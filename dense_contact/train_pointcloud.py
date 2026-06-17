"""
Training script for PointCloudNet.

The model takes a grayscale camera image and predicts the full 3D geometry
of the deformed tactile sensor skin as a structured (2048, 3) point cloud.

Loss (3 terms):
    L = λ_pc   * MSE(pred_cloud, target_cloud)      — spatial geometry (mm²)
      + λ_disp * L1(displacement, disp_target)       — normalised depth
      + λ_force* L1(force, force_target)             — normalised force

MSE on the ordered cloud works because generate_pointclouds.py produces the
same fixed 32×64 grid for every frame — point k always corresponds to grid
cell (k // 64, k % 64).

Usage:
    python dense_contact/train_pointcloud.py
"""

import json
import os
import sys
import time

import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PointCloudDataset
from model import PointCloudNet

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE          = os.path.dirname(__file__)
CSV_PATH       = os.path.join(_HERE, 'dataset.csv')
REST_POS_PATH  = os.path.join(_HERE, 'rest_positions.npy')
MODEL_DIR      = os.path.join(_HERE, 'model_pc')

# ── Hyperparameters ───────────────────────────────────────────────────────────
BATCH_SIZE  = 16     # smaller than heatmap model: point clouds (2048,3) per sample
EPOCHS      = 150
PATIENCE    = 25

LR_BACKBONE = 1e-5   # fine-tune pretrained encoder slowly
LR_HEAD     = 1e-4   # train decoder + PC head faster

# Loss weights
# λ_pc is large because MSE over 2048×3 floats in mm² units can be small
LAMBDA_PC   = 1.0
LAMBDA_DISP = 0.5
LAMBDA_FORCE= 0.5

# ── Helpers ───────────────────────────────────────────────────────────────────

VAL_FRACTION = 0.2
RANDOM_SEED  = 42


def get_sessions(csv_path: str) -> tuple[list[str], list[str]]:
    """Random 80/20 session split (same seed as DenseContactNet)."""
    import random
    sessions = sorted(pd.read_csv(csv_path)['session'].unique().tolist())
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(sessions)
    n_val = max(1, int(len(sessions) * VAL_FRACTION))
    val   = sessions[:n_val]
    train = sessions[n_val:]
    return train, val


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


# ── Training loop ─────────────────────────────────────────────────────────────

def train() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)

    device = pick_device()
    print(f"Device: {device}")

    if not os.path.exists(REST_POS_PATH):
        print(f"ERROR: rest_positions.npy not found at {REST_POS_PATH}")
        print("Run:  python dense_contact/build_rest_positions.py")
        sys.exit(1)

    train_sessions, val_sessions = get_sessions(CSV_PATH)
    print(f"Train sessions: {len(train_sessions)}  |  Val sessions: {len(val_sessions)}")

    train_ds = PointCloudDataset(CSV_PATH, train_sessions, train=True)
    val_ds   = PointCloudDataset(CSV_PATH, val_sessions,   train=False)
    print(f"Train samples:  {len(train_ds)}  |  Val samples: {len(val_ds)}")
    print(f"disp_max={train_ds.disp_max:.3f} mm  force_max={train_ds.force_max:.3f} N")

    pin = device.type == 'cuda'
    n_workers = 0 if device.type == 'mps' else 4   # num_workers>0 leaks semaphores on MPS
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=n_workers, pin_memory=pin, persistent_workers=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=n_workers, pin_memory=pin, persistent_workers=False,
    )

    model = PointCloudNet(rest_positions_path=REST_POS_PATH).to(device)

    # ── Differential learning rates ───────────────────────────────────────────
    backbone_params = (
        list(model.enc_stem.parameters()) +
        list(model.enc_pool.parameters()) +
        list(model.enc1.parameters()) +
        list(model.enc2.parameters()) +
        list(model.enc3.parameters()) +
        list(model.enc4.parameters())
    )
    head_params = (
        list(model.dec4.parameters()) +
        list(model.dec3.parameters()) +
        list(model.dec2.parameters()) +
        list(model.dec1.parameters()) +
        list(model.pc_head.parameters()) +
        list(model.scalar_head.parameters())
    )

    optimizer = torch.optim.Adam([
        {'params': backbone_params, 'lr': LR_BACKBONE},
        {'params': head_params,     'lr': LR_HEAD},
    ])
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    mse_loss = nn.MSELoss()
    l1_loss  = nn.L1Loss()

    # ── Resume from checkpoint ────────────────────────────────────────────────
    import shutil
    resume_path = os.path.join(MODEL_DIR, 'resume_checkpoint.pth')
    best_path   = os.path.join(MODEL_DIR, 'best_model.pth')
    backup_path = os.path.join(MODEL_DIR, 'best_model_backup.pth')
    start_epoch  = 1
    best_val     = float('inf')
    patience_ctr = 0

    if os.path.exists(best_path):
        shutil.copy2(best_path, backup_path)
        print(f"Backed up previous best model → {backup_path}")

    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch  = ckpt['epoch'] + 1
        best_val     = ckpt['best_val']
        patience_ctr = ckpt['patience_ctr']
        print(f"Resumed from epoch {ckpt['epoch']}  (best val={best_val:.4f})")
    else:
        print("Starting fresh training run")

    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_start = time.time()

        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        train_total = 0.0

        _tty = sys.stdout.isatty()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{EPOCHS} [train]",
                    leave=False, dynamic_ncols=True, disable=not _tty)
        for batch in pbar:
            imgs        = batch['image'].to(device)
            cloud_tgt   = batch['point_cloud'].to(device)   # (B, 2048, 3)
            disp_target = batch['displacement'].to(device)
            force_target= batch['force'].to(device)

            optimizer.zero_grad()
            out = model(imgs)

            loss = (
                LAMBDA_PC    * mse_loss(out['point_cloud'],  cloud_tgt)    +
                LAMBDA_DISP  * l1_loss( out['displacement'], disp_target)  +
                LAMBDA_FORCE * l1_loss( out['force'],        force_target)
            )

            loss.backward()
            optimizer.step()
            train_total += loss.item()
            pbar.set_postfix(loss=f'{loss.item():.4f}')

        pbar.close()
        scheduler.step()

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        val_total = 0.0
        pc_rmse_list, dp_errs, f_errs = [], [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch:3d}/{EPOCHS} [val]  ",
                              leave=False, dynamic_ncols=True, disable=not _tty):
                imgs        = batch['image'].to(device)
                cloud_tgt   = batch['point_cloud'].to(device)
                disp_target = batch['displacement'].to(device)
                force_target= batch['force'].to(device)

                out = model(imgs)

                val_total += (
                    LAMBDA_PC    * mse_loss(out['point_cloud'],  cloud_tgt)    +
                    LAMBDA_DISP  * l1_loss( out['displacement'], disp_target)  +
                    LAMBDA_FORCE * l1_loss( out['force'],        force_target)
                ).item()

                # Per-point RMSE in mm: sqrt(mean over points and xyz of squared error)
                sq_err = (out['point_cloud'] - cloud_tgt) ** 2  # (B, 2048, 3)
                rmse   = sq_err.mean(dim=(1, 2)).sqrt().mean().item()  # scalar mm
                pc_rmse_list.append(rmse)

                dp_errs.append(
                    (out['displacement'] * train_ds.disp_max
                     - batch['displacement_raw'].to(device)).abs().mean().item()
                )
                f_errs.append(
                    (out['force'] * train_ds.force_max
                     - batch['force_raw'].to(device)).abs().mean().item()
                )

        n_tr    = len(train_loader)
        n_vl    = len(val_loader)
        val_avg = val_total / n_vl
        elapsed = time.time() - epoch_start

        def _mean(lst):
            return sum(lst) / len(lst)

        print(
            f"Epoch {epoch:3d}/{EPOCHS}  "
            f"train={train_total/n_tr:.4f}  val={val_avg:.4f}  |  "
            f"pc_rmse={_mean(pc_rmse_list):.3f}mm  "
            f"disp={_mean(dp_errs):.3f}mm  force={_mean(f_errs):.4f}N  "
            f"lr={scheduler.get_last_lr()[1]:.2e}  "
            f"[{elapsed/60:.1f}min]"
        )

        # ── Checkpoint ────────────────────────────────────────────────────────
        stats = {
            'disp_max':  train_ds.disp_max,
            'force_max': train_ds.force_max,
        }

        if val_avg < best_val:
            best_val = val_avg
            patience_ctr = 0
            if os.path.exists(best_path):
                shutil.copy2(best_path, backup_path)
            torch.save(model.state_dict(), best_path)
            with open(os.path.join(MODEL_DIR, 'model_stats.json'), 'w') as fh:
                json.dump(stats, fh, indent=2)
            print(f"  ✓ best model saved (val={best_val:.4f})")
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

        torch.save({
            'epoch':        epoch,
            'model_state':  model.state_dict(),
            'optimizer':    optimizer.state_dict(),
            'scheduler':    scheduler.state_dict(),
            'best_val':     best_val,
            'patience_ctr': patience_ctr,
            'stats':        stats,
        }, os.path.join(MODEL_DIR, 'resume_checkpoint.pth'))

    print(f"\nDone. Best val loss: {best_val:.4f}")
    print(f"Weights: {MODEL_DIR}/best_model.pth")
    print(f"Stats:   {MODEL_DIR}/model_stats.json")


if __name__ == '__main__':
    train()
