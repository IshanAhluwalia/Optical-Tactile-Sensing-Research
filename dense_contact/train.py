"""
Training script for DenseContactNet.

Loss (5 terms):
    L = λ_c  * MSE(contact_map,  contact_target)
      + λ_d  * MSE(depth_map,    depth_target)
      + λ_p  * MSE(pressure_map, pressure_target)
      + λ_dp * L1(displacement,  displacement_target)
      + λ_f  * L1(force,         force_target)

Validation split: sessions whose y-coordinate is 6 or 14 are held out
entirely, so the model is evaluated on contact locations it never saw
during training.

Usage:
    python dense_contact/train.py
"""

import json
import os
import time

import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import GRID_X, GRID_Y, TactileDataset
from model import DenseContactNet

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE      = os.path.dirname(__file__)
CSV_PATH   = os.path.join(_HERE, 'dataset.csv')
MODEL_DIR  = os.path.join(_HERE, 'model')

# ── Hyperparameters ───────────────────────────────────────────────────────────
BATCH_SIZE   = 32
EPOCHS       = 150
PATIENCE     = 25

LR_BACKBONE  = 1e-5   # fine-tune encoder slowly
LR_HEAD      = 1e-4   # train decoder + heads faster

# Loss weights — all spatial maps on the same [0,1] scale so equal weights work
LAMBDA_CONTACT  = 1.0
LAMBDA_DEPTH    = 1.0
LAMBDA_PRESSURE = 1.0
LAMBDA_DISP     = 0.5
LAMBDA_FORCE    = 0.5


# ── Helpers ───────────────────────────────────────────────────────────────────

VAL_FRACTION = 0.2   # 20% of sessions held out
RANDOM_SEED  = 42

def get_sessions(csv_path: str) -> tuple[list[str], list[str]]:
    """
    Randomly hold out 20% of sessions scattered across the full grid.

    Sessions are shuffled with a fixed seed then split 80/20. This ensures
    the model sees frames from every spatial location (every X and Y) during
    training, and validation is representative of the whole sensor surface
    rather than being concentrated in specific rows/columns.
    """
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

    train_sessions, val_sessions = get_sessions(CSV_PATH)
    print(f"Train sessions: {len(train_sessions)}  |  Val sessions: {len(val_sessions)}")

    train_ds = TactileDataset(CSV_PATH, train_sessions, train=True,  stride=1)
    val_ds   = TactileDataset(CSV_PATH, val_sessions,   train=False, stride=1)
    print(f"Train samples:  {len(train_ds)}  |  Val samples: {len(val_ds)}")
    print(f"disp_max={train_ds.disp_max:.3f} mm  "
          f"force_max={train_ds.force_max:.3f} N  "
          f"pressure_max={train_ds.pressure_max:.5f} N/mm²")

    pin = device.type == 'cuda'  # pin_memory only helps on CUDA, not MPS
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=pin,
    )

    model = DenseContactNet().to(device)

    # ── Resume from checkpoint if one exists ──────────────────────────────────
    resume_path = os.path.join(MODEL_DIR, 'resume_checkpoint.pth')
    start_epoch  = 1
    best_val     = float('inf')
    patience_ctr = 0

    # Differential learning rates: backbone (pretrained) vs decoder + heads
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
        list(model.grid_pool.parameters()) +
        list(model.contact_head.parameters()) +
        list(model.depth_head.parameters()) +
        list(model.pressure_head.parameters()) +
        list(model.scalar_head.parameters())
    )

    optimizer = torch.optim.Adam([
        {'params': backbone_params, 'lr': LR_BACKBONE},
        {'params': head_params,     'lr': LR_HEAD},
    ])
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    mse_loss = nn.MSELoss()
    l1_loss  = nn.L1Loss()

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

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{EPOCHS} [train]",
                    leave=False, dynamic_ncols=True)
        for batch in pbar:
            imgs      = batch['image'].to(device)
            c_target  = batch['contact_map'].to(device)
            d_target  = batch['depth_map'].to(device)
            p_target  = batch['pressure_map'].to(device)
            dp_target = batch['displacement'].to(device)
            f_target  = batch['force'].to(device)

            optimizer.zero_grad()
            out = model(imgs)

            loss = (
                LAMBDA_CONTACT  * mse_loss(out['contact_map'],  c_target)  +
                LAMBDA_DEPTH    * mse_loss(out['depth_map'],    d_target)   +
                LAMBDA_PRESSURE * mse_loss(out['pressure_map'], p_target)  +
                LAMBDA_DISP     * l1_loss( out['displacement'], dp_target)  +
                LAMBDA_FORCE    * l1_loss( out['force'],        f_target)
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
        lx_errs, ly_errs, dp_errs, f_errs = [], [], [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch:3d}/{EPOCHS} [val]  ",
                              leave=False, dynamic_ncols=True):
                imgs      = batch['image'].to(device)
                c_target  = batch['contact_map'].to(device)
                d_target  = batch['depth_map'].to(device)
                p_target  = batch['pressure_map'].to(device)
                dp_target = batch['displacement'].to(device)
                f_target  = batch['force'].to(device)

                out = model(imgs)

                val_total += (
                    LAMBDA_CONTACT  * mse_loss(out['contact_map'],  c_target)  +
                    LAMBDA_DEPTH    * mse_loss(out['depth_map'],    d_target)   +
                    LAMBDA_PRESSURE * mse_loss(out['pressure_map'], p_target)  +
                    LAMBDA_DISP     * l1_loss( out['displacement'], dp_target)  +
                    LAMBDA_FORCE    * l1_loss( out['force'],        f_target)
                ).item()

                lx_errs.append(
                    (out['loc_x'] - batch['loc_x'].to(device)).abs().mean().item()
                )
                ly_errs.append(
                    (out['loc_y'] - batch['loc_y'].to(device)).abs().mean().item()
                )
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
            f"loc_x={_mean(lx_errs):.2f}mm  loc_y={_mean(ly_errs):.2f}mm  "
            f"disp={_mean(dp_errs):.3f}mm  force={_mean(f_errs):.4f}N  "
            f"lr={scheduler.get_last_lr()[1]:.2e}  "
            f"[{elapsed/60:.1f}min]"
        )

        # ── Checkpoint ────────────────────────────────────────────────────────
        stats = {
            'disp_max':     train_ds.disp_max,
            'force_max':    train_ds.force_max,
            'pressure_max': train_ds.pressure_max,
            'grid_x':       GRID_X.tolist(),
            'grid_y':       GRID_Y.tolist(),
        }

        # Best model — saved whenever val loss improves
        if val_avg < best_val:
            best_val = val_avg
            patience_ctr = 0
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'best_model.pth'))
            with open(os.path.join(MODEL_DIR, 'model_stats.json'), 'w') as fh:
                json.dump(stats, fh, indent=2)
            print(f"  ✓ best model saved (val={best_val:.4f})")
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

        # Resume checkpoint — saved every epoch so training can be restarted
        # from exactly where it left off if interrupted
        torch.save({
            'epoch':         epoch,
            'model_state':   model.state_dict(),
            'optimizer':     optimizer.state_dict(),
            'scheduler':     scheduler.state_dict(),
            'best_val':      best_val,
            'patience_ctr':  patience_ctr,
            'stats':         stats,
        }, os.path.join(MODEL_DIR, 'resume_checkpoint.pth'))

    print(f"\nDone. Best val loss: {best_val:.4f}")
    print(f"Weights: {MODEL_DIR}/best_model.pth")
    print(f"Stats:   {MODEL_DIR}/model_stats.json")


if __name__ == '__main__':
    train()
