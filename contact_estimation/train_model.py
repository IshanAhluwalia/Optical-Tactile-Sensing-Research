"""
Train a ResNet18 model to estimate contact location, force, and displacement
from extracted skin pattern images.

Predicts 4 outputs per frame:
  - loc_x  (mm) — contact x position
  - loc_y  (mm) — contact y position
  - displacement_mm — indentation depth
  - force_n — contact force

Validation uses held-out sessions (not seen during training).

Usage:
    python contact_estimation/train_model.py

Outputs:
    contact_estimation/model/best_model.pth
    contact_estimation/model/model_stats.json
"""

import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

BASE_DIR   = os.path.dirname(__file__)
CSV_PATH   = os.path.join(BASE_DIR, 'dataset.csv')
MODEL_DIR  = os.path.join(BASE_DIR, 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.pth')
STATS_PATH = os.path.join(MODEL_DIR, 'model_stats.json')

# Hold out two sessions spread across the y range for validation
VAL_SESSIONS = ['(-140, -6)', '(-140, -14)']

BATCH_SIZE  = 32
LR_HEAD     = 1e-4
LR_BACKBONE = 1e-5
EPOCHS      = 150
PATIENCE    = 25

TARGETS = ['loc_x', 'loc_y', 'displacement_mm', 'force_n']

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ContactDataset(Dataset):
    def __init__(self, df, stats, augment=False):
        self.df    = df.reset_index(drop=True)
        self.stats = stats

        base = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        aug = [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(8),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform = transforms.Compose(aug if augment else base)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['extracted_path']).convert('RGB')
        img = self.transform(img)

        label = torch.tensor([
            float(np.clip(
                (float(row[t]) - self.stats[t][0]) /
                max(self.stats[t][1] - self.stats[t][0], 1e-6),
                0.0, 1.0
            ))
            for t in TARGETS
        ], dtype=torch.float32)

        return img, label


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model():
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(m.fc.in_features, 128),
        nn.ReLU(),
        nn.Linear(128, len(TARGETS)),
    )
    return m


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total      = 0.0
    target_sum = torch.zeros(len(TARGETS))
    with torch.set_grad_enabled(train):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)
            loss  = criterion(preds, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total      += loss.item() * len(imgs)
            target_sum += (preds - labels).abs().sum(dim=0).cpu().detach()
    n = len(loader.dataset)
    return total / n, target_sum / n


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} samples from {df['session'].nunique()} sessions")

    val_mask = df['session'].isin(VAL_SESSIONS)
    train_df = df[~val_mask].reset_index(drop=True)
    val_df   = df[ val_mask].reset_index(drop=True)
    print(f"Train: {len(train_df)} ({train_df['session'].nunique()} sessions)")
    print(f"Val:   {len(val_df)}   ({val_df['session'].nunique()} sessions) — {VAL_SESSIONS}")

    # Stats from train set only
    stats = {t: (float(train_df[t].min()), float(train_df[t].max())) for t in TARGETS}
    with open(STATS_PATH, 'w') as f:
        json.dump(stats, f, indent=2)
    for t, (mn, mx) in stats.items():
        print(f"  {t}: [{mn:.4f}, {mx:.4f}]")

    train_ds = ContactDataset(train_df, stats, augment=True)
    val_ds   = ContactDataset(val_df,   stats, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)

    device = torch.device('mps'  if torch.backends.mps.is_available() else
                          'cuda' if torch.cuda.is_available()          else 'cpu')
    print(f"Device: {device}\n")

    model     = build_model().to(device)
    criterion = nn.L1Loss()

    backbone_params = [p for n, p in model.named_parameters() if not n.startswith('fc')]
    head_params     = list(model.fc.parameters())
    optimizer = torch.optim.Adam([
        {'params': backbone_params, 'lr': LR_BACKBONE, 'weight_decay': 1e-4},
        {'params': head_params,     'lr': LR_HEAD,     'weight_decay': 1e-4},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    ranges    = {t: max(stats[t][1] - stats[t][0], 1e-6) for t in TARGETS}
    best_val  = float('inf')
    no_improv = 0

    for epoch in range(1, EPOCHS + 1):
        t0          = time.time()
        tr_loss, _  = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        vl_loss, vl_per = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()

        mark = ''
        if vl_loss < best_val:
            best_val  = vl_loss
            no_improv = 0
            torch.save(model.state_dict(), MODEL_PATH)
            mark = ' *'
        else:
            no_improv += 1

        mae_str = '  '.join(
            f"{t.split('_')[0]}={vl_per[i].item() * ranges[t]:.3f}"
            for i, t in enumerate(TARGETS)
        )
        print(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"train={tr_loss:.4f}  val={vl_loss:.4f} | "
            f"{mae_str} | {time.time()-t0:.1f}s{mark}"
        )

        if no_improv >= PATIENCE:
            print(f"Early stopping at epoch {epoch}.")
            break

    print(f"\nBest val loss: {best_val:.4f}")
    print(f"Model  → {MODEL_PATH}")
    print(f"Stats  → {STATS_PATH}")


if __name__ == '__main__':
    main()
