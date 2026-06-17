"""
train_model.py — ResNet18 regression for contraction depth prediction.

Usage:
    python train_model.py

Outputs:
    model/best_model.pth      — best validation-loss weights
    model/model_stats.json    — label min/max for denormalization
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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CSV_PATH = "/Users/ishanahluwalia/MTS Image Data Collection/dataset/dataset.csv"
MODEL_DIR = "/Users/ishanahluwalia/MTS Image Data Collection/model"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
STATS_PATH = os.path.join(MODEL_DIR, "model_stats.json")

TRAIN_END = 2162  # ~80% of 2703
VAL_END = 2432    # ~90% of 2703
# test: frames 2432–2702

BATCH_SIZE = 16
LR_HEAD = 1e-4       # head learning rate
LR_BACKBONE = 1e-5   # backbone fine-tune learning rate
EPOCHS = 200
EARLY_STOP_PATIENCE = 25

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class SkinDataset(Dataset):
    def __init__(self, df, label_min, label_max, augment=False):
        self.df = df.reset_index(drop=True)
        self.label_min = label_min
        self.label_max = label_max

        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        img = self.transform(img)

        label_raw = float(row["crosshead_mm"])
        label_norm = (label_raw - self.label_min) / (self.label_max - self.label_min)
        label_norm = float(np.clip(label_norm, 0.0, 1.0))

        return img, torch.tensor(label_norm, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Replace final FC with dropout + regression head to combat overfitting
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 1)
    )
    # All parameters are trainable (full fine-tune)
    return model


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss = 0.0
    with torch.set_grad_enabled(train):
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device).unsqueeze(1)
            preds = model(imgs)
            loss = criterion(preds, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(imgs)
    return total_loss / len(loader.dataset)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Load CSV
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} samples from dataset.csv")

    # 2. Random split (so all contraction depths are represented in training)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_df = df.iloc[:TRAIN_END]
    val_df = df.iloc[TRAIN_END:VAL_END]
    test_df = df.iloc[VAL_END:]
    print(f"Split — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    # 3. Label stats from train set only
    label_min = float(train_df["crosshead_mm"].min())
    label_max = float(train_df["crosshead_mm"].max())
    stats = {"label_min": label_min, "label_max": label_max}
    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Label range (train): [{label_min:.4f}, {label_max:.4f}] mm")
    print(f"Saved stats to {STATS_PATH}")

    # 4. Datasets & loaders
    train_ds = SkinDataset(train_df, label_min, label_max, augment=True)
    val_ds = SkinDataset(val_df, label_min, label_max, augment=False)
    test_ds = SkinDataset(test_df, label_min, label_max, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=2, pin_memory=True)

    # 5. Model, loss, optimizer, scheduler
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    print(f"Using device: {device}")

    model = build_model().to(device)
    criterion = nn.L1Loss()
    # Differential LRs: backbone gets 10x lower LR than head
    backbone_params = [p for n, p in model.named_parameters() if not n.startswith("fc")]
    head_params = list(model.fc.parameters())
    optimizer = torch.optim.Adam([
        {"params": backbone_params, "lr": LR_BACKBONE, "weight_decay": 1e-4},
        {"params": head_params,     "lr": LR_HEAD,     "weight_decay": 1e-4},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    # 6. Training loop
    best_val_loss = float("inf")
    no_improve_count = 0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        elapsed = time.time() - t0

        scheduler.step()

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            no_improve_count = 0
            marker = " *"
        else:
            no_improve_count += 1
            marker = ""

        label_range = label_max - label_min
        print(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"train_mae={train_loss * label_range:.3f} mm | "
            f"val_mae={val_loss * label_range:.3f} mm | "
            f"{elapsed:.1f}s{marker}"
        )

        if no_improve_count >= EARLY_STOP_PATIENCE:
            print(f"Early stopping triggered after {epoch} epochs (no val improvement for {EARLY_STOP_PATIENCE} epochs).")
            break

    # 7. Test evaluation
    print("\nLoading best weights for test evaluation…")
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    test_loss = run_epoch(model, test_loader, criterion, optimizer, device, train=False)
    label_range = label_max - label_min
    test_mae_mm = test_loss * label_range
    print(f"Test MAE: {test_mae_mm:.3f} mm")
    print(f"Best model saved to {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
