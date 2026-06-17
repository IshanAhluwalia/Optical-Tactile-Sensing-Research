"""
predict.py — Run displacement model on unseen test frames and compare to ground truth.

Usage:
    python displacement_test/predict.py
"""

import csv
import json
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR        = os.path.dirname(__file__)
MODEL_PATH      = os.path.join(BASE_DIR, "model", "best_model.pth")
STATS_PATH      = os.path.join(BASE_DIR, "model", "model_stats.json")
TEST_FRAMES_DIR = os.path.join(BASE_DIR, "dataset", "test", "frames")
DAQ_PATH        = os.path.join(BASE_DIR, "raw data", "DAQ- Crosshead, … - (Timed).csv")

VIDEO_FPS    = 5.0
VIDEO_OFFSET = 1.0   # DAQ time = video_time + 1.0
DAQ_SKIP     = 8

# Frames to evaluate: 1:20–1:40 = 80s–100s = frames 400–500
EVAL_START = 400
EVAL_END   = 500

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_daq(path):
    time, crosshead = [], []
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        for _ in range(DAQ_SKIP):
            next(reader)
        for row in reader:
            if not row or not row[0].strip():
                continue
            try:
                crosshead.append(float(row[1]))
                time.append(float(row[3]))
            except (ValueError, IndexError):
                continue
    return np.array(time), np.array(crosshead)


def build_model():
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(in_features, 1))
    return model


def load_model(device):
    with open(STATS_PATH) as f:
        stats = json.load(f)
    model = build_model().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model, stats["label_min"], stats["label_max"]


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def predict_frame(model, img_path, label_min, label_max, device):
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        norm_pred = model(tensor).item()
    return norm_pred * (label_max - label_min) + label_min


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = torch.device("mps"  if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available()          else "cpu")
    model, label_min, label_max = load_model(device)

    daq_time, daq_crosshead = load_daq(DAQ_PATH)

    print(f"\n{'Frame':>6} {'Time':>6} {'Actual (mm)':>12} {'Predicted (mm)':>15} {'Error (mm)':>11}")
    print("-" * 56)

    errors = []
    for frame_idx in range(EVAL_START, EVAL_END + 1):
        img_path = os.path.join(TEST_FRAMES_DIR, f"frame_{frame_idx:05d}.jpg")
        if not os.path.exists(img_path):
            continue

        video_time = frame_idx / VIDEO_FPS
        daq_t      = video_time + VIDEO_OFFSET
        actual     = float(np.interp(daq_t, daq_time, daq_crosshead))
        predicted  = predict_frame(model, img_path, label_min, label_max, device)
        error      = predicted - actual
        errors.append(abs(error))

        timestamp = f"{int(video_time)//60}:{int(video_time)%60:02d}"
        print(f"{frame_idx:>6} {timestamp:>6} {actual:>12.3f} {predicted:>15.3f} {error:>+11.3f}")

    print("-" * 56)
    print(f"{'MAE':>36} {np.mean(errors):>11.3f} mm")
    print(f"{'Max error':>36} {np.max(errors):>11.3f} mm")


if __name__ == "__main__":
    main()
