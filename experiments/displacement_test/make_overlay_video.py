"""
make_overlay_video.py — Cut 1:19–1:39 from the original video and overlay
predicted vs actual displacement on each frame.

Output:
    displacement_test/assets/overlay_1m19s_to_1m39s.mp4
"""

import csv
import json
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(__file__)
VIDEO_PATH = os.path.join(BASE_DIR, "raw data", "video_20260403_150500.mp4")
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model.pth")
STATS_PATH = os.path.join(BASE_DIR, "model", "model_stats.json")
DAQ_PATH   = os.path.join(BASE_DIR, "raw data", "DAQ- Crosshead, … - (Timed).csv")
OUT_DIR    = os.path.join(BASE_DIR, "assets")
OUT_PATH   = os.path.join(OUT_DIR, "overlay_1m19s_to_1m39s.mp4")

VIDEO_FPS    = 5.0
VIDEO_OFFSET = 1.0
DAQ_SKIP     = 8
CLIP_START   = 79   # seconds (1:19)
CLIP_END     = 99   # seconds (1:39)

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
    m = models.resnet18(weights=None)
    m.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(m.fc.in_features, 1))
    return m


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


def predict(model, frame_bgr, label_min, label_max, device):
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    t   = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        norm = model(t).item()
    return norm * (label_max - label_min) + label_min


def draw_overlay(frame, video_time, actual, predicted):
    h, w = frame.shape[:2]

    # Semi-transparent dark bar at bottom
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 90), (w, h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    font       = cv2.FONT_HERSHEY_DUPLEX
    small_font = cv2.FONT_HERSHEY_SIMPLEX
    m, s       = divmod(int(video_time), 60)
    timestamp  = f"{m}:{s:02d}"

    # Timestamp top-left
    cv2.putText(frame, timestamp, (12, 32), small_font, 0.8, (255, 255, 255), 2)

    # Actual (blue)
    cv2.putText(frame, f"Actual:    {actual:+.3f} mm",
                (12, h - 55), font, 0.65, (100, 200, 255), 2)
    # Predicted (red)
    cv2.putText(frame, f"Predicted: {predicted:+.3f} mm",
                (12, h - 20), font, 0.65, (80, 100, 255), 2)

    # Error (white, right-aligned)
    err_text = f"Error: {predicted - actual:+.3f} mm"
    text_sz  = cv2.getTextSize(err_text, small_font, 0.65, 2)[0]
    cv2.putText(frame, err_text,
                (w - text_sz[0] - 12, h - 20), small_font, 0.65, (200, 200, 200), 2)

    return frame


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    device = torch.device("mps"  if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available()          else "cpu")
    model, label_min, label_max = load_model(device)
    daq_time, daq_crosshead     = load_daq(DAQ_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUT_PATH, fourcc, VIDEO_FPS, (w, h))

    start_frame = int(CLIP_START * VIDEO_FPS)
    end_frame   = int(CLIP_END   * VIDEO_FPS)

    print(f"Rendering frames {start_frame}–{end_frame} ({CLIP_END - CLIP_START}s)...")

    for frame_idx in range(start_frame, end_frame + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        video_time = frame_idx / VIDEO_FPS
        daq_t      = video_time + VIDEO_OFFSET
        actual     = float(np.interp(daq_t, daq_time, daq_crosshead))
        predicted  = predict(model, frame, label_min, label_max, device)

        frame = draw_overlay(frame, video_time, actual, predicted)
        writer.write(frame)

        m, s = divmod(int(video_time), 60)
        print(f"  {m}:{s:02d} (frame {frame_idx}) | actual={actual:+.3f}  predicted={predicted:+.3f}  err={predicted-actual:+.3f}")

    cap.release()
    writer.release()
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()
