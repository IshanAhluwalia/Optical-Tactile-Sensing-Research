"""
Real-time contact estimation from live camera feed.

Loads the trained model and estimates per-frame:
  - Contact location (x, y) in mm
  - Indentation displacement (mm)
  - Contact force (N)

Press Q or ESC to quit.

Usage:
    python contact_estimation/live_predict.py
"""

import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

BASE_DIR   = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'best_model.pth')
STATS_PATH = os.path.join(BASE_DIR, 'model', 'model_stats.json')
ROI_PATH   = os.path.join(BASE_DIR, '..', 'roi.json')

TARGETS = ['loc_x', 'loc_y', 'displacement_mm', 'force_n']

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def find_cam(target_w=640, target_h=480, fallback=0):
    for i in range(6):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if w == target_w and h == target_h:
                return i
    return fallback


_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def extract_pattern(bgr):
    gray     = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    enhanced = _clahe.apply(gray)
    pattern  = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        blockSize=21, C=-20,
    )
    _, gm   = cv2.threshold(enhanced, 100, 255, cv2.THRESH_BINARY)
    pattern = cv2.bitwise_and(pattern, gm)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    pattern = cv2.morphologyEx(pattern, cv2.MORPH_OPEN, kernel)
    return cv2.cvtColor(pattern, cv2.COLOR_GRAY2BGR)


def build_model():
    m = models.resnet18(weights=None)
    m.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(m.fc.in_features, 128),
        nn.ReLU(),
        nn.Linear(128, len(TARGETS)),
    )
    return m


def main():
    with open(STATS_PATH) as f:
        stats = json.load(f)
    with open(os.path.abspath(ROI_PATH)) as f:
        roi = json.load(f)
    RX, RY, RW, RH = roi['x'], roi['y'], roi['w'], roi['h']

    device = torch.device('mps'  if torch.backends.mps.is_available() else
                          'cuda' if torch.cuda.is_available()          else 'cpu')
    model = build_model().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"Model loaded ({device}). Press Q or ESC to quit.")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    cap = cv2.VideoCapture(find_cam(640, 480))
    if not cap.isOpened():
        print("Error: could not open camera.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FPS, 30)
    for _ in range(5):
        cap.read()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        crop    = frame[RY:RY + RH, RX:RX + RW]
        pattern = extract_pattern(crop)

        pil = Image.fromarray(cv2.cvtColor(pattern, cv2.COLOR_BGR2RGB))
        inp = transform(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(inp)[0].cpu().numpy()

        preds = {}
        for i, t in enumerate(TARGETS):
            mn, mx = stats[t]
            preds[t] = float(out[i]) * (mx - mn) + mn

        # Overlay
        display = frame.copy()
        cv2.rectangle(display, (RX, RY), (RX + RW, RY + RH), (0, 255, 0), 2)

        lines = [
            f"Loc:   ({preds['loc_x']:.1f}, {preds['loc_y']:.1f}) mm",
            f"Disp:  {preds['displacement_mm']:.2f} mm",
            f"Force: {preds['force_n']:.3f} N",
        ]
        y0 = 30
        for line in lines:
            cv2.putText(display, line, (10, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y0 += 28

        scale       = frame.shape[0] / pattern.shape[0]
        pat_resized = cv2.resize(pattern, (int(pattern.shape[1] * scale), frame.shape[0]))
        divider     = np.full((frame.shape[0], 4, 3), 180, dtype=np.uint8)
        combined    = np.hstack([display, divider, pat_resized])

        cv2.imshow("Contact Estimation  |  Left: predictions   Right: pattern", combined)
        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
