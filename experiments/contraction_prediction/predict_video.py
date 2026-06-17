"""
predict_video.py — Run the trained contraction-depth model on any .mp4.

Usage:
    python predict_video.py --video /path/to/video.mp4 [--fps 30]

Outputs (in the same directory as the input video):
    <name>_predicted.mp4   — original frames with prediction overlay
    <name>_predicted.csv   — frame, elapsed_sec, predicted_crosshead_mm
"""

import argparse
import csv
import json
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from extract_pattern import extract_pattern

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_DIR = "/Users/ishanahluwalia/MTS Image Data Collection/model"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
STATS_PATH = os.path.join(MODEL_DIR, "model_stats.json")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Model (must match train_model.py)
# ---------------------------------------------------------------------------
def build_model():
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 1)
    )
    return model


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def preprocess_frame(bgr_frame, transform):
    """Convert a BGR OpenCV frame to a normalized torch tensor (1, C, H, W)."""
    bgr_frame = extract_pattern(bgr_frame)
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    tensor = transform(pil).unsqueeze(0)  # (1, 3, 224, 224)
    return tensor


# ---------------------------------------------------------------------------
# Overlay helpers
# ---------------------------------------------------------------------------
def draw_overlay(frame, text, position=(20, 50)):
    """Draw white text with a dark drop-shadow for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.2
    thickness = 2
    shadow_offset = 2

    # Shadow
    cv2.putText(frame, text,
                (position[0] + shadow_offset, position[1] + shadow_offset),
                font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    # Text
    cv2.putText(frame, text, position, font, scale,
                (255, 255, 255), thickness, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Predict contraction depth from video.")
    parser.add_argument("--video", required=True, help="Path to input .mp4 file")
    parser.add_argument("--fps", type=float, default=None,
                        help="Override output FPS (defaults to source FPS)")
    args = parser.parse_args()

    video_path = args.video
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Load stats
    if not os.path.isfile(STATS_PATH):
        raise FileNotFoundError(f"model_stats.json not found at {STATS_PATH}. Train the model first.")
    with open(STATS_PATH) as f:
        stats = json.load(f)
    label_min = stats["label_min"]
    label_max = stats["label_max"]
    label_range = label_max - label_min

    # Load model
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    print(f"Using device: {device}")

    model = build_model().to(device)
    state = torch.load(BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded model from {BEST_MODEL_PATH}")

    transform = get_transform()

    # Open input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_fps = args.fps if args.fps else src_fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {width}×{height} @ {src_fps:.2f} fps, {total_frames} frames")

    # Output paths (same directory as input)
    base = os.path.splitext(video_path)[0]
    out_video_path = base + "_predicted.mp4"
    out_csv_path = base + "_predicted.csv"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video_path, fourcc, out_fps, (width, height))

    csv_rows = []

    with torch.no_grad():
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Predict
            tensor = preprocess_frame(frame, transform).to(device)
            norm_pred = model(tensor).item()
            # Denormalize
            mm_pred = norm_pred * label_range + label_min

            elapsed_sec = round(frame_idx / src_fps, 4)

            # Overlay
            text = f"Contraction: {mm_pred:.1f} mm"
            draw_overlay(frame, text)

            writer.write(frame)
            csv_rows.append({
                "frame": frame_idx,
                "elapsed_sec": elapsed_sec,
                "predicted_crosshead_mm": round(mm_pred, 4),
            })

            if frame_idx % 100 == 0:
                print(f"  Frame {frame_idx}/{total_frames}  →  {mm_pred:.2f} mm")

            frame_idx += 1

    cap.release()
    writer.release()

    # Write CSV
    with open(out_csv_path, "w", newline="") as f:
        writer_csv = csv.DictWriter(f, fieldnames=["frame", "elapsed_sec", "predicted_crosshead_mm"])
        writer_csv.writeheader()
        writer_csv.writerows(csv_rows)

    print(f"\nDone. {frame_idx} frames processed.")
    print(f"  Labeled video : {out_video_path}")
    print(f"  Predictions CSV: {out_csv_path}")


if __name__ == "__main__":
    main()
