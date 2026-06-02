"""
Live pattern extraction preview.
Shows the camera feed with the ROI highlighted, and the extracted pattern side-by-side.
Press Q or ESC to quit.
"""

import cv2
import json
import os
import sys
import numpy as np
from datetime import datetime

# ── Find camera ──────────────────────────────────────────────────────────────
def find_camera_index(target_w: int = 640, target_h: int = 480, fallback: int = 0) -> int:
    for i in range(6):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if w == target_w and h == target_h:
                return i
    return fallback

CAM_INDEX = find_camera_index(640, 480)

# ── Load ROI ─────────────────────────────────────────────────────────────────
_roi_path = os.path.join(os.path.dirname(__file__), '..', 'roi.json')
with open(os.path.abspath(_roi_path)) as f:
    _roi = json.load(f)
RX, RY, RW, RH = _roi['x'], _roi['y'], _roi['w'], _roi['h']

# ── Pattern extraction ────────────────────────────────────────────────────────
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def extract_pattern(bgr_frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    # Enhance local contrast so ink stands out even in dim lighting
    enhanced = _clahe.apply(gray)
    # THRESH_BINARY: bright ink (white on dark) → white, background → black
    # Negative C means pixel must be brighter than local mean + |C|
    pattern = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=21, C=-20,
    )
    # Global brightness floor — reject anything too dim to be ink
    _, global_mask = cv2.threshold(enhanced, 130, 255, cv2.THRESH_BINARY)
    pattern = cv2.bitwise_and(pattern, global_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    pattern = cv2.morphologyEx(pattern, cv2.MORPH_OPEN, kernel)
    return cv2.cvtColor(pattern, cv2.COLOR_GRAY2BGR)

# ── Camera ────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    print("Error: could not open camera.")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FPS, 30)
for _ in range(10):
    cap.read()
print(f"Camera opened. Press S to select ROI, R to start/stop recording, Q or ESC to quit.")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
recording = False
writer    = None
video_path = None

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Draw ROI rectangle on the full frame
    display = frame.copy()
    cv2.rectangle(display, (RX, RY), (RX + RW, RY + RH), (0, 255, 0), 2)

    # Crop and extract pattern
    crop    = frame[RY:RY + RH, RX:RX + RW]
    pattern = extract_pattern(crop)

    # Resize pattern to match full frame height for side-by-side display
    scale   = frame.shape[0] / pattern.shape[0]
    pattern_resized = cv2.resize(pattern, (int(pattern.shape[1] * scale), frame.shape[0]))

    # Side-by-side: full frame with ROI box | extracted pattern
    divider = np.full((frame.shape[0], 4, 3), 180, dtype=np.uint8)
    combined = np.hstack([display, divider, pattern_resized])

    if recording:
        writer.write(combined)
        cv2.circle(combined, (18, 22), 10, (0, 0, 255), -1)
        cv2.putText(combined, "REC", (35, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Live Extraction  |  Left: camera + ROI   Right: extracted pattern", combined)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord('s'), ord('S')):
        roi_sel = cv2.selectROI("Select ROI — drag then press Enter/Space, C to cancel",
                                frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select ROI — drag then press Enter/Space, C to cancel")
        if roi_sel[2] > 0 and roi_sel[3] > 0:
            RX, RY, RW, RH = int(roi_sel[0]), int(roi_sel[1]), int(roi_sel[2]), int(roi_sel[3])
            roi_data = {'x': RX, 'y': RY, 'w': RW, 'h': RH}
            _roi_abs = os.path.abspath(_roi_path)
            with open(_roi_abs, 'w') as _f:
                json.dump(roi_data, _f, indent=2)
            print(f"ROI saved: x={RX}, y={RY}, w={RW}, h={RH}", flush=True)
    elif key in (ord('r'), ord('R')):
        if not recording:
            stamp      = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = os.path.join(OUTPUT_DIR, f"preprocessing_{stamp}.mp4")
            fourcc     = cv2.VideoWriter_fourcc(*'mp4v')
            writer     = cv2.VideoWriter(video_path, fourcc, 30, (combined.shape[1], combined.shape[0]))
            recording  = True
            print(f"Recording: {video_path}", flush=True)
        else:
            recording = False
            writer.release()
            writer = None
            print(f"Saved: {video_path}", flush=True)
    elif key in (ord('q'), ord('Q'), 27):
        break

if recording and writer:
    writer.release()
    print(f"Saved: {video_path}", flush=True)

cap.release()
cv2.destroyAllWindows()
