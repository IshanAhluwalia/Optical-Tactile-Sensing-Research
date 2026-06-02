"""
Master recording script — camera + load cell simultaneously.
- Auto-tares load cell on startup, waits for confirmed live data before starting
- Smoothed force readings, hysteresis triggering
- Saves video_<stamp>.mp4 + data_<stamp>.csv (time_s, frame, force_n)
- Press T to re-tare, Q or ESC to quit
"""

import cv2
import serial
import csv
import json
import os
import re
import sys
import subprocess
import time
import threading
import numpy as np
from collections import deque
from datetime import datetime

SERIAL_PORT     = '/dev/tty.usbmodemF412FA6357AC2'
BAUD            = 115200

def _find_cam(target_w: int = 640, target_h: int = 480, fallback: int = 0) -> int:
    for i in range(6):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if w == target_w and h == target_h:
                return i
    return fallback

CAM_INDEX = _find_cam(640, 480)
BASE_OUTPUT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "output")
START_THRESHOLD = 0.01

# --- Load ROI ---
_roi_path = os.path.join(os.path.dirname(__file__), '..', 'roi.json')
with open(os.path.abspath(_roi_path)) as _f:
    _roi = json.load(_f)
ROI = (_roi['x'], _roi['y'], _roi['w'], _roi['h'])

_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def extract_pattern(bgr_frame):
    gray     = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    enhanced = _clahe.apply(gray)
    pattern  = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=21, C=-20,
    )
    _, global_mask = cv2.threshold(enhanced, 100, 255, cv2.THRESH_BINARY)
    pattern  = cv2.bitwise_and(pattern, global_mask)
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    pattern  = cv2.morphologyEx(pattern, cv2.MORPH_OPEN, kernel)
    return cv2.cvtColor(pattern, cv2.COLOR_GRAY2BGR)
STOP_THRESHOLD  = 0.005
SMOOTH_WINDOW   = 4
SPEED_MM_PER_S  = 10.0 / 60.0   # 10 mm/min
MAX_INDENT_MM   = 10.0           # hard stop at 10 mm

force_buffer   = deque(maxlen=SMOOTH_WINDOW)
latest_force   = {"raw": 0.0, "smooth": 0.0}
data_received  = [False]
recording       = False
csv_file        = None
csv_writer      = None
frame_count     = 0
video_path      = None
writer          = None
rec_start_time  = 0.0
contact_force   = 0.0
ser_ref         = {"ser": None}
ACTUAL_FPS      = 30.0

frames_dir = None

def start_recording(width, height, baseline_force):
    global recording, csv_file, csv_writer, frame_count, video_path, writer, rec_start_time, contact_force, frames_dir
    stamp          = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path     = os.path.join(OUTPUT_DIR, f"{session_name}.mp4")
    csv_path       = os.path.join(OUTPUT_DIR, f"{session_name}.csv")
    frames_dir     = os.path.join(OUTPUT_DIR, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    fourcc         = cv2.VideoWriter_fourcc(*'mp4v')
    writer         = cv2.VideoWriter(video_path, fourcc, ACTUAL_FPS, (width, height))
    csv_file       = open(csv_path, 'w', newline='')
    csv_writer     = csv.writer(csv_file)
    csv_writer.writerow(['time_s', 'displacement_mm', 'frame', 'force_n', 'image_path'])
    frame_count    = 0
    rec_start_time = time.monotonic()
    contact_force  = baseline_force
    recording      = True
    print(f"Contact! Baseline force={baseline_force:.3f} N -> {video_path}", flush=True)

def stop_recording():
    global recording, csv_file, csv_writer, writer
    recording = False
    writer.release()
    writer = None
    csv_file.close()
    csv_file   = None
    csv_writer = None
    print(f"Saved: {video_path}  ({frame_count} frames)", flush=True)
    post_process_frames()

def post_process_frames():
    x, y, w, h = ROI
    csv_path     = os.path.join(OUTPUT_DIR, f"{session_name}.csv")
    ext_dir      = os.path.join(frames_dir, "extracted")
    os.makedirs(ext_dir, exist_ok=True)

    print("Extracting patterns...", flush=True)
    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames + ['extracted_path']
        for row in reader:
            img = cv2.imread(row['image_path'])
            if img is not None:
                crop    = img[y:y+h, x:x+w]
                pattern = extract_pattern(crop)
                ext_path = os.path.join(ext_dir, os.path.basename(row['image_path']))
                cv2.imwrite(ext_path, pattern)
                row['extracted_path'] = ext_path
            else:
                row['extracted_path'] = ''
            rows.append(row)

    with open(csv_path, 'w', newline='') as f:
        writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(rows)
    print(f"Done. Extracted patterns saved to {ext_dir}", flush=True)

def serial_thread():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD, timeout=1)
        ser_ref["ser"] = ser
    except serial.SerialException as e:
        print(f"Serial error: {e}", flush=True)
        return

    # Collect baseline samples for Python-side tare
    print("Calibrating baseline...", flush=True)
    baseline_samples = []
    deadline = time.monotonic() + 10.0  # 10s timeout
    while len(baseline_samples) < 20 and time.monotonic() < deadline:
        try:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if not line:
                continue
            parts = line.split(',')
            force = parts[-1]  # always use last value regardless of format
            baseline_samples.append(float(force))
            print(f"  baseline sample {len(baseline_samples)}/20: {float(force):.4f} N", flush=True)
        except (ValueError, serial.SerialException):
            continue
    if not baseline_samples:
        print("Warning: no serial data — running without force.", flush=True)
        return
    baseline = sum(baseline_samples) / len(baseline_samples)
    print(f"Baseline: {baseline:.4f} N. Load cell ready.", flush=True)

    while True:
        try:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if not line or ',' not in line:
                continue
            parts = line.split(',')
            if len(parts) not in (2, 4):
                continue
            force = parts[-1]
            f = abs(float(force) - baseline)
            force_buffer.append(f)
            latest_force["raw"]    = f
            latest_force["smooth"] = sum(force_buffer) / len(force_buffer)
            data_received[0]       = True
        except (ValueError, serial.SerialException):
            continue

threading.Thread(target=serial_thread, daemon=True).start()

# --- Session name popup ---
import tkinter as tk
from tkinter import simpledialog
_root = tk.Tk()
_root.withdraw()
session_name = simpledialog.askstring("Session Name", "Enter a name for this recording:", parent=_root)
_root.destroy()
if not session_name or not session_name.strip():
    session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
else:
    session_name = session_name.strip()
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, session_name)
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Saving to: {OUTPUT_DIR}", flush=True)

# --- Camera ---
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    print("Error: could not open camera.", flush=True)
    sys.exit(1)

cap.set(cv2.CAP_PROP_FPS, 30)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

ACTUAL_FPS = 30.0
for _ in range(10):
    cap.read()
print(f"Camera ready ({width}x{height} @ 30 fps).", flush=True)

print(f"Ready. Waiting for force > {START_THRESHOLD} N. T to re-tare, Q/ESC to quit.", flush=True)

while True:
    ret, frame = cap.read()
    if not ret:
        time.sleep(0.05)
        continue

    force_smooth = latest_force["smooth"]

    if not recording and force_smooth > START_THRESHOLD:
        start_recording(width, height, force_smooth)

    if recording:
        elapsed       = round(time.monotonic() - rec_start_time, 3)
        displacement  = round(elapsed * SPEED_MM_PER_S, 4)
        rel_force     = round(max(force_smooth - contact_force, 0.0), 3)
        img_path      = os.path.join(frames_dir, f"frame_{frame_count:05d}.jpg")
        cv2.imwrite(img_path, frame)
        writer.write(frame)
        csv_writer.writerow([elapsed, displacement, frame_count, rel_force, img_path])
        frame_count += 1

        # Hard stop at 10 mm
        if displacement >= MAX_INDENT_MM:
            print(f"10 mm reached at {elapsed:.2f}s — stopping.", flush=True)
            stop_recording()
            break

    if recording:
        elapsed      = time.monotonic() - rec_start_time
        displacement = elapsed * SPEED_MM_PER_S
        rel_force    = max(force_smooth - contact_force, 0.0)
        cv2.circle(frame, (18, 22), 10, (0, 0, 255), -1)
        cv2.putText(frame, f"REC  {displacement:.2f}mm  {rel_force:.3f}N", (35, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        cv2.putText(frame, f"Force: {force_smooth:.3f} N", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1) & 0xFF

    if key in (ord('t'), ord('T')):
        try:
            if ser_ref["ser"]:
                ser_ref["ser"].write(b't')
                force_buffer.clear()
                print("Re-tared.", flush=True)
        except Exception:
            pass
    elif key in (ord('q'), ord('Q'), 27):
        break

if recording and writer:
    stop_recording()

cap.release()
cv2.destroyAllWindows()
print("Done.", flush=True)
