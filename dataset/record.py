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
import os
import sys
import time
import threading
from collections import deque
from datetime import datetime

SERIAL_PORT     = '/dev/tty.usbmodemF412FA6357AC2'
BAUD            = 115200
CAM_INDEX       = 1
OUTPUT_DIR      = os.path.abspath(os.path.dirname(__file__))
START_THRESHOLD = 0.05
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

def start_recording(width, height, baseline_force):
    global recording, csv_file, csv_writer, frame_count, video_path, writer, rec_start_time, contact_force
    stamp          = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path     = os.path.join(OUTPUT_DIR, f"video_{stamp}.mp4")
    csv_path       = os.path.join(OUTPUT_DIR, f"data_{stamp}.csv")
    fourcc         = cv2.VideoWriter_fourcc(*'mp4v')
    writer         = cv2.VideoWriter(video_path, fourcc, ACTUAL_FPS, (width, height))
    csv_file       = open(csv_path, 'w', newline='')
    csv_writer     = csv.writer(csv_file)
    csv_writer.writerow(['time_s', 'displacement_mm', 'frame', 'force_n'])
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
    while len(baseline_samples) < 20:
        try:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if not line or ',' not in line:
                continue
            parts = line.split(',')
            if len(parts) != 4:
                continue
            _, _, _, force = parts
            baseline_samples.append(float(force))
        except (ValueError, serial.SerialException):
            continue
    baseline = sum(baseline_samples) / len(baseline_samples)
    print(f"Baseline: {baseline:.4f} N. Load cell ready.", flush=True)

    while True:
        try:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if not line or ',' not in line:
                continue
            parts = line.split(',')
            if len(parts) != 4:
                continue
            _, _, _, force = parts
            f = abs(float(force) - baseline)
            force_buffer.append(f)
            latest_force["raw"]    = f
            latest_force["smooth"] = sum(force_buffer) / len(force_buffer)
            data_received[0]       = True
        except (ValueError, serial.SerialException):
            continue

threading.Thread(target=serial_thread, daemon=True).start()

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
        writer.write(frame)
        csv_writer.writerow([elapsed, displacement, frame_count, rel_force])
        frame_count += 1

        # Hard stop at 10 mm
        if displacement >= MAX_INDENT_MM:
            print(f"10 mm reached at {elapsed:.2f}s — stopping.", flush=True)
            stop_recording()

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
