"""
Camera Video Capture Script
- Press S to start/stop recording
- Press Q or ESC to quit
"""

import cv2
import os
from datetime import datetime

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "skin_test")
os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open camera. Try changing the camera index in capture.py.")
    exit(1)

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera opened ({width}x{height}). Press S to start/stop recording, Q or ESC to quit.")

recording = False
writer = None
video_path = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame from camera.")
        break

    if recording:
        writer.write(frame)

    cv2.imshow("Camera Preview", frame)

    key = cv2.waitKey(1) & 0xFF

    if key in (ord('s'), ord('S')):
        if not recording:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = os.path.join(OUTPUT_DIR, f"video_{stamp}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(video_path, fourcc, 30, (width, height))
            recording = True
            print(f"Recording: {video_path}")
        else:
            recording = False
            writer.release()
            writer = None
            print(f"Saved: {video_path}")

    elif key in (ord('q'), ord('Q'), 27):
        break

if recording and writer:
    writer.release()
    print(f"Saved: {video_path}")

cap.release()
cv2.destroyAllWindows()
