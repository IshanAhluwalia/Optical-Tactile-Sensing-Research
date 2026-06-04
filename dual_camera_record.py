"""
dual_camera_record.py

Opens two USB cameras (auto-detected at 640x480) and shows a live
side-by-side preview. Press 'q' to quit.

Usage:
    pip install opencv-python
    python dual_camera_record.py
"""

import threading
import time

import cv2
import numpy as np

# ---------------------------------------------------------------------------
TARGET_WIDTH = 640
TARGET_HEIGHT = 480
PREVIEW_HEIGHT = 480
# ---------------------------------------------------------------------------


def find_target_cameras(target_w: int, target_h: int, max_index: int = 10):
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            continue
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if w == target_w and h == target_h:
            found.append(i)
            print(f"  Found {target_w}x{target_h} camera at index {i}")
        if len(found) == 2:
            break
        time.sleep(0.1)
    return found


class CameraThread(threading.Thread):
    def __init__(self, index: int):
        super().__init__(daemon=True)
        self.index = index
        self._frame = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self.opened = False
        self.error = None

        self._cap = cv2.VideoCapture(self.index)
        if not self._cap.isOpened():
            self.error = f"Camera {self.index} could not be opened."
            return

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
        self.opened = True

    def run(self):
        if not self.opened:
            return
        while not self._stop_event.is_set():
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.005)
                continue
            with self._lock:
                self._frame = frame

    def get_frame(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        self._stop_event.set()
        self.join(timeout=2)
        if self.opened:
            self._cap.release()


def resize_to_height(frame: np.ndarray, target_height: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = target_height / h
    return cv2.resize(frame, (int(w * scale), target_height))


def main():
    print(f"Scanning for {TARGET_WIDTH}x{TARGET_HEIGHT} cameras...")
    indices = find_target_cameras(TARGET_WIDTH, TARGET_HEIGHT)
    if len(indices) < 2:
        print(f"ERROR: Found only {len(indices)} camera(s). Check connections.")
        return

    cam0 = CameraThread(indices[0])
    cam1 = CameraThread(indices[1])

    for cam in (cam0, cam1):
        if cam.error:
            print(f"WARNING: {cam.error}")

    if not cam0.opened and not cam1.opened:
        print("No cameras could be opened. Exiting.")
        return

    cam0.start()
    cam1.start()
    print(f"Live preview — camera indices {indices[0]} and {indices[1]}. Press 'q' to quit.")

    placeholder = np.zeros((PREVIEW_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
    cv2.putText(placeholder, "No signal", (10, PREVIEW_HEIGHT // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (80, 80, 80), 2)

    while True:
        f0 = cam0.get_frame() if cam0.opened else None
        f1 = cam1.get_frame() if cam1.opened else None

        display0 = resize_to_height(f0, PREVIEW_HEIGHT) if f0 is not None else placeholder
        display1 = resize_to_height(f1, PREVIEW_HEIGHT) if f1 is not None else placeholder

        cv2.imshow("Dual Camera Preview — press Q to quit", np.hstack((display0, display1)))

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam0.stop()
    cam1.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
