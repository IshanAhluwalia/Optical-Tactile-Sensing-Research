"""
Verify stereo calibration by showing rectified frames side-by-side with
horizontal epipolar lines. Lines should pass through the same features
in both images if calibration is correct.

Usage
-----
    python verify_calibration.py --cam0 0 --cam1 1 --rotate1
"""

import cv2
import numpy as np
import argparse
import time
import os
import subprocess
import sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam0",   type=int, default=0)
    ap.add_argument("--cam1",   type=int, default=1)
    ap.add_argument("--rotate0", action="store_true")
    ap.add_argument("--rotate1", action="store_true")
    ap.add_argument("--params", default="params/stereo_params.npz")
    args = ap.parse_args()

    if not os.path.exists(args.params):
        sys.exit(f"Params not found: {args.params}")

    d = np.load(args.params)
    map0x, map0y = d["map0x"], d["map0y"]
    map1x, map1y = d["map1x"], d["map1y"]
    K1, D1 = d["K1"], d["D1"]
    K2, D2 = d["K2"], d["D2"]
    R, T   = d["R"], d["T"]

    baseline_mm = float(np.linalg.norm(T))
    f_px = float(np.mean([K1[0,0], K1[1,1], K2[0,0], K2[1,1]]))
    print(f"Loaded params: baseline={baseline_mm:.1f}mm  focal={f_px:.1f}px")
    print(f"  Depth at 100px disp: {f_px * baseline_mm / 100:.1f}mm")
    print(f"  Depth at 150px disp: {f_px * baseline_mm / 150:.1f}mm")

    print(f"\nOpening cameras {args.cam0} and {args.cam1}...")
    cap0 = cv2.VideoCapture(args.cam0, cv2.CAP_AVFOUNDATION)
    cap1 = cv2.VideoCapture(args.cam1, cv2.CAP_AVFOUNDATION)
    if not cap0.isOpened(): sys.exit(f"Cannot open camera {args.cam0}")
    if not cap1.isOpened(): sys.exit(f"Cannot open camera {args.cam1}")
    cap0.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap1.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("Warming up (5s)...", end=" ", flush=True)
    time.sleep(5)
    for _ in range(10): cap0.read(); cap1.read()
    print("done")

    ret0, img0 = cap0.read()
    ret1, img1 = cap1.read()
    cap0.release(); cap1.release()

    if not (ret0 and ret1):
        sys.exit("Failed to read frames")

    print(f"Frame sizes: cam0={img0.shape[1]}x{img0.shape[0]}  cam1={img1.shape[1]}x{img1.shape[0]}")

    if args.rotate0: img0 = cv2.rotate(img0, cv2.ROTATE_180)
    if args.rotate1: img1 = cv2.rotate(img1, cv2.ROTATE_180)

    rect0 = cv2.remap(img0, map0x, map0y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    rect1 = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # Side-by-side with epipolar lines
    side = np.hstack([rect0, rect1])
    H, W = side.shape[:2]
    for y in range(0, H, 30):
        cv2.line(side, (0, y), (W, y), (0, 255, 0), 1)
    # Divider line
    cv2.line(side, (rect0.shape[1], 0), (rect0.shape[1], H), (0, 0, 255), 2)

    cv2.putText(side, "LEFT (cam0) rectified", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(side, "RIGHT (cam1) rectified", (rect0.shape[1]+10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # Also save raw side-by-side for comparison (resize to match if needed)
    if img0.shape[0] != img1.shape[0]:
        img1 = cv2.resize(img1, (img0.shape[1], img0.shape[0]))
    raw_side = np.hstack([img0, img1])

    os.makedirs("output", exist_ok=True)
    cv2.imwrite("output/verify_raw.png", raw_side)
    cv2.imwrite("output/verify_epipolar.png", side)
    print("\nSaved:")
    print("  output/verify_raw.png     — raw frames side by side")
    print("  output/verify_epipolar.png — rectified with epipolar lines")
    print("\nCheck that features (corners, edges) lie on the same green")
    print("horizontal line in both left and right rectified images.")

    subprocess.run(["open", "output/verify_epipolar.png"])

if __name__ == "__main__":
    main()
