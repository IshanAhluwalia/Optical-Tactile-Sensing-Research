"""
auto_capture_calib.py

Live auto-capture calibration collector.
Move the board around — images are saved automatically when
the detector finds a good, stable, novel pose.

Status colours:
    RED    — board not detected or too far away
    ORANGE — detected but needs more tilt, or hold still
    GREEN  — good pose, captured / ready to capture

Controls:
    Q  — quit

Usage:
    python auto_capture_calib.py --cam 0
    python auto_capture_calib.py --cam 1
"""

import argparse
import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

# ── Tuning ────────────────────────────────────────────────────────────────────
SAVE_DIR             = Path("calib_images")
CONFIG_FILE          = Path("charuco_config.json")
IMAGE_SIZE           = (640, 480)

MIN_CORNER_AREA      = 400    # px²  — board too far if below this
MAX_COND_NUMBER      = 8_000  # homography cond — rejects perfectly flat poses
STABILITY_FRAMES     = 10     # frames board must stay still before capture
STABILITY_THRESH_PX  = 2.5   # max mean corner drift (px) to count as "still"
MIN_NOVEL_DIST_PX    = 35.0  # min mean corner distance from any prior capture
MIN_CAPTURE_INTERVAL = 2.0   # seconds between captures (give time to move)
TARGET_IMAGES        = 60    # progress target shown in UI
# ─────────────────────────────────────────────────────────────────────────────


def load_config():
    with open(CONFIG_FILE) as f:
        return json.load(f)


def make_objp(cfg):
    inner_cols = cfg["cols"] - 1   # 6
    inner_rows = cfg["rows"] - 1   # 4
    sq  = cfg["square_mm"]
    objp = np.zeros((inner_cols * inner_rows, 1, 3), dtype=np.float32)
    objp[:, 0, :2] = np.mgrid[0:inner_cols, 0:inner_rows].T.reshape(-1, 2) * sq
    return objp, (inner_cols, inner_rows)


def detect_corners(gray, pattern_size):
    ret, corners = cv2.findChessboardCornersSB(
        gray, pattern_size,
        cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
    )
    if not ret:
        ret, corners = cv2.findChessboardCorners(
            gray, pattern_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if ret:
            crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
    if not ret or corners is None:
        return None
    return corners.reshape(-1, 1, 2).astype(np.float32)


def quality_check(corners, objp):
    """
    Returns (ok: bool, message: str).
    Checks board is large enough in the frame.
    """
    pts  = corners[:, 0, :]
    hull = cv2.convexHull(pts)
    area = float(cv2.contourArea(hull))
    if area < MIN_CORNER_AREA:
        return False, f"Too far away — move closer  ({area:.0f} px2)"
    return True, "Good"


def mean_dist(c1, c2):
    return float(np.mean(np.linalg.norm(
        c1.reshape(-1, 2) - c2.reshape(-1, 2), axis=1
    )))


def put_text(img, text, pos, color=(255, 255, 255), scale=0.60, thickness=2):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thickness + 2)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness)


def find_camera(w, h, max_idx=10):
    for i in range(max_idx):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            continue
        cw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ch = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if cw == w and ch == h:
            print(f"  Found {w}x{h} camera at index {i}")
            return cap
        cap.release()
        time.sleep(0.05)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam", type=int, choices=[0, 1], required=True,
                        help="Camera slot to capture for (0 or 1)")
    args = parser.parse_args()

    cfg  = load_config()
    objp, pattern_size = make_objp(cfg)

    save_dir = SAVE_DIR / f"cam{args.cam}"
    save_dir.mkdir(parents=True, exist_ok=True)

    existing    = sorted(save_dir.glob("frame_*.jpg"))
    frame_count = len(existing)
    if frame_count:
        print(f"  {frame_count} existing images found — continuing numbering.")

    print(f"Scanning for {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} camera...")
    cap = find_camera(*IMAGE_SIZE)
    if cap is None:
        print("ERROR: camera not found. Check connection.")
        return

    print(f"\nAuto-capture — CAM {args.cam}")
    print("Move the board around. Images save automatically.")
    print("Q to quit.\n")

    history          = deque(maxlen=STABILITY_FRAMES)
    captured_corners = []   # corners from images taken this session (novelty check)
    last_capture_t   = 0.0
    flash_frames     = 0
    window_title     = f"Auto-Capture — CAM {args.cam}"

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display = frame.copy()
        corners = detect_corners(gray, pattern_size)

        # ── State machine ─────────────────────────────────────────────────────
        if corners is None:
            status       = "Searching for board..."
            status_color = (60, 60, 255)   # red
            history.clear()

        else:
            ok, msg = quality_check(corners, objp)

            if not ok:
                status       = msg
                status_color = (60, 60, 255)   # red
                history.clear()
                cv2.drawChessboardCorners(display, pattern_size, corners, True)

            else:
                history.append(corners.copy())

                # Stability
                if len(history) < STABILITY_FRAMES:
                    pct    = int(100 * len(history) / STABILITY_FRAMES)
                    status = f"Hold still...  {pct}%"
                    status_color = (0, 165, 255)   # orange
                    cv2.drawChessboardCorners(display, pattern_size, corners, True)

                else:
                    ref    = history[-1]
                    stable = all(mean_dist(ref, h) < STABILITY_THRESH_PX for h in history)

                    if not stable:
                        status       = "Hold still..."
                        status_color = (0, 165, 255)
                        cv2.drawChessboardCorners(display, pattern_size, corners, True)

                    else:
                        novel    = (not captured_corners or
                                    all(mean_dist(corners, c) > MIN_NOVEL_DIST_PX
                                        for c in captured_corners))
                        time_ok  = (time.time() - last_capture_t) > MIN_CAPTURE_INTERVAL

                        if not novel:
                            status       = "Move to a new position"
                            status_color = (0, 165, 255)
                            cv2.drawChessboardCorners(display, pattern_size, corners, True)

                        elif not time_ok:
                            status       = "Ready — move board soon"
                            status_color = (0, 220, 0)
                            cv2.drawChessboardCorners(display, pattern_size, corners, True)

                        else:
                            # ── AUTO CAPTURE ──────────────────────────────────
                            frame_count += 1
                            fname = f"frame_{frame_count:03d}.jpg"
                            cv2.imwrite(str(save_dir / fname), frame)
                            captured_corners.append(corners.copy())
                            last_capture_t = time.time()
                            flash_frames   = 12
                            history.clear()
                            print(f"  Saved {fname}  ({frame_count} total)")
                            status       = "CAPTURED!  Move to new position"
                            status_color = (0, 220, 0)
                            cv2.drawChessboardCorners(display, pattern_size, corners, True)

        # ── Green flash ───────────────────────────────────────────────────────
        if flash_frames > 0:
            green  = np.full_like(display, (0, 220, 0))
            alpha  = 0.45 * (flash_frames / 12)
            display = cv2.addWeighted(display, 1 - alpha, green, alpha, 0)
            flash_frames -= 1

        # ── HUD ───────────────────────────────────────────────────────────────
        bar_w  = int(IMAGE_SIZE[0] * min(frame_count, TARGET_IMAGES) / TARGET_IMAGES)
        cv2.rectangle(display, (0, 0), (bar_w, 6), (0, 200, 80), -1)

        put_text(display, f"CAM {args.cam}  |  {frame_count}/{TARGET_IMAGES} images",
                 (10, 25), color=(255, 255, 255))
        put_text(display, status, (10, IMAGE_SIZE[1] - 12), color=status_color)

        cv2.imshow(window_title, display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone. {frame_count} images in {save_dir}/")


if __name__ == "__main__":
    main()
