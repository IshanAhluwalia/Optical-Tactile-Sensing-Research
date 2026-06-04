"""
calibrate_camera.py

Runs fisheye calibration on images collected for a single camera.
Uses plain checkerboard corner detection (more robust than ChArUco
for heavily distorted fisheye images).

Usage:
    python calibrate_camera.py --cam 0   # calibrate cam0
    python calibrate_camera.py --cam 1   # calibrate cam1

Outputs:
    calib_results/cam0_calib.npz   (or cam1_calib.npz)
    Contains: K, D, image_size, rms
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# ── Settings ──────────────────────────────────────────────────────────────────
CALIB_DIR   = Path("calib_images")
RESULTS_DIR = Path("calib_results")
CONFIG_FILE = Path("charuco_config.json")
IMAGE_SIZE  = (640, 480)   # (width, height)
# ─────────────────────────────────────────────────────────────────────────────


def load_config():
    with open(CONFIG_FILE) as f:
        return json.load(f)


def make_objp(cfg):
    """
    3-D object points for the inner corners of the checkerboard.
    A COLS x ROWS board has (COLS-1) x (ROWS-1) inner corners.
    """
    inner_cols = cfg["cols"] - 1   # 6
    inner_rows = cfg["rows"] - 1   # 4
    sq         = cfg["square_mm"]

    objp = np.zeros((inner_cols * inner_rows, 1, 3), dtype=np.float32)
    grid = np.mgrid[0:inner_cols, 0:inner_rows].T.reshape(-1, 2)
    objp[:, 0, :2] = grid * sq
    return objp, (inner_cols, inner_rows)


def detect_corners(img, pattern_size):
    """
    Detect checkerboard inner corners with subpixel refinement.
    Returns img_pts shaped (N, 1, 2) for fisheye calibration, or None.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # findChessboardCornersSB has built-in subpixel accuracy — prefer it
    ret, corners = cv2.findChessboardCornersSB(gray, pattern_size,
                                               cv2.CALIB_CB_NORMALIZE_IMAGE |
                                               cv2.CALIB_CB_EXHAUSTIVE)
    if not ret:
        # Fall back to classic detector + manual subpixel refinement
        ret, corners = cv2.findChessboardCorners(gray, pattern_size,
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH |
                                                 cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners  = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    if not ret or corners is None:
        return None

    return corners.reshape(-1, 1, 2).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam", type=int, choices=[0, 1], required=True,
                        help="Which camera to calibrate (0 or 1)")
    args = parser.parse_args()

    cfg  = load_config()
    objp, pattern_size = make_objp(cfg)
    RESULTS_DIR.mkdir(exist_ok=True)

    print(f"Board: {cfg['cols']}x{cfg['rows']} squares  →  "
          f"{pattern_size[0]}x{pattern_size[1]} inner corners  "
          f"({pattern_size[0]*pattern_size[1]} pts/image)")

    img_dir   = CALIB_DIR / f"cam{args.cam}"
    img_paths = sorted(img_dir.glob("frame_*.jpg"))
    print(f"Found {len(img_paths)} images in {img_dir}/\n")

    all_obj_pts, all_img_pts, used, skipped = [], [], [], []

    for path in img_paths:
        img = cv2.imread(str(path))
        if img is None:
            skipped.append((path.name, "could not read file"))
            continue

        img_pts = detect_corners(img, pattern_size)
        if img_pts is None:
            skipped.append((path.name, "corners not found — board may be partially out of frame"))
            continue

        # Pre-check: reject boards that are too small in the frame
        ptsimg = img_pts[:, 0, :].astype(np.float32)
        hull   = cv2.convexHull(ptsimg).reshape(-1, 2)
        area   = cv2.contourArea(hull)
        if area < 200:
            skipped.append((path.name, f"board too small in frame ({area:.0f}px²) — move closer"))
            continue

        all_obj_pts.append(objp.copy())
        all_img_pts.append(img_pts)
        used.append(path.name)

    print(f"Usable : {len(used)}")
    print(f"Skipped: {len(skipped)}")
    for name, reason in skipped:
        print(f"  {name}  — {reason}")

    if len(used) < 10:
        print("\nERROR: Need at least 10 usable images. Capture more and try again.")
        return

    # ── Run fisheye calibration ───────────────────────────────────────────────
    print(f"\nCalibrating on {len(used)} images...")

    # Provide a reasonable initial K — fisheye calibration uses it automatically
    K = np.array([[185.,   0., IMAGE_SIZE[0] / 2.],
                  [  0., 185., IMAGE_SIZE[1] / 2.],
                  [  0.,   0.,               1.  ]])
    D = np.zeros((4, 1))
    # Note: CALIB_CHECK_COND omitted — causes instability with extreme fisheye
    flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC |
             cv2.fisheye.CALIB_FIX_SKEW)

    try:
        rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
            all_obj_pts, all_img_pts,
            IMAGE_SIZE, K, D,
            flags=flags
        )
    except cv2.error:
        # Leave-one-out: find which images break InitExtrinsics and remove them
        print("  Calibration failed — scanning for problematic images (LOO)...")
        culprits = set()
        for i in range(len(all_obj_pts)):
            sub_obj = [all_obj_pts[j] for j in range(len(all_obj_pts)) if j != i]
            sub_img = [all_img_pts[j] for j in range(len(all_img_pts)) if j != i]
            K_t, D_t = K.copy(), D.copy()
            try:
                cv2.fisheye.calibrate(sub_obj, sub_img, IMAGE_SIZE, K_t, D_t, flags=flags)
                culprits.add(i)
            except cv2.error:
                pass

        if culprits:
            for i in sorted(culprits, reverse=True):
                print(f"  Dropping: {used[i]}")
            keep        = [i for i in range(len(all_obj_pts)) if i not in culprits]
            all_obj_pts = [all_obj_pts[i] for i in keep]
            all_img_pts = [all_img_pts[i] for i in keep]
            used        = [used[i]        for i in keep]

        try:
            rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                all_obj_pts, all_img_pts, IMAGE_SIZE, K, D, flags=flags)
        except cv2.error as e:
            print(f"\nCalibration failed even after LOO cleanup:\n  {e}")
            print("Capture more images with varied out-of-plane tilts and retry.")
            return

    # ── Iterative outlier rejection ───────────────────────────────────────────
    # Remove images whose per-image RMS > 2× the median, repeat until stable.
    for iteration in range(5):
        errors = []
        for obj, img, rvec, tvec in zip(all_obj_pts, all_img_pts, rvecs, tvecs):
            projected, _ = cv2.fisheye.projectPoints(obj, rvec, tvec, K, D)
            err = float(np.sqrt(np.mean((projected - img) ** 2)))
            errors.append(err)

        median_err = float(np.median(errors))
        threshold  = max(2.0 * median_err, 1.5)
        outliers   = [used[i] for i, e in enumerate(errors) if e > threshold]

        if not outliers:
            break

        print(f"\nIteration {iteration+1}: removing {len(outliers)} outlier(s) "
              f"(threshold={threshold:.2f}px):")
        for name in outliers:
            print(f"  {name}")

        keep        = [i for i, e in enumerate(errors) if e <= threshold]
        all_obj_pts = [all_obj_pts[i] for i in keep]
        all_img_pts = [all_img_pts[i] for i in keep]
        used        = [used[i]        for i in keep]

        # Keep current K/D as warm start for next iteration (do NOT reset to zeros)
        try:
            rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                all_obj_pts, all_img_pts,
                IMAGE_SIZE, K, D, flags=flags
            )
        except cv2.error:
            break

    print(f"\nFinal image count after outlier removal: {len(used)}")

    # ── Save results ──────────────────────────────────────────────────────────
    out_file = RESULTS_DIR / f"cam{args.cam}_calib.npz"
    np.savez(str(out_file), K=K, D=D, image_size=np.array(IMAGE_SIZE), rms=rms)

    # ── Report ────────────────────────────────────────────────────────────────
    cx_expected = IMAGE_SIZE[0] / 2
    cy_expected = IMAGE_SIZE[1] / 2

    print(f"\n{'='*52}")
    print(f"  Reprojection error (RMS) : {rms:.4f} px")
    print(f"{'='*52}")
    print(f"\nCamera matrix K:")
    print(f"  fx = {K[0,0]:.2f} px")
    print(f"  fy = {K[1,1]:.2f} px")
    print(f"  cx = {K[0,2]:.2f} px   (image centre = {cx_expected:.0f})")
    print(f"  cy = {K[1,2]:.2f} px   (image centre = {cy_expected:.0f})")
    print(f"\nFisheye distortion D:")
    print(f"  k1 = {D[0,0]:.6f}")
    print(f"  k2 = {D[1,0]:.6f}")
    print(f"  k3 = {D[2,0]:.6f}")
    print(f"  k4 = {D[3,0]:.6f}")
    print(f"\nSaved → {out_file}")

    if rms < 0.5:
        verdict = "EXCELLENT"
    elif rms < 1.0:
        verdict = "GOOD — proceed to next step"
    elif rms < 2.0:
        verdict = "ACCEPTABLE — consider adding more tilted images"
    else:
        verdict = "POOR — add images with strong out-of-plane tilt"
    print(f"\nResult: {verdict}")


if __name__ == "__main__":
    main()
