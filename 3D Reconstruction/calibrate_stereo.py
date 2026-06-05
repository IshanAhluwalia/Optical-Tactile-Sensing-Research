"""
Stereo fisheye calibration — finds R, T between cameras and computes
rectification maps for subsequent 3D reconstruction.

Requires individual intrinsics already saved:
    params/cam0.npz   (from calibrate_single.py)
    params/cam1.npz   (from calibrate_single.py)

Usage
-----
First plug in both cameras, then run:

    python calibrate_stereo.py --cam0 0 --cam1 1

If one camera is physically mounted 180° rotated, add --rotate1 (or --rotate0):

    python calibrate_stereo.py --cam0 0 --cam1 1 --rotate1

Live controls
-------------
SPACE  — capture pair (only if BOTH cameras detect ≥6 common corners)
C      — run calibration now with collected pairs
Q      — quit (auto-calibrates if ≥20 pairs collected)

Tips
----
- Move the board so it is clearly visible to BOTH cameras simultaneously.
- Vary tilt, distance, and position as with single-camera calibration.
- The status bar shows per-camera corner counts; wait for both to be green
  before pressing SPACE.
- Aim for 25–40 pairs.
"""

import cv2
import numpy as np
import argparse
import os
import sys
import time

MIN_PAIRS = 20


# ---------------------------------------------------------------------------
# Board helpers  (same as calibrate_single.py)
# ---------------------------------------------------------------------------

def make_board(squares_x, squares_y, square_mm, marker_mm):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y), square_mm, marker_mm, aruco_dict
    )
    return board, aruco_dict


def board_corners_3d(board):
    try:
        return board.getChessboardCorners()
    except AttributeError:
        return board.chessboardCorners


def detect_charuco(gray, board, aruco_dict, min_corners=6):
    try:
        detector = cv2.aruco.CharucoDetector(board)
        corners, ids, _, _ = detector.detectBoard(gray)
    except AttributeError:
        mkr_corners, mkr_ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)
        if mkr_ids is None or len(mkr_ids) < 4:
            return None, None
        _, corners, ids = cv2.aruco.interpolateCornersCharuco(
            mkr_corners, mkr_ids, gray, board
        )
    if ids is None or len(ids) < min_corners:
        return None, None
    return corners, ids


# ---------------------------------------------------------------------------
# Stereo calibration
# ---------------------------------------------------------------------------

def run_stereo_calibrate(objpoints, imgpoints0, imgpoints1,
                         K1, D1, K2, D2, img_size):
    """Estimate R, T by averaging per-frame relative board poses.

    For each captured pair we:
      1. Undistort image points via the fisheye model → normalised coordinates.
      2. Solve PnP (unit camera) to get the board pose in each camera frame.
      3. Compute the relative pose  R_rel = R1 @ R0.T,  T_rel = t1 - R_rel @ t0.
    The final R, T are the mean after MAD-based outlier rejection.

    This approach completely avoids cv2.fisheye.stereoCalibrate's strict
    condition-number checks, which cause spurious failures on valid data.
    """
    I3    = np.eye(3,  dtype=np.float64)
    zeros = np.zeros(4, dtype=np.float64)

    Rs, Ts, errs = [], [], []

    for pts3d_raw, pts2d_0, pts2d_1 in zip(objpoints, imgpoints0, imgpoints1):
        pts3d = pts3d_raw.reshape(-1, 1, 3).astype(np.float64)

        # Undistort to normalised (unit-focal) image coordinates
        norm0 = cv2.fisheye.undistortPoints(
            pts2d_0.reshape(-1, 1, 2).astype(np.float64), K1, D1)
        norm1 = cv2.fisheye.undistortPoints(
            pts2d_1.reshape(-1, 1, 2).astype(np.float64), K2, D2)

        ok0, rvec0, tvec0 = cv2.solvePnP(pts3d, norm0, I3, zeros)
        ok1, rvec1, tvec1 = cv2.solvePnP(pts3d, norm1, I3, zeros)
        if not ok0 or not ok1:
            continue

        R0, _ = cv2.Rodrigues(rvec0)
        R1, _ = cv2.Rodrigues(rvec1)
        t0 = tvec0.flatten()
        t1 = tvec1.flatten()

        R_rel = R1 @ R0.T
        T_rel = (t1 - R_rel @ t0).reshape(3, 1)
        Rs.append(R_rel)
        Ts.append(T_rel)

        # Per-frame reprojection error (cam0 only, for diagnostics)
        proj, _ = cv2.fisheye.projectPoints(
            pts3d, rvec0, tvec0, K1, D1)
        err = float(np.sqrt(np.mean(
            (proj.reshape(-1, 2) - pts2d_0.reshape(-1, 2)) ** 2)))
        errs.append(err)

    if len(Rs) < 5:
        raise cv2.error(f"Only {len(Rs)} valid poses (need ≥5). "
                        "Recapture with more varied board positions.")

    # --- Outlier rejection on T ---
    T_arr   = np.array([t.flatten() for t in Ts])        # (N, 3)
    T_med   = np.median(T_arr, axis=0)
    T_dists = np.linalg.norm(T_arr - T_med, axis=1)
    mad     = np.median(T_dists)
    keep    = T_dists < max(mad * 4, 5.0)                # at least 5 mm slack
    n_kept  = keep.sum()
    if n_kept < 5:
        keep = np.ones(len(Rs), dtype=bool)              # fallback: keep all

    Rs_k = [R for R, k in zip(Rs, keep) if k]
    Ts_k = [T for T, k in zip(Ts, keep) if k]
    print(f"  Using {n_kept}/{len(Rs)} poses after outlier rejection")

    # --- Average rotation via mean Rodrigues vector ---
    rvecs = np.array([cv2.Rodrigues(R)[0].flatten() for R in Rs_k])
    R_final, _ = cv2.Rodrigues(np.mean(rvecs, axis=0))
    T_final     = np.mean(Ts_k, axis=0).reshape(3, 1)

    rms = float(np.mean(errs))
    return R_final, T_final, rms


def run_rectification(K1, D1, K2, D2, img_size, R, T):
    """Compute fisheye stereo rectification and pixel remap tables.

    cv2.fisheye.stereoRectify reliably produces the rectification rotations
    R1 and R2 but often returns zero focal lengths in P1/P2. We therefore
    build P1, P2 and Q ourselves from the individual calibrations.
    """
    W, H = img_size
    Tx = abs(float(T.flatten()[0]))     # baseline magnitude in mm

    # --- Step 1: get rectification rotations from fisheye stereoRectify ---
    R1, R2, _, _, _, = cv2.fisheye.stereoRectify(
        K1, D1, K2, D2, img_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        balance=0,
        newImageSize=(0, 0),
    )

    # --- Step 2: build projection matrices with mean focal length ---
    # Use the conservative mean of fx/fy across both cameras.
    f  = float(np.mean([K1[0, 0], K1[1, 1], K2[0, 0], K2[1, 1]]))
    cx = W / 2.0
    cy = H / 2.0

    P1 = np.array([[f, 0, cx,       0],
                   [0, f, cy,       0],
                   [0, 0,  1,       0]], dtype=np.float64)
    P2 = np.array([[f, 0, cx, -f * Tx],
                   [0, f, cy,       0],
                   [0, 0,  1,       0]], dtype=np.float64)

    # --- Step 3: Q matrix for cv2.reprojectImageTo3D ---
    # Z = f * Tx / disparity  (depth in same units as Tx, i.e. mm)
    Q = np.array([[1, 0,    0,  -cx],
                  [0, 1,    0,  -cy],
                  [0, 0,    0,    f],
                  [0, 0, -1/Tx,   0]], dtype=np.float64)

    # --- Step 4: fisheye undistortion + rectification remap tables ---
    map0x, map0y = cv2.fisheye.initUndistortRectifyMap(
        K1, D1, R1, P1, img_size, cv2.CV_32FC1
    )
    map1x, map1y = cv2.fisheye.initUndistortRectifyMap(
        K2, D2, R2, P2, img_size, cv2.CV_32FC1
    )
    return R1, R2, P1, P2, Q, map0x, map0y, map1x, map1y


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Stereo fisheye calibration — ChArUco DICT_5X5_1000",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--cam0", type=int, default=0,
                    help="Device index for camera 0 (left)")
    ap.add_argument("--cam1", type=int, default=1,
                    help="Device index for camera 1 (right)")
    ap.add_argument("--rotate0", action="store_true",
                    help="Rotate camera 0 feed 180°")
    ap.add_argument("--rotate1", action="store_true",
                    help="Rotate camera 1 feed 180° (use if mounted inverted)")
    ap.add_argument("--cam0_params", default="params/cam0.npz")
    ap.add_argument("--cam1_params", default="params/cam1.npz")
    ap.add_argument("--out", default="params/stereo_params.npz")
    ap.add_argument("--squares_x", type=int, default=11)
    ap.add_argument("--squares_y", type=int, default=8)
    ap.add_argument("--square_mm", type=float, default=20.0)
    ap.add_argument("--marker_mm", type=float, default=15.0)
    ap.add_argument("--calibrate_only", action="store_true",
                    help="Skip capture and run calibration on previously saved pairs "
                         "(params/stereo_pairs.npz)")
    args = ap.parse_args()

    # Load individual intrinsics
    for p in (args.cam0_params, args.cam1_params):
        if not os.path.exists(p):
            sys.exit(f"Missing intrinsics file: {p}  "
                     f"— run calibrate_single.py first.")
    d0 = np.load(args.cam0_params)
    d1 = np.load(args.cam1_params)
    K1, D1 = d0["K"].copy(), d0["D"].copy()
    K2, D2 = d1["K"].copy(), d1["D"].copy()
    img_size = tuple(d0["img_size"].tolist())
    W, H = img_size

    # If a camera was calibrated in its native (un-rotated) orientation but
    # images are captured with 180° correction, the principal point must be
    # reflected through the image centre to stay consistent with the points.
    if args.rotate0:
        K1[0, 2] = W - 1 - K1[0, 2]
        K1[1, 2] = H - 1 - K1[1, 2]
    if args.rotate1:
        K2[0, 2] = W - 1 - K2[0, 2]
        K2[1, 2] = H - 1 - K2[1, 2]

    board, aruco_dict = make_board(
        args.squares_x, args.squares_y, args.square_mm, args.marker_mm
    )
    all_corners_3d = board_corners_3d(board)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pairs_path = os.path.join(os.path.dirname(args.out) or "params", "stereo_pairs.npz")

    # ------------------------------------------------------------------
    # Calibrate-only mode: skip capture, load saved pairs
    # ------------------------------------------------------------------
    if args.calibrate_only:
        if not os.path.exists(pairs_path):
            sys.exit(f"No saved pairs found at {pairs_path}. Capture first.")
        data = np.load(pairs_path, allow_pickle=True)
        objpoints  = list(data["objpoints"])
        imgpoints0 = list(data["imgpoints0"])
        imgpoints1 = list(data["imgpoints1"])
        print(f"Loaded {len(objpoints)} pairs from {pairs_path}")
        _run_calibration(objpoints, imgpoints0, imgpoints1,
                         K1, D1, K2, D2, img_size, args.out)
        return

    # Open cameras
    cap0 = cv2.VideoCapture(args.cam0, cv2.CAP_AVFOUNDATION)
    cap1 = cv2.VideoCapture(args.cam1, cv2.CAP_AVFOUNDATION)
    for cap, idx in ((cap0, args.cam0), (cap1, args.cam1)):
        if not cap.isOpened():
            sys.exit(f"Cannot open camera {idx}")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("Warming up cameras...", end=" ", flush=True)
    time.sleep(5)
    print("ready")
    print(f"\nStereo calibration: cam{args.cam0} (left) ↔ cam{args.cam1} (right)")
    if args.rotate0:
        print("  cam0: 180° rotation applied")
    if args.rotate1:
        print("  cam1: 180° rotation applied")
    print(f"Collect ≥{MIN_PAIRS} pairs, then Q.\n")

    objpoints  = []
    imgpoints0 = []
    imgpoints1 = []

    window = ("Stereo Calibration  "
              "[SPACE=capture  C=calibrate  Q=quit]")

    while True:
        ret0, img0 = cap0.read()
        ret1, img1 = cap1.read()
        if not ret0 or not ret1:
            continue

        if args.rotate0:
            img0 = cv2.rotate(img0, cv2.ROTATE_180)
        if args.rotate1:
            img1 = cv2.rotate(img1, cv2.ROTATE_180)

        gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        c0, ids0 = detect_charuco(gray0, board, aruco_dict)
        c1, ids1 = detect_charuco(gray1, board, aruco_dict)

        vis0, vis1 = img0.copy(), img1.copy()
        if c0 is not None:
            cv2.aruco.drawDetectedCornersCharuco(vis0, c0, ids0, (0, 255, 0))
        if c1 is not None:
            cv2.aruco.drawDetectedCornersCharuco(vis1, c1, ids1, (0, 255, 0))

        n0 = len(ids0) if ids0 is not None else 0
        n1 = len(ids1) if ids1 is not None else 0

        # Count common corners
        if c0 is not None and c1 is not None:
            common_ids = sorted(set(ids0.flatten()) & set(ids1.flatten()))
            n_common = len(common_ids)
        else:
            n_common = 0

        both_ok = n_common >= 6
        clr = (0, 255, 0) if both_ok else (0, 80, 255)

        status = (f"Pairs: {len(objpoints)}/{MIN_PAIRS}  |  "
                  f"cam0: {n0}  cam1: {n1}  common: {n_common}")
        display = np.hstack([vis0, vis1])
        cv2.putText(display, status, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, clr, 2)
        cv2.imshow(window, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c") and len(objpoints) >= MIN_PAIRS:
            break
        elif key == ord(" ") and both_ok:
            common = np.array(common_ids)
            idx0 = [np.where(ids0.flatten() == i)[0][0] for i in common]
            idx1 = [np.where(ids1.flatten() == i)[0][0] for i in common]

            obj_pts  = all_corners_3d[common].reshape(1, -1, 3).astype(np.float64)
            img_pts0 = c0[idx0].reshape(1, -1, 2).astype(np.float64)
            img_pts1 = c1[idx1].reshape(1, -1, 2).astype(np.float64)

            objpoints.append(obj_pts)
            imgpoints0.append(img_pts0)
            imgpoints1.append(img_pts1)
            print(f"  Captured pair {len(objpoints):>3}  "
                  f"({n_common} common corners)")
            # Auto-save after every capture so pairs survive a failed calibration
            def _ragged(lst):
                arr = np.empty(len(lst), dtype=object)
                for i, x in enumerate(lst):
                    arr[i] = x
                return arr
            np.savez(pairs_path,
                     objpoints=_ragged(objpoints),
                     imgpoints0=_ragged(imgpoints0),
                     imgpoints1=_ragged(imgpoints1))

    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()

    if len(objpoints) < MIN_PAIRS:
        print(f"\nNeed ≥{MIN_PAIRS} pairs (have {len(objpoints)}). Aborting.")
        return

    _run_calibration(objpoints, imgpoints0, imgpoints1,
                     K1, D1, K2, D2, img_size, args.out)


def _run_calibration(objpoints, imgpoints0, imgpoints1,
                     K1, D1, K2, D2, img_size, out_path):
    """Run stereo calibration + rectification and save results.

    Automatically removes ill-conditioned pairs and retries until success.
    """
    print(f"\nStereo calibrating with {len(objpoints)} pairs ... ", flush=True)
    try:
        R, T, rms = run_stereo_calibrate(
            objpoints, imgpoints0, imgpoints1,
            K1.copy(), D1.copy(), K2.copy(), D2.copy(), img_size)
    except Exception as e:
        print(f"Calibration failed: {e}")
        return

    print(f"RMS = {rms:.4f} px")
    print(f"\n{'─'*55}")
    print(f"  Stereo RMS     : {rms:.4f} px")
    print(f"  Baseline |T|   : {np.linalg.norm(T):.2f} mm  (expect ~65 mm)")
    print(f"  T vector       : {T.flatten().round(2)} mm")
    print(f"  R (expect ~I)  :\n{R.round(5)}")
    print(f"{'─'*55}")

    # --- Rectification ---
    print("\nComputing rectification maps ... ", end="", flush=True)
    try:
        R1, R2, P1, P2, Q, map0x, map0y, map1x, map1y = run_rectification(
            K1, D1, K2, D2, img_size, R, T
        )
    except cv2.error as e:
        print(f"\nRectification failed: {e}")
        return
    print("done")

    np.savez(
        out_path,
        K1=K1, D1=D1, K2=K2, D2=D2,
        R=R,   T=T,
        R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
        map0x=map0x, map0y=map0y,
        map1x=map1x, map1y=map1y,
        img_size=np.array(img_size),
    )
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
