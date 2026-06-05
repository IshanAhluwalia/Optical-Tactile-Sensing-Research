import cv2
import numpy as np
import argparse
import os
import sys


# Create Board

def make_board(squares_x: int, squares_y: int,
               square_mm: float, marker_mm: float):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y), square_mm, marker_mm, aruco_dict
    )
    return board, aruco_dict

def board_corners_3d(board) -> np.ndarray:
    try:
        return board.getChessboardCorners()
    except AttributeError:
        return board.chessboardCorners


# ---------------------------------------------------------------------------
# ChArUco detection  (handles both old and new OpenCV APIs)
# ---------------------------------------------------------------------------

def detect_charuco(gray: np.ndarray, board, aruco_dict,
                   min_corners: int = 6):
    """Detect ChArUco corners in a greyscale image.

    Returns
    -------
    (corners, ids) : (N×1×2 float32, N×1 int32) or (None, None) on failure.
    """
    try:
        # OpenCV 4.7+ unified detector API
        detector = cv2.aruco.CharucoDetector(board)
        charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
    except AttributeError:
        # OpenCV 4.x legacy API
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)
        if ids is None or len(ids) < 4:
            return None, None
        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, board
        )

    if charuco_ids is None or len(charuco_ids) < min_corners:
        return None, None
    return charuco_corners, charuco_ids


# ---------------------------------------------------------------------------
# Fisheye calibration
# ---------------------------------------------------------------------------

def run_fisheye_calibration(objpoints: list, imgpoints: list,
                             img_size: tuple):
    """Calibrate a fisheye camera (Kannala-Brandt model, 4 coefficients).

    Parameters
    ----------
    objpoints : list of (1, N_i, 3) float64 arrays — 3-D board corner coords.
    imgpoints : list of (1, N_i, 2) float64 arrays — detected image corners.
    img_size  : (width, height) in pixels.

    Returns
    -------
    K   : (3, 3) camera matrix.
    D   : (4, 1) distortion coefficients [k1, k2, k3, k4].
    rms : RMS reprojection error in pixels.
    """
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC |
        cv2.fisheye.CALIB_CHECK_COND |
        cv2.fisheye.CALIB_FIX_SKEW
    )
    rms, K, D, _, _ = cv2.fisheye.calibrate(
        objpoints, imgpoints, img_size, K, D, flags=flags
    )
    return K, D, rms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Single-camera fisheye calibration — ChArUco DICT_5X5_1000",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--cam", type=int, default=0,
                    help="Camera device index")
    ap.add_argument("--rotate180", action="store_true",
                    help="Rotate the live feed 180° (use for camera 1 which is "
                         "mounted inverted)")
    ap.add_argument("--out", default="params/cam0.npz",
                    help="Output path for calibration parameters")
    ap.add_argument("--squares_x", type=int, default=11,
                    help="Number of squares along board width")
    ap.add_argument("--squares_y", type=int, default=8,
                    help="Number of squares along board height")
    ap.add_argument("--square_mm", type=float, default=20.0,
                    help="Physical size of one checkerboard square in mm")
    ap.add_argument("--marker_mm", type=float, default=15.0,
                    help="Physical size of the ArUco marker inside each square in mm")
    ap.add_argument("--width",  type=int, default=640,
                    help="Capture width (pixels)")
    ap.add_argument("--height", type=int, default=480,
                    help="Capture height (pixels)")
    ap.add_argument("--generate_board", type=str, default=None,
                    help="Save a printable board PNG to this path and exit")
    args = ap.parse_args()

    board, aruco_dict = make_board(
        args.squares_x, args.squares_y, args.square_mm, args.marker_mm
    )

    # ------------------------------------------------------------------
    # Board generation mode
    # ------------------------------------------------------------------
    if args.generate_board:
        board_img = board.generateImage((1400, 900), marginSize=20, borderBits=1)
        cv2.imwrite(args.generate_board, board_img)
        print(f"Board saved → {args.generate_board}")
        print(f"  {args.squares_x} × {args.squares_y} squares, "
              f"{args.square_mm} mm squares, {args.marker_mm} mm markers")
        print(f"  Interior ChArUco corners: "
              f"{(args.squares_x - 1) * (args.squares_y - 1)}")
        return

    # ------------------------------------------------------------------
    # Capture mode
    # ------------------------------------------------------------------
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        sys.exit(f"Cannot open camera {args.cam}")

    img_size = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    all_corners_3d = board_corners_3d(board)   # shape (N_total, 3)

    objpoints: list = []
    imgpoints: list = []

    K_current = None          # set after calibration runs
    D_current = None
    show_undistorted = False

    window = f"Camera {args.cam} — Fisheye Calibration  [SPACE=capture  C=calibrate  U=undistort  Q=quit]"

    print(f"\nCamera {args.cam} | {img_size[0]}×{img_size[1]}"
          + ("  (180° rotation applied)" if args.rotate180 else ""))
    print(f"Board: {args.squares_x}×{args.squares_y} squares | "
          f"{args.square_mm} mm sq | {args.marker_mm} mm marker")
    print(f"Collect ≥{MIN_FRAMES} frames, then press C or Q.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed.")
            break

        if args.rotate180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids = detect_charuco(gray, board, aruco_dict)

        # ---- build display frame ----
        if show_undistorted and K_current is not None:
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                K_current, D_current, np.eye(3), K_current, img_size, cv2.CV_32FC1
            )
            vis = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
            cv2.putText(vis, "UNDISTORTED", (10, img_size[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        else:
            vis = frame.copy()

        if corners is not None:
            cv2.aruco.drawDetectedCornersCharuco(vis, corners, ids, (0, 255, 0))
            n_det   = len(ids)
            det_clr = (0, 255, 0)
        else:
            n_det   = 0
            det_clr = (0, 80, 255)

        hdr = (f"Frames: {len(objpoints)}/{MIN_FRAMES}  |  "
               f"Corners detected: {n_det}")
        cv2.putText(vis, hdr, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, det_clr, 2)
        cv2.imshow(window, vis)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord(" ") and corners is not None:
            # Store only the detected subset of 3-D corners
            obj_pts = all_corners_3d[ids.flatten()].reshape(1, -1, 3).astype(np.float64)
            img_pts = corners.reshape(1, -1, 2).astype(np.float64)
            objpoints.append(obj_pts)
            imgpoints.append(img_pts)
            print(f"  Captured frame {len(objpoints):>3}  ({n_det} corners)")

        elif key == ord("c") and len(objpoints) >= MIN_FRAMES:
            print(f"\nCalibrating with {len(objpoints)} frames … ", end="", flush=True)
            try:
                K_current, D_current, rms = run_fisheye_calibration(
                    objpoints, imgpoints, img_size
                )
                print(f"RMS = {rms:.4f} px")
                _print_results(K_current, D_current, rms, args.out)
                _save(K_current, D_current, img_size, rms, args)
            except cv2.error as e:
                print(f"\nCalibration failed: {e}")
                print("Try capturing more diverse poses (tilt, distance, corners).")

        elif key == ord("u") and K_current is not None:
            show_undistorted = not show_undistorted

    cap.release()
    cv2.destroyAllWindows()

    # Auto-calibrate on quit if we have enough frames and haven't yet
    if len(objpoints) >= MIN_FRAMES and K_current is None:
        print(f"\nCalibrating with {len(objpoints)} frames … ", end="", flush=True)
        try:
            K_current, D_current, rms = run_fisheye_calibration(
                objpoints, imgpoints, img_size
            )
            print(f"RMS = {rms:.4f} px")
            _print_results(K_current, D_current, rms, args.out)
            _save(K_current, D_current, img_size, rms, args)
        except cv2.error as e:
            print(f"\nCalibration failed: {e}")
    elif len(objpoints) < MIN_FRAMES:
        print(f"\nNot enough frames ({len(objpoints)}/{MIN_FRAMES}). Calibration skipped.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_results(K, D, rms, out_path):
    print(f"\n{'─'*55}")
    print(f"  RMS reprojection error : {rms:.4f} px")
    print(f"  fx = {K[0,0]:.2f}  fy = {K[1,1]:.2f}  "
          f"cx = {K[0,2]:.2f}  cy = {K[1,2]:.2f}")
    print(f"  Distortion (k1..k4)   : {D.flatten().round(6)}")
    print(f"  Output                 : {out_path}")
    print(f"{'─'*55}")
    if rms > 1.5:
        print("  WARNING: RMS > 1.5 px — consider recapturing with more "
              "varied poses.")
    elif rms < 0.5:
        print("  Excellent calibration.")
    else:
        print("  Good calibration.")


def _save(K, D, img_size, rms, args):
    np.savez(
        args.out,
        K=K, D=D,
        img_size=np.array(img_size),
        rms=rms,
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_mm=args.square_mm,
        marker_mm=args.marker_mm,
    )
    print(f"  Saved → {args.out}")


if __name__ == "__main__":
    main()
