"""
Validate stereo calibration using saved ChArUco calibration pairs.

Uses params/stereo_pairs.npz (2D detections in both cameras) and
params/stereo_params.npz (calibration) to check:

  1. Epipolar alignment  — |y0_rect - y1_rect| per matched corner
  2. Triangulation error — distance from triangulated 3D point to
                           known board position (ground truth)
  3. Scale accuracy      — reconstructed square spacing vs known 20 mm
  4. Flatness            — RMS distance of triangulated board corners
                           from a best-fit plane

Usage
-----
    python validate_stereo.py
"""

import cv2
import numpy as np


SQUARE_MM = 20.0   # known ChArUco square size


def load(path):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def undistort_rectify_points(pts_raw, K, D, R_rect, P_rect):
    """
    Fisheye-undistort then apply rectification homography to 2D points.
    pts_raw : (N, 1, 2) or (N, 2)
    Returns  : (N, 2) in rectified pixel coords
    """
    pts = pts_raw.reshape(-1, 1, 2).astype(np.float64)
    # undistortPoints with R and P gives rectified pixel coordinates
    out = cv2.fisheye.undistortPoints(pts, K, D, R=R_rect, P=P_rect)
    return out.reshape(-1, 2)


def triangulate(pts0_rect, pts1_rect, P1, P2):
    """Return (N, 3) in the reference (left camera) frame."""
    pts4d = cv2.triangulatePoints(P1, P2,
                                   pts0_rect.T.astype(np.float64),
                                   pts1_rect.T.astype(np.float64))
    return (pts4d[:3] / pts4d[3]).T.astype(np.float32)


def fit_plane(pts3d):
    """RMS distance from a best-fit plane through pts3d (N, 3)."""
    c = pts3d.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts3d - c)
    normal = Vt[-1]
    residuals = (pts3d - c) @ normal
    return float(np.sqrt(np.mean(residuals ** 2)))


def board_scale_error(pts_board_3d, pts_tri_3d):
    """
    Compare spacing between adjacent corners in board coordinates
    vs triangulated coordinates.
    Returns (measured_spacing_mm, error_mm).
    """
    if len(pts_board_3d) < 2:
        return None, None
    # All pairwise distances between adjacent board points
    dists_known = []
    dists_meas  = []
    n = len(pts_board_3d)
    for i in range(n - 1):
        dk = np.linalg.norm(pts_board_3d[i] - pts_board_3d[i + 1])
        dm = np.linalg.norm(pts_tri_3d[i]   - pts_tri_3d[i + 1])
        if SQUARE_MM * 0.5 < dk < SQUARE_MM * 1.5:   # adjacent corners only
            dists_known.append(dk)
            dists_meas.append(dm)
    if not dists_meas:
        return None, None
    mean_meas  = float(np.mean(dists_meas))
    mean_known = float(np.mean(dists_known))
    return mean_meas, mean_meas - mean_known


def main():
    print("Loading calibration data...")
    params = load("params/stereo_params.npz")
    pairs  = load("params/stereo_pairs.npz")

    K1  = params["K1"].astype(np.float64)
    D1  = params["D1"].astype(np.float64)
    K2  = params["K2"].astype(np.float64)
    D2  = params["D2"].astype(np.float64)
    R1  = params["R1"].astype(np.float64)
    R2  = params["R2"].astype(np.float64)
    P1  = params["P1"].astype(np.float64)
    P2  = params["P2"].astype(np.float64)

    objpoints  = pairs["objpoints"]   # (N,) of (1, n_corners, 3)
    imgpoints0 = pairs["imgpoints0"]  # (N,) of (1, n_corners, 2)
    imgpoints1 = pairs["imgpoints1"]  # (N,) of (1, n_corners, 2)

    n_frames = len(objpoints)
    print(f"Frames: {n_frames}\n")

    all_epi   = []
    all_tri   = []
    all_flat  = []
    all_scale = []

    for i in range(n_frames):
        obj = objpoints[i].reshape(-1, 3).astype(np.float64)   # known 3D mm
        p0  = imgpoints0[i].reshape(-1, 1, 2)
        p1  = imgpoints1[i].reshape(-1, 1, 2)
        n   = len(obj)

        if n < 4:
            continue

        # Undistort + rectify 2D detections into rectified pixel coords
        r0 = undistort_rectify_points(p0, K1, D1, R1, P1)
        r1 = undistort_rectify_points(p1, K2, D2, R2, P2)

        # 1. Epipolar error
        epi = np.abs(r0[:, 1] - r1[:, 1])

        # 2. Triangulate
        pts3d = triangulate(r0, r1, P1, P2)

        # Filter invalid depth
        valid = (pts3d[:, 2] > 0) & (pts3d[:, 2] < 1000)
        if valid.sum() < 4:
            continue
        pts3d_v = pts3d[valid]
        obj_v   = obj[valid]
        epi_v   = epi[valid]

        # 3. Triangulation error vs known board coords
        # Align via rigid transform (Procrustes) then measure RMSE
        c_tri = pts3d_v.mean(axis=0)
        c_obj = obj_v.mean(axis=0)
        A = pts3d_v - c_tri
        B = obj_v - c_obj
        H = A.T @ B
        U, _, Vt = np.linalg.svd(H)
        Rot = Vt.T @ U.T
        if np.linalg.det(Rot) < 0:
            Vt[-1] *= -1
            Rot = Vt.T @ U.T
        s = np.trace(Rot @ H) / np.sum(A ** 2)
        pts3d_aligned = s * (pts3d_v - c_tri) @ Rot.T + c_obj
        tri_err = float(np.sqrt(np.mean(np.sum((pts3d_aligned - obj_v)**2, axis=1))))

        # 4. Flatness
        flat = fit_plane(pts3d_v)

        # 5. Scale
        meas, scale_err = board_scale_error(obj_v, pts3d_v)

        all_epi.append(float(epi_v.mean()))
        all_tri.append(tri_err)
        all_flat.append(flat)
        if scale_err is not None:
            all_scale.append(abs(scale_err))

        print(f"  Frame {i+1:2d}  corners={valid.sum():2d}  "
              f"epi={epi_v.mean():.2f}px  "
              f"tri_err={tri_err:.2f}mm  "
              f"flat={flat:.2f}mm  "
              + (f"scale_err={scale_err:.2f}mm" if scale_err is not None else ""))

    print()
    print("=" * 55)
    print("STEREO VALIDATION SUMMARY")
    print("=" * 55)

    def stat(name, vals, unit, target):
        if not vals: return
        med = np.median(vals)
        mx  = np.max(vals)
        ok  = "OK" if med < target else "NEEDS ATTENTION"
        print(f"  {name:<22} median={med:.2f}{unit}  max={mx:.2f}{unit}  "
              f"target<{target}{unit}  [{ok}]")

    stat("Epipolar error",       all_epi,   "px", 1.5)
    stat("Triangulation error",  all_tri,   "mm", 2.0)
    stat("Flatness RMS",         all_flat,  "mm", 1.5)
    stat("Scale error",          all_scale, "mm", 1.0)

    print()
    if all_epi:
        epi_med = np.median(all_epi)
        if epi_med < 1.0:
            print("  Epipolar: EXCELLENT — rectification is very accurate")
        elif epi_med < 1.5:
            print("  Epipolar: GOOD — minor misalignment, acceptable for stereo")
        else:
            print("  Epipolar: POOR — rectification has significant error")
            print("    → Consider re-running stereo calibration with more pairs")

    if all_tri:
        tri_med = np.median(all_tri)
        if tri_med < 1.0:
            print("  Triangulation: EXCELLENT — 3D reconstruction is sub-mm accurate")
        elif tri_med < 2.0:
            print("  Triangulation: GOOD — 3D reconstruction is within 2mm")
        else:
            print("  Triangulation: POOR — 3D points are inaccurate")
            print("    → Stereo extrinsics may need recalibration")

    if all_flat:
        flat_med = np.median(all_flat)
        if flat_med < 0.5:
            print("  Flatness: EXCELLENT — flat board reconstructs as flat")
        elif flat_med < 1.5:
            print("  Flatness: ACCEPTABLE")
        else:
            print("  Flatness: POOR — flat board looks curved in reconstruction")


if __name__ == "__main__":
    main()
