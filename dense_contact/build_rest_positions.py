"""
Precompute the undeformed skin surface positions at the 32×64 structured grid
used by PointCloudNet and generate_pointclouds.py.

Output
------
    dense_contact/rest_positions.npy  — (2048, 3) float32  [X, Y, Z in mm]

The grid is 32 rows (along sensor length) × 64 cols (across width), sampled
evenly from the full 1000×200 surface mesh defined by the beam-deflection CSV.
Point ordering is row-major: point k = (k // 64, k % 64).

Usage
-----
    python dense_contact/build_rest_positions.py
"""

import os
import numpy as np
import pandas as pd

_HERE    = os.path.dirname(os.path.abspath(__file__))
GEOM_CSV = os.path.normpath(os.path.join(_HERE, "..", "3D Recon Geometry",
                                          "beam_deflection_profile_0.1.csv"))
OUT_PATH = os.path.join(_HERE, "rest_positions.npy")

# Grid dimensions — must match PointCloudNet and generate_pointclouds.py
GRID_ROWS = 32
GRID_COLS = 64   # 32 × 64 = 2048 points

# Geometry constants (must match contact_sim.py)
R0_MM     = 60.0
THETA_MAX = 45.0
M         = 200
WIDTH_MIN = 15.0
WIDTH_MAX = 30.0


def load_rest_surface(csv_path):
    df = pd.read_csv(csv_path)
    d  = df["beam_deflection_mm"].to_numpy(dtype=np.float64)
    z  = df["beam_height_mm"].to_numpy(dtype=np.float64)
    r  = R0_MM + d
    N  = len(z)

    z_norm        = (z - z.min()) / (z.max() - z.min())
    width_profile = WIDTH_MIN + (WIDTH_MAX - WIDTH_MIN) * np.sin(np.pi * z_norm)

    theta_A = 0.0
    theta_B = np.deg2rad(THETA_MAX)

    x_A = r * np.cos(theta_A);  y_A = r * np.sin(theta_A)
    x_B = r * np.cos(theta_B);  y_B = r * np.sin(theta_B)
    x_c = 0.5 * (x_A + x_B)
    y_c = 0.5 * (y_A + y_B)
    dx  = x_B - x_A;  dy = y_B - y_A
    dn  = np.hypot(dx, dy)
    dx /= dn;  dy /= dn

    t = np.linspace(0.0, 1.0, M)
    X = np.zeros((N, M));  Y = np.zeros((N, M));  Z = np.zeros((N, M))
    for i in range(N):
        w      = (t - 0.5) * width_profile[i]
        X[i,:] = x_c[i] + w * dx[i]
        Y[i,:] = y_c[i] + w * dy[i]
        Z[i,:] = z[i]

    return X, Y, Z, z, width_profile


def main():
    print(f"Loading geometry from {GEOM_CSV} ...")
    X, Y, Z, z_arr, width_profile = load_rest_surface(GEOM_CSV)
    N, M_cols = X.shape
    print(f"  Full surface: {N} × {M_cols}  "
          f"Z: {z_arr.min():.1f}–{z_arr.max():.1f} mm")

    row_idx = np.round(np.linspace(0, N - 1,      GRID_ROWS)).astype(int)
    col_idx = np.round(np.linspace(0, M_cols - 1, GRID_COLS)).astype(int)

    # Build (GRID_ROWS, GRID_COLS, 3) then flatten to (2048, 3)
    rest = np.zeros((GRID_ROWS, GRID_COLS, 3), dtype=np.float32)
    for i, ri in enumerate(row_idx):
        for j, cj in enumerate(col_idx):
            rest[i, j, 0] = X[ri, cj]
            rest[i, j, 1] = Y[ri, cj]
            rest[i, j, 2] = Z[ri, cj]

    rest_flat = rest.reshape(-1, 3)   # (2048, 3)
    np.save(OUT_PATH, rest_flat)

    print(f"  Grid: {GRID_ROWS} rows × {GRID_COLS} cols = {GRID_ROWS*GRID_COLS} points")
    print(f"  X range: {rest_flat[:,0].min():.1f}–{rest_flat[:,0].max():.1f} mm")
    print(f"  Y range: {rest_flat[:,1].min():.1f}–{rest_flat[:,1].max():.1f} mm")
    print(f"  Z range: {rest_flat[:,2].min():.1f}–{rest_flat[:,2].max():.1f} mm")
    print(f"  Saved → {OUT_PATH}")

    # Also expose as importable constants
    print(f"\nConstants for model.py / generate_pointclouds.py:")
    print(f"  GRID_ROWS = {GRID_ROWS}")
    print(f"  GRID_COLS = {GRID_COLS}")
    print(f"  row_idx   = np.round(np.linspace(0, {N-1}, {GRID_ROWS})).astype(int)")
    print(f"  col_idx   = np.round(np.linspace(0, {M_cols-1}, {GRID_COLS})).astype(int)")


if __name__ == "__main__":
    main()
