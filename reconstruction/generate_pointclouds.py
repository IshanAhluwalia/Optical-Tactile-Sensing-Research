"""
Generate deformed-skin point clouds for every frame in dense_contact/dataset.csv.

Physical model
--------------
The indentor always approaches VERTICALLY — i.e. along the surface normal at
the skin's centre.  At off-centre contact positions the indentor therefore hits
the curved skin at an angle.  All skin points are pushed in this single fixed
direction, not along each point's local normal.

Output
------
One .npy file per CSV row, saved alongside the camera images:
    dense_contact/images/x138_y0/frame_00001.png
 →  dense_contact/pointclouds/x138_y0/frame_00001.npy

Each file is a (PC_N_PTS, 3) = (2048, 3) float32 array of [X, Y, Z] in mm,
sampled at the SAME fixed 32×64 structured grid as rest_positions.npy.
Point k corresponds to grid cell (k // PC_GRID_COLS, k % PC_GRID_COLS) —
this ordered correspondence is required for MSE training in PointCloudNet.

Usage
-----
    python generate_pointclouds.py
    python generate_pointclouds.py --force    # overwrite existing files
"""

import argparse
import os
import numpy as np
import pandas as pd
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw): return x

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE    = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(_HERE, "..", "dense_contact", "dataset.csv")
CSV_PATH = os.path.normpath(CSV_PATH)

# ── Structured grid (must match build_rest_positions.py and PointCloudNet) ─────
PC_GRID_ROWS = 32
PC_GRID_COLS = 64
PC_N_PTS     = PC_GRID_ROWS * PC_GRID_COLS   # 2048

# ── Geometry constants (must match contact_sim.py) ─────────────────────────────
R0_MM     = 60.0
THETA_MAX = 45.0
M         = 200
WIDTH_MIN = 15.0
WIDTH_MAX = 30.0
SPREAD_U  = 5.0
SPREAD_V  = 5.0
EDGE_COMP = 0.6


# ── Geometry helpers ───────────────────────────────────────────────────────────

def load_geometry(csv_path: str):
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

    dx = x_B - x_A;  dy = y_B - y_A
    dn = np.hypot(dx, dy)
    dx /= dn;  dy /= dn

    t = np.linspace(0.0, 1.0, M)
    X = np.zeros((N, M));  Y = np.zeros((N, M));  Z = np.zeros((N, M))
    for i in range(N):
        w      = (t - 0.5) * width_profile[i]
        X[i,:] = x_c[i] + w * dx[i]
        Y[i,:] = y_c[i] + w * dy[i]
        Z[i,:] = z[i]

    return X, Y, Z, z, width_profile


def compute_normals(X, Y, Z):
    dX_di, dX_dj = np.gradient(X)
    dY_di, dY_dj = np.gradient(Y)
    dZ_di, dZ_dj = np.gradient(Z)
    Ti = np.stack([dX_di, dY_di, dZ_di], axis=-1)
    Tj = np.stack([dX_dj, dY_dj, dZ_dj], axis=-1)
    N  = np.cross(Ti, Tj)
    N /= np.linalg.norm(N, axis=-1, keepdims=True) + 1e-9
    return N[..., 0], N[..., 1], N[..., 2]


def centre_approach_vector(NX, NY, NZ):
    """
    Return the surface normal at the skin centre as the fixed approach direction.
    The MakerBot actuator is aligned with this direction — vertical at the centre,
    which means off-centre contacts hit the skin at an angle.
    """
    i_c = NX.shape[0] // 2   # row closest to z = 108 mm
    j_c = NX.shape[1] // 2   # column at width centre
    n0  = np.array([NX[i_c, j_c], NY[i_c, j_c], NZ[i_c, j_c]], dtype=np.float64)
    n0 /= np.linalg.norm(n0)
    return n0


def loc_y_to_t0(loc_y_mm: float, s0_mm: float, z_arr, width_profile) -> float:
    idx         = int(np.argmin(np.abs(z_arr - s0_mm)))
    local_width = width_profile[idx]
    return float(np.clip(0.5 + (loc_y_mm - 8.0) / local_width, 0.0, 1.0))


def apply_deformation(X, Y, Z, n0, width_profile, depth, s0, t0):
    """
    Deform the skin using a Gaussian contact model with a FIXED approach
    direction n0 (the vertical = centre-skin normal).

    All skin points are displaced by D along -n0, regardless of local surface
    orientation.  This correctly models off-centre contacts approaching at an
    angle to the curved skin.
    """
    N_rows, M_cols = X.shape
    t   = np.linspace(0.0, 1.0, M_cols)
    T   = np.tile(t[None, :], (N_rows, 1))
    W   = width_profile[:, None]

    S_mm = Z - s0
    V_mm = (T - t0) * W

    dist_edge  = np.minimum(T, 1.0 - T) / 0.5
    compliance = EDGE_COMP + (1.0 - EDGE_COMP) * np.sin(0.5 * np.pi * dist_edge) ** 2

    D = depth * np.exp(
        -(S_mm**2 / (2.0 * SPREAD_U**2) + V_mm**2 / (2.0 * SPREAD_V**2))
    ) * compliance   # shape (N_rows, M_cols)

    # Push all points along the fixed approach direction (broadcast scalar n0)
    X_def = (X - D * n0[0]).astype(np.float32)
    Y_def = (Y - D * n0[1]).astype(np.float32)
    Z_def = (Z - D * n0[2]).astype(np.float32)

    return X_def, Y_def, Z_def


def image_path_to_cloud_path(image_path: str) -> str:
    """
    dense_contact/images/x138_y0/frame_00000.png
    → dense_contact/pointclouds/x138_y0/frame_00000.npy
    """
    p = image_path.replace(os.sep + "images" + os.sep,
                           os.sep + "pointclouds" + os.sep)
    p = p.replace("/images/", "/pointclouds/")
    return os.path.splitext(p)[0] + ".npy"


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",   default=CSV_PATH,
                    help="Path to dense_contact/dataset.csv")
    ap.add_argument("--geom",  default=os.path.join(_HERE, "beam_deflection_profile_0.1.csv"),
                    help="Beam deflection CSV")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing .npy files (default: skip existing)")
    args = ap.parse_args()

    # ── Load geometry ─────────────────────────────────────────────────────────
    print(f"Loading skin geometry from {args.geom} ...")
    X, Y, Z, z_arr, width_profile = load_geometry(args.geom)
    NX, NY, NZ = compute_normals(X, Y, Z)

    n0 = centre_approach_vector(NX, NY, NZ)
    N_rows_full, M_cols_full = X.shape
    print(f"  Surface grid   : {N_rows_full} × {M_cols_full}")
    print(f"  Z range        : {z_arr.min():.1f} – {z_arr.max():.1f} mm")
    print(f"  Approach vector: [{n0[0]:.4f}, {n0[1]:.4f}, {n0[2]:.4f}]  "
          f"(fixed — normal at skin centre)")

    # ── Structured grid indices (same as build_rest_positions.py) ─────────────
    row_idx = np.round(np.linspace(0, N_rows_full - 1, PC_GRID_ROWS)).astype(int)
    col_idx = np.round(np.linspace(0, M_cols_full - 1, PC_GRID_COLS)).astype(int)
    mesh_r, mesh_c = np.ix_(row_idx, col_idx)   # for fancy indexing
    print(f"  Output grid    : {PC_GRID_ROWS} × {PC_GRID_COLS} = {PC_N_PTS} pts (structured)")

    # Pre-extract rest surface at the grid (for depth <= 0 frames)
    rest_pts = np.stack([
        X[mesh_r, mesh_c].ravel(),
        Y[mesh_r, mesh_c].ravel(),
        Z[mesh_r, mesh_c].ravel(),
    ], axis=1).astype(np.float32)   # (2048, 3) — fixed, same as rest_positions.npy

    # ── Load dataset CSV ──────────────────────────────────────────────────────
    print(f"\nLoading dataset from {args.csv} ...")
    df = pd.read_csv(args.csv)
    print(f"  {len(df):,} frames  |  {df['session'].nunique()} sessions")

    generated = 0
    skipped   = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating clouds"):
        out_path = image_path_to_cloud_path(row["image_path"])

        if not args.force and os.path.exists(out_path):
            skipped += 1
            continue

        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        depth = float(row["displacement_mm"])
        loc_x = float(row["loc_x"])
        loc_y = float(row["loc_y"])

        if depth <= 0.0:
            pts = rest_pts   # identical for all no-contact frames
        else:
            t0 = loc_y_to_t0(loc_y, loc_x, z_arr, width_profile)
            X_d, Y_d, Z_d = apply_deformation(
                X, Y, Z, n0, width_profile, depth, s0=loc_x, t0=t0
            )
            # Extract structured grid — same ordering as rest_positions.npy
            pts = np.stack([
                X_d[mesh_r, mesh_c].ravel(),
                Y_d[mesh_r, mesh_c].ravel(),
                Z_d[mesh_r, mesh_c].ravel(),
            ], axis=1).astype(np.float32)   # (2048, 3)

        np.save(out_path, pts)
        generated += 1

    print(f"\nDone.")
    print(f"  Generated : {generated:,}")
    print(f"  Skipped   : {skipped:,}  (already existed)")
    print(f"  Each cloud: ({PC_N_PTS}, 3) float32  [X, Y, Z mm]  — structured {PC_GRID_ROWS}×{PC_GRID_COLS} grid")
    print(f"  Saved under dense_contact/pointclouds/")


if __name__ == "__main__":
    main()
