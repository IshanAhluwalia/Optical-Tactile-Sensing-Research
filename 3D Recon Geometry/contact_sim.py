"""
Skin geometry contact simulator.

Builds the skin surface from a beam-deflection CSV profile and lets you
interactively explore Gaussian indentation at any location and depth.

Controls
--------
    Arrow Up / Down     – move contact along sensor length
    Arrow Left / Right  – move contact across sensor width
    + / =               – increase indentation depth
    -                   – decrease indentation depth
    Space               – toggle deformed surface on/off
    C                   – toggle camera feed window
    Q / Escape          – quit

Usage
-----
    python contact_sim.py
    python contact_sim.py --csv beam_deflection_profile_0.1.csv --depth 5.0
    python contact_sim.py --cam 0          # show camera feed from device 0
"""

import argparse
import queue
import threading
import numpy as np
import pandas as pd
import cv2
from vispy import app, scene

# ── Geometry settings ─────────────────────────────────────────────────────────
R0_MM      = 60.0   # nominal radial distance (mm)
THETA_MAX  = 45.0   # half-arc angle (degrees)
M          = 200    # points across width
WIDTH_MIN  = 15.0   # min skin width (mm)
WIDTH_MAX  = 30.0   # max skin width (mm)

# ── Contact model settings ────────────────────────────────────────────────────
SPREAD_U   = 5.0    # Gaussian σ along sensor length (mm)
SPREAD_V   = 5.0    # Gaussian σ across sensor width (mm)
EDGE_COMP  = 0.6    # edge compliance (0 = rigid edge, 1 = uniform)
DEPTH_INIT = 3.0    # initial indentation depth (mm)
DEPTH_STEP = 0.25   # depth change per keypress (mm)
DEPTH_MAX  = 20.0
DEPTH_MIN  = 0.1
S0_STEP    = 2.0    # contact shift along length per keypress (mm)
T0_STEP    = 0.05   # contact shift across width per keypress (fraction)

MAX_PTS    = 8000   # subsample cap for rendering

CAM_WIN    = "Camera Feed  [C=hide]"


# ── Camera feed thread ────────────────────────────────────────────────────────

class CamThread(threading.Thread):
    """Captures frames from a webcam and puts the latest one in a queue."""

    def __init__(self, cam_index):
        super().__init__(daemon=True)
        self.cam_index = cam_index
        self.q         = queue.Queue(maxsize=1)
        self._stop     = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        cap = cv2.VideoCapture(self.cam_index, cv2.CAP_AVFOUNDATION)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            print(f"[camera] could not open device {self.cam_index}")
            return
        print(f"[camera] device {self.cam_index} opened")
        while not self._stop.is_set():
            ret, frame = cap.read()
            if not ret:
                continue
            try:
                self.q.put_nowait(frame)
            except queue.Full:
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.q.put_nowait(frame)
                except queue.Full:
                    pass
        cap.release()


# ── Geometry ──────────────────────────────────────────────────────────────────

def load_geometry(csv_path):
    df = pd.read_csv(csv_path)
    d  = df["beam_deflection_mm"].to_numpy()
    z  = df["beam_height_mm"].to_numpy()
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

    t = np.linspace(0, 1, M)
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


# ── Deformation ───────────────────────────────────────────────────────────────

def apply_deformation(X, Y, Z, NX, NY, NZ, width_profile, depth, s0, t0):
    N, M  = X.shape
    t     = np.linspace(0, 1, M)
    T     = np.tile(t[None, :], (N, 1))
    W     = width_profile[:, None]

    S_mm = Z - s0                  # deviation along length from contact centre
    V_mm = (T - t0) * W            # deviation across width from contact centre

    dist_edge  = np.minimum(T, 1.0 - T) / 0.5
    compliance = EDGE_COMP + (1.0 - EDGE_COMP) * np.sin(0.5 * np.pi * dist_edge) ** 2

    D = depth * np.exp(
        -(S_mm**2 / (2.0 * SPREAD_U**2) + V_mm**2 / (2.0 * SPREAD_V**2))
    ) * compliance

    return X - D * NX, Y - D * NY, Z - D * NZ, D


# ── Render helpers ────────────────────────────────────────────────────────────

def _subsample(pts, colors):
    if len(pts) <= MAX_PTS:
        return pts, colors
    idx = np.random.choice(len(pts), MAX_PTS, replace=False)
    return pts[idx], colors[idx]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",   default="beam_deflection_profile_0.1.csv",
                    help="Path to beam deflection CSV")
    ap.add_argument("--depth", type=float, default=DEPTH_INIT,
                    help="Initial indentation depth (mm)")
    ap.add_argument("--cam",   type=int,   default=-1,
                    help="Camera device index (omit to disable feed)")
    args = ap.parse_args()

    print(f"Loading geometry from {args.csv} ...")
    X, Y, Z, z_arr, width_profile = load_geometry(args.csv)
    NX, NY, NZ = compute_normals(X, Y, Z)
    print(f"  Grid: {X.shape[0]} × {X.shape[1]}  "
          f"Z range: [{z_arr.min():.1f}, {z_arr.max():.1f}] mm")

    state = {
        "depth":      args.depth,
        "s0":         float(np.mean(z_arr)),
        "t0":         0.5,
        "show":       True,
        "show_cam":   args.cam >= 0,
    }

    # ── Camera thread (optional) ──────────────────────────────────────────────
    cam_thread = None
    if args.cam >= 0:
        cam_thread = CamThread(args.cam)
        cam_thread.start()

    # ── vispy canvas ─────────────────────────────────────────────────────────
    canvas = scene.SceneCanvas(
        title="Skin Contact Sim  [arrows=move  +/-=depth  space=toggle  Q=quit]",
        bgcolor="#0d0d1a",
        size=(1100, 750),
        keys="interactive",
    )
    view   = canvas.central_widget.add_view()
    view.camera = scene.cameras.TurntableCamera(
        fov=45, distance=200, elevation=5, azimuth=30,
        center=(float(np.mean(X)), float(np.mean(Y)), float(np.mean(Z)))
    )

    scatter_rest   = scene.visuals.Markers()
    scatter_deform = scene.visuals.Markers()
    view.add(scatter_rest)
    view.add(scatter_deform)
    scene.visuals.XYZAxis(parent=view.scene)

    status = scene.visuals.Text(
        "", color="white", font_size=10,
        anchor_x="left", anchor_y="bottom",
        parent=canvas.scene,
    )
    status.transform = scene.transforms.STTransform(translate=(10, 10))

    # Pre-build rest surface (static)
    rest_pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(np.float32)
    rest_col = np.full((len(rest_pts), 4), [0.3, 0.45, 0.6, 0.3], dtype=np.float32)
    rest_pts, rest_col = _subsample(rest_pts, rest_col)

    def redraw():
        depth = state["depth"]
        s0    = state["s0"]
        t0    = state["t0"]

        X_def, Y_def, Z_def, D = apply_deformation(
            X, Y, Z, NX, NY, NZ, width_profile, depth, s0, t0
        )

        def_pts = np.stack([X_def.ravel(), Y_def.ravel(), Z_def.ravel()], axis=1).astype(np.float32)
        D_flat  = D.ravel().astype(np.float32)

        # blue (no contact) → yellow (peak contact)
        norm    = np.clip(D_flat / max(depth, 0.1), 0.0, 1.0)
        cc      = np.zeros((len(norm), 4), np.float32)
        cc[:,0] = norm
        cc[:,1] = 0.3 + 0.5 * norm
        cc[:,2] = 1.0 - 0.9 * norm
        cc[:,3] = 1.0

        def_pts_s, cc_s = _subsample(def_pts, cc)

        scatter_rest.set_data(rest_pts, face_color=rest_col, size=3, edge_width=0)

        if state["show"]:
            scatter_deform.set_data(def_pts_s, face_color=cc_s, size=4, edge_width=0)
            scatter_deform.visible = True
        else:
            scatter_deform.visible = False

        t0_mm = (t0 - 0.5) * float(np.mean(width_profile))
        status.text = (
            f"depth={depth:.2f} mm   "
            f"s0={s0:.1f} mm (length)   "
            f"t0={t0:.2f} ({t0_mm:+.1f} mm across width)   "
            f"| ↑↓ length  ←→ width  +/- depth  space toggle  Q quit"
        )
        canvas.update()

    @canvas.events.key_press.connect
    def on_key(ev):
        k = ev.key
        if k in ("Q", "Escape"):
            app.quit()
        elif k in ("+", "="):
            state["depth"] = min(state["depth"] + DEPTH_STEP, DEPTH_MAX)
        elif k == "-":
            state["depth"] = max(state["depth"] - DEPTH_STEP, DEPTH_MIN)
        elif k == "Up":
            state["s0"] = min(state["s0"] + S0_STEP, float(z_arr.max()))
        elif k == "Down":
            state["s0"] = max(state["s0"] - S0_STEP, float(z_arr.min()))
        elif k == "Right":
            state["t0"] = min(state["t0"] + T0_STEP, 1.0)
        elif k == "Left":
            state["t0"] = max(state["t0"] - T0_STEP, 0.0)
        elif k == " ":
            state["show"] = not state["show"]
        redraw()

    redraw()
    canvas.show()

    print(f"\nInitial state: depth={state['depth']} mm  "
          f"s0={state['s0']:.1f} mm  t0={state['t0']}")
    print("↑↓ = along length  |  ←→ = across width  |  "
          "+/- = depth  |  space = toggle  |  Q = quit\n")

    app.run()


if __name__ == "__main__":
    main()
