"""
Interactive 3D point cloud viewer for elastomer skin reconstruction.

Camera is captured in a background thread; vispy renders the live 3D cloud.

Mouse controls (vispy TurntableCamera)
---------------------------------------
    Left-drag    – rotate
    Scroll       – zoom
    Right-drag   – pan

Keyboard
--------
    R        – capture reference (unloaded skin)
    C        – toggle contact-simulation overlay (requires reference)
    + / =    – increase simulated indentation depth
    -        – decrease simulated indentation depth
    D        – toggle disparity debug window
    Q        – quit

Usage
-----
    python reconstruct.py --cam0 2 --cam1 1 --rotate1
"""

import threading
import queue
import warnings
import numpy as np
import cv2
import argparse
import json
import os
import sys
import time
from collections import deque

try:
    from scipy.interpolate import griddata
    from scipy.ndimage import gaussian_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ── Config ────────────────────────────────────────────────────────────────────

Z_MIN_MM      = 50.0
Z_MAX_MM      = 200.0
SGBM_MIN_DISP = 60
SGBM_NUM_DISP = 208    # must be divisible by 16
SGBM_BLOCK    = 7
BUFFER_FRAMES = 6
DEFORM_RANGE  = 3.0    # ±mm
MAX_CLOUD_PTS = 5000   # subsample dense cloud for rendering

# ── Contact deformation model ─────────────────────────────────────────────────
CONTACT_SPREAD_U   = 5.0   # Gaussian σ along sensor length (mm)
CONTACT_SPREAD_V   = 5.0   # Gaussian σ across sensor width (mm)
CONTACT_EDGE_COMP  = 0.6   # Compliance at edges relative to centre
CONTACT_T0         = 0.5   # Contact centre across width (0 = one edge, 1 = other)
CONTACT_DEPTH_MM   = 3.0   # Default simulated indentation depth (mm)
CONTACT_DEPTH_STEP = 0.5   # Depth increment per keypress (mm)


# ── Stereo helpers ────────────────────────────────────────────────────────────

def load_params(path):
    d = np.load(path)
    return {k: d[k] for k in d.files}


def load_roi(path="params/skin_roi.json"):
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        return (d["x"], d["y"], d["w"], d["h"])
    return None


def make_sgbm():
    bs = SGBM_BLOCK
    return cv2.StereoSGBM_create(
        minDisparity=SGBM_MIN_DISP,
        numDisparities=SGBM_NUM_DISP,
        blockSize=bs,
        P1=8  * 3 * bs * bs,
        P2=32 * 3 * bs * bs,
        disp12MaxDiff=2,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


# ── Contact deformation helpers ───────────────────────────────────────────────

def compute_normals(X, Y, Z):
    """Per-pixel surface normals via finite-difference cross product on a 2-D grid."""
    dX_di, dX_dj = np.gradient(X)
    dY_di, dY_dj = np.gradient(Y)
    dZ_di, dZ_dj = np.gradient(Z)
    Ti = np.stack([dX_di, dY_di, dZ_di], axis=-1)
    Tj = np.stack([dX_dj, dY_dj, dZ_dj], axis=-1)
    N  = np.cross(Ti, Tj)
    N /= (np.linalg.norm(N, axis=-1, keepdims=True) + 1e-9)
    return N[..., 0], N[..., 1], N[..., 2]


def build_contact_cloud(X_map, Y_map, Z_map, depth_mm,
                        spread_u=CONTACT_SPREAD_U,
                        spread_v=CONTACT_SPREAD_V,
                        edge_comp=CONTACT_EDGE_COMP,
                        t0=CONTACT_T0):
    """
    Apply a Gaussian contact indentation to a reconstructed skin geometry.

    Displaces surface points along their inward normals; magnitude follows a
    2-D Gaussian centred at the contact location in intrinsic (length × width)
    coordinates, attenuated by an edge-compliance envelope.

    Parameters
    ----------
    X_map, Y_map, Z_map : (rh, rw) float arrays
        Reference geometry grids from the stereo reconstruction (may contain NaN).
    depth_mm : float
        Peak indentation depth in mm.
    spread_u, spread_v : float
        Gaussian σ along the sensor length and width axes (mm).
    edge_comp : float
        Minimum compliance at the edges (0 = rigid edge, 1 = uniform).
    t0 : float
        Contact centre across the width in [0, 1].

    Returns
    -------
    pts    : (N, 3) float32   deformed point positions
    depths : (N,)  float32   indentation depth per point (for colour mapping)
    """
    rh, rw = X_map.shape

    # Width-normalised column parameter t ∈ [0, 1]
    t_vec  = np.linspace(0, 1, rw)
    T_grid = np.tile(t_vec[None, :], (rh, 1))  # (rh, rw)

    # Physical width per row (mm), used to convert t → mm
    x_lo       = np.nanmin(X_map, axis=1, keepdims=True)
    x_hi       = np.nanmax(X_map, axis=1, keepdims=True)
    width_grid = np.where(np.isfinite(x_hi - x_lo), x_hi - x_lo, 1.0)

    # Intrinsic coordinates (mm)
    s0    = float(np.nanmean(Y_map))
    S_mm  = Y_map - s0                       # deviation along sensor length
    V_mm  = (T_grid - t0) * width_grid       # lateral deviation from contact centre

    # Edge-compliance envelope — tapers deformation at skin edges
    dist_edge  = np.minimum(T_grid, 1.0 - T_grid) / 0.5
    compliance = edge_comp + (1.0 - edge_comp) * np.sin(0.5 * np.pi * dist_edge) ** 2

    # Gaussian indentation depth field (positive = inward)
    D = depth_mm * np.exp(
        -(S_mm ** 2 / (2.0 * spread_u ** 2) + V_mm ** 2 / (2.0 * spread_v ** 2))
    ) * compliance

    # Surface normals (NaN holes filled before gradient)
    NX, NY, NZ = compute_normals(
        np.nan_to_num(X_map), np.nan_to_num(Y_map), np.nan_to_num(Z_map)
    )

    # Displace along inward normal
    X_def = X_map - D * NX
    Y_def = Y_map - D * NY
    Z_def = Z_map - D * NZ

    # Flatten, keep only pixels that had valid geometry
    valid  = np.isfinite(X_map) & np.isfinite(Y_map) & np.isfinite(Z_map)
    pts    = np.stack([X_def[valid], Y_def[valid], Z_def[valid]], axis=1).astype(np.float32)
    depths = D[valid].astype(np.float32)
    return pts, depths


# ── Capture thread ────────────────────────────────────────────────────────────

class CaptureThread(threading.Thread):
    def __init__(self, args, params, roi):
        super().__init__(daemon=True)
        self.args    = args
        self.params  = params
        self.roi     = roi
        self.out_q   = queue.Queue(maxsize=1)   # latest point cloud
        self.debug_q = queue.Queue(maxsize=1)   # latest debug image
        self.cmd_q   = queue.Queue()             # commands (ref, quit)
        self._stop   = threading.Event()

        Q = params["Q"].copy()
        Q[3,2] = abs(Q[3,2])
        self.f        = float(Q[2,3])
        self.cx       = float(-Q[0,3])
        self.cy       = float(-Q[1,3])
        self.baseline = abs(1.0 / Q[3,2])

        self.sgbm     = make_sgbm()
        self.disp_buf = deque(maxlen=BUFFER_FRAMES)
        self.ref_Z_map = None
        self.has_ref   = False
        self._ref_maps     = None          # (X, Y, Z) maps kept for contact model
        self.contact_depth = CONTACT_DEPTH_MM
        self.show_contact  = False

    def stop(self):
        self._stop.set()

    def run(self):
        p = self.params
        map0x, map0y = p["map0x"], p["map0y"]
        map1x, map1y = p["map1x"], p["map1y"]
        args = self.args

        cap0 = cv2.VideoCapture(args.cam0, cv2.CAP_AVFOUNDATION)
        cap1 = cv2.VideoCapture(args.cam1, cv2.CAP_AVFOUNDATION)
        cap0.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap1.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        print("Warming up cameras (5s)...", end=" ", flush=True)
        time.sleep(5)
        for _ in range(10): cap0.read(); cap1.read()
        print("ready")

        while not self._stop.is_set():
            # Process commands
            while not self.cmd_q.empty():
                cmd = self.cmd_q.get_nowait()
                if cmd == "ref":
                    self._capture_ref = True
                elif cmd == "contact_toggle":
                    self.show_contact = not self.show_contact
                    print(f"Contact simulation {'ON' if self.show_contact else 'OFF'}  "
                          f"depth={self.contact_depth:.1f}mm")
                elif cmd == "depth_up":
                    self.contact_depth = min(self.contact_depth + CONTACT_DEPTH_STEP, 20.0)
                    print(f"Contact depth → {self.contact_depth:.1f} mm")
                elif cmd == "depth_down":
                    self.contact_depth = max(self.contact_depth - CONTACT_DEPTH_STEP, 0.1)
                    print(f"Contact depth → {self.contact_depth:.1f} mm")

            ret0, img0 = cap0.read()
            ret1, img1 = cap1.read()
            if not (ret0 and ret1):
                continue

            if args.rotate0: img0 = cv2.rotate(img0, cv2.ROTATE_180)
            if args.rotate1: img1 = cv2.rotate(img1, cv2.ROTATE_180)

            rect0 = cv2.remap(img0, map0x, map0y, cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT)
            rect1 = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT)

            gray0 = cv2.cvtColor(rect0, cv2.COLOR_BGR2GRAY)
            gray1 = cv2.cvtColor(rect1, cv2.COLOR_BGR2GRAY)

            # ── SGBM dense stereo on ROI ───────────────────────────────────
            if self.roi:
                rx, ry, rw, rh = self.roi
            else:
                rx, ry, rw, rh = 0, 0, gray0.shape[1], gray0.shape[0]

            gl = gray0[ry:ry+rh, rx:rx+rw]
            gr = gray1[ry:ry+rh, rx:rx+rw]

            raw_disp = self.sgbm.compute(gl, gr).astype(np.float32) / 16.0
            raw_disp[raw_disp < SGBM_MIN_DISP] = np.nan

            self.disp_buf.append(raw_disp)

            # Temporal median over disparity buffer
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                disp_med = np.nanmedian(
                    np.stack(list(self.disp_buf), axis=0), axis=0)

            valid_mask = np.isfinite(disp_med)
            if not valid_mask.any():
                continue

            # ── Build debug overlay: colorised disparity on left frame ─────
            dbg = rect0.copy()
            disp_vis = np.zeros((rh, rw), dtype=np.uint8)
            vd = disp_med[valid_mask]
            if len(vd):
                d_lo, d_hi = float(np.nanmin(vd)), float(np.nanmax(vd))
                norm = np.clip((disp_med - d_lo) / max(d_hi - d_lo, 1.0), 0, 1)
                disp_vis = (norm * 255).astype(np.uint8)
            disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_TURBO)
            dbg[ry:ry+rh, rx:rx+rw] = cv2.addWeighted(
                dbg[ry:ry+rh, rx:rx+rw], 0.4, disp_color, 0.6, 0)
            cv2.rectangle(dbg, (rx, ry), (rx+rw, ry+rh), (0, 220, 255), 1)
            n_valid = int(valid_mask.sum())
            cv2.putText(dbg, f"SGBM  valid={n_valid}",
                        (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            try:
                self.debug_q.put_nowait(dbg)
            except queue.Full:
                try: self.debug_q.get_nowait()
                except queue.Empty: pass
                try: self.debug_q.put_nowait(dbg)
                except queue.Full: pass
            # ──────────────────────────────────────────────────────────────

            # ── Disparity → 3D ────────────────────────────────────────────
            ys_px, xs_px = np.where(valid_mask)
            us = (xs_px + rx).astype(np.float32)
            vs = (ys_px + ry).astype(np.float32)
            ds = disp_med[valid_mask]

            Z = self.f * self.baseline / ds
            X = (us - self.cx) * Z / self.f
            Y = (vs - self.cy) * Z / self.f
            depth_ok = (Z > Z_MIN_MM) & (Z < Z_MAX_MM)
            if not depth_ok.any():
                continue

            # Build a Z map per ROI pixel for deformation tracking
            Z_map = np.full((rh, rw), np.nan, dtype=np.float32)
            X_map = np.full((rh, rw), np.nan, dtype=np.float32)
            Y_map = np.full((rh, rw), np.nan, dtype=np.float32)
            ys_ok = ys_px[depth_ok]
            xs_ok = xs_px[depth_ok]
            Z_map[ys_ok, xs_ok] = Z[depth_ok]
            X_map[ys_ok, xs_ok] = X[depth_ok]
            Y_map[ys_ok, xs_ok] = Y[depth_ok]

            out_mask = np.isfinite(Z_map)
            oys, oxs = np.where(out_mask)

            # Subsample for rendering performance
            if len(oys) > MAX_CLOUD_PTS:
                idx = np.random.choice(len(oys), MAX_CLOUD_PTS, replace=False)
                oys, oxs = oys[idx], oxs[idx]

            pts_smooth = np.stack(
                [X_map[oys, oxs], Y_map[oys, oxs], Z_map[oys, oxs]], axis=1
            ).astype(np.float32)

            # Deformation vs reference
            deform = None
            if self.has_ref and self.ref_Z_map is not None:
                if self.ref_Z_map.shape == Z_map.shape:
                    dZ = Z_map[oys, oxs] - self.ref_Z_map[oys, oxs]
                    dZ = np.nan_to_num(dZ)
                    deform = dZ

            # Handle reference capture command
            if hasattr(self, '_capture_ref') and self._capture_ref:
                self.ref_Z_map  = Z_map.copy()
                self._ref_maps  = (X_map.copy(), Y_map.copy(), Z_map.copy())
                self.has_ref    = True
                self._capture_ref = False
                med_z = float(np.nanmedian(Z_map[out_mask]))
                print(f"Reference captured — {out_mask.sum()} pts, med depth {med_z:.1f}mm")

            # Contact simulation on the reference geometry
            contact_pts    = None
            contact_depths = None
            if self.show_contact and self._ref_maps is not None:
                X_ref, Y_ref, Z_ref = self._ref_maps
                contact_pts, contact_depths = build_contact_cloud(
                    X_ref, Y_ref, Z_ref, self.contact_depth
                )

            payload = {
                "pts":           pts_smooth,
                "deform":        deform,
                "n_raw":         n_valid,
                "buf":           len(self.disp_buf),
                "contact_pts":   contact_pts,
                "contact_depths": contact_depths,
                "contact_depth": self.contact_depth,
                "show_contact":  self.show_contact,
            }
            try:
                self.out_q.put_nowait(payload)
            except queue.Full:
                try:
                    self.out_q.get_nowait()
                    self.out_q.put_nowait(payload)
                except Exception:
                    pass

        cap0.release()
        cap1.release()


# ── Main (vispy viewer) ───────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam0",    type=int,   default=2)
    ap.add_argument("--cam1",    type=int,   default=1)
    ap.add_argument("--rotate0", action="store_true")
    ap.add_argument("--rotate1", action="store_true")
    ap.add_argument("--params",  default="params/stereo_params.npz")
    args = ap.parse_args()

    if not os.path.exists(args.params):
        sys.exit(f"Params not found: {args.params}")

    params = load_params(args.params)
    roi    = load_roi()
    if roi:
        print(f"ROI: {roi}")

    # ── Start capture thread ──────────────────────────────────────────────
    cap_thread = CaptureThread(args, params, roi)
    cap_thread._capture_ref = False
    cap_thread.start()

    # ── vispy setup ───────────────────────────────────────────────────────
    from vispy import app, scene

    canvas = scene.SceneCanvas(
        title="Skin Point Cloud  [R=reference  Q=quit  scroll=zoom  drag=rotate]",
        bgcolor="#0d0d1a",
        size=(1000, 700),
        keys="interactive",
    )
    view = canvas.central_widget.add_view()

    # TurntableCamera: left-drag=rotate, scroll=zoom, right-drag=pan
    view.camera = scene.cameras.TurntableCamera(
        fov=45, distance=200, elevation=25, azimuth=30
    )

    scatter = scene.visuals.Markers()
    view.add(scatter)

    # Overlay for contact-simulation geometry
    scatter_contact = scene.visuals.Markers()
    scatter_contact.visible = False
    view.add(scatter_contact)

    # Axis widget for orientation reference
    axis = scene.visuals.XYZAxis(parent=view.scene)

    status_text = scene.visuals.Text(
        "", color="white", font_size=10,
        anchor_x="left", anchor_y="bottom",
        parent=canvas.scene
    )
    status_text.transform = scene.transforms.STTransform(translate=(10, 10))

    has_data    = [False]
    show_deform = [False]
    show_debug  = [True]   # D key toggles camera tracking window

    def update_cloud(ev):
        try:
            payload = cap_thread.out_q.get_nowait()
        except queue.Empty:
            return

        pts            = payload["pts"]
        deform         = payload["deform"]
        n_raw          = payload["n_raw"]
        buf            = payload["buf"]
        contact_pts    = payload.get("contact_pts")
        contact_depths = payload.get("contact_depths")
        contact_depth  = payload.get("contact_depth", CONTACT_DEPTH_MM)
        show_contact   = payload.get("show_contact", False)

        if len(pts) == 0:
            return

        has_data[0] = True

        if deform is not None and show_deform[0]:
            # Colour by deformation
            norm  = np.clip(deform / DEFORM_RANGE * 0.5 + 0.5, 0, 1)
            cmap  = np.zeros((len(norm), 4), np.float32)
            cmap[:, 0] = norm          # red channel
            cmap[:, 2] = 1.0 - norm    # blue channel
            cmap[:, 3] = 1.0
            colors = cmap
        else:
            # Clean white-blue cloud
            colors = np.ones((len(pts), 4), np.float32)
            colors[:, 0] = 0.6   # R
            colors[:, 1] = 0.75  # G
            colors[:, 2] = 1.0   # B
            colors[:, 3] = 1.0   # A

        scatter.set_data(pts, face_color=colors, size=4, edge_width=0)

        # Contact simulation overlay
        if show_contact and contact_pts is not None and len(contact_pts) > 0:
            # Subsample for rendering
            if len(contact_pts) > MAX_CLOUD_PTS:
                idx = np.random.choice(len(contact_pts), MAX_CLOUD_PTS, replace=False)
                contact_pts    = contact_pts[idx]
                contact_depths = contact_depths[idx]
            # Colour by indentation depth: deep = warm yellow, shallow = cool blue
            d_norm = np.clip(contact_depths / max(float(contact_depth), 0.1), 0, 1)
            cc = np.zeros((len(d_norm), 4), np.float32)
            cc[:, 0] = d_norm                   # red
            cc[:, 1] = 0.3 + 0.5 * d_norm       # green
            cc[:, 2] = 1.0 - 0.9 * d_norm       # blue
            cc[:, 3] = 1.0
            scatter_contact.set_data(contact_pts, face_color=cc, size=4, edge_width=0)
            scatter_contact.visible = True
        else:
            scatter_contact.visible = False

        # Auto-set camera center on first data
        if buf == 1:
            centre = pts.mean(axis=0)
            view.camera.center = tuple(centre)

        buf_pct = int(buf / BUFFER_FRAMES * 100)
        ref_str = "REF captured" if cap_thread.has_ref else f"buf={buf_pct}% — press R when stable"
        contact_str = (f"  CONTACT {contact_depth:.1f}mm" if show_contact else "")
        status_text.text = (f"Points: {len(pts):,}  valid px: {n_raw}  {ref_str}{contact_str}  "
                            f"| R=ref  C=contact  +/-=depth  D=cam  Q=quit")

        # Debug camera tracking window
        if show_debug[0]:
            try:
                dbg = cap_thread.debug_q.get_nowait()
                cv2.imshow("Camera Tracking  [D=hide]", dbg)
                cv2.waitKey(1)
            except queue.Empty:
                cv2.waitKey(1)
        else:
            cv2.destroyWindow("Camera Tracking  [D=hide]")

    timer = app.Timer(interval=0.08, connect=update_cloud, start=True)

    @canvas.events.key_press.connect
    def on_key(event):
        if event.key == "Q" or event.key == "Escape":
            cap_thread.stop()
            app.quit()
        elif event.key == "R":
            if has_data[0]:
                cap_thread.cmd_q.put("ref")
                show_deform[0] = True
            else:
                print("No data yet — wait for buffer to fill")
        elif event.key == "C":
            if cap_thread.has_ref:
                cap_thread.cmd_q.put("contact_toggle")
            else:
                print("Capture a reference first (R), then press C")
        elif event.key in ("+", "="):
            cap_thread.cmd_q.put("depth_up")
        elif event.key == "-":
            cap_thread.cmd_q.put("depth_down")
        elif event.key == "D":
            show_debug[0] = not show_debug[0]

    canvas.show()
    print("\nvispy window open.")
    print("Left-drag = rotate  |  Scroll = zoom  |  Right-drag = pan")
    print("R = capture reference  |  C = contact simulation  |  +/- = depth  "
          "|  D = disparity view  |  Q = quit\n")

    app.run()
    cap_thread.stop()


if __name__ == "__main__":
    main()
