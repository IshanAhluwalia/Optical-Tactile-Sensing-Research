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
    R  – capture reference (unloaded skin)
    Q  – quit

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

Z_MIN_MM     = 50.0
Z_MAX_MM     = 200.0
DISP_MIN     = 60
DISP_MAX     = 260
PATCH_HALF   = 12      # NCC patch half-size → 25×25 window
NCC_THRESH   = 0.45
MAX_FEATURES = 600
BUFFER_FRAMES = 6
DEFORM_RANGE  = 3.0    # ±mm


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


def fixed_grid_points(roi, n_x=40, n_y=14):
    """Return a fixed regular grid of sample points inside the ROI.
    Same locations every frame → stable surface fitting."""
    if roi:
        x, y, w, h = roi
        margin = PATCH_HALF + 2
        xs = np.linspace(x + margin, x + w - margin, n_x)
        ys = np.linspace(y + margin, y + h - margin, n_y)
    else:
        xs = np.linspace(PATCH_HALF + 2, 637 - PATCH_HALF, n_x)
        ys = np.linspace(PATCH_HALF + 2, 477 - PATCH_HALF, n_y)
    XX, YY = np.meshgrid(xs, ys)
    return np.stack([XX.ravel(), YY.ravel()], axis=1).astype(np.float32)


def match_ncc(gray_l, gray_r, pts_l,
              patch_half=PATCH_HALF, d_min=DISP_MIN, d_max=DISP_MAX,
              thresh=NCC_THRESH):
    H, W = gray_l.shape
    p = patch_half
    good_l, good_r = [], []
    for xl, yl in pts_l:
        xl, yl = int(round(xl)), int(round(yl))
        if yl-p < 0 or yl+p >= H or xl-p < 0 or xl+p >= W:
            continue
        template = gray_l[yl-p:yl+p+1, xl-p:xl+p+1].astype(np.float32)
        xr_max = min(W-p-1, xl-d_min)
        xr_min = max(p,     xl-d_max)
        if xr_max - xr_min < 2:
            continue
        best_score, best_xr = -1.0, -1
        for yr in range(max(p, yl-1), min(H-p-2, yl+1)+1):
            strip = gray_r[yr-p:yr+p+1, xr_min:xr_max+2*p+1].astype(np.float32)
            if strip.shape[1] < template.shape[1]:
                continue
            res = cv2.matchTemplate(strip, template, cv2.TM_CCOEFF_NORMED)
            _, mx, _, mxloc = cv2.minMaxLoc(res)
            if mx > best_score:
                best_score = mx
                best_xr    = xr_min + mxloc[0] + p
        if best_score >= thresh:
            good_l.append([xl, yl])
            good_r.append([best_xr, yl])
    if not good_l:
        return np.zeros((0,2),np.float32), np.zeros((0,2),np.float32)
    return np.array(good_l,np.float32), np.array(good_r,np.float32)


def triangulate(pts_l, pts_r, f, cx, cy, baseline):
    disp = pts_l[:,0] - pts_r[:,0]
    ok   = disp > 1.0
    d    = disp[ok]
    Z    = f * baseline / d
    X    = (pts_l[ok,0] - cx) * Z / f
    Y    = (pts_l[ok,1] - cy) * Z / f
    depth_ok = (Z > Z_MIN_MM) & (Z < Z_MAX_MM)
    return np.stack([X[depth_ok], Y[depth_ok], Z[depth_ok]], axis=1).astype(np.float32)


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

        self.surf_buf = deque(maxlen=BUFFER_FRAMES)
        self.ref_Zi   = None
        self.has_ref  = False
        self.ncc_thresh = NCC_THRESH

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
                elif cmd == "thresh_up":
                    self.ncc_thresh = min(self.ncc_thresh + 0.05, 0.95)
                    print(f"NCC threshold: {self.ncc_thresh:.2f}")
                elif cmd == "thresh_dn":
                    self.ncc_thresh = max(self.ncc_thresh - 0.05, 0.05)
                    print(f"NCC threshold: {self.ncc_thresh:.2f}")

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

            feats = fixed_grid_points(self.roi)
            if len(feats) == 0:
                continue

            pl, pr = match_ncc(gray0, gray1, feats,
                                thresh=self.ncc_thresh)
            if len(pl) == 0:
                continue

            # ── Build debug overlay image ──────────────────────────────────
            dbg = rect0.copy()
            if self.roi:
                rx, ry, rw, rh = self.roi
                cv2.rectangle(dbg, (rx, ry), (rx+rw, ry+rh), (0, 220, 255), 1)
            for pt in feats:
                cv2.circle(dbg, (int(pt[0]), int(pt[1])), 2, (80, 80, 80), -1)
            if len(pl) > 0:
                disp_arr = pl[:, 0] - pr[:, 0]
                d_lo, d_hi = float(disp_arr.min()), float(disp_arr.max())
                for i in range(len(pl)):
                    t = (disp_arr[i] - d_lo) / max(d_hi - d_lo, 1.0)
                    color = (int(255*(1-t)), int(180*t), int(255*t))
                    cv2.circle(dbg, (int(pl[i,0]), int(pl[i,1])), 4, color, -1)
            cv2.putText(dbg,
                        f"matched {len(pl)}/{len(feats)}  NCC>={self.ncc_thresh:.2f}",
                        (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            try:
                self.debug_q.put_nowait(dbg)
            except queue.Full:
                try: self.debug_q.get_nowait()
                except queue.Empty: pass
                try: self.debug_q.put_nowait(dbg)
                except queue.Full: pass
            # ──────────────────────────────────────────────────────────────

            pts3d = triangulate(pl, pr, self.f, self.cx, self.cy, self.baseline)
            if len(pts3d) == 0:
                continue

            # Fit surface for smoothing
            if HAS_SCIPY:
                x0,x1 = pts3d[:,0].min(), pts3d[:,0].max()
                y0,y1 = pts3d[:,1].min(), pts3d[:,1].max()
                if x1-x0 > 1 and y1-y0 > 1:
                    xi = np.linspace(x0, x1, 60)
                    yi = np.linspace(y0, y1, 20)
                    Xi, Yi = np.meshgrid(xi, yi)
                    try:
                        Zi = griddata((pts3d[:,0], pts3d[:,1]), pts3d[:,2],
                                      (Xi, Yi), method="linear")
                        self.surf_buf.append((Xi, Yi, Zi, pts3d[:,0].min(),
                                              pts3d[:,0].max(),
                                              pts3d[:,1].min(),
                                              pts3d[:,1].max()))
                    except Exception:
                        pass

            # Build smooth cloud from buffer
            if len(self.surf_buf) > 0:
                # Use latest fit's grid but median Z
                try:
                    Xi_ref = self.surf_buf[-1][0]
                    Yi_ref = self.surf_buf[-1][1]
                    Zi_stack = np.stack([s[2] for s in self.surf_buf], axis=0)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        Zi_med = np.nanmedian(Zi_stack, axis=0)

                    valid = ~np.isnan(Zi_med)
                    pts_smooth = np.stack([Xi_ref[valid], Yi_ref[valid], Zi_med[valid]], axis=1)

                    deform = None
                    if self.has_ref and self.ref_Zi is not None:
                        if self.ref_Zi.shape == Zi_med.shape:
                            dZ = Zi_med - self.ref_Zi
                            dZ[np.isnan(dZ)] = 0
                            deform = dZ[valid]

                    # Handle reference capture
                    if hasattr(self, '_capture_ref') and self._capture_ref:
                        self.ref_Zi  = Zi_med.copy()
                        self.has_ref = True
                        self._capture_ref = False
                        n_valid = valid.sum()
                        med_z = float(np.nanmedian(Zi_med[valid]))
                        print(f"Reference captured — {n_valid} pts, med depth {med_z:.1f}mm")

                    payload = {
                        "pts":    pts_smooth,
                        "deform": deform,
                        "n_raw":  len(pl),
                        "buf":    len(self.surf_buf),
                    }
                    try:
                        self.out_q.put_nowait(payload)
                    except queue.Full:
                        try:
                            self.out_q.get_nowait()
                            self.out_q.put_nowait(payload)
                        except Exception:
                            pass
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

        pts    = payload["pts"]
        deform = payload["deform"]
        n_raw  = payload["n_raw"]
        buf    = payload["buf"]

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

        # Auto-set camera center on first data
        if buf == 1:
            centre = pts.mean(axis=0)
            view.camera.center = tuple(centre)

        buf_pct = int(buf / BUFFER_FRAMES * 100)
        ref_str = "REF captured" if cap_thread.has_ref else f"buf={buf_pct}% — press R when stable"
        status_text.text = (f"Points: {len(pts):,}  matched: {n_raw}  {ref_str}  "
                            f"| R=ref  D=cam  Q=quit  +/-=NCC")

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
        elif event.key == "D":
            show_debug[0] = not show_debug[0]
        elif event.key == "+":
            cap_thread.cmd_q.put("thresh_up")
        elif event.key == "-":
            cap_thread.cmd_q.put("thresh_dn")

    canvas.show()
    print("\nvispy window open.")
    print("Left-drag = rotate  |  Scroll = zoom  |  Right-drag = pan")
    print("R = capture reference  |  D = toggle camera view  |  Q = quit\n")

    app.run()
    cap_thread.stop()


if __name__ == "__main__":
    main()
