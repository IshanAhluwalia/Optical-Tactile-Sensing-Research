"""
Interactive 3D point cloud viewer — browse frames from dataset.csv.

Controls
--------
    Arrow Up / Down   – step through frames (deeper / shallower)
    Arrow Left / Right– switch session (different contact position)
    Mouse drag        – rotate view
    Scroll            – zoom
    Q / Escape        – quit

Usage
-----
    python view_pointclouds.py
    python view_pointclouds.py --session x174_y8
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from vispy import app, scene

_HERE    = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.normpath(os.path.join(_HERE, "..", "dense_contact", "dataset.csv"))


def cloud_path(image_path: str) -> str:
    return (image_path
            .replace("/images/", "/pointclouds/")
            .replace(".png", ".npy"))


def pts_to_color(pts):
    x_ref  = pts[:, 0].max()
    deform = x_ref - pts[:, 0]
    norm   = np.clip(deform / (deform.max() + 1e-6), 0.0, 1.0)
    c = np.zeros((len(pts), 4), dtype=np.float32)
    c[:, 0] = norm
    c[:, 1] = 0.3 + 0.5 * norm
    c[:, 2] = 1.0 - 0.9 * norm
    c[:, 3] = 0.9
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",     default=CSV_PATH)
    ap.add_argument("--session", default="x174_y8",
                    help="Starting session, e.g. x174_y8")
    args = ap.parse_args()

    df       = pd.read_csv(args.csv)
    sessions = sorted(df["session"].unique().tolist())

    if args.session not in sessions:
        args.session = sessions[0]

    state = {
        "si":    sessions.index(args.session),  # session index
        "fi":    0,                              # frame index within session
    }

    def session_frames():
        s = sessions[state["si"]]
        return df[df["session"] == s].reset_index(drop=True)

    # ── Canvas ────────────────────────────────────────────────────────────────
    canvas = scene.SceneCanvas(
        title="Point cloud viewer  [↑↓ frames  ←→ sessions  drag=rotate  scroll=zoom  Q=quit]",
        bgcolor="#0d0d1a",
        size=(1200, 800),
        keys="interactive",
    )
    view = canvas.central_widget.add_view()
    view.camera = scene.cameras.TurntableCamera(
        fov=40, distance=100, elevation=20, azimuth=30,
    )

    scatter = scene.visuals.Markers()
    view.add(scatter)
    scene.visuals.XYZAxis(parent=view.scene)

    status = scene.visuals.Text(
        "", color="white", font_size=10,
        anchor_x="left", anchor_y="bottom",
        parent=canvas.scene,
    )
    status.transform = scene.transforms.STTransform(translate=(10, 10))

    def redraw():
        frames = session_frames()
        fi     = min(state["fi"], len(frames) - 1)
        state["fi"] = fi
        row    = frames.iloc[fi]

        cp = cloud_path(row["image_path"])
        if not os.path.exists(cp):
            status.text = f"[cloud missing]  {cp}"
            canvas.update()
            return

        pts  = np.load(cp).astype(np.float32)
        cols = pts_to_color(pts)

        # Centre camera on this cloud
        ctr = pts.mean(axis=0)
        view.camera.center = (float(ctr[0]), float(ctr[1]), float(ctr[2]))

        scatter.set_data(pts, face_color=cols, size=4, edge_width=0)

        session = sessions[state["si"]]
        status.text = (
            f"Session: {session}   "
            f"loc_x={row['loc_x']:.0f} mm  loc_y={row['loc_y']:.0f} mm   "
            f"depth={row['displacement_mm']:.2f} mm   "
            f"force={row['force_n']:.3f} N   "
            f"frame {fi+1}/{len(frames)}   "
            f"| ↑↓ frames   ←→ sessions   drag rotate   scroll zoom"
        )
        canvas.update()

    @canvas.events.key_press.connect
    def on_key(ev):
        k = ev.key
        if k in ("Q", "Escape"):
            app.quit()
        elif k == "Up":
            state["fi"] = min(state["fi"] + 5, len(session_frames()) - 1)
        elif k == "Down":
            state["fi"] = max(state["fi"] - 5, 0)
        elif k == "Right":
            state["si"] = (state["si"] + 1) % len(sessions)
            state["fi"] = len(session_frames()) // 2   # start at mid-depth
        elif k == "Left":
            state["si"] = (state["si"] - 1) % len(sessions)
            state["fi"] = len(session_frames()) // 2
        redraw()

    # Start at mid-depth frame (most visible deformation)
    state["fi"] = len(session_frames()) // 2
    redraw()
    canvas.show()
    app.run()


if __name__ == "__main__":
    main()
