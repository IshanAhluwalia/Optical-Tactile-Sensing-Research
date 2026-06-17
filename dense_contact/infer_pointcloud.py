"""
Interactive viewer: PointCloudNet predicted vs ground-truth 3D mesh surfaces.

The 32×64 structured grid is triangulated into a continuous mesh so the surface
is interpolated between grid points (no gaps / discrete dots).

Controls
--------
    Arrow Right / Left  – next / previous frame
    Arrow Up / Down     – skip 10 frames forward / back
    S                   – jump to a random session
    Mouse drag          – rotate both views in sync
    Scroll              – zoom
    Q / Escape          – quit

Usage
-----
    python dense_contact/infer_pointcloud.py
    python dense_contact/infer_pointcloud.py --split val   # default
    python dense_contact/infer_pointcloud.py --split train
    python dense_contact/infer_pointcloud.py --session x174_y8
"""

import argparse
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from vispy import app, scene

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from model import PointCloudNet, PC_GRID_ROWS, PC_GRID_COLS
from dataset import homomorphic_deglare

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH      = os.path.join(_HERE, 'dataset.csv')
REST_POS_PATH = os.path.join(_HERE, 'rest_positions.npy')
MODEL_PATH    = os.path.join(_HERE, 'model_pc', 'best_model.pth')
STATS_PATH    = os.path.join(_HERE, 'model_pc', 'model_stats.json')

VAL_FRACTION  = 0.2
RANDOM_SEED   = 42

GRAY_MEAN = [0.4513]
GRAY_STD  = [0.2898]

_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(GRAY_MEAN, GRAY_STD),
])

# ── Mesh face indices (pre-computed once for the 32×64 grid) ──────────────────
def _build_faces(rows: int, cols: int) -> np.ndarray:
    faces = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            i0 = r * cols + c
            i1 = r * cols + c + 1
            i2 = (r + 1) * cols + c
            i3 = (r + 1) * cols + c + 1
            faces.append([i0, i1, i2])
            faces.append([i1, i3, i2])
    return np.array(faces, dtype=np.uint32)

FACES = _build_faces(PC_GRID_ROWS, PC_GRID_COLS)

# ── Session split (matches training) ─────────────────────────────────────────
def get_sessions(split: str):
    sessions = sorted(pd.read_csv(CSV_PATH)['session'].unique().tolist())
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(sessions)
    n_val = max(1, int(len(sessions) * VAL_FRACTION))
    val   = sessions[:n_val]
    train = sessions[n_val:]
    return val if split == 'val' else train

# ── Cloud path helper ─────────────────────────────────────────────────────────
def cloud_path(image_path: str) -> str:
    return (image_path
            .replace('/images/', '/pointclouds/')
            .replace('.png', '.npy'))

# ── Vertex colouring ──────────────────────────────────────────────────────────
def vertex_colors(pts: np.ndarray, rest: np.ndarray) -> np.ndarray:
    """Blue→red by inward Z displacement from rest surface."""
    dz   = rest[:, 2] - pts[:, 2]
    norm = np.clip(dz / (dz.max() + 1e-6), 0.0, 1.0)
    c    = np.zeros((len(pts), 4), dtype=np.float32)
    c[:, 0] = norm
    c[:, 1] = 0.15 + 0.3 * (1.0 - norm)
    c[:, 2] = 1.0 - norm
    c[:, 3] = 1.0
    return c


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split',   default='val', choices=['val', 'train'])
    ap.add_argument('--session', default=None)
    args = ap.parse_args()

    device = torch.device('mps') if torch.backends.mps.is_available() else (
             torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print(f"Device: {device}")

    model = PointCloudNet(rest_positions_path=REST_POS_PATH).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"Loaded model from {MODEL_PATH}")

    rest = np.load(REST_POS_PATH).astype(np.float32)

    import json
    with open(STATS_PATH) as f:
        stats = json.load(f)
    DISP_MAX  = stats['disp_max']
    FORCE_MAX = stats['force_max']

    df       = pd.read_csv(CSV_PATH)
    sessions = get_sessions(args.split)

    start_session = args.session if (args.session and args.session in sessions) else sessions[0]
    state = {
        'si': sessions.index(start_session),
        'fi': 0,
    }

    def session_frames():
        s = sessions[state['si']]
        return df[df['session'] == s].reset_index(drop=True)

    # ── Canvas: two side-by-side 3D mesh views ────────────────────────────────
    canvas = scene.SceneCanvas(
        title='PointCloudNet — Ground Truth (left)  |  Predicted (right)   '
              '[←→ frames  ↑↓ ×10  S=random  Q=quit]',
        bgcolor='#0d0d1a',
        size=(1600, 700),
        keys='interactive',
    )
    grid = canvas.central_widget.add_grid()

    vl = grid.add_view(row=0, col=0, bgcolor='#0d0d20')
    vl.camera = scene.cameras.TurntableCamera(fov=40, distance=60, elevation=25, azimuth=30)
    scene.visuals.XYZAxis(parent=vl.scene)

    vr = grid.add_view(row=0, col=1, bgcolor='#1a0d0d')
    vr.camera = scene.cameras.TurntableCamera(fov=40, distance=60, elevation=25, azimuth=30)
    scene.visuals.XYZAxis(parent=vr.scene)

    # Initialise meshes with rest-surface geometry
    _init_col = vertex_colors(rest, rest)
    mesh_gt = scene.visuals.Mesh(
        vertices=rest, faces=FACES, vertex_colors=_init_col,
        shading=None, parent=vl.scene,
    )
    mesh_pr = scene.visuals.Mesh(
        vertices=rest, faces=FACES, vertex_colors=_init_col,
        shading=None, parent=vr.scene,
    )

    label_gt = scene.visuals.Text(
        'Ground truth', color='#88ccff', font_size=11,
        anchor_x='left', anchor_y='top', parent=canvas.scene,
    )
    label_gt.transform = scene.transforms.STTransform(translate=(10, 30))

    label_pr = scene.visuals.Text(
        'Predicted', color='#ffaa88', font_size=11,
        anchor_x='left', anchor_y='top', parent=canvas.scene,
    )
    label_pr.transform = scene.transforms.STTransform(translate=(810, 30))

    status = scene.visuals.Text(
        '', color='white', font_size=9,
        anchor_x='left', anchor_y='bottom', parent=canvas.scene,
    )
    status.transform = scene.transforms.STTransform(translate=(10, 10))

    def redraw():
        frames = session_frames()
        fi     = min(state['fi'], len(frames) - 1)
        state['fi'] = fi
        row    = frames.iloc[fi]

        cp = cloud_path(row['image_path'])
        if not os.path.exists(cp):
            status.text = f'[GT cloud missing]  {cp}'
            canvas.update()
            return

        gt_pts = np.load(cp).astype(np.float32)

        # Model inference
        img    = Image.open(row['image_path']).convert('L')
        img    = homomorphic_deglare(img)
        tensor = _TRANSFORM(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(tensor)
        pr_pts   = out['point_cloud'][0].cpu().numpy()
        disp_mm  = out['displacement'][0].item() * DISP_MAX
        force_n  = out['force'][0].item() * FORCE_MAX

        rmse = float(np.sqrt(((pr_pts - gt_pts) ** 2).mean()))

        gt_col = vertex_colors(gt_pts, rest)
        pr_col = vertex_colors(pr_pts, rest)

        ctr = gt_pts.mean(axis=0)
        vl.camera.center = tuple(ctr)
        vr.camera.center = tuple(ctr)

        mesh_gt.set_data(vertices=gt_pts, faces=FACES, vertex_colors=gt_col)
        mesh_pr.set_data(vertices=pr_pts, faces=FACES, vertex_colors=pr_col)

        session = sessions[state['si']]
        status.text = (
            f"Session: {session}  ({args.split})   "
            f"loc_x={row['loc_x']:.0f}mm  loc_y={row['loc_y']:.0f}mm   "
            f"GT disp={float(row['displacement_mm']):.2f}mm  force={float(row['force_n']):.3f}N   "
            f"| Pred disp={disp_mm:.2f}mm  force={force_n:.3f}N   "
            f"| PC RMSE={rmse:.3f}mm   "
            f"frame {fi+1}/{len(frames)}   "
            f"[←→ frames  ↑↓ ×10  S=random  Q=quit]"
        )
        canvas.update()

    @canvas.events.key_press.connect
    def on_key(ev):
        k = ev.key
        if k in ('Q', 'Escape'):
            app.quit()
        elif k == 'Right':
            state['fi'] = min(state['fi'] + 1, len(session_frames()) - 1)
        elif k == 'Left':
            state['fi'] = max(state['fi'] - 1, 0)
        elif k == 'Up':
            state['fi'] = min(state['fi'] + 10, len(session_frames()) - 1)
        elif k == 'Down':
            state['fi'] = max(state['fi'] - 10, 0)
        elif k == 'S':
            state['si'] = random.randint(0, len(sessions) - 1)
            state['fi'] = len(session_frames()) // 2
        redraw()

    state['fi'] = len(session_frames()) // 2
    print(f"Session: {sessions[state['si']]}  ({len(session_frames())} frames)")
    redraw()
    canvas.show()
    app.run()


if __name__ == '__main__':
    main()
