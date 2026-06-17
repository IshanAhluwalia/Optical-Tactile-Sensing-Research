"""
Live 3D reconstruction from the tactile sensor camera using PointCloudNet.

Opens the camera, crops to the sensor ROI, runs PointCloudNet every frame,
and renders the predicted deformed skin surface as a continuous mesh surface
(interpolated between the 32×64 structured grid points).

Controls
--------
    Mouse drag   – rotate
    Scroll       – zoom
    R            – toggle rest surface overlay
    W            – toggle wireframe
    Q / Escape   – quit

Usage
-----
    python dense_contact/infer_pointcloud_live.py
"""

import json
import os
import sys
import time as _time

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from vispy import app, scene
from vispy.geometry import MeshData

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from model import PointCloudNet, PC_GRID_ROWS, PC_GRID_COLS, PC_N_PTS

# ── Config ────────────────────────────────────────────────────────────────────
REST_POS_PATH = os.path.join(_HERE, 'rest_positions.npy')
MODEL_PATH    = os.path.join(_HERE, 'model_pc', 'best_model.pth')
STATS_PATH    = os.path.join(_HERE, 'model_pc', 'model_stats.json')
ROI_PATH      = os.path.join(_HERE, 'roi.json')

GRAY_MEAN = [0.4513]
GRAY_STD  = [0.2898]

_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(GRAY_MEAN, GRAY_STD),
])

# ── Build mesh face index array (pre-computed once) ───────────────────────────
# Grid is PC_GRID_ROWS × PC_GRID_COLS; each quad → 2 triangles.
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

FACES = _build_faces(PC_GRID_ROWS, PC_GRID_COLS)   # (N_tri, 3)

# ── Device ────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f"Device: {device}")

# ── Load model ────────────────────────────────────────────────────────────────
model = PointCloudNet(rest_positions_path=REST_POS_PATH).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print("Model loaded.")

rest = np.load(REST_POS_PATH).astype(np.float32)   # (2048, 3)
rest_dz_range = max(rest[:, 2].max() - rest[:, 2].min(), 1e-6)

with open(STATS_PATH) as f:
    stats = json.load(f)
DISP_MAX  = stats['disp_max']
FORCE_MAX = stats['force_max']

with open(ROI_PATH) as f:
    roi = json.load(f)
X0, X1 = roi['x_start'], roi['x_end']
Y0, Y1 = roi['y_start'], roi['y_end']

# ── Camera ────────────────────────────────────────────────────────────────────
def _find_cam(target_w=640, target_h=480):
    for i in range(8):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if w == target_w and h == target_h:
                return i
    return 0

cap = cv2.VideoCapture(_find_cam())
cap.set(cv2.CAP_PROP_FPS, 30)
if not cap.isOpened():
    print("ERROR: could not open camera")
    sys.exit(1)
print("Camera ready. Press Q or Escape to quit.")

# ── Colour helpers ────────────────────────────────────────────────────────────
def vertex_colors(pts: np.ndarray) -> np.ndarray:
    """Blue→red per vertex, by inward Z displacement relative to rest."""
    dz   = rest[:, 2] - pts[:, 2]
    norm = np.clip(dz / (dz.max() + 1e-6), 0.0, 1.0)
    c    = np.zeros((len(pts), 4), dtype=np.float32)
    c[:, 0] = norm
    c[:, 1] = 0.15 + 0.3 * (1.0 - norm)
    c[:, 2] = 1.0 - norm
    c[:, 3] = 1.0
    return c

REST_COLORS = np.zeros((len(rest), 4), dtype=np.float32)
REST_COLORS[:, 2] = 0.4
REST_COLORS[:, 3] = 0.20   # semi-transparent

# ── Canvas ────────────────────────────────────────────────────────────────────
canvas = scene.SceneCanvas(
    title='PointCloudNet — Live 3D mesh   [drag=rotate  scroll=zoom  R=rest  W=wireframe  Q=quit]',
    bgcolor='#080816',
    size=(1000, 700),
    keys='interactive',
)
view = canvas.central_widget.add_view()
view.camera = scene.cameras.TurntableCamera(
    fov=40, distance=60, elevation=25, azimuth=30,
)
view.camera.center = tuple(rest.mean(axis=0))

# Initialise mesh with dummy data (will be overwritten on first timer tick)
_init_colors = vertex_colors(rest)
mesh_live = scene.visuals.Mesh(
    vertices=rest, faces=FACES,
    vertex_colors=_init_colors, shading=None,
    parent=view.scene,
)
mesh_rest = scene.visuals.Mesh(
    vertices=rest, faces=FACES,
    vertex_colors=REST_COLORS, shading=None,
    parent=view.scene,
)
scene.visuals.XYZAxis(parent=view.scene)

status = scene.visuals.Text(
    'Initialising…', color='white', font_size=10,
    anchor_x='left', anchor_y='bottom', parent=canvas.scene,
)
status.transform = scene.transforms.STTransform(translate=(10, 10))

state = {
    'show_rest': True,
    'wireframe': False,
    'fps': 0.0,
    'frame_n': 0,
    'last_t': None,
}

# ── Timer callback ─────────────────────────────────────────────────────────────
def on_timer(event):
    ret, frame = cap.read()
    if not ret:
        return

    crop    = frame[Y0:Y1, X0:X1]
    gray    = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    pil_img = Image.fromarray(gray)
    tensor  = _TRANSFORM(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(tensor)

    pts     = out['point_cloud'][0].cpu().numpy()   # (2048, 3)
    disp_mm = out['displacement'][0].item() * DISP_MAX
    force_n = out['force'][0].item() * FORCE_MAX

    cols = vertex_colors(pts)

    mesh_live.set_data(vertices=pts, faces=FACES, vertex_colors=cols)

    if state['show_rest']:
        mesh_rest.set_data(vertices=rest, faces=FACES, vertex_colors=REST_COLORS)
        mesh_rest.visible = True
    else:
        mesh_rest.visible = False

    # FPS tracking
    state['frame_n'] += 1
    now = _time.time()
    if state['last_t'] is None:
        state['last_t'] = now
    elif state['frame_n'] % 10 == 0:
        state['fps'] = 10.0 / (now - state['last_t'])
        state['last_t'] = now

    dz_peak = float((rest[:, 2] - pts[:, 2]).max())
    status.text = (
        f"disp={disp_mm:.2f}mm   force={force_n:.4f}N   "
        f"peak_dz={dz_peak:.2f}mm   {state['fps']:.1f} fps   "
        f"[R=rest {'ON' if state['show_rest'] else 'OFF'}  "
        f"W=wireframe {'ON' if state['wireframe'] else 'OFF'}  Q=quit]"
    )
    canvas.update()


timer = app.Timer(interval=1 / 30, connect=on_timer, start=True)


@canvas.events.key_press.connect
def on_key(ev):
    k = ev.key
    if k in ('Q', 'Escape'):
        cap.release()
        app.quit()
    elif k == 'R':
        state['show_rest'] = not state['show_rest']
    elif k == 'W':
        state['wireframe'] = not state['wireframe']
        mesh_live.set_data(
            vertices=mesh_live.mesh_data.get_vertices(),
            faces=FACES,
            vertex_colors=mesh_live.mesh_data.get_vertex_colors(),
        )
        # Vispy doesn't have a direct wireframe toggle on Mesh;
        # switch between filled and a line-based draw mode.
        mesh_live.mode = 'lines' if state['wireframe'] else 'triangles'


canvas.show()
app.run()

cap.release()
