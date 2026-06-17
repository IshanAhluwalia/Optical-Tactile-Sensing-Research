"""
Live inference viewer for DenseContactNet.

Opens the camera, crops to the sensor ROI, runs the model every frame,
and shows a live dashboard:

  Left  : full camera frame with ROI box drawn
  Right : contact / depth / pressure heatmaps + scalar readouts

Press Q or ESC to quit.
"""

import json
import os
import sys
import time

import cv2
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from model import DenseContactNet

# ── Load config ───────────────────────────────────────────────────────────────
with open(os.path.join(_HERE, 'roi.json')) as f:
    _roi = json.load(f)
X0, X1 = _roi['x_start'], _roi['x_end']
Y0, Y1 = _roi['y_start'], _roi['y_end']

with open(os.path.join(_HERE, 'model', 'model_stats.json')) as f:
    stats = json.load(f)
DISP_MAX  = stats['disp_max']
FORCE_MAX = stats['force_max']

# ── Image normalisation (same as training) ────────────────────────────────────
GRAY_MEAN = 0.4513
GRAY_STD  = 0.2898

# ── Device ────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f"Device: {device}")

# ── Load model ────────────────────────────────────────────────────────────────
model = DenseContactNet().to(device)
model.load_state_dict(torch.load(
    os.path.join(_HERE, 'model', 'best_model.pth'), map_location=device
))
model.eval()
print("Model loaded.")

# ── Camera ────────────────────────────────────────────────────────────────────
def _find_cam(target_w=640, target_h=480, fallback=0):
    for i in range(6):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if w == target_w and h == target_h:
                return i
    return fallback

cap = cv2.VideoCapture(_find_cam())
cap.set(cv2.CAP_PROP_FPS, 30)
if not cap.isOpened():
    print("ERROR: could not open camera")
    sys.exit(1)
print("Camera ready. Press Q or ESC to quit.")

# ── Helpers ───────────────────────────────────────────────────────────────────
def _homomorphic(gray_u8: np.ndarray, sigma: float = 60) -> np.ndarray:
    log_img = np.log(gray_u8.astype(np.float32) + 1.0)
    blur = cv2.GaussianBlur(log_img, (0, 0), sigma)
    filtered = 0.5 * blur + 1.5 * (log_img - blur)
    out = np.exp(filtered) - 1.0
    return cv2.normalize(out, None, 0.0, 255.0, cv2.NORM_MINMAX).astype(np.uint8)


def preprocess(bgr_frame):
    crop  = bgr_frame[Y0:Y1, X0:X1]
    gray  = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray  = _homomorphic(gray)
    resized = cv2.resize(gray, (224, 224))
    arr   = resized.astype(np.float32) / 255.0
    arr   = (arr - GRAY_MEAN) / GRAY_STD
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1,1,224,224)

def map_to_heatmap(arr_2d, colormap=cv2.COLORMAP_JET):
    """Convert (H,W) float array [0,1] to a BGR colour image at MAP_H x MAP_W."""
    norm = np.clip(arr_2d, 0, 1)
    u8   = (norm * 255).astype(np.uint8)
    big  = cv2.resize(u8, (MAP_W, MAP_H), interpolation=cv2.INTER_NEAREST)
    return cv2.applyColorMap(big, colormap)

def labeled_map(heatmap, label):
    out = heatmap.copy()
    cv2.putText(out, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    return out

# ── Sensor grid constants ──────────────────────────────────────────────────
_GRID_X = np.array(
    [138,140,142,144,146,148,150,152,154,156,158,160,162,164,
     166,168,170,172,174,176,178,180,182,184,
     186,188,190,192,194,196,198,200,202,204,206,208,210],
    dtype=np.float32,
)
from dataset import GRID_Y as _GRID_Y
_GX_MIN, _GX_MAX = _GRID_X[0], _GRID_X[-1]   # 138, 210
_GY_MIN, _GY_MAX = _GRID_Y[0],  _GRID_Y[-1]   # 0, 16

# Physical sensor: 72 mm wide × 16 mm tall → 4.5 : 1 aspect ratio
_PX_PER_MM   = 8                                     # pixels per mm
_GRID_MARG   = 20                                    # canvas border in px
GRID_INNER_W = int((_GX_MAX - _GX_MIN) * _PX_PER_MM)  # 576 px
GRID_INNER_H = int((_GY_MAX - _GY_MIN) * _PX_PER_MM)  # 128 px
GRID_VIZ_W   = GRID_INNER_W + 2 * _GRID_MARG          # 616 px
GRID_VIZ_H   = GRID_INNER_H + 2 * _GRID_MARG          # 168 px

def draw_sensor_grid(contact_map, loc_x, loc_y, in_contact):
    """
    Render the 37×9 sensor grid at its true physical aspect ratio (72 mm × 16 mm).
    Dot positions are mapped directly from mm → px so x and y scales are identical.
    """
    canvas = np.zeros((GRID_VIZ_H, GRID_VIZ_W, 3), dtype=np.uint8)

    # Heatmap background — stretch (9,37) map to exact inner pixel area
    hmap     = np.clip(contact_map, 0, 1)
    hmap_big = cv2.resize(hmap, (GRID_INNER_W, GRID_INNER_H), interpolation=cv2.INTER_LINEAR)
    hmap_u8  = (hmap_big * 255).astype(np.uint8)
    hmap_bgr = cv2.applyColorMap(hmap_u8, cv2.COLORMAP_HOT)
    canvas[_GRID_MARG:_GRID_MARG + GRID_INNER_H,
           _GRID_MARG:_GRID_MARG + GRID_INNER_W] = hmap_bgr

    # Grid dots — position derived from physical mm, same scale on both axes
    for ri, gy in enumerate(_GRID_Y):
        for ci, gx in enumerate(_GRID_X):
            px = _GRID_MARG + int((gx - _GX_MIN) * _PX_PER_MM)
            py = _GRID_MARG + int((gy - _GY_MIN) * _PX_PER_MM)
            val = float(contact_map[ri, ci])
            b   = int(50 + val * 205)
            cv2.circle(canvas, (px, py), 2, (b, b, b), -1)

    # Crosshair at predicted contact location
    if in_contact:
        cx = _GRID_MARG + int((loc_x - _GX_MIN) * _PX_PER_MM)
        cy = _GRID_MARG + int((loc_y - _GY_MIN) * _PX_PER_MM)
        cx = int(np.clip(cx, _GRID_MARG, _GRID_MARG + GRID_INNER_W - 1))
        cy = int(np.clip(cy, _GRID_MARG, _GRID_MARG + GRID_INNER_H - 1))
        cv2.drawMarker(canvas, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 16, 2)
        cv2.circle(canvas, (cx, cy), 6, (0, 255, 0), 2)

    # Labels
    cv2.putText(canvas, "contact grid", (8, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(canvas, "138 mm", (_GRID_MARG, GRID_VIZ_H - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (110, 110, 110), 1)
    cv2.putText(canvas, "210 mm", (GRID_VIZ_W - 55, GRID_VIZ_H - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (110, 110, 110), 1)
    return canvas

# ── Main loop ─────────────────────────────────────────────────────────────────
fps_t = time.time()
fps   = 0.0
frame_n = 0

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # ── Inference ─────────────────────────────────────────────────────────────
    tensor = preprocess(frame).to(device)
    with torch.no_grad():
        out = model(tensor)

    contact  = out['contact_map'][0].cpu().numpy()    # (9,37)
    depth    = out['depth_map'][0].cpu().numpy()      # (9,37)
    pressure = out['pressure_map'][0].cpu().numpy()   # (9,37)
    loc_x    = out['loc_x'][0].item()
    loc_y    = out['loc_y'][0].item()
    disp_mm  = out['displacement'][0].item() * DISP_MAX
    force_n  = out['force'][0].item() * FORCE_MAX

    contact_peak = float(contact.max())
    in_contact   = contact_peak > 0.03

    # ── FPS ───────────────────────────────────────────────────────────────────
    frame_n += 1
    if frame_n % 10 == 0:
        fps = 10.0 / (time.time() - fps_t)
        fps_t = time.time()

    # ── Left panel: camera with ROI box ──────────────────────────────────────
    display = frame.copy()
    cv2.rectangle(display, (X0, Y0), (X1, Y1), (0, 255, 0), 2)
    cv2.putText(display, f"{fps:.1f} fps", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # ── Right panel: grid (correct aspect) on top, scalars below ─────────────
    grid_panel   = draw_sensor_grid(contact, loc_x, loc_y, in_contact)

    scalar_h     = display.shape[0] - GRID_VIZ_H   # fills remaining height
    scalar_panel = np.zeros((scalar_h, GRID_VIZ_W, 3), dtype=np.uint8)

    if in_contact:
        lines  = [
            f"loc_x : {loc_x:6.1f} mm",
            f"loc_y : {loc_y:6.1f} mm",
            f"disp  : {disp_mm:6.3f} mm",
            f"force : {force_n:6.4f} N",
        ]
        colors = [(0, 255, 0)] * 4
    else:
        lines  = ["  -- no contact --", "", "", ""]
        colors = [(80, 80, 80)] * 4

    lines  += ["", f"confidence: {contact_peak:.3f}"]
    colors += [(180, 180, 180), (180, 180, 180)]

    for i, (line, color) in enumerate(zip(lines, colors)):
        cv2.putText(scalar_panel, line, (10, 50 + i * 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    right_panel = np.vstack([grid_panel, scalar_panel])

    # ── Camera panel at natural resolution, hstack with right panel ──────────
    combined = np.hstack([display, right_panel])
    cv2.imshow("DenseContactNet — Live", combined)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), 27):
        break

cap.release()
cv2.destroyAllWindows()
