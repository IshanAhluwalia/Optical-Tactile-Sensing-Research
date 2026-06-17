"""
Flow computation — fake wave generation dataset.

For every frame, compute dense optical flow (Farneback) against the reference
frame (frame_00000 = unloaded skin).  The full per-pixel flow field is then
aggregated into a spatial grid so we get a mean displacement vector (u, v) and
magnitude for each region of the skin.

Outputs saved to fake_wave_generation/:
  flow_data.npz
    flow_uv  : (N, GRID_ROWS, GRID_COLS, 2)  — mean (u, v) per cell, per frame
    flow_mag : (N, GRID_ROWS, GRID_COLS)     — ||uv|| per cell, per frame
    roi      : [x1, y1, x2, y2]             — pattern band bounds used
    grid     : [GRID_COLS, GRID_ROWS]        — grid dimensions
"""

import cv2
import numpy as np
import os
import glob

# ── Config ────────────────────────────────────────────────────────────────────
FRAMES_DIR = os.path.join(os.path.dirname(__file__), 'frames')
OUT_DIR    = os.path.dirname(__file__)

GRID_COLS  = 16   # horizontal divisions across the skin
GRID_ROWS  = 6    # vertical divisions

# Farneback dense optical flow parameters:
#   pyr_scale  — pyramid downscale ratio (0.5 = halve each level)
#   levels     — number of pyramid levels (more = captures larger motions)
#   winsize    — averaging window size (larger = smoother, less detail)
#   iterations — passes per pyramid level
#   poly_n     — pixel neighbourhood for polynomial fit (5 or 7)
#   poly_sigma — Gaussian smoothing for polynomial (1.1 for poly_n=5)
FB_PARAMS = dict(
    pyr_scale  = 0.5,
    levels     = 3,
    winsize    = 15,
    iterations = 3,
    poly_n     = 5,
    poly_sigma = 1.1,
    flags      = 0,
)

# ── 1. Load reference frame ───────────────────────────────────────────────────
# Frame 0 = unloaded skin, used as the baseline for all measurements.
# Every subsequent frame's flow is relative to this state.
ref_path = os.path.join(FRAMES_DIR, 'frame_00000.png')
ref      = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
H, W     = ref.shape
print(f'Reference frame loaded: {W}x{H}px')

# ── 2. Auto-detect the pattern band ROI ──────────────────────────────────────
# The pattern band is a horizontal strip; top and bottom are black.
# We find the bounding box of non-black pixels in the reference frame so flow
# is only computed where there is actual maze texture.
row_sum  = ref.sum(axis=1).astype(float)  # brightness summed across each row
col_sum  = ref.sum(axis=0).astype(float)  # brightness summed down each col

row_thresh = row_sum.max() * 0.05   # rows with at least 5% of peak brightness
col_thresh = col_sum.max() * 0.05

active_rows = np.where(row_sum > row_thresh)[0]
active_cols = np.where(col_sum > col_thresh)[0]

y1, y2 = int(active_rows[0]),  int(active_rows[-1])
x1, x2 = int(active_cols[0]),  int(active_cols[-1])

print(f'Pattern ROI detected: x={x1}:{x2}, y={y1}:{y2}  →  {x2-x1}×{y2-y1}px')

ref_roi       = ref[y1:y2, x1:x2]
roi_h, roi_w  = ref_roi.shape

# ── 3. Grid cell boundaries ───────────────────────────────────────────────────
# Divide the ROI evenly into GRID_ROWS × GRID_COLS cells.
# Each cell gets a single representative (u, v) vector = spatial average of all
# pixel-level flow vectors inside it.
cell_h = roi_h // GRID_ROWS
cell_w = roi_w // GRID_COLS
print(f'Grid: {GRID_COLS} cols × {GRID_ROWS} rows  |  cell size: ~{cell_w}×{cell_h}px')

# ── 4. Process all frames ─────────────────────────────────────────────────────
frame_paths = sorted(glob.glob(os.path.join(FRAMES_DIR, 'frame_*.png')))
N           = len(frame_paths)
print(f'\nProcessing {N} frames...')

# Pre-allocate output arrays
flow_uv  = np.zeros((N, GRID_ROWS, GRID_COLS, 2), dtype=np.float32)
flow_mag = np.zeros((N, GRID_ROWS, GRID_COLS),    dtype=np.float32)

for i, path in enumerate(frame_paths):
    frame   = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    roi     = frame[y1:y2, x1:x2]

    # Dense flow: returns array of shape (roi_h, roi_w, 2)
    #   flow[y, x, 0] = u = horizontal displacement in pixels
    #   flow[y, x, 1] = v = vertical   displacement in pixels
    flow = cv2.calcOpticalFlowFarneback(ref_roi, roi, None, **FB_PARAMS)

    # Aggregate: mean (u, v) within each grid cell
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            ys = slice(r * cell_h, (r + 1) * cell_h)
            xs = slice(c * cell_w, (c + 1) * cell_w)
            cell_uv          = flow[ys, xs, :]            # (cell_h, cell_w, 2)
            mean_uv          = cell_uv.mean(axis=(0, 1))  # (2,)
            flow_uv[i, r, c] = mean_uv
            flow_mag[i, r, c]= np.linalg.norm(mean_uv)

    if (i + 1) % 100 == 0:
        print(f'  {i + 1}/{N}', flush=True)

print(f'  {N}/{N} — done')

# ── 5. Save results ───────────────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, 'flow_data.npz')
np.savez_compressed(
    out_path,
    flow_uv  = flow_uv,
    flow_mag = flow_mag,
    roi      = np.array([x1, y1, x2, y2]),
    grid     = np.array([GRID_COLS, GRID_ROWS]),
)

print(f'\nSaved: {out_path}')
print(f'  flow_uv  shape: {flow_uv.shape}   (frames × rows × cols × uv)')
print(f'  flow_mag shape: {flow_mag.shape}  (frames × rows × cols)')
print(f'  Max magnitude:  {flow_mag.max():.3f} px')
print(f'  Mean magnitude: {flow_mag.mean():.3f} px')
