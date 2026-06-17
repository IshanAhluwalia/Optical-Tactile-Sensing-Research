"""
Flow visualization — fake wave generation dataset.

Produces two outputs:
  fake_wave_generation/flow_summary.png  — static figure showing key frames
  fake_wave_generation/flow_video.mp4    — animated overlay on the pattern frames

Each frame is visualized with two layers:
  1. Magnitude heatmap — semi-transparent color fill per grid cell
                         (blue = still, red = high displacement)
  2. Quiver arrows     — direction and magnitude of flow per cell
"""

import cv2
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

FRAMES_DIR = os.path.join(os.path.dirname(__file__), 'frames')
OUT_DIR    = os.path.dirname(__file__)

# ── 1. Load flow data ─────────────────────────────────────────────────────────
data     = np.load(os.path.join(OUT_DIR, 'flow_data.npz'))
flow_uv  = data['flow_uv']    # (N, GRID_ROWS, GRID_COLS, 2)
flow_mag = data['flow_mag']   # (N, GRID_ROWS, GRID_COLS)
x1, y1, x2, y2 = data['roi']
GRID_COLS, GRID_ROWS = data['grid']

N      = flow_uv.shape[0]
roi_w  = x2 - x1
roi_h  = y2 - y1
cell_w = roi_w // GRID_COLS
cell_h = roi_h // GRID_ROWS

# Global max magnitude — used to normalise colours consistently across all frames
MAG_MAX = float(np.percentile(flow_mag, 99))  # 99th percentile avoids outlier saturation

# Cells with displacement below this are treated as stationary and shown dark in the
# flow grid.  Raise to filter more noise, lower to show subtler motion.
FLOW_THRESHOLD = MAG_MAX * 0.60

print(f'Loaded {N} frames  |  ROI {roi_w}×{roi_h}px  |  Grid {GRID_COLS}×{GRID_ROWS}')
print(f'Magnitude scale: 0 – {MAG_MAX:.2f}px  (99th percentile)')
print(f'Flow threshold:  {FLOW_THRESHOLD:.2f}px  (15% of MAG_MAX)')

# ── 2. Helper: render one frame as a BGR overlay image ────────────────────────
# Returns a BGR image of the ROI with heatmap + quiver drawn on top.
def render_frame(frame_idx: int, alpha: float = 0.55) -> np.ndarray:
    # Load the pattern frame and crop to ROI
    path  = os.path.join(FRAMES_DIR, f'frame_{frame_idx:05d}.png')
    gray  = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    crop  = gray[y1:y2, x1:x2]
    bgr   = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)

    overlay = bgr.copy()

    uv  = flow_uv[frame_idx]   # (GRID_ROWS, GRID_COLS, 2)
    mag = flow_mag[frame_idx]  # (GRID_ROWS, GRID_COLS)

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            # Cell pixel bounds
            cx1 = c * cell_w;  cx2 = cx1 + cell_w
            cy1 = r * cell_h;  cy2 = cy1 + cell_h

            # ── Heatmap fill ──
            # Normalise magnitude to [0, 1] and map to BGR via 'plasma' colormap
            norm  = float(np.clip(mag[r, c] / MAG_MAX, 0, 1))
            rgba  = cm.plasma(norm)                        # (R,G,B,A) in [0,1]
            color = (int(rgba[2]*255), int(rgba[1]*255), int(rgba[0]*255))  # BGR
            cv2.rectangle(overlay, (cx1, cy1), (cx2, cy2), color, -1)

            # ── Quiver arrow ──
            cx = cx1 + cell_w // 2   # cell centre x
            cy = cy1 + cell_h // 2   # cell centre y
            u, v = float(uv[r, c, 0]), float(uv[r, c, 1])

            # Scale arrow length: 1px displacement → 2px arrow for visibility
            SCALE = 2.0
            ex = int(cx + u * SCALE)
            ey = int(cy + v * SCALE)

            if mag[r, c] > 0.3:   # only draw arrows where there is meaningful motion
                cv2.arrowedLine(overlay, (cx, cy), (ex, ey),
                                (255, 255, 255), 1, tipLength=0.35)

    # Blend overlay with original frame for semi-transparency
    out = cv2.addWeighted(bgr, 1 - alpha, overlay, alpha, 0)
    return out


# ── 3. Flow grid renderer ─────────────────────────────────────────────────────
# Renders a standalone direction grid panel.
# Each cell is coloured by the angle of its flow vector (HSV colour wheel).
# Cells below FLOW_THRESHOLD are shown dark — no false arrows on stationary skin.
def render_flow_grid(frame_idx: int) -> np.ndarray:
    panel = np.full((GRID_ROWS * cell_h, GRID_COLS * cell_w, 3), 18, dtype=np.uint8)

    uv  = flow_uv[frame_idx]
    mag = flow_mag[frame_idx]

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            cx1 = c * cell_w;  cx2 = cx1 + cell_w
            cy1 = r * cell_h;  cy2 = cy1 + cell_h

            if mag[r, c] > FLOW_THRESHOLD:
                # Hue = direction (0-179 in OpenCV HSV)
                angle  = np.arctan2(float(uv[r, c, 1]), float(uv[r, c, 0]))
                hue    = int(((angle + np.pi) / (2 * np.pi)) * 179)
                val    = int(np.clip(mag[r, c] / MAG_MAX, 0, 1) * 255)
                hsv_px = np.array([[[hue, 220, val]]], dtype=np.uint8)
                bgr_px = cv2.cvtColor(hsv_px, cv2.COLOR_HSV2BGR)[0, 0]
                cv2.rectangle(panel, (cx1, cy1), (cx2, cy2),
                              (int(bgr_px[0]), int(bgr_px[1]), int(bgr_px[2])), -1)

                # Fixed-length arrow showing direction (not scaled by magnitude)
                cx_mid = cx1 + cell_w // 2
                cy_mid = cy1 + cell_h // 2
                u_, v_ = float(uv[r, c, 0]), float(uv[r, c, 1])
                length = min(cell_w, cell_h) * 0.38
                norm_  = mag[r, c] + 1e-6
                ex = int(cx_mid + (u_ / norm_) * length)
                ey = int(cy_mid + (v_ / norm_) * length)
                cv2.arrowedLine(panel, (cx_mid, cy_mid), (ex, ey),
                                (255, 255, 255), 1, tipLength=0.35)

            # Subtle cell border so the grid structure is always visible
            cv2.rectangle(panel, (cx1, cy1), (cx2 - 1, cy2 - 1), (45, 45, 45), 1)

    return panel


# ── 4. Static summary figure ──────────────────────────────────────────────────
# Pick: frame 0 (reference), frame at max deformation, and 3 evenly spaced frames
peak_frame = int(flow_mag.mean(axis=(1, 2)).argmax())
key_frames = sorted(set([
    0,
    N // 4,
    N // 2,
    3 * N // 4,
    peak_frame,
]))

print(f'\nKey frames for summary: {key_frames}  (peak at frame {peak_frame})')

fig, axes = plt.subplots(3, len(key_frames), figsize=(4 * len(key_frames), 10),
                         facecolor='#0F1117')
fig.suptitle('Skin Flow Analysis — Fake Wave Dataset', color='white', fontsize=14, y=0.98)

col_labels = [f'Frame {f}\n(t={f/30:.1f}s)' for f in key_frames]

for col, fidx in enumerate(key_frames):
    # Row 0: rendered overlay (heatmap + quiver)
    img = render_frame(fidx)
    axes[0, col].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, col].set_title(col_labels[col], color='white', fontsize=9)
    axes[0, col].axis('off')

    # Row 1: magnitude heatmap only (clean, no background)
    axes[1, col].imshow(flow_mag[fidx], cmap='plasma', vmin=0, vmax=MAG_MAX,
                        aspect='auto', interpolation='nearest')
    axes[1, col].set_title(f'max={flow_mag[fidx].max():.1f}px', color='white', fontsize=8)
    axes[1, col].axis('off')

    # Row 2: quiver only (u horizontal, v vertical)
    ax = axes[2, col]
    ax.set_facecolor('#0F1117')
    xs = np.arange(GRID_COLS) + 0.5
    ys = np.arange(GRID_ROWS) + 0.5
    XX, YY = np.meshgrid(xs, ys)
    U = flow_uv[fidx, :, :, 0]
    V = flow_uv[fidx, :, :, 1]
    mag_norm = flow_mag[fidx] / (MAG_MAX + 1e-6)
    ax.quiver(XX, YY, U, -V,   # flip V so +y points up in the plot
              mag_norm, cmap='plasma', scale=40, width=0.008, clim=[0, 1])
    ax.set_xlim(0, GRID_COLS); ax.set_ylim(0, GRID_ROWS)
    ax.invert_yaxis()
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')

# Row labels
for ax, label in zip(axes[:, 0], ['Overlay', 'Magnitude Heatmap', 'Flow Vectors']):
    ax.set_ylabel(label, color='white', fontsize=9, rotation=90, labelpad=6)

plt.tight_layout()
summary_path = os.path.join(OUT_DIR, 'flow_summary.png')
plt.savefig(summary_path, dpi=140, bbox_inches='tight', facecolor='#0F1117')
plt.close()
print(f'Saved: {summary_path}')


# ── 5. Animated video ─────────────────────────────────────────────────────────
# Layout (top → bottom):
#   [ heatmap overlay on pattern frame  ]   out_h px
#   [ 4px dark divider                  ]
#   [ label bar                         ]   24px
#   [ flow direction grid               ]   GRID_ROWS * cell_h px
#   [ timestamp bar                     ]   28px
print('\nRendering video...')

sample_overlay = render_frame(0)
sample_grid    = render_flow_grid(0)
out_h, out_w   = sample_overlay.shape[:2]
grid_h         = sample_grid.shape[0]

DIVIDER_H  = 4
LABEL_H    = 24
STAMP_H    = 28
VIDEO_H    = out_h + DIVIDER_H + LABEL_H + grid_h + STAMP_H

video_path = os.path.join(OUT_DIR, 'flow_video.mp4')
fourcc     = cv2.VideoWriter_fourcc(*'mp4v')
writer     = cv2.VideoWriter(video_path, fourcc, 30, (out_w, VIDEO_H))

mean_mag_series = flow_mag.mean(axis=(1, 2))

# Pre-build static label bar (same every frame)
label_bar = np.full((LABEL_H, out_w, 3), 30, dtype=np.uint8)
cv2.putText(label_bar,
            f'FLOW DIRECTION GRID  |  colour = direction  |  threshold = {FLOW_THRESHOLD:.2f}px',
            (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

divider = np.full((DIVIDER_H, out_w, 3), 15, dtype=np.uint8)

for i in range(N):
    overlay   = render_frame(i)
    flow_grid = render_flow_grid(i)

    # Pad grid to match video width if there is a small rounding difference
    if flow_grid.shape[1] < out_w:
        pad = np.full((grid_h, out_w - flow_grid.shape[1], 3), 18, dtype=np.uint8)
        flow_grid = np.hstack([flow_grid, pad])

    # Timestamp bar
    stamp_bar = np.full((STAMP_H, out_w, 3), 10, dtype=np.uint8)
    t_str = f't={i/30:.2f}s   mean disp={mean_mag_series[i]:.2f}px   threshold={FLOW_THRESHOLD:.2f}px'
    cv2.putText(stamp_bar, t_str, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)

    composed = np.vstack([overlay, divider, label_bar, flow_grid, stamp_bar])
    writer.write(composed)

    if (i + 1) % 100 == 0:
        print(f'  {i+1}/{N}', flush=True)

writer.release()
print(f'Saved: {video_path}')
print('Done.')
