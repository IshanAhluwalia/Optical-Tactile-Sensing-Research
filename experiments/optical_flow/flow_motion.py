"""
Flow velocity, direction, and propagation analysis.

Three things computed from flow_data.npz:

  1. VELOCITY  — frame-to-frame change in displacement (how fast each skin
                 region is actively moving at each moment, not just how far
                 it has drifted from rest)

  2. DIRECTION — angle of the displacement vector at each cell, shown as a
                 colour wheel (hue = direction, brightness = speed)

  3. PROPAGATION — when each region first starts moving, used to estimate
                   how fast the deformation wave travels across the skin

Outputs:
  flow_motion_summary.png  — static 3-row figure
  flow_motion_video.mp4    — animated direction + velocity side by side
"""

import cv2
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

FRAMES_DIR = os.path.join(os.path.dirname(__file__), 'frames')
OUT_DIR    = os.path.dirname(__file__)
FPS        = 30.0

# ── Load flow data ────────────────────────────────────────────────────────────
data     = np.load(os.path.join(OUT_DIR, 'flow_data.npz'))
flow_uv  = data['flow_uv']    # (N, ROWS, COLS, 2)
flow_mag = data['flow_mag']   # (N, ROWS, COLS)
x1, y1, x2, y2 = data['roi']
GRID_COLS, GRID_ROWS = data['grid']

N      = flow_uv.shape[0]
roi_w  = x2 - x1
roi_h  = y2 - y1
cell_w = roi_w // GRID_COLS
cell_h = roi_h // GRID_ROWS
times  = np.arange(N) / FPS   # time axis in seconds

MAG_MAX = float(np.percentile(flow_mag, 99))
print(f'Loaded {N} frames  |  Grid {GRID_COLS}×{GRID_ROWS}  |  MAG_MAX={MAG_MAX:.2f}px')

# ═══════════════════════════════════════════════════════════════════════════════
# 1. VELOCITY  (frame-to-frame change in displacement)
# ─────────────────────────────────────────────────────────────────────────────
# flow_uv[t] tells us how far each cell has moved FROM REST by frame t.
# The derivative of that (difference between consecutive frames) tells us how
# fast the skin is currently moving at frame t.
#
#   vel_uv[t]  = flow_uv[t+1] - flow_uv[t]   shape: (N-1, ROWS, COLS, 2)
#   vel_mag[t] = ||vel_uv[t]||                shape: (N-1, ROWS, COLS)
# ─────────────────────────────────────────────────────────────────────────────
vel_uv  = np.diff(flow_uv, axis=0)                        # (N-1, ROWS, COLS, 2)
vel_mag = np.linalg.norm(vel_uv, axis=-1)                 # (N-1, ROWS, COLS)

# Pixels per second (multiply by FPS since each step = 1/FPS seconds)
vel_pps = vel_mag * FPS                                   # pixels / second

VEL_MAX = float(np.percentile(vel_pps, 99))
print(f'Max velocity: {VEL_MAX:.1f} px/s  (99th percentile)')

# Mean velocity across all cells — one number per frame (good for time series)
mean_vel = vel_pps.mean(axis=(1, 2))                      # (N-1,)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. DIRECTION  (angle of the displacement vector)
# ─────────────────────────────────────────────────────────────────────────────
# atan2(v, u) gives the angle in radians [-π, π].
# We map this to a hue [0, 1] for the HSV colour wheel so every direction gets
# a unique colour.  Brightness encodes speed — fast motion = bright, still = dark.
# ─────────────────────────────────────────────────────────────────────────────
u = flow_uv[..., 0]   # (N, ROWS, COLS)
v = flow_uv[..., 1]

angle_rad = np.arctan2(v, u)                              # [-π, π]
angle_norm = (angle_rad + np.pi) / (2 * np.pi)           # [0, 1]  hue

# ═══════════════════════════════════════════════════════════════════════════════
# 3. PROPAGATION  (when each cell first starts moving)
# ─────────────────────────────────────────────────────────────────────────────
# For each grid cell we find the first frame where displacement magnitude
# exceeds a threshold.  Plotting onset time as a heatmap reveals the direction
# the wave travels: early-onset cells are upstream, late-onset cells downstream.
#
# From the onset map we also estimate wave speed:
#   speed [px/s] = distance between two cells [px] / time difference [s]
# ─────────────────────────────────────────────────────────────────────────────
ONSET_THRESH = MAG_MAX * 0.15          # 15% of peak = "meaningfully moving"

onset_frame = np.full((GRID_ROWS, GRID_COLS), np.nan)
for r in range(GRID_ROWS):
    for c in range(GRID_COLS):
        hits = np.where(flow_mag[:, r, c] > ONSET_THRESH)[0]
        if len(hits) > 0:
            onset_frame[r, c] = hits[0]

onset_time = onset_frame / FPS        # seconds

# Estimate wave propagation speed across columns (horizontal direction)
# For each column, take the median onset time across all rows.
# A linear fit of median_onset vs column index gives px/frame → convert to px/s.
col_onset = np.nanmedian(onset_time, axis=0)   # (COLS,) one time per column
valid     = ~np.isnan(col_onset)

if valid.sum() >= 2:
    col_positions = np.arange(GRID_COLS)[valid] * cell_w   # pixel x of each col centre
    col_times     = col_onset[valid]
    # Linear fit: position = speed × time + offset
    coeffs   = np.polyfit(col_times, col_positions, 1)
    wave_speed_pps = abs(coeffs[0])   # px/s
    print(f'Estimated horizontal wave speed: {wave_speed_pps:.1f} px/s')
else:
    wave_speed_pps = None
    print('Not enough onset data for wave speed estimate.')

# ═══════════════════════════════════════════════════════════════════════════════
# STATIC SUMMARY FIGURE
# ═══════════════════════════════════════════════════════════════════════════════
peak_frame = int(mean_vel.argmax())
key_frames = sorted(set([0, N//4, N//2, peak_frame, min(peak_frame+30, N-2)]))
key_frames = [f for f in key_frames if f < N-1]   # velocity has N-1 frames

fig = plt.figure(figsize=(4*len(key_frames), 14), facecolor='#0F1117')
fig.suptitle('Flow Velocity · Direction · Propagation', color='white', fontsize=14, y=0.99)

gs = fig.add_gridspec(4, len(key_frames), hspace=0.4, wspace=0.15)

for col, fidx in enumerate(key_frames):
    t_label = f't={fidx/FPS:.1f}s'

    # ── Row 0: Velocity magnitude heatmap ─────────────────────────────────────
    ax = fig.add_subplot(gs[0, col])
    ax.imshow(vel_pps[fidx], cmap='inferno', vmin=0, vmax=VEL_MAX,
              aspect='auto', interpolation='nearest')
    ax.set_title(f'{t_label}\nmax={vel_pps[fidx].max():.1f}px/s',
                 color='white', fontsize=8)
    ax.axis('off')

    # ── Row 1: Direction (HSV colour wheel) ───────────────────────────────────
    ax = fig.add_subplot(gs[1, col])
    hue = angle_norm[fidx]                                  # (ROWS, COLS)
    sat = np.ones_like(hue)
    val = np.clip(flow_mag[fidx] / MAG_MAX, 0, 1)          # brightness = speed
    hsv_img = np.stack([hue, sat, val], axis=-1)            # (ROWS, COLS, 3)
    rgb_img = mcolors.hsv_to_rgb(hsv_img)
    ax.imshow(rgb_img, aspect='auto', interpolation='nearest')
    ax.set_title(f'{t_label}\ndirection (colour) + speed (brightness)',
                 color='white', fontsize=7)
    ax.axis('off')

    # ── Row 2: Quiver arrows coloured by direction ────────────────────────────
    ax = fig.add_subplot(gs[2, col])
    ax.set_facecolor('#0F1117')
    xs = np.arange(GRID_COLS) + 0.5
    ys = np.arange(GRID_ROWS) + 0.5
    XX, YY = np.meshgrid(xs, ys)
    U_f = flow_uv[fidx, :, :, 0]
    V_f = flow_uv[fidx, :, :, 1]
    colors_flat = rgb_img.reshape(-1, 3)
    ax.quiver(XX, YY, U_f, -V_f, color=colors_flat,
              scale=60, width=0.012)
    ax.set_xlim(0, GRID_COLS); ax.set_ylim(0, GRID_ROWS)
    ax.invert_yaxis()
    ax.set_title(f'{t_label}\nflow vectors', color='white', fontsize=8)
    for spine in ax.spines.values(): spine.set_edgecolor('#444')
    ax.tick_params(colors='white', labelsize=6)

# ── Row 3: Propagation onset map (same for all columns) ───────────────────────
ax_prop = fig.add_subplot(gs[3, :])
im = ax_prop.imshow(onset_time, cmap='viridis', aspect='auto',
                    interpolation='nearest')
ax_prop.set_title(
    f'Wave Propagation — onset time per cell (seconds)\n'
    f'Estimated horizontal wave speed: '
    f'{wave_speed_pps:.1f} px/s' if wave_speed_pps else 'Wave Propagation',
    color='white', fontsize=9
)
ax_prop.set_xlabel('Grid column (left → right)', color='white', fontsize=8)
ax_prop.set_ylabel('Grid row', color='white', fontsize=8)
ax_prop.tick_params(colors='white')
cb = plt.colorbar(im, ax=ax_prop, orientation='horizontal', pad=0.2, fraction=0.03)
cb.set_label('Onset time (s)', color='white', fontsize=8)
cb.ax.yaxis.set_tick_params(color='white')
plt.setp(cb.ax.xaxis.get_ticklabels(), color='white')
for spine in ax_prop.spines.values(): spine.set_edgecolor('#444')

# Time series inset on propagation row
ax_ts = fig.add_subplot(gs[3, -1])
ax_ts.set_facecolor('#0F1117')
ax_ts.plot(times[:-1], mean_vel, color='#e06c75', linewidth=1.2)
ax_ts.axvline(peak_frame/FPS, color='yellow', linewidth=0.8, linestyle='--',
              label=f'peak t={peak_frame/FPS:.1f}s')
ax_ts.set_xlabel('Time (s)', color='white', fontsize=7)
ax_ts.set_ylabel('Mean velocity (px/s)', color='white', fontsize=7)
ax_ts.set_title('Mean skin velocity over time', color='white', fontsize=8)
ax_ts.tick_params(colors='white', labelsize=6)
ax_ts.legend(fontsize=6, labelcolor='white', facecolor='#0F1117')
for spine in ax_ts.spines.values(): spine.set_edgecolor('#444')

summary_path = os.path.join(OUT_DIR, 'flow_motion_summary.png')
plt.savefig(summary_path, dpi=140, bbox_inches='tight', facecolor='#0F1117')
plt.close()
print(f'Saved: {summary_path}')

# ═══════════════════════════════════════════════════════════════════════════════
# ANIMATED VIDEO  — direction (left) | velocity (right)
# ═══════════════════════════════════════════════════════════════════════════════
print('\nRendering video...')

CELL_PX = 60    # each grid cell rendered as CELL_PX × CELL_PX pixels in the video
PANEL_W = GRID_COLS * CELL_PX
PANEL_H = GRID_ROWS * CELL_PX
DIVIDER = 6
VIDEO_W = PANEL_W * 2 + DIVIDER
VIDEO_H = PANEL_H + 40     # +40 for text bar at bottom

video_path = os.path.join(OUT_DIR, 'flow_motion_video.mp4')
fourcc     = cv2.VideoWriter_fourcc(*'mp4v')
writer     = cv2.VideoWriter(video_path, fourcc, FPS, (VIDEO_W, VIDEO_H))

# Inferno LUT for velocity
lut_inferno = np.array([plt.cm.inferno(i / 255.0)[:3] for i in range(256)],
                        dtype=np.float32)  # (256, 3) RGB

def mag_to_bgr(mag_grid, vmax):
    """Map a (ROWS, COLS) magnitude array to a (PANEL_H, PANEL_W, 3) BGR image."""
    idx   = np.clip((mag_grid / (vmax + 1e-6)) * 255, 0, 255).astype(np.uint8)
    panel = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            rgb   = lut_inferno[idx[r, c]]
            color = (int(rgb[2]*255), int(rgb[1]*255), int(rgb[0]*255))  # BGR
            y0, y1_ = r*CELL_PX, (r+1)*CELL_PX
            x0, x1_ = c*CELL_PX, (c+1)*CELL_PX
            panel[y0:y1_, x0:x1_] = color
            # Arrow
            cx, cy = x0 + CELL_PX//2, y0 + CELL_PX//2
    return panel

def dir_to_bgr(angle_grid, mag_grid):
    """Map direction + magnitude to HSV-coloured BGR panel."""
    hue = ((angle_grid + np.pi) / (2 * np.pi)).astype(np.float32)
    val = np.clip(mag_grid / (MAG_MAX + 1e-6), 0, 1).astype(np.float32)
    panel = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            h, v = float(hue[r, c]), float(val[r, c])
            rgb  = mcolors.hsv_to_rgb([h, 1.0, v])
            color = (int(rgb[2]*255), int(rgb[1]*255), int(rgb[0]*255))
            panel[r*CELL_PX:(r+1)*CELL_PX, c*CELL_PX:(c+1)*CELL_PX] = color
    # Draw arrows on top
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if mag_grid[r, c] > MAG_MAX * 0.1:
                cx = c*CELL_PX + CELL_PX//2
                cy = r*CELL_PX + CELL_PX//2
                u_  = float(flow_uv[0, r, c, 0])   # placeholder, overridden below
                ex  = cx + int(u_ * 2)
                ey  = cy
                cv2.arrowedLine(panel, (cx, cy), (ex, ey), (255,255,255), 1, tipLength=0.3)
    return panel

for i in range(N - 1):
    angle_i = np.arctan2(flow_uv[i, :, :, 1], flow_uv[i, :, :, 0])

    # Direction panel (HSV)
    hue = ((angle_i + np.pi) / (2 * np.pi)).astype(np.float32)
    val = np.clip(flow_mag[i] / (MAG_MAX + 1e-6), 0, 1).astype(np.float32)
    dir_panel = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            rgb   = mcolors.hsv_to_rgb([float(hue[r,c]), 1.0, float(val[r,c])])
            color = (int(rgb[2]*255), int(rgb[1]*255), int(rgb[0]*255))
            dir_panel[r*CELL_PX:(r+1)*CELL_PX, c*CELL_PX:(c+1)*CELL_PX] = color
            # Arrow showing direction
            if flow_mag[i, r, c] > MAG_MAX * 0.1:
                cx = c*CELL_PX + CELL_PX//2
                cy = r*CELL_PX + CELL_PX//2
                u_ = float(flow_uv[i, r, c, 0])
                v_ = float(flow_uv[i, r, c, 1])
                scale = CELL_PX * 0.35 / (flow_mag[i, r, c] + 1e-6)
                ex = int(cx + u_ * scale)
                ey = int(cy + v_ * scale)
                cv2.arrowedLine(dir_panel, (cx, cy), (ex, ey),
                                (255, 255, 255), 1, tipLength=0.3)

    # Velocity panel (inferno heatmap)
    vel_i = vel_pps[i]
    idx   = np.clip((vel_i / (VEL_MAX + 1e-6)) * 255, 0, 255).astype(np.uint8)
    vel_panel = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            rgb   = lut_inferno[idx[r, c]]
            color = (int(rgb[2]*255), int(rgb[1]*255), int(rgb[0]*255))
            vel_panel[r*CELL_PX:(r+1)*CELL_PX, c*CELL_PX:(c+1)*CELL_PX] = color

    # Compose side by side
    divider_col = np.full((PANEL_H, DIVIDER, 3), 40, dtype=np.uint8)
    top = np.hstack([dir_panel, divider_col, vel_panel])

    # Text bar
    bar = np.zeros((40, VIDEO_W, 3), dtype=np.uint8)
    cv2.putText(bar, 'DIRECTION  (colour=angle  brightness=displacement)',
                (10, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200,200,200), 1)
    cv2.putText(bar, 'VELOCITY  (brightness=speed px/s)',
                (PANEL_W + DIVIDER + 10, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200,200,200), 1)
    cv2.putText(bar,
                f't={i/FPS:.2f}s   vel={mean_vel[i]:.1f}px/s',
                (VIDEO_W//2 - 100, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,220,80), 1)

    frame_out = np.vstack([top, bar])
    writer.write(frame_out)

    if (i + 1) % 100 == 0:
        print(f'  {i+1}/{N-1}', flush=True)

writer.release()
print(f'Saved: {video_path}')
print('Done.')
