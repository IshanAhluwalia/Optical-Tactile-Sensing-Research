"""
Two rows of images:
  Top row:    real frames from x174_y8 at 5 depths (2mm → 10mm)
  Bottom row: synthetically generated same frames with R=5mm (half radius)
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE   = os.path.dirname(os.path.abspath(__file__))
VIZ_DIR = os.path.join(_HERE, 'synthetic_viz')
os.makedirs(VIZ_DIR, exist_ok=True)

SESSION = 'x174_y8'
LOC_X   = 174.0
LOC_Y   = 8.0
R_REAL  = 10.0
R_HALF  = 5.0

IMG_W, IMG_H = 538, 163
X_MIN, X_RANGE = 138.0, 72.0
Y_MIN, Y_RANGE = 0.0, 16.0

# pixel distance from contact centre (in mm)
PX, PY = np.meshgrid(np.arange(IMG_W, dtype=np.float32),
                     np.arange(IMG_H, dtype=np.float32))
cx_px = (LOC_X - X_MIN) / X_RANGE * (IMG_W - 1)
cy_px = (LOC_Y - Y_MIN) / Y_RANGE * (IMG_H - 1)
DX    = (PX - cx_px) / (IMG_W - 1) * X_RANGE
DY    = (PY - cy_px) / (IMG_H - 1) * Y_RANGE
R2    = DX**2 + DY**2

# compliance (edges stiffer)
MIN_C = 0.3
COMP  = MIN_C + (1 - MIN_C) * (
    np.sin(np.pi * np.arange(IMG_W) / (IMG_W - 1))[np.newaxis, :] *
    np.sin(np.pi * np.arange(IMG_H) / (IMG_H - 1))[:, np.newaxis]
)

from scipy.ndimage import map_coordinates

def half_radius(rest, frame, delta):
    """
    Spatially compress the deformation field to simulate a smaller indentor.

    For Hertz contact, the surface deformation profile is:
        u(r) = delta * (1 - r^2/a^2)  for r < a

    A point at radius r in the R=5mm contact has the same deformation value
    as a point at r*(a1/a2) in the R=10mm contact.  So we warp by sampling
    the real deformation field at scaled-up coordinates, then cut off outside a2.

    Result: same bright centre, but the deformation falls to zero at a2 instead
    of a1 — the contact footprint is physically smaller.
    """
    if delta <= 0:
        return frame.copy()

    a1    = np.sqrt(R_REAL * delta)
    a2    = np.sqrt(R_HALF * delta)
    scale = a1 / a2          # > 1: we sample further out to compress inward

    D = frame.astype(np.float32) - rest.astype(np.float32)

    # contact centre in pixel coords
    cx_px = (LOC_X - X_MIN) / X_RANGE * (IMG_W - 1)
    cy_px = (LOC_Y - Y_MIN) / Y_RANGE * (IMG_H - 1)

    PX_g, PY_g = np.meshgrid(np.arange(IMG_W, dtype=np.float32),
                              np.arange(IMG_H, dtype=np.float32))

    # For each output pixel, sample the real D field at scale * offset from centre
    src_x = cx_px + (PX_g - cx_px) * scale
    src_y = cy_px + (PY_g - cy_px) * scale

    D_warped = map_coordinates(D, [src_y.ravel(), src_x.ravel()],
                               order=1, mode='constant', cval=0.0
                               ).reshape(IMG_H, IMG_W)

    # Hard cut-off outside smaller contact zone
    r_mm = np.sqrt(R2)
    mask = 1.0 / (1.0 + np.exp((r_mm - a2) / 0.5))

    return np.clip(rest.astype(np.float32) + D_warped * mask * COMP,
                   0, 255).astype(np.uint8)

# load session
df  = pd.read_csv(os.path.join(_HERE, 'dataset.csv'))
sdf = df[df['session'] == SESSION].sort_values('displacement_mm').reset_index(drop=True)

rest = np.array(Image.open(sdf.iloc[0]['image_path']).convert('L'))

# pick 5 depths
depths = [2.0, 4.0, 6.0, 8.0, 10.0]
frames = []
for t in depths:
    row   = sdf.iloc[(sdf['displacement_mm'] - t).abs().argsort().iloc[0]]
    real  = np.array(Image.open(row['image_path']).convert('L'))
    synth = half_radius(rest, real, float(row['displacement_mm']))
    frames.append((float(row['displacement_mm']), real, synth))

# ── plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(4, len(frames), figsize=(len(frames) * 4, 10))

for col, (delta, real, synth) in enumerate(frames):
    a1 = np.sqrt(R_REAL * delta)
    a2 = np.sqrt(R_HALF * delta)

    # deformation signal: amplify 4x so it's visible
    amp = 4
    d_real  = np.clip(128 + (real.astype(np.float32)  - rest.astype(np.float32)) * amp, 0, 255).astype(np.uint8)
    d_synth = np.clip(128 + (synth.astype(np.float32) - rest.astype(np.float32)) * amp, 0, 255).astype(np.uint8)

    def _show(ax, img, border, cmap='gray'):
        ax.imshow(img, cmap=cmap, vmin=0, vmax=255, aspect='auto')
        for sp in ax.spines.values():
            sp.set_edgecolor(border); sp.set_linewidth(4)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    axes[0, col].set_title(f'δ={delta:.1f}mm\na={a1:.1f}mm', fontsize=9)
    _show(axes[0, col], real,    '#2196F3')
    _show(axes[1, col], synth,   '#F44336')
    _show(axes[2, col], d_real,  '#2196F3', cmap='RdBu_r')
    _show(axes[3, col], d_synth, '#F44336', cmap='RdBu_r')

axes[0, 0].set_ylabel('REAL image\n(R=10mm)',          fontsize=10, fontweight='bold', color='#2196F3')
axes[1, 0].set_ylabel('SYNTHETIC image\n(R=5mm)',      fontsize=10, fontweight='bold', color='#F44336')
axes[2, 0].set_ylabel('REAL deformation\n(×4 amplified)', fontsize=10, fontweight='bold', color='#2196F3')
axes[3, 0].set_ylabel('SYNTH deformation\n(×4 amplified)', fontsize=10, fontweight='bold', color='#F44336')

plt.suptitle(f'Session {SESSION}  —  Real (R=10mm) vs Synthetic half-radius (R=5mm)',
             fontsize=12, fontweight='bold')
plt.tight_layout()

out = os.path.join(VIZ_DIR, 'half_radius_comparison.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"open {out}")
