"""
Model pipeline visualisation.

Shows how DenseContactNet processes a skin image end-to-end, sampling frames
from one session across six indentation depths (0 → 10 mm).

Each column = one depth level.  Each row = one stage of the model pipeline:

  Row 0 : Skin image (model input)
  Row 1 : Contact probability map     (sigmoid output)
  Row 2 : Depth profile map           (Hertz depth, ReLU output)
  Row 3 : Pressure distribution map   (Hertz pressure, ReLU output)
  Row 4 : Sensor grid                 (heatmap + contact circle + location crosshair)

Usage:
    python dense_contact/visualize_model.py
    python dense_contact/visualize_model.py --session x174_y8
    python dense_contact/visualize_model.py --session x174_y8 --save pipeline.png
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from PIL import Image
from torchvision import transforms

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from model import DenseContactNet
from dataset import GRID_X, GRID_Y

# ── Constants ──────────────────────────────────────────────────────────────────
with open(os.path.join(_HERE, 'model', 'model_stats.json')) as f:
    _stats = json.load(f)
DISP_MAX  = _stats['disp_max']
FORCE_MAX = _stats['force_max']

GRAY_MEAN  = 0.4513
GRAY_STD   = 0.2898
R_INDENTOR = 10.0   # mm

_GRID_X = np.asarray(GRID_X, dtype=np.float32)
_GRID_Y = np.asarray(GRID_Y, dtype=np.float32)
GX_MIN, GX_MAX = float(_GRID_X[0]), float(_GRID_X[-1])
GY_MIN, GY_MAX = float(_GRID_Y[0]), float(_GRID_Y[-1])

TARGET_DEPTHS = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]   # mm — columns
DEPTH_TOL     = 0.5                                  # ± mm window

_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([GRAY_MEAN], [GRAY_STD]),
])

# ── Device + model ────────────────────────────────────────────────────────────
def _get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def _load_model(device):
    m = DenseContactNet().to(device)
    m.load_state_dict(
        torch.load(os.path.join(_HERE, 'model', 'best_model.pth'), map_location=device)
    )
    m.eval()
    return m

# ── Grid drawing helper ───────────────────────────────────────────────────────
def _draw_sensor_grid(ax, contact_map, loc_x, loc_y, disp_mm, in_contact,
                      col_label=''):
    """Draw sensor grid onto a matplotlib axes."""
    ax.set_facecolor('#060612')
    ax.set_xlim(GX_MIN - 2, GX_MAX + 2)
    ax.set_ylim(GY_MAX + 1.5, GY_MIN - 1.5)   # y=0 at top

    # Contact heatmap
    ax.imshow(
        contact_map,
        extent=[GX_MIN, GX_MAX, GY_MAX, GY_MIN],
        cmap='hot', vmin=0, vmax=1,
        aspect='auto', origin='upper', alpha=0.8, zorder=1,
    )

    # Sparse reference dots
    xx, yy = np.meshgrid(_GRID_X[::3], _GRID_Y[::3])
    ax.scatter(xx.ravel(), yy.ravel(),
               c='#1a1a33', s=6, zorder=2, linewidths=0)

    if in_contact:
        a_mm = float(np.sqrt(max(R_INDENTOR * disp_mm, 0.0)))
        circ = mpatches.Circle(
            (loc_x, loc_y), a_mm,
            fill=False, edgecolor='#00ccff', linewidth=1.8, zorder=5,
        )
        ax.add_patch(circ)
        ax.plot(loc_x, loc_y, '+',
                color='#00ff88', markersize=14, markeredgewidth=2.2, zorder=6)

    ax.tick_params(left=False, bottom=False,
                   labelleft=False, labelbottom=False)
    for sp in ax.spines.values():
        sp.set_edgecolor('#2a2a44')


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--session', default='x174_y8',
                    help='Dataset session ID (default: x174_y8)')
    ap.add_argument('--save', default='model_pipeline.png',
                    help='Output file path')
    args = ap.parse_args()

    device = _get_device()
    print(f"Device: {device}")
    model = _load_model(device)
    print("Model loaded.")

    csv_path = os.path.join(_HERE, 'dataset.csv')
    df  = pd.read_csv(csv_path)
    sdf = df[df['session'] == args.session].reset_index(drop=True)
    if sdf.empty:
        sys.exit(f"Session '{args.session}' not found in dataset.csv")

    loc_x_gt = float(sdf['loc_x'].iloc[0])
    loc_y_gt = float(sdf['loc_y'].iloc[0])
    print(f"Session {args.session}: GT location ({loc_x_gt}, {loc_y_gt}) mm")

    # ── Collect one frame per target depth ────────────────────────────────────
    frames = []
    for td in TARGET_DEPTHS:
        diffs = (sdf['displacement_mm'] - td).abs()
        if diffs.min() > DEPTH_TOL:
            print(f"  No frame within {DEPTH_TOL} mm of {td} mm — skipping")
            continue
        row = sdf.loc[diffs.idxmin()]

        pil    = Image.open(row['image_path']).convert('L')
        tensor = _TRANSFORM(pil).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(tensor)

        frames.append({
            'depth_target': td,
            'depth_actual': float(row['displacement_mm']),
            'image':        np.array(pil.resize((224, 224), Image.BILINEAR), dtype=np.uint8),
            'contact_map':  out['contact_map'][0].cpu().numpy(),
            'depth_map':    out['depth_map'][0].cpu().numpy(),
            'pressure_map': out['pressure_map'][0].cpu().numpy(),
            'loc_x':        out['loc_x'][0].item(),
            'loc_y':        out['loc_y'][0].item(),
            'disp_mm':      out['displacement'][0].item() * DISP_MAX,
            'force_n':      out['force'][0].item() * FORCE_MAX,
        })
        print(f"  {td:.0f} mm → pred ({out['loc_x'][0].item():.1f}, "
              f"{out['loc_y'][0].item():.1f}) mm  "
              f"disp={out['displacement'][0].item()*DISP_MAX:.2f} mm  "
              f"force={out['force'][0].item()*FORCE_MAX:.4f} N")

    n_cols = len(frames)
    n_rows = 5   # skin | contact | depth | pressure | grid

    # ── Row labels ────────────────────────────────────────────────────────────
    row_labels = [
        'Input\n(skin image)',
        'Contact\nprobability',
        'Depth\nprofile',
        'Pressure\ndistribution',
        'Sensor grid\n+ prediction',
    ]
    row_cmaps  = [None, 'hot', 'plasma', 'inferno', None]

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(2.8 * n_cols + 1.6, 3.2 * n_rows + 1.4),
                     facecolor='#0a0a14')

    fig.suptitle(
        f'DenseContactNet — Model Pipeline\n'
        f'Session {args.session}  ·  GT location ({loc_x_gt:.0f}, {loc_y_gt:.1f}) mm',
        color='white', fontsize=13, y=0.995,
    )

    gs = GridSpec(
        n_rows, n_cols + 1,           # extra col for row labels
        figure=fig,
        left=0.08, right=0.97,
        top=0.96, bottom=0.04,
        hspace=0.12, wspace=0.08,
        width_ratios=[0.18] + [1] * n_cols,
    )

    # ── Column headers (depth labels) ─────────────────────────────────────────
    for ci, fr in enumerate(frames):
        ax_hdr = fig.add_subplot(gs[0, ci + 1])
        ax_hdr.set_facecolor('#0a0a14')
        ax_hdr.axis('off')
        ax_hdr.set_title(
            f'{fr["depth_target"]:.0f} mm target\n'
            f'({fr["depth_actual"]:.2f} mm actual)',
            color='#ccccdd', fontsize=9, pad=3,
        )

    # ── Row label column ──────────────────────────────────────────────────────
    for ri, label in enumerate(row_labels):
        ax_lbl = fig.add_subplot(gs[ri, 0])
        ax_lbl.set_facecolor('#0a0a14')
        ax_lbl.axis('off')
        ax_lbl.text(0.9, 0.5, label,
                    transform=ax_lbl.transAxes,
                    color='#aaaacc', fontsize=8.5, ha='right', va='center',
                    linespacing=1.5)

    # ── Cell axes ─────────────────────────────────────────────────────────────
    # We'll store one axes per (row, col) to add colorbars later
    for ci, fr in enumerate(frames):
        in_contact = fr['contact_map'].max() > 0.05

        # Row 0 — skin image (column header already drawn, reuse space below it)
        ax0 = fig.add_subplot(gs[0, ci + 1])
        ax0.imshow(fr['image'], cmap='gray', origin='upper', aspect='auto')
        ax0.axis('off')
        # Overlay: predicted location as a small dot
        if in_contact:
            # Convert mm → pixel coords in the 224×224 image is not straightforward
            # (image is the raw skin, not the grid) — just skip overlay here
            pass

        # Row 1 — contact map
        ax1 = fig.add_subplot(gs[1, ci + 1])
        ax1.imshow(fr['contact_map'], cmap='hot', vmin=0, vmax=1,
                   origin='upper', aspect='auto')
        ax1.axis('off')
        if ci == 0:
            ax1.set_ylabel('Contact prob.', color='#888899', fontsize=7)

        # Row 2 — depth map
        dmap = fr['depth_map']
        ax2 = fig.add_subplot(gs[2, ci + 1])
        ax2.imshow(dmap, cmap='plasma', vmin=0, vmax=dmap.max() + 1e-6,
                   origin='upper', aspect='auto')
        ax2.axis('off')

        # Row 3 — pressure map
        pmap = fr['pressure_map']
        ax3 = fig.add_subplot(gs[3, ci + 1])
        ax3.imshow(pmap, cmap='inferno', vmin=0, vmax=pmap.max() + 1e-6,
                   origin='upper', aspect='auto')
        ax3.axis('off')

        # Row 4 — sensor grid
        ax4 = fig.add_subplot(gs[4, ci + 1])
        _draw_sensor_grid(ax4, fr['contact_map'],
                          fr['loc_x'], fr['loc_y'], fr['disp_mm'], in_contact)
        # Annotation: predicted scalars
        a_mm = float(np.sqrt(max(R_INDENTOR * fr['disp_mm'], 0.0)))
        label_txt = (
            f"({fr['loc_x']:.1f}, {fr['loc_y']:.1f}) mm\n"
            f"δ={fr['disp_mm']:.2f} mm  F={fr['force_n']:.3f} N\n"
            f"r={a_mm:.1f} mm"
        )
        ax4.text(0.5, -0.02, label_txt,
                 transform=ax4.transAxes,
                 color='#99aacc', fontsize=6.5, ha='center', va='top',
                 linespacing=1.4)

    # ── Shared colorbars (one per map type, placed at right of last column) ───
    # We use fig.add_axes for compact bars
    def _add_cbar(cmap, vmin, vmax, label, ypos):
        cax = fig.add_axes([0.975, ypos, 0.008, 0.14])
        sm  = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin, vmax))
        sm.set_array([])
        cb  = fig.colorbar(sm, cax=cax)
        cb.set_label(label, color='#888899', fontsize=6.5, labelpad=3)
        cb.ax.yaxis.set_tick_params(color='#666677', labelsize=5.5)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color='#888899')

    _add_cbar('hot',     0, 1,   'Contact\nprob.',  0.665)
    _add_cbar('plasma',  0, 1,   'Norm.\ndepth',    0.475)
    _add_cbar('inferno', 0, 1,   'Norm.\npressure', 0.285)

    # ── Arrow annotation on the figure showing the pipeline flow ─────────────
    # Left-side vertical arrow
    fig.text(0.025, 0.5, '← model outputs',
             color='#555577', fontsize=7, rotation=90, va='center', ha='center')

    plt.savefig(args.save, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"\nSaved → {args.save}")
    plt.close(fig)


if __name__ == '__main__':
    main()
