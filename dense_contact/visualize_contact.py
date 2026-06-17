"""
Contact visualisation: skin image + predicted contact location and area.

Side-by-side layout
-------------------
  Left   : skin indentation image (raw) + preprocessed model input
  Right  : sensor grid with contact probability heatmap, a circle showing
           the predicted contact area (radius = sqrt(R_indentor * disp_mm)),
           and a crosshair at the predicted contact location.

Usage
-----
    python dense_contact/visualize_contact.py                          # random frame at ~6 mm
    python dense_contact/visualize_contact.py --session x174_y8 --depth 6
    python dense_contact/visualize_contact.py --image /path/to/frame.png
    python dense_contact/visualize_contact.py --save contact_vis.png
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
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

GRAY_MEAN = 0.4513
GRAY_STD  = 0.2898
R_INDENTOR = 10.0   # mm — spherical indentor radius for Hertz contact area

_GRID_X = np.asarray(GRID_X, dtype=np.float32)
_GRID_Y = np.asarray(GRID_Y, dtype=np.float32)
GX_MIN, GX_MAX = float(_GRID_X[0]), float(_GRID_X[-1])   # 138, 210
GY_MIN, GY_MAX = float(_GRID_Y[0]), float(_GRID_Y[-1])   # 0, 16

# Transform that matches training (no extra homomorphic — applied at record time)
_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([GRAY_MEAN], [GRAY_STD]),
])

# ── Device + model ────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

_model = None
def _get_model():
    global _model
    if _model is None:
        _model = DenseContactNet().to(DEVICE)
        _model.load_state_dict(
            torch.load(os.path.join(_HERE, 'model', 'best_model.pth'),
                       map_location=DEVICE)
        )
        _model.eval()
        print(f"Model loaded on {DEVICE}")
    return _model


@torch.no_grad()
def _infer(tensor: torch.Tensor) -> dict:
    out = _get_model()(tensor.to(DEVICE))
    return {k: v.cpu() for k, v in out.items()}


# ── Matplotlib figure ─────────────────────────────────────────────────────────
def _make_figure(skin_img_gray, contact_map, loc_x, loc_y,
                 disp_mm, force_n, gt=None, save_path=None):
    """
    skin_img_gray : (H, W) uint8  — raw sensor image from disk
    contact_map   : (H, W) float  — contact probability [0, 1]
    gt            : dict with 'loc_x', 'loc_y', 'disp_mm' ground-truth values (optional)
    save_path     : write PNG to this path instead of displaying interactively
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import Normalize

    in_contact = float(contact_map.max()) > 0.05
    a_mm = float(np.sqrt(max(R_INDENTOR * disp_mm, 0.0)))

    fig = plt.figure(figsize=(17, 7), facecolor='#0d0d1a')
    fig.suptitle('DenseContactNet — Skin Image + Predicted Contact',
                 color='white', fontsize=13, y=0.98)

    # 2-row, 3-col grid: [skin img | readouts | sensor grid (spans both rows)]
    gs = fig.add_gridspec(
        2, 3,
        left=0.04, right=0.97, top=0.91, bottom=0.06,
        hspace=0.45, wspace=0.32,
        width_ratios=[1, 1, 2.6],
    )

    # ── Panel A: skin image ──────────────────────────────────────────────────
    ax_skin = fig.add_subplot(gs[0, 0])
    ax_skin.imshow(skin_img_gray, cmap='gray', origin='upper', aspect='auto')
    ax_skin.set_title('Skin image (raw)', color='white', fontsize=9)
    ax_skin.axis('off')

    # ── Panel B: preprocessed 224×224 (what the model sees) ─────────────────
    pil = Image.fromarray(skin_img_gray).convert('L')
    proc_224 = np.array(pil.resize((224, 224), Image.BILINEAR), dtype=np.uint8)

    ax_proc = fig.add_subplot(gs[1, 0])
    ax_proc.imshow(proc_224, cmap='gray', origin='upper', aspect='auto')
    ax_proc.set_title('Resized model input (224×224)', color='white', fontsize=9)
    ax_proc.axis('off')

    # ── Panel C: scalar readouts ─────────────────────────────────────────────
    ax_info = fig.add_subplot(gs[:, 1])
    ax_info.set_facecolor('#0d0d1a')
    ax_info.axis('off')

    fields = [
        ('Predicted X',     f'{loc_x:.2f} mm',          '#00ff88'),
        ('Predicted Y',     f'{loc_y:.2f} mm',          '#00ff88'),
        ('Displacement',    f'{disp_mm:.3f} mm',         '#88aaff'),
        ('Force',           f'{force_n:.4f} N',          '#ffaa44'),
        ('Contact radius',  f'{a_mm:.2f} mm',            '#00ccff'),
        ('Confidence',      f'{contact_map.max():.3f}',  '#aaaaaa'),
    ]
    if gt:
        fields += [
            ('GT X',   f"{gt['loc_x']:.1f} mm",   '#ff6666'),
            ('GT Y',   f"{gt['loc_y']:.1f} mm",   '#ff6666'),
            ('GT disp',f"{gt['disp_mm']:.3f} mm", '#ff9966'),
        ]

    for i, (label, value, color) in enumerate(fields):
        y = 0.95 - i * 0.093
        ax_info.text(0.04, y, label + ':',
                     transform=ax_info.transAxes, fontsize=9.5,
                     color='#888888', ha='left', va='center')
        ax_info.text(0.55, y, value,
                     transform=ax_info.transAxes, fontsize=9.5,
                     color=color, ha='left', va='center', fontweight='bold')

    # ── Panel D: sensor grid ─────────────────────────────────────────────────
    ax_grid = fig.add_subplot(gs[:, 2])
    ax_grid.set_facecolor('#0a0a14')
    # y-axis: 0 mm at top, 16 mm at bottom (matching physical sensor orientation)
    ax_grid.set_xlim(GX_MIN - 3, GX_MAX + 3)
    ax_grid.set_ylim(GY_MAX + 2, GY_MIN - 2)
    ax_grid.set_xlabel('X — sensor length (mm)', color='#aaaaaa', fontsize=9)
    ax_grid.set_ylabel('Y — sensor width (mm)',  color='#aaaaaa', fontsize=9)
    ax_grid.set_title('Sensor Grid — Contact Prediction', color='white', fontsize=10, pad=6)
    ax_grid.tick_params(colors='#777777', labelsize=8)
    for sp in ax_grid.spines.values():
        sp.set_edgecolor('#333355')

    # Contact probability heatmap as image background
    ax_grid.imshow(
        contact_map,
        extent=[GX_MIN, GX_MAX, GY_MAX, GY_MIN],
        cmap='hot', vmin=0, vmax=1,
        aspect='auto', origin='upper',
        alpha=0.75, zorder=1,
    )

    # Reference dots at sensor grid positions
    xx, yy = np.meshgrid(_GRID_X, _GRID_Y)
    ax_grid.scatter(xx.ravel(), yy.ravel(),
                    c='#2a2a44', s=10, zorder=2, linewidths=0)

    if in_contact:
        # Contact area circle
        circle = mpatches.Circle(
            (loc_x, loc_y), a_mm,
            fill=False, edgecolor='#00ccff', linewidth=2.5,
            zorder=5, label=f'Contact area  r = {a_mm:.1f} mm',
        )
        ax_grid.add_patch(circle)

        # Predicted location crosshair
        ax_grid.plot(
            loc_x, loc_y, '+',
            color='#00ff88', markersize=22, markeredgewidth=2.8,
            zorder=6, label=f'Predicted  ({loc_x:.1f}, {loc_y:.1f}) mm',
        )

        # Ground-truth marker
        if gt:
            ax_grid.plot(
                gt['loc_x'], gt['loc_y'], 'x',
                color='#ff4444', markersize=14, markeredgewidth=2.5,
                zorder=6, label=f"Ground truth  ({gt['loc_x']:.0f}, {gt['loc_y']:.1f}) mm",
            )

        leg = ax_grid.legend(
            loc='upper right', fontsize=8,
            facecolor='#1a1a2e', edgecolor='#444466',
            labelcolor='white', framealpha=0.85,
        )
    else:
        ax_grid.text(0.5, 0.5, '— no contact detected —',
                     transform=ax_grid.transAxes,
                     ha='center', va='center', color='#666688', fontsize=13)

    # Colorbar
    sm = plt.cm.ScalarMappable(
        cmap='hot', norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax_grid, fraction=0.025, pad=0.015)
    cb.set_label('Contact probability', color='#aaaaaa', fontsize=8)
    cb.ax.yaxis.set_tick_params(color='#777777', labelsize=7)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='#aaaaaa')

    out = save_path or 'contact_vis.png'
    plt.savefig(out, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"Saved → {out}")
    plt.close(fig)


# ── Static mode ───────────────────────────────────────────────────────────────
def _run_static(args):
    import pandas as pd

    if args.image:
        img_path = args.image
        gt = None
    else:
        csv_path = os.path.join(_HERE, 'dataset.csv')
        if not os.path.exists(csv_path):
            sys.exit(f"dataset.csv not found at {csv_path}")
        df = pd.read_csv(csv_path)

        if args.session:
            sdf = df[df['session'] == args.session].reset_index(drop=True)
            if sdf.empty:
                sys.exit(f"Session '{args.session}' not in dataset.csv")
            diffs = (sdf['displacement_mm'] - args.depth).abs()
            row   = sdf.loc[diffs.idxmin()]
        else:
            # Pick a random in-contact frame near the requested depth
            tol = 0.5
            sub = df[(df['displacement_mm'] - args.depth).abs() < tol]
            if sub.empty:
                sub = df[df['displacement_mm'] > 0.1]
            row = sub.sample(1, random_state=args.seed).iloc[0]

        img_path = row['image_path']
        gt = {
            'loc_x':   float(row['loc_x']),
            'loc_y':   float(row['loc_y']),
            'disp_mm': float(row['displacement_mm']),
        }
        print(f"Session: {row['session']}  "
              f"GT: ({gt['loc_x']:.0f}, {gt['loc_y']:.1f}) mm  "
              f"depth: {gt['disp_mm']:.3f} mm")

    print(f"Image: {img_path}")
    pil      = Image.open(img_path).convert('L')
    skin_u8  = np.array(pil, dtype=np.uint8)
    tensor   = _TRANSFORM(pil).unsqueeze(0)   # (1,1,224,224)

    out = _infer(tensor)

    contact = out['contact_map'][0].numpy()
    loc_x   = out['loc_x'][0].item()
    loc_y   = out['loc_y'][0].item()
    disp_mm = out['displacement'][0].item() * DISP_MAX
    force_n = out['force'][0].item() * FORCE_MAX

    print(f"Predicted: ({loc_x:.2f}, {loc_y:.2f}) mm  "
          f"disp: {disp_mm:.3f} mm  force: {force_n:.4f} N  "
          f"conf: {contact.max():.3f}")

    _make_figure(
        skin_img_gray=skin_u8,
        contact_map=contact,
        loc_x=loc_x, loc_y=loc_y,
        disp_mm=disp_mm, force_n=force_n,
        gt=gt,
        save_path=args.save,
    )


# ── CLI ────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description='Skin image + sensor grid contact visualisation')
    ap.add_argument('--image',   default=None,
                    help='Path to a saved skin image (overrides dataset lookup)')
    ap.add_argument('--session', default=None,
                    help='Dataset session ID, e.g. x174_y8')
    ap.add_argument('--depth',   type=float, default=6.0,
                    help='Target indentation depth for frame selection (mm, default 6)')
    ap.add_argument('--save',    default=None,
                    help='Output file path (default: contact_vis.png)')
    ap.add_argument('--seed',    type=int, default=0,
                    help='Random seed for frame selection')
    args = ap.parse_args()

    _run_static(args)


if __name__ == '__main__':
    main()
