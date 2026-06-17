"""
Validation grid visualisation for DenseContactNet.

For each held-out validation session the model runs on frames sampled at
fixed indentation depths (2, 4, 6, 8, 10 mm).  Results are shown as a
2-D spatial grid (sensor X × Y axes) at each depth level:

  Top row    : actual displacement (ground truth)
  Bottom row : predicted displacement

A final column shows the overall predicted vs actual scatter.

Usage:
    python dense_contact/eval_grid.py
    python dense_contact/eval_grid.py --save eval_grid.png
"""

import argparse, json, os, random, sys
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, os.path.dirname(__file__))
from model import DenseContactNet
from dataset import GRID_X, GRID_Y

# ── Config ─────────────────────────────────────────────────────────────────────
_HERE      = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(_HERE, 'dataset.csv')
MODEL_PATH = os.path.join(_HERE, 'model', 'best_model.pth')
STATS_PATH = os.path.join(_HERE, 'model', 'model_stats.json')

GRAY_MEAN, GRAY_STD = [0.4513], [0.2898]
TARGET_DEPTHS = [2.0, 4.0, 6.0, 8.0, 10.0]   # mm — columns in the grid plot
DEPTH_TOL     = 0.4                             # ± mm tolerance for frame selection

VAL_FRACTION = 0.2
RANDOM_SEED  = 42

_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(GRAY_MEAN, GRAY_STD),
])

# ── Session split (must match training) ───────────────────────────────────────
def get_val_sessions():
    sessions = sorted(pd.read_csv(CSV_PATH)['session'].unique().tolist())
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(sessions)
    n_val = max(1, int(len(sessions) * VAL_FRACTION))
    return sessions[:n_val]

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save', default='eval_grid.png')
    args = ap.parse_args()

    device = (torch.device('mps') if torch.backends.mps.is_available() else
              torch.device('cuda') if torch.cuda.is_available() else
              torch.device('cpu'))
    print(f"Device: {device}")

    model = DenseContactNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    with open(STATS_PATH) as f:
        stats = json.load(f)
    DISP_MAX  = stats['disp_max']
    FORCE_MAX = stats['force_max']

    df           = pd.read_csv(CSV_PATH)
    val_sessions = get_val_sessions()
    print(f"Val sessions ({len(val_sessions)}): {val_sessions}")

    # ── Run inference ──────────────────────────────────────────────────────────
    records = []   # {session, loc_x, loc_y, target_depth, actual_disp, pred_disp, pred_x, pred_y, pred_force, actual_force}

    for sess in val_sessions:
        sdf    = df[df['session'] == sess].reset_index(drop=True)
        loc_x  = float(sdf['loc_x'].iloc[0])
        loc_y  = float(sdf['loc_y'].iloc[0])

        for td in TARGET_DEPTHS:
            # Pick the frame closest to this target depth
            diffs = (sdf['displacement_mm'] - td).abs()
            if diffs.min() > DEPTH_TOL:
                continue
            row = sdf.loc[diffs.idxmin()]

            img    = Image.open(row['image_path']).convert('L')
            tensor = _TRANSFORM(img).unsqueeze(0).to(device)

            with torch.no_grad():
                out = model(tensor)

            records.append({
                'session':      sess,
                'loc_x':        loc_x,
                'loc_y':        loc_y,
                'target_depth': td,
                'actual_disp':  float(row['displacement_mm']),
                'pred_disp':    out['displacement'][0].item() * DISP_MAX,
                'pred_x':       out['loc_x'][0].item(),
                'pred_y':       out['loc_y'][0].item(),
                'actual_force': float(row['force_n']),
                'pred_force':   out['force'][0].item() * FORCE_MAX,
            })

    res = pd.DataFrame(records)
    print(res[['session','target_depth','actual_disp','pred_disp','pred_x','pred_y']].to_string(index=False))

    # ── Plot ───────────────────────────────────────────────────────────────────
    n_depths = len(TARGET_DEPTHS)
    fig = plt.figure(figsize=(5 * n_depths + 3, 14), facecolor='#0d0d1a')

    cmap    = plt.cm.plasma
    vmin, vmax = 0.0, 10.5

    def draw_sensor_grid(ax, title):
        ax.set_facecolor('#0d0d1a')
        ax.set_xlim(134, 214)
        ax.set_ylim(-1, 17)
        ax.set_xlabel('X — sensor length (mm)', color='#aaaaaa', fontsize=8)
        ax.set_ylabel('Y — sensor width (mm)',  color='#aaaaaa', fontsize=8)
        ax.set_title(title, color='white', fontsize=9, pad=4)
        ax.tick_params(colors='#777777', labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor('#333333')
        # faint grid dots at sensor positions
        for gx in GRID_X[::4]:
            for gy in GRID_Y[::4]:
                ax.plot(gx, gy, '.', color='#222244', markersize=2, zorder=1)

    # ── Rows 1 & 2: spatial grid at each depth ─────────────────────────────────
    gs = fig.add_gridspec(3, n_depths + 1,
                          left=0.05, right=0.97, top=0.93, bottom=0.06,
                          hspace=0.45, wspace=0.3,
                          width_ratios=[1]*n_depths + [0.07])

    for ci, td in enumerate(TARGET_DEPTHS):
        sub = res[res['target_depth'] == td]

        # Row 0: actual
        ax_a = fig.add_subplot(gs[0, ci])
        draw_sensor_grid(ax_a, f'{td:.0f} mm  —  Actual')
        sc = ax_a.scatter(sub['loc_x'], sub['loc_y'],
                          c=sub['actual_disp'], cmap=cmap, vmin=vmin, vmax=vmax,
                          s=220, edgecolors='white', linewidths=0.5, zorder=3)

        # Row 1: predicted displacement at actual location + arrow to predicted location
        ax_p = fig.add_subplot(gs[1, ci])
        draw_sensor_grid(ax_p, f'{td:.0f} mm  —  Predicted')
        # Draw arrow from actual → predicted location
        for _, r in sub.iterrows():
            color = cmap((r['pred_disp'] - vmin) / (vmax - vmin))
            ax_p.annotate('', xy=(r['pred_x'], r['pred_y']),
                          xytext=(r['loc_x'], r['loc_y']),
                          arrowprops=dict(arrowstyle='->', color='#888888',
                                          lw=0.8, mutation_scale=8))
            ax_p.scatter(r['loc_x'],  r['loc_y'],  c='#334455', s=80,
                         edgecolors='#556677', linewidths=0.5, zorder=3)
            ax_p.scatter(r['pred_x'], r['pred_y'],
                         c=[color], s=220,
                         edgecolors='white', linewidths=0.5, zorder=4)

    # shared colourbar
    cax = fig.add_subplot(gs[0:2, -1])
    sm  = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cb  = fig.colorbar(sm, cax=cax)
    cb.set_label('Displacement (mm)', color='white', fontsize=8)
    cb.ax.yaxis.set_tick_params(color='white', labelsize=7)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='white')

    # ── Row 2: scatter — predicted vs actual for disp + location ──────────────
    ax_disp = fig.add_subplot(gs[2, :2])
    ax_disp.set_facecolor('#0d0d1a')
    ax_disp.scatter(res['actual_disp'], res['pred_disp'],
                    c=res['actual_disp'], cmap=cmap, vmin=vmin, vmax=vmax,
                    s=60, edgecolors='white', linewidths=0.3, zorder=3, alpha=0.85)
    lim = [0, 11]
    ax_disp.plot(lim, lim, '--', color='#666688', lw=1, zorder=2)
    mae_d = (res['pred_disp'] - res['actual_disp']).abs().mean()
    ax_disp.set_title(f'Displacement  MAE = {mae_d:.3f} mm', color='white', fontsize=9)
    ax_disp.set_xlabel('Actual (mm)',    color='#aaaaaa', fontsize=8)
    ax_disp.set_ylabel('Predicted (mm)', color='#aaaaaa', fontsize=8)
    ax_disp.set_xlim(*lim); ax_disp.set_ylim(*lim)
    ax_disp.tick_params(colors='#777777', labelsize=7)
    for sp in ax_disp.spines.values(): sp.set_edgecolor('#333333')

    ax_x = fig.add_subplot(gs[2, 2])
    ax_x.set_facecolor('#0d0d1a')
    ax_x.scatter(res['loc_x'], res['pred_x'],
                 c=res['actual_disp'], cmap=cmap, vmin=vmin, vmax=vmax,
                 s=60, edgecolors='white', linewidths=0.3, alpha=0.85)
    ax_x.plot([138,210],[138,210], '--', color='#666688', lw=1)
    mae_x = (res['pred_x'] - res['loc_x']).abs().mean()
    ax_x.set_title(f'Location X  MAE = {mae_x:.2f} mm', color='white', fontsize=9)
    ax_x.set_xlabel('Actual X (mm)',    color='#aaaaaa', fontsize=8)
    ax_x.set_ylabel('Predicted X (mm)', color='#aaaaaa', fontsize=8)
    ax_x.tick_params(colors='#777777', labelsize=7)
    for sp in ax_x.spines.values(): sp.set_edgecolor('#333333')

    ax_y = fig.add_subplot(gs[2, 3])
    ax_y.set_facecolor('#0d0d1a')
    ax_y.scatter(res['loc_y'], res['pred_y'],
                 c=res['actual_disp'], cmap=cmap, vmin=vmin, vmax=vmax,
                 s=60, edgecolors='white', linewidths=0.3, alpha=0.85)
    ax_y.plot([0,16],[0,16], '--', color='#666688', lw=1)
    mae_y = (res['pred_y'] - res['loc_y']).abs().mean()
    ax_y.set_title(f'Location Y  MAE = {mae_y:.2f} mm', color='white', fontsize=9)
    ax_y.set_xlabel('Actual Y (mm)',    color='#aaaaaa', fontsize=8)
    ax_y.set_ylabel('Predicted Y (mm)', color='#aaaaaa', fontsize=8)
    ax_y.tick_params(colors='#777777', labelsize=7)
    for sp in ax_y.spines.values(): sp.set_edgecolor('#333333')

    ax_f = fig.add_subplot(gs[2, 4])
    ax_f.set_facecolor('#0d0d1a')
    ax_f.scatter(res['actual_force'], res['pred_force'],
                 c=res['actual_disp'], cmap=cmap, vmin=vmin, vmax=vmax,
                 s=60, edgecolors='white', linewidths=0.3, alpha=0.85)
    fmax = max(res['actual_force'].max(), res['pred_force'].max()) * 1.05
    ax_f.plot([0,fmax],[0,fmax], '--', color='#666688', lw=1)
    mae_f = (res['pred_force'] - res['actual_force']).abs().mean()
    ax_f.set_title(f'Force  MAE = {mae_f:.4f} N', color='white', fontsize=9)
    ax_f.set_xlabel('Actual (N)',    color='#aaaaaa', fontsize=8)
    ax_f.set_ylabel('Predicted (N)', color='#aaaaaa', fontsize=8)
    ax_f.tick_params(colors='#777777', labelsize=7)
    for sp in ax_f.spines.values(): sp.set_edgecolor('#333333')

    fig.suptitle('DenseContactNet — Validation Set: Predicted vs Actual\n'
                 'Top row: actual location + depth  |  Middle row: predicted location + depth (arrow = shift)  |  Bottom: scatter',
                 color='white', fontsize=10, y=0.98)

    plt.savefig(args.save, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"Saved → {args.save}")

if __name__ == '__main__':
    main()
