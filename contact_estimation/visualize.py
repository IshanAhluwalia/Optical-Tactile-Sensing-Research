"""
Visualize model performance on the full dataset.

Generates:
  contact_estimation/assets/performance.png  — scatter plots + sample predictions
  contact_estimation/assets/pipeline.png     — pipeline diagram

Usage:
    python contact_estimation/visualize.py
"""

import json
import os

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

BASE_DIR   = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'best_model.pth')
STATS_PATH = os.path.join(BASE_DIR, 'model', 'model_stats.json')
CSV_PATH   = os.path.join(BASE_DIR, 'dataset.csv')
ASSETS_DIR = os.path.join(BASE_DIR, 'assets')
os.makedirs(ASSETS_DIR, exist_ok=True)

TARGETS       = ['loc_x', 'loc_y', 'displacement_mm', 'force_n']
TARGETS_LABEL = ['Location X (mm)', 'Location Y (mm)', 'Displacement (mm)', 'Force (N)']
VAL_SESSIONS  = ['(-140, -6)', '(-140, -14)']

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def build_model():
    m = models.resnet18(weights=None)
    m.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(m.fc.in_features, 128),
        nn.ReLU(),
        nn.Linear(128, len(TARGETS)),
    )
    return m


def run_inference():
    with open(STATS_PATH) as f:
        stats = json.load(f)

    device = torch.device('mps'  if torch.backends.mps.is_available() else
                          'cuda' if torch.cuda.is_available()          else 'cpu')
    model = build_model().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    df = pd.read_csv(CSV_PATH)
    all_preds  = {t: [] for t in TARGETS}
    all_truths = {t: [] for t in TARGETS}
    is_val     = []

    print(f"Running inference on {len(df)} samples...")
    batch, batch_meta = [], []
    BATCH = 64

    def flush(batch, batch_meta):
        if not batch:
            return
        inp = torch.stack(batch).to(device)
        with torch.no_grad():
            out = model(inp).cpu().numpy()
        for i, (row, pred_norm) in enumerate(zip(batch_meta, out)):
            for j, t in enumerate(TARGETS):
                mn, mx = stats[t]
                all_preds[t].append(float(pred_norm[j]) * (mx - mn) + mn)
                all_truths[t].append(float(row[t]))
            is_val.append(row['session'] in VAL_SESSIONS)

    for _, row in df.iterrows():
        try:
            img = Image.open(row['extracted_path']).convert('RGB')
            batch.append(transform(img))
            batch_meta.append(row)
        except Exception:
            continue
        if len(batch) == BATCH:
            flush(batch, batch_meta)
            batch, batch_meta = [], []
    flush(batch, batch_meta)

    is_val = np.array(is_val)
    return all_preds, all_truths, is_val, stats


def mae(preds, truths):
    return np.mean(np.abs(np.array(preds) - np.array(truths)))


def plot_performance(all_preds, all_truths, is_val, stats):
    # Only plot loc_y, displacement, force (loc_x is constant)
    plot_targets = ['loc_y', 'displacement_mm', 'force_n']
    plot_labels  = ['Location Y (mm)', 'Displacement (mm)', 'Force (N)']
    colors_train = '#4C9BE8'
    colors_val   = '#E8714C'

    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor('#0F1117')

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35,
                           left=0.07, right=0.97, top=0.90, bottom=0.07)

    fig.suptitle('Contact Estimation Model — Performance', fontsize=18,
                 fontweight='bold', color='white', y=0.96)

    for col, (t, label) in enumerate(zip(plot_targets, plot_labels)):
        p = np.array(all_preds[t])
        g = np.array(all_truths[t])
        p_tr, g_tr = p[~is_val], g[~is_val]
        p_vl, g_vl = p[ is_val], g[ is_val]

        # ── Row 0: scatter predicted vs actual ──────────────────────────────
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor('#1A1D27')
        lim = [min(g.min(), p.min()) - 0.5, max(g.max(), p.max()) + 0.5]
        ax.plot(lim, lim, '--', color='#FFFFFF', lw=1, alpha=0.4, zorder=1)
        ax.scatter(g_tr, p_tr, s=2, alpha=0.3, color=colors_train, zorder=2, label='Train')
        ax.scatter(g_vl, p_vl, s=4, alpha=0.6, color=colors_val,   zorder=3, label='Val (unseen)')
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel(f'Actual {label}', color='#AAAAAA', fontsize=9)
        ax.set_ylabel(f'Predicted {label}', color='#AAAAAA', fontsize=9)
        ax.set_title(f'{label}\nMAE train={mae(p_tr,g_tr):.3f}  val={mae(p_vl,g_vl):.3f}',
                     color='white', fontsize=9)
        ax.tick_params(colors='#AAAAAA', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#333344')
        if col == 0:
            ax.legend(fontsize=7, facecolor='#1A1D27', labelcolor='white',
                      edgecolor='#333344', markerscale=4)

        # ── Row 1: residuals ─────────────────────────────────────────────────
        ax2 = fig.add_subplot(gs[1, col])
        ax2.set_facecolor('#1A1D27')
        ax2.axhline(0, color='white', lw=1, alpha=0.4)
        ax2.scatter(g_tr, p_tr - g_tr, s=2, alpha=0.3, color=colors_train)
        ax2.scatter(g_vl, p_vl - g_vl, s=4, alpha=0.6, color=colors_val)
        ax2.set_xlabel(f'Actual {label}', color='#AAAAAA', fontsize=9)
        ax2.set_ylabel('Residual (pred − actual)', color='#AAAAAA', fontsize=9)
        ax2.set_title('Residuals', color='white', fontsize=9)
        ax2.tick_params(colors='#AAAAAA', labelsize=8)
        for spine in ax2.spines.values():
            spine.set_edgecolor('#333344')

        # ── Row 2: error over displacement (force/loc) or over time ─────────
        ax3 = fig.add_subplot(gs[2, col])
        ax3.set_facecolor('#1A1D27')
        disp_all = np.array(all_truths['displacement_mm'])
        err_all  = np.abs(p - g)
        # Bin by displacement
        bins = np.linspace(0, 10, 21)
        bin_idx = np.digitize(disp_all, bins) - 1
        bin_mae_tr, bin_mae_vl = [], []
        bin_centers = (bins[:-1] + bins[1:]) / 2
        for b in range(len(bins) - 1):
            mask_tr = (~is_val) & (bin_idx == b)
            mask_vl = ( is_val) & (bin_idx == b)
            bin_mae_tr.append(err_all[mask_tr].mean() if mask_tr.sum() > 0 else np.nan)
            bin_mae_vl.append(err_all[mask_vl].mean() if mask_vl.sum() > 0 else np.nan)
        ax3.plot(bin_centers, bin_mae_tr, color=colors_train, lw=1.5, label='Train')
        ax3.plot(bin_centers, bin_mae_vl, color=colors_val,   lw=1.5, label='Val')
        ax3.set_xlabel('Displacement (mm)', color='#AAAAAA', fontsize=9)
        ax3.set_ylabel(f'MAE ({label})', color='#AAAAAA', fontsize=9)
        ax3.set_title('Error vs Indentation Depth', color='white', fontsize=9)
        ax3.tick_params(colors='#AAAAAA', labelsize=8)
        for spine in ax3.spines.values():
            spine.set_edgecolor('#333344')

    out = os.path.join(ASSETS_DIR, 'performance.png')
    plt.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved → {out}")


def plot_pipeline():
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.patch.set_facecolor('#0F1117')
    fig.suptitle('Contact Estimation Pipeline', fontsize=16, fontweight='bold',
                 color='white', y=1.02)

    # Find a verified loadable frame from session CSVs (not the aggregated CSV)
    import csv as _csv, glob as _glob
    OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'dataset', 'output')
    raw_path, ext_path, sample_disp, sample_force = None, None, 0.0, 0.0
    for session_dir in sorted(os.listdir(OUTPUT_DIR)):
        csvs = _glob.glob(os.path.join(OUTPUT_DIR, session_dir, '*.csv'))
        if not csvs:
            continue
        with open(csvs[0]) as f:
            for row in _csv.DictReader(f):
                d = float(row['displacement_mm'])
                if abs(d - 5.0) < 0.5:
                    ip, ep = row.get('image_path',''), row.get('extracted_path','')
                    if ip and ep and os.path.exists(ip) and os.path.exists(ep):
                        img_test = cv2.imread(ip)
                        ext_test = cv2.imread(ep)
                        if img_test is not None and ext_test is not None and img_test.sum() > 0:
                            raw_path, ext_path = ip, ep
                            sample_disp  = d
                            sample_force = float(row['force_n'])
                            break
        if raw_path:
            break

    raw = cv2.cvtColor(cv2.imread(raw_path), cv2.COLOR_BGR2RGB)
    pat = cv2.cvtColor(cv2.imread(ext_path), cv2.COLOR_BGR2RGB)

    step_titles = ['1. Raw Camera Frame', '2. ROI Crop', '3. Pattern Extraction', '4. Model Output']
    step_colors = ['#4C9BE8', '#48B87E', '#E8B84C', '#E8714C']

    # Load ROI json
    roi_path = os.path.join(BASE_DIR, '..', 'roi.json')
    with open(os.path.abspath(roi_path)) as f:
        roi = json.load(f)
    RX, RY, RW, RH = roi['x'], roi['y'], roi['w'], roi['h']

    imgs = [
        raw,
        raw[RY:RY+RH, RX:RX+RW],
        pat,
        None,
    ]

    for i, (ax, title, color) in enumerate(zip(axes, step_titles, step_colors)):
        ax.set_facecolor('#1A1D27')
        ax.set_title(title, color=color, fontsize=11, fontweight='bold', pad=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        if i < 3:
            ax.imshow(imgs[i])
            if i == 0:
                rect = plt.Rectangle((RX, RY), RW, RH, linewidth=2,
                                     edgecolor='#48B87E', facecolor='none')
                ax.add_patch(rect)
                ax.text(RX, RY - 6, 'ROI', color='#48B87E', fontsize=9, fontweight='bold')
        else:
            # Predictions panel
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_facecolor('#1A1D27')
            preds_display = [
                ('Location Y', '-8.0 mm', '#E8714C'),
                ('Displacement', f'{sample_disp:.2f} mm', '#4C9BE8'),
                ('Force', f'{sample_force:.3f} N', '#48B87E'),
            ]
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.axis('off')
            for k, (lbl, val, c) in enumerate(preds_display):
                y_pos = 0.80 - k * 0.24
                ax.text(0.08, y_pos, lbl, color='#AAAAAA', fontsize=11,
                        transform=ax.transAxes, va='bottom')
                ax.text(0.08, y_pos - 0.10, val, color=c, fontsize=17,
                        fontweight='bold', transform=ax.transAxes, va='bottom')
                ax.axhline(y_pos - 0.13, xmin=0.05, xmax=0.95, color='#333344', lw=0.8)
            ax.text(0.5, 0.04, 'ground truth labels', color='#555566',
                    fontsize=8, ha='center', transform=ax.transAxes)

    # Arrows between panels
    for i in range(3):
        fig.text(0.245 + i * 0.185, 0.5, '→', fontsize=24, color='#444455',
                 ha='center', va='center')

    plt.tight_layout()
    out = os.path.join(ASSETS_DIR, 'pipeline.png')
    plt.savefig(out, dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Saved → {out}")


if __name__ == '__main__':
    all_preds, all_truths, is_val, stats = run_inference()
    print("\nOverall MAE:")
    for t, label in zip(TARGETS, TARGETS_LABEL):
        p, g = np.array(all_preds[t]), np.array(all_truths[t])
        print(f"  {label}: train={mae(p[~is_val], g[~is_val]):.4f}  val={mae(p[is_val], g[is_val]):.4f}")
    plot_performance(all_preds, all_truths, is_val, stats)
    plot_pipeline()
    print("\nDone.")
