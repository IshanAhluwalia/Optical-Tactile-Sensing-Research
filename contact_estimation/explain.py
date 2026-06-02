"""
Two-part model explanation figure:

  1. Grad-CAM — which regions of the dot pattern the model attends to
     when estimating location, displacement, and force at different depths.

  2. Prediction traces — model-predicted vs ground-truth force and displacement
     over a full 0→10 mm press on the two held-out (unseen) validation sessions.

Usage:
    python contact_estimation/explain.py

Output:
    contact_estimation/assets/gradcam.png
    contact_estimation/assets/prediction_traces.png
"""

import csv
import json
import os

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

BASE_DIR   = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'best_model.pth')
STATS_PATH = os.path.join(BASE_DIR, 'model', 'model_stats.json')
ASSETS_DIR = os.path.join(BASE_DIR, 'assets')
OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'dataset', 'output')
os.makedirs(ASSETS_DIR, exist_ok=True)

TARGETS      = ['loc_x', 'loc_y', 'displacement_mm', 'force_n']
VAL_SESSIONS = ['(-140, -6)', '(-140, -14)']

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

BG = '#0F1117'
PANEL_BG = '#1A1D27'
GRID_COLOR = '#252535'


# ── Model ────────────────────────────────────────────────────────────────────
def build_model():
    m = models.resnet18(weights=None)
    m.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(m.fc.in_features, 128),
        nn.ReLU(),
        nn.Linear(128, len(TARGETS)),
    )
    return m


def load_model():
    with open(STATS_PATH) as f:
        stats = json.load(f)
    device = torch.device('mps'  if torch.backends.mps.is_available() else
                          'cuda' if torch.cuda.is_available()          else 'cpu')
    model = build_model().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model, stats, device


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ── Grad-CAM ─────────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model):
        self.model      = model
        self.activations = None
        self.gradients   = None
        target_layer = model.layer4[-1]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _, __, output):
        self.activations = output.detach()

    def _save_gradient(self, _, __, grad_output):
        self.gradients = grad_output[0].detach()

    def compute(self, inp, target_idx):
        self.model.zero_grad()
        out = self.model(inp)
        out[0, target_idx].backward(retain_graph=True)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam[0, 0].cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def overlay_cam(img_rgb, cam, alpha=0.5):
    cam_up = cv2.resize(cam, (img_rgb.shape[1], img_rgb.shape[0]))
    heatmap = cv2.applyColorMap((cam_up * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return (img_rgb * (1 - alpha) + heatmap * alpha).astype(np.uint8)


def find_frame(session, target_disp):
    import glob
    csvs = glob.glob(os.path.join(OUTPUT_DIR, session, '*.csv'))
    if not csvs:
        return None
    best, best_d = None, 1e9
    with open(csvs[0]) as f:
        for row in csv.DictReader(f):
            d = abs(float(row['displacement_mm']) - target_disp)
            ep = row.get('extracted_path', '')
            if d < best_d and ep and os.path.exists(ep):
                img = cv2.imread(ep)
                if img is not None and img.sum() > 0:
                    best_d, best = d, row
    return best


# ── Figure 1: Grad-CAM ───────────────────────────────────────────────────────
def plot_gradcam(model, stats, device):
    cam_engine = GradCAM(model)

    # 3 target depths × a train session (y=0) and a val session (y=-6)
    sessions   = ['(-140, 0)', '(-140, -6)']
    depths     = [1.5, 5.0, 9.0]
    target_names = ['Location Y', 'Displacement', 'Force']
    target_idxs  = [1, 2, 3]          # indices in TARGETS list
    target_colors = ['#E8714C', '#4C9BE8', '#48B87E']

    n_sessions = len(sessions)
    n_depths   = len(depths)
    n_targets  = len(target_names)

    # Layout: rows = sessions × depths, cols = raw + 3 cam outputs
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor(BG)
    fig.suptitle('Grad-CAM  —  Where the model looks to estimate each output',
                 fontsize=14, fontweight='bold', color='white', y=0.98)

    n_rows = n_sessions * n_depths
    n_cols = 1 + n_targets   # raw pattern + one cam per target
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                           hspace=0.05, wspace=0.04,
                           left=0.06, right=0.98, top=0.93, bottom=0.06)

    col_labels = ['Extracted Pattern'] + target_names
    col_colors = ['white'] + target_colors
    for c, (lbl, col) in enumerate(zip(col_labels, col_colors)):
        fig.text((0.06 + c * (0.92 / n_cols) + 0.92 / n_cols / 2),
                 0.955, lbl, color=col, fontsize=10, fontweight='bold',
                 ha='center', va='bottom')

    row_idx = 0
    for s_idx, session in enumerate(sessions):
        label = f"y = {session.split(',')[1].strip().rstrip(')')}"
        split = 'val (unseen)' if session in VAL_SESSIONS else 'train'
        for d_idx, depth in enumerate(depths):
            row = find_frame(session, depth)
            if row is None:
                row_idx += 1
                continue

            pat = cv2.imread(row['extracted_path'])
            pat_rgb = cv2.cvtColor(pat, cv2.COLOR_BGR2RGB)

            pil = Image.fromarray(pat_rgb)
            inp = transform(pil).unsqueeze(0).to(device)
            inp.requires_grad_(True)

            # raw pattern
            ax = fig.add_subplot(gs[row_idx, 0])
            ax.imshow(pat_rgb)
            ax.axis('off')
            ax.set_facecolor(PANEL_BG)
            disp_val = float(row['displacement_mm'])
            force_val = float(row['force_n'])
            side_label = f"{label} [{split}]\n{disp_val:.1f} mm  {force_val:.3f} N"
            if d_idx == 1:
                fig.text(0.01, ax.get_position().y0 + ax.get_position().height / 2,
                         side_label, color='#AAAAAA', fontsize=7.5,
                         ha='left', va='center', rotation=90)

            # Grad-CAM for each target
            for t_idx, (t_name, t_color, t_out_idx) in enumerate(
                    zip(target_names, target_colors, target_idxs)):
                cam = cam_engine.compute(inp, t_out_idx)
                overlay = overlay_cam(
                    cv2.resize(pat_rgb, (224, 224)), cam, alpha=0.55)

                ax2 = fig.add_subplot(gs[row_idx, t_idx + 1])
                ax2.imshow(overlay)
                ax2.axis('off')

                for spine in ax2.spines.values():
                    spine.set_edgecolor(t_color)
                    spine.set_linewidth(1.5)

            row_idx += 1

    out = os.path.join(ASSETS_DIR, 'gradcam.png')
    plt.savefig(out, dpi=150, facecolor=BG, bbox_inches='tight')
    plt.close()
    print(f"Saved → {out}")


# ── Figure 2: Prediction traces ──────────────────────────────────────────────
def plot_prediction_traces(model, stats, device):
    import glob

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.patch.set_facecolor(BG)
    fig.suptitle('Prediction Traces on Held-Out Validation Sessions  (never seen during training)',
                 fontsize=13, fontweight='bold', color='white', y=0.98)

    output_cfg = [
        ('displacement_mm', 'Displacement (mm)', '#4C9BE8', 2),
        ('force_n',         'Force (N)',          '#48B87E', 3),
    ]

    for col, session in enumerate(VAL_SESSIONS):
        csvs = glob.glob(os.path.join(OUTPUT_DIR, session, '*.csv'))
        if not csvs:
            continue

        times, gt_disp, gt_force = [], [], []
        pred_disp, pred_force    = [], []

        with open(csvs[0]) as f:
            rows = list(csv.DictReader(f))

        print(f"  Running trace for {session} ({len(rows)} frames)...")
        for row in rows:
            ep = row.get('extracted_path', '')
            if not ep or not os.path.exists(ep):
                continue
            img = cv2.imread(ep)
            if img is None or img.sum() == 0:
                continue

            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            inp = transform(pil).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(inp)[0].cpu().numpy()

            mn_d, mx_d = stats['displacement_mm']
            mn_f, mx_f = stats['force_n']

            times.append(float(row['time_s']))
            gt_disp.append(float(row['displacement_mm']))
            gt_force.append(float(row['force_n']))
            pred_disp.append(float(out[2]) * (mx_d - mn_d) + mn_d)
            pred_force.append(float(out[3]) * (mx_f - mn_f) + mn_f)

        times    = np.array(times)
        gt_d     = np.array(gt_disp)
        gt_f     = np.array(gt_force)
        pred_d   = np.array(pred_disp)
        pred_f   = np.array(pred_force)

        mae_d = np.mean(np.abs(pred_d - gt_d))
        mae_f = np.mean(np.abs(pred_f - gt_f))

        sess_label = f"y = {session.split(',')[1].strip().rstrip(')')}"

        for row_idx, (gt, pred, label, color, mae) in enumerate([
            (gt_d, pred_d, 'Displacement (mm)', '#4C9BE8', mae_d),
            (gt_f, pred_f, 'Force (N)',          '#48B87E', mae_f),
        ]):
            ax = axes[row_idx][col]
            ax.set_facecolor(PANEL_BG)
            ax.grid(True, color=GRID_COLOR, lw=0.8)

            ax.plot(times, gt,   color='white',  lw=1.5, alpha=0.8, label='Ground truth')
            ax.plot(times, pred, color=color,    lw=1.5, alpha=0.9, label=f'Predicted  (MAE={mae:.3f})')
            ax.fill_between(times, gt, pred, alpha=0.15, color=color)

            ax.set_title(f'{sess_label}  —  {label}', color='white', fontsize=10, pad=6)
            ax.set_xlabel('Time (s)', color='#AAAAAA', fontsize=9)
            ax.set_ylabel(label, color='#AAAAAA', fontsize=9)
            ax.tick_params(colors='#AAAAAA', labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor('#333344')
            ax.legend(fontsize=8, facecolor=PANEL_BG, labelcolor='white',
                      edgecolor='#333344')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(ASSETS_DIR, 'prediction_traces.png')
    plt.savefig(out, dpi=150, facecolor=BG)
    plt.close()
    print(f"Saved → {out}")


if __name__ == '__main__':
    model, stats, device = load_model()
    print("Generating Grad-CAM figure...")
    plot_gradcam(model, stats, device)
    print("Generating prediction traces figure...")
    plot_prediction_traces(model, stats, device)
    print("\nDone.")
