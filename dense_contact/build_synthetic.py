"""
build_synthetic.py — Synthetically generate small-indentation tactile images.

Physics model
-------------
The sensor membrane is clamped at its edges and free at the centre.
Stiffness increases toward the boundary (like a drumhead).

We model this with a compliance map C(px, py) that is high (soft, deforms
freely) at the image centre and low (stiff, resists deformation) at the edges:

    C(px, py) = C_min + (1 - C_min) * sin(π*px/W) * sin(π*py/H)

C_min = 0.3 means edge pixels contribute only 30% as much deformation as
centre pixels for the same applied depth — consistent with a clamped membrane.

Synthesis
---------
For a real frame at depth δ_src with rest image I_rest:
    deformation D = I_src - I_rest                         (signed float)
    I_synth(δ_new) = I_rest + (δ_new/δ_src) * D * C       (stiffness-modulated)
    F_new          = F_src  * (δ_new/δ_src)^1.5           (Hertz scaling)

Outputs
-------
  dense_contact/images/{session}/synth_*.png   — synthetic images
  dense_contact/dataset_synthetic.csv           — metadata for new frames
  dense_contact/synthetic_viz/                  — visualisation figures
"""

import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')   # no display needed — saves to file
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

_HERE      = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(_HERE, 'dataset.csv')
OUT_CSV    = os.path.join(_HERE, 'dataset_synthetic.csv')
IMAGES_DIR = os.path.join(_HERE, 'images')
VIZ_DIR    = os.path.join(_HERE, 'synthetic_viz')
os.makedirs(VIZ_DIR, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────

# Synthetic target depths (mm) — all shallower than MIN_SOURCE_DEPTH
SYNTH_DEPTHS = [0.3, 0.6, 1.0, 1.5, 2.0, 3.0]

# Only use source frames deeper than this (need enough deformation signal)
MIN_SOURCE_DEPTH = 5.0   # mm

# How many source frames to sample per session (evenly spaced in the deep zone)
N_SOURCE_FRAMES = 5

# Compliance model: edges have MIN_COMPLIANCE × centre compliance
MIN_COMPLIANCE = 0.3

IMG_H, IMG_W = 163, 538   # grayscale crop dimensions


# ── Stiffness / compliance model ──────────────────────────────────────────────

def make_compliance_map(H: int = IMG_H, W: int = IMG_W) -> np.ndarray:
    """
    Sinusoidal compliance map in [MIN_COMPLIANCE, 1.0].

    Physically motivated by a membrane clamped at all four edges:
    - sin(π*x/W) = 0 at left/right edges, 1 at horizontal centre
    - sin(π*y/H) = 0 at top/bottom edges, 1 at vertical centre
    - Product gives zero at any edge, 1.0 only at the membrane centre

    Edges are stiffer → lower compliance → less visible deformation per mm depth.
    """
    px = np.arange(W, dtype=np.float32)
    py = np.arange(H, dtype=np.float32)
    cx = np.sin(np.pi * px / (W - 1))   # (W,)
    cy = np.sin(np.pi * py / (H - 1))   # (H,)
    CX, CY = np.meshgrid(cx, cy)         # (H, W)
    compliance = MIN_COMPLIANCE + (1.0 - MIN_COMPLIANCE) * CX * CY
    return compliance.astype(np.float32)


COMPLIANCE = make_compliance_map()


def synthesise(rest: np.ndarray, src: np.ndarray,
               delta_src: float, delta_new: float,
               force_src: float) -> tuple[np.ndarray, float]:
    """
    Return (synthetic_image_uint8, synthetic_force).
    """
    alpha      = delta_new / delta_src
    deform     = src.astype(np.float32) - rest.astype(np.float32)
    synth      = rest.astype(np.float32) + alpha * deform * COMPLIANCE
    synth      = np.clip(synth, 0, 255).astype(np.uint8)
    force_new  = force_src * (delta_new / delta_src) ** 1.5
    return synth, force_new


# ── Visualisation ─────────────────────────────────────────────────────────────

def visualise_compliance():
    """Save compliance map figure."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    im = axes[0].imshow(COMPLIANCE, cmap='RdYlGn', vmin=0, vmax=1,
                        aspect='auto')
    plt.colorbar(im, ax=axes[0], label='Compliance (1=soft, 0.3=stiff)')
    axes[0].set_title('Membrane Compliance Map\n(edges clamped → stiffer)')
    axes[0].set_xlabel('Pixel X (along sensor width, 538px)')
    axes[0].set_ylabel('Pixel Y (along sensor height, 163px)')

    # 1D cross-sections
    mid_row = COMPLIANCE[IMG_H // 2, :]
    mid_col = COMPLIANCE[:, IMG_W // 2]
    axes[1].plot(mid_row, label='Horizontal cross-section (y = centre)')
    axes[1].plot(mid_col, label='Vertical cross-section (x = centre)')
    axes[1].axhline(MIN_COMPLIANCE, color='gray', linestyle='--',
                    label=f'Edge minimum ({MIN_COMPLIANCE})')
    axes[1].set_ylim(0, 1.1)
    axes[1].set_xlabel('Pixel position')
    axes[1].set_ylabel('Compliance')
    axes[1].set_title('Cross-sections through compliance map')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(VIZ_DIR, 'compliance_map.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


def visualise_synthesis(df: pd.DataFrame):
    """
    For 3 different sessions (edge, centre, middle), show:
    rest | deep source | synthesised depths 0.3→3mm | deformation signal
    """
    example_sessions = ['x138_y0', 'x174_y8', 'x210_y16']
    example_sessions = [s for s in example_sessions
                        if s in df['session'].unique()][:3]

    n_depths = len(SYNTH_DEPTHS)
    n_cols   = 2 + n_depths + 1   # rest, source, synths, deformation
    n_rows   = len(example_sessions)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 2.5, n_rows * 2.5))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    col_labels = (['Rest (δ=0)', f'Source (δ≈{MIN_SOURCE_DEPTH}mm)'] +
                  [f'Synth δ={d}mm' for d in SYNTH_DEPTHS] +
                  ['Deformation\nsignal'])

    for row_i, session in enumerate(example_sessions):
        sdf = df[df['session'] == session].sort_values('displacement_mm')

        # Rest frame
        rest_row = sdf.iloc[0]
        rest_img = np.array(Image.open(rest_row['image_path']).convert('L'))

        # Source frame (deepest available above MIN_SOURCE_DEPTH)
        src_candidates = sdf[sdf['displacement_mm'] >= MIN_SOURCE_DEPTH]
        if src_candidates.empty:
            continue
        src_row   = src_candidates.iloc[len(src_candidates) // 2]
        src_delta = float(src_row['displacement_mm'])
        src_force = float(src_row['force_n'])
        src_img   = np.array(Image.open(src_row['image_path']).convert('L'))

        deform = src_img.astype(np.float32) - rest_img.astype(np.float32)

        col = 0

        def _show(ax, img, title='', cmap='gray', vmin=None, vmax=None):
            ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
            ax.set_title(title, fontsize=8)
            ax.axis('off')

        loc_x = float(sdf['loc_x'].iloc[0])
        loc_y = float(sdf['loc_y'].iloc[0])
        row_label = f"{session}\n(x={loc_x:.0f}, y={loc_y:.0f}mm)"

        _show(axes[row_i, col], rest_img, 'Rest (δ=0mm)')
        axes[row_i, col].set_ylabel(row_label, fontsize=8)
        col += 1

        _show(axes[row_i, col], src_img, f'Source (δ={src_delta:.1f}mm)')
        col += 1

        for d in SYNTH_DEPTHS:
            synth, _ = synthesise(rest_img, src_img, src_delta, d, src_force)
            _show(axes[row_i, col], synth, f'Synth δ={d}mm')
            col += 1

        # Deformation signal (signed, centred at 128)
        deform_vis = np.clip(deform + 128, 0, 255).astype(np.uint8)
        _show(axes[row_i, col], deform_vis, 'Deformation\n(grey=zero)',
              cmap='RdBu_r', vmin=0, vmax=255)

    for c, lbl in enumerate(col_labels):
        axes[0, c].set_title(lbl, fontsize=8, fontweight='bold')

    plt.suptitle('Synthetic Image Generation — Stiffness-Modulated Interpolation',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(VIZ_DIR, 'synthetic_examples.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


def visualise_stiffness_effect(df: pd.DataFrame):
    """
    Show how the compliance map changes the deformation at edge vs centre sessions,
    at a fixed synthetic depth of 1.5mm.
    """
    sessions_to_compare = {
        'Edge   (x=138, y=0)':   'x138_y0',
        'Middle (x=174, y=8)':   'x174_y8',
    }
    sessions_to_compare = {k: v for k, v in sessions_to_compare.items()
                           if v in df['session'].unique()}

    fig, axes = plt.subplots(len(sessions_to_compare), 4,
                             figsize=(14, 3.5 * len(sessions_to_compare)))
    if len(sessions_to_compare) == 1:
        axes = axes[np.newaxis, :]

    for row_i, (label, session) in enumerate(sessions_to_compare.items()):
        sdf = df[df['session'] == session].sort_values('displacement_mm')
        rest_row  = sdf.iloc[0]
        src_row   = sdf[sdf['displacement_mm'] >= MIN_SOURCE_DEPTH].iloc[2]
        rest_img  = np.array(Image.open(rest_row['image_path']).convert('L'))
        src_img   = np.array(Image.open(src_row['image_path']).convert('L'))
        src_delta = float(src_row['displacement_mm'])
        src_force = float(src_row['force_n'])

        # Without stiffness model (uniform α scaling)
        alpha    = 1.5 / src_delta
        no_stiff = np.clip(rest_img.astype(np.float32) +
                           alpha * (src_img.astype(np.float32) - rest_img.astype(np.float32)),
                           0, 255).astype(np.uint8)

        # With stiffness model
        with_stiff, _ = synthesise(rest_img, src_img, src_delta, 1.5, src_force)

        diff = np.abs(with_stiff.astype(np.float32) - no_stiff.astype(np.float32))

        def _show(ax, img, title, cmap='gray', vmin=None, vmax=None):
            ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
            ax.set_title(title, fontsize=8)
            ax.axis('off')

        axes[row_i, 0].set_ylabel(label, fontsize=9)
        _show(axes[row_i, 0], rest_img,    'Rest')
        _show(axes[row_i, 1], no_stiff,    'δ=1.5mm\n(no stiffness model)')
        _show(axes[row_i, 2], with_stiff,  'δ=1.5mm\n(with stiffness model)')
        _show(axes[row_i, 3], diff,        'Difference\n(stiffness effect)',
              cmap='hot', vmin=0, vmax=20)

    plt.suptitle('Effect of Stiffness Model on Synthetic Images',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(VIZ_DIR, 'stiffness_effect.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


# ── Dataset generation ────────────────────────────────────────────────────────

def process_session(args):
    session, session_df = args
    rows = []
    session_img_dir = os.path.join(IMAGES_DIR, session)
    if not os.path.isdir(session_img_dir):
        return rows

    sdf       = session_df.sort_values('displacement_mm')
    rest_row  = sdf.iloc[0]
    rest_img  = np.array(Image.open(rest_row['image_path']).convert('L'))
    loc_x     = float(sdf['loc_x'].iloc[0])
    loc_y     = float(sdf['loc_y'].iloc[0])

    src_pool  = sdf[sdf['displacement_mm'] >= MIN_SOURCE_DEPTH]
    if src_pool.empty:
        return rows

    # Pick N_SOURCE_FRAMES evenly spaced source frames
    indices   = np.linspace(0, len(src_pool) - 1, N_SOURCE_FRAMES, dtype=int)
    src_pool  = src_pool.iloc[indices]

    for _, src_row in src_pool.iterrows():
        src_delta = float(src_row['displacement_mm'])
        src_force = float(src_row['force_n'])
        src_img   = np.array(Image.open(src_row['image_path']).convert('L'))
        src_stem  = os.path.splitext(os.path.basename(src_row['image_path']))[0]

        for d_new in SYNTH_DEPTHS:
            if d_new >= src_delta:
                continue

            synth, f_new = synthesise(rest_img, src_img, src_delta, d_new, src_force)

            fname     = f"synth_{src_stem}_d{d_new:.1f}.png"
            save_path = os.path.join(session_img_dir, fname)

            if not os.path.exists(save_path):
                Image.fromarray(synth).save(save_path)

            rows.append({
                'image_path':     save_path,
                'loc_x':          loc_x,
                'loc_y':          loc_y,
                'displacement_mm': d_new,
                'force_n':        round(f_new, 6),
                'session':        session,
            })

    return rows


def build_dataset(df: pd.DataFrame):
    sessions  = df['session'].unique().tolist()
    args_list = [(s, df[df['session'] == s]) for s in sessions]

    all_rows = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(process_session, a): a[0] for a in args_list}
        for fut in tqdm(as_completed(futs), total=len(futs),
                        desc="Generating synthetic data"):
            all_rows.extend(fut.result())

    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"\nSynthetic dataset: {len(out_df)} rows → {OUT_CSV}")
    return out_df


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} real frames from {len(df['session'].unique())} sessions")

    print("\n── Generating visualisations ─────────────────────────────────────")
    visualise_compliance()
    visualise_synthesis(df)
    visualise_stiffness_effect(df)

    print("\n── Generating synthetic dataset ──────────────────────────────────")
    synth_df = build_dataset(df)

    print("\nDone. Open these to see results:")
    print(f"  open {VIZ_DIR}/compliance_map.png")
    print(f"  open {VIZ_DIR}/synthetic_examples.png")
    print(f"  open {VIZ_DIR}/stiffness_effect.png")
