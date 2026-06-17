"""
Glare removal comparison for tactile sensor images.

Shows four methods side-by-side on a sample frame:
  1. Original (cropped ROI)
  2. Homomorphic filtering  (no reference needed)
  3. Dataset-median background division
  4. Homomorphic + CLAHE (current pipeline baseline)

Usage:
    python preprocessing/deglare_demo.py
    python preprocessing/deglare_demo.py --frame dataset/output/x160_y6/frames/frame_00050.jpg
    python preprocessing/deglare_demo.py --save comparison_deglare.png
"""

import argparse
import glob
import os
import random
import sys

import cv2
import numpy as np

# ── ROI ───────────────────────────────────────────────────────────────────────
import json
_roi_path = os.path.join(os.path.dirname(__file__), '..', 'roi.json')
with open(os.path.abspath(_roi_path)) as f:
    _roi = json.load(f)
RX, RY, RW, RH = _roi['x'], _roi['y'], _roi['w'], _roi['h']

# ── Methods ───────────────────────────────────────────────────────────────────

def crop_roi(bgr):
    return bgr[RY:RY + RH, RX:RX + RW]


def homomorphic(gray, sigma=60, alpha=0.5, beta=1.5):
    """
    Separate illumination (low-freq) from reflectance (high-freq) in log space.
    alpha scales the low-freq component down, beta scales high-freq up.
    """
    img = gray.astype(np.float32) + 1.0
    log_img = np.log(img)
    blur = cv2.GaussianBlur(log_img, (0, 0), sigma)
    filtered = alpha * blur + beta * (log_img - blur)
    out = np.exp(filtered) - 1.0
    out = np.clip(out, 0, 255).astype(np.uint8)
    # stretch to full range
    out = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX)
    return out


def build_median_background(n_frames=120, seed=42):
    """
    Sample frames uniformly across the dataset and compute per-pixel median.
    The contact pattern is random; the illumination/glare is fixed → median = glare.
    """
    all_frames = sorted(glob.glob(
        os.path.join(os.path.dirname(__file__), '..', 'dataset', 'output',
                     '*', 'frames', '*.jpg')
    ))
    if not all_frames:
        return None
    rng = random.Random(seed)
    chosen = rng.sample(all_frames, min(n_frames, len(all_frames)))
    stack = []
    for p in chosen:
        bgr = cv2.imread(p)
        if bgr is None:
            continue
        crop = crop_roi(bgr)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        stack.append(gray.astype(np.float32))
    if not stack:
        return None
    median_bg = np.median(np.stack(stack, axis=0), axis=0).astype(np.float32)
    print(f"Background computed from {len(stack)} frames.")
    return median_bg


def background_division(gray, bg):
    """Divide frame by background, re-scale to 0-255."""
    gray_f = gray.astype(np.float32)
    ratio = gray_f / (bg + 1e-3)
    # map to [0, 255]
    ratio = cv2.normalize(ratio, None, 0.0, 255.0, cv2.NORM_MINMAX)
    return ratio.astype(np.uint8)


_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def homo_plus_clahe(gray, sigma=60):
    homo = homomorphic(gray, sigma=sigma)
    return _clahe.apply(homo)


# ── Annotated tile ────────────────────────────────────────────────────────────

def tile(img_gray, label, w=559, h=160):
    rgb = cv2.cvtColor(cv2.resize(img_gray, (w, h)), cv2.COLOR_GRAY2BGR)
    cv2.putText(rgb, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    return rgb


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frame', default=None,
                    help='Path to a specific frame JPG (default: random from dataset)')
    ap.add_argument('--save', default=None,
                    help='If given, save comparison image to this path instead of displaying')
    ap.add_argument('--sigma', type=float, default=60,
                    help='Gaussian sigma for homomorphic filter (default 60)')
    args = ap.parse_args()

    # Pick frame
    if args.frame:
        frame_path = args.frame
    else:
        candidates = sorted(glob.glob(
            os.path.join(os.path.dirname(__file__), '..', 'dataset', 'output',
                         '*', 'frames', 'frame_0005*.jpg')
        ))
        if not candidates:
            print("No frames found. Use --frame <path>")
            sys.exit(1)
        frame_path = candidates[0]

    print(f"Frame: {frame_path}")
    bgr = cv2.imread(frame_path)
    if bgr is None:
        print(f"Could not load {frame_path}")
        sys.exit(1)

    crop = crop_roi(bgr)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Method 1: original
    t1 = tile(gray, "1. Original")

    # Method 2: homomorphic
    homo = homomorphic(gray, sigma=args.sigma)
    t2 = tile(homo, f"2. Homomorphic (sigma={args.sigma:.0f})")

    # Method 3: background division
    print("Building median background (sampling dataset)...")
    bg = build_median_background()
    if bg is not None:
        bgdiv = background_division(gray, bg)
        t3 = tile(bgdiv, "3. Background division (median)")
    else:
        t3 = tile(gray, "3. Background div (no data)")

    # Method 4: homomorphic + CLAHE
    hc = homo_plus_clahe(gray, sigma=args.sigma)
    t4 = tile(hc, "4. Homomorphic + CLAHE")

    # Stack 2x2
    gap = np.zeros((10, 559 * 2 + 4, 3), dtype=np.uint8)
    divider_v = np.zeros((160, 4, 3), dtype=np.uint8)
    row1 = np.hstack([t1, divider_v, t2])
    row2 = np.hstack([t3, divider_v, t4])
    comparison = np.vstack([row1, gap, row2])

    if args.save:
        cv2.imwrite(args.save, comparison)
        print(f"Saved to {args.save}")
    else:
        cv2.imshow("Glare removal comparison  (Q to quit)", comparison)
        while True:
            if cv2.waitKey(0) & 0xFF in (ord('q'), 27):
                break
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
