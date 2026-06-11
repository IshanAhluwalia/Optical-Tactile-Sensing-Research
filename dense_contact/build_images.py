"""
Build the grayscale PNG dataset from raw frames.

For every session (excluding y=18):
  - Reads each raw frame JPG
  - Crops to the selected ROI
  - Converts to grayscale
  - Saves as PNG under dense_contact/images/{session}/frame_NNNNN.png

Then writes dense_contact/dataset.csv with columns:
  image_path, loc_x, loc_y, displacement_mm, force_n, session

Supports resume: skips PNGs that already exist.
Uses multiprocessing for speed.

Usage:
    python dense_contact/build_images.py
"""

import csv
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
_HERE       = Path(__file__).parent
DATA_DIR    = _HERE.parent / 'dataset' / 'output'
IMAGES_DIR  = _HERE / 'images'
OUT_CSV     = _HERE / 'dataset.csv'
ROI_FILE    = _HERE / 'roi.json'

# Load ROI
with open(ROI_FILE) as f:
    _roi = json.load(f)
X0, X1 = _roi['x_start'], _roi['x_end']
Y0, Y1 = _roi['y_start'], _roi['y_end']

CROP_W = X1 - X0   # 538
CROP_H = Y1 - Y0   # 163


# ── Per-session worker (runs in subprocess) ───────────────────────────────────

def process_session(session_dir: str) -> list[dict]:
    """
    Crop + greyscale every frame in one session.
    Returns list of row dicts for the master CSV.
    Skips PNGs that already exist (resume support).
    """
    m = re.match(r'x(\d+)_y(\d+)$', session_dir)
    if not m:
        return []
    loc_x, loc_y = int(m.group(1)), int(m.group(2))
    if loc_y == 18:
        return []

    session_path = DATA_DIR / session_dir
    csv_path     = session_path / f'{session_dir}.csv'
    out_dir      = IMAGES_DIR / session_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        return []

    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_path = row.get('image_path', '').strip()
            if not raw_path or not os.path.exists(raw_path):
                continue

            frame_num = int(row['frame'])
            png_name  = f'frame_{frame_num:05d}.png'
            png_path  = out_dir / png_name

            # Crop + greyscale (skip if already done)
            if not png_path.exists():
                img = cv2.imread(raw_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                crop = img[Y0:Y1, X0:X1]
                cv2.imwrite(str(png_path), crop)

            rows.append({
                'image_path':      str(png_path),
                'loc_x':           loc_x,
                'loc_y':           loc_y,
                'displacement_mm': float(row['displacement_mm']),
                'force_n':         float(row['force_n']),
                'session':         session_dir,
            })

    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    sessions = sorted(
        d for d in os.listdir(DATA_DIR)
        if re.match(r'x\d+_y\d+$', d) and '_y18' not in d
    )

    print(f'Sessions to process : {len(sessions)}')
    print(f'ROI                 : x=[{X0}:{X1}]  y=[{Y0}:{Y1}]  →  {CROP_W}×{CROP_H} px')
    print(f'Output images       : {IMAGES_DIR}')
    print(f'Output CSV          : {OUT_CSV}')
    print()

    all_rows = []
    done = 0

    # Use up to 8 workers — each handles one session independently
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(process_session, s): s for s in sessions}
        for future in as_completed(futures):
            rows = future.result()
            all_rows.extend(rows)
            done += 1
            session = futures[future]
            print(f'  [{done:3d}/{len(sessions)}]  {session}  →  {len(rows)} frames', flush=True)

    # Sort by session then frame for reproducibility
    all_rows.sort(key=lambda r: (r['session'], r['image_path']))

    # Write master CSV
    fieldnames = ['image_path', 'loc_x', 'loc_y', 'displacement_mm', 'force_n', 'session']
    with open(OUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f'\nDone.')
    print(f'  Total frames   : {len(all_rows):,}')
    print(f'  Sessions       : {len(sessions)}')
    print(f'  Crop size      : {CROP_W} x {CROP_H} px')
    print(f'  CSV            : {OUT_CSV}')
    print(f'  Images folder  : {IMAGES_DIR}')


if __name__ == '__main__':
    main()
