"""
Aggregate all recorded sessions into a single CSV for training.
Parses (loc_x, loc_y) from x{N}_y{N} folder names.

Usage:
    python dense_contact/build_dataset.py

Output:
    dense_contact/dataset.csv
"""

import csv
import glob
import os
import re

import pandas as pd

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'output')
OUT_CSV    = os.path.join(os.path.dirname(__file__), 'dataset.csv')


def main():
    rows = []

    for session_dir in sorted(os.listdir(OUTPUT_DIR)):
        full_path = os.path.join(OUTPUT_DIR, session_dir)
        if not os.path.isdir(full_path):
            continue

        m = re.match(r'x(\d+)_y(\d+)$', session_dir)
        if not m:
            print(f"  skip {session_dir!r} — unrecognised name")
            continue
        loc_x, loc_y = int(m.group(1)), int(m.group(2))

        csvs = glob.glob(os.path.join(full_path, '*.csv'))
        if not csvs:
            print(f"  skip {session_dir!r} — no CSV found")
            continue
        csv_path = csvs[0]

        n = 0
        with open(csv_path, newline='') as f:
            for row in csv.DictReader(f):
                ep = row.get('extracted_path', '').strip()
                if not ep or not os.path.exists(ep):
                    continue
                rows.append({
                    'extracted_path':  ep,
                    'loc_x':           loc_x,
                    'loc_y':           loc_y,
                    'displacement_mm': float(row['displacement_mm']),
                    'force_n':         float(row['force_n']),
                    'session':         session_dir,
                })
                n += 1
        print(f"  {session_dir}: {n} frames  (loc=({loc_x}, {loc_y}))")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    print(f"\nDataset: {len(df)} samples across {df['session'].nunique()} sessions")
    print(f"X range: {df['loc_x'].min()} to {df['loc_x'].max()} mm")
    print(f"Y range: {df['loc_y'].min()} to {df['loc_y'].max()} mm")
    print(f"Displacement range: {df['displacement_mm'].min():.3f} to {df['displacement_mm'].max():.3f} mm")
    print(f"Force range: {df['force_n'].min():.3f} to {df['force_n'].max():.3f} N")
    print(f"Output: {OUT_CSV}")


if __name__ == '__main__':
    main()
