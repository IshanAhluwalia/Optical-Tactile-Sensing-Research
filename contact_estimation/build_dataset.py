"""
Aggregate all recorded sessions into a single CSV for training.
Parses (loc_x, loc_y) from session folder names.

Usage:
    python contact_estimation/build_dataset.py

Output:
    contact_estimation/dataset.csv
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

        m = re.match(r'\(?(-?\d+),\s*(-?\d+)\)?', session_dir)
        if not m:
            print(f"  skip {session_dir!r} — no coordinates in name")
            continue
        loc_x, loc_y = int(m.group(1)), int(m.group(2))

        # Find the CSV inside the folder (name may differ from folder)
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
                    'extracted_path': ep,
                    'loc_x':          loc_x,
                    'loc_y':          loc_y,
                    'displacement_mm': float(row['displacement_mm']),
                    'force_n':         float(row['force_n']),
                    'session':         session_dir,
                })
                n += 1
        print(f"  {session_dir}: {n} frames  (loc=({loc_x}, {loc_y}))")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nDataset: {len(df)} samples, {df['session'].nunique()} sessions → {OUT_CSV}")
    print(df.groupby('session')[['loc_y', 'displacement_mm', 'force_n']].agg(['min', 'max']).to_string())


if __name__ == '__main__':
    main()
