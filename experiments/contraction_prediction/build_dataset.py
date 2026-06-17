"""
Dataset Builder
Correlates the first 30 seconds of each video with MTS displacement/load data.
Appends to any existing dataset.csv rather than overwriting it.
Outputs:
  - dataset/frames/     : extracted JPEG frames
  - dataset/dataset.csv : frame, elapsed_sec, crosshead_mm, load_N, cycle, image_path
"""

import csv
import cv2
import os
import numpy as np
from extract_pattern import extract_pattern

# ── Config ──────────────────────────────────────────────────────────────────
VIDEOS = [
    {
        "video": "/Users/ishanahluwalia/Desktop/ContractionRealVideo.mp4",
        "mts":   "/Users/ishanahluwalia/Desktop/Contract_dataset.csv",
        "prefix": "v1",
    },
    {
        "video": "/Users/ishanahluwalia/Desktop/skincontract60mmVideo2.mp4",
        "mts":   "/Users/ishanahluwalia/Desktop/Test Run 2 03-3-26 15 25 30 PM/DAQ- Crosshead, … - (Timed).txt",
        "prefix": "v2",
    },
    {
        "video": "/Users/ishanahluwalia/Desktop/skincontract60mmVideo3.mp4",
        "mts":   "/Users/ishanahluwalia/Desktop/Test Run 3 03-3-26 15 29 28 PM/DAQ- Crosshead, … - (Timed).csv",
        "prefix": "v3",
    },
]

OUTPUT_DIR    = os.path.join(os.path.dirname(__file__), "dataset")
MTS_SKIP_ROWS = 8      # metadata/header rows at top of each MTS file
CLIP_DURATION = 30.0   # seconds of video to use
# ────────────────────────────────────────────────────────────────────────────


def load_mts(mts_path):
    """Load MTS CSV/TXT file, auto-detecting tab vs comma delimiter."""
    delimiter = '\t' if mts_path.endswith('.txt') else ','
    time, crosshead, load, cycle = [], [], [], []
    with open(mts_path, newline='', encoding='utf-8-sig') as f:
        reader = csv.reader(f, delimiter=delimiter)
        for _ in range(MTS_SKIP_ROWS):
            next(reader)
        for row in reader:
            if not row or not row[0].strip():
                continue
            try:
                cycle.append(int(float(row[0])))
                crosshead.append(float(row[1]))
                load.append(float(row[2]))
                time.append(float(row[3]))
            except (ValueError, IndexError):
                continue
    return (np.array(time), np.array(crosshead),
            np.array(load), np.array(cycle, dtype=int))


def process_video(video_path, mts_path, prefix, frames_dir):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\nVideo:    {video_path}")
    print(f"FPS: {fps:.2f}  |  Total frames: {total_frames}")
    print(f"MTS data: {mts_path}")

    frames = [{'frame': i, 'elapsed': i / fps}
              for i in range(total_frames)
              if i / fps <= CLIP_DURATION]
    print(f"Frames in first {CLIP_DURATION:.0f}s: {len(frames)}")

    mts_time, mts_crosshead, mts_load, mts_cycle = load_mts(mts_path)
    print(f"MTS rows loaded: {len(mts_time)}  ({mts_time[0]:.2f}s – {mts_time[-1]:.2f}s)")

    out_rows = []
    for fr in frames:
        elapsed   = fr['elapsed']
        crosshead = float(np.interp(elapsed, mts_time, mts_crosshead))
        load      = float(np.interp(elapsed, mts_time, mts_load))
        cycle     = int(mts_cycle[int(np.argmin(np.abs(mts_time - elapsed)))])

        cap.set(cv2.CAP_PROP_POS_FRAMES, fr['frame'])
        ret, img = cap.read()
        if not ret:
            print(f"  Warning: could not read frame {fr['frame']}, skipping.")
            continue

        img_name = f"{prefix}_frame_{fr['frame']:05d}.jpg"
        img_path = os.path.join(frames_dir, img_name)
        cv2.imwrite(img_path, extract_pattern(img))

        out_rows.append({
            'frame':        fr['frame'],
            'elapsed_sec':  round(elapsed, 4),
            'crosshead_mm': round(crosshead, 6),
            'load_N':       round(load, 6),
            'cycle':        cycle,
            'image_path':   img_path,
        })

    cap.release()
    return out_rows


# ── Main ─────────────────────────────────────────────────────────────────────
frames_dir = os.path.join(OUTPUT_DIR, "frames")
os.makedirs(frames_dir, exist_ok=True)

out_csv    = os.path.join(OUTPUT_DIR, "dataset.csv")
fieldnames = ['frame', 'elapsed_sec', 'crosshead_mm', 'load_N', 'cycle', 'image_path']

# Determine which prefixes are already in the dataset to skip re-processing
existing_prefixes = set()
if os.path.isfile(out_csv):
    with open(out_csv, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img = os.path.basename(row.get('image_path', ''))
            if '_frame_' in img:
                existing_prefixes.add(img.split('_frame_')[0])
    print(f"Existing dataset found. Prefixes already processed: {existing_prefixes}")

total_new = 0
for entry in VIDEOS:
    prefix = entry['prefix']
    if prefix in existing_prefixes:
        print(f"\nSkipping {prefix} — already in dataset.")
        continue

    rows = process_video(entry['video'], entry['mts'], prefix, frames_dir)

    # Append to CSV (write header only if file doesn't exist yet)
    write_header = not os.path.isfile(out_csv)
    with open(out_csv, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    print(f"  Appended {len(rows)} rows for {prefix}.")
    total_new += len(rows)

print(f"\nDone. {total_new} new rows added to {out_csv}")
