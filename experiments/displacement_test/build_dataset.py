"""
Dataset Builder — Displacement Test
Correlates video frames with MTS crosshead displacement data.

Video starts 1 second AFTER the DAQ, so:
    DAQ lookup time = video_elapsed_time + VIDEO_OFFSET

Train split : first 1:19 (79s) of video  → dataset/train/
Test  split : remaining frames            → dataset/test/  (no labels)

Outputs:
    dataset/train/frames/   - JPEG frames
    dataset/train/dataset.csv - frame, elapsed_sec, crosshead_mm, load_N, cycle, image_path
    dataset/test/frames/    - JPEG frames (unlabeled, for evaluation)
"""

import csv
import os
import cv2
import numpy as np

# ── Config ───────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(__file__)
RAW_DIR    = os.path.join(BASE_DIR, "raw data")
VIDEO_PATH = os.path.join(RAW_DIR, "video_20260403_150500.mp4")
DAQ_PATH   = os.path.join(RAW_DIR, "DAQ- Crosshead, … - (Timed).csv")

TRAIN_DURATION = 79.0   # seconds of video used for training (1:19)
VIDEO_OFFSET   = 1.0    # DAQ starts 1s before video; daq_time = video_time + offset
DAQ_SKIP_ROWS  = 8      # metadata/header rows at top of DAQ file

OUT_TRAIN_FRAMES = os.path.join(BASE_DIR, "dataset", "train", "frames")
OUT_TEST_FRAMES  = os.path.join(BASE_DIR, "dataset", "test",  "frames")
OUT_CSV          = os.path.join(BASE_DIR, "dataset", "train", "dataset.csv")
# ─────────────────────────────────────────────────────────────────────────────


def load_daq(path):
    time, crosshead, load, cycle = [], [], [], []
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        for _ in range(DAQ_SKIP_ROWS):
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


def main():
    os.makedirs(OUT_TRAIN_FRAMES, exist_ok=True)
    os.makedirs(OUT_TEST_FRAMES,  exist_ok=True)

    # Load DAQ
    daq_time, daq_crosshead, daq_load, daq_cycle = load_daq(DAQ_PATH)
    print(f"DAQ loaded: {len(daq_time)} rows  ({daq_time[0]:.2f}s – {daq_time[-1]:.2f}s)")

    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration     = total_frames / fps
    print(f"Video: {fps:.1f} FPS, {total_frames} frames, {duration:.1f}s")

    train_cutoff = int(TRAIN_DURATION * fps)  # last train frame index (exclusive)
    print(f"Train frames: 0 – {train_cutoff - 1}  ({TRAIN_DURATION:.0f}s)")
    print(f"Test  frames: {train_cutoff} – {total_frames - 1}  ({duration - TRAIN_DURATION:.0f}s)")

    fieldnames = ['frame', 'elapsed_sec', 'crosshead_mm', 'load_N', 'cycle', 'image_path']
    train_rows = []

    for frame_idx in range(total_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, img = cap.read()
        if not ret:
            print(f"  Warning: could not read frame {frame_idx}, skipping.")
            continue

        video_time = frame_idx / fps
        is_train   = frame_idx < train_cutoff

        if is_train:
            img_name = f"frame_{frame_idx:05d}.jpg"
            img_path = os.path.join(OUT_TRAIN_FRAMES, img_name)
            cv2.imwrite(img_path, img)

            daq_t      = video_time + VIDEO_OFFSET
            crosshead  = float(np.interp(daq_t, daq_time, daq_crosshead))
            load       = float(np.interp(daq_t, daq_time, daq_load))
            nearest    = int(np.argmin(np.abs(daq_time - daq_t)))
            cycle      = int(daq_cycle[nearest])

            train_rows.append({
                'frame':        frame_idx,
                'elapsed_sec':  round(video_time, 4),
                'crosshead_mm': round(crosshead, 6),
                'load_N':       round(load, 6),
                'cycle':        cycle,
                'image_path':   img_path,
            })
        else:
            img_name = f"frame_{frame_idx:05d}.jpg"
            cv2.imwrite(os.path.join(OUT_TEST_FRAMES, img_name), img)

    cap.release()

    # Write train CSV
    with open(OUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(train_rows)

    print(f"\nDone.")
    print(f"  Train: {len(train_rows)} frames + labels → {OUT_CSV}")
    print(f"  Test : {total_frames - train_cutoff} frames (unlabeled) → {OUT_TEST_FRAMES}")


if __name__ == "__main__":
    main()
