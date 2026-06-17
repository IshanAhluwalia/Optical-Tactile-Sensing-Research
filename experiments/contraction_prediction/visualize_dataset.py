
"""
Generates a contact sheet showing sample frames with their displacement labels.
Opens the result automatically in Preview.
"""

import csv
import os
import subprocess
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

DATASET_CSV = os.path.join(os.path.dirname(__file__), "dataset", "dataset.csv")
OUTPUT_IMG  = os.path.join(os.path.dirname(__file__), "dataset", "preview.png")

# Every 5 seconds: frames at t=0, 5, 10, 15, 20, 25, 30
STEP_SEC = 5
with open(DATASET_CSV, newline='') as f:
    all_rows = list(csv.DictReader(f))

targets = np.arange(0, 31, STEP_SEC)  # 0, 5, 10, 15, 20, 25, 30
samples = []
for t in targets:
    closest = min(all_rows, key=lambda r: abs(float(r['elapsed_sec']) - t))
    samples.append(closest)

n = len(samples)
cols = 4
rows_count = int(np.ceil(n / cols))

fig, axes = plt.subplots(rows_count, cols, figsize=(cols * 5, rows_count * 4))
fig.patch.set_facecolor('#1a1a1a')
axes = axes.flatten()

for i, row in enumerate(samples):
    img = cv2.imread(row['image_path'])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    axes[i].imshow(img)
    axes[i].axis('off')
    axes[i].set_title(
        f"Frame {row['frame']}  |  t = {float(row['elapsed_sec']):.1f}s\n"
        f"Crosshead: {float(row['crosshead_mm']):.2f} mm  |  Load: {float(row['load_N']):.2f} N",
        color='white', fontsize=9, pad=6
    )

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Skin Contraction Dataset — Every 5 Seconds", color='white', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(OUTPUT_IMG, dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
plt.close()

print(f"Saved: {OUTPUT_IMG}")
subprocess.run(["open", OUTPUT_IMG])
