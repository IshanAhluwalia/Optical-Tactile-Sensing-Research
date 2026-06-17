"""
Plot force vs displacement curves for all y positions, one PNG per x value.
Output saved to: 3D Recon/force_vs_displacement_x{X}.png
"""

import os
import matplotlib
matplotlib.use("Agg")
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset", "output")
OUT_DIR  = os.path.join(BASE_DIR, "3D Recon")
os.makedirs(OUT_DIR, exist_ok=True)

Y_VALS = list(range(0, 19, 2))  # 0, 2, 4, ..., 18

# Collect all unique x values present in the dataset folder
x_vals = sorted(set(
    int(d.split("_")[0][1:])
    for d in os.listdir(DATA_DIR)
    if d.startswith("x") and "_y" in d
))

cmap = matplotlib.colormaps["plasma"].resampled(len(Y_VALS))

for x in x_vals:
    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = 0

    for i, y in enumerate(Y_VALS):
        session  = f"x{x}_y{y}"
        csv_path = os.path.join(DATA_DIR, session, f"{session}.csv")
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        df["force_smooth"] = (
            df["force_n"].rolling(window=5, center=True, min_periods=1).mean()
        )

        ax.plot(
            df["displacement_mm"],
            df["force_smooth"],
            color=cmap(i),
            linewidth=1.5,
            label=f"y = {y} mm",
        )
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        continue

    ax.set_xlabel("Displacement (mm)", fontsize=13)
    ax.set_ylabel("Force (N)", fontsize=13)
    ax.set_title(f"Force vs Displacement — x = {x}  (y = 0–18 mm)", fontsize=14)
    ax.legend(title="Strip position (y)", bbox_to_anchor=(1.02, 1),
              loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"force_vs_displacement_x{x}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

print("Done.")
