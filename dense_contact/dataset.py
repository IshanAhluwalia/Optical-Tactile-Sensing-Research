"""
PyTorch dataset for dense contact estimation.

For each frame the dataset produces:
  - image          : (1, 224, 224) normalised grayscale tensor
  - contact_map    : (H, W) Gaussian blob in [0, 1], peak = 1 at contact centre
  - depth_map      : (H, W) Hertz depth profile in [0, 1], normalised by disp_max
  - pressure_map   : (H, W) Hertz pressure profile in [0, 1], normalised by pressure_max
  - displacement   : scalar in [0, 1], normalised total indentation depth
  - force          : scalar in [0, 1], normalised total force
  - displacement_raw / force_raw / loc_x / loc_y : raw mm / N values for evaluation
"""

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# ── Sensor grid ────────────────────────────────────────────────────────────────
# Actual mm coordinates of each sampled position on the sensor surface.
# Full continuous grid: 138 to 210 mm every 2 mm (37 columns), 0 to 16 mm (9 rows).
GRID_X = np.array(
    [138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158, 160, 162, 164,
     166, 168, 170, 172, 174, 176, 178, 180, 182, 184,
     186, 188, 190, 192, 194, 196, 198, 200, 202, 204, 206, 208, 210],
    dtype=np.float32,
)
GRID_Y = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16], dtype=np.float32)

GRID_H = len(GRID_Y)   # 9
GRID_W = len(GRID_X)   # 37

R_INDENTOR = 10.0  # mm  (1 cm spherical tip)

# Pre-build meshgrid once (H, W) — reused in every label call
_XX, _YY = np.meshgrid(GRID_X, GRID_Y)  # both (H, W)


# ── Hertzian pseudo-label generation ──────────────────────────────────────────

def make_hertz_labels(
    loc_x: float,
    loc_y: float,
    delta: float,
    force: float,
    disp_max: float,
    pressure_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate spatially-resolved pseudo-labels for one frame.

    contact_map : Gaussian blob with sigma = contact radius a.
                  Peak = 1 at contact centre, decays to ~0 outside contact zone.
    depth_map   : Hertz surface deformation  d(r) = delta * max(0, 1 - r²/a²)
                  Normalised to [0, 1] by disp_max.
    pressure_map: Hertz pressure distribution p(r) = p0 * max(0, sqrt(1 - r²/a²))
                  Normalised to [0, 1] by pressure_max.

    When delta <= 0 all maps are zero (no contact yet).
    """
    H, W = GRID_H, GRID_W
    contact_map  = np.zeros((H, W), dtype=np.float32)
    depth_map    = np.zeros((H, W), dtype=np.float32)
    pressure_map = np.zeros((H, W), dtype=np.float32)

    if delta <= 0.0 or force <= 0.0:
        return contact_map, depth_map, pressure_map

    a  = np.sqrt(R_INDENTOR * delta)       # contact radius (mm)
    a2 = a * a
    p0 = 3.0 * force / (2.0 * np.pi * a2)  # peak pressure (N/mm²)

    r2 = (_XX - loc_x) ** 2 + (_YY - loc_y) ** 2  # (H, W)

    # Contact probability: Gaussian with sigma = a (smooth, extends slightly outside)
    contact_map = np.exp(-r2 / (2.0 * a2)).astype(np.float32)

    # Depth and pressure: Hertz profile, zero outside contact circle
    inside = r2 < a2
    r2_norm = np.where(inside, r2 / a2, 1.0)

    depth_map[inside]    = (delta * (1.0 - r2_norm[inside]) / disp_max)
    pressure_map[inside] = (p0 * np.sqrt(1.0 - r2_norm[inside]) / pressure_max)

    return contact_map, depth_map, pressure_map


# ── Dataset ───────────────────────────────────────────────────────────────────
# Grayscale normalisation stats computed from 500 sampled images in the dataset.
_GRAY_MEAN = [0.4513]
_GRAY_STD  = [0.2898]

_TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(8),
    transforms.ToTensor(),                          # → (1, H, W) in [0, 1]
    transforms.Normalize(_GRAY_MEAN, _GRAY_STD),
])

_VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(_GRAY_MEAN, _GRAY_STD),
])


class TactileDataset(Dataset):
    """
    Args:
        csv_path : path to dataset.csv (built by build_images.py)
        sessions : list of session strings to include (e.g. ['x144_y8', ...])
        train    : whether to apply training augmentations
        stride   : keep every Nth frame per session (default 5).
                   Adjacent frames in a 60s press differ by ~0.022mm — highly
                   redundant. stride=5 keeps ~90 frames/session while preserving
                   the full 0→10mm indentation curve.
    """

    def __init__(self, csv_path: str, sessions: list[str], train: bool = True, stride: int = 5):
        # Load full CSV to compute global normalisation stats
        full_df = pd.read_csv(csv_path)
        self.disp_max = float(full_df['displacement_mm'].max())
        self.force_max = float(full_df['force_n'].max())

        # pressure_max: highest Hertz peak pressure that actually occurs in the data.
        # Peak pressure p0 = 3F/(2πa²) = 3F/(2πRδ), which is largest at shallow
        # indentations where the contact area is still small but force is non-trivial.
        # Computing it analytically from (max_delta, max_force) gives the wrong answer.
        valid = full_df[(full_df['displacement_mm'] > 0) & (full_df['force_n'] > 0)]
        a_vals  = np.sqrt(R_INDENTOR * valid['displacement_mm'].values)
        p0_vals = 3.0 * valid['force_n'].values / (2.0 * np.pi * a_vals ** 2)
        self.pressure_max = float(p0_vals.max())

        # Subset to requested sessions, then subsample every `stride` frames
        subset = full_df[full_df['session'].isin(sessions)]
        subset = subset.groupby('session', group_keys=False).apply(
            lambda g: g.iloc[::stride]
        )
        self.df = subset.reset_index(drop=True)
        self.transform = _TRAIN_TRANSFORM if train else _VAL_TRANSFORM

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        img = Image.open(row['image_path']).convert('L')  # grayscale
        img = self.transform(img)

        delta = float(row['displacement_mm'])
        force = float(row['force_n'])
        loc_x = float(row['loc_x'])
        loc_y = float(row['loc_y'])

        contact_map, depth_map, pressure_map = make_hertz_labels(
            loc_x, loc_y, delta, force,
            self.disp_max, self.pressure_max,
        )

        return {
            'image':            img,
            'contact_map':      torch.from_numpy(contact_map),
            'depth_map':        torch.from_numpy(depth_map),
            'pressure_map':     torch.from_numpy(pressure_map),
            # Normalised scalars for training
            'displacement':     torch.tensor(delta / self.disp_max,  dtype=torch.float32),
            'force':            torch.tensor(force / self.force_max, dtype=torch.float32),
            # Raw values for evaluation metrics
            'displacement_raw': torch.tensor(delta,  dtype=torch.float32),
            'force_raw':        torch.tensor(force,  dtype=torch.float32),
            'loc_x':            torch.tensor(loc_x,  dtype=torch.float32),
            'loc_y':            torch.tensor(loc_y,  dtype=torch.float32),
        }
