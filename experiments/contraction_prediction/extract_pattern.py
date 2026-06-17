"""
extract_pattern.py — Extract the speckle/ink pattern from a tissue image.

The adaptive threshold isolates the dark ink lines on the lighter tissue,
stripping out the background (grips, glare, lighting variation).

Can be used as a module:
    from extract_pattern import extract_pattern
    pattern_bgr = extract_pattern(bgr_frame)

Or run directly to preview extraction on any image:
    python extract_pattern.py path/to/image.jpg
    python extract_pattern.py path/to/image.jpg --save output.jpg
"""

import argparse
import os
import sys

import cv2
import numpy as np


def extract_pattern(bgr_frame: np.ndarray) -> np.ndarray:
    """
    Extract the speckle pattern from a BGR frame.

    Args:
        bgr_frame: OpenCV BGR image (any size).

    Returns:
        BGR image of the same size — white pattern lines on black background.
    """
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold: finds dark ink relative to local neighbourhood,
    # so it handles uneven lighting across the frame automatically.
    pattern = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,   # dark ink → white, background → black
        blockSize=31,
        C=8,
    )

    # Remove small noise specks (keep only ink-sized connected regions)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    pattern = cv2.morphologyEx(pattern, cv2.MORPH_OPEN, kernel)

    return cv2.cvtColor(pattern, cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------------------------
# CLI — preview / save extraction on a single image
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Extract speckle pattern from an image.")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--save", default=None, help="Path to save extracted pattern (optional)")
    parser.add_argument("--side-by-side", action="store_true",
                        help="Show original and extracted side by side")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print(f"Error: file not found: {args.image}")
        sys.exit(1)

    img = cv2.imread(args.image)
    if img is None:
        print(f"Error: could not read image: {args.image}")
        sys.exit(1)

    pattern = extract_pattern(img)
    coverage = round((pattern[:, :, 0] > 0).mean() * 100, 1)
    print(f"Image   : {args.image}  ({img.shape[1]}×{img.shape[0]})")
    print(f"Pattern coverage: {coverage}%")

    if args.save:
        cv2.imwrite(args.save, pattern)
        print(f"Saved   : {args.save}")

    if args.side_by_side:
        divider = np.full((img.shape[0], 4, 3), 180, dtype=np.uint8)
        display = np.hstack([img, divider, pattern])
    else:
        display = pattern

    cv2.imshow("Pattern Extraction (any key to close)", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
