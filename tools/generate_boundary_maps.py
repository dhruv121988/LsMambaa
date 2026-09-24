#!/usr/bin/env python3
"""
Generate boundary ground-truth maps from LoveDA segmentation masks.
===================================================================

Reads the *converted* (0-indexed) segmentation masks from ``masks_png_convert/``
and produces binary boundary maps in ``masks_png_boundary/``.

A pixel is marked as boundary (255) if it differs from at least one neighbour
in its neighbourhood (4-connected or 8-connected).  An optional morphological
dilation widens the boundary strip.

Ignore-index pixels (class 7 in converted LoveDA masks, value = num_classes)
are excluded: they are neither boundary nor interior.

Directory layout produced::

    data/LoveDA/
    ├── Train/
    │   ├── Urban/
    │   │   ├── masks_png_convert/   ← input  (existing)
    │   │   └── masks_png_boundary/  ← output (NEW)
    │   └── Rural/
    │       ├── masks_png_convert/
    │       └── masks_png_boundary/
    └── Val/
        ├── Urban/...
        └── Rural/...

Usage::

    python tools/generate_boundary_maps.py \\
        --data-root data/LoveDA \\
        --splits Train Val \\
        --regions Urban Rural \\
        --connectivity 8 \\
        --dilation 0 \\
        --input-dir masks_png_convert \\
        --output-dir masks_png_boundary

After generation the script prints boundary-pixel coverage statistics
(percentage of valid pixels that are boundary) for the whole dataset.
"""

import argparse
import glob
import multiprocessing as mp
import multiprocessing.pool as mpp
import os
import sys
import time
from functools import partial

import cv2
import numpy as np


# =====================================================================
# LoveDA constants
# =====================================================================

CLASSES = (
    "background", "building", "road", "water",
    "barren", "forest", "agricultural",
)
NUM_CLASSES = len(CLASSES)           # 7
IGNORE_INDEX = NUM_CLASSES           # 7  (in converted masks)


# =====================================================================
# Core boundary extraction
# =====================================================================

def extract_boundary(
    mask: np.ndarray,
    connectivity: int = 8,
    dilation: int = 0,
    ignore_index: int = IGNORE_INDEX,
) -> np.ndarray:
    """
    Detect class-transition edges in a segmentation mask.

    A pixel is a boundary pixel if at least one of its neighbours (in
    the chosen connectivity) belongs to a different valid class.

    Parameters
    ----------
    mask : np.ndarray, shape (H, W), dtype uint8
        Segmentation label map (0-indexed classes).
    connectivity : {4, 8}
        Neighbourhood connectivity.
    dilation : int
        Number of iterations of binary dilation applied to the raw
        boundary map (widens the boundary strip).  0 = no dilation.
    ignore_index : int
        Label value to treat as "don't care" — neither boundary nor interior.

    Returns
    -------
    boundary : np.ndarray, shape (H, W), dtype uint8
        255 at boundary pixels, 0 at interior / ignore pixels.
    """
    H, W = mask.shape
    boundary = np.zeros((H, W), dtype=np.uint8)

    # valid pixel mask (not ignore)
    valid = mask != ignore_index

    # offsets for the chosen connectivity
    if connectivity == 4:
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    elif connectivity == 8:
        offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            ( 0, -1),          ( 0, 1),
            ( 1, -1), ( 1, 0), ( 1, 1),
        ]
    else:
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")

    for dy, dx in offsets:
        # shifted mask (with border padding via np.roll + masking)
        shifted = np.roll(np.roll(mask, -dy, axis=0), -dx, axis=1)
        shifted_valid = np.roll(np.roll(valid.astype(np.uint8), -dy, axis=0),
                                -dx, axis=1).astype(bool)

        # A pixel is boundary if it differs from this neighbour,
        # AND both pixels are valid (not ignore).
        diff = (mask != shifted) & valid & shifted_valid
        boundary[diff] = 255

    # Handle border rows/cols: np.roll wraps around, which can create
    # false positives at image edges.  Zero out the wrapped borders.
    for dy, dx in offsets:
        if dy < 0:
            boundary[:abs(dy), :] &= 0 if abs(dy) <= 1 else boundary[:abs(dy), :]
            boundary[0, :] = 0
        if dy > 0:
            boundary[-dy:, :] = 0
        if dx < 0:
            boundary[:, :abs(dx)] = 0
        if dx > 0:
            boundary[:, -dx:] = 0

    # Optional dilation to thicken boundaries
    if dilation > 0:
        if connectivity == 4:
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        boundary = cv2.dilate(boundary, kernel, iterations=dilation)
        # Re-mask ignore regions after dilation
        boundary[~valid] = 0

    return boundary


# =====================================================================
# Parallel processing worker
# =====================================================================

def _process_one(
    args_tuple,
    connectivity: int,
    dilation: int,
    ignore_index: int,
):
    """Worker function for multiprocessing."""
    mask_path, output_path = args_tuple

    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        print(f"  [WARNING] Could not read: {mask_path}", file=sys.stderr)
        return 0, 0  # boundary_pixels, valid_pixels

    boundary = extract_boundary(
        mask, connectivity=connectivity,
        dilation=dilation, ignore_index=ignore_index,
    )
    cv2.imwrite(output_path, boundary)

    valid = mask != ignore_index
    n_boundary = int((boundary > 0).sum())
    n_valid = int(valid.sum())
    return n_boundary, n_valid


# =====================================================================
# Main
# =====================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate boundary GT maps from LoveDA segmentation masks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data-root", type=str, default="data/LoveDA",
        help="Root directory of the LoveDA dataset.",
    )
    p.add_argument(
        "--splits", nargs="+", default=["Train", "Val"],
        help="Dataset splits to process.",
    )
    p.add_argument(
        "--regions", nargs="+", default=["Urban", "Rural"],
        help="Region subfolders to process.",
    )
    p.add_argument(
        "--input-dir", type=str, default="masks_png_convert",
        help="Name of the input mask subfolder (must contain 0-indexed masks).",
    )
    p.add_argument(
        "--output-dir", type=str, default="masks_png_boundary",
        help="Name of the output boundary-map subfolder.",
    )
    p.add_argument(
        "--connectivity", type=int, default=8, choices=[4, 8],
        help="Pixel neighbourhood connectivity (4 or 8).",
    )
    p.add_argument(
        "--dilation", type=int, default=0,
        help="Number of morphological dilation iterations (0 = no dilation).",
    )
    p.add_argument(
        "--ignore-index", type=int, default=IGNORE_INDEX,
        help="Label value to ignore.",
    )
    p.add_argument(
        "--num-workers", type=int, default=0,
        help="Number of parallel workers (0 = auto = cpu_count).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    num_workers = args.num_workers if args.num_workers > 0 else mp.cpu_count()
    print("=" * 65)
    print("  Boundary Ground-Truth Generation for LoveDA")
    print("=" * 65)
    print(f"  data_root    : {args.data_root}")
    print(f"  splits       : {args.splits}")
    print(f"  regions      : {args.regions}")
    print(f"  input_dir    : {args.input_dir}")
    print(f"  output_dir   : {args.output_dir}")
    print(f"  connectivity : {args.connectivity}")
    print(f"  dilation     : {args.dilation}")
    print(f"  ignore_index : {args.ignore_index}")
    print(f"  num_workers  : {num_workers}")
    print("-" * 65)

    total_boundary_pixels = 0
    total_valid_pixels = 0
    total_images = 0

    per_split_stats = {}

    t_start = time.time()

    for split in args.splits:
        for region in args.regions:
            input_root = os.path.join(args.data_root, split, region, args.input_dir)
            output_root = os.path.join(args.data_root, split, region, args.output_dir)

            if not os.path.isdir(input_root):
                print(f"  [SKIP] Not found: {input_root}")
                continue

            os.makedirs(output_root, exist_ok=True)

            mask_paths = sorted(glob.glob(os.path.join(input_root, "*.png")))
            if not mask_paths:
                print(f"  [SKIP] No PNGs in: {input_root}")
                continue

            # Build (input, output) pairs
            pairs = []
            for mp_ in mask_paths:
                fname = os.path.basename(mp_)
                out_path = os.path.join(output_root, fname)
                pairs.append((mp_, out_path))

            # Process
            worker_fn = partial(
                _process_one,
                connectivity=args.connectivity,
                dilation=args.dilation,
                ignore_index=args.ignore_index,
            )

            print(f"\n  Processing {split}/{region}  ({len(pairs)} images) ...")

            if num_workers > 1:
                with mpp.Pool(processes=num_workers) as pool:
                    results = pool.map(worker_fn, pairs)
            else:
                results = [worker_fn(p) for p in pairs]

            # Accumulate stats
            split_bnd = sum(r[0] for r in results)
            split_valid = sum(r[1] for r in results)
            split_key = f"{split}/{region}"
            coverage = 100.0 * split_bnd / max(split_valid, 1)
            per_split_stats[split_key] = {
                "images": len(pairs),
                "boundary_pixels": split_bnd,
                "valid_pixels": split_valid,
                "coverage_pct": coverage,
            }

            total_boundary_pixels += split_bnd
            total_valid_pixels += split_valid
            total_images += len(pairs)

            print(f"    → {len(pairs)} maps written to {output_root}")
            print(f"    → boundary coverage: {coverage:.4f}%"
                  f"  ({split_bnd:,} / {split_valid:,} pixels)")

    t_elapsed = time.time() - t_start

    # ---- Summary ----
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(f"  {'Split/Region':<25s} {'Images':>8s} {'Coverage %':>12s}"
          f" {'Boundary px':>14s} {'Valid px':>14s}")
    print("  " + "-" * 63)
    for key, stats in per_split_stats.items():
        print(f"  {key:<25s} {stats['images']:>8d}"
              f" {stats['coverage_pct']:>11.4f}%"
              f" {stats['boundary_pixels']:>14,}"
              f" {stats['valid_pixels']:>14,}")
    print("  " + "-" * 63)

    overall_coverage = 100.0 * total_boundary_pixels / max(total_valid_pixels, 1)
    print(f"  {'TOTAL':<25s} {total_images:>8d}"
          f" {overall_coverage:>11.4f}%"
          f" {total_boundary_pixels:>14,}"
          f" {total_valid_pixels:>14,}")

    print(f"\n  Overall boundary-pixel coverage: {overall_coverage:.4f}%")
    print(f"  (Paper's reported figure: ~2.36% — compare above)")
    print(f"  Time elapsed: {t_elapsed:.1f}s")
    print("=" * 65)


if __name__ == "__main__":
    main()
