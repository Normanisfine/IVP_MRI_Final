import argparse
import os
from typing import Dict, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch visualize overlays of GT and prediction masks on MRI images."
    )
    parser.add_argument("--out", type=str, required=True, help="Output directory to save figures.")
    parser.add_argument("--gt_dir", type=str, required=True, help="Directory with ground-truth .nii.gz masks.")
    parser.add_argument("--pred_dir", type=str, required=True, help="Directory with predicted .nii.gz masks.")
    parser.add_argument("--img_dir", type=str, required=True, help="Directory with base .nii.gz images.")
    parser.add_argument("--num_classes", type=int, required=True, help="Total number of classes including background (0).")

    return parser.parse_args()


def imshow_with_mask(ax, img, mask, title, num_classes: int):
    ax.axis("off")
    if img is None:
        ax.set_title(f"{title} (missing)")
        return
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    # Discrete colormap: label 0 transparent, 1..(num_classes-1) colored
    colors = [(0.0, 0.0, 0.0, 0.0)]
    base_cmap = plt.get_cmap("tab20")
    for i in range(1, num_classes):
        rgba = base_cmap(float(i % base_cmap.N) / base_cmap.N)
        colors.append((rgba[0], rgba[1], rgba[2], 0.6))
    cmap = mcolors.ListedColormap(colors)
    boundaries = [-0.5] + [j + 0.5 for j in range(num_classes)]
    norm = mcolors.BoundaryNorm(boundaries, cmap.N)
    ax.imshow(mask, cmap=cmap, norm=norm, interpolation="nearest")


def normalize(img):
    ptp = img.max() - img.min()
    return (img - img.min()) / ptp if ptp != 0 else np.zeros_like(img)


def visualize_case(img_path: str, gt_path: str, pred_path: str, out_dir: str, name: str, num_classes: int) -> None:
    try:
        base_image_data = nib.load(img_path).get_fdata()
    except Exception as e:
        print(f"Error loading base image: {e}")
        print(f"Image path: {img_path}")
        return

    try:
        gt_data = nib.load(gt_path).get_fdata()
    except Exception as e:
        print(f"Error loading GT mask: {e}")
        print(f"GT path: {gt_path}")
        return

    try:
        pred_data = nib.load(pred_path).get_fdata()
    except Exception as e:
        print(f"Error loading Pred mask: {e}")
        print(f"Pred path: {pred_path}")
        return

    if base_image_data.shape != gt_data.shape or base_image_data.shape != pred_data.shape:
        print(f"Shape mismatch for {name}: img {base_image_data.shape}, gt {gt_data.shape}, pred {pred_data.shape}")
        return

    gt_data = gt_data.astype(int)
    pred_data = pred_data.astype(int)

    num_slices = base_image_data.shape[2]
    pairs_per_row = 6  # 6 slices per row, each slice uses two panels (GT, Pred)
    rows = np.ceil(num_slices / pairs_per_row).astype(int)
    cols = pairs_per_row * 2
    fig, ax = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5), squeeze=False)

    # Label each column: left=GT, right=Pred
    for p in range(pairs_per_row):
        c_left = p * 2
        c_right = c_left + 1
        if c_left < cols:
            ax[0, c_left].set_title("GT")
        if c_right < cols:
            ax[0, c_right].set_title("Pred")

    for i in range(num_slices):
        r = i // pairs_per_row
        c_left = (i % pairs_per_row) * 2
        c_right = c_left + 1
        imshow_with_mask(
            ax[r, c_left],
            normalize(base_image_data[:, :, i]),
            gt_data[:, :, i],
            f"Slice {i} - GT",
            num_classes,
        )
        imshow_with_mask(
            ax[r, c_right],
            normalize(base_image_data[:, :, i]),
            pred_data[:, :, i],
            f"Slice {i} - Pred",
            num_classes,
        )

    used_panels = num_slices * 2
    num_panels = rows * cols
    for k in range(used_panels, num_panels):
        r = k // cols
        c = k % cols
        ax[r, c].axis("off")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{name}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def collect_common_cases(gt_dir: str, pred_dir: str, img_dir: str):
    gt_files = {f[:12] for f in os.listdir(gt_dir) if f.endswith(".nii.gz")}
    pred_files = {f[:12] for f in os.listdir(pred_dir) if f.endswith(".nii.gz")}
    img_files = {f[:12] for f in os.listdir(img_dir) if f.endswith("_0000.nii.gz")}
    common = sorted(gt_files.intersection(pred_files).intersection(img_files))
    if not common:
        print("No common .nii.gz files found across gt/pred/img directories.")
    return common


def main() -> None:
    args = parse_args()
    cases = collect_common_cases(args.gt_dir, args.pred_dir, args.img_dir)
    for fname in cases:
        fname = f"{fname}_0000.nii.gz"
        name = os.path.splitext(os.path.splitext(fname)[0])[0]
        img_path = os.path.join(args.img_dir, fname)
        gt_path = os.path.join(args.gt_dir, fname[:12]+".nii.gz")
        pred_path = os.path.join(args.pred_dir, fname[:12]+".nii.gz")
        visualize_case(img_path, gt_path, pred_path, args.out, name, args.num_classes)

if __name__ == "__main__":
    main()