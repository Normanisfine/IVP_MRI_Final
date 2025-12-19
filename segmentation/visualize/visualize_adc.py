import argparse
import os
from typing import Optional

import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize ADC maps."
    )
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV file.")
    parser.add_argument("--id", type=str, required=True, default=24, help="Case ID to visualize.")
    parser.add_argument("--vis_path", type=str, required=True, help="Path to save the visualization figure.")
    parser.add_argument("--save_path", type=str, required=False, default=None, help="Path to save the figure and mask.")
    parser.add_argument("--all", action="store_true", help="Visualize all slices.")
    return parser.parse_args()


def imshow_with_mask(ax, img, mask, title):
    ax.axis("off")
    if img is None:
        ax.set_title(f"{title} (missing)")
        return
    print(img.shape, mask.shape)
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax.imshow(mask, cmap="winter", vmin=0, vmax=2, alpha=0.15)


def save_with_mask(img, mask, out_name):
    img_path = str(out_name) + ".npy"
    mask_path = str(out_name) + "_mask.npy"
    img = img.astype(np.float32)
    mask = mask.astype(np.float32)
    np.save(img_path, img)
    np.save(mask_path, mask)


def normalize(img):
    # Avoid division by zero if the image is constant
    img_min = img.min()
    img_max = img.max()
    if img_max == img_min:
        return np.zeros_like(img, dtype=np.float32)
    return (img - img_min) / (img_max - img_min)


def visualize_adc(row: pd.Series, out_dir: str, csv_base_dir: str, save_path: Optional[str] = None) -> None:
    paths = {
        "adc": os.path.normpath(os.path.join(csv_base_dir, row.get("adc", None))),
        "adc_tumor": os.path.normpath(os.path.join(csv_base_dir, row.get("adc_tumor_reader1", None)))
    }

    try:
        adc = nib.load(paths["adc"]).get_fdata()
        adc_tumor = nib.load(paths["adc_tumor"]).get_fdata()
    except Exception as e:
        print(f"Error loading ADC image: {e}")
        print(f"ADC image path: {paths['adc']}")
        return

    rows = np.ceil(adc.shape[2] / 6).astype(int)

    fig, ax = plt.subplots(rows, 6, figsize=(10, 10))

    if save_path is not None:
        save_dir = os.path.join(save_path, f"{row['ID']}_slice")
        os.makedirs(save_dir, exist_ok=True)

    for i in range(adc.shape[2]):
        imshow_with_mask(ax[i // 6, i % 6], normalize(adc[:, :, i]), adc_tumor[:, :, i], f"ADC slice {i}")

        if save_path is not None:
            save_with_mask(adc[:, :, i], adc_tumor[:, :, i], os.path.join(save_dir, f"{row['ID']}_slice{i}"))

    for j in range(adc.shape[2] % 6, 6):
        ax[rows - 1, j].axis("off")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"ADC_ID{row['ID']}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)
    csv_base_dir = os.path.dirname(os.path.abspath(args.csv))
    matches = df[df["ID"].astype(str) == str(args.id)]
    os.makedirs(args.vis_path, exist_ok=True)
    if args.save_path is not None:
        os.makedirs(args.save_path, exist_ok=True)
    if matches.empty:
        raise ValueError(f"ID {args.id} not found. Available IDs include: {df['ID'].head(10).tolist()} ...")
    row = matches.iloc[0]
    if args.all:
        for _, row in df.iterrows():
            visualize_adc(row, args.vis_path, csv_base_dir, args.save_path)
    else:
        visualize_adc(row, args.vis_path, csv_base_dir, args.save_path)


if __name__ == "__main__":
    main()


