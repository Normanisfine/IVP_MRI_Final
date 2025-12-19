import argparse
import os
from typing import Dict, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize T2-weighted MRI images."
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
    return (img - img.min()) / (img.max() - img.min())


def visualize_t2(row: pd.Series, out_dir: str, csv_base_dir: str, save_path: Optional[str] = None) -> None:
    paths = {
        "t2": os.path.normpath(os.path.join(csv_base_dir, row.get("t2", None))),
        "t2_gland": os.path.normpath(os.path.join(csv_base_dir, row.get("t2_anatomy_reader1", None))),
        "t2_tumor": os.path.normpath(os.path.join(csv_base_dir, row.get("t2_tumor_reader1", None)))
    }

    try:
        # t2 = nib.load(paths["t2"].replace(".nii.gz", "_resized.nii.gz")).get_fdata()
        # t2_gland = nib.load(paths["t2_gland"].replace(".nii.gz", "_resized.nii.gz")).get_fdata()
        t2 = nib.load(paths["t2"]).get_fdata()
        t2_gland = nib.load(paths["t2_gland"]).get_fdata()
        t2_tumor = nib.load(paths["t2_tumor"]).get_fdata()
    except Exception as e:
        print(f"Error loading T2 image: {e}")
        print(f"T2 image path: {paths['t2']}")
        return

    # print(f"T2 shape: {t2.shape}")
    rows = np.ceil(t2.shape[2] / 6).astype(int)

    fig, ax = plt.subplots(rows, 6, figsize=(10, 10))

    if save_path is not None:
        save_dir = os.path.join(save_path, f"{row['ID']}_slice")
        os.makedirs(save_dir, exist_ok=True)

    for i in range(t2.shape[2]):
        imshow_with_mask(ax[i//6, i%6], normalize(t2[:,:,i]), t2_gland[:,:,i], f"T2 slice {i}")
        imshow_with_mask(ax[i//6, i%6], normalize(t2[:,:,i]), t2_tumor[:,:,i], f"T2 slice {i}")
        # print(t2_gland[:,:,i].max())
        # print(f"count: {(t2_gland[:,:,i] == 2.0).sum()}")

        # if t2_gland[:,:,i].sum() > 0 and save_path is not None:
        if save_path is not None:
            save_with_mask(t2[:,:,i], t2_gland[:,:,i], os.path.join(save_dir, f"{row['ID']}_slice{i}"))
            


    for j in range(t2.shape[2] % 6, 6):
        ax[rows - 1, j].axis("off")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"T2_ID{row['ID']}.png")
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
        for index, row in df.iterrows():
            visualize_t2(row, args.vis_path, csv_base_dir, args.save_path)
    else:
        visualize_t2(row, args.vis_path, csv_base_dir, args.save_path)


if __name__ == "__main__":
    main()