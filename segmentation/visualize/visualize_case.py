import argparse
import os
from typing import Dict, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize prostate MRI modalities (T2, ADC, DWI) and overlay available masks "
            "for a single case from a CSV file."
        )
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to CSV with columns: ID,t2,adc,dwi,t2_anatomy_reader1,t2_tumor_reader1,adc_tumor_reader1,t2_anatomy_reader2,adc_tumor_reader2",
    )
    parser.add_argument(
        "--id",
        type=str,
        default=None,
        help="Case ID to visualize. If not provided, uses --row index.",
    )
    parser.add_argument(
        "--row",
        type=int,
        default=0,
        help="Row index to visualize if --id not provided (0-based).",
    )
    parser.add_argument(
        "--slice",
        type=int,
        default=None,
        help="Axial slice index to display. Default: middle slice.",
    )
    parser.add_argument(
        "--dwi-volume",
        type=int,
        default=None,
        help=(
            "DWI volume index to display for 4D DWI. Default: last volume if 4D; "
            "ignored for 3D volumes."
        ),
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Path to save the figure (e.g., /tmp/vis.png). If not set, shows the plot.",
    )
    return parser.parse_args()


def robust_normalize(image: np.ndarray, lower_pct: float = 1.0, upper_pct: float = 99.0) -> np.ndarray:
    finite_vals = image[np.isfinite(image)]
    if finite_vals.size == 0:
        return np.zeros_like(image)
    low = np.percentile(finite_vals, lower_pct)
    high = np.percentile(finite_vals, upper_pct)
    if high <= low:
        high = low + 1e-6
    image_clipped = np.clip(image, low, high)
    return (image_clipped - low) / (high - low + 1e-12)


def load_nifti(path: Optional[str]) -> Optional[np.ndarray]:
    if path is None or (isinstance(path, float) and np.isnan(path)):
        return None
    if not os.path.exists(path):
        return None
    data = nib.load(path).get_fdata()
    return data.astype(np.float32)


def select_axial_slice(volume: np.ndarray, index: Optional[int] = None) -> Tuple[np.ndarray, int]:
    if volume.ndim < 3:
        raise ValueError("Volume must be at least 3D")
    z_dim = volume.shape[2]
    z_index = z_dim // 2 if index is None else int(np.clip(index, 0, z_dim - 1))
    slice_2d = volume[:, :, z_index]
    return slice_2d, z_index


def select_dwi_volume(dwi: np.ndarray, volume_index: Optional[int]) -> np.ndarray:
    if dwi.ndim == 4:
        vdim = dwi.shape[3]
        vidx = vdim - 1 if volume_index is None else int(np.clip(volume_index, 0, vdim - 1))
        return dwi[:, :, :, vidx]
    return dwi


def overlay_mask(
    base_image_norm: np.ndarray,
    mask: Optional[np.ndarray],
    color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    alpha: float = 0.35,
) -> np.ndarray:
    h, w = base_image_norm.shape
    rgb = np.stack([base_image_norm] * 3, axis=-1)
    if mask is None:
        return rgb
    mask_bool = mask.astype(bool)
    overlay = rgb.copy()
    for c in range(3):
        overlay[..., c] = np.where(mask_bool, color[c], overlay[..., c])
    blended = (1 - alpha) * rgb + alpha * overlay
    return blended


def resolve_path(path_value: Optional[str], base_dir: str) -> Optional[str]:
    if path_value is None or (isinstance(path_value, float) and np.isnan(path_value)):
        return None
    path_str = str(path_value)
    if os.path.isabs(path_str):
        return path_str
    return os.path.normpath(os.path.join(base_dir, path_str))


def try_load(paths: Dict[str, Optional[str]]) -> Dict[str, Optional[np.ndarray]]:
    return {k: load_nifti(v) for k, v in paths.items()}


def visualize_case(row: pd.Series, slice_index: Optional[int], dwi_vol_idx: Optional[int], out_path: Optional[str], base_dir: str) -> None:
    paths = {
        "t2": resolve_path(row.get("t2", None), base_dir),
        "adc": resolve_path(row.get("adc", None), base_dir),
        "dwi": resolve_path(row.get("dwi", None), base_dir),
        "t2_anatomy_reader1": resolve_path(row.get("t2_anatomy_reader1", None), base_dir),
        "t2_tumor_reader1": resolve_path(row.get("t2_tumor_reader1", None), base_dir),
        "adc_tumor_reader1": resolve_path(row.get("adc_tumor_reader1", None), base_dir),
        "t2_anatomy_reader2": resolve_path(row.get("t2_anatomy_reader2", None), base_dir),
        "adc_tumor_reader2": resolve_path(row.get("adc_tumor_reader2", None), base_dir),
    }

    vols = try_load(paths)

    # Prepare base images
    t2 = vols.get("t2")
    adc = vols.get("adc")
    dwi = vols.get("dwi")

    if dwi is not None:
        dwi = select_dwi_volume(dwi, dwi_vol_idx)

    # Select slices
    t2_slice = select_axial_slice(t2, slice_index)[0] if t2 is not None else None
    adc_slice = select_axial_slice(adc, slice_index)[0] if adc is not None else None
    dwi_slice = select_axial_slice(dwi, slice_index)[0] if dwi is not None else None

    # Normalize for display
    t2_norm = robust_normalize(t2_slice) if t2_slice is not None else None
    adc_norm = robust_normalize(adc_slice) if adc_slice is not None else None
    dwi_norm = robust_normalize(dwi_slice) if dwi_slice is not None else None

    # Masks (match the same slice)
    def mask_slice(name: str) -> Optional[np.ndarray]:
        m = vols.get(name)
        if m is None or m.ndim < 3:
            return None
        return select_axial_slice(m, slice_index)[0]

    t2_gland_r1 = mask_slice("t2_anatomy_reader1")
    t2_tumor_r1 = mask_slice("t2_tumor_reader1")
    adc_tumor_r1 = mask_slice("adc_tumor_reader1")
    adc_tumor_r2 = mask_slice("adc_tumor_reader2")

    # Build figure
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.tight_layout(pad=3.0)

    # Row 1: base images
    def imshow(ax, img, title):
        ax.axis("off")
        if img is None:
            ax.set_title(f"{title} (missing)")
            return
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)

    imshow(axes[0, 0], t2_norm, "T2")
    imshow(axes[0, 1], adc_norm, "ADC")
    imshow(axes[0, 2], dwi_norm, "DWI")

    # Row 2: overlays
    def overlay(ax, base_norm, mask, title, color=(1.0, 0.0, 0.0)):
        ax.axis("off")
        if base_norm is None:
            ax.set_title(f"{title} (missing base)")
            return
        blended = overlay_mask(base_norm, mask, color=color, alpha=0.35)
        ax.imshow(blended, vmin=0, vmax=1)
        ax.set_title(title)

    overlay(axes[1, 0], t2_norm, t2_gland_r1, "T2 + Gland (R1)", color=(0.1, 0.8, 0.1))
    overlay(axes[1, 1], t2_norm, t2_tumor_r1, "T2 + Tumor (R1)", color=(1.0, 0.2, 0.2))
    overlay(axes[1, 2], adc_norm, adc_tumor_r2, "ADC + Tumor (R1)", color=(1.0, 0.5, 0.1))

    case_id = row.get("ID", "unknown")
    fig.suptitle(f"Case {case_id}")

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.subplots_adjust(wspace=0, hspace=0)
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved: {out_path}")
    else:
        plt.show()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)
    csv_base_dir = os.path.dirname(os.path.abspath(args.csv))
    row: pd.Series
    if args.id is not None and "ID" in df.columns:
        matches = df[df["ID"].astype(str) == str(args.id)]
        if matches.empty:
            raise ValueError(f"ID {args.id} not found in CSV")
        row = matches.iloc[0]
    else:
        if args.row < 0 or args.row >= len(df):
            raise IndexError(f"Row index {args.row} out of range (0..{len(df)-1})")
        row = df.iloc[args.row]

    visualize_case(
        row=row,
        slice_index=args.slice,
        dwi_vol_idx=args.dwi_volume,
        out_path=args.out,
        base_dir=csv_base_dir,
    )


if __name__ == "__main__":
    main()
