#!/usr/bin/env python3

import argparse
import contextlib
import gc
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import torch
import h5py

import matplotlib

# Use non-GUI backend for server/HPC environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fastmri
import fastmri.data.transforms as T
from fastmri.data import SliceDataset
from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.pl_modules import VarNetModule


@dataclass
class CaseResult:
    method: str
    h5_path: Path
    slice_idx: int
    out_recon_npy: Path
    out_label_npy: Path
    mse: float
    psnr: float
    wall_time_sec: float


def _psnr_from_mse(mse: float, data_range: float) -> float:
    if mse <= 0:
        return float("inf")
    if data_range <= 0:
        return float("nan")
    return 20.0 * math.log10(data_range / math.sqrt(mse))


def compute_mse_psnr(label: np.ndarray, recon: np.ndarray) -> Tuple[float, float]:
    label = np.asarray(label, dtype=np.float64)
    recon = np.asarray(recon, dtype=np.float64)
    if label.shape != recon.shape:
        raise ValueError(f"Shape mismatch: label {label.shape} vs recon {recon.shape}")
    mse = float(np.mean((recon - label) ** 2))
    data_range = float(np.max(np.abs(label)))
    psnr = float(_psnr_from_mse(mse, data_range))
    return mse, psnr


def summarize(vals: List[float]) -> Tuple[float, float]:
    if not vals:
        return float("nan"), float("nan")
    arr = np.asarray(vals, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1) if len(arr) > 1 else 0.0)


def middle_slice_idx(h5_path: Path) -> int:
    # Prefer kspace if present; fall back to reconstruction_rss.
    with h5py.File(h5_path, "r") as hf:
        if "kspace" in hf:
            n = int(hf["kspace"].shape[0])
        elif "reconstruction_rss" in hf:
            n = int(hf["reconstruction_rss"].shape[0])
        else:
            raise KeyError(f"{h5_path} missing 'kspace' and 'reconstruction_rss'")
    return n // 2


def load_label_from_h5(h5_path: Path, slice_idx: int) -> np.ndarray:
    with h5py.File(h5_path, "r") as hf:
        if "reconstruction_rss" in hf:
            return np.asarray(hf["reconstruction_rss"][slice_idx])
        # As a fallback, derive RSS from fully-sampled kspace if available.
        if "kspace" in hf:
            kspace = np.asarray(hf["kspace"][slice_idx])  # (coils, H, W) complex64 typically
            img = np.fft.fftshift(
                np.fft.ifft2(np.fft.ifftshift(kspace, axes=(-2, -1)), axes=(-2, -1)),
                axes=(-2, -1),
            )
            rss = np.sqrt(np.sum(np.abs(img) ** 2, axis=0))
            return rss
    raise KeyError(f"{h5_path} missing label sources")


def load_original_from_h5(h5_path: Path, slice_idx: int) -> np.ndarray:
    """
    "Original" here means a fully-sampled RSS image reconstructed from k-space.
    Falls back to reconstruction_rss if kspace is unavailable.
    """
    with h5py.File(h5_path, "r") as hf:
        if "kspace" in hf:
            kspace = np.asarray(hf["kspace"][slice_idx])  # (coils, H, W) complex64 typically
            img = np.fft.fftshift(
                np.fft.ifft2(np.fft.ifftshift(kspace, axes=(-2, -1)), axes=(-2, -1)),
                axes=(-2, -1),
            )
            rss = np.sqrt(np.sum(np.abs(img) ** 2, axis=0))
            return rss
        if "reconstruction_rss" in hf:
            return np.asarray(hf["reconstruction_rss"][slice_idx])
    raise KeyError(f"{h5_path} missing kspace and reconstruction_rss")


def run_varnet_on_batch(batch, varnet, device: str) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Returns (recon_2d, label_2d, wall_time_sec).
    """
    crop_size = batch.crop_size
    masked_kspace = batch.masked_kspace.to(device)
    mask = batch.mask.to(device)

    # Label: prefer transform-provided target if present, else load from H5.
    target = getattr(batch, "target", None)
    if target is not None:
        target = target.to(device)
    else:
        # batch.fname is like "file1000060.h5"
        # batch.slice_num is tensor([idx])
        # We'll load label outside using h5py: caller provides path map.
        raise RuntimeError("Expected batch.target from VarNetDataTransform, but got None")

    with torch.no_grad():
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        output = varnet(masked_kspace, mask)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        wall = time.perf_counter() - start

    # Handle FLAIR-like edge case (copied from your `reconstruct.py`)
    if output.shape[-1] < crop_size[1]:
        crop_size = (output.shape[-1], output.shape[-1])

    # Center-crop both to the same size; drop batch dimension.
    output = T.center_crop(output, crop_size)[0]
    target = T.center_crop(target, crop_size)[0]

    recon = output.detach().cpu().numpy()
    label = target.detach().cpu().numpy()
    return recon, label, wall


def _as_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 2:
        return x
    if x.ndim == 3 and x.shape[0] == 1:
        return x[0]
    return np.squeeze(x)


def _center_crop_np(img2d: np.ndarray, crop_size: Tuple[int, int]) -> np.ndarray:
    t = torch.from_numpy(np.asarray(img2d))[None, ...]  # (1, H, W)
    t = T.center_crop(t, crop_size)[0]
    return t.numpy()


def _robust_vmax(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 1.0
    vmax = float(np.percentile(np.abs(x), 99.5))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.max(np.abs(x)))
    return float(vmax) if vmax > 0 else 1.0


def _fft_mag_image(x2d: np.ndarray) -> np.ndarray:
    """
    Log-magnitude FFT visualization (centered).
    """
    x = np.asarray(x2d, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    F = np.fft.fftshift(np.fft.fft2(x))
    # log scaling is critical for visualization (FFT magnitude has huge dynamic range)
    mag = np.log10(1.0 + np.abs(F))
    return mag.astype(np.float32)


def _save_gray_png(path: Path, img2d: np.ndarray, title: str, vmin: float, vmax: float) -> None:
    plt.figure(figsize=(6, 6))
    plt.imshow(img2d, cmap="gray", vmin=vmin, vmax=vmax)
    plt.title(title, fontsize=12)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def _save_cmap_png(
    path: Path,
    img2d: np.ndarray,
    title: str,
    *,
    cmap: str = "magma",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    colorbar: bool = False,
) -> None:
    plt.figure(figsize=(6, 6))
    im = plt.imshow(img2d, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(title, fontsize=12)
    plt.axis("off")
    if colorbar:
        plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def save_case_images(
    *,
    recon: np.ndarray,
    label: np.ndarray,
    original: np.ndarray,
    input_image: np.ndarray,
    out_dir: Path,
    tag: str,
    mse: float,
    psnr: float,
) -> Tuple[Path, Path, Path]:
    """
    Save PNGs for quick visual inspection:
    - {tag}_label.png
    - {tag}_original.png
    - {tag}_input.png
    - {tag}_recon.png
    and their frequency (FFT magnitude) images:
    - {tag}_label_freq.png
    - {tag}_original_freq.png
    - {tag}_input_freq.png
    - {tag}_recon_freq.png
    plus a panel:
    - {tag}_panel.png  (label, input, recon, |diff|)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    recon2d = _as_2d(recon)
    label2d = _as_2d(label)
    original2d = _as_2d(original)
    input2d = _as_2d(input_image)
    diff2d = np.abs(recon2d - label2d)

    vmax = _robust_vmax(label2d)
    dvmax = _robust_vmax(diff2d)

    label_png = out_dir / f"{tag}_label.png"
    original_png = out_dir / f"{tag}_original.png"
    input_png = out_dir / f"{tag}_input.png"
    recon_png = out_dir / f"{tag}_recon.png"
    panel_png = out_dir / f"{tag}_panel.png"

    # Spatial images
    _save_gray_png(label_png, label2d, f"Label: {tag}", vmin=0.0, vmax=vmax)
    _save_gray_png(original_png, original2d, f"Original (from k-space): {tag}", vmin=0.0, vmax=vmax)
    _save_gray_png(input_png, input2d, f"Input (zero-filled): {tag}", vmin=0.0, vmax=vmax)
    _save_gray_png(
        recon_png,
        recon2d,
        f"Reconstructed (VarNet): {tag}\nPSNR={psnr:.2f} dB, MSE={mse:.3e}",
        vmin=0.0,
        vmax=vmax,
    )

    # Frequency (FFT magnitude) images
    label_freq = _fft_mag_image(label2d)
    original_freq = _fft_mag_image(original2d)
    input_freq = _fft_mag_image(input2d)
    recon_freq = _fft_mag_image(recon2d)

    # Use a shared robust scale across all FFT images so they are comparable by eye.
    all_freq = np.stack([label_freq, original_freq, input_freq, recon_freq], axis=0)
    finite = all_freq[np.isfinite(all_freq)]
    if finite.size:
        f_lo = float(np.percentile(finite, 1.0))
        f_hi = float(np.percentile(finite, 99.7))
        if not np.isfinite(f_lo):
            f_lo = float(np.min(finite))
        if not np.isfinite(f_hi) or f_hi <= f_lo:
            f_hi = float(np.max(finite))
    else:
        f_lo, f_hi = 0.0, 1.0

    label_freq_png = out_dir / f"{tag}_label_freq.png"
    original_freq_png = out_dir / f"{tag}_original_freq.png"
    input_freq_png = out_dir / f"{tag}_input_freq.png"
    recon_freq_png = out_dir / f"{tag}_recon_freq.png"

    _save_cmap_png(
        label_freq_png,
        label_freq,
        f"log10(1+|FFT|) Label: {tag}",
        vmin=f_lo,
        vmax=f_hi,
        colorbar=True,
    )
    _save_cmap_png(
        original_freq_png,
        original_freq,
        f"log10(1+|FFT|) Original: {tag}",
        vmin=f_lo,
        vmax=f_hi,
        colorbar=True,
    )
    _save_cmap_png(
        input_freq_png,
        input_freq,
        f"log10(1+|FFT|) Input: {tag}",
        vmin=f_lo,
        vmax=f_hi,
        colorbar=True,
    )
    _save_cmap_png(
        recon_freq_png,
        recon_freq,
        f"log10(1+|FFT|) Recon: {tag}",
        vmin=f_lo,
        vmax=f_hi,
        colorbar=True,
    )

    # Panel (label, input, recon, diff)
    fig, axes = plt.subplots(1, 4, figsize=(21, 6))
    axes[0].imshow(label2d, cmap="gray", vmin=0.0, vmax=vmax)
    axes[0].set_title("Label", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(input2d, cmap="gray", vmin=0.0, vmax=vmax)
    axes[1].set_title("Input (zero-filled)", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(recon2d, cmap="gray", vmin=0.0, vmax=vmax)
    axes[2].set_title("VarNet recon", fontsize=12)
    axes[2].axis("off")

    axes[3].imshow(diff2d, cmap="hot", vmin=0.0, vmax=dvmax)
    axes[3].set_title("|Recon - Label|", fontsize=12)
    axes[3].axis("off")

    fig.suptitle(f"{tag}  |  PSNR={psnr:.2f} dB  MSE={mse:.3e}", fontsize=14)
    plt.tight_layout()
    plt.savefig(panel_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Return the main spatial images + panel (for convenience)
    return label_png, input_png, recon_png


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate trained VarNet on the 5 compare .h5 files, using the same "
            "mask setup (equispaced_fraction, acc=4, center_fraction=0.08) and "
            "report per-file MSE/PSNR/time plus mean/std. Saves recon/label .npy."
        )
    )
    p.add_argument(
        "--compare_dir",
        type=Path,
        default=Path("/scratch/ml8347/MRI/train/train_dataset/compare"),
        help="Directory containing 5 compare .h5 files",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/scratch/ml8347/MRI/train/varnet/checkpoints/last.ckpt"),
        help="Path to VarNet Lightning checkpoint (.ckpt)",
    )
    p.add_argument(
        "--save_dir",
        type=Path,
        default=Path("/scratch/ml8347/MRI/train/compare/varnet/compare_outputs_equi_acc4"),
        help="Where to save recon/label .npy outputs",
    )
    p.add_argument(
        "--report",
        type=Path,
        default=Path("/scratch/ml8347/MRI/train/compare/varnet/results_compare_equi_acc4.txt"),
        help="Output txt report path",
    )

    # Mask config (match training)
    p.add_argument("--mask_type", type=str, default="equispaced_fraction")
    p.add_argument("--acc_factor", type=int, default=4)
    p.add_argument("--center_fraction", type=float, default=0.08)

    # Slice selection
    p.add_argument(
        "--slice_idx",
        type=int,
        default=None,
        help="Optional fixed slice index (default: middle slice per file)",
    )

    # Runtime
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = p.parse_args()

    h5_files = sorted(args.compare_dir.glob("*.h5"))
    if len(h5_files) != 5:
        raise RuntimeError(f"Expected 5 .h5 files in {args.compare_dir}, found {len(h5_files)}")

    # Choose slice per file (middle by default, to match `test_score_compare.py` semantics)
    target_slice: Dict[str, int] = {}
    for h5 in h5_files:
        target_slice[h5.name] = int(args.slice_idx if args.slice_idx is not None else middle_slice_idx(h5))

    args.save_dir.mkdir(parents=True, exist_ok=True)
    (args.save_dir / "varnet" / "recon").mkdir(parents=True, exist_ok=True)
    (args.save_dir / "varnet" / "label").mkdir(parents=True, exist_ok=True)
    (args.save_dir / "varnet" / "images").mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    # Load VarNet
    device = args.device
    lightning_model = VarNetModule.load_from_checkpoint(str(args.checkpoint))
    varnet = lightning_model.varnet.eval().to(device)

    # Dataset/transform (same mask setup)
    mask_func = create_mask_for_mask_type(args.mask_type, [args.center_fraction], [args.acc_factor])
    transform = T.VarNetDataTransform(mask_func=mask_func)
    dataset = SliceDataset(root=args.compare_dir, transform=transform, challenge="multicoil")
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=args.num_workers, shuffle=False)

    # Evaluate only the 5 selected (file, slice) pairs
    results: List[CaseResult] = []
    want = {(h5.name, target_slice[h5.name]) for h5 in h5_files}

    for batch in loader:
        fname = batch.fname[0]
        slice_num = int(batch.slice_num[0])
        if (fname, slice_num) not in want:
            continue

        # Run model + get label; if transform didn't provide label, load from H5
        try:
            recon, label, wall = run_varnet_on_batch(batch, varnet, device)
        except RuntimeError:
            # Fallback path if `batch.target` is None in your environment.
            # Still run model, but load label from the file.
            crop_size = batch.crop_size
            masked_kspace = batch.masked_kspace.to(device)
            mask = batch.mask.to(device)

            with torch.no_grad():
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                start = time.perf_counter()
                output = varnet(masked_kspace, mask)
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                wall = time.perf_counter() - start

            if output.shape[-1] < crop_size[1]:
                crop_size = (output.shape[-1], output.shape[-1])

            output = T.center_crop(output, crop_size)[0]
            recon = output.detach().cpu().numpy()

            h5_path = args.compare_dir / fname
            label = load_label_from_h5(h5_path, slice_num)
            # Match crop
            label_t = torch.from_numpy(label)[None, ...]
            label = T.center_crop(label_t, crop_size)[0].numpy()

        mse, psnr = compute_mse_psnr(label=label, recon=recon)

        tag = Path(fname).stem + f"_slice_{slice_num}"
        out_recon = args.save_dir / "varnet" / "recon" / f"{tag}.npy"
        out_label = args.save_dir / "varnet" / "label" / f"{tag}.npy"
        np.save(out_recon, recon)
        np.save(out_label, label)

        # Save quick-look images
        # - label: ground-truth target (from transform or H5)
        # - original: fully-sampled RSS from k-space (H5)
        # - input: zero-filled RSS from masked k-space (batch)
        crop_size = batch.crop_size
        input_img = fastmri.ifft2c(batch.masked_kspace)
        input_img = fastmri.complex_abs(input_img)
        input_img = fastmri.rss(input_img, dim=1)[0].cpu().numpy()

        # Match the same crop used for recon/label
        recon2d = _as_2d(recon)
        crop_size_t = (int(recon2d.shape[-2]), int(recon2d.shape[-1]))
        input_img = _center_crop_np(input_img, crop_size_t)

        h5_path = args.compare_dir / fname
        original_img = load_original_from_h5(h5_path, slice_num)
        original_img = _center_crop_np(original_img, crop_size_t)

        save_case_images(
            recon=recon,
            label=label,
            original=original_img,
            input_image=input_img,
            out_dir=(args.save_dir / "varnet" / "images"),
            tag=tag,
            mse=mse,
            psnr=psnr,
        )

        results.append(
            CaseResult(
                method="VARNET",
                h5_path=args.compare_dir / fname,
                slice_idx=slice_num,
                out_recon_npy=out_recon,
                out_label_npy=out_label,
                mse=mse,
                psnr=psnr,
                wall_time_sec=wall,
            )
        )

        # Free memory between cases
        with contextlib.suppress(Exception):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        gc.collect()

        if len(results) == 5:
            break

    if len(results) != 5:
        got = {(r.h5_path.name, r.slice_idx) for r in results}
        raise RuntimeError(f"Only evaluated {len(results)}/5 cases. Got: {sorted(got)}")

    # Report (same “shape” as `test_score_compare.py`)
    lines: List[str] = []
    lines.append("VarNet compare evaluation")
    lines.append(f"Mask: {args.mask_type}, acc_factor={args.acc_factor}, center_fraction={args.center_fraction}")
    lines.append(f"slice_idx={'middle(default)' if args.slice_idx is None else args.slice_idx}")
    lines.append(f"Checkpoint: {args.checkpoint}")
    lines.append(f"Save dir: {args.save_dir}")
    lines.append("")
    header = f"{'method':<6}  {'file':<14}  {'slice':>5}  {'MSE':>12}  {'PSNR(dB)':>10}  {'time(s)':>10}"
    lines.append(header)
    lines.append("-" * len(header))

    # Keep file order consistent with compare_dir listing
    by_file = {(r.h5_path.name, r.slice_idx): r for r in results}
    for h5 in h5_files:
        r = by_file[(h5.name, target_slice[h5.name])]
        lines.append(
            f"{r.method:<6}  {r.h5_path.name:<14}  {r.slice_idx:5d}  {r.mse:12.6e}  {r.psnr:10.3f}  {r.wall_time_sec:10.3f}"
        )

    lines.append("")
    mse_mean, mse_std = summarize([r.mse for r in results])
    psnr_mean, psnr_std = summarize([r.psnr for r in results])
    t_mean, t_std = summarize([r.wall_time_sec for r in results])
    lines.append(f"VARNET summary over {len(results)} files:")
    lines.append(f"  MSE mean/std  : {mse_mean:.6e} / {mse_std:.6e}")
    lines.append(f"  PSNR mean/std : {psnr_mean:.3f} / {psnr_std:.3f} dB")
    lines.append(f"  Time mean/std : {t_mean:.3f} / {t_std:.3f} s")
    lines.append("")

    args.report.write_text("\n".join(lines) + "\n")
    print(f"Wrote report: {args.report}")


if __name__ == "__main__":
    main()