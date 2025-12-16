#!/usr/bin/env python3

import argparse
import contextlib
import gc
import math
import os
import runpy
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class CaseResult:
    method: str
    h5_path: Path
    slice_idx: Optional[int]
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

    # Guard shape mismatches
    if label.shape != recon.shape:
        raise ValueError(f"Shape mismatch: label {label.shape} vs recon {recon.shape}")

    mse = float(np.mean((recon - label) ** 2))
    data_range = float(np.max(np.abs(label)))
    psnr = float(_psnr_from_mse(mse, data_range))
    return mse, psnr


def run_inference(
    repo_root: Path,
    method: str,
    h5_path: Path,
    checkpoint: Path,
    save_dir: Path,
    mask_type: str,
    acc_factor: int,
    center_fraction: float,
    N: int,
    m: int,
    slice_idx: Optional[int],
    ssos_parallel: bool,
    save_progress: bool,
    save_every: int,
) -> Tuple[Path, Path, float]:
    score_mod_dir = repo_root / "MRI" / "scoreMRI" / "score-MRI-mod"
    if method.upper() == "SENSE":
        script = score_mod_dir / "inference_multi-coil_SENSE_h5.py"
    elif method.upper() == "SSOS":
        script = score_mod_dir / "inference_multi-coil_SSOS_h5.py"
    else:
        raise ValueError(f"Unknown method: {method}")

    if not script.exists():
        raise FileNotFoundError(f"Missing inference script: {script}")

    argv: List[str] = [
        str(script),
        "--data",
        str(h5_path),
        "--checkpoint",
        str(checkpoint),
        "--save_dir",
        str(save_dir),
        "--mask_type",
        str(mask_type),
        "--acc_factor",
        str(acc_factor),
        "--center_fraction",
        str(center_fraction),
        "--N",
        str(N),
        "--m",
        str(m),
    ]

    if slice_idx is not None:
        argv += ["--slice_idx", str(slice_idx)]

    if method.upper() == "SSOS" and ssos_parallel:
        argv += ["--parallel"]

    if save_progress:
        argv += ["--save_progress", "--save_every", str(int(save_every))]

    # Run in-process (no subprocess). We temporarily:
    # - set cwd to score-MRI-mod so relative imports work
    # - prepend score-MRI-mod to sys.path
    # - set sys.argv for argparse
    start = time.perf_counter()
    old_argv = sys.argv[:]
    old_cwd = Path.cwd()
    old_syspath = sys.path[:]
    try:
        sys.argv = argv
        sys.path.insert(0, str(score_mod_dir))
        os.chdir(str(score_mod_dir))
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = old_argv
        os.chdir(str(old_cwd))
        sys.path = old_syspath
        # Help free GPU/CPU memory between runs
        with contextlib.suppress(Exception):
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        gc.collect()
    wall = time.perf_counter() - start

    # Infer output paths based on inference scripts.
    fname = h5_path.stem
    if slice_idx is not None:
        fname = f"{fname}_slice_{slice_idx}"

    method_dir = "SENSE" if method.upper() == "SENSE" else "SSOS"
    out_root = save_dir / "multi-coil" / method_dir
    recon_npy = out_root / "recon" / f"{fname}.npy"
    label_npy = out_root / "label" / f"{fname}.npy"

    if not recon_npy.exists():
        raise FileNotFoundError(f"Expected recon not found: {recon_npy}")
    if not label_npy.exists():
        raise FileNotFoundError(f"Expected label not found: {label_npy}")

    return recon_npy, label_npy, wall


def summarize(vals: List[float]) -> Tuple[float, float]:
    if not vals:
        return float("nan"), float("nan")
    arr = np.asarray(vals, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1) if len(arr) > 1 else 0.0)


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Run score-based MRI inference on 5 compare files (SENSE + SSOS), "
            "using equispaced_fraction mask with acc=4, and report PSNR/MSE and time."
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
        default=Path(
            "/scratch/ml8347/MRI/train/score/workdir3/fastmri_multicoil_knee_320/checkpoints/checkpoint_50.pth"
        ),
        help="Path to diffusion model checkpoint",
    )
    p.add_argument(
        "--save_dir",
        type=Path,
        default=Path("/scratch/ml8347/MRI/train/compare/outputs_equi_acc4"),
        help="Where inference scripts will write outputs",
    )
    p.add_argument(
        "--report",
        type=Path,
        default=Path("/scratch/ml8347/MRI/train/compare/results_equi_acc4.txt"),
        help="Output txt report path",
    )
    p.add_argument("--mask_type", type=str, default="equispaced_fraction")
    p.add_argument("--acc_factor", type=int, default=4)
    p.add_argument("--center_fraction", type=float, default=0.08)
    p.add_argument("--N", type=int, default=600)
    p.add_argument("--m", type=int, default=1)
    p.add_argument(
        "--slice_idx",
        type=int,
        default=None,
        help="Optional fixed slice index (default: middle slice per file)",
    )
    p.add_argument(
        "--no_ssos_parallel",
        action="store_true",
        help="Disable SSOS --parallel",
    )
    p.add_argument(
        "--no_save_progress",
        action="store_true",
        help="Disable saving intermediate recon PNGs + per-step PSNR logs",
    )
    p.add_argument(
        "--save_every",
        type=int,
        default=50,
        help="Save intermediate recon/metrics every N steps (default: 50)",
    )

    args = p.parse_args()

    repo_root = Path("/scratch/ml8347")

    h5_files = sorted(args.compare_dir.glob("*.h5"))
    if len(h5_files) != 5:
        raise RuntimeError(f"Expected 5 .h5 files in {args.compare_dir}, found {len(h5_files)}")

    args.save_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    results: List[CaseResult] = []

    for method in ["SENSE", "SSOS"]:
        for h5_path in h5_files:
            recon_npy, label_npy, wall = run_inference(
                repo_root=repo_root,
                method=method,
                h5_path=h5_path,
                checkpoint=args.checkpoint,
                save_dir=args.save_dir,
                mask_type=args.mask_type,
                acc_factor=args.acc_factor,
                center_fraction=args.center_fraction,
                N=args.N,
                m=args.m,
                slice_idx=args.slice_idx,
                ssos_parallel=(not args.no_ssos_parallel),
                save_progress=(not args.no_save_progress),
                save_every=args.save_every,
            )

            recon = np.load(recon_npy)
            label = np.load(label_npy)
            mse, psnr = compute_mse_psnr(label=label, recon=recon)

            results.append(
                CaseResult(
                    method=method,
                    h5_path=h5_path,
                    slice_idx=args.slice_idx,
                    out_recon_npy=recon_npy,
                    out_label_npy=label_npy,
                    mse=mse,
                    psnr=psnr,
                    wall_time_sec=wall,
                )
            )

    # Write report
    lines: List[str] = []
    lines.append("Score MRI compare evaluation")
    lines.append(f"Mask: {args.mask_type}, acc_factor={args.acc_factor}, center_fraction={args.center_fraction}")
    lines.append(f"N={args.N}, m={args.m}, slice_idx={'middle(default)' if args.slice_idx is None else args.slice_idx}")
    lines.append(f"Checkpoint: {args.checkpoint}")
    lines.append(f"Save dir: {args.save_dir}")
    lines.append("")

    header = f"{'method':<6}  {'file':<14}  {'MSE':>12}  {'PSNR(dB)':>10}  {'time(s)':>10}"
    lines.append(header)
    lines.append("-" * len(header))

    for r in results:
        lines.append(
            f"{r.method:<6}  {r.h5_path.name:<14}  {r.mse:12.6e}  {r.psnr:10.3f}  {r.wall_time_sec:10.3f}"
        )

    lines.append("")

    for method in ["SENSE", "SSOS"]:
        mres = [r for r in results if r.method == method]
        mse_mean, mse_std = summarize([r.mse for r in mres])
        psnr_mean, psnr_std = summarize([r.psnr for r in mres])
        t_mean, t_std = summarize([r.wall_time_sec for r in mres])
        lines.append(f"{method} summary over {len(mres)} files:")
        lines.append(f"  MSE mean/std  : {mse_mean:.6e} / {mse_std:.6e}")
        lines.append(f"  PSNR mean/std : {psnr_mean:.3f} / {psnr_std:.3f} dB")
        lines.append(f"  Time mean/std : {t_mean:.3f} / {t_std:.3f} s")
        lines.append("")

    args.report.write_text("\n".join(lines) + "\n")
    print(f"Wrote report: {args.report}")


if __name__ == "__main__":
    main()
