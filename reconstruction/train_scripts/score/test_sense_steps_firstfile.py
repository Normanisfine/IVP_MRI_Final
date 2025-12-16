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
class StepResult:
    N: int
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

    if label.shape != recon.shape:
        raise ValueError(f"Shape mismatch: label {label.shape} vs recon {recon.shape}")

    mse = float(np.mean((recon - label) ** 2))
    data_range = float(np.max(np.abs(label)))
    psnr = float(_psnr_from_mse(mse, data_range))
    return mse, psnr


def run_sense_inference(
    repo_root: Path,
    h5_path: Path,
    checkpoint: Path,
    save_dir: Path,
    mask_type: str,
    acc_factor: int,
    center_fraction: float,
    N: int,
    m: int,
    slice_idx: Optional[int],
    save_progress: bool,
) -> Tuple[Path, Path, float]:
    """Run score-MRI-mod SENSE inference in-process and return (recon_npy, label_npy, wall_time)."""

    score_mod_dir = repo_root / "MRI" / "scoreMRI" / "score-MRI-mod"
    script = score_mod_dir / "inference_multi-coil_SENSE_h5.py"
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

    if save_progress:
        argv += ["--save_progress"]

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
        with contextlib.suppress(Exception):
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        gc.collect()

    wall = time.perf_counter() - start

    # Output path convention used by inference_multi-coil_SENSE_h5.py
    fname = h5_path.stem
    if slice_idx is not None:
        fname = f"{fname}_slice_{slice_idx}"

    out_root = save_dir / "multi-coil" / "SENSE"
    recon_npy = out_root / "recon" / f"{fname}.npy"
    label_npy = out_root / "label" / f"{fname}.npy"

    if not recon_npy.exists():
        raise FileNotFoundError(f"Expected recon not found: {recon_npy}")
    if not label_npy.exists():
        raise FileNotFoundError(f"Expected label not found: {label_npy}")

    return recon_npy, label_npy, wall


def parse_steps(s: str) -> List[int]:
    """Parse steps like '100,200,300' or '100-800:100'."""
    s = s.strip()
    if not s:
        return []

    if ":" in s and "-" in s:
        # format: start-end:step
        range_part, step_part = s.split(":", 1)
        start_s, end_s = range_part.split("-", 1)
        start, end, step = int(start_s), int(end_s), int(step_part)
        if step <= 0:
            raise ValueError("step must be > 0")
        if end < start:
            raise ValueError("end must be >= start")
        return list(range(start, end + 1, step))

    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Sweep SENSE score-POCS steps (N) on ONLY the first compare file, "
            "using the same setup as MRI/train/compare/test_score_compare.py."
        )
    )
    p.add_argument(
        "--compare_dir",
        type=Path,
        default=Path("/scratch/ml8347/MRI/train/train_dataset/compare"),
        help="Directory containing compare .h5 files (we use only the first)",
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
        "--save_root",
        type=Path,
        default=Path("/scratch/ml8347/MRI/train/score/outputs_sense_steps_firstfile_equi_acc4"),
        help="Root directory; per-N outputs will be saved into save_root/N{N}/...",
    )
    p.add_argument(
        "--report",
        type=Path,
        default=Path("/scratch/ml8347/MRI/train/score/sweep_sense_steps_firstfile_equi_acc4.txt"),
        help="Output txt report path",
    )
    p.add_argument("--mask_type", type=str, default="equispaced_fraction")
    p.add_argument("--acc_factor", type=int, default=4)
    p.add_argument("--center_fraction", type=float, default=0.08)
    p.add_argument("--m", type=int, default=1)
    p.add_argument(
        "--slice_idx",
        type=int,
        default=None,
        help="Optional fixed slice index (default: middle slice)",
    )
    p.add_argument(
        "--steps",
        type=str,
        default="100,200,300,400,500,600,700,800",
        help="Steps (N) list, e.g. '100,200,...' or range '100-800:100'",
    )
    p.add_argument(
        "--save_progress",
        action="store_true",
        help="Pass --save_progress to inference (saves intermediate recon_progress)",
    )

    args = p.parse_args()

    repo_root = Path("/scratch/ml8347")

    h5_files = sorted(args.compare_dir.glob("*.h5"))
    if len(h5_files) < 1:
        raise RuntimeError(f"No .h5 files found in {args.compare_dir}")

    h5_path = h5_files[0]
    steps = parse_steps(args.steps)
    if not steps:
        raise RuntimeError("No steps parsed from --steps")

    args.save_root.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    results: List[StepResult] = []

    print(f"Using ONLY first file: {h5_path}")
    print(f"Sweeping N over: {steps}")

    for N in steps:
        save_dir = args.save_root / f"N{N}"
        save_dir.mkdir(parents=True, exist_ok=True)

        recon_npy, label_npy, wall = run_sense_inference(
            repo_root=repo_root,
            h5_path=h5_path,
            checkpoint=args.checkpoint,
            save_dir=save_dir,
            mask_type=args.mask_type,
            acc_factor=args.acc_factor,
            center_fraction=args.center_fraction,
            N=N,
            m=args.m,
            slice_idx=args.slice_idx,
            save_progress=args.save_progress,
        )

        recon = np.load(recon_npy)
        label = np.load(label_npy)
        mse, psnr = compute_mse_psnr(label=label, recon=recon)

        results.append(
            StepResult(
                N=N,
                h5_path=h5_path,
                slice_idx=args.slice_idx,
                out_recon_npy=recon_npy,
                out_label_npy=label_npy,
                mse=mse,
                psnr=psnr,
                wall_time_sec=wall,
            )
        )

        print(f"N={N:4d}  MSE={mse:12.6e}  PSNR={psnr:8.3f} dB  time={wall:8.3f}s")

    # Write report
    lines: List[str] = []
    lines.append("SENSE steps (N) sweep on first compare file")
    lines.append(f"File: {h5_path}")
    lines.append(
        f"Mask: {args.mask_type}, acc_factor={args.acc_factor}, center_fraction={args.center_fraction}"
    )
    lines.append(f"m={args.m}, slice_idx={'middle(default)' if args.slice_idx is None else args.slice_idx}")
    lines.append(f"Checkpoint: {args.checkpoint}")
    lines.append(f"Save root: {args.save_root}")
    lines.append("")

    header = f"{'N':>4}  {'MSE':>12}  {'PSNR(dB)':>10}  {'time(s)':>10}  {'recon_npy':<40}"
    lines.append(header)
    lines.append("-" * len(header))
    for r in results:
        lines.append(
            f"{r.N:4d}  {r.mse:12.6e}  {r.psnr:10.3f}  {r.wall_time_sec:10.3f}  {str(r.out_recon_npy)}"
        )

    args.report.write_text("\n".join(lines) + "\n")
    print(f"Wrote report: {args.report}")


if __name__ == "__main__":
    main()
