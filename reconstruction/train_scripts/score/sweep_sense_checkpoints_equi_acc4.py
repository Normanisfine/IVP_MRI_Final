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
class SweepResult:
    checkpoint_step: int
    checkpoint_path: Path
    save_dir: Path
    recon_npy: Path
    label_npy: Path
    mse: float
    psnr: float
    wall_time_sec: float
    ok: bool
    error: Optional[str] = None


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
        raise ValueError("Shape mismatch: label %s vs recon %s" % (label.shape, recon.shape))
    mse = float(np.mean((recon - label) ** 2))
    data_range = float(np.max(np.abs(label)))
    psnr = float(_psnr_from_mse(mse, data_range))
    return mse, psnr


def run_sense_inference_inprocess(
    repo_root: Path,
    data_h5: Path,
    checkpoint: Path,
    save_dir: Path,
    mask_type: str,
    acc_factor: int,
    center_fraction: float,
    N: int,
    m: int,
    slice_idx: Optional[int],
) -> Tuple[Path, Path, float]:
    score_mod_dir = repo_root / "MRI" / "scoreMRI" / "score-MRI-mod"
    script = score_mod_dir / "inference_multi-coil_SENSE_h5.py"

    if not script.exists():
        raise FileNotFoundError("Missing inference script: %s" % script)

    argv: List[str] = [
        str(script),
        "--data",
        str(data_h5),
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

        # Best-effort memory cleanup between checkpoints
        with contextlib.suppress(Exception):
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        gc.collect()

    wall = time.perf_counter() - start

    fname = data_h5.stem
    if slice_idx is not None:
        fname = "%s_slice_%d" % (fname, slice_idx)

    out_root = save_dir / "multi-coil" / "SENSE"
    recon_npy = out_root / "recon" / (fname + ".npy")
    label_npy = out_root / "label" / (fname + ".npy")

    if not recon_npy.exists():
        raise FileNotFoundError("Expected recon not found: %s" % recon_npy)
    if not label_npy.exists():
        raise FileNotFoundError("Expected label not found: %s" % label_npy)

    return recon_npy, label_npy, wall


def _summarize(vals: List[float]) -> Tuple[float, float]:
    if not vals:
        return float("nan"), float("nan")
    arr = np.asarray(vals, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1) if len(arr) > 1 else 0.0)
    return mean, std


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Sweep SENSE checkpoints using equispaced_fraction acc=4, compute PSNR/MSE, "
            "and save a txt report + plot. Runs in-process (no subprocess)."
        )
    )

    p.add_argument(
        "--data",
        type=Path,
        default=Path("/scratch/ml8347/MRI/train/train_dataset/multicoil_val/file1000015.h5"),
        help="Input FastMRI multicoil .h5 file",
    )
    p.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=Path("/scratch/ml8347/MRI/train/score/workdir3/fastmri_multicoil_knee_320/checkpoints"),
        help="Directory containing checkpoint_*.pth",
    )
    p.add_argument(
        "--checkpoints",
        type=int,
        nargs="*",
        default=[50, 70, 90, 110, 130, 150, 160],
        help="Checkpoint steps to evaluate (e.g. 50 70 90 ... 160)",
    )
    p.add_argument(
        "--save_root",
        type=Path,
        default=Path("/scratch/ml8347/MRI/train/score/recon_sweep_equi_acc4"),
        help="Root folder to write per-checkpoint recon outputs",
    )
    p.add_argument(
        "--report",
        type=Path,
        default=Path("/scratch/ml8347/MRI/train/score/sweep_sense_equi_acc4.txt"),
        help="Output txt report path",
    )
    p.add_argument(
        "--plot",
        type=Path,
        default=Path("/scratch/ml8347/MRI/train/score/sweep_sense_equi_acc4.png"),
        help="Output plot image path",
    )

    p.add_argument("--mask_type", type=str, default="equispaced_fraction")
    p.add_argument("--acc_factor", type=int, default=4)
    p.add_argument("--center_fraction", type=float, default=0.08)
    p.add_argument("--N", type=int, default=500)
    p.add_argument("--m", type=int, default=1)
    p.add_argument(
        "--slice_idx",
        type=int,
        default=None,
        help="Optional fixed slice index (default: middle slice per file)",
    )

    args = p.parse_args()

    repo_root = Path("/scratch/ml8347")

    args.save_root.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.plot.parent.mkdir(parents=True, exist_ok=True)

    results: List[SweepResult] = []

    for step in args.checkpoints:
        ckpt = args.checkpoint_dir / ("checkpoint_%d.pth" % int(step))
        save_dir = args.save_root / ("checkpoint_%d" % int(step))
        save_dir.mkdir(parents=True, exist_ok=True)

        if not ckpt.exists():
            results.append(
                SweepResult(
                    checkpoint_step=int(step),
                    checkpoint_path=ckpt,
                    save_dir=save_dir,
                    recon_npy=Path(""),
                    label_npy=Path(""),
                    mse=float("nan"),
                    psnr=float("nan"),
                    wall_time_sec=float("nan"),
                    ok=False,
                    error="missing checkpoint",
                )
            )
            continue

        try:
            recon_npy, label_npy, wall = run_sense_inference_inprocess(
                repo_root=repo_root,
                data_h5=args.data,
                checkpoint=ckpt,
                save_dir=save_dir,
                mask_type=args.mask_type,
                acc_factor=args.acc_factor,
                center_fraction=args.center_fraction,
                N=args.N,
                m=args.m,
                slice_idx=args.slice_idx,
            )

            recon = np.load(str(recon_npy))
            label = np.load(str(label_npy))
            mse, psnr = compute_mse_psnr(label=label, recon=recon)

            results.append(
                SweepResult(
                    checkpoint_step=int(step),
                    checkpoint_path=ckpt,
                    save_dir=save_dir,
                    recon_npy=recon_npy,
                    label_npy=label_npy,
                    mse=mse,
                    psnr=psnr,
                    wall_time_sec=wall,
                    ok=True,
                    error=None,
                )
            )
        except Exception as e:
            results.append(
                SweepResult(
                    checkpoint_step=int(step),
                    checkpoint_path=ckpt,
                    save_dir=save_dir,
                    recon_npy=Path(""),
                    label_npy=Path(""),
                    mse=float("nan"),
                    psnr=float("nan"),
                    wall_time_sec=float("nan"),
                    ok=False,
                    error=str(e),
                )
            )

    ok_results = [r for r in results if r.ok]

    # Text report
    lines: List[str] = []
    lines.append("SENSE checkpoint sweep (equispaced_fraction acc=4)")
    lines.append("Data: %s" % args.data)
    lines.append("Checkpoint dir: %s" % args.checkpoint_dir)
    lines.append(
        "Mask: %s, acc_factor=%d, center_fraction=%s" % (args.mask_type, args.acc_factor, str(args.center_fraction))
    )
    lines.append("N=%d, m=%d, slice_idx=%s" % (args.N, args.m, "middle(default)" if args.slice_idx is None else str(args.slice_idx)))
    lines.append("Save root: %s" % args.save_root)
    lines.append("")

    header = "%7s  %10s  %12s  %10s  %10s  %s" % ("ckpt", "ok", "MSE", "PSNR(dB)", "time(s)", "error")
    lines.append(header)
    lines.append("-" * len(header))

    for r in sorted(results, key=lambda x: x.checkpoint_step):
        lines.append(
            "%7d  %10s  %12.6e  %10.3f  %10.3f  %s"
            % (
                r.checkpoint_step,
                str(bool(r.ok)),
                r.mse,
                r.psnr,
                r.wall_time_sec,
                "" if r.ok else (r.error or ""),
            )
        )

    lines.append("")

    if ok_results:
        best_by_psnr = max(ok_results, key=lambda x: x.psnr)
        best_by_mse = min(ok_results, key=lambda x: x.mse)
        lines.append("Best checkpoint by PSNR : checkpoint_%d.pth (PSNR=%.3f dB, MSE=%.6e)" % (best_by_psnr.checkpoint_step, best_by_psnr.psnr, best_by_psnr.mse))
        lines.append("Best checkpoint by MSE  : checkpoint_%d.pth (MSE=%.6e, PSNR=%.3f dB)" % (best_by_mse.checkpoint_step, best_by_mse.mse, best_by_mse.psnr))

        mse_mean, mse_std = _summarize([r.mse for r in ok_results])
        psnr_mean, psnr_std = _summarize([r.psnr for r in ok_results])
        t_mean, t_std = _summarize([r.wall_time_sec for r in ok_results])
        lines.append("")
        lines.append("Summary over %d successful checkpoints:" % len(ok_results))
        lines.append("  MSE  mean/std : %.6e / %.6e" % (mse_mean, mse_std))
        lines.append("  PSNR mean/std : %.3f / %.3f dB" % (psnr_mean, psnr_std))
        lines.append("  Time mean/std : %.3f / %.3f s" % (t_mean, t_std))

    args.report.write_text("\n".join(lines) + "\n")

    # Plot
    try:
        import matplotlib.pyplot as plt

        xs = [r.checkpoint_step for r in ok_results]
        ys_psnr = [r.psnr for r in ok_results]
        ys_mse = [r.mse for r in ok_results]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), constrained_layout=True)

        ax1.plot(xs, ys_psnr, marker="o")
        ax1.set_title("SENSE checkpoint sweep (equi acc=4)")
        ax1.set_xlabel("Checkpoint step")
        ax1.set_ylabel("PSNR (dB)")
        ax1.grid(True, alpha=0.3)

        ax2.plot(xs, ys_mse, marker="o")
        ax2.set_xlabel("Checkpoint step")
        ax2.set_ylabel("MSE")
        ax2.set_yscale("log")
        ax2.grid(True, alpha=0.3, which="both")

        fig.savefig(str(args.plot), dpi=150)
        plt.close(fig)
    except Exception:
        # Plotting is best-effort; metrics are still written.
        pass


if __name__ == "__main__":
    main()
