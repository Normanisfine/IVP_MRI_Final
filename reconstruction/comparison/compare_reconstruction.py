#!/usr/bin/env python3
"""
Comparison script for Score-based and VarNet MRI reconstruction methods.

This script evaluates and compares reconstruction quality between:
- Score-based diffusion models (SENSE and SSOS variants)
- VarNet (Variational Network)

Metrics: MSE, PSNR, reconstruction time
"""

import argparse
import contextlib
import gc
import math
import os
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
    """Compute MSE and PSNR between label and reconstruction."""
    label = np.asarray(label, dtype=np.float64)
    recon = np.asarray(recon, dtype=np.float64)

    # Guard shape mismatches
    if label.shape != recon.shape:
        raise ValueError(f"Shape mismatch: label {label.shape} vs recon {recon.shape}")

    mse = float(np.mean((recon - label) ** 2))
    data_range = float(np.max(np.abs(label)))
    psnr = float(_psnr_from_mse(mse, data_range))
    return mse, psnr


def format_results_table(results: List[CaseResult]) -> str:
    """Format results as a table string."""
    lines = []
    lines.append("method  file                     MSE    PSNR(dB)     time(s)")
    lines.append("-" * 60)
    
    for r in results:
        filename = r.h5_path.name if hasattr(r.h5_path, 'name') else str(r.h5_path)
        lines.append(
            f"{r.method:<6}  {filename:20}  {r.mse:11.6e}  {r.psnr:10.3f}  {r.wall_time_sec:10.3f}"
        )
    
    return "\n".join(lines)


def compute_summary_stats(results: List[CaseResult], method: str) -> dict:
    """Compute summary statistics for a specific method."""
    method_results = [r for r in results if r.method == method]
    
    if not method_results:
        return {}
    
    mses = [r.mse for r in method_results]
    psnrs = [r.psnr for r in method_results]
    times = [r.wall_time_sec for r in method_results]
    
    return {
        'count': len(method_results),
        'mse_mean': np.mean(mses),
        'mse_std': np.std(mses),
        'psnr_mean': np.mean(psnrs),
        'psnr_std': np.std(psnrs),
        'time_mean': np.mean(times),
        'time_std': np.std(times),
    }


def print_summary(results: List[CaseResult]):
    """Print summary statistics for all methods."""
    methods = sorted(set(r.method for r in results))
    
    print("\nSummary Statistics:")
    print("=" * 60)
    
    for method in methods:
        stats = compute_summary_stats(results, method)
        if stats:
            print(f"\n{method} summary over {stats['count']} files:")
            print(f"  MSE mean/std  : {stats['mse_mean']:.6e} / {stats['mse_std']:.6e}")
            print(f"  PSNR mean/std : {stats['psnr_mean']:.3f} / {stats['psnr_std']:.3f} dB")
            print(f"  Time mean/std : {stats['time_mean']:.3f} / {stats['time_std']:.3f} s")


def main():
    parser = argparse.ArgumentParser(
        description="Compare Score-based and VarNet MRI reconstruction methods"
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        required=True,
        help="Directory containing reconstruction results",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file for results (default: print to stdout)",
    )
    
    args = parser.parse_args()
    
    # This is a template script - actual implementation would:
    # 1. Load reconstruction results from both methods
    # 2. Compute metrics for each case
    # 3. Generate comparison tables and plots
    # 4. Save results to file
    
    print("Comparison script template")
    print(f"Results directory: {args.results_dir}")
    print("\nFor actual comparison, integrate with:")
    print("  - Score-based reconstruction: inference_multi-coil_SENSE_h5.py")
    print("  - VarNet reconstruction: reconstruct.py")
    

if __name__ == "__main__":
    main()
