#!/usr/bin/env python3

import os
import argparse
from glob import glob
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm


def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Compute Dice coefficient between prediction and ground truth.
    Uses same calculation as scripts/compute_dice.py for consistency.

    Args:
        pred: Predicted binary mask
        gt: Ground truth binary mask

    Returns:
        Dice coefficient (0-1)
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    intersection = np.logical_and(pred, gt).sum(dtype=np.int64)
    pred_sum = pred.sum(dtype=np.int64)
    gt_sum = gt.sum(dtype=np.int64)
    denom = pred_sum + gt_sum

    if denom == 0:
        # Both empty => define Dice as 1.0 for background agreement
        return 1.0

    return 2.0 * intersection / float(denom)


def compute_hausdorff_95(pred: np.ndarray, gt: np.ndarray, spacing: np.ndarray = None) -> float:
    """
    Compute 95th percentile Hausdorff distance.

    Args:
        pred: Predicted binary mask
        gt: Ground truth binary mask
        spacing: Voxel spacing (optional)

    Returns:
        HD95 in mm (or voxels if spacing not provided)
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    if np.sum(gt) == 0 and np.sum(pred) == 0:
        return 0.0
    if np.sum(gt) == 0 or np.sum(pred) == 0:
        return np.inf

    if spacing is None:
        spacing = np.array([1.0, 1.0, 1.0])

    # Compute surface voxels (boundary of the mask)
    pred_dt = distance_transform_edt(pred)
    gt_dt = distance_transform_edt(gt)
    pred_surface = np.logical_and(pred, pred_dt <= 1)
    gt_surface = np.logical_and(gt, gt_dt <= 1)

    # Get surface coordinates
    pred_coords = np.array(np.where(pred_surface)).T * spacing
    gt_coords = np.array(np.where(gt_surface)).T * spacing

    if len(pred_coords) == 0 or len(gt_coords) == 0:
        return np.inf

    # Compute distances from pred to gt
    pred_to_gt = []
    for p in pred_coords:
        distances = np.sqrt(np.sum((gt_coords - p) ** 2, axis=1))
        pred_to_gt.append(np.min(distances))

    # Compute distances from gt to pred
    gt_to_pred = []
    for g in gt_coords:
        distances = np.sqrt(np.sum((pred_coords - g) ** 2, axis=1))
        gt_to_pred.append(np.min(distances))

    # Combine and get 95th percentile
    all_distances = np.concatenate([pred_to_gt, gt_to_pred])
    return np.percentile(all_distances, 95)


def compute_surface_distance(pred: np.ndarray, gt: np.ndarray, spacing: np.ndarray = None) -> float:
    """
    Compute average symmetric surface distance.

    Args:
        pred: Predicted binary mask
        gt: Ground truth binary mask
        spacing: Voxel spacing (optional)

    Returns:
        Average surface distance in mm (or voxels if spacing not provided)
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    if np.sum(gt) == 0 and np.sum(pred) == 0:
        return 0.0
    if np.sum(gt) == 0 or np.sum(pred) == 0:
        return np.inf

    if spacing is None:
        spacing = np.array([1.0, 1.0, 1.0])

    # Compute distance transform of the complement
    pred_dist = distance_transform_edt(~pred, sampling=spacing)
    gt_dist = distance_transform_edt(~gt, sampling=spacing)

    # Get surface voxels (boundary of the mask)
    pred_dt = distance_transform_edt(pred)
    gt_dt = distance_transform_edt(gt)
    pred_surface = np.logical_and(pred, pred_dt <= 1)
    gt_surface = np.logical_and(gt, gt_dt <= 1)

    # Average distance from pred surface to gt
    if np.sum(pred_surface) > 0:
        pred_to_gt = np.mean(gt_dist[pred_surface])
    else:
        pred_to_gt = 0

    # Average distance from gt surface to pred
    if np.sum(gt_surface) > 0:
        gt_to_pred = np.mean(pred_dist[gt_surface])
    else:
        gt_to_pred = 0

    return (pred_to_gt + gt_to_pred) / 2


def evaluate_case(npz_path: str) -> dict:
    """
    Evaluate a single case.

    Args:
        npz_path: Path to prediction NPZ file

    Returns:
        Dictionary with metrics
    """
    data = np.load(npz_path, allow_pickle=True)
    pred = data['segs']
    gt = data['gts']
    spacing = data.get('spacing', np.array([1.0, 1.0, 1.0]))

    # Binarize
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)

    # Compute metrics
    dice = compute_dice(pred, gt)
    hd95 = compute_hausdorff_95(pred, gt, spacing)
    asd = compute_surface_distance(pred, gt, spacing)

    # Volume metrics
    pred_volume = np.sum(pred) * np.prod(spacing)  # in mm^3
    gt_volume = np.sum(gt) * np.prod(spacing)
    volume_diff = abs(pred_volume - gt_volume)

    return {
        'dice': dice,
        'hd95': hd95,
        'asd': asd,
        'pred_volume_mm3': pred_volume,
        'gt_volume_mm3': gt_volume,
        'volume_diff_mm3': volume_diff,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate MedSAM2 prostate segmentation")
    parser.add_argument(
        "--pred", "-p",
        type=str,
        required=True,
        help="Directory containing prediction NPZ files"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output directory for evaluation results"
    )

    args = parser.parse_args()

    pred_dir = Path(args.pred)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get prediction files
    pred_files = sorted(glob(str(pred_dir / "*_pred.npz")))
    print(f"Found {len(pred_files)} prediction files")

    # Evaluate each case
    results = OrderedDict()
    results['case'] = []
    results['dice'] = []
    results['hd95'] = []
    results['asd'] = []
    results['pred_volume_mm3'] = []
    results['gt_volume_mm3'] = []
    results['volume_diff_mm3'] = []

    for pred_path in tqdm(pred_files, desc="Evaluating"):
        case_name = os.path.basename(pred_path).replace('_pred.npz', '')

        try:
            metrics = evaluate_case(pred_path)
            results['case'].append(case_name)
            results['dice'].append(metrics['dice'])
            results['hd95'].append(metrics['hd95'])
            results['asd'].append(metrics['asd'])
            results['pred_volume_mm3'].append(metrics['pred_volume_mm3'])
            results['gt_volume_mm3'].append(metrics['gt_volume_mm3'])
            results['volume_diff_mm3'].append(metrics['volume_diff_mm3'])
        except Exception as e:
            print(f"  Error evaluating {case_name}: {e}")

    # Create DataFrame
    results_df = pd.DataFrame(results)

    # Save per-case results
    results_df.to_csv(output_dir / "per_case_metrics.csv", index=False)

    # Compute and save aggregate statistics
    # Filter out inf values for HD95 and ASD
    valid_hd95 = [x for x in results['hd95'] if x != np.inf]
    valid_asd = [x for x in results['asd'] if x != np.inf]

    summary = {
        'metric': ['dice', 'hd95', 'asd', 'volume_diff_mm3'],
        'mean': [
            np.mean(results['dice']),
            np.mean(valid_hd95) if valid_hd95 else np.nan,
            np.mean(valid_asd) if valid_asd else np.nan,
            np.mean(results['volume_diff_mm3']),
        ],
        'std': [
            np.std(results['dice']),
            np.std(valid_hd95) if valid_hd95 else np.nan,
            np.std(valid_asd) if valid_asd else np.nan,
            np.std(results['volume_diff_mm3']),
        ],
        'median': [
            np.median(results['dice']),
            np.median(valid_hd95) if valid_hd95 else np.nan,
            np.median(valid_asd) if valid_asd else np.nan,
            np.median(results['volume_diff_mm3']),
        ],
        'min': [
            np.min(results['dice']),
            np.min(valid_hd95) if valid_hd95 else np.nan,
            np.min(valid_asd) if valid_asd else np.nan,
            np.min(results['volume_diff_mm3']),
        ],
        'max': [
            np.max(results['dice']),
            np.max(valid_hd95) if valid_hd95 else np.nan,
            np.max(valid_asd) if valid_asd else np.nan,
            np.max(results['volume_diff_mm3']),
        ],
    }
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(output_dir / "summary_metrics.csv", index=False)

    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Total cases: {len(results['case'])}")
    print(f"\nDice Coefficient:")
    print(f"  Mean: {np.mean(results['dice']):.4f} +/- {np.std(results['dice']):.4f}")
    print(f"  Median: {np.median(results['dice']):.4f}")
    print(f"  Range: [{np.min(results['dice']):.4f}, {np.max(results['dice']):.4f}]")

    if valid_hd95:
        print(f"\nHausdorff Distance 95% (mm):")
        print(f"  Mean: {np.mean(valid_hd95):.2f} +/- {np.std(valid_hd95):.2f}")
        print(f"  Median: {np.median(valid_hd95):.2f}")
        print(f"  Range: [{np.min(valid_hd95):.2f}, {np.max(valid_hd95):.2f}]")

    if valid_asd:
        print(f"\nAverage Surface Distance (mm):")
        print(f"  Mean: {np.mean(valid_asd):.2f} +/- {np.std(valid_asd):.2f}")
        print(f"  Median: {np.median(valid_asd):.2f}")
        print(f"  Range: [{np.min(valid_asd):.2f}, {np.max(valid_asd):.2f}]")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
