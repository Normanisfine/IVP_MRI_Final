#!/usr/bin/env python3
import argparse
import csv
import os
import sys
from typing import Dict, List, Set, Tuple

import numpy as np

try:
    import nibabel as nib
except Exception as e:
    print("Failed to import nibabel. Please install it: pip install nibabel", file=sys.stderr)
    raise


def _apply_lcc_to_prediction(pred: np.ndarray, num_classes: int, background: int = 0) -> np.ndarray:
    """Keep only the largest connected component for each foreground class in `pred`.

    This emulates MONAI's KeepLargestConnectedComponent applied to predictions after discretization
    and one-hot conversion (include_background=False effectively means classes 1..num_classes-1).
    """
    try:
        from scipy import ndimage as ndi
    except Exception:
        return pred

    refined = pred.copy()
    for label in range(1, int(num_classes)):
        class_mask = refined == label
        if not class_mask.any():
            continue
        labeled, num = ndi.label(class_mask)
        if num <= 1:
            continue
        counts = np.bincount(labeled.ravel())
        counts[0] = 0
        largest_id = int(np.argmax(counts))
        keep_mask = labeled == largest_id
        refined[class_mask] = background
        refined[keep_mask] = label
    return refined

def load_nii_int(path: str) -> np.ndarray:
    img = nib.load(path)
    # Use get_fdata for compatibility; cast to int for label maps
    data = img.get_fdata(dtype=np.float32)
    return data.astype(np.int32)


def dice_score_binary(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    intersection = np.logical_and(a, b).sum(dtype=np.int64)
    a_sum = a.sum(dtype=np.int64)
    b_sum = b.sum(dtype=np.int64)
    denom = a_sum + b_sum
    if denom == 0:
        # Both empty => define Dice as 1.0 for background agreement
        return 1.0
    return 2.0 * intersection / float(denom)


def dice_score_label(gt: np.ndarray, pred: np.ndarray, label: int) -> float:
    gt_mask = gt == label
    pred_mask = pred == label
    a_sum = gt_mask.sum(dtype=np.int64)
    b_sum = pred_mask.sum(dtype=np.int64)
    if a_sum + b_sum == 0:
        # If the label is absent in both, return NaN so it won't affect averages
        return float("nan")
    return dice_score_binary(gt_mask, pred_mask)


def collect_case_list(gt_dir: str, pred_dir: str) -> List[str]:
    gt_files = {f for f in os.listdir(gt_dir) if f.endswith(".nii.gz")}
    pred_files = {f for f in os.listdir(pred_dir) if f.endswith(".nii.gz")}
    common = sorted(gt_files.intersection(pred_files))
    if not common:
        raise RuntimeError("No common .nii.gz files found between GT and predictions.")
    return common


def compute_dice_for_cases(
    gt_dir: str,
    pred_dir: str,
    cases: List[str],
    num_classes: int = 0,
    apply_lcc: bool = True,
) -> Tuple[Dict[str, float], Dict[str, Dict[int, float]], Set[int]]:
    per_case_fg: Dict[str, float] = {}
    per_case_labels: Dict[str, Dict[int, float]] = {}
    all_labels: Set[int] = set()

    for fname in cases:
        gt_path = os.path.join(gt_dir, fname)
        pred_path = os.path.join(pred_dir, fname)

        gt = load_nii_int(gt_path)
        pred = load_nii_int(pred_path)

        if gt.shape != pred.shape:
            raise ValueError(f"Shape mismatch for {fname}: gt {gt.shape} vs pred {pred.shape}")

        if num_classes and num_classes > 0:
            case_num_classes = int(num_classes)
        else:
            case_max = int(max(gt.max(initial=0), pred.max(initial=0)))
            case_num_classes = case_max + 1  # background + labels

        if apply_lcc:
            pred = _apply_lcc_to_prediction(pred, num_classes=case_num_classes, background=0)

        per_case_fg[fname] = dice_score_binary(gt > 0, pred > 0)

        labels = np.arange(1, int(case_num_classes), dtype=np.int32)

        label_scores: Dict[int, float] = {}
        for label in labels.tolist():
            ds = dice_score_label(gt, pred, int(label))
            label_scores[int(label)] = ds
            all_labels.add(int(label))

        per_case_labels[fname] = label_scores

    return per_case_fg, per_case_labels, all_labels


def write_csv(out_csv: str, cases: List[str], per_case_fg: Dict[str, float], per_case_labels: Dict[str, Dict[int, float]], all_labels: Set[int]) -> None:
    label_cols = [f"dice_label_{l}" for l in sorted(all_labels)]
    fieldnames = ["case", "dice_fg"] + label_cols
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for fname in cases:
            row = {"case": os.path.splitext(os.path.splitext(fname)[0])[0], "dice_fg": per_case_fg.get(fname, float("nan"))}
            label_scores = per_case_labels.get(fname, {})
            for l in sorted(all_labels):
                key = f"dice_label_{l}"
                val = label_scores.get(l, float("nan"))
                row[key] = val
            writer.writerow(row)


def summarize(per_case_fg: Dict[str, float], per_case_labels: Dict[str, Dict[int, float]]) -> Tuple[float, Dict[int, float]]:
    fg_values = [v for v in per_case_fg.values() if np.isfinite(v)]
    mean_fg = float(np.mean(fg_values)) if fg_values else float("nan")

    label_to_values: Dict[int, List[float]] = {}
    for label_scores in per_case_labels.values():
        for l, v in label_scores.items():
            if np.isnan(v):
                continue
            label_to_values.setdefault(l, []).append(v)

    mean_per_label: Dict[int, float] = {}
    for l, vals in label_to_values.items():
        if len(vals) > 0:
            mean_per_label[l] = float(np.mean(vals))
        else:
            mean_per_label[l] = float("nan")

    return mean_fg, mean_per_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Dice between label maps and predictions (NIfTI .nii.gz)")
    parser.add_argument("--gt", required=True, help="Ground-truth directory containing .nii.gz labels")
    parser.add_argument("--pred", required=True, help="Predictions directory containing .nii.gz labels")
    parser.add_argument("--out", required=False, default=None, help="Optional path to save CSV results")
    parser.add_argument("--num_classes", type=int, default=0, help="Total number of classes (including background). If 0, infer per case.")
    parser.add_argument("--no_lcc", action="store_true", help="Disable Largest Connected Component filtering on predictions.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gt_dir = args.gt
    pred_dir = args.pred
    out_csv = args.out
    num_classes = int(args.num_classes) if hasattr(args, "num_classes") else 0
    apply_lcc = not getattr(args, "no_lcc", False)

    if not os.path.isdir(gt_dir):
        raise NotADirectoryError(f"GT directory not found: {gt_dir}")
    if not os.path.isdir(pred_dir):
        raise NotADirectoryError(f"Prediction directory not found: {pred_dir}")

    cases = collect_case_list(gt_dir, pred_dir)
    per_case_fg, per_case_labels, all_labels = compute_dice_for_cases(
        gt_dir, pred_dir, cases, num_classes=num_classes, apply_lcc=apply_lcc
    )

    if out_csv is None:
        out_csv = os.path.join(pred_dir, "dice_results.csv")
    write_csv(out_csv, cases, per_case_fg, per_case_labels, all_labels)

    mean_fg, mean_per_label = summarize(per_case_fg, per_case_labels)

    all_dice_values = []
    for label_scores in per_case_labels.values():
        for v in label_scores.values():
            if np.isfinite(v):
                all_dice_values.append(v)
    overall_mean_dice = float(np.mean(all_dice_values)) if all_dice_values else float("nan")

    print(f"Saved per-case Dice to: {out_csv}")
    print(f"Overall MeanDice (trainer-equivalent): {overall_mean_dice:.4f}")
    print(f"Mean foreground Dice: {mean_fg:.4f}")
    if mean_per_label:
        parts = [f"label {l}: {m:.4f}" for l, m in sorted(mean_per_label.items())]
        print("Mean per-label Dice: " + ", ".join(parts))


if __name__ == "__main__":
    main()



