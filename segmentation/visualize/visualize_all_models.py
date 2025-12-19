#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from glob import glob

def get_bbox_from_mask(mask_2d, margin=5):
    y_indices, x_indices = np.where(mask_2d > 0)
    if len(x_indices) == 0:
        return None
    H, W = mask_2d.shape
    x_min = max(0, np.min(x_indices) - margin)
    x_max = min(W - 1, np.max(x_indices) + margin)
    y_min = max(0, np.min(y_indices) - margin)
    y_max = min(H - 1, np.max(y_indices) + margin)
    return [x_min, y_min, x_max, y_max]


def draw_bbox(ax, bbox, color='yellow', linewidth=2):
    if bbox is None:
        return
    x_min, y_min, x_max, y_max = bbox
    rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                          fill=False, edgecolor=color, linewidth=linewidth)
    ax.add_patch(rect)


def visualize_case(case_id, output_dir):
    test_npz = np.load(f"npz/test/prostate_{case_id:03d}.npz")
    imgs = test_npz['imgs']
    gts = test_npz['gts']
    z_prompt = int(test_npz.get('z_prompt', len(imgs) // 2))

    baseline_npz = np.load(f"inference/baseline_box/prostate_{case_id:03d}_pred.npz")
    single_npz = np.load(f"inference/finetuned_box/prostate_{case_id:03d}_pred.npz")
    dual_npz = np.load(f"inference/dual_box/prostate_{case_id:03d}_pred.npz")

    baseline_pred = baseline_npz['segs']
    single_pred = single_npz['segs']
    dual_pred = dual_npz['segs']

    tumor_slices = np.where(np.any(gts > 0, axis=(1, 2)))[0]
    if len(tumor_slices) == 0:
        print(f"No tumor in case {case_id}")
        return

    if z_prompt in tumor_slices:
        selected = [z_prompt]
    else:
        selected = [tumor_slices[len(tumor_slices) // 2]]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    z = selected[0]
    img = imgs[z]
    gt = gts[z]
    bl_pred = baseline_pred[z]
    sg_pred = single_pred[z]
    dl_pred = dual_pred[z]
    bbox = get_bbox_from_mask(gt, margin=5)

    axes[0].imshow(img, cmap='gray')
    if bbox:
        draw_bbox(axes[0], bbox, color='yellow', linewidth=2)
    axes[0].set_title(f'Slice {z}: Image + Box', fontsize=10)
    axes[0].axis('off')

    axes[1].imshow(img, cmap='gray')
    axes[1].contour(gt, colors='lime', linewidths=2)
    if bbox:
        draw_bbox(axes[1], bbox, color='yellow', linewidth=2)
    axes[1].set_title('Ground Truth', fontsize=10)
    axes[1].axis('off')

    axes[2].imshow(img, cmap='gray')
    axes[2].contour(gt, colors='lime', linewidths=1.5, linestyles='--')
    if np.any(bl_pred > 0):
        axes[2].contour(bl_pred, colors='blue', linewidths=2)
    axes[2].set_title('Baseline (blue)', fontsize=10)
    axes[2].axis('off')

    axes[3].imshow(img, cmap='gray')
    axes[3].contour(gt, colors='lime', linewidths=1.5, linestyles='--')
    if np.any(sg_pred > 0):
        axes[3].contour(sg_pred, colors='red', linewidths=2)
    axes[3].set_title('Single/ADC (red)', fontsize=10)
    axes[3].axis('off')

    axes[4].imshow(img, cmap='gray')
    axes[4].contour(gt, colors='lime', linewidths=1.5, linestyles='--')
    if np.any(dl_pred > 0):
        axes[4].contour(dl_pred, colors='magenta', linewidths=2)
    axes[4].set_title('Dual/T2+ADC (magenta)', fontsize=10)
    axes[4].axis('off')

    plt.suptitle(f'Case {case_id:03d}: Baseline vs Single vs Dual Modality (Green dashed = GT)', fontsize=12)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/case_{case_id:03d}_all_models.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization for case {case_id:03d}")


def create_summary_plot(output_dir):
    baseline = pd.read_csv("scores/baseline_box_scores.csv/per_case_metrics.csv")
    single = pd.read_csv("scores/finetuned_box_scores.csv/per_case_metrics.csv")
    dual = pd.read_csv("scores/dual_box_scores.csv/per_case_metrics.csv")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    x = np.arange(len(baseline))
    width = 0.25
    axes[0].bar(x - width, baseline['dice'], width, label='Baseline', color='steelblue')
    axes[0].bar(x, single['dice'], width, label='Single (ADC)', color='coral')
    axes[0].bar(x + width, dual['dice'], width, label='Dual (T2+ADC)', color='mediumorchid')
    axes[0].set_xlabel('Case')
    axes[0].set_ylabel('Dice Coefficient')
    axes[0].set_title('Dice: All Models')
    axes[0].legend()
    axes[0].axhline(y=baseline['dice'].mean(), color='steelblue', linestyle='--', alpha=0.5)
    axes[0].axhline(y=single['dice'].mean(), color='coral', linestyle='--', alpha=0.5)
    axes[0].axhline(y=dual['dice'].mean(), color='mediumorchid', linestyle='--', alpha=0.5)

    axes[1].bar(x - width, baseline['hd95'], width, label='Baseline', color='steelblue')
    axes[1].bar(x, single['hd95'], width, label='Single (ADC)', color='coral')
    axes[1].bar(x + width, dual['hd95'], width, label='Dual (T2+ADC)', color='mediumorchid')
    axes[1].set_xlabel('Case')
    axes[1].set_ylabel('HD95 (mm)')
    axes[1].set_title('HD95: All Models (lower is better)')
    axes[1].legend()

    data = [baseline['dice'], single['dice'], dual['dice']]
    bp = axes[2].boxplot(data, tick_labels=['Baseline', 'Single\n(ADC)', 'Dual\n(T2+ADC)'], patch_artist=True)
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][1].set_facecolor('coral')
    bp['boxes'][2].set_facecolor('mediumorchid')
    axes[2].set_ylabel('Dice Coefficient')
    axes[2].set_title('Dice Distribution')

    means = [baseline['dice'].mean(), single['dice'].mean(), dual['dice'].mean()]
    for i, m in enumerate(means):
        axes[2].text(i+1, m + 0.02, f'{m:.3f}', ha='center', fontsize=9)

    plt.suptitle('MedSAM2: Baseline vs Single-Modality (ADC) vs Dual-Modality (T2+ADC)', fontsize=14)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/summary_all_models.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved summary plot to {output_dir}/summary_all_models.png")

    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Metric':<15} {'Baseline':>12} {'Single(ADC)':>12} {'Dual(T2+ADC)':>12}")
    print("-"*60)
    print(f"{'Dice':<15} {baseline['dice'].mean():>12.3f} {single['dice'].mean():>12.3f} {dual['dice'].mean():>12.3f}")
    print(f"{'HD95 (mm)':<15} {baseline['hd95'].mean():>12.1f} {single['hd95'].mean():>12.1f} {dual['hd95'].mean():>12.1f}")
    print(f"{'ASD (mm)':<15} {baseline['asd'].mean():>12.2f} {single['asd'].mean():>12.2f} {dual['asd'].mean():>12.2f}")
    print("="*60)


if __name__ == "__main__":
    output_dir = "visualizations/all_models"

    test_files = sorted(glob("npz/test/prostate_*.npz"))
    case_ids = [int(Path(f).stem.split('_')[1]) for f in test_files]

    print(f"Found {len(case_ids)} test cases")

    for case_id in case_ids:
        try:
            visualize_case(case_id, output_dir)
        except Exception as e:
            print(f"Error visualizing case {case_id}: {e}")

    try:
        create_summary_plot(output_dir)
    except Exception as e:
        print(f"Error creating summary plot: {e}")

    print(f"\nAll visualizations saved to: {output_dir}/")
