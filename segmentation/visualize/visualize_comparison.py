#!/usr/bin/env python3
"""
Visualize comparison between finetuned and baseline MedSAM2 predictions.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob

def get_bbox_from_mask(mask_2d, margin=5):
    """Get bounding box from a 2D mask with margin."""
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
    """Draw bounding box on axis."""
    if bbox is None:
        return
    x_min, y_min, x_max, y_max = bbox
    rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                          fill=False, edgecolor=color, linewidth=linewidth, linestyle='-')
    ax.add_patch(rect)


def visualize_case(case_id, output_dir):
    """Visualize a single case with GT, finetuned pred, and baseline pred."""

    # Load data
    test_npz = np.load(f"npz/test/prostate_{case_id:03d}.npz")
    imgs = test_npz['imgs']
    gts = test_npz['gts']
    z_prompt = int(test_npz.get('z_prompt', len(imgs) // 2))
    bbox_stored = test_npz.get('bbox', None)

    # Load predictions
    finetuned_npz = np.load(f"inference/finetuned_box/prostate_{case_id:03d}_pred.npz")
    baseline_npz = np.load(f"inference/baseline_box/prostate_{case_id:03d}_pred.npz")

    finetuned_pred = finetuned_npz['segs']
    baseline_pred = baseline_npz['segs']

    # Find slices with tumor
    tumor_slices = np.where(np.any(gts > 0, axis=(1, 2)))[0]
    if len(tumor_slices) == 0:
        print(f"No tumor in case {case_id}")
        return

    # Select 3 representative slices including prompt slice
    n_slices = min(3, len(tumor_slices))
    if n_slices == 1:
        selected = [tumor_slices[0]]
    elif n_slices == 2:
        selected = [tumor_slices[0], tumor_slices[-1]]
    else:
        # Include prompt slice if it has tumor
        if z_prompt in tumor_slices:
            mid_idx = np.where(tumor_slices == z_prompt)[0][0]
        else:
            mid_idx = len(tumor_slices) // 2
        selected = [tumor_slices[0], tumor_slices[mid_idx], tumor_slices[-1]]
        # Ensure prompt slice is included
        if z_prompt not in selected and z_prompt in tumor_slices:
            selected[1] = z_prompt

    # Create figure
    fig, axes = plt.subplots(n_slices, 4, figsize=(16, 4 * n_slices))
    if n_slices == 1:
        axes = axes.reshape(1, -1)

    for row, z in enumerate(selected):
        img = imgs[z]
        gt = gts[z]
        ft_pred = finetuned_pred[z]
        bl_pred = baseline_pred[z]

        # Compute bbox for this slice from GT
        bbox = get_bbox_from_mask(gt, margin=5)
        is_prompt_slice = (z == z_prompt)

        # Use same color for all bounding boxes
        box_color = 'yellow'
        box_lw = 2

        # Original image with bbox
        axes[row, 0].imshow(img, cmap='gray')
        if bbox is not None:
            draw_bbox(axes[row, 0], bbox, color=box_color, linewidth=box_lw)
        slice_label = f'Slice {z} (PROMPT)' if is_prompt_slice else f'Slice {z}'
        axes[row, 0].set_title(f'{slice_label}: Image + Box', fontsize=10)
        axes[row, 0].axis('off')

        # Ground truth with bbox
        axes[row, 1].imshow(img, cmap='gray')
        axes[row, 1].contour(gt, colors='lime', linewidths=2)
        if bbox is not None:
            draw_bbox(axes[row, 1], bbox, color=box_color, linewidth=box_lw)
        axes[row, 1].set_title('Ground Truth + Box')
        axes[row, 1].axis('off')

        # Finetuned prediction with bbox
        axes[row, 2].imshow(img, cmap='gray')
        axes[row, 2].contour(gt, colors='lime', linewidths=2, linestyles='--')
        if np.any(ft_pred > 0):
            axes[row, 2].contour(ft_pred, colors='red', linewidths=2)
        if bbox is not None:
            draw_bbox(axes[row, 2], bbox, color=box_color, linewidth=box_lw)
        axes[row, 2].set_title('Finetuned (red) vs GT (green)')
        axes[row, 2].axis('off')

        # Baseline prediction with bbox
        axes[row, 3].imshow(img, cmap='gray')
        axes[row, 3].contour(gt, colors='lime', linewidths=2, linestyles='--')
        if np.any(bl_pred > 0):
            axes[row, 3].contour(bl_pred, colors='blue', linewidths=2)
        if bbox is not None:
            draw_bbox(axes[row, 3], bbox, color=box_color, linewidth=box_lw)
        axes[row, 3].set_title('Baseline (blue) vs GT (green)')
        axes[row, 3].axis('off')

    plt.suptitle(f'Case {case_id:03d}: Finetuned vs Baseline MedSAM2 (Yellow Box = GT Bounding Box)', fontsize=12)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/case_{case_id:03d}_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization for case {case_id:03d}")


def create_summary_plot(output_dir):
    """Create summary bar chart comparing metrics."""
    import pandas as pd

    # Load scores
    ft_scores = pd.read_csv("scores/finetuned_box_scores.csv/per_case_metrics.csv")
    bl_scores = pd.read_csv("scores/baseline_box_scores.csv/per_case_metrics.csv")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Dice comparison
    x = np.arange(len(ft_scores))
    width = 0.35
    axes[0].bar(x - width/2, ft_scores['dice'], width, label='Finetuned', color='coral')
    axes[0].bar(x + width/2, bl_scores['dice'], width, label='Baseline', color='steelblue')
    axes[0].set_xlabel('Case')
    axes[0].set_ylabel('Dice Coefficient')
    axes[0].set_title('Dice: Finetuned vs Baseline')
    axes[0].legend()
    axes[0].axhline(y=ft_scores['dice'].mean(), color='coral', linestyle='--', alpha=0.7)
    axes[0].axhline(y=bl_scores['dice'].mean(), color='steelblue', linestyle='--', alpha=0.7)

    # HD95 comparison
    axes[1].bar(x - width/2, ft_scores['hd95'], width, label='Finetuned', color='coral')
    axes[1].bar(x + width/2, bl_scores['hd95'], width, label='Baseline', color='steelblue')
    axes[1].set_xlabel('Case')
    axes[1].set_ylabel('HD95 (mm)')
    axes[1].set_title('HD95: Finetuned vs Baseline (lower is better)')
    axes[1].legend()

    # Summary box plot
    data = [ft_scores['dice'], bl_scores['dice']]
    bp = axes[2].boxplot(data, labels=['Finetuned', 'Baseline'], patch_artist=True)
    bp['boxes'][0].set_facecolor('coral')
    bp['boxes'][1].set_facecolor('steelblue')
    axes[2].set_ylabel('Dice Coefficient')
    axes[2].set_title('Dice Distribution')

    plt.suptitle('MedSAM2 Finetuning Results on Prostate Tumor Segmentation', fontsize=14)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/summary_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved summary plot to {output_dir}/summary_comparison.png")


if __name__ == "__main__":
    output_dir = "visualizations/comparison"

    # Get all test case IDs
    test_files = sorted(glob("npz/test/prostate_*.npz"))
    case_ids = [int(Path(f).stem.split('_')[1]) for f in test_files]

    print(f"Found {len(case_ids)} test cases")

    # Visualize each case
    for case_id in case_ids:
        try:
            visualize_case(case_id, output_dir)
        except Exception as e:
            print(f"Error visualizing case {case_id}: {e}")

    # Create summary plot
    try:
        create_summary_plot(output_dir)
    except Exception as e:
        print(f"Error creating summary plot: {e}")

    print(f"\nAll visualizations saved to: {output_dir}/")
