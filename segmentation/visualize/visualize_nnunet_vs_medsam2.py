#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nibabel as nib
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


def load_nnunet_prediction(case_id, tumor_label=3):
    nnunet_id = case_id - 1
    nifti_path = f"inference/adc_t2_tumor/Prostate_{nnunet_id:03d}.nii.gz"

    if not os.path.exists(nifti_path):
        return None

    img = nib.load(nifti_path)
    data = img.get_fdata().astype(np.int32)

    tumor_mask = (data == tumor_label).astype(np.uint8)
    tumor_mask = np.transpose(tumor_mask, (2, 1, 0))  # (Z, H, W)

    return tumor_mask


def load_medsam_prediction(pred_dir, case_id):
    npz_path = f"{pred_dir}/prostate_{case_id:03d}_pred.npz"
    if not os.path.exists(npz_path):
        return None
    data = np.load(npz_path)
    return (data['segs'] > 0).astype(np.uint8)


def load_test_data(case_id):
    npz_path = f"npz/test/prostate_{case_id:03d}.npz"
    if not os.path.exists(npz_path):
        return None
    data = np.load(npz_path)
    return {
        'imgs': data['imgs'],
        'gts': data['gts'],
        'z_prompt': int(data.get('z_prompt', len(data['imgs']) // 2)),
    }


def visualize_case(case_id, output_dir):
    test_data = load_test_data(case_id)
    if test_data is None:
        print(f"  Skipping case {case_id}: test data not found")
        return

    imgs = test_data['imgs']
    gts = test_data['gts']
    z_prompt = test_data['z_prompt']

    nnunet_pred = load_nnunet_prediction(case_id)
    medsam_adc = load_medsam_prediction('inference/finetuned_box', case_id)
    medsam_dual = load_medsam_prediction('inference/dual_box', case_id)

    tumor_slices = np.where(np.any(gts > 0, axis=(1, 2)))[0]
    if len(tumor_slices) == 0:
        print(f"  Skipping case {case_id}: no tumor in GT")
        return

    if z_prompt in tumor_slices:
        z = z_prompt
    else:
        z = tumor_slices[len(tumor_slices) // 2]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    img = imgs[z]
    gt = gts[z]
    bbox = get_bbox_from_mask(gt, margin=5)

    axes[0].imshow(img, cmap='gray')
    if bbox:
        draw_bbox(axes[0], bbox, color='yellow', linewidth=2)
    axes[0].set_title(f'Slice {z}: Image + Box', fontsize=10)
    axes[0].axis('off')

    axes[1].imshow(img, cmap='gray')
    axes[1].contour(gt, colors='lime', linewidths=2)
    axes[1].set_title('Ground Truth', fontsize=10)
    axes[1].axis('off')

    axes[2].imshow(img, cmap='gray')
    axes[2].contour(gt, colors='lime', linewidths=1.5, linestyles='--')
    if nnunet_pred is not None and z < len(nnunet_pred) and np.any(nnunet_pred[z] > 0):
        axes[2].contour(nnunet_pred[z], colors='blue', linewidths=2)
    axes[2].set_title('nnUNet (blue)', fontsize=10)
    axes[2].axis('off')

    axes[3].imshow(img, cmap='gray')
    axes[3].contour(gt, colors='lime', linewidths=1.5, linestyles='--')
    if medsam_adc is not None and np.any(medsam_adc[z] > 0):
        axes[3].contour(medsam_adc[z], colors='red', linewidths=2)
    axes[3].set_title('MedSAM2-ADC (red)', fontsize=10)
    axes[3].axis('off')

    axes[4].imshow(img, cmap='gray')
    axes[4].contour(gt, colors='lime', linewidths=1.5, linestyles='--')
    if medsam_dual is not None and np.any(medsam_dual[z] > 0):
        axes[4].contour(medsam_dual[z], colors='magenta', linewidths=2)
    axes[4].set_title('MedSAM2-Dual (magenta)', fontsize=10)
    axes[4].axis('off')

    plt.suptitle(f'Case {case_id:03d}: nnUNet vs MedSAM2 (Green dashed = GT)', fontsize=12)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/case_{case_id:03d}_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()


def load_all_scores():
    scores = {}

    scores['medsam_adc'] = pd.read_csv('scores/finetuned_box_scores.csv/per_case_metrics.csv')
    scores['medsam_dual'] = pd.read_csv('scores/dual_box_scores.csv/per_case_metrics.csv')

    nnunet_df = pd.read_csv('inference/adc_t2_tumor/dice_result.csv')
    scores['nnunet'] = pd.DataFrame({
        'case': [f"prostate_{int(row['case'].split('_')[1])+1:03d}" for _, row in nnunet_df.iterrows()],
        'dice': nnunet_df['dice_label_3'].values
    })

    return scores


def create_summary_plots(scores, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    models = ['nnunet', 'medsam_adc', 'medsam_dual']
    labels = ['nnUNet\n(T2+ADC)', 'MedSAM2\n(ADC)', 'MedSAM2\n(T2+ADC)']
    colors = ['steelblue', 'coral', 'mediumorchid']

    n_cases = min(len(scores[m]['dice']) for m in models)

    x = np.arange(n_cases)
    width = 0.25

    for i, (model, label, color) in enumerate(zip(models, labels, colors)):
        dice_values = scores[model]['dice'].values[:n_cases]
        axes[0, 0].bar(x + i*width, dice_values, width, label=label.replace('\n', ' '), color=color)

    axes[0, 0].set_xlabel('Case')
    axes[0, 0].set_ylabel('Dice Coefficient')
    axes[0, 0].set_title('Tumor Dice Score per Case')
    axes[0, 0].legend(loc='upper right')
    axes[0, 0].set_xticks(x + width)
    axes[0, 0].set_xticklabels([f'{i+1}' for i in range(n_cases)], fontsize=8)

    data = [scores[m]['dice'].dropna().values[:n_cases] for m in models]
    bp = axes[0, 1].boxplot(data, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    axes[0, 1].set_ylabel('Dice Coefficient')
    axes[0, 1].set_title('Tumor Dice Distribution')

    for i, d in enumerate(data):
        mean_val = np.mean(d)
        axes[0, 1].annotate(f'{mean_val:.3f}', xy=(i+1, mean_val),
                           xytext=(i+1.2, mean_val+0.05), fontsize=9)

    means = [np.nanmean(scores[m]['dice'].values[:n_cases]) for m in models]
    stds = [np.nanstd(scores[m]['dice'].values[:n_cases]) for m in models]
    x_pos = np.arange(len(models))

    bars = axes[1, 0].bar(x_pos, means, yerr=stds, capsize=5, color=colors, alpha=0.8)
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(labels)
    axes[1, 0].set_ylabel('Mean Dice')
    axes[1, 0].set_title('Mean Tumor Dice Comparison')
    axes[1, 0].set_ylim(0, 0.8)

    for i, (m, s) in enumerate(zip(means, stds)):
        axes[1, 0].text(i, m + s + 0.02, f'{m:.3f}', ha='center', fontsize=10, fontweight='bold')

    axes
    table_data = []
    for model, label in zip(models, ['nnUNet (T2+ADC)', 'MedSAM2 (ADC)', 'MedSAM2 (T2+ADC)']):
        dice = scores[model]['dice'].dropna().values[:n_cases]
        table_data.append([
            label,
            f'{np.mean(dice):.3f}',
            f'{np.std(dice):.3f}',
            f'{np.median(dice):.3f}',
            f'{np.min(dice):.3f}',
            f'{np.max(dice):.3f}'
        ])

    table = axes[1, 1].table(
        cellText=table_data,
        colLabels=['Model', 'Mean', 'Std', 'Median', 'Min', 'Max'],
        loc='center',
        cellLoc='center',
        colColours=['lightgray']*6
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    axes[1, 1].set_title('Tumor Segmentation Statistics (Dice)', fontsize=12, pad=20)

    plt.suptitle('nnUNet vs MedSAM2: Prostate Tumor Segmentation', fontsize=14, fontweight='bold')
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/summary_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    return means, stds


def main():
    output_dir = "visualizations/nnunet_vs_medsam2"

    test_files = sorted(glob("npz/test/prostate_*.npz"))
    case_ids = [int(Path(f).stem.split('_')[1]) for f in test_files]

    print(f"Found {len(case_ids)} test cases")
    print(f"Output directory: {output_dir}")

    print("\nGenerating per-case visualizations...")
    for case_id in case_ids:
        try:
            visualize_case(case_id, output_dir)
            print(f"  Saved case {case_id:03d}")
        except Exception as e:
            print(f"  Error case {case_id}: {e}")

    print("\nLoading scores...")
    scores = load_all_scores()

    print("Creating summary plots...")
    means, stds = create_summary_plots(scores, output_dir)

    print(f"\nAll visualizations saved to: {output_dir}/")


if __name__ == "__main__":
    main()
