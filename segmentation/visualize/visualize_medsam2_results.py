#!/usr/bin/env python3

import os
import argparse
from glob import glob
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from tqdm import tqdm


def load_case_data(npz_dir: Path, case_name: str) -> dict:
    npz_path = npz_dir / f"{case_name}.npz"
    if not npz_path.exists():
        return None

    data = np.load(npz_path, allow_pickle=True)
    return {
        'imgs': data['imgs'],
        'imgs_adc': data.get('imgs_adc', data['imgs']),
        'imgs_t2': data.get('imgs_t2', data['imgs']),
        'gts': data['gts'],
        'bbox': data['bbox'],
        'z_prompt': int(data['z_prompt']),
        'case_id': data.get('case_id', case_name),
    }


def load_prediction(pred_dir: Path, case_name: str) -> np.ndarray:
    pred_path = pred_dir / f"{case_name}_pred.npz"
    if not pred_path.exists():
        return None

    data = np.load(pred_path, allow_pickle=True)
    return data['segs']


def create_overlay(img: np.ndarray, mask: np.ndarray, color: tuple, alpha: float = 0.5) -> np.ndarray:
    img_norm = img.astype(float)
    if img_norm.max() > 0:
        img_norm = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min())

    rgb = np.stack([img_norm, img_norm, img_norm], axis=-1)

    mask_bool = mask > 0
    for i, c in enumerate(color):
        rgb[:, :, i] = np.where(mask_bool, rgb[:, :, i] * (1 - alpha) + c * alpha, rgb[:, :, i])

    return rgb


def visualize_case(
    case_name: str,
    npz_dir: Path,
    inference_dirs: dict,
    output_dir: Path,
    prompt_types: list = None,
):
    if prompt_types is None:
        prompt_types = ['box', 'center', 'multi', 'grid']

    data = load_case_data(npz_dir, case_name)
    if data is None:
        print(f"  Could not load data for {case_name}")
        return False

    img = data['imgs_adc']
    gt = data['gts']
    z_prompt = data['z_prompt']
    bbox = data['bbox']

    predictions = {}
    for pt in prompt_types:
        if pt in inference_dirs:
            pred = load_prediction(inference_dirs[pt], case_name)
            if pred is not None:
                predictions[pt] = pred

    if not predictions:
        print(f"  No predictions found for {case_name}")
        return False

    tumor_slices = np.where(np.any(gt > 0, axis=(1, 2)))[0]
    if len(tumor_slices) == 0:
        tumor_slices = [z_prompt]

    slices_to_show = sorted(set([
        tumor_slices[0],
        z_prompt,
        tumor_slices[-1],
    ]))

    colors = {
        'gt': (0, 1, 0),      # Green for ground truth
        'box': (1, 0, 0),     # Red
        'center': (0, 0, 1),  # Blue
        'multi': (1, 1, 0),   # Yellow
        'grid': (1, 0, 1),    # Magenta
    }

    n_cols = 2 + len(predictions)
    n_rows = len(slices_to_show)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(f'{case_name} - MedSAM2 Segmentation Results', fontsize=14, fontweight='bold')

    for row_idx, slice_idx in enumerate(slices_to_show):
        img_slice = img[slice_idx]
        gt_slice = gt[slice_idx]

        ax = axes[row_idx, 0]
        ax.imshow(img_slice, cmap='gray')
        if slice_idx == z_prompt:
            x_min, y_min, x_max, y_max = bbox
            rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                  linewidth=2, edgecolor='cyan', facecolor='none')
            ax.add_patch(rect)
        ax.set_title(f'ADC (z={slice_idx})' + (' [prompt]' if slice_idx == z_prompt else ''))
        ax.axis('off')

        ax = axes[row_idx, 1]
        overlay = create_overlay(img_slice, gt_slice, colors['gt'], alpha=0.4)
        ax.imshow(overlay)
        ax.set_title('Ground Truth')
        ax.axis('off')

        for col_idx, pt in enumerate(predictions.keys()):
            ax = axes[row_idx, col_idx + 2]
            pred_slice = predictions[pt][slice_idx]

            overlay = create_overlay(img_slice, pred_slice, colors.get(pt, (1, 0, 0)), alpha=0.4)
            ax.imshow(overlay)

            if np.any(gt_slice > 0):
                ax.contour(gt_slice, levels=[0.5], colors=['lime'], linewidths=1.5)

            pred_bool = pred_slice > 0
            gt_bool = gt_slice > 0
            intersection = np.logical_and(pred_bool, gt_bool).sum()
            union = pred_bool.sum() + gt_bool.sum()
            dice = 2 * intersection / union if union > 0 else 1.0

            ax.set_title(f'{pt} (Dice={dice:.2f})')
            ax.axis('off')

    legend_elements = [
        mpatches.Patch(facecolor='lime', edgecolor='lime', alpha=0.7, label='Ground Truth'),
    ]
    for pt in predictions.keys():
        color = colors.get(pt, (1, 0, 0))
        legend_elements.append(
            mpatches.Patch(facecolor=color, alpha=0.5, label=f'{pt} prediction')
        )

    fig.legend(handles=legend_elements, loc='lower center', ncol=len(legend_elements),
               bbox_to_anchor=(0.5, 0.02), fontsize=10)

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])

    output_path = output_dir / f"{case_name}_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return True


def visualize_summary(
    npz_dir: Path,
    inference_dirs: dict,
    scores_dirs: dict,
    output_dir: Path,
):
    import pandas as pd

    metrics = {}
    for pt, scores_dir in scores_dirs.items():
        csv_path = scores_dir / "per_case_metrics.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            metrics[pt] = df

    if not metrics:
        print("No metrics files found")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('MedSAM2 Prostate Tumor Segmentation - Prompt Type Comparison',
                 fontsize=14, fontweight='bold')

    prompt_types = list(metrics.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(prompt_types)))

    ax = axes[0, 0]
    dice_data = [metrics[pt]['dice'].values for pt in prompt_types]
    bp = ax.boxplot(dice_data, labels=prompt_types, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Dice Coefficient')
    ax.set_title('Dice Score Distribution by Prompt Type')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    x = np.arange(len(prompt_types))
    width = 0.25

    mean_dice = [metrics[pt]['dice'].mean() for pt in prompt_types]
    mean_hd95 = [metrics[pt]['hd95'].mean() / 100 for pt in prompt_types]  # Scale down
    mean_asd = [metrics[pt]['asd'].mean() / 10 for pt in prompt_types]   # Scale down

    bars1 = ax.bar(x - width, mean_dice, width, label='Dice', color='steelblue')
    bars2 = ax.bar(x, mean_hd95, width, label='HD95/100', color='coral')
    bars3 = ax.bar(x + width, mean_asd, width, label='ASD/10', color='forestgreen')

    ax.set_ylabel('Value')
    ax.set_title('Mean Metrics by Prompt Type')
    ax.set_xticks(x)
    ax.set_xticklabels(prompt_types)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

    ax = axes[1, 0]
    cases = metrics[prompt_types[0]]['case'].values
    x = np.arange(len(cases))
    width = 0.2

    for i, pt in enumerate(prompt_types):
        dice_vals = metrics[pt]['dice'].values
        ax.bar(x + i * width, dice_vals, width, label=pt, alpha=0.8)

    ax.set_ylabel('Dice Coefficient')
    ax.set_title('Per-Case Dice Comparison')
    ax.set_xticks(x + width * (len(prompt_types) - 1) / 2)
    ax.set_xticklabels([c.replace('prostate_', '') for c in cases], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1, 1]
    ax.axis('off')

    table_data = []
    for pt in prompt_types:
        df = metrics[pt]
        table_data.append([
            pt,
            f"{df['dice'].mean():.3f} ± {df['dice'].std():.3f}",
            f"{df['hd95'].mean():.1f} ± {df['hd95'].std():.1f}",
            f"{df['asd'].mean():.1f} ± {df['asd'].std():.1f}",
        ])

    table = ax.table(
        cellText=table_data,
        colLabels=['Prompt Type', 'Dice (mean±std)', 'HD95 (mm)', 'ASD (mm)'],
        loc='center',
        cellLoc='center',
        colColours=['lightgray'] * 4,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    ax.set_title('Summary Metrics', pad=20)

    plt.tight_layout()

    output_path = output_dir / "summary_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved summary to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize MedSAM2 segmentation results")
    parser.add_argument(
        "--npz_dir",
        type=str,
        default="npz/test",
        help="Directory containing NPZ files"
    )
    parser.add_argument(
        "--inference_dir",
        type=str,
        default="inference",
        help="Base directory containing inference results"
    )
    parser.add_argument(
        "--scores_dir",
        type=str,
        default="scores",
        help="Base directory containing score results"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="visual/medsam2",
        help="Output directory for visualizations"
    )
    parser.add_argument(
        "--case",
        type=str,
        default=None,
        help="Specific case to visualize (e.g., prostate_001). If not specified, visualize all."
    )
    parser.add_argument(
        "--prompt_types",
        type=str,
        nargs='+',
        default=['box', 'center', 'multi', 'grid'],
        help="Prompt types to compare"
    )

    args = parser.parse_args()

    npz_dir = Path(args.npz_dir)
    inference_base = Path(args.inference_dir)
    scores_base = Path(args.scores_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    inference_dirs = {}
    scores_dirs = {}
    for pt in args.prompt_types:
        inf_dir = inference_base / pt
        if inf_dir.exists():
            inference_dirs[pt] = inf_dir
        scores_dir = scores_base / pt
        if scores_dir.exists():
            scores_dirs[pt] = scores_dir

    print(f"Found inference dirs: {list(inference_dirs.keys())}")
    print(f"Found scores dirs: {list(scores_dirs.keys())}")

    if args.case:
        cases = [args.case]
    else:
        npz_files = sorted(glob(str(npz_dir / "*.npz")))
        cases = [Path(f).stem for f in npz_files]

    print(f"Visualizing {len(cases)} cases...")

    for case_name in tqdm(cases, desc="Creating visualizations"):
        success = visualize_case(
            case_name=case_name,
            npz_dir=npz_dir,
            inference_dirs=inference_dirs,
            output_dir=output_dir,
            prompt_types=args.prompt_types,
        )
        if not success:
            print(f"  Failed to visualize {case_name}")

    print("Creating summary visualization...")
    visualize_summary(npz_dir, inference_dirs, scores_dirs, output_dir)

    print(f"\nVisualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
