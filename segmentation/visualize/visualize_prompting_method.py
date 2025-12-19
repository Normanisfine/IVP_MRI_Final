#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from glob import glob
from pathlib import Path


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


def load_example_case(case_id=3):
    npz_path = f"npz/test/prostate_{case_id:03d}.npz"
    data = np.load(npz_path)
    return {
        'imgs': data['imgs'],
        'gts': data['gts'],
        'z_prompt': int(data.get('z_prompt', len(data['imgs']) // 2)),
    }


def create_slice_selection_figure(output_dir):

    data = load_example_case(case_id=3)
    imgs = data['imgs']
    gts = data['gts']
    z_prompt = data['z_prompt']

    tumor_slices = np.where(np.any(gts > 0, axis=(1, 2)))[0]

    fig, ax = plt.subplots(figsize=(12, 5))

    n_show = min(7, len(imgs))
    start_idx = max(0, z_prompt - n_show // 2)
    end_idx = min(len(imgs), start_idx + n_show)

    slice_strip = []
    for i, z in enumerate(range(start_idx, end_idx)):
        slice_strip.append(imgs[z])

    strip_img = np.hstack(slice_strip)
    ax.imshow(strip_img, cmap='gray')

    slice_width = imgs.shape[2]
    z_prompt_rel = z_prompt - start_idx

    rect = patches.Rectangle(
        (z_prompt_rel * slice_width, 0),
        slice_width, imgs.shape[1],
        linewidth=4, edgecolor='yellow', facecolor='none'
    )
    ax.add_patch(rect)

    for i, z in enumerate(range(start_idx, end_idx)):
        label_color = 'yellow' if z == z_prompt else 'white'
        weight = 'bold' if z == z_prompt else 'normal'
        fontsize = 14 if z == z_prompt else 12
        ax.text(i * slice_width + slice_width/2, -15, f'z={z}',
                ha='center', fontsize=fontsize, color=label_color, fontweight=weight)

    for i, z in enumerate(range(start_idx, end_idx)):
        if z in tumor_slices:
            ax.plot(i * slice_width + slice_width/2, imgs.shape[1] + 20,
                    'o', color='lime', markersize=12)

    ax.set_xlim(-10, strip_img.shape[1] + 10)
    ax.set_ylim(imgs.shape[1] + 45, -40)
    ax.axis('off')

    ax.text(strip_img.shape[1]/2, imgs.shape[1] + 40,
            'Green dots = tumor-containing slices; Yellow box = selected prompt slice (middle tumor slice)',
            ha='center', fontsize=11, color='white',
            bbox=dict(boxstyle='round', facecolor='darkgreen', alpha=0.8))

    plt.title('Prompt Slice Selection ($z_{prompt}$ = middle tumor-containing slice)',
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()

    output_path = f"{output_dir}/medsam2_slice_selection.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def create_bbox_figure(output_dir):

    data = load_example_case(case_id=3)
    imgs = data['imgs']
    gts = data['gts']
    z_prompt = data['z_prompt']

    fig, ax = plt.subplots(figsize=(8, 8))

    img_prompt = imgs[z_prompt]
    gt_prompt = gts[z_prompt]
    bbox = get_bbox_from_mask(gt_prompt, margin=5)
    bbox_tight = get_bbox_from_mask(gt_prompt, margin=0)

    ax.imshow(img_prompt, cmap='gray')

    ax.contour(gt_prompt, colors='lime', linewidths=3)

    if bbox_tight:
        x_min, y_min, x_max, y_max = bbox_tight
        rect_tight = patches.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            linewidth=3, edgecolor='red', facecolor='none', linestyle='--'
        )
        ax.add_patch(rect_tight)

    if bbox:
        x_min, y_min, x_max, y_max = bbox
        rect_margin = patches.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            linewidth=4, edgecolor='yellow', facecolor='none'
        )
        ax.add_patch(rect_margin)

        ax.annotate('', xy=(x_max + 3, (y_min + y_max)/2),
                    xytext=(bbox_tight[2] + 3, (y_min + y_max)/2),
                    arrowprops=dict(arrowstyle='<->', color='cyan', lw=2))
        ax.text(x_max + 12, (y_min + y_max)/2, '5px\nmargin',
                fontsize=12, color='cyan', va='center', fontweight='bold')

    ax.axis('off')

    ax.plot([], [], color='lime', linewidth=3, label='Ground truth tumor mask')
    ax.plot([], [], color='red', linewidth=3, linestyle='--', label='Tight bounding box')
    ax.plot([], [], color='yellow', linewidth=4, label='Prompt box (+5px margin)')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)

    plt.title('Bounding Box Prompt Generation\n(tight box around GT mask + 5-pixel margin)',
              fontsize=14, fontweight='bold', pad=10)
    plt.tight_layout()

    output_path = f"{output_dir}/medsam2_bbox_generation.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def create_propagation_figure(output_dir):

    fig, ax = plt.subplots(figsize=(10, 8))

    n_slices = 5
    slice_height = 0.10
    slice_width_prop = 0.18

    y_positions = np.linspace(0.82, 0.18, n_slices)
    z_labels = ['$z_{min}$', '...', '$z_{prompt}$', '...', '$z_{max}$']
    colors = ['lightblue', 'lightblue', 'gold', 'lightblue', 'lightblue']

    for i, (y_pos, label, color) in enumerate(zip(y_positions, z_labels, colors)):
        rect = patches.FancyBboxPatch(
            (0.15, y_pos - slice_height/2), slice_width_prop, slice_height,
            boxstyle="round,pad=0.01", facecolor=color, edgecolor='black', linewidth=2
        )
        ax.add_patch(rect)

        ax.text(0.15 + slice_width_prop/2, y_pos, label,
                ha='center', va='center', fontsize=14, fontweight='bold')

        pred_color = 'lime' if label == '$z_{prompt}$' else 'lightgreen'
        pred_x = 0.55
        pred_rect = patches.FancyBboxPatch(
            (pred_x, y_pos - slice_height/2), slice_width_prop, slice_height,
            boxstyle="round,pad=0.01", facecolor=pred_color, edgecolor='black', linewidth=2
        )
        ax.add_patch(pred_rect)

        seg_label = 'Seg' if label != '...' else '...'
        ax.text(pred_x + slice_width_prop/2, y_pos, seg_label,
                ha='center', va='center', fontsize=12, fontweight='bold')

        ax.annotate('', xy=(pred_x - 0.02, y_pos), xytext=(0.15 + slice_width_prop + 0.02, y_pos),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    arrow_x = 0.55 + slice_width_prop + 0.06

    ax.annotate('', xy=(arrow_x, y_positions[0] + 0.02), xytext=(arrow_x, y_positions[2] - 0.04),
                arrowprops=dict(arrowstyle='->', color='blue', lw=3,
                               connectionstyle="arc3,rad=0.3"))
    ax.text(arrow_x + 0.06, (y_positions[0] + y_positions[2])/2, 'Backward\npropagation',
            fontsize=12, color='blue', ha='left', va='center', fontweight='bold')

    ax.annotate('', xy=(arrow_x, y_positions[-1] - 0.02), xytext=(arrow_x, y_positions[2] + 0.04),
                arrowprops=dict(arrowstyle='->', color='red', lw=3,
                               connectionstyle="arc3,rad=-0.3"))
    ax.text(arrow_x + 0.06, (y_positions[-1] + y_positions[2])/2, 'Forward\npropagation',
            fontsize=12, color='red', ha='left', va='center', fontweight='bold')

    ax.text(0.15 + slice_width_prop/2, 0.92, 'Input\nSlices', ha='center', fontsize=13, fontweight='bold')
    ax.text(0.55 + slice_width_prop/2, 0.92, 'Segmentation\nOutput', ha='center', fontsize=13, fontweight='bold')

    ax.annotate('Box prompt\nprovided here', xy=(0.15, y_positions[2]), xytext=(0.02, y_positions[2] + 0.08),
                fontsize=11, ha='center', va='center', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='orange', lw=2),
                bbox=dict(boxstyle='round', facecolor='orange', alpha=0.3))

    explanation = (
        "MedSAM2 treats 3D volume as video frames.\n"
        "Prompt on single slice propagates bidirectionally\n"
        "using learned temporal consistency priors."
    )
    ax.text(0.5, 0.05, explanation, ha='center', va='bottom', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
            transform=ax.transAxes)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.title('3D Propagation via Video Object Segmentation (VOS)',
              fontsize=14, fontweight='bold', pad=10)
    plt.tight_layout()

    output_path = f"{output_dir}/medsam2_3d_propagation.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    output_dir = "IVP_final"
    os.makedirs(output_dir, exist_ok=True)

    create_slice_selection_figure(output_dir)
    create_bbox_figure(output_dir)
    create_propagation_figure(output_dir)
    print("Done!")


if __name__ == "__main__":
    main()
