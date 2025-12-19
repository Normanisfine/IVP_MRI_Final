#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


METHODS = [
    'nnU-Net\n(T2+ADC)',
    'MedSAM2\nBaseline',
    'MedSAM2\nFinetuned\n(ADC)',
    'MedSAM2\nFinetuned\n(T2+ADC)'
]
DICE_SCORES = [0.315, 0.220, 0.526, 0.522]
HD95_SCORES = [None, 49.4, 13.1, 8.6]
ASD_SCORES = [None, 7.8, 2.2, 1.6]
COLORS = ['steelblue', 'lightcoral', 'coral', 'mediumorchid']


def create_dice_figure(output_dir):
    fig, ax = plt.subplots(figsize=(10, 7))

    x = np.arange(len(METHODS))
    bars = ax.bar(x, DICE_SCORES, color=COLORS, edgecolor='black', linewidth=2, width=0.7)

    for i, (bar, score) in enumerate(zip(bars, DICE_SCORES)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                f'{score:.3f}', ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax.annotate('', xy=(2, 0.526), xytext=(0, 0.315),
                arrowprops=dict(arrowstyle='->', color='green', lw=3,
                               connectionstyle="arc3,rad=-0.3"))
    ax.text(1, 0.48, '+67%', fontsize=14, color='green', fontweight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_ylabel('Dice Coefficient', fontsize=14)
    ax.set_xlabel('Method', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(METHODS, fontsize=11)
    ax.set_ylim(0, 0.7)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=2)
    ax.text(3.5, 0.51, 'Dice = 0.5', fontsize=10, color='gray')
    ax.grid(axis='y', alpha=0.3)

    plt.title('Tumor Dice Score Comparison\n(higher is better)', fontsize=16, fontweight='bold', pad=15)
    plt.tight_layout()

    output_path = f"{output_dir}/medsam2_dice_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def create_hd95_figure(output_dir):
    fig, ax = plt.subplots(figsize=(9, 7))

    methods_hd = ['MedSAM2\nBaseline', 'MedSAM2\nFinetuned\n(ADC)', 'MedSAM2\nFinetuned\n(T2+ADC)']
    hd95_vals = [49.4, 13.1, 8.6]
    colors_hd = ['lightcoral', 'coral', 'mediumorchid']

    x_hd = np.arange(len(methods_hd))
    bars_hd = ax.bar(x_hd, hd95_vals, color=colors_hd, edgecolor='black', linewidth=2, width=0.6)

    for bar, score in zip(bars_hd, hd95_vals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 1.5,
                f'{score:.1f} mm', ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax.annotate('', xy=(2, 8.6), xytext=(0, 49.4),
                arrowprops=dict(arrowstyle='->', color='green', lw=3,
                               connectionstyle="arc3,rad=0.3"))
    ax.text(1, 32, '-83%', fontsize=14, color='green', fontweight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_ylabel('HD95 (mm)', fontsize=14)
    ax.set_xlabel('Method', fontsize=14)
    ax.set_xticks(x_hd)
    ax.set_xticklabels(methods_hd, fontsize=11)
    ax.set_ylim(0, 60)
    ax.grid(axis='y', alpha=0.3)

    plt.title('Hausdorff Distance 95% Comparison\n(lower is better)', fontsize=16, fontweight='bold', pad=15)
    plt.tight_layout()

    output_path = f"{output_dir}/medsam2_hd95_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def create_timeline_figure(output_dir):
    fig, ax = plt.subplots(figsize=(14, 6))

    y_timeline = 0.5
    x_positions = [0.12, 0.37, 0.62, 0.87]

    ax.plot([0.05, 0.95], [y_timeline, y_timeline], 'k-', linewidth=4)

    point_sizes = [400, 300, 600, 600]

    for i, (x_pos, method, dice, color, size) in enumerate(zip(
            x_positions, METHODS, DICE_SCORES, COLORS, point_sizes)):

        ax.scatter([x_pos], [y_timeline], s=size, c=[color], edgecolors='black',
                   linewidths=3, zorder=3)

        ax.text(x_pos, y_timeline + 0.28, method.replace('\n', ' '),
                ha='center', va='bottom', fontsize=12, fontweight='bold')

        ax.text(x_pos, y_timeline - 0.18, f'Dice: {dice:.3f}',
                ha='center', va='top', fontsize=13, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.4, edgecolor='black'))

    annotations = [
        (0.245, y_timeline + 0.12, 'Domain gap\n(worse than nnU-Net)', 'red'),
        (0.495, y_timeline + 0.12, 'Fine-tuning\n(+2.4x Dice)', 'green'),
        (0.745, y_timeline + 0.12, 'Dual-modality\n(better boundary)', 'blue'),
    ]

    for x_ann, y_ann, text, color in annotations:
        ax.text(x_ann, y_ann, text, fontsize=11, ha='center',
                color=color, fontweight='bold')

    for i in range(len(x_positions) - 1):
        ax.annotate('', xy=(x_positions[i+1] - 0.08, y_timeline),
                    xytext=(x_positions[i] + 0.08, y_timeline),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=3))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.title('Methodology Journey: From nnU-Net to Fine-tuned MedSAM2',
              fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()

    output_path = f"{output_dir}/medsam2_timeline.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    output_dir = "IVP_final"
    os.makedirs(output_dir, exist_ok=True)

    create_dice_figure(output_dir)
    create_hd95_figure(output_dir)
    create_timeline_figure(output_dir)


if __name__ == "__main__":
    main()
