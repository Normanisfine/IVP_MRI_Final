import os
os.environ['MPLBACKEND'] = 'Agg'  # Set backend before any matplotlib import

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for HPC
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Define paths
base_path = Path("/scratch/ml8347/MRI/train/compare/outputs_equi_acc4/multi-coil")
files = ['file1000060', 'file1000089', 'file1000094', 'file1000311', 'file1000331']

# Collect data for both methods
sense_data = []
ssos_data = []

for file in files:
    # Read SENSE data
    sense_csv = base_path / "SENSE/progress" / file / "recon_progress/psnr.csv"
    df_sense = pd.read_csv(sense_csv)
    sense_data.append(df_sense)
    
    # Read SSOS data
    ssos_csv = base_path / "SSOS/progress" / file / "recon_progress/psnr.csv"
    df_ssos = pd.read_csv(ssos_csv)
    ssos_data.append(df_ssos)

# Calculate mean and std across cases
steps = sense_data[0]['step'].values

sense_psnr_all = np.array([df['psnr_db'].values for df in sense_data])
ssos_psnr_all = np.array([df['psnr_db'].values for df in ssos_data])

sense_mean = sense_psnr_all.mean(axis=0)
sense_std = sense_psnr_all.std(axis=0)
ssos_mean = ssos_psnr_all.mean(axis=0)
ssos_std = ssos_psnr_all.std(axis=0)

# Create the plot
plt.figure(figsize=(12, 7))

# Plot mean curves with error bands
plt.plot(steps, sense_mean, 'b-', linewidth=2, label='SCORE-SENSE', marker='o', markersize=6)
plt.fill_between(steps, sense_mean - sense_std, sense_mean + sense_std, alpha=0.2, color='blue')

plt.plot(steps, ssos_mean, 'r-', linewidth=2, label='SCORE-SSOS', marker='s', markersize=6)
plt.fill_between(steps, ssos_mean - ssos_std, ssos_mean + ssos_std, alpha=0.2, color='red')

# Mark optimal points
sense_optimal_idx = np.argmax(sense_mean)
ssos_optimal_idx = np.argmax(ssos_mean)

plt.scatter(steps[sense_optimal_idx], sense_mean[sense_optimal_idx], 
           s=200, c='blue', marker='*', edgecolors='black', linewidths=2, 
           label=f'SENSE optimal (step {steps[sense_optimal_idx]:.0f})', zorder=5)

plt.scatter(steps[ssos_optimal_idx], ssos_mean[ssos_optimal_idx], 
           s=200, c='red', marker='*', edgecolors='black', linewidths=2, 
           label=f'SSOS optimal (step {steps[ssos_optimal_idx]:.0f})', zorder=5)

# Add annotations
plt.annotate(f'{sense_mean[sense_optimal_idx]:.2f} dB', 
            xy=(steps[sense_optimal_idx], sense_mean[sense_optimal_idx]),
            xytext=(10, 20), textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.5', fc='blue', alpha=0.3),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

plt.annotate(f'{ssos_mean[ssos_optimal_idx]:.2f} dB', 
            xy=(steps[ssos_optimal_idx], ssos_mean[ssos_optimal_idx]),
            xytext=(10, -30), textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.5', fc='red', alpha=0.3),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

# Mark key milestones
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
plt.axvline(x=300, color='gray', linestyle=':', alpha=0.3, linewidth=1)
plt.text(300, plt.ylim()[1]*0.95, 'Step 300\n(positive PSNR)', 
         ha='center', fontsize=9, alpha=0.7)

# Formatting
plt.xlabel('Diffusion Sampling Steps', fontsize=14, fontweight='bold')
plt.ylabel('PSNR (dB)', fontsize=14, fontweight='bold')
plt.title('SCORE Reconstruction Convergence Analysis\n(Averaged over 5 FastMRI Test Cases)', 
         fontsize=16, fontweight='bold')
plt.legend(loc='lower right', fontsize=11, framealpha=0.9)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()

# Save figure
plt.savefig('score_convergence.png', dpi=300, bbox_inches='tight')
plt.savefig('score_convergence.pdf', bbox_inches='tight')
print("Figure saved as score_convergence.png and score_convergence.pdf")
plt.close()

# Print optimal step information
print("\n" + "="*60)
print("OPTIMAL STEP ANALYSIS")
print("="*60)
print(f"\nSENSE optimal: Step {steps[sense_optimal_idx]:.0f} → {sense_mean[sense_optimal_idx]:.2f} ± {sense_std[sense_optimal_idx]:.2f} dB")
print(f"SSOS optimal:  Step {steps[ssos_optimal_idx]:.0f} → {ssos_mean[ssos_optimal_idx]:.2f} ± {ssos_std[ssos_optimal_idx]:.2f} dB")

print(f"\nCurrent reported (Step 550):")
print(f"  SENSE: {sense_mean[-1]:.2f} ± {sense_std[-1]:.2f} dB")
print(f"  SSOS:  {ssos_mean[-1]:.2f} ± {ssos_std[-1]:.2f} dB")

if sense_optimal_idx != len(steps) - 1:
    improvement = sense_mean[sense_optimal_idx] - sense_mean[-1]
    print(f"\n⚠️  SENSE could improve by {improvement:.2f} dB using step {steps[sense_optimal_idx]:.0f}")

print("\nPlot generation complete!")