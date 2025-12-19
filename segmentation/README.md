# Segmentation

Prostate MRI segmentation using MedSAM2, ResNet-UNet, and nnU-Net.

## Structure

```
segmentation/
├── train_scripts/          # Training scripts
│   ├── medsam2/           # MedSAM2 fine-tuning & inference
│   ├── resnetunet/        # ResNet18-UNet baseline
│   └── nnunet/            # nnU-Net v2
├── data/                  # Data preparation
├── comparison/            # Evaluation & metrics
├── visualize/             # Visualization
├── configs/medsam2/       # Custom MedSAM2 configs
├── patches/               # MedSAM2 patches
└── sbatch/                # SLURM batch scripts
```

## Quick Start

**1. Setup MedSAM2:**
```bash
git submodule update --init --recursive
./setup_medsam2.sh
```

**2. Prepare data:**
```bash
python data/convert_prostate_to_npz.py --data_root /path/to/data --output /path/to/npz
```

**3. Train:**
```bash
# MedSAM2
sbatch sbatch/finetune_medsam2.sbatch

# ResNet-UNet
sbatch sbatch/train_resnetunet.sbatch

# nnU-Net
sbatch sbatch/train_nnunet.sbatch
```

**4. Evaluate:**
```bash
python comparison/compute_dice.py --gt /path/to/gt --pred /path/to/pred
```

