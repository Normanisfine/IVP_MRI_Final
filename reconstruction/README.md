# Reconstruction Scripts

Training and evaluation for MRI reconstruction methods.

## Structure

```
reconstruction/
├── train_scripts/          # Training scripts
│   ├── score/             # Score-based diffusion
│   └── varnet/            # VarNet
├── data_split/            # Dataset splitting
└── comparison/            # Results & comparison
```

## Quick Start

**1. Split dataset:**
```bash
cd data_split
python split_data.py
```

**2. Train models:**
```bash
# Score (local)
cd train_scripts/score
./train.sh

# Score (HPC)
sbatch score.sbatch

# VarNet
cd train_scripts/varnet
./run.sh
```

**3. Evaluate:**
```bash
cd comparison
python test_score_compare.py
python test_varnet_compare.py
```

## Results (4× Acceleration)

| Method | PSNR (dB) | Time | Performance |
|--------|-----------|------|-------------|
| VarNet | 34.6 ± 3.8 | 0.24 s/slice | **Best quality & speed** |
| SSOS | 27.6 ± 1.5 | ~97 min | 7 dB lower PSNR |
| SENSE | 24.4 ± 4.6 | ~7.7 min | 10 dB lower PSNR |

**Dataset**: 197 FastMRI knee files (137 train, 29 val, 31 test)

## Best Checkpoint

- Checkpoint: checkpoint_50.pth
- PSNR: 24.3 dB
- Tested steps: 50, 70, 90, 110, 130, 150, 160

## Modified Score-MRI

Uses: [github.com/Normanisfine/score-MRI-mod](https://github.com/Normanisfine/score-MRI-mod)

Key improvements:
- HPC training (TensorFlow/PyTorch fix)
- SENSE inference (12.5× speedup)
- FastMRI compatibility
