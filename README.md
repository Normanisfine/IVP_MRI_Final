# IVP_MRI_Final

Final project for MRI reconstruction and segmentation using deep learning.

## Structure

```
IVP_MRI_Final/
├── reconstruction/         # MRI reconstruction
├── segmentation/          # Prostate MRI segmentation
└── IVP_final.pdf          # Final report
```

## Reconstruction Results

**FastMRI Knee Multi-Coil (4× acceleration):**

| Method | PSNR (dB) | Time | Notes |
|--------|-----------|------|-------|
| VarNet | 34.6 ± 3.8 | 0.24 s/slice | **Best quality & fastest** |
| Score SSOS | 27.6 ± 1.5 | ~97 min | High quality |
| Score SENSE | 24.4 ± 4.6 | ~7.7 min | Fast |

**Dataset**: 197 FastMRI files (70% train, 15% val, 15% test)

## Modified Score-MRI

This project uses a modified score-based implementation optimized for HPC:

**Repository**: [https://github.com/Normanisfine/score-MRI-mod](https://github.com/Normanisfine/score-MRI-mod)

**Key features**:
- HPC training with TensorFlow/PyTorch conflict fix
- SENSE inference (12.5× speedup over SSOS)
- FastMRI multi-coil support

## Usage

See `reconstruction/README.md` for detailed instructions.

```bash
cd reconstruction

# Train models
./train_scripts/score/train.sh
./train_scripts/varnet/run.sh

# Compare methods
python comparison/test_score_compare.py
python comparison/test_varnet_compare.py
```

## Segmentation

Prostate MRI segmentation using MedSAM2, ResNet-UNet, and nnU-Net. See `segmentation/README.md`.
Raw and processed data is available at [Google Drive](https://drive.google.com/file/d/1VgHM50tVTUltojJX5XuHhXkGFjRJBNkT/view?usp=sharing)

## References

- Modified Score-MRI: [github.com/Normanisfine/score-MRI-mod](https://github.com/Normanisfine/score-MRI-mod)
- FastMRI: [github.com/facebookresearch/fastMRI](https://github.com/facebookresearch/fastMRI)
- MedSAM2: [github.com/bowang-lab/MedSAM2](https://github.com/bowang-lab/MedSAM2)
