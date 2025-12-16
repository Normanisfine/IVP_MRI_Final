# VarNet Training

Train VarNet model for MRI reconstruction.

## Files

- `train_varnet.py` - PyTorch Lightning training script
- `run.sh` - Training execution script

## Quick Start

```bash
./run.sh
```

## Configuration

- Architecture: 12 cascades
- Image size: 320×320
- Batch size: 1
- Learning rate: 0.0003
- Mixed precision: 16-bit

## Training

Runs for 50 epochs with:
- Automatic checkpointing (top 3 + last)
- Early stopping (patience=10)
- TensorBoard logging

Checkpoints saved to `varnet_training/checkpoints/`

## Reference

Based on FastMRI: [https://github.com/facebookresearch/fastMRI](https://github.com/facebookresearch/fastMRI)
