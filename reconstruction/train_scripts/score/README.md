# Score-based Training

Train NCSN++ diffusion model for MRI reconstruction.

## Files

**Training:**
- `config.py` - Training configuration
- `train.sh` - Local training script
- `score.sbatch` - SLURM batch job

**Evaluation:**
- `sweep_sense_checkpoints_equi_acc4.py` - Test multiple checkpoints
- `sweep_sense_equi_acc4.txt` - Checkpoint results
- `test_sense_steps_firstfile.py` - Test different step counts

**Logs & Docs:**
- `score_*.err/.out` - Training logs
- `SEGFAULT_FIX_LOG.md` - TensorFlow/PyTorch fix (important!)

## Quick Start

**Local:**
```bash
./train.sh
```

**HPC:**
```bash
sbatch score.sbatch
```

## Configuration

- Model: NCSN++ (score-based diffusion)
- Image size: 320×320
- Batch size: 1
- Learning rate: 1e-4

## Best Results

- Checkpoint: checkpoint_50.pth
- PSNR: 24.3 dB
- Time: ~389 seconds/image

## Important

Read `SEGFAULT_FIX_LOG.md` if you get GPU errors. Shows how to fix TensorFlow/PyTorch conflicts.

## Modified Implementation

Uses: [https://github.com/Normanisfine/score-MRI-mod](https://github.com/Normanisfine/score-MRI-mod)
