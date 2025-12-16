# Method Comparison

Evaluation results for Score-based and VarNet reconstruction methods.

## Files

**Scripts:**
- `test_score_compare.py` - Score (SENSE & SSOS) evaluation
- `test_varnet_compare.py` - VarNet evaluation
- `plot.py` - Visualization

**Results:**
- `results_equi_acc4.txt` - Detailed metrics (5 test files)
- `results_summary.txt` - Summary
- `score_convergence.png` - Training convergence plot

**Logs:**
- `score_*.err/.out` - Comparison job logs

## Results (4× Acceleration)

**All Methods:**

| Method | PSNR (dB) | MSE | Time |
|--------|-----------|-----|------|
| VarNet | 34.6 ± 3.8 | 2.72e-11 | 0.24 s/slice |
| SSOS | 27.6 ± 1.5 | 2.49e-03 | ~97 min |
| SENSE | 24.4 ± 4.6 | 9.80e-03 | ~7.7 min |

**Key Findings:**
- VarNet achieves **best quality** (7 dB better than SSOS)
- VarNet is **fastest** (24,000× faster than SSOS)
- SENSE offers 12.5× speedup over SSOS for score-based methods

## Usage

```bash
# Score comparison
python test_score_compare.py \
  --data /path/to/file.h5 \
  --checkpoint checkpoint_50.pth

# VarNet comparison
python test_varnet_compare.py \
  --data /path/to/file.h5
```

## Recommendation

- **Best choice**: Use VarNet (highest quality + fastest)
- **Score-based alternative**: Use SSOS for diffusion-based reconstruction
- **Fast score-based**: Use SENSE for 12.5× speedup over SSOS
