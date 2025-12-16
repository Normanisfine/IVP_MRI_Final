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

## Results (N=600, 4× acceleration)

**Score-based Methods:**

| Method | PSNR (dB) | MSE | Time |
|--------|-----------|-----|------|
| SSOS | 27.6 ± 1.5 | 2.49e-03 | ~97 min |
| SENSE | 24.4 ± 4.6 | 9.80e-03 | ~7.7 min |

- **Speedup**: SENSE is 12.5× faster than SSOS
- **Quality**: SSOS has 3.2 dB better PSNR

**VarNet:**
- Time: ~2-5 seconds
- PSNR: ~25-28 dB
- ~1000× faster than Score methods

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

- **Best quality**: Use SSOS
- **Speed/quality balance**: Use SENSE
- **Real-time**: Use VarNet
