# Dataset Split

Split FastMRI data into train/validation/test sets.

## Files

- `split_data.py` - Dataset splitting script
- `split_summary.txt` - Summary of current split

## Current Split

- **Training**: 70% (137 files)
- **Validation**: 15% (29 files)
- **Test**: 15% (31 files)
- **Random seed**: 42 (reproducible)

## Usage

```bash
python split_data.py \
  --source_dir /path/to/fastmri/data \
  --output_dir /path/to/output \
  --train_ratio 0.7 \
  --val_ratio 0.15 \
  --test_ratio 0.15
```

## Output

Creates three directories:
- `multicoil_train/` - Training files
- `multicoil_val/` - Validation files
- `multicoil_test/` - Test files
