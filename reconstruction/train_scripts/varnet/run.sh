#!/bin/bash

# Activate conda environment
source /ext3/miniforge3/etc/profile.d/conda.sh
export PATH=/ext3/miniforge3/bin:$PATH
conda activate fastmri

# Set paths
DATA_PATH="/scratch/ml8347/MRI/train/train_dataset"
OUTPUT_DIR="./varnet_training"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Starting VarNet training..."
echo "Dataset: $DATA_PATH"
echo "Output directory: $OUTPUT_DIR"

# Run training
python train_varnet.py \
  --data_path "$DATA_PATH" \
  --challenge multicoil \
  --mask_type equispaced_fraction \
  --center_fractions 0.08 \
  --accelerations 4 \
  --num_cascades 12 \
  --batch_size 1 \
  --num_workers 4 \
  --max_epochs 50 \
  --lr 0.0003 \
  --default_root_dir "$OUTPUT_DIR" \
  --num_gpus 1 \
  --use_amp

echo "Training completed!"
