#!/bin/bash

# Set GPU device
export CUDA_VISIBLE_DEVICES=0

# PyTorch CUDA settings
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=1
export TENSORBOARD_DISABLED=1

# Set paths
SCORE_MRI_DIR="/scratch/ml8347/MRI/scoreMRI/score-MRI-mod"
DATASET_DIR="/scratch/ml8347/MRI/train/train_dataset"
WORKDIR="/scratch/ml8347/MRI/train/score/workdir3/fastmri_multicoil_knee_320"
EVAL_DIR="/scratch/ml8347/MRI/train/score/eval3/fastmri_multicoil_knee_320"

# Create directories
mkdir -p "$WORKDIR"
mkdir -p "$EVAL_DIR"

# Change to score-MRI directory
cd "$SCORE_MRI_DIR"

echo "Starting multicoil knee training..."
echo "Dataset: $DATASET_DIR"
echo "Workdir: $WORKDIR"
echo "Eval dir: $EVAL_DIR"

# Run training (TensorFlow GPU conflict fixed in main_fastmri.py)
python main_fastmri.py \
  --config=/scratch/ml8347/MRI/train/score/config.py \
  --eval_folder="$EVAL_DIR" \
  --mode='train' \
  --workdir="$WORKDIR"

echo "Training completed!"
