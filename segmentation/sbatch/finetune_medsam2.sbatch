#!/bin/bash
#
# SLURM Job Script: MedSAM2 Fine-tuning for Prostate Segmentation
#
# IMPORTANT: This script requires customization for your HPC environment.
# See docs/hpc_setup.md for configuration instructions.
#
# Required customization:
#   - Line 13: Update --account to your allocation
#   - Line 18-19: Update Singularity overlay and container paths
#   - Line 28: Update PROSTATE_DATA_DIR to your data path
#   - Line 37: Update output path for checkpoints
#
#SBATCH --job-name=medsam2_ft
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --account=pr_31_tandon_advanced

module purge

OVERLAY="${SINGULARITY_OVERLAY:-/vast/sw5672/mri/prostate/overlay-15GB-500K.ext3:ro}"
SIF="${SINGULARITY_SIF:-/scratch/work/public/singularity/cuda12.5.1-cudnn9.2.1-ubuntu22.04.5.sif}"

singularity exec --overlay "${OVERLAY}" --nv "${SIF}" /bin/bash -c '

    source /ext3/env.sh
    conda activate medsam2

    WORK_DIR="${PROSTATE_WORK_DIR:-/vast/sw5672/mri/prostate}"
    cd "${WORK_DIR}/MedSAM2"

    export PROSTATE_DATA_DIR="${PROSTATE_DATA_DIR:-/vast/sw5672/mri/prostate_data/npz}"
    export CUDA_VISIBLE_DEVICES=0

    mkdir -p "${WORK_DIR}/logs"

    OUTPUT_DIR="${PROSTATE_OUTPUT_DIR:-/vast/sw5672/mri/prostate/finetune_prostate}"
    python training/train.py \
        -c sam2/configs/sam2.1_hiera_tiny_finetune_prostate.yaml \
        --output-path "${OUTPUT_DIR}" \
        --use-cluster 0 \
        --num-gpus 1 \
        --num-nodes 1 \
        2>&1 | tee "${WORK_DIR}/logs/finetune_medsam2_${SLURM_JOB_ID}.log"
'

echo "Job finished."
