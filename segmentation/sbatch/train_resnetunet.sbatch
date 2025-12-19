#!/bin/bash

#SBATCH --job-name=train_resnetunet
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=20GB
#SBATCH --time=6:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --account=pr_31_tandon_advanced

module purge

OVERLAY="${SINGULARITY_OVERLAY:-/vast/sw5672/mri/prostate/overlay-15GB-500K.ext3:ro}"
SIF="${SINGULARITY_SIF:-/scratch/work/public/singularity/cuda12.5.1-cudnn9.2.1-ubuntu22.04.5.sif}"

singularity exec --overlay "${OVERLAY}" --nv "${SIF}" /bin/bash -c '

    source /ext3/env.sh

    WORK_DIR="${PROSTATE_WORK_DIR:-/vast/sw5672/mri/prostate}"
    DATA_DIR="${PROSTATE_DATA_DIR:-/vast/sw5672/mri/prostate_data/raw/prostate158_train}"

    python "${WORK_DIR}/scripts/train_resnetunet.py" \
        --train_dir "${DATA_DIR}/train_sliced" \
        --val_dir "${DATA_DIR}/val_sliced" \
        --log_dir "${WORK_DIR}/logs/resnetunet_${SLURM_JOB_ID}" \
        --epochs 150 \
        --batch_size 64 \
        2>&1 | tee "${WORK_DIR}/logs/train_resnetunet_${SLURM_JOB_ID}.log"
'

echo "Job finished."
