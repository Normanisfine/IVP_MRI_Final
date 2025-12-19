#!/bin/bash

#SBATCH --job-name=medsam2_dual
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --account=pr_31_tandon_advanced

module purge

# Execute MedSAM2 finetuning with DUAL MODALITY (T2 + ADC)
singularity exec --overlay /vast/sw5672/mri/prostate/MedSAM2/overlay-15GB-500K.ext3:ro --nv \
    /scratch/work/public/singularity/cuda12.5.1-cudnn9.2.1-ubuntu22.04.5.sif \
    /bin/bash -c '

    source /ext3/env.sh
    conda activate medsam2

    cd /vast/sw5672/mri/prostate/MedSAM2

    export PROSTATE_DATA_DIR=/vast/sw5672/mri/prostate_data/npz
    export CUDA_VISIBLE_DEVICES=0

    mkdir -p /vast/sw5672/mri/prostate/logs

    echo "=== Starting DUAL MODALITY (T2 + ADC) Training ==="
    echo "Channel mapping: R=T2, G=ADC, B=average(T2,ADC)"

    python training/train.py \
        -c configs/sam2.1_hiera_tiny_finetune_prostate_dual.yaml \
        --output-path /vast/sw5672/mri/prostate/finetune_prostate_dual \
        --use-cluster 0 \
        --num-gpus 1 \
        --num-nodes 1 \
        2>&1 | tee /vast/sw5672/mri/prostate/logs/finetune_medsam2_dual_${SLURM_JOB_ID}.log
'

echo "Job finished."
