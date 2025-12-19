#!/bin/bash

#SBATCH --job-name=train_nnunet
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128GB
#SBATCH --time=23:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --account=pr_31_tandon_advanced

module purge

singularity exec --overlay /vast/sw5672/mri/prostate/overlay-25GB-500K.ext3:ro --nv \
    /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
    /bin/bash -c '

    # Source the environment script to activate Conda
    source /ext3/env.sh
    
    # Run the training script with the converted data
    python /vast/sw5672/mri/prostate/train_nnunetv2.py &> /vast/sw5672/mri/prostate/logs/train_nnunet-${SLURM_JOB_ID}
'

echo "Job finished."