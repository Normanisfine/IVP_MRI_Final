#!/bin/bash

#SBATCH --job-name=3d_fullres
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --account=pr_31_tandon_advanced

module purge

# Execute the 3d_fullres training within the Singularity container
singularity exec --overlay /vast/sw5672/mri/prostate/overlay-25GB-500K.ext3:ro --nv \
    /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
    /bin/bash -c '

    # Source the environment script to activate Conda
    source /ext3/env.sh

    # Set nnU-Net environment paths
    export nnUNet_raw=/vast/sw5672/mri/prostate_data/raw/prostate158_train
    export nnUNet_preprocessed=/vast/sw5672/mri/prostate_data/raw/prostate158_train/preprocessed
    export nnUNet_results=/vast/sw5672/mri/prostate_data/raw/prostate158_train/results

    # Train Dataset300 3d_fullres, fold all
    nnUNetv2_train 1000 3d_fullres 0 -p nnUNetPlans -tr nnUNetTrainer --npz \
        &> /vast/sw5672/mri/prostate/logs/train_nnunet-3d_fullres-${SLURM_JOB_ID}
'

echo "Job finished."

