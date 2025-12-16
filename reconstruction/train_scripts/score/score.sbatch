#!/bin/bash

#SBATCH --job-name=score
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=015:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

module purge

# Navigate to the project directory
cd /scratch/ml8347/MRI/train/score || { echo "Error: Could not find 'gaussian-splatting' directory."; exit 1; }

# Define the path 
DATASET_PATH="/scratch/ml8347/3dgs/input/1"
OUTPUT_PATH="/scratch/ml8347/3dgs/output/2" 
OVERLAY_PATH="/scratch/ml8347/MRI/scoreMRI/env/overlay-25GB-500K.ext3"
IMAGE_PATH="/scratch/ml8347/MRI/base_env/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif"


# Check paths
echo "Starting training job..."
echo "Using dataset at: ${DATASET_PATH}"
echo "Output will be saved to: ${OUTPUT_PATH}"
echo "Overlay: ${OVERLAY_PATH}"
echo "Singularity image: ${IMAGE_PATH}"

# Execute the training script within the Singularity container
singularity exec --overlay ${OVERLAY_PATH}:ro --nv \
    ${IMAGE_PATH} \
    /bin/bash -c '

     # Source the environment script to activate Conda
    source /ext3/env.sh

    # Activate the pre-existing Conda environment
    conda activate score-POCS
    ./train.sh
'

echo "Job finished."
