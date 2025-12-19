#!/bin/bash

#SBATCH --job-name=visualize
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --account=pr_31_tandon_advanced

module purge

singularity exec --overlay /vast/sw5672/mri/prostate/MedSAM2/overlay-15GB-500K.ext3:ro \
    /scratch/work/public/singularity/cuda12.5.1-cudnn9.2.1-ubuntu22.04.5.sif \
    /bin/bash -c '

    source /ext3/env.sh
    conda activate medsam2

    cd /vast/sw5672/mri/prostate

    python visualize_comparison.py
'

echo "Visualization complete."
