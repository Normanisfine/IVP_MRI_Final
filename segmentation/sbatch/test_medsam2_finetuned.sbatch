#!/bin/bash

#SBATCH --job-name=medsam2_test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --account=pr_31_tandon_advanced

module purge

# Execute MedSAM2 inference and evaluation within the Singularity container
singularity exec --overlay /vast/sw5672/mri/prostate/MedSAM2/overlay-15GB-500K.ext3:ro --nv \
    /scratch/work/public/singularity/cuda12.5.1-cudnn9.2.1-ubuntu22.04.5.sif \
    /bin/bash -c '

    source /ext3/env.sh
    conda activate medsam2

    cd /vast/sw5672/mri/prostate

    echo "=== Running inference with finetuned model (box prompt) ==="
    python medsam2_infer_prostate.py \
        --input npz/test \
        --output inference/finetuned_box \
        --checkpoint finetune_prostate/checkpoints/checkpoint_50.pt \
        --prompt_type box

    echo "=== Running inference with finetuned model (center point) ==="
    python medsam2_infer_prostate.py \
        --input npz/test \
        --output inference/finetuned_center \
        --checkpoint finetune_prostate/checkpoints/checkpoint_50.pt \
        --prompt_type center_point

    echo "=== Evaluating finetuned box predictions ==="
    python evaluate_medsam2.py \
        --pred inference/finetuned_box \
        --output scores/finetuned_box_scores.csv

    echo "=== Evaluating finetuned center predictions ==="
    python evaluate_medsam2.py \
        --pred inference/finetuned_center \
        --output scores/finetuned_center_scores.csv

    echo "=== Comparing with baseline (pretrained) model ==="
    python medsam2_infer_prostate.py \
        --input npz/test \
        --output inference/baseline_box \
        --checkpoint MedSAM2/checkpoints/MedSAM2_latest.pt \
        --prompt_type box

    python evaluate_medsam2.py \
        --pred inference/baseline_box \
        --output scores/baseline_box_scores.csv

    echo "=== All done ==="
'

echo "Job finished."
