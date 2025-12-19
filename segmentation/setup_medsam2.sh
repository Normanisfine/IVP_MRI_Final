#!/bin/bash

set -e
cd "$(dirname "$0")"
cp configs/medsam2/*.yaml MedSAM2/sam2/configs/
cat patches/npz_dual_modality_dataset.py >> MedSAM2/training/dataset/vos_raw_dataset.py
cd MedSAM2 && pip install -e ".[dev]"

