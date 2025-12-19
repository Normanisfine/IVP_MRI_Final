#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path
import json

def environ_setup(nnunet_root):
    os.environ["nnUNet_raw"] = nnunet_root
    os.environ["nnUNet_preprocessed"] = nnunet_root + "/preprocessed"
    os.environ["nnUNet_results"] = nnunet_root + "/results"
    print("nnUNet_raw:", os.environ["nnUNet_raw"])
    print("nnUNet_preprocessed:", os.environ["nnUNet_preprocessed"])
    print("nnUNet_results:", os.environ["nnUNet_results"])

def assert_dataset_dir_exists(nnunet_root, dataset_id):
    expected_dir = Path(nnunet_root) / f"Dataset{dataset_id}_Prostate"
    if not expected_dir.exists():
        print(f"Expected dataset directory not found: {expected_dir}")
        try:
            available = [p.name for p in Path(nnunet_root).iterdir() if p.is_dir()]
            print("Available directories:", available)
        except Exception:
            pass
        sys.exit(1)

def preprocess(nnunet_root, dataset_id):
    try:
        subprocess.run(["nnUNetv2_plan_and_preprocess", "-d", f"{dataset_id}", "--verify_dataset_integrity"], check=True)
    except Exception as e:
        print(e)
        print("Preprocessing failed")
        sys.exit(1)

def get_available_configs(nnunet_root, dataset_id):
    plans_path = Path(nnunet_root) / "preprocessed" / f"Dataset{dataset_id}_Prostate" / "nnUNetPlans.json"
    try:
        with open(plans_path, "r") as f:
            plans = json.load(f)
        configs_dict = plans.get("configurations", {})
        configs = list(configs_dict.keys()) if isinstance(configs_dict, dict) else list(configs_dict)
        # Prefer typical order
        priority = {"2d": 0, "3d_fullres": 1, "3d_lowres": 2}
        configs.sort(key=lambda c: priority.get(c, 99))
        if not configs:
            return ["2d", "3d_fullres"]
        return configs
    except Exception:
        return ["2d", "3d_fullres"]

def find_best_config(nnunet_root, dataset_id):
    try:
        subprocess.run(["nnUNetv2_find_best_configuration", f"{dataset_id}"], check=True)
    except Exception as e:
        print(e)
        print("Finding best configuration failed")
        sys.exit(1)

def train(nnunet_root, dataset_id, config, fold="0", trainer="nnUNetTrainer", plans="nnUNetPlans", save_npz=True):
    try:
        cmd = [
            "nnUNetv2_train",
            f"{dataset_id}",
            f"{config}",
            f"{fold}",
            "-p", plans,
            "-tr", trainer,
        ]
        if save_npz:
            cmd.append("--npz")
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(e)
        print("Training failed")
        sys.exit(1)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train nnU-Net v2 on prostate dataset")
    parser.add_argument("--nnunet_root", type=str, required=True,
                        help="nnU-Net root directory containing Dataset*_Prostate folder")
    parser.add_argument("--dataset_id", type=str, default="100",
                        help="Dataset ID (default: 100)")
    parser.add_argument("--config", type=str, default="3d_fullres",
                        help="nnU-Net configuration (default: 3d_fullres)")
    parser.add_argument("--find_best", action="store_true",
                        help="Run nnUNetv2_find_best_configuration after training")
    args = parser.parse_args()

    nnunet_root = args.nnunet_root
    train_path = nnunet_root + "/imagesTr"
    val_path = nnunet_root + "/imagesTs"
    dataset_id = args.dataset_id
    find_best = args.find_best
    config = args.config

    environ_setup(nnunet_root)
    assert_dataset_dir_exists(nnunet_root, dataset_id)

    preprocess(nnunet_root, dataset_id)

    # Train all available configurations across all folds (0-4)
    available = get_available_configs(nnunet_root, dataset_id)
    print("Available configurations:", available)
    # folds = ["0"]
    # for cfg in available:
    #     for fold in folds:
    #         print(f"Training config={cfg}, fold={fold}")
    #         train(nnunet_root, dataset_id, cfg, fold=fold)

    # if find_best:
    #     find_best_config(nnunet_root, dataset_id)
if __name__ == "__main__":
    main()