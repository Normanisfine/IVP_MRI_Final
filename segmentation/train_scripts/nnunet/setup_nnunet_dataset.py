import glob
import os
from pathlib import Path
import shutil
import json
import pandas as pd
import SimpleITK as sitk
import numpy as np

def move_resized_to_dir(mri_root, nnunet_root, create_test=False):
    t2_path = sorted(
        glob.glob(mri_root + "/*/t2_resized.nii.gz"),
        key=lambda p: int(Path(p).parents[0].name)
    )
    print("T2 files found:", len(t2_path))
    Path(nnunet_root + "/imagesTr").mkdir(parents=True, exist_ok=True)
    Path(nnunet_root + "/labelsTr").mkdir(parents=True, exist_ok=True)
    train_ids = pd.read_csv(Path(mri_root).parents[0] / "train.csv")["ID"].tolist()
    val_ids = pd.read_csv(Path(mri_root).parents[0] / "valid.csv")["ID"].tolist()
    train_ids = train_ids + val_ids
    if create_test:
        Path(nnunet_root + "/imagesTs").mkdir(parents=True, exist_ok=True)
        Path(nnunet_root + "/labelsTs").mkdir(parents=True, exist_ok=True)
        test_ids = [int(x) for x in pd.read_csv(Path(mri_root).parents[0] / "test.csv")["ID"].tolist()]
        print("Test ids:", len(test_ids))
        print(test_ids)
    label_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

    for i, t2 in enumerate(t2_path):
        subject_id = int(Path(t2).parents[0].name)
        anatomy_src = Path(t2).with_name('t2_anatomy_reader1_resized.nii.gz')

        img_dir = nnunet_root + "/imagesTr/" + 'Prostate_' + f"{i:03d}" + "_0000.nii.gz"
        label_dir = nnunet_root + "/labelsTr/" + 'Prostate_' + f"{i:03d}" + ".nii.gz"
        if create_test and subject_id in test_ids:
            img_dir = nnunet_root + "/imagesTs/" + 'Prostate_' + f"{i:03d}" + "_0000.nii.gz"
            label_dir = nnunet_root + "/labelsTs/" + 'Prostate_' + f"{i:03d}" + ".nii.gz"


        t2_img = sitk.ReadImage(t2)

        t2_dir = img_dir.replace("_0000.nii.gz", "_0000.nii.gz")
        print('Writing T2 to', t2_dir)
        sitk.WriteImage(t2_img, t2_dir)

        adc_src = Path(t2).with_name('adc_resized.nii.gz')
        adc_dir = img_dir.replace("_0000.nii.gz", "_0001.nii.gz")
        if adc_src.exists():
            adc_img = sitk.ReadImage(str(adc_src))
            adc_resampled = sitk.Resample(
                adc_img,
                t2_img,
                sitk.Transform(),
                sitk.sitkLinear,
                0.0,
                sitk.sitkFloat32
            )
        else:
            adc_resampled = sitk.Image(t2_img.GetSize(), sitk.sitkFloat32)
            adc_resampled.CopyInformation(t2_img)
        print('Writing ADC to', adc_dir)
        sitk.WriteImage(adc_resampled, adc_dir)

        if anatomy_src.exists():
            anatomy_img = sitk.ReadImage(str(anatomy_src))
            anatomy_arr = sitk.GetArrayFromImage(anatomy_img)
        else:
            anatomy_arr = np.zeros(t2_img.GetSize()[::-1], dtype=np.uint8)
        
        t2_tumor_src = Path(t2).with_name('t2_tumor_reader1_resized.nii.gz')
        if t2_tumor_src.exists():
            t2_tumor_img = sitk.ReadImage(str(t2_tumor_src))
            t2_tumor_arr = sitk.GetArrayFromImage(t2_tumor_img)
        else:
            t2_tumor_arr = np.zeros(t2_img.GetSize()[::-1], dtype=np.uint8)
        
        adc_tumor_src = Path(t2).with_name('adc_tumor_reader1_resized.nii.gz')
        if adc_tumor_src.exists():
            adc_tumor_img = sitk.ReadImage(str(adc_tumor_src))
            adc_tumor_arr = sitk.GetArrayFromImage(adc_tumor_img)
        else:
            adc_tumor_arr = np.zeros(t2_img.GetSize()[::-1], dtype=np.uint8)
        
        combined_arr = anatomy_arr.copy()
        combined_arr[t2_tumor_arr > 0] = 3
        combined_arr[adc_tumor_arr > 0] = 4
        
        combined = sitk.GetImageFromArray(combined_arr)
        combined.CopyInformation(t2_img)

        print('Writing combined label to', label_dir)
        sitk.WriteImage(combined, label_dir)

        vals, cnts = np.unique(combined_arr, return_counts=True)
        for v, c in zip(vals.tolist(), cnts.tolist()):
            v_int = int(v)
            if v_int not in label_counts:
                label_counts[v_int] = 0
            label_counts[v_int] += int(c)

    stats = {
        "num_training_subjects": len(train_ids),
        "label_counts": {str(k): int(v) for k, v in label_counts.items()},
        "present_labels": [int(k) for k, v in label_counts.items() if v > 0],
    }
    stats_path = nnunet_root + "/label_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print("Wrote label statistics to", stats_path)

    missing = [l for l in (1, 2, 3, 4) if label_counts.get(l, 0) == 0]
    if missing:
        print(f"WARNING: Missing foreground labels in dataset (no voxels): {missing}. This can cause NaNs during evaluation.")

    return len(train_ids)

def create_json(nnunet_root, json_data):
    json_path = nnunet_root + "/dataset.json"
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=4)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Setup nnU-Net dataset from prostate MRI data")
    parser.add_argument("--mri_root", type=str, required=True,
                        help="Path to train directory with subject folders containing resized MRI files")
    parser.add_argument("--nnunet_root", type=str, required=True,
                        help="Output path for nnU-Net dataset (e.g., path/to/Dataset1000_Prostate)")
    parser.add_argument("--create_test", action="store_true",
                        help="Create test split from test.csv")
    args = parser.parse_args()

    mri_root = args.mri_root
    nnunet_root = args.nnunet_root
    num_training = move_resized_to_dir(mri_root, nnunet_root, create_test=args.create_test)
    json_data = { 
                "channel_names": {
                    "0": "T2",
                    "1": "ADC"
                }, 
                "labels": { 
                    "background": 0,
                    "TZ": 1,
                    "PZ": 2,
                    "T2_tumor": 3,
                    "ADC_tumor": 4
                }, 
                "numTraining": num_training, 
                "file_ending": ".nii.gz",
                "overwrite_image_reader_writer": "SimpleITKIO"
                }
    create_json(nnunet_root, json_data)
if __name__ == "__main__":
    main()