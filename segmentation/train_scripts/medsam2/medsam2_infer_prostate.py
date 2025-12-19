#!/usr/bin/env python3
"""
    python medsam2_infer_prostate.py --input npz/test --output inference/box --prompt_type box
    python medsam2_infer_prostate.py --input npz/test --output inference/center --prompt_type center_point
    python medsam2_infer_prostate.py --input npz/test --output inference/multi --prompt_type multi_point
"""

import os
import sys
import argparse
import time
from glob import glob
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from PIL import Image
from tqdm import tqdm

# Add MedSAM2 to path
MEDSAM2_PATH = Path(__file__).parent / "MedSAM2"
sys.path.insert(0, str(MEDSAM2_PATH))

from sam2.build_sam import build_sam2_video_predictor_npz

# Set random seeds for reproducibility
torch.set_float32_matmul_precision('high')
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
np.random.seed(2024)


def resize_grayscale_to_rgb_and_resize(array: np.ndarray, image_size: int) -> np.ndarray:
    d, h, w = array.shape
    resized_array = np.zeros((d, 3, image_size, image_size))

    for i in range(d):
        img_pil = Image.fromarray(array[i].astype(np.uint8))
        img_rgb = img_pil.convert("RGB")
        img_resized = img_rgb.resize((image_size, image_size))
        img_array = np.array(img_resized).transpose(2, 0, 1)
        resized_array[i] = img_array

    return resized_array


def get_center_point(mask_2d: np.ndarray) -> np.ndarray:
    y_indices, x_indices = np.where(mask_2d > 0)
    if len(x_indices) == 0:
        return None
    center_x = int(np.mean(x_indices))
    center_y = int(np.mean(y_indices))
    return np.array([[center_x, center_y]])


def get_multi_points(mask_2d: np.ndarray, n_points: int = 5) -> np.ndarray:
    y_indices, x_indices = np.where(mask_2d > 0)
    if len(x_indices) == 0:
        return None
    if len(x_indices) < n_points:
        n_points = len(x_indices)

    indices = np.random.choice(len(x_indices), size=n_points, replace=False)
    points = np.stack([x_indices[indices], y_indices[indices]], axis=1)
    return points


def sample_points_in_bbox_grid(bbox: np.ndarray, n: int = 9) -> np.ndarray:
    x_min, y_min, x_max, y_max = bbox
    grid_size = int(np.ceil(np.sqrt(n)))

    x_vals = np.linspace(x_min, x_max, grid_size, dtype=int)
    y_vals = np.linspace(y_min, y_max, grid_size, dtype=int)

    xv, yv = np.meshgrid(x_vals, y_vals)
    coords = np.stack([xv.ravel(), yv.ravel()], axis=1)

    return coords[:n]


@torch.inference_mode()
def infer_case(
    npz_path: str,
    predictor,
    prompt_type: str,
    output_dir: Path,
    save_nifti: bool = True,
    save_overlay: bool = False,
    shift: int = 5
) -> tuple:
    start_time = time.time()

    npz_name = os.path.basename(npz_path)
    case_name = npz_name.replace('.npz', '')

    # Load data
    npz_data = np.load(npz_path, allow_pickle=True)
    img_3d = npz_data['imgs']  # (D, H, W) uint8
    gts = npz_data['gts']      # (D, H, W) ground truth
    bbox = npz_data['bbox']    # [x_min, y_min, x_max, y_max]
    spacing = npz_data['spacing']
    z_prompt = int(npz_data['z_prompt'])

    assert np.max(img_3d) < 256, f'Input data should be in range [0, 255], got max {np.max(img_3d)}'

    segs_3d = np.zeros(img_3d.shape, dtype=np.uint8)

    video_height, video_width = img_3d.shape[1:3]
    img_resized = resize_grayscale_to_rgb_and_resize(img_3d, 512)
    img_resized = img_resized / 255.0
    img_resized = torch.from_numpy(img_resized).cuda()

    img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None].cuda()
    img_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None].cuda()
    img_resized -= img_mean
    img_resized /= img_std

    with torch.autocast("cuda", dtype=torch.bfloat16):
        inference_state = predictor.init_state(img_resized, video_height, video_width)

        if prompt_type == "box":
            H, W = img_3d.shape[1:3]
            bbox_shifted = np.array([
                max(0, bbox[0] - shift),
                max(0, bbox[1] - shift),
                min(W - 1, bbox[2] + shift),
                min(H - 1, bbox[3] + shift)
            ])
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=z_prompt,
                obj_id=1,
                box=bbox_shifted,
            )

        elif prompt_type == "center_point":
            points = get_center_point(gts[z_prompt])
            if points is None:
                print(f"  Warning: No tumor found on prompt slice for {case_name}")
                return case_name, 0

            labels = np.ones(len(points))
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=z_prompt,
                obj_id=1,
                points=points,
                labels=labels,
            )

        elif prompt_type == "multi_point":
            points = get_multi_points(gts[z_prompt], n_points=5)
            if points is None:
                print(f"  Warning: No tumor found on prompt slice for {case_name}")
                return case_name, 0

            labels = np.ones(len(points))
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=z_prompt,
                obj_id=1,
                points=points,
                labels=labels,
            )

        elif prompt_type == "grid_point":
            points = sample_points_in_bbox_grid(bbox, n=9)
            labels = np.ones(len(points))
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=z_prompt,
                obj_id=1,
                points=points,
                labels=labels,
            )

        else:
            raise ValueError(f"Unknown prompt type: {prompt_type}")

        mask_prompt = (out_mask_logits[0] > 0.0).squeeze(0).cpu().numpy().astype(np.uint8)

        frame_idx, object_ids, masks = predictor.add_new_mask(
            inference_state, frame_idx=z_prompt, obj_id=1, mask=mask_prompt
        )
        segs_3d[z_prompt] = (masks[0] > 0.0).cpu().numpy()[0]

        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
            inference_state, start_frame_idx=z_prompt, reverse=False
        ):
            segs_3d[out_frame_idx] = (out_mask_logits[0] > 0.0).cpu().numpy()[0]

        predictor.reset_state(inference_state)
        inference_state = predictor.init_state(img_resized, video_height, video_width)

        frame_idx, object_ids, masks = predictor.add_new_mask(
            inference_state, frame_idx=z_prompt, obj_id=1, mask=mask_prompt
        )

        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
            inference_state, start_frame_idx=z_prompt, reverse=True
        ):
            segs_3d[out_frame_idx] = (out_mask_logits[0] > 0.0).cpu().numpy()[0]

        predictor.reset_state(inference_state)

    np.savez_compressed(
        output_dir / f"{case_name}_pred.npz",
        segs=segs_3d,
        gts=gts,
        spacing=spacing,
        z_prompt=z_prompt,
    )

    if save_nifti:
        nifti_dir = output_dir / "nifti"
        nifti_dir.mkdir(exist_ok=True)

        sitk_seg = sitk.GetImageFromArray(segs_3d)
        sitk_seg.SetSpacing(spacing.tolist())
        sitk.WriteImage(sitk_seg, str(nifti_dir / f"{case_name}_pred.nii.gz"))

        sitk_gt = sitk.GetImageFromArray(gts)
        sitk_gt.SetSpacing(spacing.tolist())
        sitk.WriteImage(sitk_gt, str(nifti_dir / f"{case_name}_gt.nii.gz"))

        sitk_img = sitk.GetImageFromArray(img_3d)
        sitk_img.SetSpacing(spacing.tolist())
        sitk.WriteImage(sitk_img, str(nifti_dir / f"{case_name}_img.nii.gz"))

    duration = time.time() - start_time
    return case_name, duration


def main():
    parser = argparse.ArgumentParser(description="MedSAM2 inference for prostate tumor segmentation")
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input directory containing NPZ files"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output directory for predictions"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="MedSAM2/checkpoints/MedSAM2_latest.pt",
        help="Path to MedSAM2 checkpoint"
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="MedSAM2/sam2/configs",
        help="Path to config directory"
    )
    parser.add_argument(
        "--prompt_type",
        type=str,
        default="box",
        choices=["box", "center_point", "multi_point", "grid_point"],
        help="Type of prompt to use"
    )
    parser.add_argument(
        "--shift",
        type=int,
        default=5,
        help="Margin to add to bounding box"
    )
    parser.add_argument(
        "--save_nifti",
        action="store_true",
        default=True,
        help="Save NIfTI output files"
    )
    parser.add_argument(
        "--save_overlay",
        action="store_true",
        default=False,
        help="Save overlay visualizations"
    )

    args = parser.parse_args()

    script_dir = Path(__file__).parent
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = script_dir / args.checkpoint

    cfg_dir = script_dir / args.cfg
    cfg = "//" + str(cfg_dir) + "/sam2.1_hiera_t512.yaml"

    print(f"Loading model from {checkpoint}")
    print(f"Config: {cfg}")

    predictor = build_sam2_video_predictor_npz(cfg, str(checkpoint))

    npz_files = sorted(glob(str(input_dir / "*.npz")))
    print(f"Found {len(npz_files)} NPZ files to process")

    results = OrderedDict()
    results['case'] = []
    results['duration'] = []

    for npz_path in tqdm(npz_files, desc=f"Inference ({args.prompt_type})"):
        case_name, duration = infer_case(
            npz_path=npz_path,
            predictor=predictor,
            prompt_type=args.prompt_type,
            output_dir=output_dir,
            save_nifti=args.save_nifti,
            save_overlay=args.save_overlay,
            shift=args.shift,
        )
        results['case'].append(case_name)
        results['duration'].append(duration)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "inference_times.csv", index=False)

    print(f"\nInference complete!")
    print(f"  Total cases: {len(npz_files)}")
    print(f"  Mean time per case: {np.mean(results['duration']):.2f}s")
    print(f"  Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
