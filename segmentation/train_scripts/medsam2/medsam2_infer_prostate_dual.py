#!/usr/bin/env python3
""" 
    python medsam2_infer_prostate_dual.py --input npz/test --output inference/dual_box --prompt_type box
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
    """
    Resize a 3D grayscale NumPy array to RGB and resize to target size.

    Args:
        array: Input array of shape (D, H, W)
        image_size: Desired size for width and height

    Returns:
        Resized array of shape (D, 3, image_size, image_size)
    """
    d, h, w = array.shape
    resized_array = np.zeros((d, 3, image_size, image_size))

    for i in range(d):
        img_pil = Image.fromarray(array[i].astype(np.uint8))
        img_rgb = img_pil.convert("RGB")
        img_resized = img_rgb.resize((image_size, image_size))
        img_array = np.array(img_resized).transpose(2, 0, 1)
        resized_array[i] = img_array

    return resized_array


def resize_dual_modality(t2: np.ndarray, adc: np.ndarray, image_size: int) -> np.ndarray:
    """
    Resize dual modality (T2 and ADC) images and combine into RGB channels.

    Channel mapping: R=T2, G=ADC, B=average(T2,ADC)

    Args:
        t2: T2 image array of shape (D, H, W) uint8 [0-255]
        adc: ADC image array of shape (D, H, W) uint8 [0-255]
        image_size: Desired size for width and height

    Returns:
        Resized array of shape (D, 3, image_size, image_size) float [0-1]
    """
    d, h, w = t2.shape
    resized_array = np.zeros((d, 3, image_size, image_size), dtype=np.float32)

    for i in range(d):
        # Resize T2
        t2_pil = Image.fromarray(t2[i].astype(np.uint8))
        t2_resized = np.array(t2_pil.resize((image_size, image_size))) / 255.0

        # Resize ADC
        adc_pil = Image.fromarray(adc[i].astype(np.uint8))
        adc_resized = np.array(adc_pil.resize((image_size, image_size))) / 255.0

        # Combine: R=T2, G=ADC, B=average
        avg_resized = (t2_resized + adc_resized) / 2.0

        resized_array[i, 0] = t2_resized   # R channel = T2
        resized_array[i, 1] = adc_resized  # G channel = ADC
        resized_array[i, 2] = avg_resized  # B channel = average

    return resized_array


def get_center_point(mask_2d: np.ndarray) -> np.ndarray:
    """
    Get center point of a 2D mask.

    Args:
        mask_2d: 2D binary mask (H, W)

    Returns:
        Center point as [[x, y]]
    """
    y_indices, x_indices = np.where(mask_2d > 0)
    if len(x_indices) == 0:
        return None
    center_x = int(np.mean(x_indices))
    center_y = int(np.mean(y_indices))
    return np.array([[center_x, center_y]])


def get_multi_points(mask_2d: np.ndarray, n_points: int = 5) -> np.ndarray:
    """
    Sample multiple points from a 2D mask.

    Args:
        mask_2d: 2D binary mask (H, W)
        n_points: Number of points to sample

    Returns:
        Points as array of shape (n_points, 2) with [x, y] coordinates
    """
    y_indices, x_indices = np.where(mask_2d > 0)
    if len(x_indices) == 0:
        return None
    if len(x_indices) < n_points:
        n_points = len(x_indices)

    indices = np.random.choice(len(x_indices), size=n_points, replace=False)
    points = np.stack([x_indices[indices], y_indices[indices]], axis=1)
    return points


def sample_points_in_bbox_grid(bbox: np.ndarray, n: int = 9) -> np.ndarray:
    """
    Uniformly sample n grid-aligned (x, y) points inside the bbox.

    Args:
        bbox: [x_min, y_min, x_max, y_max]
        n: Number of points to sample

    Returns:
        Points as array of shape (n, 2)
    """
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
    """
    Run MedSAM2 inference on a single case.

    Args:
        npz_path: Path to input NPZ file
        predictor: MedSAM2 predictor
        prompt_type: Type of prompt ("box", "center_point", "multi_point", "grid_point")
        output_dir: Output directory
        save_nifti: Whether to save NIfTI output
        save_overlay: Whether to save overlay visualization
        shift: Margin to add to bounding box

    Returns:
        Tuple of (case_name, inference_time)
    """
    start_time = time.time()

    npz_name = os.path.basename(npz_path)
    case_name = npz_name.replace('.npz', '')

    # Load data - DUAL MODALITY
    npz_data = np.load(npz_path, allow_pickle=True)

    # Load T2 and ADC images
    if 'imgs_t2' in npz_data and 'imgs_adc' in npz_data:
        t2_3d = npz_data['imgs_t2']   # (D, H, W) uint8
        adc_3d = npz_data['imgs_adc']  # (D, H, W) uint8
    else:
        # Fallback to single modality
        t2_3d = npz_data['imgs']
        adc_3d = npz_data['imgs']

    img_3d = npz_data['imgs']  # Keep for saving
    gts = npz_data['gts']      # (D, H, W) ground truth
    bbox = npz_data['bbox']    # [x_min, y_min, x_max, y_max]
    spacing = npz_data['spacing']
    z_prompt = int(npz_data['z_prompt'])

    # Initialize output
    segs_3d = np.zeros(t2_3d.shape, dtype=np.uint8)

    # Resize dual modality images to 512x512 and combine into RGB
    video_height, video_width = t2_3d.shape[1:3]
    img_resized = resize_dual_modality(t2_3d, adc_3d, 512)  # Already normalized to [0,1]
    img_resized = torch.from_numpy(img_resized).cuda()

    # ImageNet normalization
    img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None].cuda()
    img_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None].cuda()
    img_resized -= img_mean
    img_resized /= img_std

    # Get prompt based on type
    with torch.autocast("cuda", dtype=torch.bfloat16):
        inference_state = predictor.init_state(img_resized, video_height, video_width)

        if prompt_type == "box":
            # Add margin to bbox
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
            # Get center point from ground truth on prompt slice
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
            # Sample multiple points from ground truth
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
            # Sample grid points within bounding box
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

        # Get initial mask from prompt
        mask_prompt = (out_mask_logits[0] > 0.0).squeeze(0).cpu().numpy().astype(np.uint8)

        # Add mask and propagate forward
        frame_idx, object_ids, masks = predictor.add_new_mask(
            inference_state, frame_idx=z_prompt, obj_id=1, mask=mask_prompt
        )
        segs_3d[z_prompt] = (masks[0] > 0.0).cpu().numpy()[0]

        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
            inference_state, start_frame_idx=z_prompt, reverse=False
        ):
            segs_3d[out_frame_idx] = (out_mask_logits[0] > 0.0).cpu().numpy()[0]

        # Reset and propagate backward
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

    # Save results
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

        # Save prediction
        sitk_seg = sitk.GetImageFromArray(segs_3d)
        sitk_seg.SetSpacing(spacing.tolist())
        sitk.WriteImage(sitk_seg, str(nifti_dir / f"{case_name}_pred.nii.gz"))

        # Save ground truth
        sitk_gt = sitk.GetImageFromArray(gts)
        sitk_gt.SetSpacing(spacing.tolist())
        sitk.WriteImage(sitk_gt, str(nifti_dir / f"{case_name}_gt.nii.gz"))

        # Save image
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

    # Setup paths
    script_dir = Path(__file__).parent
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = script_dir / args.checkpoint

    # Config path needs special handling for Hydra
    # Format: //<absolute_path>/sam2.1_hiera_t512.yaml
    cfg_dir = script_dir / args.cfg
    cfg = "//" + str(cfg_dir) + "/sam2.1_hiera_t512.yaml"

    print(f"Loading model from {checkpoint}")
    print(f"Config: {cfg}")

    # Build predictor
    predictor = build_sam2_video_predictor_npz(cfg, str(checkpoint))

    # Get input files
    npz_files = sorted(glob(str(input_dir / "*.npz")))
    print(f"Found {len(npz_files)} NPZ files to process")

    # Run inference
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

    # Save timing results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "inference_times.csv", index=False)

    print(f"\nInference complete!")
    print(f"  Total cases: {len(npz_files)}")
    print(f"  Mean time per case: {np.mean(results['duration']):.2f}s")
    print(f"  Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
