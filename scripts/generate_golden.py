#!/usr/bin/env python3
"""
generate_golden.py — Golden Fixtures Generator for Milestone 1.

Loads official Depth Anything V2 Small (vits) checkpoint, processes demo01.jpg through
demo10.jpg from `depth-anything-v2-official/assets/examples/`, and generates:
- `golden/images/demo01.jpg` .. `demo10.jpg`
- `golden/depths/demo01_depth.npy` .. `demo10_depth.npy` (raw float32 depth arrays)
- `golden/vis/demo01_vis.png` .. `demo10_vis.png` (visualized depth maps)
- `golden/fixture_manifest.json` (metadata index with sample parameters, dimensions, statistics, sha256 checksums)
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch

# Ensure project root & official repo in sys.path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
OFFICIAL_REPO = PROJECT_ROOT / "depth-anything-v2-official"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if OFFICIAL_REPO.exists() and str(OFFICIAL_REPO) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_REPO))

from depth_anything_v2.dpt import DepthAnythingV2


def compute_sha256(filepath: Path) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def generate_golden_fixtures(
    checkpoint_path: Path,
    output_dir: Path,
    input_size: int = 518,
    num_samples: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    print("==================================================================")
    print("        DepthLab Golden Fixtures Generator (Milestone 1)          ")
    print("==================================================================")
    print(f"Checkpoint  : {checkpoint_path}")
    print(f"Output Dir  : {output_dir}")
    print(f"Input Size  : {input_size}")
    print(f"Device      : {device}")
    print(f"Num Samples : {num_samples}")
    print("------------------------------------------------------------------")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found at '{checkpoint_path}'. Run scripts/download_checkpoints.py first.")

    # Prepare output directories
    images_dir = output_dir / "images"
    depths_dir = output_dir / "depths"
    vis_dir = output_dir / "vis"

    images_dir.mkdir(parents=True, exist_ok=True)
    depths_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Instantiate model
    model_config = {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
    print("[GOLDEN] Instantiating DepthAnythingV2 (vits)...")
    model = DepthAnythingV2(**model_config)
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    print("[GOLDEN] Model loaded successfully.")

    # Locate source sample images (demo01.jpg .. demo10.jpg)
    src_examples_dir = OFFICIAL_REPO / "assets" / "examples"
    sample_names = [f"demo{i:02d}.jpg" for i in range(1, num_samples + 1)]

    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    manifest_samples = []

    for idx, sample_name in enumerate(sample_names, start=1):
        src_img_path = src_examples_dir / sample_name
        if not src_img_path.exists():
            raise FileNotFoundError(f"Source sample image not found at '{src_img_path}'.")

        base_name = f"demo{idx:02d}"
        dst_img_path = images_dir / f"{base_name}.jpg"
        dst_npy_path = depths_dir / f"{base_name}_depth.npy"
        dst_vis_path = vis_dir / f"{base_name}_vis.png"

        # Copy original image
        shutil.copy2(src_img_path, dst_img_path)

        # Read image
        raw_image = cv2.imread(str(dst_img_path))
        if raw_image is None:
            raise ValueError(f"Failed to read image at '{dst_img_path}'")

        h, w, c = raw_image.shape

        # Run inference in FP32
        with torch.no_grad():
            depth = model.infer_image(raw_image, input_size=input_size)

        # Confirm shape and dtype
        assert depth.shape == (h, w), f"Shape mismatch: expected ({h}, {w}), got {depth.shape}"
        depth_float32 = depth.astype(np.float32)

        # Save raw float32 numpy array
        np.save(dst_npy_path, depth_float32)

        # Compute visualization map
        min_val = float(depth_float32.min())
        max_val = float(depth_float32.max())
        mean_val = float(depth_float32.mean())
        std_val = float(depth_float32.std())

        val_range = max_val - min_val if max_val > min_val else 1.0
        depth_norm = (depth_float32 - min_val) / val_range * 255.0
        depth_uint8 = depth_norm.astype(np.uint8)

        depth_vis_bgr = (cmap(depth_uint8)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        cv2.imwrite(str(dst_vis_path), depth_vis_bgr)

        # Compute SHA256 checksums
        img_sha = compute_sha256(dst_img_path)
        npy_sha = compute_sha256(dst_npy_path)
        vis_sha = compute_sha256(dst_vis_path)

        manifest_samples.append({
            "sample_id": base_name,
            "image_filename": dst_img_path.name,
            "depth_npy_filename": dst_npy_path.name,
            "vis_png_filename": dst_vis_path.name,
            "image_path": str(dst_img_path.relative_to(output_dir.parent)),
            "depth_npy_path": str(dst_npy_path.relative_to(output_dir.parent)),
            "vis_png_path": str(dst_vis_path.relative_to(output_dir.parent)),
            "height": h,
            "width": w,
            "channels": c,
            "min_depth": min_val,
            "max_depth": max_val,
            "mean_depth": mean_val,
            "std_depth": std_val,
            "image_sha256": img_sha,
            "depth_npy_sha256": npy_sha,
            "vis_png_sha256": vis_sha,
        })

        print(f"[GOLDEN] Processed [{idx:02d}/{num_samples:02d}] {sample_name} -> shape ({h}x{w}), range [{min_val:.4f}, {max_val:.4f}]")

    # Build fixture manifest
    manifest_data = {
        "generator": "scripts/generate_golden.py",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_variant": "vits",
        "checkpoint_filename": checkpoint_path.name,
        "checkpoint_sha256": compute_sha256(checkpoint_path),
        "input_size": input_size,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": device,
        "samples_count": len(manifest_samples),
        "samples": manifest_samples,
    }

    manifest_path = output_dir / "fixture_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)

    print("------------------------------------------------------------------")
    print(f"[GOLDEN] Manifest generated at: {manifest_path}")
    print("[GOLDEN] Golden fixture generation completed successfully.")
    print("==================================================================")


def main():
    parser = argparse.ArgumentParser(description="Generate Golden Reference Fixtures for DepthLab")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints" / "depth_anything_v2_vits.pth", help="Path to DA2 Small checkpoint")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "golden", help="Output directory for golden fixtures")
    parser.add_argument("--input-size", type=int, default=518, help="Inference resolution")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of sample images")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Target device")

    args = parser.parse_args()
    generate_golden_fixtures(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        input_size=args.input_size,
        num_samples=args.num_samples,
        device=args.device,
    )


if __name__ == "__main__":
    main()
