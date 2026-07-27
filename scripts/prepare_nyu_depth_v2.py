#!/usr/bin/env python3
"""Convert the NYU Depth V2 labeled MATLAB archive into DepthLab assets."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


def _sample_axis(shape: tuple[int, ...], samples: int) -> int:
    axes = [axis for axis, size in enumerate(shape) if size == samples]
    if len(axes) != 1:
        raise ValueError(f"Could not identify the sample axis in shape {shape} for {samples} samples.")
    return axes[0]


def _sample_count(image_shape: tuple[int, ...], depth_shape: tuple[int, ...]) -> int:
    shared = set(image_shape).intersection(depth_shape)
    if not shared:
        raise ValueError(f"Image and depth arrays have no shared sample dimension: {image_shape}, {depth_shape}.")
    return max(shared)


def _read_sample(dataset: h5py.Dataset, index: int, sample_axis: int) -> np.ndarray:
    selector = [slice(None)] * dataset.ndim
    selector[sample_axis] = index
    return np.asarray(dataset[tuple(selector)])


def _rgb_hwc(array: np.ndarray) -> np.ndarray:
    if array.ndim != 3:
        raise ValueError(f"Expected RGB image with three dimensions, got {array.shape}.")
    channel_axes = [axis for axis, size in enumerate(array.shape) if size == 3]
    if len(channel_axes) != 1:
        raise ValueError(f"Could not identify RGB channel axis in shape {array.shape}.")
    return np.moveaxis(array, channel_axes[0], -1)


def _depth_hw(array: np.ndarray) -> np.ndarray:
    if array.ndim != 2:
        raise ValueError(f"Expected depth map with two dimensions, got {array.shape}.")
    return np.asarray(array, dtype=np.float32)


def convert(input_path: Path, output_dir: Path, validation_fraction: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    depth_dir = output_dir / "depths"
    image_dir.mkdir(exist_ok=True)
    depth_dir.mkdir(exist_ok=True)

    with h5py.File(input_path, "r") as archive:
        images = archive["images"]
        depths = archive["depths"]
        sample_count = _sample_count(images.shape, depths.shape)
        image_axis = _sample_axis(images.shape, sample_count)
        depth_axis = _sample_axis(depths.shape, sample_count)
        rows: list[dict[str, str]] = []
        for index in range(sample_count):
            image = _rgb_hwc(_read_sample(images, index, image_axis))
            depth = _depth_hw(_read_sample(depths, index, depth_axis))
            image_path = image_dir / f"{index:04d}.png"
            depth_path = depth_dir / f"{index:04d}.npy"
            Image.fromarray(image.astype(np.uint8)).save(image_path)
            np.save(depth_path, depth)
            rows.append(
                {
                    "id": f"nyu-v2-{index:04d}",
                    "split": "val" if index % round(1 / validation_fraction) == 0 else "train",
                    "image_path": str(image_path.relative_to(output_dir.parent)),
                    "depth_path": str(depth_path.relative_to(output_dir.parent)),
                    "valid": "true",
                }
            )

    manifest_path = output_dir.parent / "depth_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("id", "split", "image_path", "depth_path", "valid"))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Converted {len(rows)} NYU Depth V2 samples to {output_dir}.")
    print(f"Wrote manifest: {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/nyu_depth_v2"))
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    args = parser.parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if not 0 < args.validation_fraction < 1:
        raise ValueError("--validation-fraction must be between zero and one")
    convert(args.input, args.output_dir, args.validation_fraction)


if __name__ == "__main__":
    main()
