#!/usr/bin/env python3
"""Run a trained SEF head on one image and save depth/entropy arrays and PNGs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from depthlab.backbone.loader import load_da2_checkpoint
from depthlab.evaluation import save_depth_visualization, save_entropy_visualization
from depthlab.heads import get_head


def load_image(path: Path) -> torch.Tensor:
    from PIL import Image
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--backbone-checkpoint", default="checkpoints/depth_anything_v2_vits.pth")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--variant", default="vits")
    parser.add_argument("--n-bins", type=int, default=96)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = load_da2_checkpoint(args.variant, args.backbone_checkpoint, str(device)).eval()
    head = get_head("sef", encoder_variant=args.variant, n_bins=args.n_bins).to(device).eval()
    state = torch.load(args.checkpoint, map_location=device)
    head.load_state_dict(state.get("head_state_dict", state))
    with torch.no_grad():
        result = head(backbone.features(load_image(Path(args.input)).to(device)))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    depth, entropy = result["depth"][0].cpu().numpy(), result["entropy"][0].cpu().numpy()
    np.save(output / "depth.npy", depth.astype(np.float16))
    np.save(output / "entropy.npy", entropy.astype(np.float16))
    save_depth_visualization(depth, output / "depth.png")
    save_entropy_visualization(entropy, output / "entropy.png")


if __name__ == "__main__":
    main()
