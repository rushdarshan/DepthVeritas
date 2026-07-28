#!/usr/bin/env python3
"""Compare official DA2 and a trained DepthLab head on one validation split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from depthlab.backbone.loader import load_da2_checkpoint
from depthlab.data import get_dataset
from depthlab.data.transforms import normalize_image_tensor
from depthlab.heads import get_head
from depthlab.metrics.dispatcher import MetricDispatcher


def _dispatcher(config: dict) -> MetricDispatcher:
    evaluation = config.get("eval", {})
    return MetricDispatcher(
        align_scale_shift=bool(evaluation.get("align_scale_shift", False)),
        min_depth=float(evaluation.get("min_depth", 1e-3)),
        max_depth=float(evaluation.get("max_depth", 80.0)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare DA2 with a trained DepthLab head")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--sef-checkpoint", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/sef-end-to-end/da2_comparison.json"))
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone_config = config.get("model", {}).get("backbone", {})
    backbone = load_da2_checkpoint(
        variant=backbone_config.get("variant", "vits"),
        checkpoint_path=backbone_config.get("checkpoint_path", "checkpoints/depth_anything_v2_vits.pth"),
        device=str(device),
        freeze=True,
    ).eval()

    head_config = dict(config.get("model", {}).get("head", {}))
    head_name = head_config.pop("name", "relative_depth")
    head_config.setdefault("encoder_variant", backbone_config.get("variant", "vits"))
    head = get_head(head_name, **head_config).to(device).eval()
    checkpoint = torch.load(args.sef_checkpoint, map_location=device)
    head.load_state_dict(checkpoint.get("head_state_dict", checkpoint))

    dataset_config = dict(config.get("dataset", {}))
    dataset_name = dataset_config.pop("name", "file_list")
    dataset = get_dataset(dataset_name, split="val", **dataset_config)
    loader = DataLoader(
        dataset,
        batch_size=int(dataset_config.get("batch_size", 2)),
        shuffle=False,
        num_workers=int(dataset_config.get("num_workers", 0)),
    )

    da2_raw, da2_aligned = _dispatcher(config), _dispatcher({"eval": {**config.get("eval", {}), "align_scale_shift": True}})
    sef_raw, sef_aligned = _dispatcher(config), _dispatcher({"eval": {**config.get("eval", {}), "align_scale_shift": True}})
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            target = batch["depth"].to(device)
            mask = batch.get("valid_mask")
            mask = mask.to(device) if mask is not None else None
            da2_prediction = backbone(normalize_image_tensor(images))
            sef_prediction = head(backbone.features(images)).get("depth")
            if sef_prediction.shape[-2:] != target.shape[-2:]:
                sef_prediction = F.interpolate(
                    sef_prediction.unsqueeze(1), size=target.shape[-2:], mode="bilinear", align_corners=False
                ).squeeze(1)
            da2_raw.update(da2_prediction, target, mask)
            da2_aligned.update(da2_prediction, target, mask)
            sef_raw.update(sef_prediction, target, mask)
            sef_aligned.update(sef_prediction, target, mask)

    result = {
        "dataset": {"name": dataset_name, "split": "val", "samples": len(dataset)},
        "protocol": {
            "shared_loader": True,
            "target_resolution": "392x392",
            "da2_input": "official ImageNet normalization",
            "sef_input": "training-time raw RGB",
            "sef_output": "bilinearly upsampled from patch grid",
        },
        "da2": {"raw": da2_raw.compute(), "aligned": da2_aligned.compute()},
        "sef": {"raw": sef_raw.compute(), "aligned": sef_aligned.compute()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
