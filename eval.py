#!/usr/bin/env python3
"""
eval.py — DepthLab Standalone Evaluation Script & Feature Ablation Engine.

Usage:
    python eval.py --config experiments/nyu_relative.yaml --checkpoint outputs/nyu_relative_vits/checkpoint_best.pth
    python eval.py --config experiments/nyu_relative.yaml --checkpoint outputs/nyu_relative_vits/checkpoint_best.pth --ablate-layer 0 1
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional
import yaml
import torch
from torch.utils.data import DataLoader

from depthlab.backbone.loader import load_da2_checkpoint
from depthlab.heads import get_head
from depthlab.data import get_dataset
from depthlab.metrics.dispatcher import MetricDispatcher, apply_feature_ablation


def main():
    parser = argparse.ArgumentParser(description="DepthLab Evaluation Script & Feature Ablation Suite")
    parser.add_argument("--config", type=Path, required=True, help="Path to experiment YAML config")
    parser.add_argument("--checkpoint", type=Path, required=False, default=None, help="Path to head checkpoint (.pth)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory to save evaluation results")
    parser.add_argument("--ablate-layer", type=int, nargs="*", default=None, help="Stage indices to ablate (e.g. --ablate-layer 0 1)")
    args = parser.parse_args()

    if not args.config.exists():
        print(f"Error: Config file '{args.config}' not found.", file=sys.stderr)
        sys.exit(1)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] Starting evaluation on device: {device}")

    # Load Backbone
    bb_cfg = config.get("model", {}).get("backbone", {})
    backbone = load_da2_checkpoint(
        variant=bb_cfg.get("variant", "vits"),
        checkpoint_path=Path(bb_cfg.get("checkpoint_path", "checkpoints/depth_anything_v2_vits.pth")),
        device=str(device)
    ).eval()

    # Load Head
    head_cfg = config.get("model", {}).get("head", {})
    head = get_head(
        head_cfg.get("name", "relative_depth"),
        encoder_variant=bb_cfg.get("variant", "vits"),
        features=head_cfg.get("features", 64),
        out_channels=head_cfg.get("out_channels", [48, 96, 192, 384])
    ).to(device)

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        out_d = Path(config.get("experiment", {}).get("output_dir", "outputs/default"))
        ckpt_path = out_d / "checkpoints" / "checkpoint_latest.pth"

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt.get("head_state_dict", ckpt)
        head.load_state_dict(state_dict)
        print(f"[Eval] Loaded head state dict from {ckpt_path}")
    else:
        print(f"Warning: Checkpoint '{ckpt_path}' not found! Running evaluation with initialized head weights.", file=sys.stderr)

    head.eval()

    # Setup Dataset & Loader
    data_cfg = config.get("dataset", {})
    data_kwargs = dict(data_cfg)
    ds_name = data_kwargs.pop("name", "synthetic")
    ds = get_dataset(ds_name, split="val", **data_kwargs)
    loader = DataLoader(ds, batch_size=int(data_cfg.get("batch_size", 4)), shuffle=False, num_workers=int(data_cfg.get("num_workers", 0)))

    # Dispatcher
    eval_cfg = config.get("eval", {})
    dispatcher = MetricDispatcher(
        align_scale_shift=eval_cfg.get("align_scale_shift", True),
        min_depth=float(eval_cfg.get("min_depth", 1e-3)),
        max_depth=float(eval_cfg.get("max_depth", 80.0))
    )

    ablate_layers = args.ablate_layer
    if ablate_layers:
        print(f"[Eval] Feature Ablation Active: zeroing stages {ablate_layers}")

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["depth"].to(device)

            features = backbone.features(images)
            if ablate_layers:
                features = apply_feature_ablation(features, ablate_layers)

            preds = head(features)
            pred_depth = preds.get("depth", preds.get("predicted_depth"))
            dispatcher.update(pred_depth, targets)

    results = dispatcher.compute()
    print("\n================ Evaluation Results ================")
    print(dispatcher.summary_table())

    output_dir = args.output_dir or Path(config.get("experiment", {}).get("output_dir", "outputs/eval"))
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_data = {
        "config": str(args.config),
        "checkpoint": str(ckpt_path),
        "ablated_layers": ablate_layers,
        "metrics": results
    }

    res_json = output_dir / "eval_results.json"
    with open(res_json, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"[Eval] Saved evaluation results to {res_json}")


if __name__ == "__main__":
    main()
