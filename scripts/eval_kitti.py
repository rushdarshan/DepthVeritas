#!/usr/bin/env python3
"""Evaluate precomputed prediction/ground-truth pairs after optional anchoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from depthlab.anchoring import AnchoringConfig, anchor_scale
from depthlab.metrics import compute_depth_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="JSON list with prediction_path and depth_path")
    parser.add_argument("--output", default="results/kitti_anchoring.json")
    parser.add_argument("--scene-type", default="street")
    args = parser.parse_args()
    rows = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    metrics = []
    for row in rows:
        prediction = np.load(row["prediction_path"])
        anchored = anchor_scale(prediction, AnchoringConfig(scene_type=args.scene_type))["depth"]
        metrics.append(compute_depth_metrics(torch.from_numpy(anchored), torch.from_numpy(np.load(row["depth_path"]))))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({key: float(np.mean([row[key] for row in metrics])) for key in metrics[0]}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
