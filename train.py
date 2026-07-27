#!/usr/bin/env python3
"""
train.py — DepthLab Single-Entrypoint Training Script.

Usage:
    python train.py --config config/default.yaml
    python train.py --config experiments/nyu_relative.yaml
"""

import argparse
import sys
from pathlib import Path
import yaml
from depthlab.preflight import gpu_details, validate_training_config


def main():
    parser = argparse.ArgumentParser(description="DepthLab Single-Entrypoint Training Script")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML training configuration file"
    )
    parser.add_argument("--preflight", action="store_true", help="Validate config, data paths, checkpoint, and hardware without training")
    args = parser.parse_args()

    if not args.config.exists():
        print(f"Error: Configuration file not found at '{args.config}'", file=sys.stderr)
        sys.exit(1)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.preflight:
        report = validate_training_config(config or {}, Path(__file__).parent)
        report.details["hardware"] = gpu_details()
        print(yaml.safe_dump(report.to_dict(), sort_keys=False))
        sys.exit(0 if report.ok else 1)

    from depthlab.trainer import Trainer
    trainer = Trainer(config)
    trainer.fit()


if __name__ == "__main__":
    main()
