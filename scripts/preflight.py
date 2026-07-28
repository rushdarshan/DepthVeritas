#!/usr/bin/env python3
"""Validate an experiment config without loading a model or dataset tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from depthlab.preflight import gpu_details, validate_training_config


def main() -> None:
    parser = argparse.ArgumentParser(description="DepthLab experiment preflight")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    if not args.config.is_file():
        print(json.dumps({"ok": False, "errors": [f"Config does not exist: {args.config}"]}, indent=2))
        raise SystemExit(1)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    report = validate_training_config(config, PROJECT_ROOT)
    report.details["hardware"] = gpu_details()
    print(json.dumps(report.to_dict(), indent=2))
    raise SystemExit(0 if report.ok else 1)


if __name__ == "__main__":
    main()
