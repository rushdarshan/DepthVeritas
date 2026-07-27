#!/usr/bin/env python3
"""Compatibility entry point for SEF end-to-end fine-tuning."""

from pathlib import Path
import subprocess
import sys


if __name__ == "__main__":
    config = Path(__file__).parent / "config" / "sef_end_to_end.yaml"
    raise SystemExit(subprocess.call([sys.executable, "train.py", "--config", str(config)]))
