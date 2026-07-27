#!/usr/bin/env python3
"""Compatibility entry point for SEF head warmup."""

from pathlib import Path
import subprocess
import sys


if __name__ == "__main__":
    config = Path(__file__).parent / "config" / "sef_warmup.yaml"
    raise SystemExit(subprocess.call([sys.executable, "train.py", "--config", str(config)]))
