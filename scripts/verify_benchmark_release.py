#!/usr/bin/env python3
"""Reject a benchmark freeze until its curation evidence is complete."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from depthlab.benchmark.release import validate_benchmark_release


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a DepthLab benchmark release candidate")
    parser.add_argument("--manifest", default="data/manifest.json", type=Path)
    args = parser.parse_args()
    report = validate_benchmark_release(args.manifest)
    print(json.dumps(report.to_dict(), indent=2))
    raise SystemExit(0 if report.ok else 1)


if __name__ == "__main__":
    main()
