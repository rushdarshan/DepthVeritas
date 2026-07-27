#!/usr/bin/env python3
"""Build a versioned benchmark manifest from a JSONL candidate export.

This script never downloads or redistributes datasets. It normalizes metadata,
assigns deterministic strata, and leaves uncertain samples flagged for review.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from depthlab.benchmark import BENCHMARK_VERSION
from assign_stratum import assign_stratum


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True, help="JSONL rows with paths, provenance, and tags")
    parser.add_argument("--output", default="data/manifest.json")
    args = parser.parse_args()
    samples = []
    for line in Path(args.candidates).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        row.update(assign_stratum(row.get("tags", [])))
        row.setdefault("split", "test")
        row.setdefault("known_train_leakage", row.get("dataset", "").lower() in {"nyuv2", "kitti"})
        samples.append(row)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"benchmark_version": BENCHMARK_VERSION, "samples": samples}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
