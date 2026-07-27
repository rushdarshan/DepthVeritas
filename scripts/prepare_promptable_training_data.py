#!/usr/bin/env python3
"""Create training manifest rows from precomputed token and relative-depth files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def optimal_scale(relative_depth: np.ndarray, metric_depth: np.ndarray) -> float:
    valid = np.isfinite(relative_depth) & np.isfinite(metric_depth) & (relative_depth > 0) & (metric_depth > 0)
    return float((relative_depth[valid] * metric_depth[valid]).sum() / max((relative_depth[valid] ** 2).sum(), 1e-8))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV with feature_path,relative_depth_path,depth_path,category")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with Path(args.input).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["s_star"] = optimal_scale(np.load(row["relative_depth_path"]), np.load(row["depth_path"]))
    with Path(args.output).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
