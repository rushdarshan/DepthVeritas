#!/usr/bin/env python3
"""Normalize RGB-D manifest paths and create deterministic train/validation splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
from pathlib import Path


REQUIRED_COLUMNS = ("image_path", "depth_path")


def stable_split(sample_id: str, validation_fraction: float) -> str:
    value = int(hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "val" if value < validation_fraction else "train"


def prepare_rows(source: Path, validation_fraction: float, output_dir: Path | None = None) -> list[dict[str, str]]:
    with source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not set(REQUIRED_COLUMNS).issubset(rows[0]):
        raise ValueError(f"Input must contain columns: {', '.join(REQUIRED_COLUMNS)}")
    prepared = []
    for index, row in enumerate(rows):
        sample_id = row.get("id") or f"sample-{index:06d}"
        image = Path(row["image_path"])
        depth = Path(row["depth_path"])
        image_absolute = image if image.is_absolute() else source.parent / image
        depth_absolute = depth if depth.is_absolute() else source.parent / depth
        if output_dir is not None:
            image = Path(os.path.relpath(image_absolute, output_dir))
            depth = Path(os.path.relpath(depth_absolute, output_dir))
        prepared.append({
            "id": sample_id,
            "split": row.get("split") or stable_split(sample_id, validation_fraction),
            "image_path": str(image),
            "depth_path": str(depth),
            "valid": str(image_absolute.is_file() and depth_absolute.is_file()).lower(),
        })
    return prepared


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 1:
        raise ValueError("--validation-fraction must be between zero and one")
    rows = prepare_rows(args.input, args.validation_fraction, args.output.parent)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("id", "split", "image_path", "depth_path", "valid"))
        writer.writeheader()
        writer.writerows(rows)
    invalid = sum(row["valid"] == "false" for row in rows)
    print(f"Wrote {len(rows)} samples to {args.output}; {invalid} rows reference missing assets.")


if __name__ == "__main__":
    main()
