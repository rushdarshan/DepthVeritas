#!/usr/bin/env python3
"""Score benchmark predictions against a manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from depthlab.benchmark.scoring import score_samples, write_markdown_table
from depthlab.preflight import validate_benchmark_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="DepthLab benchmark scorer")
    parser.add_argument("--manifest", default="data/manifest.json", help="Benchmark manifest JSON")
    parser.add_argument("--pred_dir", required=True, help="Directory containing prediction .npy files")
    parser.add_argument("--output", default="results.json", help="Output JSON path")
    parser.add_argument("--no-align", action="store_true", help="Write raw scores only (default writes raw and aligned scores)")
    parser.add_argument("--preflight", action="store_true", help="Validate manifest and prediction directory without scoring")
    args = parser.parse_args()

    if args.preflight:
        report = validate_benchmark_manifest(args.manifest, args.pred_dir)
        print(json.dumps(report.to_dict(), indent=2))
        raise SystemExit(0 if report.ok else 1)

    output = Path(args.output)
    result = score_samples(args.manifest, args.pred_dir, align=False)
    if not args.no_align:
        result["aligned_scores"] = score_samples(args.manifest, args.pred_dir, align=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown_table(result.get("aligned_scores", result), output.with_suffix(".md"))


if __name__ == "__main__":
    main()
