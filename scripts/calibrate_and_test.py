"""Reproducible calibration artifact creation and locked-test evaluation.

Usage:
    python scripts/calibrate_and_test.py --calibration-split data/nyu_calib \
        --dev-split data/nyu_dev --test-split data/nyu_test \
        [--model-id sef-vits] [--target-fur 0.01] [--tile-size 32]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from depthlab.metrics.calibration import depth_error_event, _tiles
from depthlab.risk.artifact import build_calibration_artifact, CalibrationArtifact
from depthlab.risk.evaluate import evaluate_locked_test
from depthlab.risk.risk_score import NormalizedCombiner


def _load_split(path: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    p = Path(path)
    if not p.is_dir() and p.suffix == ".pt":
        data = torch.load(p)
        required = {"pred", "target", "entropy"}
        missing = required.difference(data)
        if missing:
            raise ValueError(f"{p} is missing required tensors: {sorted(missing)}")
        return data["pred"], data["target"], data["entropy"]
    raise FileNotFoundError(f"Split not found: {path}")


def _compute_tile_errors(pred: torch.Tensor, target: torch.Tensor, tile_size: int) -> torch.Tensor:
    err = depth_error_event(pred, target, threshold_ratio=0.1, align_scale_shift=True)
    return _tiles(err.float(), tile_size, agg="any")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate risk thresholds and evaluate on locked test split")
    parser.add_argument("--calibration-split", required=True, help="Path to calibration split .pt file or dir")
    parser.add_argument("--dev-split", required=True, help="Path to development split .pt file or dir")
    parser.add_argument("--test-split", required=True, help="Path to test split .pt file or dir")
    parser.add_argument("--model-id", default="sef-vits", help="Model identifier for the artifact")
    parser.add_argument("--target-fur", type=float, default=0.01, help="Target false-usable rate")
    parser.add_argument("--tile-size", type=int, default=32, help="Tile size in pixels")
    parser.add_argument("--output", default="artifacts/calibration.json", help="Output artifact path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred_cal, target_cal, entropy_cal = _load_split(args.calibration_split)
    pred_dev, target_dev, entropy_dev = _load_split(args.dev_split)
    pred_test, target_test, entropy_test = _load_split(args.test_split)

    pred_cal, target_cal, entropy_cal = pred_cal.to(device), target_cal.to(device), entropy_cal.to(device)
    pred_dev, target_dev, entropy_dev = pred_dev.to(device), target_dev.to(device), entropy_dev.to(device)
    pred_test, target_test, entropy_test = pred_test.to(device), target_test.to(device), entropy_test.to(device)

    cal_errors = _compute_tile_errors(pred_cal, target_cal, args.tile_size)
    dev_errors = _compute_tile_errors(pred_dev, target_dev, args.tile_size)

    cal_entropy = _tiles(entropy_cal, args.tile_size, agg="mean")
    dev_entropy = _tiles(entropy_dev, args.tile_size, agg="mean")
    test_entropy = _tiles(entropy_test, args.tile_size, agg="mean")
    combiner = NormalizedCombiner({"entropy": cal_entropy.flatten()})

    print("Calibrating thresholds on calibration split...")
    cal_risk = combiner.combine({"entropy": cal_entropy.flatten()}, enforce_shape=False)
    artifact = build_calibration_artifact(
        cal_risk.flatten(), cal_errors.flatten(), combiner, args.model_id,
        {"calibration": args.calibration_split, "development": args.dev_split, "test": args.test_split},
        target_fur=args.target_fur, n_steps=100,
    )
    if artifact is None:
        print("ERROR: No feasible threshold pair found on calibration split.")
        sys.exit(1)

    out_path = Path(args.output)
    artifact.save(out_path)
    print(f"Artifact saved to {out_path}")
    print(f"  Version: {artifact.version}")
    print(f"  t_usable={artifact.threshold_usable:.4f}, t_abstain={artifact.threshold_abstain:.4f}")
    print(f"  Achieved coverage: {artifact.achieved_coverage:.2%}")

    print("\nEvaluating on development split...")
    dev_risk = combiner.combine({"entropy": dev_entropy.flatten()}, enforce_shape=False)
    dev_result = evaluate_locked_test(artifact, dev_risk.flatten(), dev_errors.flatten())
    for k, v in dev_result.items():
        print(f"  {k}: {v}")

    print("\n=== LOCKED TEST EVALUATION ===")
    test_errors = _compute_tile_errors(pred_test, target_test, args.tile_size)
    test_risk = combiner.combine({"entropy": test_entropy.flatten()}, enforce_shape=False)
    test_result = evaluate_locked_test(artifact, test_risk.flatten(), test_errors.flatten())
    for k, v in test_result.items():
        print(f"  {k}: {v}")

    report_path = out_path.with_name(out_path.stem + "_report.json")
    report = {"artifact": str(out_path), "development": dev_result, "test": test_result}
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
