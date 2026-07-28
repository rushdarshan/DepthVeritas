"""Benchmark scoring helpers for depth predictions."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch

from depthlab.metrics.depth_metrics import compute_depth_metrics
from depthlab.metrics.temporal import temporal_metrics
from depthlab.benchmark import BENCHMARK_VERSION, PRIORITY_WEIGHTS


def load_manifest(path: str | Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "samples" in data:
        return list(data["samples"])
    if isinstance(data, list):
        return data
    raise ValueError("Manifest must be a list or an object with a 'samples' list.")


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def _prediction_path(pred_dir: Path, sample: Dict[str, Any]) -> Path:
    if "prediction_path" in sample:
        return _resolve(pred_dir, sample["prediction_path"])
    sample_id = str(sample.get("id") or sample.get("sample_id"))
    if not sample_id:
        raise ValueError("Sample must contain id/sample_id or prediction_path.")
    return pred_dir / f"{sample_id}.npy"


def _load_depth(path: Path) -> torch.Tensor:
    if path.suffix.lower() != ".npy":
        raise ValueError(f"Depth files must be .npy arrays: {path}")
    return torch.from_numpy(np.load(path)).float()


def _mean_dict(items: Iterable[Dict[str, float]]) -> Dict[str, float]:
    rows = list(items)
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    return {key: float(np.mean([row[key] for row in rows if key in row])) for key in keys}


def score_samples(manifest_path: str | Path, pred_dir: str | Path, align: bool) -> Dict[str, Any]:
    manifest_path = Path(manifest_path)
    pred_dir = Path(pred_dir)
    base = manifest_path.parent
    samples = load_manifest(manifest_path)

    per_sample: List[Dict[str, Any]] = []
    for sample in samples:
        pred = _load_depth(_prediction_path(pred_dir, sample))
        if "depth_path" not in sample:
            per_sample.append({"id": sample.get("id", sample.get("sample_id")), "stratum": sample.get("stratum", "unknown"),
                               "sub_stratum": sample.get("sub_stratum", "unknown"), "metrics": {}, "missing_ground_truth": True})
            continue
        target = _load_depth(_resolve(base, sample["depth_path"]))
        mask = None
        if "mask_path" in sample:
            mask = _load_depth(_resolve(base, sample["mask_path"])).bool()

        metrics = compute_depth_metrics(pred, target, mask=mask, align=align)
        per_sample.append({
            "id": sample.get("id", sample.get("sample_id")),
            "stratum": sample.get("stratum", "unknown"),
            "sub_stratum": sample.get("sub_stratum", "unknown"),
            "metrics": metrics,
            "leakage": bool(sample.get("known_train_leakage", sample.get("leakage", False))),
        })

    by_stratum: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    by_sub: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    for row in per_sample:
        by_stratum[row["stratum"]].append(row["metrics"])
        by_sub[f"{row['stratum']}:{row['sub_stratum']}"].append(row["metrics"])

    weighted = []
    for stratum, rows in by_stratum.items():
        weight = PRIORITY_WEIGHTS.get(stratum, 1.0)
        weighted.extend([_mean_dict(rows)] * int(weight))
    clean_rows = [row["metrics"] for row in per_sample if not row.get("leakage") and row["metrics"]]
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "aligned": align,
        "per_sample": per_sample,
        "per_stratum": {key: _mean_dict(value) for key, value in sorted(by_stratum.items())},
        "per_sub_stratum": {key: _mean_dict(value) for key, value in sorted(by_sub.items())},
        "composite": _mean_dict(weighted),
        "clean_composite": _mean_dict(clean_rows),
    }


def write_markdown_table(result: Dict[str, Any], path: str | Path) -> None:
    rows = result.get("per_stratum", {})
    metric_names = ["abs_rel", "d1", "rmse", "log10"]
    lines = ["| Stratum | " + " | ".join(metric_names) + " |", "|" + "---|" * (len(metric_names) + 1)]
    for stratum, metrics in rows.items():
        values = [f"{metrics.get(name, 0.0):.4f}" for name in metric_names]
        lines.append(f"| {stratum} | " + " | ".join(values) + " |")
    lines.append("| composite | " + " | ".join(f"{result.get('composite', {}).get(name, 0.0):.4f}" for name in metric_names) + " |")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
