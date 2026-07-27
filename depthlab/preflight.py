"""Non-destructive readiness checks for DepthLab experiments and benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import csv
import json


@dataclass
class PreflightReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "errors": self.errors, "warnings": self.warnings, "details": self.details}


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _manifest_rows(manifest: Path) -> list[dict[str, str]]:
    with manifest.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_depth_manifest(manifest: str | Path) -> PreflightReport:
    manifest = Path(manifest)
    report = PreflightReport(details={"manifest": str(manifest)})
    if not manifest.is_file():
        report.errors.append(f"Dataset manifest does not exist: {manifest}")
        return report
    try:
        rows = _manifest_rows(manifest)
    except (csv.Error, UnicodeDecodeError) as exc:
        report.errors.append(f"Dataset manifest could not be read: {exc}")
        return report
    if not rows:
        report.errors.append("Dataset manifest has no samples.")
        return report
    required = {"image_path", "depth_path"}
    missing_columns = required - set(rows[0])
    if missing_columns:
        report.errors.append(f"Dataset manifest is missing columns: {', '.join(sorted(missing_columns))}")
        return report
    root = manifest.parent
    missing = []
    for index, row in enumerate(rows, start=2):
        for field_name in sorted(required):
            value = row.get(field_name, "")
            if not value or not resolve_path(root, value).is_file():
                missing.append(f"row {index} {field_name}: {value or '<empty>'}")
    if missing:
        report.errors.extend(missing)
    report.details["samples"] = len(rows)
    report.details["splits"] = sorted({row.get("split", "unspecified") for row in rows})
    return report


def validate_training_config(config: dict[str, Any], project_root: str | Path) -> PreflightReport:
    root = Path(project_root)
    report = PreflightReport()
    backbone = config.get("model", {}).get("backbone", {})
    checkpoint = resolve_path(root, backbone.get("checkpoint_path", "checkpoints/depth_anything_v2_vits.pth"))
    report.details["checkpoint"] = str(checkpoint)
    if not checkpoint.is_file():
        report.errors.append(f"Backbone checkpoint does not exist: {checkpoint}")
    dataset = config.get("dataset", {})
    dataset_name = dataset.get("name", "synthetic")
    report.details["dataset"] = dataset_name
    if dataset_name == "file_list":
        report = merge_reports(report, validate_depth_manifest(resolve_path(root, dataset.get("manifest", ""))))
    elif dataset_name == "video_manifest":
        path = resolve_path(root, dataset.get("manifest", ""))
        if not path.is_file():
            report.errors.append(f"Video manifest does not exist: {path}")
    elif dataset_name != "synthetic":
        report.warnings.append(f"No specialized preflight validator is registered for dataset '{dataset_name}'.")
    return report


def validate_benchmark_manifest(manifest: str | Path, prediction_dir: str | Path | None = None) -> PreflightReport:
    manifest = Path(manifest)
    report = PreflightReport(details={"manifest": str(manifest)})
    if not manifest.is_file():
        report.errors.append(f"Benchmark manifest does not exist: {manifest}")
        return report
    try:
        document = json.loads(manifest.read_text(encoding="utf-8"))
        samples: Iterable[dict[str, Any]] = document.get("samples", []) if isinstance(document, dict) else document
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        report.errors.append(f"Benchmark manifest could not be read: {exc}")
        return report
    samples = list(samples)
    if not samples:
        report.errors.append("Benchmark manifest has no samples.")
        return report
    root = manifest.parent
    for index, sample in enumerate(samples):
        if not sample.get("id", sample.get("sample_id")):
            report.errors.append(f"sample {index}: missing id or sample_id")
        if "depth_path" in sample and not resolve_path(root, sample["depth_path"]).is_file():
            report.errors.append(f"sample {index}: missing depth_path {sample['depth_path']}")
        if not sample.get("stratum") or not sample.get("sub_stratum"):
            report.errors.append(f"sample {index}: missing stratum or sub_stratum")
    if prediction_dir is not None and not Path(prediction_dir).is_dir():
        report.errors.append(f"Prediction directory does not exist: {prediction_dir}")
    report.details["samples"] = len(samples)
    return report


def gpu_details() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"torch_available": False}
    cuda = torch.cuda.is_available()
    return {
        "torch_available": True,
        "cuda_available": cuda,
        "device": torch.cuda.get_device_name(0) if cuda else "CPU",
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2) if cuda else 0,
    }


def merge_reports(*reports: PreflightReport) -> PreflightReport:
    merged = PreflightReport()
    for report in reports:
        merged.errors.extend(report.errors)
        merged.warnings.extend(report.warnings)
        merged.details.update(report.details)
    return merged
