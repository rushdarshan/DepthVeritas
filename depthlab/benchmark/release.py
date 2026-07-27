"""Release gate for a provenance-complete failure-mode benchmark."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from depthlab.benchmark import PRIMARY_STRATA
from depthlab.benchmark.scoring import load_manifest
from depthlab.preflight import PreflightReport


MINIMUM_SAMPLES = {
    "transparent_reflective": 200,
    "thin_structures": 200,
    "low_light": 100,
    "extreme_fov": 100,
    "hdr_specular": 100,
    "video_flicker": 100,
}
REQUIRED_PROVENANCE_FIELDS = (
    "id",
    "dataset",
    "dataset_version",
    "source_url",
    "license",
    "license_url",
    "reviewed_by",
    "reviewed_at",
)


def validate_benchmark_release(manifest_path: str | Path) -> PreflightReport:
    """Validate all evidence required to freeze benchmark version 1.0.0."""
    manifest_path = Path(manifest_path)
    report = PreflightReport(details={"manifest": str(manifest_path), "required_counts": MINIMUM_SAMPLES})
    if not manifest_path.is_file():
        report.errors.append(f"Benchmark manifest does not exist: {manifest_path}")
        return report
    try:
        samples = load_manifest(manifest_path)
    except (OSError, ValueError) as exc:
        report.errors.append(f"Benchmark manifest could not be read: {exc}")
        return report

    counts: Counter[str] = Counter()
    for index, sample in enumerate(samples):
        stratum = str(sample.get("stratum", ""))
        counts[stratum] += 1
        expected_sub_strata = PRIMARY_STRATA.get(stratum)
        if expected_sub_strata is None:
            report.errors.append(f"sample {index}: unknown stratum '{stratum or '<missing>'}'")
        elif sample.get("sub_stratum") not in expected_sub_strata:
            report.errors.append(f"sample {index}: invalid sub_stratum '{sample.get('sub_stratum', '<missing>')}' for {stratum}")
        missing = [field for field in REQUIRED_PROVENANCE_FIELDS if not sample.get(field)]
        if missing:
            report.errors.append(f"sample {index}: missing provenance fields: {', '.join(missing)}")
        if sample.get("needs_human_review", True):
            report.errors.append(f"sample {index}: human review is incomplete")

    report.details["counts"] = dict(sorted(counts.items()))
    for stratum, minimum in MINIMUM_SAMPLES.items():
        count = counts[stratum]
        if count < minimum:
            report.errors.append(f"{stratum}: requires {minimum} reviewed samples, found {count}")
    return report
