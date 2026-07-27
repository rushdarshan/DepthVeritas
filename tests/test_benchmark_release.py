import json
from pathlib import Path

from depthlab.benchmark.release import MINIMUM_SAMPLES, validate_benchmark_release
from depthlab.benchmark import PRIMARY_STRATA


def _sample(stratum: str, sub_stratum: str, index: int) -> dict[str, str | bool]:
    return {
        "id": f"{stratum}-{index}",
        "stratum": stratum,
        "sub_stratum": sub_stratum,
        "dataset": "fixture",
        "dataset_version": "1",
        "source_url": "https://example.test/source",
        "license": "CC BY 4.0",
        "license_url": "https://example.test/license",
        "reviewed_by": "reviewer",
        "reviewed_at": "2026-07-27",
        "needs_human_review": False,
    }


def test_release_gate_reports_stratum_shortfall(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"samples": []}), encoding="utf-8")
    report = validate_benchmark_release(path)
    assert not report.ok
    assert "transparent_reflective: requires 200 reviewed samples, found 0" in report.errors


def test_release_gate_accepts_complete_reviewed_manifest(tmp_path: Path) -> None:
    samples = []
    for stratum, minimum in MINIMUM_SAMPLES.items():
        sub_stratum = PRIMARY_STRATA[stratum][0]
        samples.extend(_sample(stratum, sub_stratum, index) for index in range(minimum))
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"samples": samples}), encoding="utf-8")
    assert validate_benchmark_release(path).ok
