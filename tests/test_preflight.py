import csv
from pathlib import Path

from depthlab.preflight import validate_depth_manifest, validate_training_config


def _write_manifest(path: Path, image: Path, depth: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("split", "image_path", "depth_path"))
        writer.writeheader()
        writer.writerow({"split": "train", "image_path": image.name, "depth_path": depth.name})


def test_valid_depth_manifest_passes(tmp_path: Path) -> None:
    image, depth = tmp_path / "sample.png", tmp_path / "sample.npy"
    image.write_bytes(b"image")
    depth.write_bytes(b"depth")
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, image, depth)
    assert validate_depth_manifest(manifest).ok


def test_missing_assets_are_reported(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, tmp_path / "missing.png", tmp_path / "missing.npy")
    report = validate_depth_manifest(manifest)
    assert not report.ok
    assert "missing.png" in "\n".join(report.errors)


def test_config_checks_checkpoint_before_model_load(tmp_path: Path) -> None:
    report = validate_training_config({"model": {"backbone": {"checkpoint_path": "missing.pth"}}, "dataset": {"name": "synthetic"}}, tmp_path)
    assert not report.ok
    assert "Backbone checkpoint does not exist" in report.errors[0]
