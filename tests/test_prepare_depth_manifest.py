import csv
from pathlib import Path

from scripts.prepare_depth_manifest import prepare_rows


def test_prepare_rows_assigns_stable_ids_and_splits(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("image_path", "depth_path"))
        writer.writeheader()
        writer.writerow({"image_path": "missing.png", "depth_path": "missing.npy"})
    first = prepare_rows(source, 0.2)
    assert first == prepare_rows(source, 0.2)
    assert first[0]["id"] == "sample-000000"
    assert first[0]["valid"] == "false"


def test_prepare_rows_rebases_paths_for_output_manifest(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "image.png").write_bytes(b"image")
    (assets / "depth.npy").write_bytes(b"depth")
    source = tmp_path / "source.csv"
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("image_path", "depth_path"))
        writer.writeheader()
        writer.writerow({"image_path": "assets/image.png", "depth_path": "assets/depth.npy"})
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    row = prepare_rows(source, 0.2, output_dir)[0]
    assert (output_dir / row["image_path"]).resolve() == (assets / "image.png").resolve()
    assert row["valid"] == "true"
