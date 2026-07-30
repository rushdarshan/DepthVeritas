from __future__ import annotations

import json

import torch

from depthlab.evidence import Capability, capability_matrix, file_sha256, write_evidence_bundle
from scripts.nyu_failure_studio import apply_perturbation
from scripts.profile_da2 import profile


def test_evidence_bundle_hashes_inputs_and_preserves_tier(tmp_path) -> None:
    input_file = tmp_path / "input.txt"
    input_file.write_text("depthlab", encoding="utf-8")
    output = write_evidence_bundle(tmp_path / "evidence.json", command="python test.py", tier="synthetic", inputs=[input_file], capabilities=[Capability("fixture", "synthetic", "unit test")])
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["tier"] == "synthetic"
    assert payload["inputs"][0]["sha256"] == file_sha256(input_file)
    assert payload["capabilities"][0]["name"] == "fixture"


def test_capability_matrix_marks_missing_research_inputs(tmp_path) -> None:
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "depth_anything_v2_vits.pth").write_bytes(b"weights")
    statuses = {item.name: item.status for item in capability_matrix(tmp_path)}
    assert statuses["DA2 checkpoint"] == "measured"
    assert statuses["Six-stratum reviewed benchmark"] == "blocked"


def test_failure_studio_crop_keeps_image_and_depth_shapes() -> None:
    image = torch.rand(3, 20, 24)
    depth = torch.ones(20, 24)
    cropped_image, cropped_depth = apply_perturbation("center_crop", image, depth)
    assert cropped_image.shape == image.shape
    assert cropped_depth.shape == depth.shape


def test_profile_rejects_non_patch_aligned_size_without_loading_model() -> None:
    result = profile(torch.nn.Identity(), torch.device("cpu"), 256, 1)
    assert result["status"] == "rejected"
