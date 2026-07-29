from __future__ import annotations

import torch

from depthlab.data.procedural import validate_procedural_manifest
from depthlab.export import relative_point_cloud, write_colored_ply
from depthlab.fusion import fuse_depth_measurements
from depthlab.heads import list_heads
from depthlab.backbone.adapter import FeatureBundle, FeatureStage
from depthlab.heads.sparse import SparseDepthHead, select_sparse_candidates
from depthlab.ordinal import ordinal_accuracy, ordinal_ranking_loss
from depthlab.physics import fit_plane_from_depth
from depthlab.probes import affinity_probe
from depthlab.streaming import MaskedEMA, validate_video_manifest


def test_sparse_selection_is_confidence_ranked_and_spatially_separated() -> None:
    candidates = torch.tensor([[[0.1, 0.1, 1.0, 0.9, 0.1], [0.11, 0.1, 2.0, 0.8, 0.1], [0.9, 0.9, 3.0, 0.7, 0.1]]])
    selected, mask = select_sparse_candidates(candidates, max_candidates=2, radius=0.05)
    assert mask.tolist() == [[True, True]]
    assert torch.allclose(selected[0, :, 3], torch.tensor([0.9, 0.7]))
    _, abstained = select_sparse_candidates(candidates, max_candidates=2, confidence_threshold=0.95)
    assert not abstained.any()
    assert "sparse_depth" in list_heads()


def test_sparse_head_decodes_patch_grid() -> None:
    head = SparseDepthHead(hidden_dim=8, max_candidates=3)
    bundle = FeatureBundle([FeatureStage(torch.randn(1, 4, 384), None, 3, 384)])
    output = head(bundle)
    assert output["candidates"].shape == (1, 4, 5)
    assert output["selected"].shape == (1, 3, 5)


def test_fusion_reduces_variance_and_rejects_disagreement() -> None:
    good_mean, good_variance, accepted = fuse_depth_measurements([torch.tensor([2.0]), torch.tensor([2.1])], [torch.tensor([1.0]), torch.tensor([1.0])])
    assert accepted.item() and good_variance.item() < 1.0 and 2.0 < good_mean.item() < 2.1
    rejected_mean, rejected_variance, _ = fuse_depth_measurements([torch.tensor([2.0]), torch.tensor([20.0])], [torch.tensor([0.01]), torch.tensor([0.01])])
    assert torch.allclose(rejected_mean, torch.tensor([2.0]))
    assert torch.allclose(rejected_variance, torch.tensor([0.01]))


def test_plane_fit_recovers_synthetic_flat_floor() -> None:
    depth = torch.full((1, 8, 8), 2.0)
    intrinsics = torch.tensor([[[2.0, 0.0, 3.5], [0.0, 2.0, 3.5], [0.0, 0.0, 1.0]]])
    result = fit_plane_from_depth(depth, intrinsics, torch.ones_like(depth, dtype=torch.bool), min_points=8)
    assert result["confidence"].item() > 0
    assert result["median_residual"].item() < 1e-5


def test_ordinal_loss_and_accuracy_follow_target_order() -> None:
    target = torch.tensor([[[1.0, 2.0, 3.0]]])
    prediction = target.clone().requires_grad_()
    assert ordinal_accuracy(prediction, target) == 1.0
    loss = ordinal_ranking_loss(prediction, target)
    assert loss.item() == 0.0
    loss.backward()


def test_ordinal_sampling_is_bounded_for_image_sized_inputs() -> None:
    target = torch.linspace(1.0, 2.0, 10_000).reshape(1, 100, 100)
    loss = ordinal_ranking_loss(target, target, max_pairs=64)
    assert torch.isfinite(loss)


def test_procedural_and_video_manifest_validation() -> None:
    record = {"scene_id": "thin-wire-1", "seed": 7, "renderer": "blender", "render_version": "4.0", "asset_licenses": "CC0", "depth_convention": "first_visible_surface"}
    assert validate_procedural_manifest([record]) == []
    assert validate_video_manifest([{"scene_id": "clip", "source_image_path": "a.png", "target_image_path": "b.png", "intrinsics": [1], "T_target_from_source": [1], "valid_mask": "m.png"}]) == []


def test_masked_ema_resets_and_preserves_current_on_invalid_prior() -> None:
    ema = MaskedEMA(0.5)
    ema.update(torch.tensor([1.0, 1.0]))
    assert torch.equal(ema.update(torch.tensor([3.0, 5.0]), torch.tensor([True, False])), torch.tensor([2.0, 5.0]))
    assert torch.equal(ema.update(torch.tensor([9.0, 9.0]), scene_cut=True), torch.tensor([9.0, 9.0]))


def test_affinity_probe_and_relative_export(tmp_path) -> None:
    result = affinity_probe(torch.tensor([[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]]), torch.tensor([[1.0, 1.1, 4.0]]))
    assert result["affinity_abs_depth_difference"] < result["reversed_index_abs_depth_difference"]
    points, colors = relative_point_cloud(torch.ones(2, 2), torch.ones(3, 2, 2), torch.eye(3))
    ply = write_colored_ply(tmp_path / "relative.ply", points, colors)
    assert "relative scale" in ply.read_text(encoding="ascii")
