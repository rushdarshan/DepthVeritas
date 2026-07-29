"""Tests for U0: SEF entropy baseline — depth-error event and risk-coverage curves."""

from __future__ import annotations

import torch

from depthlab.metrics.calibration import (
    depth_error_event,
    _tiles,
    risk_coverage_curve,
)


class TestDepthErrorEvent:

    def test_flags_high_error_pixels(self):
        pred = torch.tensor([[1.0, 5.0]])
        target = torch.tensor([[1.0, 1.0]])
        err = depth_error_event(pred, target, threshold_ratio=0.1, align_scale_shift=False)
        assert not err[0, 0].item()
        assert err[0, 1].item()

    def test_all_correct_at_low_threshold(self):
        pred = torch.full((3, 16, 16), 2.0)
        target = torch.full((3, 16, 16), 2.0)
        err = depth_error_event(pred, target, threshold_ratio=0.5)
        assert not err.any().item()

    def test_shape_match(self):
        pred = torch.randn(4, 32, 48)
        target = torch.randn(4, 32, 48)
        err = depth_error_event(pred, target, threshold_ratio=0.1)
        assert err.shape == pred.shape
        assert err.dtype == torch.bool

    def test_alignment_pathway_exists(self):
        pred = torch.randn(2, 16, 16)
        target = torch.randn(2, 16, 16)
        err = depth_error_event(pred, target, align_scale_shift=True)
        assert err.shape == pred.shape

    def test_mask_suppresses_error_outside_valid_region(self):
        pred = torch.tensor([[1.0, 5.0]])
        target = torch.tensor([[1.0, 1.0]])
        mask = torch.tensor([[True, False]])
        err = depth_error_event(pred, target, threshold_ratio=0.1, mask=mask)
        assert not err[0, 0].item()
        assert not err[0, 1].item()

    def test_mask_leaves_error_inside_valid_region(self):
        pred = torch.tensor([[1.0, 5.0]])
        target = torch.tensor([[1.0, 1.0]])
        mask = torch.tensor([[True, True]])
        err = depth_error_event(pred, target, threshold_ratio=0.1, mask=mask)
        assert not err[0, 0].item()
        assert err[0, 1].item()


class TestTileOps:

    def test_mean_aggregation_2d(self):
        x = torch.zeros(8, 8)
        x[2:4, 2:4] = 1.0
        tiles = _tiles(x, tile_size=4, agg="mean")
        assert tiles.shape == (4,)
        assert tiles[0].item() == 0.25

    def test_max_aggregation_2d(self):
        x = torch.arange(16.0).reshape(4, 4)
        tiles = _tiles(x, tile_size=2, agg="max")
        assert (tiles == torch.tensor([5.0, 7.0, 13.0, 15.0])).all()

    def test_any_aggregation_2d(self):
        x = torch.zeros(4, 4)
        x[0, 0] = 1.0
        tiles = _tiles(x, tile_size=2, agg="any")
        assert tiles[0] == 1.0
        assert (tiles[1:] == 0).all()

    def test_3d_tensor(self):
        x = torch.zeros(2, 4, 4)
        x[0, 0, 0] = 1.0
        tiles = _tiles(x, tile_size=2, agg="mean")
        assert tiles.shape == (2, 4)
        assert tiles[0, 0] > 0

    def test_non_divisible_edge(self):
        x = torch.ones(1, 10, 10)
        tiles = _tiles(x, tile_size=4, agg="mean")
        assert tiles.shape == (1, 4)

    def test_sum_aggregation(self):
        x = torch.zeros(1, 4, 4)
        x[:, :2, :2] = 1.0
        tiles = _tiles(x, tile_size=2, agg="sum")
        assert tiles.shape == (1, 4)
        assert tiles[0, 0].item() == 4.0

    def test_image_smaller_than_tile_returns_empty(self):
        x = torch.ones(1, 4, 4)
        tiles = _tiles(x, tile_size=8, agg="mean")
        assert tiles.shape == (1, 0)

    def test_raises_on_zero_tile_size(self):
        try:
            _tiles(torch.ones(4, 4), tile_size=0)
            assert False, "expected ValueError"
        except ValueError:
            pass


class TestRiskCoverageCurve:

    def test_perfect_risk_perfect_coverage(self):
        risk = torch.zeros(16, 16)
        risk[:8, :8] = 1.0
        errors = torch.zeros(16, 16)
        errors[:8, :8] = 1.0
        result = risk_coverage_curve(risk, errors, tile_size=4, n_steps=10)
        selective = result["risk_selective"]
        assert selective.dim() == 1
        assert selective[-1] == 0.0

    def test_random_baseline_is_flat(self):
        risk = torch.rand(32, 32)
        errors = torch.randint(0, 2, (32, 32)).float()
        result = risk_coverage_curve(risk, errors, tile_size=8, n_steps=20)
        baseline = result["random_baseline"]
        assert baseline.dim() == 1
        assert baseline.numel() == 20

    def test_batch_input(self):
        risk = torch.rand(2, 16, 16)
        errors = torch.randint(0, 2, (2, 16, 16)).float()
        result = risk_coverage_curve(risk, errors, tile_size=4, n_steps=5)
        assert result["risk_selective"].shape == (2, 5)
