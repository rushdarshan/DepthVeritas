"""Tests for U1: risk signal computation — entropy, aleatoric, ensemble, combiner."""

from __future__ import annotations

import torch

from depthlab.risk.risk_score import (
    tile_entropy_risk,
    tile_aleatoric_risk,
    tile_ensemble_risk,
    NormalizedCombiner,
)


class TestTileEntropyRisk:

    def test_output_shape(self):
        entropy = torch.rand(2, 64, 64)
        risk = tile_entropy_risk(entropy, tile_size=16)
        assert risk.shape == (2, 16)

    def test_valid_mask_aggregates_only_valid_pixels(self):
        entropy = torch.ones(1, 32, 32)
        mask = torch.zeros(1, 32, 32, dtype=torch.bool)
        mask[:, 8:24, 8:24] = True
        risk = tile_entropy_risk(entropy, tile_size=8, valid_mask=mask)
        assert risk[0, 0] == 1.0
        assert risk[0, 5] > 0

    def test_fully_invalid_tile_forced_to_max_risk(self):
        entropy = torch.ones(1, 32, 32)
        mask = torch.zeros(1, 32, 32, dtype=torch.bool)
        risk = tile_entropy_risk(entropy, tile_size=8, valid_mask=mask)
        assert (risk == 1.0).all()

    def test_2d_input(self):
        entropy = torch.rand(32, 32)
        risk = tile_entropy_risk(entropy, tile_size=8)
        assert risk.ndim == 1


class TestTileAleatoricRisk:

    def test_output_shape(self):
        var = torch.rand(1, 64, 64)
        risk = tile_aleatoric_risk(var, tile_size=16)
        assert risk.shape == (1, 16)


class TestTileEnsembleRisk:

    def test_variance_is_computed_correctly(self):
        d1 = torch.zeros(1, 4, 4)
        d2 = torch.ones(1, 4, 4)
        d3 = torch.full((1, 4, 4), 2.0)
        risk = tile_ensemble_risk([d1, d2, d3], tile_size=4)
        assert risk.shape == (1, 1)
        assert risk[0, 0] > 0

    def test_zero_variance_for_identical_heads(self):
        d = torch.rand(1, 16, 16)
        risk = tile_ensemble_risk([d, d], tile_size=8)
        assert risk.abs().max() < 1e-6


class TestNormalizedCombiner:

    def test_combine_two_signals(self):
        s1 = torch.tensor([0.0, 1.0])
        s2 = torch.tensor([1.0, 0.0])
        combiner = NormalizedCombiner({"a": s1, "b": s2})
        combined = combiner.combine({"a": s1, "b": s2})
        expected = 0.5 * (s1 + s2)
        assert torch.allclose(combined, expected)

    def test_single_signal_passthrough(self):
        s = torch.tensor([0.0, 1.0])
        combiner = NormalizedCombiner({"x": s})
        combined = combiner.combine({"x": s})
        assert torch.allclose(combined, s)

    def test_degenerate_range(self):
        s = torch.zeros(4)
        combiner = NormalizedCombiner({"x": s})
        norm = combiner.normalize("x", s)
        assert (norm == 0).all()

    def test_raises_on_empty(self):
        combiner = NormalizedCombiner({})
        try:
            combiner.combine({})
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_normalize_uses_stored_min_not_batch_min(self):
        calib_signal = torch.tensor([2.0, 4.0])
        combiner = NormalizedCombiner({"x": calib_signal})
        new_batch = torch.tensor([1.0, 3.0])
        norm = combiner.normalize("x", new_batch)
        expected = (new_batch - 2.0) / 2.0
        assert torch.allclose(norm, expected)

    def test_unknown_signal_raises(self):
        combiner = NormalizedCombiner({"a": torch.tensor([0.0, 1.0])})
        try:
            combiner.normalize("b", torch.tensor([0.0, 1.0]))
            assert False, "expected KeyError"
        except KeyError:
            pass

    def test_shape_mismatch_raises(self):
        combiner = NormalizedCombiner({"a": torch.tensor([0.0, 1.0])})
        try:
            combiner.combine({"a": torch.tensor([0.0, 1.0, 2.0])})
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_state_dict_roundtrip(self):
        combiner = NormalizedCombiner({"a": torch.tensor([0.0, 2.0]), "b": torch.tensor([5.0, 10.0])})
        d = combiner.state_dict()
        combiner2 = NormalizedCombiner.from_state_dict(d)
        assert combiner2.mins == {"a": 0.0, "b": 5.0}
        assert combiner2.ranges == {"a": 2.0, "b": 5.0}
        combined = combiner2.combine({"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([7.0, 10.0])})
        assert combined is not None
