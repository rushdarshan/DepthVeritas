"""Tests for U2: threshold calibration and triage policy."""

from __future__ import annotations

import torch

from depthlab.risk.calibrate import TriagePolicy, calibrate_thresholds


class TestTriagePolicy:

    def test_classify_below_usable(self):
        pol = TriagePolicy(0.3, 0.7)
        risk = torch.tensor([0.1, 0.5, 0.9])
        labels = pol.classify(risk)
        expected = torch.tensor([0.0, 1.0, 2.0])
        assert (labels == expected).all(), f"got {labels}"

    def test_usable_review_abstain_labels(self):
        pol = TriagePolicy(0.3, 0.7)
        risk = torch.tensor([0.1, 0.5, 0.9])
        labels = pol.classify(risk)
        assert labels[0].item() == 0.0
        assert labels[1].item() == 1.0
        assert labels[2].item() == 2.0

    def test_invalid_thresholds_raises(self):
        try:
            TriagePolicy(0.7, 0.3)
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_state_dict_roundtrip(self):
        pol = TriagePolicy(0.2, 0.8)
        d = pol.state_dict()
        pol2 = TriagePolicy.from_state_dict(d)
        assert pol2.t_usable == 0.2
        assert pol2.t_abstain == 0.8


class TestCalibrateThresholds:

    def test_returns_thresholds_with_clean_separation(self):
        torch.manual_seed(0)
        risk = torch.rand(200)
        err = torch.cat([torch.zeros(180), torch.ones(20)])
        result = calibrate_thresholds(risk, err, target_fur=0.05, n_steps=50)
        assert result is not None
        t_u, t_a = result
        assert t_u < t_a
        assert 0 <= t_u <= 1
        assert 0 <= t_a <= 1

    def test_returns_none_when_no_feasible_threshold(self):
        risk = torch.ones(50)
        err = torch.ones(50)
        result = calibrate_thresholds(risk, err, target_fur=0.01, n_steps=20)
        assert result is None

    def test_calibrate_result_feeds_triage_policy(self):
        torch.manual_seed(0)
        risk = torch.rand(200)
        err = torch.cat([torch.zeros(180), torch.ones(20)])
        result = calibrate_thresholds(risk, err, target_fur=0.05, n_steps=50)
        assert result is not None
        t_u, t_a = result
        pol = TriagePolicy(t_u, t_a)
        labels = pol.classify(risk)
        assert labels.shape == risk.shape

    def test_none_result_cannot_construct_policy(self):
        risk = torch.ones(50)
        err = torch.ones(50)
        result = calibrate_thresholds(risk, err, target_fur=0.01, n_steps=20)
        assert result is None

    def test_shape_mismatch_raises(self):
        try:
            calibrate_thresholds(torch.rand(10), torch.rand(5))
            assert False, "expected ValueError"
        except ValueError:
            pass
