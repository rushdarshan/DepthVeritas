from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

from depthlab.risk.artifact import (
    CalibrationArtifact,
    SplitMetadata,
    build_calibration_artifact,
)
from depthlab.risk.calibrate import TriagePolicy
from depthlab.risk.evaluate import evaluate_locked_test, evaluate_on_tiles
from depthlab.risk.risk_score import NormalizedCombiner


class TestSplitMetadata:

    def test_from_dict(self):
        d = {"calibration": "nyu_calib", "development": "nyu_dev", "test": "nyu_test",
             "calibration_size": 100, "development_size": 50, "test_size": 50, "extra": "ignored"}
        sm = SplitMetadata.from_dict(d)
        assert sm.calibration == "nyu_calib"
        assert sm.development == "nyu_dev"
        assert sm.test == "nyu_test"
        assert not hasattr(sm, "extra")


class TestCalibrationArtifact:

    def test_save_load_roundtrip(self):
        artifact = CalibrationArtifact(
            version="1.0.0",
            created_at="2026-07-29T00:00:00+00:00",
            model_id="sef-vits",
            combiner_state={"mins": {"entropy": 0.0}, "ranges": {"entropy": 1.0}},
            threshold_usable=0.3,
            threshold_abstain=0.7,
            splits=SplitMetadata("calib", "dev", "test"),
            target_false_usable_rate=0.01,
            achieved_coverage=0.85,
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            artifact.save(path)
            loaded = CalibrationArtifact.load(path)
            assert loaded.version == "1.0.0"
            assert loaded.model_id == "sef-vits"
            assert loaded.threshold_usable == 0.3
            assert loaded.threshold_abstain == 0.7
            assert loaded.splits.calibration == "calib"
            assert loaded.splits.development == "dev"
            assert loaded.splits.test == "test"
            assert loaded.target_false_usable_rate == 0.01
            assert loaded.achieved_coverage == 0.85
        finally:
            path.unlink(missing_ok=True)

    def test_to_triage_policy(self):
        artifact = CalibrationArtifact(
            version="1.0.0", created_at="now", model_id="m",
            combiner_state={},
            threshold_usable=0.2, threshold_abstain=0.8,
            splits=SplitMetadata("c", "d", "t"),
            target_false_usable_rate=0.01, achieved_coverage=0.9,
        )
        policy = artifact.to_triage_policy()
        assert isinstance(policy, TriagePolicy)
        assert policy.t_usable == 0.2
        assert policy.t_abstain == 0.8


class TestBuildCalibrationArtifact:

    def test_builds_artifact_with_valid_data(self):
        torch.manual_seed(0)
        risk = torch.rand(200)
        err = torch.cat([torch.zeros(180), torch.ones(20)])
        combiner = NormalizedCombiner({"x": torch.tensor([0.0, 1.0])})
        artifact = build_calibration_artifact(
            risk, err, combiner, "test-model",
            {"calibration": "nyu_calib", "development": "nyu_dev", "test": "nyu_test"},
            target_fur=0.05, n_steps=50,
        )
        assert artifact is not None
        assert artifact.model_id == "test-model"
        assert artifact.splits.calibration == "nyu_calib"
        assert artifact.threshold_usable < artifact.threshold_abstain
        assert artifact.achieved_coverage > 0

    def test_returns_none_when_infeasible(self):
        risk = torch.ones(50)
        err = torch.ones(50)
        combiner = NormalizedCombiner({"x": torch.tensor([0.0, 1.0])})
        artifact = build_calibration_artifact(
            risk, err, combiner, "test-model",
            {"calibration": "c", "development": "d", "test": "t"},
            target_fur=0.01, n_steps=20,
        )
        assert artifact is None


class TestEvaluateLockedTest:

    def test_evaluate_returns_metrics(self):
        artifact = CalibrationArtifact(
            version="1.0.0", created_at="now", model_id="m",
            combiner_state={},
            threshold_usable=0.3, threshold_abstain=0.7,
            splits=SplitMetadata("c", "d", "t"),
            target_false_usable_rate=0.01, achieved_coverage=0.85,
        )
        risk = torch.tensor([0.1, 0.2, 0.5, 0.8, 0.9])
        err = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0])
        result = evaluate_locked_test(artifact, risk, err)
        assert "false_usable_rate" in result
        assert "coverage" in result
        assert "usable_rate" in result
        assert "review_rate" in result
        assert "abstain_rate" in result
        assert result["artifact_version"] == "1.0.0"

    def test_all_abstain_when_all_high_risk(self):
        artifact = CalibrationArtifact(
            version="1.0.0", created_at="now", model_id="m",
            combiner_state={},
            threshold_usable=0.3, threshold_abstain=0.7,
            splits=SplitMetadata("c", "d", "t"),
            target_false_usable_rate=0.01, achieved_coverage=0.0,
        )
        risk = torch.tensor([0.9, 1.0])
        err = torch.tensor([1.0, 1.0])
        result = evaluate_locked_test(artifact, risk, err)
        assert result["usable_rate"] == 0.0
        assert result["abstain_rate"] == 1.0

    def test_shape_mismatch_raises(self):
        artifact = CalibrationArtifact(
            version="1.0.0", created_at="now", model_id="m",
            combiner_state={},
            threshold_usable=0.3, threshold_abstain=0.7,
            splits=SplitMetadata("c", "d", "t"),
            target_false_usable_rate=0.01, achieved_coverage=0.0,
        )
        try:
            evaluate_locked_test(artifact, torch.rand(5), torch.rand(3))
            assert False
        except ValueError:
            pass

    def test_2d_input_raises(self):
        artifact = CalibrationArtifact(
            version="1.0.0", created_at="now", model_id="m",
            combiner_state={}, threshold_usable=0.3, threshold_abstain=0.7,
            splits=SplitMetadata("c", "d", "t"),
            target_false_usable_rate=0.01, achieved_coverage=0.0,
        )
        try:
            evaluate_locked_test(artifact, torch.rand(5, 5), torch.rand(5))
            assert False
        except ValueError:
            pass


class TestEvaluateOnTiles:

    def test_evaluate_on_tiles_routes_correctly(self):
        artifact = CalibrationArtifact(
            version="1.0.0", created_at="now", model_id="m",
            combiner_state={}, threshold_usable=0.3, threshold_abstain=0.7,
            splits=SplitMetadata("c", "d", "t"),
            target_false_usable_rate=0.01, achieved_coverage=0.85,
        )
        risk_map = torch.rand(64, 64)
        err_map = torch.randint(0, 2, (64, 64)).float()
        result = evaluate_on_tiles(artifact, risk_map, err_map, tile_size=16)
        assert isinstance(result, dict)
        assert "coverage" in result
