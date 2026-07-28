"""Focused unit coverage for the offline research components.

These tests are intentionally not executed as part of the current power-saving
session. Run them once dependencies and the laptop power are available.
"""

import numpy as np
import torch

from depthlab.anchoring import AnchoringConfig, anchor_scale
from depthlab.heads.sef import SEFHead
from depthlab.metrics.calibration import expected_calibration_error
from depthlab.promptable import ScalePredictor, get_click_features
from depthlab.refinement import refine_depth


def test_sef_probabilities_and_centers() -> None:
    output = SEFHead(8, n_bins=4)(torch.randn(2, 8, 3, 3))
    assert output["bin_probs"].shape == (2, 3, 3, 4)
    assert torch.allclose(output["bin_probs"].sum(-1), torch.ones(2, 3, 3), atol=1e-5)
    assert torch.all(output["bin_centers"][:, 1:] > output["bin_centers"][:, :-1])


def test_sef_single_bin_entropy_is_zero() -> None:
    assert torch.equal(SEFHead(4, n_bins=1)(torch.randn(1, 4, 2, 2))["entropy"], torch.zeros(1, 2, 2))


def test_promptable_scale_is_positive_and_small() -> None:
    model = ScalePredictor()
    assert model.num_params < 10_000
    assert (model(torch.zeros(2, 768)) > 0).all()


def test_click_feature_mapping() -> None:
    tokens = torch.arange(17 * 2, dtype=torch.float32).reshape(1, 17, 2)
    output = get_click_features(tokens, 15, 0, 56, 56)
    assert output["row"].item() == 0 and output["col"].item() == 1
    assert torch.equal(output["patch_token"], tokens[:, 2])


def test_refinement_preserves_shape() -> None:
    depth = torch.ones(1, 8, 8)
    assert refine_depth(depth, torch.rand_like(depth), torch.rand(1, 4, 8, 8)).shape == depth.shape


def test_perfect_ece_is_zero() -> None:
    assert expected_calibration_error(torch.ones(4), torch.ones(4, dtype=torch.bool)) == 0.0


def test_anchor_returns_depth() -> None:
    depth = np.tile(np.linspace(2, 3, 32), (32, 1))
    output = anchor_scale(depth, AnchoringConfig(scene_type="street", ransac_iterations=8))
    assert output["depth"].shape == depth.shape
