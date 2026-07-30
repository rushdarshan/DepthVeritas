import torch
from depthlab.adaptation.correction_net import CorrectionNet


def test_forward_shape():
    net = CorrectionNet(feature_dim=384, hidden_dim=64)
    features = torch.randn(2, 384, 28, 28)
    base_depth = torch.randn(2, 1, 392, 392).abs() + 0.1
    corrected, gate, residual = net(features, base_depth)
    assert corrected.shape == (2, 1, 392, 392)
    assert gate.shape == (2, 1, 392, 392)
    assert residual.shape == (2, 1, 392, 392)


def test_zero_init_identity():
    net = CorrectionNet(feature_dim=384, hidden_dim=64)
    features = torch.randn(1, 384, 28, 28)
    base_depth = torch.randn(1, 1, 392, 392).abs() + 0.1
    corrected, gate, residual = net(features, base_depth)
    assert torch.allclose(corrected, base_depth, atol=1e-6), "Zero-init should preserve base depth"


def test_gate_bounds():
    net = CorrectionNet(feature_dim=384, hidden_dim=64)
    features = torch.randn(1, 384, 28, 28)
    base_depth = torch.randn(1, 1, 392, 392).abs() + 0.1
    _, gate, _ = net(features, base_depth)
    assert gate.min() >= 0.0
    assert gate.max() <= 1.0


def test_residual_bounds():
    net = CorrectionNet(feature_dim=384, hidden_dim=64)
    features = torch.randn(1, 384, 28, 28)
    base_depth = torch.randn(1, 1, 392, 392).abs() + 0.1
    _, _, residual = net(features, base_depth)
    assert residual.min() >= -1.0
    assert residual.max() <= 1.0


def test_negative_base_depth_raises():
    net = CorrectionNet(feature_dim=384, hidden_dim=64)
    features = torch.randn(1, 384, 28, 28)
    base_depth = -torch.ones(1, 1, 392, 392)
    try:
        net(features, base_depth)
        assert False, "Should have raised"
    except ValueError:
        pass


def test_param_count():
    net = CorrectionNet(feature_dim=384, hidden_dim=64, max_params=2_000_000)
    n = sum(p.numel() for p in net.parameters())
    assert n < 2_000_000, f"CorrectionNet has {n} params, exceeds limit"
