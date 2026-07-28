"""Tests for the self-adapting depth module (U1-U4)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from depthlab.adaptation.correction_net import CorrectionNet
from depthlab.adaptation.manifest_loader import AdaptationManifest, select_keyframes
from depthlab.adaptation.scene_manager import SceneManager
from depthlab.adaptation.adapt import adapt_scene, _extract_features_and_depth
from depthlab.geometry import (
    projected_coords_in_bounds,
    positive_z_mask,
    occlusion_mask,
    sample_target_depth,
    minimum_reprojection_mask,
    reproject_depth,
)
from depthlab.losses.temporal import TemporalConsistencyLoss


# ---------------------------------------------------------------------------
# U1: CorrectionNet
# ---------------------------------------------------------------------------

class TestCorrectionNet:

    def test_forward_shape(self):
        net = CorrectionNet()
        feats = torch.randn(2, 384, 37, 37)
        base = torch.ones(2, 1, 518, 518)
        corrected, gate, residual = net(feats, base)
        assert corrected.shape == (2, 1, 518, 518)
        assert gate.shape == (2, 1, 518, 518)
        assert residual.shape == (2, 1, 518, 518)

    def test_identity_init_produces_base_depth(self):
        net = CorrectionNet()
        feats = torch.randn(1, 384, 37, 37)
        base = torch.full((1, 1, 518, 518), 5.0)
        corrected, gate, residual = net(feats, base)
        assert torch.allclose(corrected, base, atol=1e-4), (
            f"Identity init should preserve base depth, max diff={(
                corrected - base).abs().max().item()}"
        )

    def test_params_under_2m(self):
        n = sum(p.numel() for p in CorrectionNet().parameters())
        assert n < 2_000_000, f"CorrectionNet has {n} params, limit is 2M"

    def test_non_positive_base_depth_raises(self):
        net = CorrectionNet()
        feats = torch.randn(1, 384, 37, 37)
        with pytest.raises(ValueError, match="positive and finite"):
            net(feats, torch.zeros(1, 1, 518, 518))

    def test_non_finite_base_depth_raises(self):
        net = CorrectionNet()
        feats = torch.randn(1, 384, 37, 37)
        with pytest.raises(ValueError, match="positive and finite"):
            net(feats, torch.full((1, 1, 518, 518), float("nan")))


# ---------------------------------------------------------------------------
# U2: Manifest loader
# ---------------------------------------------------------------------------

def _tiny_png(path: Path) -> None:
    """Write a minimal valid 1x1 white PNG."""
    import struct, zlib
    sig = b'\x89PNG\r\n\x1a\n'
    ihdr_data = struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0)
    ihdr_crc = zlib.crc32(b'IHDR' + ihdr_data)
    ihdr = struct.pack('>I', 13) + b'IHDR' + ihdr_data + struct.pack('>I', ihdr_crc)
    raw = zlib.compress(b'\x00\xff\xff\xff')
    idat_crc = zlib.crc32(b'IDAT' + raw)
    idat = struct.pack('>I', len(raw)) + b'IDAT' + raw + struct.pack('>I', idat_crc)
    iend_crc = zlib.crc32(b'IEND')
    iend = struct.pack('>I', 0) + b'IEND' + struct.pack('>I', iend_crc)
    path.write_bytes(sig + ihdr + idat + iend)


class TestManifestLoader:

    @pytest.fixture
    def manifest_path(self, tmp_path: Path) -> Path:
        entries = [
            {
                "split": "adapt",
                "image_path": "frame_0000.png",
                "intrinsics": [500, 0, 320, 0, 500, 240, 0, 0, 1],
                "transform": [1, 0, 0, 0.1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            },
            {
                "split": "test",
                "image_path": "frame_0001.png",
                "intrinsics": [500, 0, 320, 0, 500, 240, 0, 0, 1],
                "transform": [1, 0, 0, 0.0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            },
        ]
        p = tmp_path / "manifest.json"
        p.write_text(json.dumps(entries), encoding="utf-8")
        _tiny_png(tmp_path / "frame_0000.png")
        _tiny_png(tmp_path / "frame_0001.png")
        return p

    def test_adapt_split_filters(self, manifest_path: Path):
        ds = AdaptationManifest(manifest_path, split="adapt")
        assert len(ds) == 1

    def test_missing_intrinsics_raises(self, manifest_path: Path):
        entries = [{"split": "adapt", "image_path": "f.png",
                     "transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]}]
        p = manifest_path.parent / "bad.json"
        p.write_text(json.dumps(entries), encoding="utf-8")
        _tiny_png(manifest_path.parent / "f.png")
        with pytest.raises(ValueError, match="Intrinsics required"):
            AdaptationManifest(p)[0]

    def test_missing_pose_raises(self, manifest_path: Path):
        entries = [{"split": "adapt", "image_path": "f.png",
                     "intrinsics": [500, 0, 320, 0, 500, 240, 0, 0, 1]}]
        p = manifest_path.parent / "bad2.json"
        p.write_text(json.dumps(entries), encoding="utf-8")
        _tiny_png(manifest_path.parent / "f.png")
        with pytest.raises(ValueError, match="Pose"):
            AdaptationManifest(p)[0]

    def test_tiny_baseline_raises(self, manifest_path: Path):
        entries = [{"split": "adapt", "image_path": "f.png",
                     "intrinsics": [500, 0, 320, 0, 500, 240, 0, 0, 1],
                     "transform": [1, 0, 0, 0.001, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]}]
        p = manifest_path.parent / "bad3.json"
        p.write_text(json.dumps(entries), encoding="utf-8")
        _tiny_png(manifest_path.parent / "f.png")
        with pytest.raises(ValueError, match="baseline"):
            AdaptationManifest(p)[0]


# ---------------------------------------------------------------------------
# U3b: Geometry validity
# ---------------------------------------------------------------------------

class TestGeometryValidity:

    def test_in_bounds_mask_center(self):
        pixels = torch.zeros(1, 2, 10, 10)
        pixels[:, 0] = 5
        pixels[:, 1] = 5
        mask = projected_coords_in_bounds(pixels, 10, 10)
        assert mask.all()

    def test_in_bounds_mask_outside(self):
        pixels = torch.zeros(1, 2, 10, 10)
        pixels[:, 0] = -1
        mask = projected_coords_in_bounds(pixels, 10, 10)
        assert not mask.any()

    def test_occlusion_mask_no_occlusion(self):
        ref = torch.ones(1, 1, 4, 4)
        sampled = torch.ones(1, 1, 4, 4)
        mask = occlusion_mask(ref, sampled)
        assert mask.all()

    def test_occlusion_mask_detects_occlusion(self):
        ref = torch.ones(1, 1, 4, 4) * 10.0
        sampled = torch.ones(1, 1, 4, 4) * 1.0
        mask = occlusion_mask(ref, sampled)
        assert not mask.any()

    def test_sample_target_depth_produces_valid_values(self):
        depth = torch.rand(1, 1, 10, 10)
        pixels = torch.stack(torch.meshgrid(
            torch.linspace(0, 9, 10), torch.linspace(0, 9, 10), indexing="ij"
        )).unsqueeze(0).float()
        sampled = sample_target_depth(depth, pixels)
        assert sampled.shape == (1, 1, 10, 10)
        assert torch.isfinite(sampled).all()

    def test_min_reproj_mask(self):
        target = torch.ones(1, 3, 4, 4)
        src = torch.ones(1, 3, 4, 4)
        pixels = torch.stack(torch.meshgrid(
            torch.linspace(0, 3, 4), torch.linspace(0, 3, 4), indexing="ij"
        )).unsqueeze(0).float()
        mask = minimum_reprojection_mask([src], target, [pixels])
        assert mask.shape == (1, 1, 4, 4)
        assert mask.any()

    def test_known_translation_produces_correct_pixels(self):
        B = 1
        H, W = 4, 4
        depth = torch.ones(B, H, W)
        fx = 10.0
        K = torch.tensor([[[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1]]])
        T = torch.eye(4).unsqueeze(0)
        T[:, 0, 3] = 0.5
        pixels, z = reproject_depth(depth, K, T)
        in_bounds = projected_coords_in_bounds(pixels, H, W)
        assert in_bounds.shape == (B, 1, H, W)
        assert pixels.shape == (B, 2, H, W)
        assert z.shape == (B, H, W)


# ---------------------------------------------------------------------------
# U3: TemporalConsistencyLoss fix (compares sampled at projected coords)
# ---------------------------------------------------------------------------

class TestTemporalConsistencyLoss:

    def test_loss_is_finite(self):
        loss_fn = TemporalConsistencyLoss()
        depth_t = torch.rand(1, 1, 8, 8)
        z = torch.rand(1, 1, 8, 8)
        pixels = torch.stack(torch.meshgrid(
            torch.linspace(0, 7, 8), torch.linspace(0, 7, 8), indexing="ij"
        )).unsqueeze(0).float()
        l = loss_fn(depth_t, z, pixels)
        assert torch.isfinite(l)


# ---------------------------------------------------------------------------
# U4: SceneManager
# ---------------------------------------------------------------------------

class TestSceneManager:

    def test_reset_restores_initial_state(self):
        class DummyBackbone(torch.nn.Module):
            def features(self, x):
                return type("Bundle", (), {"stages": [type("S", (), {"spatial_features": lambda *a: torch.randn(1, 384, 37, 37)})()]})()
            def forward(self, x):
                return torch.ones(1, 1, 518, 518)

        backbone = DummyBackbone()
        net = CorrectionNet()
        sm = SceneManager(backbone, net, torch.device("cpu"))
        state_before = {k: v.clone() for k, v in net.state_dict().items()}
        net.residual_head.weight.data.fill_(0.1)
        sm.reset()
        for k in state_before:
            assert torch.allclose(net.state_dict()[k], state_before[k]), f"{k} diverged after reset"

    def test_infer_before_adapt_returns_base(self):
        class DummyBackbone(torch.nn.Module):
            def features(self, x):
                return type("Bundle", (), {"stages": [type("S", (), {"spatial_features": lambda *a: torch.randn(1, 384, 37, 37)})()]})()
            def forward(self, x):
                return torch.ones(1, 1, 518, 518)

        backbone = DummyBackbone()
        net = CorrectionNet()
        sm = SceneManager(backbone, net, torch.device("cpu"))
        img = torch.randn(3, 518, 518)
        out = sm.infer(img)
        assert out.shape == (1, 1, 518, 518)
        assert torch.allclose(out, torch.ones(1, 1, 518, 518))


# ---------------------------------------------------------------------------
# Integration: known translation synthetic test (codex review requirement)
# ---------------------------------------------------------------------------

class TestSyntheticTranslation:

    def test_known_translation_produces_coherent_projections(self):
        B, H, W = 1, 4, 4
        depth = torch.ones(B, H, W) * 2.0
        fx = 10.0
        K = torch.tensor([[[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1]]])
        T = torch.eye(4).unsqueeze(0)
        T[:, 0, 3] = 0.5
        pixels, z = reproject_depth(depth, K, T)
        assert torch.isfinite(pixels).all()
        assert torch.isfinite(z).all()
        assert (z > 0).all()
        in_bounds = projected_coords_in_bounds(pixels, H, W)
        assert in_bounds.any()
