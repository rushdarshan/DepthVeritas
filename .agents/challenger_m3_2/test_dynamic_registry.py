"""
Dynamic Registry Verification Test Suite
Tests multi-head lookup, duplicate registration attempts, type validation, and error handling.
"""

import sys
import unittest
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import depthlab.heads as heads_module
from depthlab.heads import register_head, get_head, list_heads, BaseHead
from depthlab.backbone.adapter import FeatureBundle, FeatureStage


class MockHeadA(BaseHead):
    def __init__(self, encoder_variant="vits", **kwargs):
        super().__init__()
        self.encoder_variant = encoder_variant
        self.conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, features: FeatureBundle) -> dict:
        feat = features.stages[0].patch_tokens
        b, n, c = feat.shape
        h = w = int(n ** 0.5)
        feat = feat.permute(0, 2, 1).view(b, c, h, w)
        out = self.conv(feat)
        return {"depth": out}

    def compute_loss(self, preds: dict, targets: dict) -> torch.Tensor:
        return torch.tensor(0.5, requires_grad=True)

    def compute_metrics(self, preds: dict, targets: dict) -> dict:
        return {"metric_a": 0.1}


class MockHeadB(BaseHead):
    def __init__(self, encoder_variant="vits", **kwargs):
        super().__init__()
        self.encoder_variant = encoder_variant
        self.linear = nn.Linear(64, 2)

    def forward(self, features: FeatureBundle) -> dict:
        return {"uncertainty": self.linear(features.stages[0].cls_token)}

    def compute_loss(self, preds: dict, targets: dict) -> torch.Tensor:
        return torch.tensor(0.2, requires_grad=True)

    def compute_metrics(self, preds: dict, targets: dict) -> dict:
        return {"metric_b": 0.05}


class TestDynamicRegistry(unittest.TestCase):
    def setUp(self):
        # Save snapshot of registry
        self._saved_registry = dict(heads_module._HEAD_REGISTRY)

    def tearDown(self):
        # Restore original registry state
        heads_module._HEAD_REGISTRY.clear()
        heads_module._HEAD_REGISTRY.update(self._saved_registry)

    def test_01_multi_head_registration_and_lookup(self):
        # Register two distinct heads
        register_head("mock_head_a")(MockHeadA)
        register_head("mock_head_b")(MockHeadB)

        self.assertIn("mock_head_a", list_heads())
        self.assertIn("mock_head_b", list_heads())

        # Lookup and instantiate both heads
        head_a = get_head("mock_head_a", encoder_variant="vits")
        head_b = get_head("mock_head_b", encoder_variant="vits")

        self.assertIsInstance(head_a, MockHeadA)
        self.assertIsInstance(head_b, MockHeadB)

        # Create dummy feature bundle
        patch_tokens = torch.randn(2, 16, 64)
        cls_token = torch.randn(2, 64)
        stage = FeatureStage(patch_tokens=patch_tokens, cls_token=cls_token, stage_index=0, embed_dim=64)
        bundle = FeatureBundle(stages=[stage])

        out_a = head_a(bundle)
        out_b = head_b(bundle)

        self.assertIn("depth", out_a)
        self.assertIn("uncertainty", out_b)
        self.assertEqual(out_a["depth"].shape, (2, 1, 4, 4))
        self.assertEqual(out_b["uncertainty"].shape, (2, 2))

    def test_02_duplicate_same_class_registration(self):
        register_head("mock_head_a")(MockHeadA)
        # Register exact same class again under same name — should succeed (idempotent)
        res1 = register_head("mock_head_a")(MockHeadA)
        self.assertEqual(res1, MockHeadA)

    def test_03_duplicate_different_class_registration_fails(self):
        register_head("mock_head_a")(MockHeadA)

        # Register a different class under existing name — should raise ValueError
        class ConflictingHead(BaseHead):
            def __init__(self, **kwargs):
                super().__init__()
            def forward(self, features): return {}
            def compute_loss(self, preds, targets): return torch.tensor(0.0)
            def compute_metrics(self, preds, targets): return {}

        with self.assertRaises(ValueError) as ctx:
            register_head("mock_head_a")(ConflictingHead)
        self.assertIn("already registered", str(ctx.exception))

    def test_04_invalid_subclass_registration_fails(self):
        # Register a class that does not inherit from BaseHead
        class NotABaseHead:
            pass

        with self.assertRaises(TypeError) as ctx:
            register_head("invalid_head")(NotABaseHead)
        self.assertIn("must inherit from BaseHead", str(ctx.exception))

    def test_05_nonexistent_head_lookup_fails(self):
        with self.assertRaises(KeyError) as ctx:
            get_head("non_existent_head_xyz")
        self.assertIn("not found in registry", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
