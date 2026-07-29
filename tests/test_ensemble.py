from __future__ import annotations

import torch
import torch.nn as nn

from depthlab.risk.ensemble import HeadEnsemble, tile_ensemble_risk_from_members


def _dummy_head_factory(seed: int) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Conv2d(3, 1, kernel_size=1)


class TestHeadEnsemble:

    def test_three_members_created(self):
        ensemble = HeadEnsemble(_dummy_head_factory, n_members=3, seeds=[42, 73, 91])
        assert len(ensemble.members) == 3

    def test_forward_returns_variance(self):
        ensemble = HeadEnsemble(_dummy_head_factory, n_members=3)
        x = torch.randn(1, 3, 8, 8)
        out = ensemble(x)
        assert out["mean"].shape == (1, 1, 8, 8)
        assert out["variance"].shape == (1, 1, 8, 8)
        assert torch.all(out["variance"] >= 0)

    def test_member_count_mismatch_raises(self):
        try:
            HeadEnsemble(_dummy_head_factory, n_members=3, seeds=[42, 73])
            assert False
        except ValueError:
            pass

    def test_load_member_state(self):
        ensemble = HeadEnsemble(_dummy_head_factory, n_members=2, seeds=[1, 2])
        sds = [m.state_dict() for m in ensemble.members]
        ensemble2 = HeadEnsemble(_dummy_head_factory, n_members=2, seeds=[3, 4])
        ensemble2.load_member_state(sds)
        for m1, m2 in zip(ensemble.members, ensemble2.members):
            k1 = list(m1.state_dict())[0]
            assert torch.equal(m1.state_dict()[k1], m2.state_dict()[k1])

    def test_load_member_state_wrong_count_raises(self):
        ensemble = HeadEnsemble(_dummy_head_factory, n_members=2)
        try:
            ensemble.load_member_state([_dummy_head_factory(0).state_dict()])
            assert False
        except ValueError:
            pass
