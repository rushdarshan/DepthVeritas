import torch
from depthlab.geometry import (
    projected_coords_in_bounds,
    positive_z_mask,
    occlusion_mask,
    sample_target_depth,
    minimum_reprojection_mask,
)


def test_projected_coords_in_bounds():
    pixels = torch.tensor([[[[50.0]], [[100.0]]]])
    mask = projected_coords_in_bounds(pixels, 200, 200)
    assert mask.all()


def test_projected_coords_out_of_bounds():
    pixels = torch.tensor([[[[-10.0]], [[100.0]]]])
    mask = projected_coords_in_bounds(pixels, 200, 200)
    assert not mask.any()


def test_projected_coords_all_outside():
    pixels = torch.tensor([[[[-100.0]], [[-100.0]]]])
    mask = projected_coords_in_bounds(pixels, 200, 200)
    assert not mask.any()


def test_positive_z_mask():
    z = torch.tensor([1.0, 0.5, 0.0, -0.1])
    result = positive_z_mask(z)
    assert result.shape == z.shape


def test_occlusion_detected():
    depth_ref = torch.ones(1, 1, 4, 4) * 5.0
    depth_sampled = torch.ones(1, 1, 4, 4) * 0.5
    mask = occlusion_mask(depth_ref, depth_sampled, threshold=2.0)
    assert not mask.all()


def test_no_occlusion():
    depth_ref = torch.ones(1, 1, 4, 4) * 5.0
    depth_sampled = torch.ones(1, 1, 4, 4) * 4.5
    mask = occlusion_mask(depth_ref, depth_sampled, threshold=2.0)
    assert mask.all()


def test_sample_target_depth_shape():
    depth = torch.randn(1, 1, 100, 100)
    pixels = torch.randn(1, 2, 100, 100)
    sampled = sample_target_depth(depth, pixels)
    assert sampled.shape == (1, 1, 100, 100)


def test_min_reproj_mask_shape():
    target = torch.randn(1, 3, 50, 50)
    sources = [torch.randn(1, 3, 50, 50)]
    pixels_list = [torch.randn(1, 2, 50, 50)]
    mask = minimum_reprojection_mask(sources, target, pixels_list)
    assert mask.shape == (1, 1, 50, 50)
    assert mask.dtype == torch.bool
