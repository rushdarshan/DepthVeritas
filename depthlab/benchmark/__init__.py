"""DepthLab failure-mode benchmark version."""

BENCHMARK_VERSION = "1.0.0-dev"

PRIMARY_STRATA = {
    "transparent_reflective": ["glass", "mirror", "water", "polished_metal"],
    "thin_structures": ["wires", "poles", "railings", "branches", "hair"],
    "low_light": ["indoor_dark", "night_outdoor", "near_infrared"],
    "extreme_fov": ["fisheye", "wide_angle_120+", "panoramic"],
    "hdr_specular": ["studio_lighting", "specular_cg", "glossy_floors"],
    "video_flicker": ["slow_motion", "fast_motion", "static_scene"],
}

PRIORITY_WEIGHTS = {"transparent_reflective": 2.0, "thin_structures": 2.0}
