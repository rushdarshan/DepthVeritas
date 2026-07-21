## ADDED Requirements

### Requirement: Stratified evaluation benchmark

The system SHALL provide a frozen evaluation benchmark with six primary strata: Transparent/Reflective, Thin Structures, Low-Light/Night, Extreme FOV, HDR/Specular, and Video Flicker. Each stratum SHALL contain at least 200 annotated samples (P0 strata) or 100 samples (P1/P2 strata) from existing public datasets.

#### Scenario: Benchmark samples are curated
- **WHEN** the curation script completes
- **THEN** each P0 stratum has ≥200 samples and each P1/P2 stratum has ≥100 samples, all from public datasets

### Requirement: Error taxonomy sub-strata

Each primary stratum SHALL be divided into error taxonomy sub-strata for per-category reporting: Transparent/Reflective (glass, mirror, water, polished_metal), Thin Structures (wires, poles, railings, branches, hair), Low-Light (indoor_dark, night_outdoor, near_infrared), Extreme FOV (fisheye, wide_angle_120+, panoramic), HDR/Specular (studio_lighting, specular_cg, glossy_floors), and Video Flicker (slow_motion, fast_motion, static_scene).

#### Scenario: Sub-strata tagged in manifest
- **WHEN** curation completes
- **THEN** each sample in `manifest.json` has a `sub_stratum` tag from the defined taxonomy

### Requirement: Standardized scoring protocol

The system SHALL provide a single scoring CLI: `python score.py --pred_dir ./predictions --output results.json`. It SHALL output per-sample, per-stratum, per-sub-stratum, and composite scores in JSON and markdown table formats. Per-frame metrics SHALL include AbsRel, δ1 accuracy, RMSE, and log10. Per-video metrics SHALL include frame-to-frame depth variance and temporal consistency error.

#### Scenario: Scoring script produces valid output
- **WHEN** `python score.py --pred_dir predictions --output results.json` is run on a directory with valid predictions
- **THEN** `results.json` contains per-sample, per-stratum, and composite scores; a markdown table is also written

### Requirement: Benchmark versioning

The benchmark SHALL use semantic versioning stored in `benchmark/__init__.py` as `BENCHMARK_VERSION`. Every experiment SHALL record which benchmark version it was evaluated against. Version patches SHALL NOT change scores; minor versions SHALL add strata; major versions SHALL change protocol and invalidate direct comparison.

#### Scenario: Benchmark version recorded in experiment
- **WHEN** an experiment evaluates against the benchmark
- **THEN** the experiment's metadata includes `benchmark_version` matching `benchmark/__init__.py`

### Requirement: Scale/shift alignment

The scoring script SHALL apply least-squares scale/shift alignment per image for affine-invariant predictions. Both aligned and raw scores SHALL be reported.

#### Scenario: Aligned and raw scores differ
- **WHEN** scoring affine-invariant predictions
- **THEN** both `results_aligned.json` and `results_raw.json` are produced with different metric values
