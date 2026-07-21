## ADDED Requirements

### Requirement: Relative depth head reproduces DPT decoder

The system SHALL provide a `RelativeDepthHead` that implements the same DPT decoder architecture as the official DA2 DPT head, consuming 4 feature maps from `FeatureBundle` and producing a single-channel depth map.

#### Scenario: Output shape matches input resolution
- **WHEN** `RelativeDepthHead(variant='vits').forward(features)` is called with a FeatureBundle from a 518×518 input
- **THEN** the output depth tensor has spatial dimensions matching the input resolution

#### Scenario: Training on NYUv2 converges
- **WHEN** `RelativeDepthHead` is trained on NYUv2 with the official training recipe
- **THEN** AbsRel converges to within 1% of the published DA2-Small result (~0.128)

### Requirement: Loss function uses SILog

The relative depth head SHALL use `SILogLoss` from `metric_depth/util/loss.py` as its primary training loss.

#### Scenario: SILog loss decreases during training
- **WHEN** training with SILog loss on NYUv2
- **THEN** the loss decreases monotonically and validation AbsRel improves

### Requirement: Evaluation uses official metrics

The head SHALL report `abs_rel`, `d1`, `d2`, `d3`, `rmse`, `rmse_log`, `sq_rel`, `log10`, and `silog` using the official `eval_depth()` implementation.

#### Scenario: Metrics match expected format
- **WHEN** `head.compute_metrics(preds, batch)` is called
- **THEN** the returned dict contains all nine metric keys with float values
