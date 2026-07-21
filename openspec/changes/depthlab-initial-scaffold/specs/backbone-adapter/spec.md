## ADDED Requirements

### Requirement: DA2Backbone loads official DepthAnythingV2

The system SHALL provide a `DA2Backbone` class that wraps the official `DepthAnythingV2` model from `depth-anything-v2-official/`. Loading SHALL support all four variants (`vits`, `vitb`, `vitl`, `vitg`) using the official model configs. The loaded model SHALL be placed in eval mode with frozen weights by default.

#### Scenario: Load DA2-Small variant
- **WHEN** `DA2Backbone(variant='vits')` is instantiated
- **THEN** the model loads the `vits` architecture (25M params, 384 embed dim) with frozen encoder and no CUDA errors

#### Scenario: Invalid variant raises error
- **WHEN** `DA2Backbone(variant='invalid')` is instantiated
- **THEN** a `ValueError` is raised listing available variants

### Requirement: features() returns FeatureBundle

The system SHALL provide a `features()` method that calls the official model's `get_intermediate_layers()` and returns a `FeatureBundle` containing 4 `FeatureStage` entries. Each stage SHALL contain `patch_tokens`, `cls_token`, `stage_index`, and `embed_dim`.

#### Scenario: FeatureBundle has correct structure
- **WHEN** `backbone.features(tensor)` is called on a (1, 3, 518, 518) input
- **THEN** it returns a `FeatureBundle` with exactly 4 stages, each containing `patch_tokens` of shape (1, N, D), `cls_token` of shape (1, D), sequential stage indices (0-3), and embed_dim matching the variant

### Requirement: forward() returns depth tensor

The system SHALL provide a `forward()` method that produces a depth map via the official model's DPT head, matching the output of the official `infer_image()`.

#### Scenario: Depth output matches official
- **WHEN** `backbone.forward(tensor)` is called on a preprocessed input tensor
- **THEN** the output is a 2D float tensor of shape (H, W) with non-negative values, visually matching the official model's output

### Requirement: Load checkpoint

The system SHALL provide a `load()` method that loads pretrained weights from a `.pth` file.

#### Scenario: Checkpoint loads successfully
- **WHEN** `backbone.load('checkpoints/depth_anything_v2_vits.pth')` is called
- **THEN** the model weights are updated and inference produces valid depth output

### Requirement: Variant config resolution

The system SHALL resolve model variant names to architecture parameters using a single lookup table.

#### Scenario: Variant config is correct
- **WHEN** variant 'vits' is resolved
- **THEN** it maps to features=64, out_channels=[48, 96, 192, 384], intermediate_layers=[2, 5, 8, 11], img_size=518, patch_size=14
