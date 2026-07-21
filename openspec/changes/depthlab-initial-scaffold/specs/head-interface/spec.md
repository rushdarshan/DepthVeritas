## ADDED Requirements

### Requirement: BaseHead is an abstract class

The system SHALL provide an abstract `BaseHead` class defining three methods: `forward(features: FeatureBundle) -> dict`, `compute_loss(preds: dict, batch: dict) -> Tensor`, and `compute_metrics(preds: dict, batch: dict) -> dict[str, float]`. Attempting to instantiate `BaseHead` directly SHALL raise `TypeError`.

#### Scenario: BaseHead cannot be instantiated directly
- **WHEN** `BaseHead()` is called
- **THEN** `TypeError` is raised

#### Scenario: Subclass with all methods is valid
- **WHEN** a subclass implements `forward`, `compute_loss`, and `compute_metrics`
- **THEN** it can be instantiated and called without error

### Requirement: forward() accepts FeatureBundle

The `forward()` method SHALL accept a single `FeatureBundle` argument and return a dict. The dict SHALL contain at minimum a `'depth'` key with the depth tensor.

#### Scenario: Forward pass produces depth output
- **WHEN** `head.forward(feature_bundle)` is called
- **THEN** the returned dict contains `'depth'` as a (H, W) tensor, and any additional head-specific keys

### Requirement: Head registry

The system SHALL provide a decorator-based head registry (`@register_head(name)`) that maps string names to head classes. The registry SHALL be queryable and support config-driven head selection.

#### Scenario: Register and resolve head
- **WHEN** `@register_head('relative')` decorates a head class, then `build_head('relative', config)` is called
- **THEN** the registered class is instantiated with the given config

#### Scenario: Unknown head raises error
- **WHEN** `build_head('nonexistent', config)` is called
- **THEN** a `KeyError` is raised listing available heads
