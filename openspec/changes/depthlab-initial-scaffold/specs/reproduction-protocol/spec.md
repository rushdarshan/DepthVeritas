## ADDED Requirements

### Requirement: Environment validation

The system SHALL provide a script that validates the compute environment and produces `docs/reproduction/environment.md` containing: Python version, PyTorch version, CUDA version, GPU model, driver version, git commit hash, and checkpoint SHA256.

#### Scenario: Environment report generated
- **WHEN** the validation script completes
- **THEN** `docs/reproduction/environment.md` exists with all required fields populated

### Requirement: Inference reproduction

The system SHALL run the official `run.py` on sample images without modification and save golden outputs to `golden/`. Each golden sample SHALL consist of the input image, the official depth output (`.npy` and `.png`), and a manifest.

#### Scenario: Golden samples created
- **WHEN** inference reproduction completes on 10 sample images
- **THEN** `golden/` contains 10 entries with `image_NNN.png`, `depth_NNN.npy`, `depth_NNN.png`, and a `manifest.json`

### Requirement: Evaluation reproduction

The system SHALL run the official evaluation and compare metrics against published values. Results SHALL be within 1% of published AbsRel, δ1, and RMSE on NYUv2 Eigen split.

#### Scenario: Metrics match published values
- **WHEN** official evaluation runs on NYUv2 Eigen split
- **THEN** AbsRel is within 1% of the published DA2-Small value

### Requirement: verify.py regression suite

The system SHALL provide `verify.py` that, when run, validates: checkpoint loads, inference produces correct output shape, output values are non-negative, no NaNs present, output MAE against golden sample is < 1e-6, and runtime is within expected bounds.

#### Scenario: verify.py passes on clean install
- **WHEN** `python verify.py` is run after a successful reproduction
- **THEN** all checks pass and exit code is 0

#### Scenario: verify.py detects corrupted checkpoint
- **WHEN** `python verify.py` is run with a corrupted checkpoint file
- **THEN** checkpoint loading check fails with an appropriate error message

### Requirement: Reproduction report

The system SHALL produce a reproduction report containing: official commit, checkpoint path and SHA256, dataset name and version, metric implementation SHA256, CUDA/PyTorch/GPU versions, and a results table with all nine depth metrics.

#### Scenario: Report contains all required fields
- **WHEN** reproduction report is generated
- **THEN** all fields per ADR-002 §5 are present and non-empty
