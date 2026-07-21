## ADDED Requirements

### Requirement: Config-driven experiment execution

The system SHALL support a single entry point `train.py --config <path>` that reads a YAML config specifying model variant, head type, dataset, optimizer, learning rate, batch size, fp16, gradient accumulation, max epochs, and checkpoint path.

#### Scenario: Training run starts from config
- **WHEN** `python train.py --config experiments/EXP-001/experiment.yaml` is executed
- **THEN** the model, head, dataset, and training loop are configured from the YAML and training begins

### Requirement: EXP-NNN experiment folders

Each training run SHALL create a self-contained experiment folder at `runs/EXP-NNN/` containing: `experiment.yaml` (config snapshot), `checkpoints/` (backbone and head state dicts), `tensorboard/` (event logs), `visualizations/` (generated figures), and `metrics.json` (per-eval metrics).

#### Scenario: Experiment folder structure created
- **WHEN** training starts
- **THEN** a directory `runs/EXP-NNN/` is created with `experiment.yaml`, `checkpoints/`, `tensorboard/`, and `metrics.json`

### Requirement: Automatic metadata capture

Every experiment SHALL automatically capture and log: git commit hash, CUDA version, PyTorch version, random seed, dataset version, benchmark version, and start/end timestamps. These SHALL be written to `experiment.yaml` under a `_metadata` key at run start.

#### Scenario: Metadata captured at run start
- **WHEN** training begins
- **THEN** the `experiment.yaml` in the run directory includes a `_metadata` section with all required fields

### Requirement: One-command reproducibility

Running `python train.py --config runs/EXP-NNN/experiment.yaml` on a different machine SHALL reproduce the identical experiment configuration (seeds, hyperparameters, metadata preserved; new run gets a child EXP-ID).

#### Scenario: Config is self-contained
- **WHEN** the experiment.yaml from a completed run is used to start a new run
- **THEN** the new run starts with the same configuration (new EXP-ID, same parameters)

### Requirement: Checkpointing and resumption

The trainer SHALL save checkpoints after every epoch and support resumption from the latest checkpoint. Backbone and head SHALL be saved separately to avoid redundant backbone copies.

#### Scenario: Training resumes from checkpoint
- **WHEN** training is interrupted and restarted with the same config
- **THEN** training resumes from the latest epoch, restoring optimizer state and epoch counter

### Requirement: TensorBoard logging

The trainer SHALL log training loss, validation metrics, learning rate, and GPU memory usage to TensorBoard.

#### Scenario: TensorBoard events are written
- **WHEN** training runs for at least 10 steps
- **THEN** TensorBoard event files exist in the run directory with scalar logs
