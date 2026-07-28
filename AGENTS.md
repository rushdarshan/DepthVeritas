# Repository Guidelines

## Project Structure & Module Organization
DepthLab is a Python research scaffold for reproducible depth estimation. Core
package code lives in `depthlab/`: `backbone/` wraps Depth Anything V2,
`heads/` contains registered prediction heads, `metrics/` contains evaluation
metrics, and `trainer.py` drives training. Top-level entry points are
`train.py`, `eval.py`, and `verify.py`.

Keep `depth-anything-v2-official/` as upstream reference code. Store regression
fixtures in `golden/`, config in `config/`, helper scripts in `scripts/`, and
design or reproduction records in `docs/` and `openspec/`. Large local outputs
belong in ignored paths such as `checkpoints/`, `outputs/`, and `scratch/`.

## Build, Test, and Development Commands
- `python verify.py` runs the default regression and environment checks.
- `python verify.py --all` runs the full verification suite against golden
  fixtures and reproduction metadata.
- `python scripts/download_checkpoints.py` downloads checkpoints.
- `python scripts/generate_golden.py` refreshes golden fixtures.
- `python train.py --config config/default.yaml` starts training from a YAML
  config.
- `python eval.py --config config/default.yaml --checkpoint outputs/<run>/checkpoint_best.pth`
  evaluates a trained head and writes `eval_results.json`.

Install dependencies in a virtual environment. Upstream requirements are in
`depth-anything-v2-official/requirements.txt`.

## Coding Style & Naming Conventions
Use Python 3 with 4-space indentation, type hints at module boundaries, and
concise docstrings for public entry points. Prefer `Path` over string paths and
structured YAML/JSON parsing. Name modules and functions in `snake_case`,
classes in `PascalCase`, and config keys in lowercase snake case. Register
extensible components through existing registries such as `@register_head`.

## Testing Guidelines
The main test gate is `verify.py`; run it before submitting changes to models,
metrics, fixtures, or reproducibility reports. When changing golden outputs,
regenerate fixtures deliberately and include report updates. Keep new checks
deterministic, lightweight by default, and CPU-compatible unless GPU behavior is
explicitly under test.

## Commit & Pull Request Guidelines
The current history uses descriptive, sentence-style commit subjects. Use
concise imperative subjects that name the affected area, such as `Add relative
depth evaluation metrics`.

Pull requests should summarize the change, list commands run, call out
checkpoint or dataset assumptions, and link relevant docs, OpenSpec changes, or
issues. Include screenshots or metric tables when changing visual outputs or
evaluation behavior.

## Security & Configuration Tips
Do not commit checkpoints, datasets, generated outputs, or local environment
files. Keep paths configurable through YAML files, and document any required
external model weights in `docs/reproduction/` or the PR description.
