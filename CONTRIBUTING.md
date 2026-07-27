# Contributing

Keep `depth-anything-v2-official/` unchanged. Add new methods through the
`depthlab` registries, use manifest-driven data paths, and add CPU-safe tests.

Before a pull request, run `python -m pytest tests -q` and `python verify.py
--all` when the checkpoint and fixtures are available. Do not commit datasets,
checkpoints, generated outputs, or local environment files. Benchmark changes
must include provenance, license, split/leakage metadata, and stratum tags.
