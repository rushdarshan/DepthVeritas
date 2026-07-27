# Implementation Status

## Completed Runtime Evidence

- DA2-Small checkpoint loaded and verified on an RTX 4050 6 GB GPU.
- NYU Depth V2 labeled archive was converted into 1,449 local RGB/depth pairs
  with deterministic 1,159/290 train/validation splits.
- SEF completed 20 frozen-backbone warm-up epochs and 10 end-to-end epochs.
- Final best-checkpoint evaluation is recorded in
  `runs/sef-end-to-end/eval_results.json`; the tracked run summary is
  `docs/reproduction/nyu-sef-experiment.md`.
- The focused test suite passes locally (`14 passed`).

## Remaining External Research Gate

Benchmark `1.0.0-dev` is deliberately not frozen. The OpenSpec benchmark
requires licensed, human-reviewed samples across six failure strata, including
temporal video data, plus baseline predictions for every curated sample. NYU
Depth V2 alone cannot satisfy those provenance or coverage requirements. The
manifest preparation, preflight, scoring, and reporting paths are implemented
and will fail clearly until those inputs are supplied.
