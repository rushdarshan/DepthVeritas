---
title: feat: Failure-Aware Benchmark
type: feat
status: active
date: 2026-07-29
---

# Failure-Aware Benchmark

## Overview

Six-stratum benchmark (transparent, reflective, thin structures, low-light, HDR/specular, video flicker) with provenance, human review, locked evaluation. Contribution is benchmark + baseline study.

## Implementation Units

- U1. **Sample curation**: Collect 200 samples/stratum (1200 total) with provenance fields, license review, human annotation.
- U2. **Evaluation harness**: Locked evaluation code, stratum-specific metrics, deterministic scoring.
- U3. **Baseline runs**: DA2, SEF, and any trained head on the frozen benchmark.
- U4. **Publication prep**: Dataset paper for next eligible venue call.

## Decisions
- 1200+ samples target. 300 is internal pilot only.
- Contribution = benchmark + reproducible baseline study
- NeurIPS Datasets track (missed May 2026 deadline) or next available venue
- CVPR workshop for early community feedback or challenge
