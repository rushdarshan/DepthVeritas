---
title: feat: Adversarial Augmentation Loop
type: feat
status: active
date: 2026-07-29
---

# Adversarial Augmentation Loop

## Overview

Staged experiments: start with one stratum (thin structures), prove transfer improves held-out real data without harming NYU. Scale to multi-stratum curriculum only after gate passes.

## Key Decisions
- Start with one stratum (thin structures or low light) + explicit target-depth convention (first visible surface)
- Blender procedural gen for v1 (not differentiable renderer)
- Pinned Blender version, scene assets, seeds, render metadata
- Benchmark analysis as detector (not adversarial discriminator)
- Quarantine benchmark test split — only use development set for generator tuning
- Needs 2+ rounds for publication — each round selects its distribution using development evidence
- Include ordinary augmentation and static procedural augmentation baselines
- Define source-domain and real-domain evaluation, compute budget, minimum effect sizes

## Implementation Units

- U0. **First slice: thin-structure dataset**
  Generate small, reproducible thin-structure dataset with depth + masks. Train head, evaluate on held-out real thin-structure development split + standard NYU validation. Proceed only if transfer gate passes (improves thin-structure without harming NYU).

- U1. **Blender procedural generator**: Script generates scenes with controlled materials/lighting/geometry. Define ground-truth convention (first visible surface). Pinned versions, asset licenses, seeds, render metadata.

- U2. **Weakness analysis from benchmark**: Development-split benchmark analysis identifies which failure modes dominate for targeted generation.

- U3. **Augment → retrain loop** (2+ rounds): Generate targeted samples based on development evidence → add to training set → retrain → re-evaluate. Report held-out stratum scores, normal-scene regression, render distribution, convergence curve across rounds.
