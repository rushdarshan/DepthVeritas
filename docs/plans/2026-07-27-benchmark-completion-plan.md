---
artifact_contract: ce-unified-plan/v1
artifact_readiness: implementation-ready
execution: code
title: Public Failure-Mode Benchmark Completion
date: 2026-07-27
---

# Public Failure-Mode Benchmark Completion

## Objective

Produce a versioned benchmark release candidate with real public RGB-D/video
samples, DA2 predictions, scored baselines, provenance records, and release
documentation. Do not claim version `1.0.0` until every source license and
human-review record is present.

## Units

1. Add source descriptors and a curation audit format that records dataset
   version, license URL, source identifier, checksum, assigned stratum, and
   reviewer decision for every sample.
2. Acquire only sources whose redistribution and access terms permit local
   processing without an account-bound acceptance flow. Normalize candidates
   through the existing manifest and preflight validators.
3. Generate DA2 predictions, raw/aligned per-stratum scores, and baseline
   tables. Record missing ground truth rather than deriving it.
4. Add a release gate that rejects a freeze unless all six strata meet their
   counts and every sample has provenance plus a human-review decision.
5. Publish the release candidate artifacts, run verification, review changes,
   commit, push, and watch CI.

## Acceptance

- The manifest preflight validates every local asset and provenance field.
- The release gate reports exact missing counts or review records.
- Baseline generation and scoring are deterministic for available samples.
- A benchmark is frozen only when the OpenSpec count and provenance rules pass.
