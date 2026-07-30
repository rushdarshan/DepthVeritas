---
title: feat: Risk-Aware Depth — Calibrated Abstention Policy
type: feat
status: active
date: 2026-07-29
origin: docs/ideation/2026-07-28-depthlab-high-impact-directions.md
---

# Risk-Aware Depth

## Overview

Add a calibrated risk score and threshold-based triage policy that labels each region as usable / review-needed / abstain. Start with SEF entropy alone as baseline, then add aleatoric variance + ensemble.

---

## Problem Frame

Depth models output a number per pixel with no confidence signal. Downstream systems cannot distinguish "confidently correct" from "confidently wrong." This project adds calibrated abstention so depth can be used in safety-aware applications.

---

## Requirements Trace

- R1. First slice: SEF categorical entropy alone as risk score. Define depth-error event on held-out NYU data
- R2. Risk-coverage curves and tile-level selective error vs random rejection baseline
- R3. Then add aleatoric variance (uncertainty head) + 3-member ensemble variance
- R4. Each signal calibrated independently (normalized on development split) before combining
- R5. Combined risk via calibrated monotonic combiner trained only on calibration split
- R6. Threshold policy: risk < t_usable → usable, t_usable ≤ risk < t_abstain → review, risk ≥ t_abstain → abstain
- R7. Thresholds chosen to satisfy target false-usable rate on calibration split
- R8. Policy on small tiles (e.g. 32×32), not isolated pixels
- R9. Final evaluation once on locked test split

---

## Scope Boundaries

- No "unsafe" label — use "abstain" / "review-needed"
- No safety certification — calibrated abstention, not safety guarantee
- No learned policy classifier in v1 — threshold only
- No metric depth claim — relative depth only

---

## Key Decisions

- **Calibrate each signal independently** before combining. Aleatoric variance, SEF entropy, and ensemble variance have different units/ranges — cannot sum directly
- **Define tile-level correctness label** for the single prediction being risk-scored (SEF or uncertainty head output)
- **Ensemble heads**: distinct initialization + data order + saved checkpoints. Shared backbone is fine, but variance from nearly identical heads is under-dispersed
- **Dashboard loads fixed versioned calibration artifacts** — cannot recompute thresholds from an uploaded image with no labels

## Implementation Units

- U0. **First slice: SEF entropy baseline**

**Files:**
- Modify: `depthlab/metrics/calibration.py`
- Test: `tests/test_calibration_triage.py`

**Approach:**
- Define depth-error event for relative depth (ordinal or boundary error threshold on evaluation set)
- Implement risk-coverage curves and tile-level selective error
- Use SEF entropy alone as risk signal
- Compare against random rejection baseline
- Add aleatoric and ensemble signals only after baseline has versioned calibration/test split and reproducible artifact

- U1. **Risk score computation**

**Files:**
- Create: `depthlab/risk/risk_score.py`
- Test: `tests/test_risk_score.py`

**Approach:**
- Reuse existing uncertainty head output (aleatoric log_variance → variance) — calibrate independently on dev split
- Add SEF head entropy computation (categorical distribution entropy) — calibrate independently on dev split
- Add 3-member compact head ensemble (shared backbone, distinct seeds + data order) — variance across heads, calibrate independently
- Combine normalized signals via fixed rule or calibrated monotonic combiner trained only on calibration split
- Apply per small tile (e.g. 32×32), not per pixel
- Define tile aggregation, border handling, tile-level correctness label

- U2. **Threshold calibration**

**Files:**
- Create: `depthlab/risk/calibrate.py`
- Test: `tests/test_calibrate.py`

**Approach:**
- On held-out calibration split (separate from dev and test), sweep t_usable, t_abstain
- Choose thresholds to satisfy target false-usable-rate (e.g. <1%) on calibration split
- False-usable rate requires pre-declared error event definition
- Report operating point: coverage at target false-usable rate
- Output: threshold values + calibration curve (risk vs actual error rate)
- Test only once on locked test split

- U3. **Dashboard integration**

**Files:**
- Modify: `dashboard.py`

**Approach:**
- Color overlay: green = usable, yellow = review, red = abstain
- Show risk score on hover
- Load fixed versioned calibration artifacts — no threshold recomputation from uploaded images
- Display selected operating point

---

## Risks

| Risk | Mitigation |
|------|------------|
| Ensemble 3× compute | Shared backbone → single forward pass; 3 heads add negligible cost |
| Ensemble variance under-dispersed | Distinct init + data order + separate checkpoints |
| Calibration overfits to calibration split | Use separate calibration, validation, test splits |
| Low coverage at target false-usable rate | Report honestly — coverage is the metric |