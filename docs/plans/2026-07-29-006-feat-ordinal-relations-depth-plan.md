---
title: feat: Depth as Ordinal Relations
type: feat
status: active
date: 2026-07-29
---

# Depth as Ordinal Relations

## Overview

Add pairwise ranking loss to standard depth head. Evaluate with ordinal accuracy + Kendall's τ instead of AbsRel.

## Implementation Units

- U1. **Pairwise ranking loss**: Sample valid pixel pairs (i,j) with ground-truth depth difference > threshold. Margin loss on predicted inverse depth ordering.
- U2. **SfM auxiliary pairs**: Use only high-confidence triangulated pairs from external SfM. Weight below RGB-D supervision.
- U3. **Evaluation**: Ordinal accuracy, Kendall's τ, tie handling, confidence-stratified accuracy. Compare DA2 and SEF on same pair set.

## Decisions
- Ranking loss on standard depth head (not separate Siamese)
- SfM for auxiliary pairs only — not sole signal
- Report aligned AbsRel/RMSE as guardrails alongside ordinal metrics
