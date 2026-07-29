---
title: feat: Interactive Scene Understanding
type: feat
status: active
date: 2026-07-29
---

# Interactive Scene Understanding

## Overview

Companion dashboard demo: click a pixel → see depth, uncertainty, boundary quality → export colored PLY point cloud.

## Implementation Units

- U1. **Click interaction**: Add Streamlit canvas component for pixel coordinate selection. On click: show depth, SEF entropy, confidence.
- U2. **PLY export**: Export colored point cloud using declared approximate intrinsics. Label as "relative scale."
- U3. **Metadata bundle**: Download prediction + confidence + PLY with model/checkpoint metadata.

## Decisions
- Companion demo, not research claim
- SAM 2 optional (adds complexity, separate failure mode)
- Explicit "relative scale" label on all exported data
- Use Streamlit canvas/image-coordinate component for reliable click handling
