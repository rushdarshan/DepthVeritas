# Plan: Test-Time Scale Anchoring via Physical Priors

**Date:** 2026-07-21
**Tracking:** Issue #006
**Source:** `docs/brainstorms/2026-07-21-da2-scale-anchoring-requirements.md`
**Design doc:** `docs/designs/2026-07-21-da2-scale-anchoring-design.md`

---

## Implementation Units

### IU1: Ground plane + camera height solver

Core engine. Given a DA2 depth map + camera height prior, estimate ground plane orientation from depth gradient, derive horizon line, and solve the convex (scale, shift) system.

**Files:**
- `src/da2_anchoring/plane_solver.py` — `fit_ground_plane(depth_map)` returns plane coefficients (a,b,c) via RANSAC on (u,v,depth) gradients. `solve_scale_shift(plane_coeffs, depth_map, camera_height, focal_length_guess)` builds the 2x2 system and returns (s, t). `compute_horizon_line(plane_coeffs)` returns horizon pixel coordinate.
- `src/da2_anchoring/__init__.py` — exports top-level `anchor_scale(depth_map, config)` function.

**Tests (in-file):** Self-check asserts on synthetic data — render a perfect ground plane with known (s,t) and verify solve recovers them within 1e-4 tolerance.

**Depends on:** numpy, DA2 inference output (HxW float disparity).

---

### IU2: Camera height prior config

Default priors + validation. Provides typed config and a resolution rule when multiple cues exist.

**Files:**
- `src/da2_anchoring/config.py` — dataclass `AnchoringConfig` with `camera_height: float` (default 1.6), `fallback_enabled: bool` (True), `plane_fit_threshold: float` (0.05), `grounding_dino_confidence: float` (0.3). Height presets: `from_scene_type(scene: str) -> float` — valid scene types: `"indoor"`, `"street"`, `"aerial"`, `"drone"`.
- `src/da2_anchoring/prior_resolver.py` — merges user override → scene-type default → hardcoded fallback. One-hot decision.

---

### IU3: Object-fallback scale solver

When plane-fit residual > threshold, detect known-scale objects via Grounding DINO + DA2 depth contours and solve (s, t) from known physical dimensions.

**Files:**
- `src/da2_anchoring/object_fallback.py` — `detect_known_scale_objects(rgb_image)` returns candidate bounding boxes + class labels. `solve_from_object_depth(depth_map, bbox, class_label)` applies the two closed-form variants from design doc §4 (person-height and car-height solvers). `select_best_candidate(candidates)` picks the lowest-residual across all detected objects.
- `scripts/download_grounding_dino.py` — one-shot download of GDINO checkpoint and config (idempotent, checksum-verified).

**Tests (in-file):** Synthetic rendering of a person (1.7m tall cylinder) at known depth; verify recovered scale within 3% error.

**Depends on:** Grounding DINO (PyTorch), torchvision, IU1 solver helpers.

---

### IU4: KITTI evaluation harness

End-to-end eval on KITTI metric depth split. Reports AbsRel, δ1-3, RMSE. One command to reproduce.

**Files:**
- `scripts/eval_kitti.py` — downloads KITTI split, runs `anchor_scale` per frame, collates metrics. Outputs table + histogram of scale errors.
- `scripts/benchmark_latency.py` — times each sub-module (plane fit, solve, object detection, total) across 100 frames on A100. Writes to `results/benchmark.json`.

**Tests (in-file):** `assert` that computed metrics match a golden value on a 10-frame canned subset.

**Depends on:** IU1 + IU3, KITTI raw dataset (~170GB, download handled by script), DA2 Large.

---

## Dependency Order

```
IU1 (solver engine)
 ├── IU2 (config)
 └── IU3 (object fallback) ── depends on IU1 solve helpers
      └── IU4 (eval) ── depends on IU1 + IU3
```

Implement in IU1→IU2→IU3→IU4 order. IU2 is trivially parallelizable with IU1.

## Acceptance

- `scripts/eval_kitti.py --method plane` → AbsRel < 0.10 (tgt: 0.085).
- Scale error histogram median < 8%.
- Plane-fallback coverage ≥ 80% on KITTI (fallback triggers on ≤ 20% of frames).
- Single-frame solve < 1s on A100 (w/o Grounding DINO). Grounding DINO adds ~0.5s.

---

## Scientific Rigor

### Hypothesis

H0: A closed-form (scale, shift) solve from a RANSAC ground-plane prior recovers metric depth from DA2 disparity with AbsRel < 0.10 on KITTI, requiring no learned refinement.

H1: Object-fallback (person/car priors) recovers scale when plane-fit residual exceeds threshold, maintaining AbsRel < 0.12 on the subset where plane-only fails.

Each is falsifiable — failure means the physical-prior assumptions (planar ground, known camera height, known object dimensions) do not transfer to the DA2 latent space as modeled.

### 3x Seeds

All stochastic components seeded across 3 runs:
- RANSAC plane fit inlier sampling.
- Grounding DINO NMS threshold tie-breaking.
- KITTI frame ordering for any subsampled eval.

Seeds: `{42, 1337, 20260721}`. Report mean ± std across seeds for all metrics. Non-determinism from GPU ops logged via `torch.use_deterministic_algorithms(True)`; if unsupported ops force non-determinism, note which and increase seed count to 5.

### Ablation Chains

Chain A — plane-solver contribution:
1. No anchoring (raw DA2 disparity, unit scale).
2. Oracle scale (ground-truth median ratio).
3. Plane-only solve (IU1).
4. Plane + object fallback (IU1+IU3).

Chain B — fallback component isolation:
1. Person-only fallback.
2. Car-only fallback.
3. Person + car fusion.
4. All detected classes.

Chain C — robustness:
1. Camera height perturbed ±20%.
2. Focal length perturbed ±10%.
3. Plane fit threshold varied {0.01, 0.05, 0.10, 0.20}.
4. Grounding DINO confidence threshold varied {0.1, 0.3, 0.5}.

Results for all ablation conditions in `results/ablations/` as JSON + markdown table.

### Threats to Validity

- **KITTI distribution mismatch**: KITTI is street-driving, well-conditioned ground planes. Degraded scene types (indoor cluttered, aerial, handheld) not represented. External validity limited until tested on NYUv2 or Middlebury.
- **Object-fallback label leakage**: Grounding DINO trained on internet images that may include KITTI-like street scenes. Cross-dataset eval (nuScenes, Waymo) needed to confirm generalization.
- **Camera height ambiguity**: Prior hardcoded at 1.6m for street scenes. Dashcam, hood-mount, or truck POV violates this. Report sensitivity (Ablation C1).
- **Temporal consistency not modeled**: Frame-by-frame independent solve — no smoothness prior. Error per-frame independent; variance across frames is not yet bounded.
- **RANSAC degeneracy**: On near-degenerate depth gradients (empty ground, extreme close-up), RANSAC may fit a numerically unstable plane. Log condition number of the solved linear system per frame; flag frames where condition number > threshold.

### Risk Management

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Plane fit fails on non-planar scenes | Medium | High — fallback has higher latency | Hard residual gate; if residual > 2× threshold, skip anchor and emit warning |
| Grounding DINO checkpoint download fails at inference time | Low | High — object-fallback unusable | Download in `scripts/download_grounding_dino.py` with retry (3×, exponential backoff); ship pre-downloaded hash-verified checkpoint as CI artifact backup |
| Camera height prior wrong for deployment scene | Medium | Medium — biased scale | Height presets per scene type; user override always wins; log chosen height at INFO |
| RANSAC timing variance across resolutions | Low | Low — < 1s target relaxed to < 2s | Downsample depth map to 320×240 for plane fit; upsample recovered scale to full resolution |
| DA2 model swap invalidates depth distribution | Medium | Medium — solve constants may drift | IU1 config exposes `depth_mean_prior` and `depth_std_prior` tunable; pin DA2 version in CI |

### Computational Budget

| Component | Budget (A100 frame) | Tracking |
|-----------|--------------------|----------|
| Plane fit + solve | 200 ms | Timer in `plane_solver.py` |
| Object detection (GDINO) | 500 ms | Script-level per-call timer |
| Total with fallback | 800 ms | Benchmark script |
| Total without fallback | 250 ms | Benchmark script |
| CI eval run (KITTI, 200 frames) | 5 min | GitHub Actions timeout |
| Ablation sweep (Chain A+B+C, 3 seeds) | 45 min | Orchestrated by `scripts/run_ablations.py` |

GPU-hour budget: **≤ 15 A100-hours** for all experiments (KITTI eval + 3-seed ablations). Log cumulative GPU-hours to `results/cost.json`. If budget exceeded, file a budget-expansion issue before continuing.

### Cost Metrics

Logged per run to `results/cost.json`:
- `gpu_hours`: wall-clock × GPU count.
- `gpu_model`: e.g. `"A100-SXM-80GB"`.
- `co2_kg_estimated`: using `codecarbon` if installed; otherwise Power Usage Effectiveness (PUE = 1.1) × TDP × runtime formula.
- `total_frames_processed`.
- `inference_only_gpu_hours`: excludes eval harness overhead.
