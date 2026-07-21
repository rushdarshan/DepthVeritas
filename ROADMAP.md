# DepthLab — Research Roadmap

> A reproducible research platform for monocular depth estimation built on Depth Anything V2.

---

## Phase 0.1: Environment Validation (1-2 hours)
**Status:** ❌ Not started

| Check | Pass Condition |
|-------|---------------|
| Official repo | Clones and checkpoint downloads without error |
| Inference | One image runs on CUDA without warnings |
| VRAM | Peak ≤ 2GB for DA2-Small at 518×518 |
| FP16 | Inference works in mixed precision |

- [ ] Deliverable: `docs/reproduction/environment.md` (Python, PyTorch, CUDA, GPU, driver, commit hash, checkpoint SHA256)

---

## Phase 0.2: Inference Reproduction (1 day)
**Status:** ❌ Not started

| Check | Pass Condition |
|-------|---------------|
| Visual output | Depth maps match official examples |
| Runtime | Single image ≤ 1 second at 518×518 |
| No regressions | No NaNs, output shape correct (H×W), non-negative range |

- [ ] Run official `run.py` on sample images — do not modify anything
- [ ] Golden sample: save 10 images with official depth outputs as `golden/` fixtures
- [ ] Create `verify.py`: checkpoint loads, inference runs, output shape, no NaNs, runtime bounded

---

## Phase 0.3: Evaluation Reproduction
**Status:** ❌ Not started

| Check | Pass Condition |
|-------|---------------|
| Metrics | Within 1% of published values on NYUv2 Eigen split (AbsRel, δ1, RMSE) |
| Golden regression | `verify.py` reports MAE < 1e-6 against golden sample |
| Report | All fields populated (see ADR-002 §5 for required fields) |

- [ ] Run official evaluation on NYUv2 or KITTI Eigen split
- [ ] **GO / NO-GO gate:** If metrics don't match, investigate — do not continue to Phase 1
- [ ] Deliverable: Reproduction report with checkpoint hash, commit, metrics table, screenshots, runtime, VRAM

---

## Phase 0.4: Framework Extraction
**Status:** ❌ Not started

- [ ] Extract DA2 as a git submodule (`depth-anything-v2-official/`)
- [ ] Build `depthlab/` directory structure per ADR-001 + ADR-002
- [ ] Implement Layer 1 adapter (`backbone.py`, `transforms.py`, `datasets.py`, `metrics.py`)
- [ ] Implement `FeatureBundle` dataclass with `FeatureStage`
- [ ] Implement `BaseHead` contract around `FeatureBundle`
- [ ] Implement `train.py` entry point with config-driven execution
- [ ] Implement `verify.py` — tests against golden sample (MAE < 1e-6)
- [ ] Implement EXP-NNN experiment folders with metadata capture

---

## Phase 0.5: Literature Gap Analysis
**Status:** ❌ Not started

- [ ] Survey 20-30 recent papers (CVPR/ICCV/ECCV/NeurIPS) on monocular depth
- [ ] Identify which gaps existing work already covers
- [ ] Select research head based on uncovered gap + benchmark results

---

## Phase 2: Failure-Mode Benchmark
**Status:** ❌ Not started

- [ ] Curate 6 strata from public datasets
- [ ] Error taxonomy sub-strata (glass, mirror, wires, etc.)
- [ ] Scoring script + evaluation protocol
- [ ] Baseline inference with DA2
- [ ] Benchmark versioning (v1.0.0)
- [ ] Documentation + arXiv paper

---

## Phase 3: Flagship Research Contribution
**Status:** ❌ Not started

- [ ] Implement primary research head (TBD from gap analysis)
- [ ] Evaluate on frozen benchmark
- [ ] Run ablations (Baseline → A → B → A+B)
- [ ] 3 seeds per experiment, report mean ± std
- [ ] Cost metrics (FPS, VRAM, params, FLOPs)

---

## Phase 4: Papers

### Paper 1 — DepthLab Platform
**Status:** ❌ Not started
- [ ] Framework description and design decisions
- [ ] Benchmark v1.0 documentation and baseline results
- [ ] Experiment management and reproducibility demonstration
- [ ] arXiv submission (dataset/benchmark track or systems track)

### Paper 2 — Flagship Research Head
**Status:** ❌ Not started
- [ ] Generate publication-quality figures
- [ ] Hypothesis, ablation chains, significance tests
- [ ] Failure analysis + error taxonomy tables
- [ ] Cost metrics and trade-off analysis
- [ ] Paper draft

### Papers 3+ (if time permits)
**Status:** ❌ Not started
- [ ] Paper 3: Uncertainty Refinement (builds on SEF if SEF was flagship)
- [ ] Paper 4: Application-focused (Scale Anchoring or Promptable Metric)

---

## Planned Timeline

| Phase | Time | Deliverable |
|-------|------|-------------|
| 0.1 | 1-2h | Environment validated, golden sample created |
| 0.2 | 1 day | Inference matches official |
| 0.3 | 1-2 days | Metrics match published ≤1% — GO/NO-GO gate |
| 0.4 | 1 week | Framework extracted, adapter built, verify.py passes |
| 0.5 | 1 week | Gap analysis complete, research head selected |
| 1 | 1-2 weeks | Framework operational |
| 2 | 1 week | Benchmark v1.0.0 frozen |
| 3 | 3-4 weeks | One research contribution with full eval |
| 4 | 1 week | Paper draft + release |
