---
date: 2026-07-21
topic: da2-research-platform-restructure
focus: consolidate 7 extensions into a cohesive, reproducible research platform
mode: elsewhere-software
---

# Ideation: DA2 Research Platform Restructure

## Grounding Context
Seven Depth Anything V2 extension plans exist (SEF, temporal consistency, multi-head framework, failure benchmark, promptable metric, scale anchoring, uncertainty refinement). External critique identified the core problem: building 7 parallel extensions dilutes the research story. The solution: restructure into a single research platform with one flagship contribution.

## Ranked Ideas

### 1. Reproduce DA2 First (Phase 0)
**Description:** Before building anything new, reproduce the official DA2 results exactly on the target hardware (RTX 4050 6GB, 16GB RAM). This verifies the baseline and gives a trusted starting point for all future experiments.
**Warrant:** `reasoned:` If reproduction fails, failures in later experiments could come from the benchmark, implementation, or research idea — three unknown variables instead of one. A verified baseline isolates the research variable.
**Rationale:** This is the single most important scientific discipline move. Without it, every negative result is uninterpretable.
**Downsides:** 1-2 weeks before any novel work begins.
**Confidence:** 95%
**Complexity:** Low (reproduce existing pipeline)
**Status:** Unexplored

### 2. Build and Freeze the Failure-Mode Benchmark (Phase 1)
**Description:** After DA2 is reproduced, build the stratified failure-mode benchmark. Freeze it before any research head development begins. Every future model evaluated on the same protocol.
**Warrant:** `direct:` Already defined in `docs/brainstorms/2026-07-21-da2-failure-benchmark-requirements.md` and `docs/plans/2026-07-21-004-feat-da2-benchmark-plan.md`.
**Rationale:** The benchmark transforms from "just another component" into infrastructure. Researchers think Benchmark → Model, not Model → Benchmark.
**Downsides:** Requires 1-2 weeks of curation effort before research begins.
**Confidence:** 95%
**Complexity:** Medium (curation + scoring script)
**Status:** Unexplored

### 3. One Flagship Research Contribution
**Description:** Pick exactly one of SEF, temporal consistency, uncertainty refinement, promptable metric, or scale anchoring. Build it, evaluate it on the frozen benchmark, run ablations, significance tests, and cost analysis. The rest are deferred.
**Warrant:** `direct:` Current 7 plans are independent peers — this restructures them into parent-child with the platform.
**Rationale:** "I advanced one research direction and thoroughly evaluated it" beats "I built six partially finished ideas" in every context (viva, interview, paper).
**Downsides:** Requires choosing — other ideas may feel like opportunity cost.
**Confidence:** 90%
**Complexity:** Decision (choice, not implementation)
**Status:** Unexplored

#### The strongest asset: Framework + Benchmark + Flagship
The combination of all three is stronger than any single component. Many students modify a model; very few build reusable research infrastructure. The framework+benchmark together tell a story of a reproducible research platform, with the flagship contribution as the first demonstration.

#### Decision filter — 4 questions before choosing
Do not choose SEF vs Temporal based on which sounds cooler. Decide after:
1. What limitation is still unsolved in the latest literature?
2. Can I improve it on consumer hardware (6GB)?
3. Can I evaluate it rigorously with the benchmark?
4. Is the improvement measurable with the benchmark's metrics?

If any answer is "no," don't build it.

#### Publication Priority Ranking

| Rank | Topic | Novelty | Success Chance | Overall Recommendation |
|------|-------|---------|---------------|----------------------|
| 1 | Failure Benchmark | High | Very High | ⭐⭐⭐⭐⭐ — guaranteed useful artifact |
| 2 | Temporal Consistency | High | High | ⭐⭐⭐⭐⭐ — well-defined, fits 6GB |
| 3 | Surface Existence Field | Very High | Medium | ⭐⭐⭐⭐☆ — highest ceiling, highest risk |
| 4 | Uncertainty Refinement | Medium-High | Medium | ⭐⭐⭐⭐☆ — builds naturally on SEF |
| 5 | Scale Anchoring | Medium | High | ⭐⭐⭐⭐☆ — clean but narrower |
| 6 | Promptable Metric Depth | Medium | High | ⭐⭐⭐☆☆ — practical, less novel |

#### Publication Strategy

| Paper | Topic | Type | Content |
|-------|-------|------|---------|
| Paper 1 | DepthLab Platform | Engineering + Benchmark | Framework, experiment management, benchmark v1.0, standardized eval |
| Paper 2 | Flagship Head | Research | SEF or Temporal — full rigour, ablations, significance, cost metrics |
| Paper 3 | Uncertainty Refinement | Extension | Builds on SEF (if SEF was chosen); natural sequel |
| Paper 4 | Scale Anchoring or Promptable | Application | Application-focused; lowest priority |

### 4. Experimental Rigor Infrastructure
**Description:** Every experiment automatically saves: git commit, config YAML, random seed, dataset version, CUDA version, PyTorch version, checkpoint, TensorBoard logs, and metrics JSON. One-command reproducibility: `python train.py --config experiments/exp.yaml`.
**Warrant:** `reasoned:` If someone cannot reproduce an experiment, the model quality is irrelevant. Industrial labs expect this as baseline.
**Rationale:** Low effort (~2 days), high return. Transforms project from "student code" to "research infrastructure."
**Downsides:** Adds ~2 days of framework development.
**Confidence:** 95%
**Complexity:** Low
**Status:** Unexplored

### 5. Hypotheses + Risk Management per Head
**Description:** Every research head starts with: hypothesis, null hypothesis, success criteria, expected outcome, possible failure modes, and fallback experiments. This forces scientific framing rather than feature implementation.
**Warrant:** `reasoned:` Most student projects implement features and then try to retrofit a research story. Starting with a hypothesis inverts this — the experiment answers a question rather than demonstrating a capability.
**Rationale:** Even negative results become publishable ("our hypothesis was wrong; here's why"). Currently no plan has this framing.
**Downsides:** Requires thinking time before coding.
**Confidence:** 90%
**Complexity:** Low (adds ~1 page per plan)
**Status:** Unexplored

### 6. Statistical Significance + Cost Metrics
**Description:** Every evaluation reports mean ± std across 3+ seeds. Always include cost metrics (FPS, VRAM, params, FLOPs, training time) alongside accuracy.
**Warrant:** `reasoned:` 0.084 ± 0.002 is scientifically defensible. 0.084 is not — it could be random variation. `direct:` External critique explicitly flagged this gap.
**Rationale:** Without significance bars, a reviewer can always say "maybe this is random." With them, the result stands.
**Downsides:** 3x training cost per experiment.
**Confidence:** 90%
**Complexity:** Low (3x compute, trivial code change)
**Status:** Unexplored

### 7. Ablation Chains
**Description:** Don't compare Baseline → Mine. Compare Baseline → Baseline+A → Baseline+B → Baseline+A+B. Isolate each component's contribution.
**Warrant:** `reasoned:` Without ablations, you don't know which component actually helps. A-B testing is standard science.
**Rationale:** Prevents the common failure mode of claiming a complex system works when only one simple component matters.
**Downsides:** 3-5x more experiments per contribution.
**Confidence:** 85%
**Complexity:** Medium (designing orthogonal components)
**Status:** Unexplored

### 8. Literature Gap Analysis Phase (Phase 0.5)
**Description:** After reproducing DA2 but before building the benchmark, do a structured literature gap analysis. Identify what conditions current models fail on and what existing work already covers. The research head should come from a gap, not from a menu.
**Warrant:** `reasoned:` A paper's opening line is "Current methods fail under three conditions. Existing work solves two. This project addresses the third." Without the gap analysis, the research question is a preference, not a discovery.
**Rationale:** Prevents choosing a head based on excitement rather than evidence. The benchmark results + gap analysis together should point to the most impactful direction.
**Downsides:** ~3-5 days of reading. May narrow options.
**Confidence:** 90%
**Complexity:** Low (reading + synthesis)
**Status:** Unexplored

### 9. Threats to Validity Section
**Description:** Every research contribution document includes a structured threats to validity analysis: internal validity (dataset bias, initialization sensitivity), external validity (scene type limitations), construct validity (metric fidelity), and statistical validity (run count).
**Warrant:** `reasoned:` This is standard in rigorous ML papers. It preempts reviewer criticism and shows awareness of limitations.
**Rationale:** Makes the work feel mature and defended rather than optimistic. Also serves as a checklist for what the ablation suite must cover.
**Downsides:** Adds ~1 page per document.
**Confidence:** 90%
**Complexity:** Low (writing, not code)
**Status:** Unexplored

### 10. Experiment Versioning (EXP-NNN)
**Description:** Replace ad-hoc "Run 1/2/3" with structured EXP-NNN folders. Each experiment references: code version (git commit), dataset version, benchmark version, config path, checkpoint path, and paper figures it supports. One-command reproducibility.
**Warrant:** `reasoned:` Months later, "Run 1" is meaningless. "EXP-004 with git commit a3f2c1" is exact. Makes paper figure regeneration trivial.
**Rationale:** This is what separates research infrastructure from student code. Each experiment folder contains everything needed to understand and reproduce it.
**Downsides:** Requires discipline to maintain.
**Confidence:** 90%
**Complexity:** Low (folder conventions + metadata logging)
**Status:** Unexplored

#### Experiment Manifest Extension
In addition to standard metadata, each experiment record includes:
- `experiment_id` — EXP-NNN
- `paper_section` — which paper section this experiment supports
- `figure_number` — which figure(s) this experiment generates
- `benchmark_version` — which benchmark version was used
- `git_commit` — pinned code version
- `dataset_version` — pinned dataset version

This makes figure regeneration fully automatic: re-run EXP-042 and the figure for §4.2/Fig 3 is reproduced identically.

### 11. Benchmark Versioning
**Description:** Once frozen, the benchmark uses semantic versions (v1.0, v1.1, v2.0). Every experiment records which benchmark version it used. Never silently change the benchmark.
**Warrant:** `reasoned:` Without versioning, later comparisons are ambiguous — did the model improve, or did the benchmark change?
**Rationale:** Prevents the common failure mode of silently updating the evaluation set and getting non-comparable results.
**Downsides:** Requires discipline to not tweak.
**Confidence:** 95%
**Complexity:** Trivial (version string in config)
**Status:** Unexplored

### 12. Error Taxonomy
**Description:** Categorize failures by type (glass, mirrors, rain, fog, night, thin wires, transparent objects, textureless walls, moving objects). Report per-category improvement, not just aggregate metrics.
**Warrant:** `reasoned:` "SEF improves glass +12%, water +18%, mirror +3%" is more compelling than "AbsRel improved 0.5%." The per-category story is the paper's narrative.
**Rationale:** Aggregate metrics hide what actually changed. Error taxonomy makes the contribution visible and defendable.
**Downsides:** More evaluation surface area.
**Confidence:** 90%
**Complexity:** Low (extend benchmark strata with sub-strata)
**Status:** Unexplored

### 13. Computational Budget
**Description:** Explicit resource tracking: GPU memory ≤6GB, single experiment ≤24h, full ablation suite ≤7 days, storage ≤500GB, fixed checkpoint retention.
**Warrant:** `reasoned:` On a 6GB GPU, compute is the binding constraint. A budget prevents designing experiments that are infeasible before implementation begins.
**Rationale:** Industrial research groups track this. It forces realistic planning and prevents wasted implementation time.
**Downsides:** None.
**Confidence:** 95%
**Complexity:** Trivial (table in plan)
**Status:** Unexplored

### 14a. Benchmark as the Moat
**Description:** The benchmark's stratified design and error taxonomy sub-strata are the project's strongest differentiator. Anyone can compare AbsRel; very few compare glass vs mirrors vs thin wires vs HDR vs low light vs temporal flicker. If external researchers start using the benchmark, that is a significant contribution independent of any model improvement.
**Warrant:** `direct:` The benchmark plan defines 6 strata with sub-strata — this level of failure-mode granularity is absent from NYUv2, KITTI, DIODE, and Sintel.
**Rationale:** The benchmark is the project's moat because it's a reusable community resource that outlives any single model. It's also the safest contribution — even if every research head fails, the benchmark alone is publishable.
**Downsides:** None — already planned.
**Confidence:** 95%
**Complexity:** Already scoped
**Status:** Unexplored

### 14b. Separate Engineering from Science
**Description:** The framework has two distinct layers. Engineering layer: config, logging, checkpointing, plugin system, visualization. Science layer: hypotheses, benchmark, evaluation, ablations, statistical analysis. Keep them separate so the platform can evolve without affecting conclusions.
**Warrant:** `reasoned:` When engineering and science are coupled, platform bugs can invalidate scientific results, and platform improvements can change the evaluation surface.
**Rationale:** Clean separation means the platform and research can evolve independently.
**Downsides:** Requires architectural discipline.
**Confidence:** 85%
**Complexity:** Medium (framework re-architecture)
**Status:** Unexplored

### 15. Publication Roadmap (Research Milestones)
**Description:** Define research milestones rather than code milestones: DA2 reproduced → benchmark validated → baseline on benchmark → novel contribution → ablations complete → paper figures generated → paper draft → public release.
**Warrant:** `reasoned:` Code milestones (U1-U5) measure implementation progress. Research milestones measure scientific progress. They are not the same.
**Rationale:** Prevents shipping code while the paper is unwritten. The milestone is "paper figures generated," not "training loop works."
**Downsides:** None — orthogonal to code plan.
**Confidence:** 90%
**Complexity:** Trivial (parallel tracking)
**Status:** Unexplored

### 16. Concrete Next-Steps Checklist
**Description:** A precise execution checklist that operationalizes the whole roadmap into day-1 actions:
1. Reproduce DA2 exactly on RTX 4050 — don't write new model code yet
2. Read 20-30 recent papers (CVPR/ICCV/ECCV/NeurIPS) on monocular depth to identify the remaining gap
3. Finish the framework so every experiment is reproducible
4. Build and freeze the benchmark
5. Run DA2 on the benchmark to identify its real weaknesses
6. Let those results determine whether SEF or Temporal becomes the first flagship

**Warrant:** `reasoned:` The biggest technical risk is choosing the wrong research question. Steps 1-6 invert the usual "pick an idea and build it" pattern into "understand the gap first, then pick." The literature survey and benchmark results together provide evidence for the choice, not preference.
**Rationale:** This checklist is the operational heart of the entire restructure. Everything else (framework, benchmark, hypotheses, rigor) supports this sequence.
**Downsides:** None.
**Confidence:** 95%
**Complexity:** Trivial (sequence discipline)
**Status:** Unexplored

## Rejection Summary

| # | Idea | Reason Rejected |
|---|------|-----------------|
| 1 | Plugin auto-discovery (P2) | Worthwhile but lower priority than experimental rigor |
| 2 | Build benchmark before reproducing DA2 | Without verified baseline, later failures are uninterpretable |
| 3 | All 7 heads in parallel | Dilutes research story |
| 4 | Fancy developer experience features | Better experiments > fancy plugin loading |
| 5 | Locking in one head before baseline analysis | Let benchmark + gap analysis inform the choice, not upfront preference |

## Consolidated Roadmap

| Priority | Task | Time |
|----------|------|------|
| P0 | Reproduce DA2 exactly (baseline) | Weeks 1-2 |
| P0.5 | Literature gap analysis | Week 2 (alongside DA2 repro) |
| P1 | Build shared framework with reproducibility built in | Weeks 1-2 (parallel with P0) |
| P1 | Build and freeze the benchmark | Week 3 |
| P1 | Choose research head from gap analysis + benchmark results | End of Week 3 |
| P2 | Implement one flagship research head | Weeks 4-6 |
| P2 | Run ablations, 3x seeds, significance tests, cost analysis | Week 7 |
| P3 | Add failure analysis, error taxonomy, publication visuals | Week 8 |
| P3 | Write paper draft (Paper 1: DepthLab + Paper 2: Flagship), clean codebase | Week 8 |
| P4 | Paper 3 (uncertainty refinement) if SEF was flagship; else Paper 2 extension | If time |
| P4 | Paper 4 (application-focused), Polish DX | If time |

### Publication Pipeline

| Paper | Title | Content |
|-------|-------|---------|
| Paper 1 | DepthLab: A Reproducible Research Platform for Monocular Depth Estimation | Framework + Benchmark v1.0 + standardized evaluation — mostly engineering, but valuable |
| Paper 2 | Flagship head (SEF or Temporal Consistency) | Full scientific contribution with hypothesis, ablations, significance, failure analysis |
| Paper 3 | Uncertainty Refinement (builds on SEF) | Natural sequel if SEF was Paper 2; demonstrates platform extensibility |
| Paper 4 | Scale Anchoring or Promptable Metric | Application-focused; lowest priority |

## Revisions Needed in Existing Plans

The following plans need updates to reflect this restructure:

1. **Framework plan** (003): Add EXP-NNN metadata capture, one-command reproducibility, separate engineering/science layers, computational budget, benchmark versioning
2. **Benchmark plan** (004): Add benchmark versioning, error taxonomy sub-strata, semantic version release process
3. **All head plans** (001, 002, 005, 006, 007): Add hypothesis + null hypothesis + success criteria + risk management + threats to validity + failure modes + computational budget
4. **All head plans**: Add 3x seeds, mean±std, confidence intervals, cost metrics to evaluation sections
5. **All head plans**: Add ablation chains to experiment design
6. **New file**: ROADMAP.md with research milestones + publication roadmap
7. **Project branding**: Rename to "DepthLab" or similar — a research platform, not an extension set
