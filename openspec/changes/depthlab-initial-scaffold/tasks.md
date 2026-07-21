## 1. Environment Validation (Phase 0.1)

- [ ] 1.1 Verify official DA2 repo cloned and checkpoint downloads work
- [ ] 1.2 Run one inference image on CUDA without warnings
- [ ] 1.3 Verify VRAM usage ≤ 2GB for DA2-Small at 518×518
- [ ] 1.4 Verify FP16 inference works
- [ ] 1.5 Create `docs/reproduction/environment.md` with Python, PyTorch, CUDA, GPU, driver, commit hash, checkpoint SHA256

## 2. Inference Reproduction (Phase 0.2)

- [ ] 2.1 Run official `run.py` on 10 sample images without modifications
- [ ] 2.2 Save golden sample outputs to `golden/` (input image + depth.npy + depth.png per sample)
- [ ] 2.3 Create `golden/manifest.json` indexing all golden samples

## 3. Evaluation Reproduction (Phase 0.3)

- [ ] 3.1 Run official evaluation on NYUv2 Eigen split using DA2-Small
- [ ] 3.2 Verify AbsRel, δ1, RMSE within 1% of published values
- [ ] 3.3 Create reproduction report with all required fields per ADR-002 §5
- [ ] 3.4 **GO/NO-GO gate:** Confirm metrics match before proceeding to framework

## 4. Adapter Layer (Phase 0.4)

- [ ] 4.1 Create `depthlab/` project structure with `research/`, `runs/`, `benchmark/`, `golden/` directories
- [ ] 4.2 Set up `depth-anything-v2-official/` as repo dependency (pinned commit)
- [ ] 4.3 Implement `research/transforms.py` with single shared `OFFICIAL_TRANSFORM`
- [ ] 4.4 Implement `research/backbone.py` with `DA2Backbone` class wrapping official `DepthAnythingV2`
- [ ] 4.5 Implement `FeatureBundle` and `FeatureStage` dataclasses
- [ ] 4.6 Implement variant config resolution in `research/configs/variants.yaml`
- [ ] 4.7 Implement `research/metrics.py` re-exporting official `eval_depth()`
- [ ] 4.8 Implement `research/datasets.py` re-exporting official dataset loaders
- [ ] 4.9 Implement `research/losses.py` re-exporting official `SILogLoss`

## 5. Head Interface + Relative Depth Head

- [ ] 5.1 Implement `research/heads/base.py` with `BaseHead` abstract class and `@register_head` decorator
- [ ] 5.2 Implement `research/heads/relative.py` with `RelativeDepthHead` reproducing official DPT decoder
- [ ] 5.3 Verify relative depth head training converges on NYUv2
- [ ] 5.4 Verify head metrics match official evaluation within tolerance

## 6. Experiment Management

- [ ] 6.1 Implement `research/train.py` with config-driven training loop
- [ ] 6.2 Implement EXP-NNN experiment folder creation with auto-naming
- [ ] 6.3 Implement automatic metadata capture (git commit, CUDA, PyTorch, seed, timestamps)
- [ ] 6.4 Implement checkpointing backbone and head separately with resumption
- [ ] 6.5 Implement TensorBoard logging
- [ ] 6.6 Implement `research/eval.py` evaluation entry point

## 7. Regression Tests

- [ ] 7.1 Implement `research/verify.py` with golden sample regression tests
- [ ] 7.2 Verify: checkpoint loads, inference shape, output range, no NaNs, MAE < 1e-6 against golden
- [ ] 7.3 Add runtime and VRAM checks to verify.py

## 8. Benchmark Framework

- [ ] 8.1 Implement `benchmark/__init__.py` with `BENCHMARK_VERSION`
- [ ] 8.2 Implement `benchmark/protocol.py` with standardized scoring protocol
- [ ] 8.3 Implement per-stratum and per-sub-stratum scoring
- [ ] 8.4 Implement composite score with priority weighting and scale/shift alignment
- [ ] 8.5 Curate P0 strata (transparent/reflective, thin structures) from public datasets
- [ ] 8.6 Curate P1/P2 strata (low-light, extreme FOV, HDR, video flicker)
- [ ] 8.7 Tag all samples with error taxonomy sub-strata in manifest
- [ ] 8.8 Run DA2 baseline inference on all curated samples
- [ ] 8.9 Freeze benchmark v1.0.0 with documentation
