## 2026-07-21T18:10:00Z
You are Worker M3 (worker_m3) implementing Milestone 3: Phase 0.4 Multi-Head Extension Framework & Scaffolding in depthlab/.
Your working directory is C:\Users\rushd\Downloads\prj-res\.agents\worker_m3.

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A Forensic Auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Instructions & Task Scope:
1. Read Explorer handoff reports:
   - C:\Users\rushd\Downloads\prj-res\.agents\explorer_m3_1\handoff.md
   - C:\Users\rushd\Downloads\prj-res\.agents\explorer_m3_2\handoff.md
   - C:\Users\rushd\Downloads\prj-res\.agents\explorer_m3_3\handoff.md
2. Construct the depthlab/ modular framework adhering strictly to ADR-001 and ADR-002:
   a. `depthlab/__init__.py`
   b. `depthlab/backbone/loader.py` & `depthlab/backbone/adapter.py`:
      - `FeatureStage(patch_tokens, cls_token, stage_index, embed_dim)` dataclass.
      - `FeatureBundle` holding list of FeatureStage objects across stages.
      - `DA2Backbone`: class wrapping frozen DA2 backbone checkpoint loading (depth-anything-v2-official/) with optional adapter interface (`IdentityAdapter`, `ResidualFeatureAdapter`, `LoRAAdapter`). Zero files in depth-anything-v2-official/ modified!
      - `load_da2_checkpoint` factory.
   c. `depthlab/heads/base.py` & `depthlab/heads/__init__.py`:
      - `BaseHead` abstract contract (`forward(FeatureBundle) -> dict`, `compute_loss(preds, batch) -> Tensor`, `compute_metrics(preds, batch) -> dict`).
      - `@register_head(name)` decorator and registry dispatch (`get_head`, `list_heads`).
   d. `depthlab/heads/relative_depth.py`:
      - `RelativeDepthHead` implementing DINOv2 DPT decoder head registered via `@register_head('relative_depth')`.
      - `SILogLoss` scale-invariant logarithmic loss function.
      - Metric calculations (AbsRel, delta_1 to delta_3, RMSE, SILog).
   e. `depthlab/data/`:
      - `depthlab/data/__init__.py`: `@register_dataset(name)` decorator and registry dispatch (`get_dataset`, `list_datasets`).
      - `depthlab/data/datasets.py`: `NYUv2Dataset`, `KITTIDataset`, and synthetic `DummyDataset` loaders.
      - `depthlab/data/transforms.py`: `OFFICIAL_TRANSFORM` augmentation factory matching official DA2 image prep.
   f. `depthlab/metrics/`:
      - `depthlab/metrics/depth_metrics.py`: metric functions.
      - `depthlab/metrics/dispatcher.py`: `MetricDispatcher` accumulator and `apply_feature_ablation` hook.
   g. `depthlab/trainer.py` & `train.py`:
      - Single YAML config-driven training harness with PyTorch AMP FP16, GradScaler, AdamW, cosine LR schedule, checkpointing, TensorBoard logging.
      - Automatic metadata tracking writing `experiment.yaml` in output folder (`git commit`, `checkpoint hash`, `CUDA version`, `PyTorch version`, `seed`, `dataset version`, `benchmark version`).
   h. `eval.py`:
      - Standalone evaluation script with metric dispatcher and `--ablate-layer` hook.
   i. Configuration files: `config/default.yaml` and `experiments/nyu_relative.yaml`.
   j. `verify.py`:
      - Enhance verify.py to include `check_framework_scaffolding` as test 12 to validate end-to-end framework execution.
3. Test and Verify your implementation:
   - Run `python train.py --config config/default.yaml` to ensure synthetic training pass succeeds and outputs experiment.yaml.
   - Run `python train.py --config experiments/nyu_relative.yaml` to ensure config parsing and dataset loading succeed.
   - Run `python verify.py` or `python verify.py --all` to ensure all 12 verification checks pass 100% cleanly!
4. Write your implementation report and test results to C:\Users\rushd\Downloads\prj-res\.agents\worker_m3\handoff.md.
5. Send a send_message tool call to parent (Recipient: "78803110-53f8-4299-8bf1-e0ba882962fe") summarizing your work and pointing to handoff.md.
