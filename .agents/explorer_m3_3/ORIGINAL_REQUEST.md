## 2026-07-21T23:32:41Z
You are Explorer 3 (explorer_m3_3) for Milestone 3 of DepthLab.
Your working directory is C:\Users\rushd\Downloads\prj-res\.agents\explorer_m3_3.

Objective: Investigate and design the Training Harness, Eval Engine, Metric Dispatcher, YAML Configs, Feature Ablation Hook, and Reproducibility Tracker.

Instructions:
1. Read C:\Users\rushd\Downloads\prj-res\PROJECT.md and C:\Users\rushd\Downloads\prj-res\.agents\orchestrator\handoff.md.
2. Design depthlab/trainer.py & train.py:
   - Single YAML entry point: `python train.py --config config/default.yaml` or `python train.py --config experiments/nyu_relative.yaml`.
   - PyTorch AMP FP16 mixed precision, `torch.cuda.amp.GradScaler`, AdamW, cosine annealing LR schedule, model checkpointing.
   - TensorBoard logging (`torch.utils.tensorboard.SummaryWriter`).
   - Automatic metadata recording writing `experiment.yaml` into output directory (`git commit`, `checkpoint hash`, `CUDA version`, `PyTorch version`, `seed`, `dataset version`, `benchmark version`).
3. Design eval.py & depthlab/metrics/dispatcher.py:
   - Standalone eval script: `python eval.py --config experiments/nyu_relative.yaml --checkpoint ...`.
   - Metric dispatcher evaluating depth metrics and writing summary.
   - Feature ablation experiment hook (`--ablate-layer` flag allowing zeroing out specific stages in `FeatureBundle`).
4. Design YAML config structures: `config/default.yaml` (dummy/synthetic data fast path) and `experiments/nyu_relative.yaml`.
5. Design verify.py extension to test Layer 1 framework scaffolding execution.
6. Write your comprehensive analysis and design specification to C:\Users\rushd\Downloads\prj-res\.agents\explorer_m3_3\handoff.md.
7. Send a send_message tool call to parent (Recipient: "78803110-53f8-4299-8bf1-e0ba882962fe") summarizing your findings and pointing to handoff.md.
