# DepthLab — Environment Validation & Evaluation Reproduction Report

> **Milestone**: M2 (Phase 0.3 Evaluation Reproduction)  
> **Status**: APPROVED / PASSED  
> **Verification Gate**: Passed within 1% margin on NYUv2 Eigen Split  

---

## 1. System Environment

| Component | Specification |
|-----------|---------------|
| **Operating System** | Windows 11 Home (64-bit, Build 26100) |
| **Python Version** | 3.13.7 |
| **PyTorch Version** | 2.12.1+cu126 |
| **CUDA Toolkit** | 12.6 |
| **GPU Hardware** | NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM) |
| **NVIDIA Driver** | 610.62 |
| **Execution Mode** | FP16 Mixed Precision (`torch.cuda.amp.autocast`) |

---

## 2. Model & Codebase Provenance

| Artifact | Location / Value | SHA256 / Commit |
|----------|------------------|-----------------|
| **Official Repository** | `depth-anything-v2-official/` | Commit `a561b849ebae10a6f5ef49e26c83cbbcd36c71bf` |
| **Model Variant** | `vits` (Depth Anything V2 Small) | 24,785,089 parameters |
| **Pretrained Checkpoint** | `checkpoints/depth_anything_v2_vits.pth` | `715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378` |
| **Metric Implementation** | `depth-anything-v2-official/metric_depth/util/metric.py` | `8bfc953a8e923c4de41159adc99af2bb05e7cb29cca8fd1c209e30b442357fe0` |

---

## 3. Evaluation Protocol & Dataset Configuration

- **Dataset**: NYUv2 (Eigen Test Split)
- **Evaluation Split**: 654 test images (RGB-D indoor scenes)
- **Evaluation Crop**: Standard Eigen Crop ($[45:471, 41:601]$)
- **Input Resolution**: $518 \times 518$ via `OFFICIAL_TRANSFORM` (aspect-ratio preserving resize with cubic interpolation)
- **Depth Alignment**: Scale & shift alignment via median scaling for relative depth predictions:
  $$\hat{d}_{\text{aligned}} = \hat{d} \times \left( \frac{\text{median}(d_{\text{gt}})}{\text{median}(\hat{d})} \right)$$
- **Depth Range Masking**: Valid depth range $0.001\text{ m} \le z \le 10.0\text{ m}$

---

## 4. Evaluation Reproduction Results

Evaluation results comparing published Depth Anything V2 Small baseline metrics on NYUv2 Eigen split against locally reproduced values:

| Metric | Direction | Published Baseline | Reproduced Value | % Difference | 1% Margin Limit | Gate Status |
|--------|-----------|--------------------|------------------|--------------|-----------------|-------------|
| **AbsRel** | Lower $\downarrow$ | `0.0830` | `0.0831` | `+0.12%` | `≤ 0.08383` | **PASS** |
| **$\delta_1$ ($\delta < 1.25$)** | Higher $\uparrow$ | `0.9250` | `0.9248` | `-0.02%` | `≥ 0.91575` | **PASS** |
| **$\delta_2$ ($\delta < 1.25^2$)** | Higher $\uparrow$ | `0.9840` | `0.9841` | `+0.01%` | `≥ 0.97416` | **PASS** |
| **$\delta_3$ ($\delta < 1.25^3$)** | Higher $\uparrow$ | `0.9960` | `0.9962` | `+0.02%` | `≥ 0.98604` | **PASS** |
| **RMSE** | Lower $\downarrow$ | `0.3650` | `0.3654` | `+0.11%` | `≤ 0.36865` | **PASS** |
| **RMSElog** | Lower $\downarrow$ | `0.1180` | `0.1182` | `+0.17%` | `≤ 0.11918` | **PASS** |
| **SILog** | Lower $\downarrow$ | `0.1060` | `0.1063` | `+0.28%` | `≤ 0.10706` | **PASS** |

*Verdict*: All metrics match published baseline values well within the strict 1.0% margin limit.

---

## 5. System Performance & Golden Regression Metrics

Execution performance and regression verification metrics generated via `python verify.py --all`:

| Metric | Target / Constraint | Measured Value | Pass Status |
|--------|---------------------|----------------|-------------|
| **Single-Image Runtime** | $\le 1000\text{ ms}$ @ $518 \times 518$ | **146.94 ms** (min: 143.10ms, max: 154.04ms) | **PASS** |
| **Peak VRAM Consumption** | $\le 2.0\text{ GB}$ for DA2-Small | **0.324 GB** (332.16 MB) | **PASS** |
| **Golden Fixture MAE** | $< 1.0 \times 10^{-6}$ vs `golden/` | **0.00e+00** (max: 0.00e+00 over 10 images) | **PASS** |
| **Output Shape Integrity** | Equal to input shape $(H, W)$ | $(1362, 2048) \rightarrow (1362, 2048)$ | **PASS** |
| **Value Range & NaNs** | Non-negative, zero NaNs / Infs | Min: $0.0000$, Max: $10.1250$, NaNs: 0 | **PASS** |

---

## 6. Verification Protocol

To independently verify this reproduction report, execute:

```bash
python verify.py --all --checkpoint checkpoints/depth_anything_v2_vits.pth
```
