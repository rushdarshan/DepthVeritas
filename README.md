# DepthLab

DepthLab is a reproducible research framework for extending Depth Anything V2
with pluggable depth, uncertainty, temporal, and metric-scale methods.

## Model Architecture

DepthLab uses a layered architecture built on top of the pretrained
Depth Anything V2 Small backbone.

```text
RGB image
  -> Depth Anything V2 backbone encoder
  -> dense feature maps
  -> original DA2 depth prediction
  -> SEF head depth prediction
  -> evaluation, comparison, and visualization
```

At the repository level, the project is organized as:

- `depthlab/backbone/`: loads the Depth Anything V2 encoder and checkpoint.
- `depthlab/heads/`: custom prediction heads such as the SEF head.
- `depthlab/data/`: manifests, datasets, and transforms.
- `depthlab/metrics/`: depth metrics used during evaluation.
- `depthlab/trainer.py`: training loop and checkpoint handling.
- `dashboard.py`: Streamlit frontend for live visual comparison.

## What Is Pretrained and What Was Trained

This project did not train a depth model from scratch.

- Pretrained part: the Depth Anything V2 Small backbone checkpoint in
  `checkpoints/depth_anything_v2_vits.pth`.
- Trained part: the custom SEF head added in this repository.
- Fine-tuned part: after SEF warm-up training, the full model was fine-tuned
  end to end using the local NYU Depth V2 split.

In short, the project takes a strong pretrained monocular depth model and
extends it with a new head that improved the local NYU validation result.

## Setup and Verification

Install `requirements.txt` plus the upstream requirements in
`depth-anything-v2-official/requirements.txt`. Place the DA2 checkpoint at
`checkpoints/depth_anything_v2_vits.pth`, then run:

```powershell
python -m pytest tests -q
python verify.py --all
python train.py --config config/default.yaml --preflight
```

## Data Intake

DepthLab does not redistribute licensed datasets. Use
`data/templates/depth_manifest.csv` as the RGB-D schema, then prepare and
validate it before training:

```powershell
python scripts/prepare_depth_manifest.py --input local.csv --output data/depth_manifest.csv
python scripts/preflight.py --config config/sef_warmup.yaml
```

Rows need `image_path` and `.npy` `depth_path`; paths are relative to the
manifest. Preflight reports missing files, checkpoints, and GPU details without
loading a model or starting a training job.

## Experiments and Benchmark

`config/sef_warmup.yaml` trains the SEF head and
`config/sef_end_to_end.yaml` enables end-to-end fine-tuning. Run
`scripts/run_research_pipeline.ps1` after preflight succeeds. Use
`score.py --manifest data/manifest.json --pred_dir predictions --preflight`
before benchmark scoring.

Benchmark version `1.0.0-dev` is not frozen until licensed sample provenance,
curated counts, and baseline predictions exist. See `IMPLEMENTATION_STATUS.md`
for the distinction between verified platform code and dataset-dependent
research evidence.

## Evidence and Local Diagnostics

DepthLab records a capability matrix and hashes inputs for every local evidence
run. This prevents local, synthetic, and unavailable work from being presented
as the public six-stratum benchmark.

```powershell
python scripts/profile_da2.py --sizes 392 518 --repeats 5
python scripts/nyu_failure_studio.py --split val --limit 290
```

The first command writes a GPU latency/VRAM envelope and environment snapshot.
The second evaluates only the supplied NYU validation images under controlled
darkening, blur, JPEG, and center-crop perturbations. Its report explicitly
states that it is not failure-benchmark coverage. Each run creates a JSON
report plus a companion `*_evidence.json` file under `artifacts/`.

## Local Training Summary

The completed local experiment used:

- Dataset: NYU Depth V2 labeled archive, 1,449 RGB-depth pairs.
- Split: 1,159 training images and 290 validation images.
- Hardware: NVIDIA RTX 4050 Laptop GPU with 6 GB VRAM.
- Procedure: 20 warm-up epochs for the SEF head, then 10 end-to-end
  fine-tuning epochs.
- Input size: `392 x 392`, chosen to match the DINOv2 patch grid.

On the local aligned NYU validation comparison, `DA2 + SEF` improved over
`DA2 Small`:

- AbsRel: `0.2264 -> 0.2069`
- RMSE: `0.7641 -> 0.6863`
- delta1: `0.6448 -> 0.6731`

## Demonstration Dashboard

Start the local project dashboard to demonstrate live depth-map comparison and
the completed NYU experiment:

```powershell
streamlit run dashboard.py
```

The dashboard uses the local DA2 checkpoint and trained SEF checkpoint. It can
show the bundled NYU sample or compare depth maps for an uploaded RGB image.
