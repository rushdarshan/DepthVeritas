# DepthLab

DepthLab is a reproducible research framework for extending Depth Anything V2
with pluggable depth, uncertainty, temporal, and metric-scale methods.

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

## Demonstration Dashboard

Start the local project dashboard to demonstrate live depth-map comparison and
the completed NYU experiment:

```powershell
streamlit run dashboard.py
```

The dashboard uses the local DA2 checkpoint and trained SEF checkpoint. It can
show the bundled NYU sample or compare depth maps for an uploaded RGB image.
