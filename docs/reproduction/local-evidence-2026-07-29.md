# Local Evidence Report

## Scope

This report records reproducible local evidence from the supplied NYU Depth V2
data and DA2 Small checkpoint. It is not a public benchmark release, a
six-stratum failure study, or evidence for video, DINOv2, or Blender-dependent
claims.

## GPU Operating Envelope

Command: `python scripts/profile_da2.py --sizes 256 392 518 --repeats 3`

- Device: NVIDIA GeForce RTX 4050 Laptop GPU, 6,438,780,928 bytes VRAM.
- 256: rejected because DA2 inputs must be divisible by its 14-pixel patch
  size.
- 392: 58.974 ms median inference latency and 172.07 MB peak allocated VRAM.
- 518: 52.906 ms median inference latency and 225.01 MB peak allocated VRAM.

The run used CUDA 12.6, PyTorch 2.12.1+cu126, and NVIDIA driver 610.62. The
full measurement and hashed checkpoint provenance are in
`artifacts/da2_operating_envelope.json` and
`artifacts/da2_operating_envelope_evidence.json`.

## NYU Perturbation Study

Command: `python scripts/nyu_failure_studio.py --split val --limit 290 --output artifacts/nyu_failure_studio_full.json`

This deterministic local validation study measured DA2 Small after per-image
scale-and-shift alignment. Baseline AbsRel was 0.166313 and delta1 was 0.785768.
Darkening increased AbsRel to 0.170365; blur to 0.183891; JPEG compression to
0.169987. Center crop is an image-content change, not a degradation ordering,
and produced 0.131956 AbsRel on this validation split.

The evidence is reproducible from the tracked command and is explicitly
limited to the supplied 290-image NYU validation split. See
`artifacts/nyu_failure_studio_full.json` and
`artifacts/nyu_failure_studio_full_evidence.json` for all metrics, environment
facts, and input hashes.

## Blocked Evidence

The evidence capability matrix still reports these unavailable or blocked
inputs: DINOv2 weights, trusted pose/flow video data, Blender renders, and the
licensed human-reviewed six-stratum benchmark. They are not represented as
completed experiments.
