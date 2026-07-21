import json
import hashlib
import numpy as np
import torch
import sys
from pathlib import Path

root = Path('.').resolve()
golden_dir = root / 'golden'
manifest_path = golden_dir / 'fixture_manifest.json'

print("=== 1. MANIFEST SCHEMA & SAMPLE COUNTS ===")
with open(manifest_path, 'r') as f:
    manifest = json.load(f)

required_top_keys = [
    'generator', 'generated_at', 'model_variant', 'checkpoint_filename',
    'checkpoint_sha256', 'input_size', 'torch_version', 'cuda_available',
    'device', 'samples_count', 'samples'
]
missing_keys = [k for k in required_top_keys if k not in manifest]
print("Missing top-level keys:", missing_keys)
print("Samples count in manifest header:", manifest.get('samples_count'))
print("Actual samples list length:", len(manifest.get('samples', [])))
assert len(manifest.get('samples', [])) == 10, "Expected 10 samples"

print("\n=== 2. SHA256 CHECKSUM INTEGRITY ===")
def get_sha256(p):
    sha = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            sha.update(chunk)
    return sha.hexdigest()

ckpt_path = root / 'checkpoints' / manifest['checkpoint_filename']
ckpt_sha = get_sha256(ckpt_path)
print(f"Checkpoint SHA256 match: {ckpt_sha == manifest['checkpoint_sha256']} ({ckpt_sha})")

sample_keys = [
    'sample_id', 'image_filename', 'depth_npy_filename', 'vis_png_filename',
    'image_path', 'depth_npy_path', 'vis_png_path', 'height', 'width',
    'channels', 'min_depth', 'max_depth', 'mean_depth', 'std_depth',
    'image_sha256', 'depth_npy_sha256', 'vis_png_sha256'
]

checksum_errors = []
stats_errors = []

for idx, sample in enumerate(manifest['samples']):
    s_id = sample['sample_id']
    m_keys = [k for k in sample_keys if k not in sample]
    if m_keys:
        checksum_errors.append(f"{s_id} missing sample keys: {m_keys}")
    
    img_p = root / sample['image_path']
    npy_p = root / sample['depth_npy_path']
    vis_p = root / sample['vis_png_path']
    
    img_sha = get_sha256(img_p)
    npy_sha = get_sha256(npy_p)
    vis_sha = get_sha256(vis_p)
    
    if img_sha != sample['image_sha256']:
        checksum_errors.append(f"{s_id} image SHA mismatch: calc {img_sha} vs manifest {sample['image_sha256']}")
    if npy_sha != sample['depth_npy_sha256']:
        checksum_errors.append(f"{s_id} depth npy SHA mismatch: calc {npy_sha} vs manifest {sample['depth_npy_sha256']}")
    if vis_sha != sample['vis_png_sha256']:
        checksum_errors.append(f"{s_id} vis png SHA mismatch: calc {vis_sha} vs manifest {sample['vis_png_sha256']}")
        
    depth_arr = np.load(npy_p)
    if depth_arr.dtype != np.float32:
        stats_errors.append(f"{s_id} depth npy dtype is {depth_arr.dtype}, expected float32")
    h, w = depth_arr.shape
    if (h, w) != (sample['height'], sample['width']):
        stats_errors.append(f"{s_id} shape mismatch: arr ({h},{w}) vs manifest ({sample['height']},{sample['width']})")
        
    c_min = float(np.min(depth_arr))
    c_max = float(np.max(depth_arr))
    c_mean = float(np.mean(depth_arr))
    c_std = float(np.std(depth_arr))
    
    if not np.isclose(c_min, sample['min_depth'], rtol=1e-5, atol=1e-6):
        stats_errors.append(f"{s_id} min_depth mismatch: arr {c_min} vs manifest {sample['min_depth']}")
    if not np.isclose(c_max, sample['max_depth'], rtol=1e-5, atol=1e-6):
        stats_errors.append(f"{s_id} max_depth mismatch: arr {c_max} vs manifest {sample['max_depth']}")
    if not np.isclose(c_mean, sample['mean_depth'], rtol=1e-5, atol=1e-6):
        stats_errors.append(f"{s_id} mean_depth mismatch: arr {c_mean} vs manifest {sample['mean_depth']}")
    if not np.isclose(c_std, sample['std_depth'], rtol=1e-5, atol=1e-6):
        stats_errors.append(f"{s_id} std_depth mismatch: arr {c_std} vs manifest {sample['std_depth']}")

print("Checksum errors count:", len(checksum_errors))
for err in checksum_errors:
    print("  ERROR:", err)

print("Stats errors count:", len(stats_errors))
for err in stats_errors:
    print("  ERROR:", err)

if not missing_keys and ckpt_sha == manifest['checkpoint_sha256'] and len(checksum_errors) == 0 and len(stats_errors) == 0:
    print("\nALL MANIFEST, CHECKSUM, AND DEPTH STATS CHECKS PASSED PERFECTLY!")
