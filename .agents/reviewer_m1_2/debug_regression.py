import cv2
import numpy as np
import torch
from pathlib import Path
import json
import sys

root = Path('.').resolve()
official_repo = root / 'depth-anything-v2-official'
if str(official_repo) not in sys.path:
    sys.path.insert(0, str(official_repo))

from depth_anything_v2.dpt import DepthAnythingV2

model_config = {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
ckpt_path = root / 'checkpoints' / 'depth_anything_v2_vits.pth'

model = DepthAnythingV2(**model_config)
state_dict = torch.load(ckpt_path, map_location='cpu')
model.load_state_dict(state_dict)
model = model.to('cuda').eval()

golden_dir = root / 'golden'
manifest_path = golden_dir / 'fixture_manifest.json'
with open(manifest_path, 'r') as f:
    manifest = json.load(f)

sample_img = cv2.imread(str(root / manifest['samples'][0]['image_path']))

print("1. Running check_fp16_inference...")
with torch.no_grad():
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        d_fp16 = model.infer_image(sample_img, input_size=518)

print("2. Running check_runtime_bounds (5 iterations in FP32 or what?)...")
for _ in range(5):
    torch.cuda.synchronize()
    with torch.no_grad():
        _ = model.infer_image(sample_img, input_size=518)
    torch.cuda.synchronize()

print("3. Running check_vram_bounds...")
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
with torch.no_grad():
    _ = model.infer_image(sample_img, input_size=518)

print("4. Running check_golden_regression...")
with torch.no_grad():
    pred_depth = model.infer_image(sample_img, input_size=518)

golden_depth = np.load(root / manifest['samples'][0]['depth_npy_path'])
mae = float(np.mean(np.abs(pred_depth - golden_depth)))
print(f"Resulting MAE: {mae:.6e}")
