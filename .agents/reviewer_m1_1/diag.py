import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
OFFICIAL_REPO = PROJECT_ROOT / "depth-anything-v2-official"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(OFFICIAL_REPO))

import torch
import cv2
import numpy as np
from depth_anything_v2.dpt import DepthAnythingV2

model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
model.load_state_dict(torch.load(PROJECT_ROOT / 'checkpoints/depth_anything_v2_vits.pth', map_location='cpu'))
model = model.to('cuda').eval()

manifest = json.load(open(PROJECT_ROOT / 'golden/fixture_manifest.json'))

print("=== Sample FP32 vs Golden ===")
for s in manifest['samples']:
    img_path = PROJECT_ROOT / s['image_path']
    npy_path = PROJECT_ROOT / s['depth_npy_path']
    img = cv2.imread(str(img_path))
    golden = np.load(str(npy_path))
    with torch.no_grad():
        pred_fp32 = model.infer_image(img, input_size=518)
    mae_fp32 = np.mean(np.abs(pred_fp32 - golden))
    max_fp32 = np.max(np.abs(pred_fp32 - golden))
    print(f"{s['sample_id']}: FP32 MAE={mae_fp32:.2e}, Max={max_fp32:.2e}")

print("\n=== Sample FP16 vs Golden ===")
for s in manifest['samples']:
    img_path = PROJECT_ROOT / s['image_path']
    npy_path = PROJECT_ROOT / s['depth_npy_path']
    img = cv2.imread(str(img_path))
    golden = np.load(str(npy_path))
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.float16):
            pred_fp16 = model.infer_image(img, input_size=518)
    mae_fp16 = np.mean(np.abs(pred_fp16 - golden))
    max_fp16 = np.max(np.abs(pred_fp16 - golden))
    print(f"{s['sample_id']}: FP16 MAE={mae_fp16:.2e}, Max={max_fp16:.2e}")
