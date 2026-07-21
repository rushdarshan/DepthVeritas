# Handoff Report — Explorer 2 (explorer_m3_2)

**From**: Explorer 2 (`explorer_m3_2`)  
**To**: Project Orchestrator / Worker M3 (`worker_m3`)  
**Date**: 2026-07-21  
**Working Directory**: `C:\Users\rushd\Downloads\prj-res\.agents\explorer_m3_2`  
**Milestone**: Milestone 3 (Multi-Head Framework & Scaffolding)  

---

## 1. Observation

Direct investigation of the codebase and official Depth Anything V2 repository yielded the following facts:

### 1.1 Official DPT Decoder & Preprocessing (`depth-anything-v2-official/`)
- **DPT Decoder Architecture** (`depth_anything_v2/dpt.py:38-150`):
  - Consists of `DPTHead` receiving feature maps from 4 intermediate ViT/DINOv2 backbone layers.
  - Encoder variants feature dimension mappings:
    - `vits`: `in_channels=384`, `features=64`, `out_channels=[48, 96, 192, 384]`
    - `vitb`: `in_channels=768`, `features=128`, `out_channels=[96, 192, 384, 768]`
    - `vitl`: `in_channels=1024`, `features=256`, `out_channels=[256, 512, 1024, 1024]`
    - `vitg`: `in_channels=1536`, `features=384`, `out_channels=[1536, 1536, 1536, 1536]`
  - Spatial token reshape: `x.permute(0, 2, 1).reshape(B, C, patch_h, patch_w)` where $patch\_h = patch\_w = H_{img} / 14$.
  - RefineNet blocks (`refinenet4` $\to$ `refinenet3` $\to$ `refinenet2` $\to$ `refinenet1`) assemble multi-scale feature maps.
  - Final projection (`output_conv1` $\to$ bilinear upsample to $patch\_h \times 14, patch\_w \times 14$ $\to$ `output_conv2` $\to$ `ReLU`).

- **Official Preprocessing Transform** (`depth_anything_v2/util/transform.py`):
  - `Resize`: Keeps aspect ratio, constrains dimensions to multiple of 14, lower bound resize method using `cv2.INTER_CUBIC`.
  - `NormalizeImage`: Mean `[0.485, 0.456, 0.406]`, Std `[0.229, 0.224, 0.225]`.
  - `PrepareForNet`: Converts HWC $\to$ CHW float32 arrays.

- **Loss & Metrics** (`metric_depth/util/loss.py` & `metric_depth/util/metric.py`):
  - **SiLogLoss**: $\mathcal{L}_{SILog} = \sqrt{\frac{1}{N} \sum d_i^2 - \lambda (\frac{1}{N} \sum d_i)^2}$ where $d_i = \log y_i - \log \hat{y}_i$ and default $\lambda = 0.5$.
  - **Evaluation Metrics**:
    - `AbsRel`: $\frac{1}{N} \sum \frac{|y_i - \hat{y}_i|}{y_i}$
    - `RMSE`: $\sqrt{\frac{1}{N} \sum (y_i - \hat{y}_i)^2}$
    - $\delta_1, \delta_2, \delta_3$: Percentage of pixels where $\max(\frac{y_i}{\hat{y}_i}, \frac{\hat{y}_i}{y_i}) < 1.25^k$ for $k \in \{1, 2, 3\}$.

---

## 2. Logic Chain & System Design

To satisfy ADR-001 and ADR-002 while maintaining 100% immutability of Layer 0 (`depth-anything-v2-official/`), the `depthlab/heads/` and `depthlab/data/` modules are designed with pluggable decorator registries (`@register_head` and `@register_dataset`).

```
                              FeatureBundle (Layer 1)
                                         │
                                         ▼
                      ┌──────────────────────────────────────┐
                      │      BaseHead (Abstract Interface)   │
                      │  - forward(FeatureBundle) -> dict    │
                      │  - compute_loss(preds, batch)        │
                      │  - compute_metrics(preds, batch)     │
                      └──────────────────┬───────────────────┘
                                         │
               ┌─────────────────────────┴────────────────────────┐
               ▼                                                  ▼
   RelativeDepthHead (@register_head('relative_depth'))    UncertaintyHead (M4)
   - DPT Decoder (DINOv2 features)                         - Learned log-var
   - SILogLoss                                             - Negative log-likelihood
   - Metrics (AbsRel, RMSE, d1-d3)
```

---

## 3. Detailed Specifications & Implementation Contracts

### 3.1 Head Registry & Abstract Base Class (`depthlab/heads/`)

#### File 1: `depthlab/heads/base.py`
Defines the `BaseHead` abstract base class.

```python
"""
depthlab/heads/base.py — Abstract Base Class for DepthLab Heads.
"""
import abc
from typing import Dict, Any
import torch
import torch.nn as nn


class BaseHead(nn.Module, abc.ABC):
    """
    Abstract Base Class for all prediction heads in DepthLab (Layer 2).
    
    All head implementations must inherit from BaseHead and implement:
      1. forward(feature_bundle) -> Dict[str, torch.Tensor]
      2. compute_loss(preds, batch) -> torch.Tensor
      3. compute_metrics(preds, batch) -> Dict[str, float]
    """

    @abc.abstractmethod
    def forward(self, feature_bundle: Any) -> Dict[str, torch.Tensor]:
        """
        Forward pass converting backbone FeatureBundle into output predictions.
        
        Args:
            feature_bundle: FeatureBundle object containing multi-stage backbone features.
            
        Returns:
            Dict containing predicted tensors, e.g. {'predicted_depth': Tensor[B, H, W]}.
        """
        pass

    @abc.abstractmethod
    def compute_loss(
        self, 
        preds: Dict[str, torch.Tensor], 
        batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Computes scalar training loss.
        
        Args:
            preds: Output dict from forward pass.
            batch: Data dictionary containing ground truth ('depth', 'valid_mask', etc.).
            
        Returns:
            Scalar Loss Tensor.
        """
        pass

    @abc.abstractmethod
    def compute_metrics(
        self, 
        preds: Dict[str, torch.Tensor], 
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Computes evaluation metrics.
        
        Args:
            preds: Output dict from forward pass.
            batch: Data dictionary containing ground truth.
            
        Returns:
            Dict mapping metric names to scalar float values.
        """
        pass
```

#### File 2: `depthlab/heads/__init__.py`
Defines the decorator-based Head Registry dispatch mechanism.

```python
"""
depthlab/heads/__init__.py — Head Registry & Dispatcher.
"""
from typing import Dict, Type, List, Any
from depthlab.heads.base import BaseHead

_HEAD_REGISTRY: Dict[str, Type[BaseHead]] = {}


def register_head(name: str):
    """
    Decorator to register a prediction head implementation in DepthLab.
    
    Usage:
        @register_head('relative_depth')
        class RelativeDepthHead(BaseHead):
            ...
    """
    def decorator(cls: Type[BaseHead]):
        if name in _HEAD_REGISTRY:
            raise ValueError(f"Head name '{name}' is already registered to {_HEAD_REGISTRY[name]}.")
        if not issubclass(cls, BaseHead):
            raise TypeError(f"Class '{cls.__name__}' must inherit from BaseHead to be registered.")
        _HEAD_REGISTRY[name] = cls
        return cls
    return decorator


def get_head(name: str, **kwargs: Any) -> BaseHead:
    """
    Factory function to instantiate a registered head by name.
    """
    if name not in _HEAD_REGISTRY:
        available = list_heads()
        raise KeyError(f"Head '{name}' not found in registry. Available heads: {available}")
    return _HEAD_REGISTRY[name](**kwargs)


def list_heads() -> List[str]:
    """
    Returns list of all registered head names.
    """
    return sorted(list(_HEAD_REGISTRY.keys()))
```

#### File 3: `depthlab/heads/relative_depth.py`
Implements `RelativeDepthHead`, `SILogLoss`, and relative depth metrics (`AbsRel`, `RMSE`, $\delta_1$-$\delta_3$).

```python
"""
depthlab/heads/relative_depth.py — DINOv2 DPT Relative Depth Head & Loss Suite.
"""
from typing import Dict, List, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from depth_anything_v2.dpt import DPTHead
from depthlab.heads.base import BaseHead
from depthlab.heads import register_head


class SILogLoss(nn.Module):
    """
    Scale-Invariant Logarithmic (SILog) Loss.
    """
    def __init__(self, lambd: float = 0.5, eps: float = 1e-6):
        super().__init__()
        self.lambd = lambd
        self.eps = eps

    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor, 
        valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if valid_mask is None:
            valid_mask = (target > 0) & (~torch.isnan(target)) & (~torch.isinf(target))
        else:
            valid_mask = valid_mask.detach()

        pred_val = torch.clamp(pred[valid_mask], min=self.eps)
        target_val = torch.clamp(target[valid_mask], min=self.eps)

        if pred_val.numel() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        diff_log = torch.log(target_val) - torch.log(pred_val)
        loss = torch.sqrt(
            torch.pow(diff_log, 2).mean() - self.lambd * torch.pow(diff_log.mean(), 2) + 1e-8
        )
        return loss


def compute_relative_depth_metrics(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    valid_mask: Optional[torch.Tensor] = None,
    align_scale: bool = True
) -> Dict[str, float]:
    """
    Computes standard depth evaluation metrics: AbsRel, RMSE, d1, d2, d3.
    Optionally performs median scale alignment for relative depth predictions.
    """
    if valid_mask is None:
        valid_mask = (target > 0) & (~torch.isnan(target)) & (~torch.isinf(target))

    pred_val = pred[valid_mask]
    target_val = target[valid_mask]

    if pred_val.numel() == 0:
        return {'abs_rel': 0.0, 'rmse': 0.0, 'd1': 0.0, 'd2': 0.0, 'd3': 0.0}

    if align_scale:
        scale = torch.median(target_val) / (torch.median(pred_val) + 1e-8)
        pred_val = pred_val * scale

    pred_val = torch.clamp(pred_val, min=1e-3)
    target_val = torch.clamp(target_val, min=1e-3)

    thresh = torch.max((target_val / pred_val), (pred_val / target_val))
    d1 = (thresh < 1.25).float().mean().item()
    d2 = (thresh < 1.25 ** 2).float().mean().item()
    d3 = (thresh < 1.25 ** 3).float().mean().item()

    diff = pred_val - target_val
    abs_rel = (torch.abs(diff) / target_val).mean().item()
    rmse = torch.sqrt(torch.pow(diff, 2).mean()).item()

    return {
        'abs_rel': abs_rel,
        'rmse': rmse,
        'd1': d1,
        'd2': d2,
        'd3': d3,
    }


@register_head('relative_depth')
class RelativeDepthHead(BaseHead):
    """
    DPT Decoder Head registered for Relative Depth Estimation.
    """
    def __init__(
        self,
        encoder_variant: str = 'vits',
        in_channels: int = 384,
        features: int = 64,
        out_channels: Optional[List[int]] = None,
        use_bn: bool = False,
        use_clstoken: bool = False,
        lambd: float = 0.5,
    ):
        super().__init__()
        
        # Configure variant defaults matching official DA2
        variant_configs = {
            'vits': {'in_channels': 384, 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'in_channels': 768, 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'in_channels': 1024, 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'in_channels': 1536, 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        
        if encoder_variant in variant_configs:
            cfg = variant_configs[encoder_variant]
            in_channels = cfg['in_channels']
            features = cfg['features']
            out_channels = cfg['out_channels']
        elif out_channels is None:
            out_channels = [48, 96, 192, 384]

        self.dpt = DPTHead(
            in_channels=in_channels,
            features=features,
            use_bn=use_bn,
            out_channels=out_channels,
            use_clstoken=use_clstoken
        )
        self.loss_fn = SILogLoss(lambd=lambd)

    def forward(self, feature_bundle: Any) -> Dict[str, torch.Tensor]:
        """
        Extracts intermediate feature tensors from FeatureBundle and passes through DPT decoder.
        """
        out_features = []
        for stage in feature_bundle.stages:
            if self.dpt.use_clstoken:
                out_features.append((stage.patch_tokens, stage.cls_token))
            else:
                out_features.append((stage.patch_tokens,))

        tokens = feature_bundle.stages[0].patch_tokens
        num_patches = tokens.shape[1]
        patch_h = int(num_patches ** 0.5)
        patch_w = patch_h

        depth = self.dpt(out_features, patch_h, patch_w)
        depth = F.relu(depth)
        depth = depth.squeeze(1)  # Shape: [B, H, W]

        return {"predicted_depth": depth}

    def compute_loss(
        self, 
        preds: Dict[str, torch.Tensor], 
        batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        pred_depth = preds["predicted_depth"]
        target_depth = batch["depth"]
        valid_mask = batch.get("valid_mask", target_depth > 0)
        return self.loss_fn(pred_depth, target_depth, valid_mask)

    def compute_metrics(
        self, 
        preds: Dict[str, torch.Tensor], 
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        pred_depth = preds["predicted_depth"]
        target_depth = batch["depth"]
        valid_mask = batch.get("valid_mask", target_depth > 0)
        return compute_relative_depth_metrics(pred_depth, target_depth, valid_mask, align_scale=True)
```

---

### 3.2 Dataset Registry, Loaders & Augmentations (`depthlab/data/`)

#### File 1: `depthlab/data/__init__.py`
Dataset Registry & Decorator.

```python
"""
depthlab/data/__init__.py — Dataset Registry & Dispatcher.
"""
from typing import Dict, Type, List, Any
from torch.utils.data import Dataset

_DATASET_REGISTRY: Dict[str, Type[Dataset]] = {}


def register_dataset(name: str):
    """
    Decorator to register a dataset class in DepthLab.
    
    Usage:
        @register_dataset('nyuv2')
        class NYUv2Dataset(Dataset):
            ...
    """
    def decorator(cls: Type[Dataset]):
        if name in _DATASET_REGISTRY:
            raise ValueError(f"Dataset '{name}' is already registered to {_DATASET_REGISTRY[name]}.")
        if not issubclass(cls, Dataset):
            raise TypeError(f"Class '{cls.__name__}' must inherit from torch.utils.data.Dataset.")
        _DATASET_REGISTRY[name] = cls
        return cls
    return decorator


def get_dataset(name: str, **kwargs: Any) -> Dataset:
    """
    Factory function to instantiate a registered dataset loader by name.
    """
    if name not in _DATASET_REGISTRY:
        available = list_datasets()
        raise KeyError(f"Dataset '{name}' not found in registry. Available datasets: {available}")
    return _DATASET_REGISTRY[name](**kwargs)


def list_datasets() -> List[str]:
    """
    Returns a sorted list of registered dataset names.
    """
    return sorted(list(_DATASET_REGISTRY.keys()))
```

#### File 2: `depthlab/data/transforms.py`
`OFFICIAL_TRANSFORM` Preprocessing & Augmentation Factory.

```python
"""
depthlab/data/transforms.py — OFFICIAL_TRANSFORM Augmentation & Preprocessing Factory.
"""
import cv2
from typing import Tuple, Union
from torchvision.transforms import Compose

from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet


def get_official_transform(
    input_size: Union[int, Tuple[int, int]] = 518, 
    is_train: bool = False
) -> Compose:
    """
    Factory returning the official Depth Anything V2 image & depth transform pipeline.
    
    Args:
        input_size: Resolution (default 518x518).
        is_train: Whether to resize target depth maps during training.
        
    Returns:
        torchvision.transforms.Compose pipeline.
    """
    if isinstance(input_size, int):
        net_w, net_h = input_size, input_size
    else:
        net_w, net_h = input_size

    return Compose([
        Resize(
            width=net_w,
            height=net_h,
            resize_target=is_train,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])


# Standard static reference matching official 518x518 pipeline
OFFICIAL_TRANSFORM = get_official_transform(input_size=518, is_train=False)
```

#### File 3: `depthlab/data/datasets.py`
`NYUv2Dataset`, `KITTIDataset`, and synthetic `DummyDataset` implementations.

```python
"""
depthlab/data/datasets.py — Dataset Loaders for NYUv2, KITTI, and Synthetic Dummy Testing.
"""
import os
from typing import Dict, Any, Optional, Tuple, List
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from depthlab.data import register_dataset
from depthlab.data.transforms import get_official_transform


@register_dataset('nyuv2')
class NYUv2Dataset(Dataset):
    """
    NYUv2 Dataset Loader for Monocular Depth Estimation.
    """
    def __init__(
        self, 
        filelist_path: str, 
        split: str = 'val', 
        input_size: int = 518,
        data_dir: Optional[str] = None
    ):
        self.split = split
        self.input_size = input_size
        self.data_dir = data_dir or ""

        if os.path.exists(filelist_path):
            with open(filelist_path, 'r') as f:
                self.filelist = [line.strip() for line in f.readlines() if line.strip()]
        else:
            self.filelist = []

        self.transform = get_official_transform(input_size=input_size, is_train=(split == 'train'))

    def __len__(self) -> int:
        return len(self.filelist)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        line = self.filelist[idx]
        parts = line.split()
        img_path = os.path.join(self.data_dir, parts[0])
        depth_path = os.path.join(self.data_dir, parts[1])

        raw_img = cv2.imread(img_path)
        if raw_img is None:
            raise FileNotFoundError(f"Failed to read image at: {img_path}")
        image = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB) / 255.0

        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        # Normalize depth if stored in millimeters (NYUv2 raw standard)
        if depth.max() > 100.0:
            depth = depth / 1000.0

        sample = self.transform({'image': image, 'depth': depth})
        image_tensor = torch.from_numpy(sample['image']).float()
        depth_tensor = torch.from_numpy(sample['depth']).float()
        valid_mask = (depth_tensor > 0.0) & (~torch.isnan(depth_tensor))

        return {
            'image': image_tensor,
            'depth': depth_tensor,
            'valid_mask': valid_mask,
            'image_path': img_path
        }


@register_dataset('kitti')
class KITTIDataset(Dataset):
    """
    KITTI Dataset Loader for Monocular Metric/Relative Depth Estimation.
    """
    def __init__(
        self, 
        filelist_path: str, 
        split: str = 'val', 
        input_size: int = 518,
        data_dir: Optional[str] = None
    ):
        self.split = split
        self.input_size = input_size
        self.data_dir = data_dir or ""

        if os.path.exists(filelist_path):
            with open(filelist_path, 'r') as f:
                self.filelist = [line.strip() for line in f.readlines() if line.strip()]
        else:
            self.filelist = []

        self.transform = get_official_transform(input_size=input_size, is_train=(split == 'train'))

    def __len__(self) -> int:
        return len(self.filelist)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        line = self.filelist[idx]
        parts = line.split()
        img_path = os.path.join(self.data_dir, parts[0])
        depth_path = os.path.join(self.data_dir, parts[1])

        raw_img = cv2.imread(img_path)
        if raw_img is None:
            raise FileNotFoundError(f"Failed to read image at: {img_path}")
        image = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB) / 255.0

        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth = depth / 256.0  # Convert KITTI raw depth uint16 to meters

        sample = self.transform({'image': image, 'depth': depth})
        image_tensor = torch.from_numpy(sample['image']).float()
        depth_tensor = torch.from_numpy(sample['depth']).float()
        valid_mask = (depth_tensor > 0.0) & (depth_tensor < 80.0) & (~torch.isnan(depth_tensor))

        return {
            'image': image_tensor,
            'depth': depth_tensor,
            'valid_mask': valid_mask,
            'image_path': img_path
        }


@register_dataset('dummy')
class DummyDataset(Dataset):
    """
    Synthetic Synthetic/Dummy Dataset for fast unit tests, benchmarking, and debugging.
    """
    def __init__(
        self, 
        length: int = 16, 
        input_size: Tuple[int, int] = (518, 518),
        seed: int = 42
    ):
        self.length = length
        self.input_size = input_size
        self.rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        h, w = self.input_size
        # Generate synthetic image CHW normalized
        image = torch.tensor(self.rng.randn(3, h, w), dtype=torch.float32)
        # Generate synthetic positive depth map HW
        depth = torch.tensor(self.rng.uniform(0.5, 10.0, size=(h, w)), dtype=torch.float32)
        valid_mask = torch.ones((h, w), dtype=torch.bool)

        return {
            'image': image,
            'depth': depth,
            'valid_mask': valid_mask,
            'image_path': f'dummy_sample_{idx:04d}.jpg'
        }
```

---

## 4. Caveats & Assumptions

1. **FeatureBundle Integration**: `RelativeDepthHead.forward` relies on the `FeatureBundle` dataclass defined in `depthlab/backbone/adapter.py` by `explorer_m3_1`. The contract assumes `feature_bundle.stages` contains 4 `FeatureStage` objects ordered from shallowest to deepest stage.
2. **Median Scale Alignment**: For relative depth metric computation (`compute_relative_depth_metrics`), median scaling (`scale = median(gt) / median(pred)`) is enabled by default (`align_scale=True`) to reflect standard monocular relative depth benchmark evaluation.
3. **Dataset Filelists**: Real NYUv2 / KITTI dataset loaders gracefully check file existence; if filelists are missing or dataset paths are not mounted, unit testing falls back to `get_dataset('dummy')`.

---

## 5. Conclusion & Recommendations

The Head Registry (`@register_head`), Abstract `BaseHead`, `RelativeDepthHead`, `SILogLoss`, Dataset Registry (`@register_dataset`), and Preprocessing (`OFFICIAL_TRANSFORM`) specifications are fully defined and ready for implementation.

### Implementation Checklist for Worker M3:
1. Create `depthlab/heads/base.py`, `depthlab/heads/__init__.py`, and `depthlab/heads/relative_depth.py`.
2. Create `depthlab/data/__init__.py`, `depthlab/data/transforms.py`, and `depthlab/data/datasets.py`.
3. Verify that `@register_head('relative_depth')` and `@register_dataset('dummy')` can be instantiated dynamically without errors.

---

## 6. Verification Method

To independently verify the implementation once written by Worker M3:

1. **Registry Verification**:
   ```python
   from depthlab.heads import list_heads, get_head
   from depthlab.data import list_datasets, get_dataset

   assert 'relative_depth' in list_heads()
   assert 'dummy' in list_datasets()
   ```

2. **Synthetic Integration Test**:
   - Instantiate `DummyDataset` and retrieve sample `batch`.
   - Instantiate `DA2Backbone` and pass `batch['image'].unsqueeze(0)` to obtain `FeatureBundle`.
   - Pass `FeatureBundle` through `RelativeDepthHead` instance.
   - Assert `preds['predicted_depth']` has shape `[1, 518, 518]`.
   - Call `compute_loss(preds, batch)` and assert loss is a positive scalar float tensor.
   - Call `compute_metrics(preds, batch)` and assert metrics dictionary contains `abs_rel`, `rmse`, `d1`, `d2`, `d3`.
