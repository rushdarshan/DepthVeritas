"""Presentation dashboard for DepthLab's local NYU experiment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from depthlab.backbone.loader import load_da2_checkpoint
from depthlab.data.transforms import normalize_image_tensor
from depthlab.heads import get_head


st.set_page_config(page_title="DepthLab | Depth Intelligence", page_icon="DL", layout="wide")


def _load_comparison() -> dict:
    path = ROOT / "runs" / "sef-end-to-end" / "da2_comparison.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


@st.cache_resource(show_spinner="Loading trained depth models...")
def load_models() -> tuple[torch.nn.Module, torch.nn.Module, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = load_da2_checkpoint("vits", ROOT / "checkpoints" / "depth_anything_v2_vits.pth", str(device)).eval()
    head = get_head("sef", encoder_variant="vits", n_bins=96, hidden_dim=256).to(device).eval()
    checkpoint = torch.load(ROOT / "runs" / "sef-end-to-end" / "checkpoints" / "checkpoint_best.pth", map_location=device)
    head.load_state_dict(checkpoint["head_state_dict"])
    return backbone, head, device


def prepare_image(image: Image.Image) -> tuple[Image.Image, torch.Tensor]:
    image = image.convert("RGB").resize((392, 392))
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0)
    return image, tensor


def colorize(depth: torch.Tensor) -> np.ndarray:
    values = depth.detach().float().cpu().numpy()
    values = (values - values.min()) / max(values.max() - values.min(), 1e-8)
    red = np.clip(1.5 - np.abs(4 * values - 3), 0, 1)
    green = np.clip(1.5 - np.abs(4 * values - 2), 0, 1)
    blue = np.clip(1.5 - np.abs(4 * values - 1), 0, 1)
    return (np.stack((red, green, blue), axis=-1) * 255).astype(np.uint8)


@torch.no_grad()
def infer(image: Image.Image) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    backbone, head, device = load_models()
    display, tensor = prepare_image(image)
    tensor = tensor.to(device)
    da2 = backbone(normalize_image_tensor(tensor))[0]
    sef = head(backbone.features(tensor))["depth"]
    sef = F.interpolate(sef.unsqueeze(1), size=(392, 392), mode="bilinear", align_corners=False)[0, 0]
    return display, colorize(da2), colorize(sef)


st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: Geist, sans-serif; }
.stApp { background: #101614; color: #eaf1eb; }
[data-testid="stSidebar"] { background: #17201c; border-right: 1px solid #2d3a34; }
.block-container { max-width: 1440px; padding-top: 2rem; padding-bottom: 4rem; }
.topline { color: #8bd6b3; font-family: 'DM Mono', monospace; font-size: .75rem; text-transform: uppercase; letter-spacing: .08em; }
.project-title { font-size: clamp(2.25rem, 4.5vw, 4.5rem); line-height: 1.04; letter-spacing: 0; max-width: 72rem; margin: .2rem 0 .8rem; }
.project-copy { color: #b8c6bd; max-width: 48rem; font-size: 1.08rem; line-height: 1.6; }
[data-testid="stMetric"] { background: #19231e; border: 1px solid #314238; border-radius: 7px; padding: 1rem; }
[data-testid="stMetricLabel"] { color: #a6b9ad; }
[data-testid="stMetricValue"] { color: #f3f7f4; }
.evidence { background: #19231e; border: 1px solid #314238; border-radius: 7px; padding: 1.2rem; min-height: 100%; }
.evidence h3 { margin: 0 0 .55rem; color: #f3f7f4; }
.evidence p { color: #b8c6bd; line-height: 1.55; margin: 0; }
.status-line { border-left: 3px solid #e3b260; padding: .7rem 1rem; background: #25251d; color: #f4e2bc; margin: 1rem 0 1.5rem; }
.stButton > button { background: #daf5e7; color: #102017; border: 0; border-radius: 5px; font-weight: 700; }
.stButton > button:hover { background: #ffffff; color: #102017; border: 0; }
</style>
""",
    unsafe_allow_html=True,
)

comparison = _load_comparison()
aligned_da2 = comparison.get("da2", {}).get("aligned", {})
aligned_sef = comparison.get("sef", {}).get("aligned", {})

with st.sidebar:
    st.markdown("## DepthLab")
    st.caption("Depth intelligence research dashboard")
    page = st.radio("View", ("Live comparison", "Experiment evidence", "How it works"), label_visibility="collapsed")
    st.divider()
    st.caption("NYU Depth V2 local experiment")
    st.caption("RTX 4050 6 GB | DA2-Small + SEF")

if page == "Live comparison":
    st.markdown('<div class="topline">DepthLab demonstration</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="project-title">Turn one image into a distance-aware scene.</h1>', unsafe_allow_html=True)
    st.markdown('<p class="project-copy">Upload an image to compare the original Depth Anything V2 output with the trained Surface Existence Field depth head.</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Choose an RGB image", type=("png", "jpg", "jpeg"))
    default_path = ROOT / "data" / "nyu_depth_v2" / "images" / "0000.png"
    source = Image.open(uploaded) if uploaded is not None else Image.open(default_path)
    if st.button("Run depth comparison", type="primary", use_container_width=True):
        with st.spinner("Generating depth maps on the local model..."):
            display, da2_map, sef_map = infer(source)
        original, baseline, enhanced = st.columns(3)
        original.image(display, caption="Input image", use_container_width=True)
        baseline.image(da2_map, caption="Original Depth Anything V2", use_container_width=True)
        enhanced.image(sef_map, caption="DepthLab SEF output", use_container_width=True)
        st.markdown('<div class="status-line">Color represents relative distance. Cooler colors are nearer; warmer colors are farther away. This is a live visual comparison, not a metric evaluation of the uploaded image.</div>', unsafe_allow_html=True)
    else:
        st.image(source, caption="Ready to run: use the provided NYU sample or upload your own image.", use_container_width=False, width=420)

elif page == "Experiment evidence":
    st.markdown('<div class="topline">Validated local result</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="project-title">SEF improved the local NYU validation result.</h1>', unsafe_allow_html=True)
    st.markdown('<p class="project-copy">Both models were evaluated on the same 290-image NYU validation split. Relative-depth predictions use standard per-image scale and shift alignment.</p>', unsafe_allow_html=True)
    metrics = st.columns(3)
    metrics[0].metric("AbsRel", f"{aligned_sef.get('abs_rel', 0):.4f}", f"{aligned_da2.get('abs_rel', 0) - aligned_sef.get('abs_rel', 0):.4f} lower")
    metrics[1].metric("RMSE", f"{aligned_sef.get('rmse', 0):.4f}", f"{aligned_da2.get('rmse', 0) - aligned_sef.get('rmse', 0):.4f} lower")
    metrics[2].metric("delta1", f"{aligned_sef.get('delta1', 0):.4f}", f"+{aligned_sef.get('delta1', 0) - aligned_da2.get('delta1', 0):.4f}")
    left, right = st.columns(2)
    with left:
        st.markdown('<div class="evidence"><h3>Original DA2</h3><p>AbsRel 0.2264<br>RMSE 0.7641<br>delta1 0.6448</p></div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="evidence"><h3>DA2 + SEF</h3><p>AbsRel 0.2069<br>RMSE 0.6863<br>delta1 0.6731</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="status-line">This is a local comparison, not a claim about the official NYUv2 benchmark or all real-world scenes.</div>', unsafe_allow_html=True)

else:
    st.markdown('<div class="topline">Research workflow</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="project-title">A repeatable path from depth data to measurable improvement.</h1>', unsafe_allow_html=True)
    columns = st.columns(3)
    columns[0].markdown('<div class="evidence"><h3>Prepare</h3><p>Convert RGB and ground-truth depth pairs into a validated local manifest. DepthLab checks assets before GPU work starts.</p></div>', unsafe_allow_html=True)
    columns[1].markdown('<div class="evidence"><h3>Train</h3><p>Warm up the SEF head, then fine-tune end to end. Every run records configuration, hardware, checkpoint, and result metadata.</p></div>', unsafe_allow_html=True)
    columns[2].markdown('<div class="evidence"><h3>Evaluate</h3><p>Compare depth error and accuracy against the original model on the same validation data, then preserve the result for review.</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="status-line">Completed locally: 1,449 NYU RGB-depth pairs, 20 warm-up epochs, 10 fine-tuning epochs, and a 290-image model comparison.</div>', unsafe_allow_html=True)
