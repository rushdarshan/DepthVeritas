"""Machine-readable provenance for measured, synthetic, and blocked research work."""

from __future__ import annotations

import hashlib
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import torch


EvidenceTier = Literal["measured", "synthetic", "blocked", "unavailable"]


@dataclass(frozen=True)
class Capability:
    name: str
    status: EvidenceTier
    reason: str
    required_paths: tuple[str, ...] = ()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_snapshot() -> dict[str, Any]:
    """Capture only portable, locally observable runtime facts."""
    snapshot: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        snapshot["cuda"] = {
            "device": properties.name,
            "total_memory_bytes": properties.total_memory,
            "capability": list(torch.cuda.get_device_capability(0)),
            "torch_cuda": torch.version.cuda,
        }
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version,temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
        if output:
            driver, temperature, power = [part.strip() for part in output.splitlines()[0].split(",")]
            snapshot["nvidia_smi"] = {"driver": driver, "temperature_c": float(temperature), "power_w": float(power)}
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        snapshot["nvidia_smi"] = None
    return snapshot


def capability_matrix(root: str | Path) -> list[Capability]:
    """Describe the current delivery boundary without treating absent inputs as passes."""
    project = Path(root)
    checks = [
        ("DA2 checkpoint", "measured", project / "checkpoints" / "depth_anything_v2_vits.pth", "Local DA2 checkpoint is available."),
        ("NYU RGB-D", "measured", project / "data" / "nyu_depth_v2", "Local NYU RGB-D data is available."),
        ("SEF checkpoint", "measured", project / "runs" / "sef-end-to-end" / "checkpoints" / "checkpoint_best.pth", "Local SEF checkpoint is available."),
        ("DINOv2 weights", "unavailable", project / "checkpoints" / "dinov2.pth", "No DINOv2 weights were supplied."),
        ("Trusted video pose and flow", "unavailable", project / "data" / "video_manifest.json", "No trusted video manifest with pose/flow was supplied."),
        ("Six-stratum reviewed benchmark", "blocked", project / "data" / "failure_benchmark" / "manifest.json", "Requires licensed, human-reviewed samples in every stratum."),
        ("Blender procedural renders", "unavailable", project / "data" / "procedural" / "manifest.json", "No rendered procedural dataset manifest is available."),
    ]
    results: list[Capability] = []
    for name, success_status, path, reason in checks:
        if path.exists() and success_status == "measured":
            results.append(Capability(name, "measured", reason, (str(path.relative_to(project)),)))
        elif path.exists():
            results.append(Capability(name, "synthetic", f"{reason} A manifest exists but needs independent review.", (str(path.relative_to(project)),)))
        else:
            results.append(Capability(name, success_status, reason, (str(path.relative_to(project)),)))
    return results


def write_evidence_bundle(
    path: str | Path, *, command: str, tier: EvidenceTier, inputs: Iterable[str | Path] = (),
    outputs: Iterable[str | Path] = (), capabilities: Iterable[Capability] = (), extra: dict[str, Any] | None = None,
) -> Path:
    """Write a self-contained evidence record; inputs must exist and are hashed."""
    input_records = []
    for input_path in inputs:
        source = Path(input_path)
        if not source.is_file():
            raise FileNotFoundError(f"Evidence input does not exist or is not a file: {source}")
        input_records.append({"path": str(source), "sha256": file_sha256(source), "bytes": source.stat().st_size})
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "tier": tier,
        "command": command,
        "environment": environment_snapshot(),
        "inputs": input_records,
        "outputs": [str(Path(output)) for output in outputs],
        "capabilities": [asdict(capability) for capability in capabilities],
        "extra": extra or {},
    }
    import json

    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return destination
