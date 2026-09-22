"""Compute device detection shared across Von.

These helpers used to live inside the superseded cross-encoder backend, which
made every consumer import a benchmark baseline just to resolve a device. They
are model-agnostic, so they live on their own now.
"""

from __future__ import annotations

import os
from typing import Optional

import torch


def _detect_device(device_str: Optional[str] = None) -> torch.device:
    d_str = (device_str or os.environ.get("VON_DEVICE", "auto")).lower().strip()

    # Handle AMD ROCm / HIP aliases
    if d_str in ("rocm", "hip"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "AMD ROCm requested, but PyTorch CUDA/ROCm is not available. "
                "Ensure PyTorch was installed with ROCm support."
            )
        return torch.device("cuda")

    # Handle DirectML (Windows AMD / Intel)
    if d_str in ("dml", "directml"):
        try:
            import torch_directml  # type: ignore[import-not-found]  # optional extra
            return torch_directml.device()
        except ImportError:
            raise RuntimeError(
                "DirectML requested, but 'torch-directml' is not installed. "
                "Run 'pip install torch-directml'."
            )

    if d_str != "auto":
        return torch.device(d_str)

    # Auto-detection: First-party native accelerators only (CUDA/ROCm -> Apple Silicon MPS -> CPU).
    # DirectML on Windows is supported via explicit opt-in (--device dml) to avoid third-party driver clashing.
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_device_description(device: torch.device) -> str:
    if device.type == "cuda":
        dev_name = torch.cuda.get_device_name(device) if torch.cuda.is_available() else "CUDA"
        if getattr(torch.version, "hip", None) or any(w in dev_name.lower() for w in ["amd", "radeon", "instinct"]):
            return f"AMD GPU [ROCm: {dev_name}]"
        return f"NVIDIA GPU [CUDA: {dev_name}]"
    elif device.type == "mps":
        return "Apple Silicon [MPS]"
    elif str(device).startswith("privateuseone"):
        return "DirectML GPU [AMD/Intel]"
    return "CPU"
