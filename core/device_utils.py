"""Pick the best available torch device (CUDA > MPS > CPU)."""
from __future__ import annotations

import torch


def get_torch_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
