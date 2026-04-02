from __future__ import annotations

import torch


def select_device(preference: str = "auto") -> torch.device:
    normalized = preference.lower()
    if normalized not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError(f"Unsupported device preference: {preference}")

    if normalized in {"auto", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda")
    if normalized in {"auto", "mps"} and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
