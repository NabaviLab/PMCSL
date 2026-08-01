from __future__ import annotations

import torch


def resolve_device(requested: str | None = None) -> torch.device:
    """
    Resolve the requested compute device.

    Examples:
        "cuda"
        "cuda:0"
        "cpu"
        None
    """
    if requested is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    requested = requested.lower()

    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA was requested but is unavailable; using CPU.")
        return torch.device("cpu")

    return torch.device(requested)


def describe_device(device: torch.device) -> dict:
    """Return concise hardware information for logging."""
    info = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
    }

    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        info.update(
            {
                "gpu_name": torch.cuda.get_device_name(index),
                "gpu_count": torch.cuda.device_count(),
                "cuda_version": torch.version.cuda,
            }
        )

    return info
