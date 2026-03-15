"""Torch device selection and configuration helpers."""

from __future__ import annotations

import torch


def get_preferred_device() -> str:
    """Pick the best available torch device."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def configure_torch_device() -> str:
    """
    Configure the process-wide default torch device.

    Returns the selected device string.
    """
    selected = get_preferred_device()
    _ = torch.set_default_device(selected)
    return selected
