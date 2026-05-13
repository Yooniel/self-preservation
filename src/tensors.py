"""Helpers for loading tensor payloads from .pt files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def extract_tensor_payload(payload: Any, path: Path, keys: tuple[str, ...]) -> torch.Tensor:
    if isinstance(payload, torch.Tensor):
        return payload
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a tensor or tensor dict in {path}.")

    for key in keys:
        value = payload.get(key)
        if isinstance(value, torch.Tensor):
            return value

    tensor_items = [value for value in payload.values() if isinstance(value, torch.Tensor)]
    if len(tensor_items) == 1:
        return tensor_items[0]
    if not tensor_items:
        raise TypeError(f"No tensor found in tensor dict at {path}.")
    raise ValueError(f"Multiple tensors found in tensor dict at {path}; expected one of {keys}.")
