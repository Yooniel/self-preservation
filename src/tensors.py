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


def get_nested_payload(payload: Any, key: str) -> Any:
    value = payload
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(f"Could not find tensor key path: {key}")
        value = value[part]
    return value


def iter_tensor_paths(payload: Any, prefix: str = "") -> list[str]:
    if isinstance(payload, torch.Tensor):
        return [prefix] if prefix else ["<root>"]
    if not isinstance(payload, dict):
        return []

    paths: list[str] = []
    for key, value in payload.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        paths.extend(iter_tensor_paths(value, path))
    return paths


def load_tensor_by_key(path: Path, key: str | None = None) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    if key is not None:
        tensor = get_nested_payload(payload, key)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Payload key {key!r} in {path} is not a tensor.")
        return tensor

    if isinstance(payload, torch.Tensor):
        return payload

    tensor_paths = iter_tensor_paths(payload)
    if len(tensor_paths) == 1:
        return get_nested_payload(payload, tensor_paths[0])

    if not tensor_paths:
        raise TypeError(f"No tensors found in {path}.")
    raise ValueError(
        f"Multiple tensors found in {path}; pass --key. Available keys: "
        + ", ".join(tensor_paths)
    )
