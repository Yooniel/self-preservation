"""Transformer-layer discovery and activation hook helpers."""

from __future__ import annotations

from typing import Any

import torch

from .modeling import first_model_device


def get_nested_attr(obj: Any, path: str) -> Any:
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def get_transformer_layers(model) -> list[torch.nn.Module]:
    """Find transformer blocks for common Hugging Face model layouts."""
    candidate_paths = [
        "model.layers",
        "language_model.model.layers",
        "transformer.h",
        "gpt_neox.layers",
    ]
    for path in candidate_paths:
        try:
            return list(get_nested_attr(model, path))
        except AttributeError:
            continue
    raise ValueError(
        "Could not locate transformer layers. Add your model's layer path to "
        "get_transformer_layers()."
    )


def extract_hidden(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def replace_hidden(output: Any, hidden: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (hidden,) + output[1:]
    return hidden


def parse_int_list(value: str, *, name: str = "value") -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise ValueError(f"--{name} must contain at least one integer.")
    return values


def parse_layer_spec(layer_spec: str, n_layers: int) -> list[int]:
    if layer_spec == "all":
        return list(range(n_layers))

    layers: list[int] = []
    for piece in layer_spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            start_text, end_text = piece.split("-", maxsplit=1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"Invalid descending layer range: {piece}")
            layers.extend(range(start, end + 1))
        else:
            layers.append(int(piece))

    deduped: list[int] = []
    seen = set()
    for layer_idx in layers:
        if not 0 <= layer_idx < n_layers:
            raise ValueError(f"Layer {layer_idx} is outside model layers [0, {n_layers - 1}].")
        if layer_idx not in seen:
            deduped.append(layer_idx)
            seen.add(layer_idx)
    if not deduped:
        raise ValueError("--layers did not select any layers.")
    return deduped


def extract_layer_means_for_text(
    *,
    model,
    tokenizer,
    text: str,
    layer_indices: list[int],
    layers: list[torch.nn.Module],
    start_token: int,
    max_length: int | None,
) -> torch.Tensor:
    if start_token < 1:
        raise ValueError("--start-token must be at least 1.")

    start_idx = start_token - 1
    layer_means: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer_idx: int):
        def hook_fn(_module, _inputs, output):
            hidden = extract_hidden(output)
            if hidden.shape[1] <= start_idx:
                raise ValueError(
                    f"Text has {hidden.shape[1]} tokens, fewer than --start-token {start_token}."
                )
            layer_means[layer_idx] = (
                hidden[:, start_idx:, :]
                .mean(dim=1)
                .squeeze(0)
                .detach()
                .to(device="cpu", dtype=torch.float32)
            )

        return hook_fn

    try:
        for layer_idx in layer_indices:
            handles.append(layers[layer_idx].register_forward_hook(make_hook(layer_idx)))

        tokenize_kwargs: dict[str, Any] = {
            "return_tensors": "pt",
            "return_token_type_ids": False,
            "add_special_tokens": False,
            "truncation": max_length is not None,
        }
        if max_length is not None:
            tokenize_kwargs["max_length"] = max_length

        inputs = tokenizer(text, **tokenize_kwargs)
        inputs = {key: value.to(first_model_device(model)) for key, value in inputs.items()}
        with torch.inference_mode():
            model(**inputs, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    missing = [layer_idx for layer_idx in layer_indices if layer_idx not in layer_means]
    if missing:
        raise RuntimeError(f"Did not capture activations for layers: {missing}")
    return torch.stack([layer_means[layer_idx] for layer_idx in layer_indices], dim=0)
