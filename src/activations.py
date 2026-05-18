"""Transformer-layer discovery and activation hook helpers."""

from __future__ import annotations

from typing import Any

import torch

from .modeling import first_model_device


def get_nested_attr(obj: Any, path: str) -> Any:
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def _unwrap_peft_model(model):
    """Return the underlying HF model when PEFT wraps it."""
    return getattr(getattr(model, "base_model", None), "model", model)


def get_transformer_layers(model) -> list[torch.nn.Module]:
    """Find transformer blocks, with Gemma 3's text stack handled first."""
    model = _unwrap_peft_model(model)

    if hasattr(model, "model"):
        backbone = model.model
        if hasattr(backbone, "language_model") and hasattr(backbone.language_model, "layers"):
            return list(backbone.language_model.layers)
        if hasattr(backbone, "layers"):
            return list(backbone.layers)

    if hasattr(model, "language_model") and hasattr(model.language_model, "model"):
        language_backbone = model.language_model.model
        if hasattr(language_backbone, "layers"):
            return list(language_backbone.layers)

    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return list(model.gpt_neox.layers)

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


def extract_layer_means_for_texts(
    *,
    model,
    tokenizer,
    texts: list[str],
    layer_indices: list[int],
    layers: list[torch.nn.Module],
    start_token: int,
    max_length: int | None,
) -> torch.Tensor:
    if start_token < 1:
        raise ValueError("--start-token must be at least 1.")
    if not texts:
        return torch.empty((0, len(layer_indices), 0), dtype=torch.float32)

    layer_means: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer_idx: int):
        def hook_fn(_module, _inputs, output):
            hidden = extract_hidden(output)
            attention_mask = hook_fn.attention_mask.to(device=hidden.device)
            token_numbers = attention_mask.cumsum(dim=1)
            mean_mask = attention_mask.bool() & (token_numbers >= start_token)
            mask = mean_mask.unsqueeze(-1).to(dtype=hidden.dtype)
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp_min(1)
            layer_means[layer_idx] = (
                (summed / counts)
                .detach()
                .to(device="cpu", dtype=torch.float32)
            )

        return hook_fn

    try:
        hooks_by_layer = {}
        for layer_idx in layer_indices:
            hook = make_hook(layer_idx)
            hooks_by_layer[layer_idx] = hook
            handles.append(layers[layer_idx].register_forward_hook(hook))

        tokenize_kwargs: dict[str, Any] = {
            "return_tensors": "pt",
            "return_token_type_ids": False,
            "add_special_tokens": False,
            "padding": True,
            "truncation": max_length is not None,
        }
        if max_length is not None:
            tokenize_kwargs["max_length"] = max_length

        inputs = tokenizer(texts, **tokenize_kwargs)
        attention_mask = inputs["attention_mask"]
        token_counts = attention_mask.sum(dim=1)
        if torch.any(token_counts < start_token):
            raise ValueError(
                f"At least one text has fewer than --start-token {start_token} "
                f"tokens after tokenization: {token_counts.tolist()}."
            )

        for hook in hooks_by_layer.values():
            hook.attention_mask = attention_mask

        inputs = {key: value.to(first_model_device(model)) for key, value in inputs.items()}
        with torch.inference_mode():
            model(**inputs, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    missing = [layer_idx for layer_idx in layer_indices if layer_idx not in layer_means]
    if missing:
        raise RuntimeError(f"Did not capture activations for layers: {missing}")

    return torch.stack([layer_means[layer_idx] for layer_idx in layer_indices], dim=1)
