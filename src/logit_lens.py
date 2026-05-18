"""Logit-lens helpers for projecting hidden-state vectors to vocabulary logits."""

from __future__ import annotations

from typing import Any

import torch


def get_nested_attr(obj: Any, path: str) -> Any:
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def find_first_attr(obj: Any, paths: tuple[str, ...]) -> Any:
    for path in paths:
        try:
            return get_nested_attr(obj, path)
        except AttributeError:
            continue
    raise AttributeError(f"Could not find any of: {', '.join(paths)}")


def module_device(module: torch.nn.Module) -> torch.device:
    return next(module.parameters()).device


def module_dtype(module: torch.nn.Module) -> torch.dtype:
    return next(module.parameters()).dtype


def find_final_norm(model) -> torch.nn.Module:
    return find_first_attr(
        model,
        (
            "model.norm",
            "model.language_model.norm",
            "model.language_model.model.norm",
            "language_model.model.norm",
            "language_model.norm",
            "transformer.ln_f",
        ),
    )


def find_unembedding(model) -> torch.nn.Module:
    try:
        return find_first_attr(
            model,
            (
                "lm_head",
                "model.lm_head",
                "language_model.lm_head",
                "model.language_model.lm_head",
            ),
        )
    except AttributeError:
        output_embeddings = model.get_output_embeddings()
        if output_embeddings is None:
            raise
        return output_embeddings


@torch.inference_mode()
def project_vector_to_logits(
    *,
    model,
    vector: torch.Tensor,
    apply_final_norm: bool = True,
) -> torch.Tensor:
    unembedding = find_unembedding(model)
    unembed_weight = unembedding.weight
    hidden = vector.to(device=unembed_weight.device, dtype=unembed_weight.dtype)

    if apply_final_norm:
        final_norm = find_final_norm(model)
        hidden = hidden.to(device=module_device(final_norm), dtype=module_dtype(final_norm))
        hidden = final_norm(hidden.unsqueeze(0)).squeeze(0)
        hidden = hidden.to(device=unembed_weight.device, dtype=unembed_weight.dtype)

    return (unembed_weight @ hidden).float()


def top_tokens_from_logits(tokenizer, logits: torch.Tensor, top_k: int) -> list[dict[str, Any]]:
    probs = torch.softmax(logits, dim=-1)
    values, token_ids = torch.topk(logits, k=top_k)
    rows: list[dict[str, Any]] = []
    for rank, (token_id, logit) in enumerate(zip(token_ids.tolist(), values.tolist()), start=1):
        rows.append(
            {
                "rank": rank,
                "token_id": int(token_id),
                "token": tokenizer.decode([token_id]),
                "logit": float(logit),
                "prob": float(probs[token_id].item()),
            }
        )
    return rows
