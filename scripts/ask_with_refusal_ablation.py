#!/usr/bin/env python3
"""
Ask JSON questions with refusal-direction ablation.

This script assumes you already have a refusal direction tensor saved as a
PyTorch .pt file with shape (num_layers, hidden_dim). It loads a Hugging Face
causal language model, installs one forward hook per transformer layer covered
by the tensor, and subtracts the projection of each hidden state onto that
layer's refusal direction during generation. It uses the optimized 03d
Gemma-3-27B per-region ablation weights by default.

Examples:
    python scripts/ask_with_refusal_ablation.py \
        --model-id google/gemma-3-27b-it \
        --questions-json questions.json \
        --refusal-directions analysis/03d_refusal_abliteration/gemma3_27b/refusal_directions.pt \
        --output answers.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.activations import (
    extract_hidden,
    get_transformer_layers,
    replace_hidden,
)
from src.io import load_questions
from src.modeling import format_chat_prompt, generate_answer, load_tokenizer_and_model


DEFAULT_REGION_WEIGHTS = {
    "very_early_a": 0.010190365613071925,
    "very_early_b": 0.09976487098474057,
    "very_early_c": 0.009846349798252014,
    "very_early_d": 0.010714741304450688,
    "early_a": 0.023812035217103455,
    "early_b": 0.006873821994170306,
    "early_c": 0.0023568060724657135,
    "early_d": 0.11762696391562547,
    "pre_key_a": 0.024324361266584712,
    "pre_key_b": 0.009936585603088419,
    "key_a": 0.000533052460819306,
    "key_b": 0.0057508808893361974,
    "mid_a": 0.020646470409482434,
    "mid_b": 0.02205567035624907,
    "mid_c": 0.004716948598867072,
    "mid_d": 0.003251529189292551,
    "late_a": 0.07694211978232157,
    "late_b": 0.03330589279564281,
    "final_a": 2.358688691270255e-05,
    "final_b": 0.003955462234418926,
}


def region_weight_for_layer(layer_idx: int) -> float:
    """Return the optimized 03d refusal-ablation weight for a Gemma-3-27B layer."""
    if layer_idx <= 2:
        return DEFAULT_REGION_WEIGHTS["very_early_a"]
    if layer_idx <= 5:
        return DEFAULT_REGION_WEIGHTS["very_early_b"]
    if layer_idx <= 8:
        return DEFAULT_REGION_WEIGHTS["very_early_c"]
    if layer_idx <= 10:
        return DEFAULT_REGION_WEIGHTS["very_early_d"]
    if layer_idx <= 13:
        return DEFAULT_REGION_WEIGHTS["early_a"]
    if layer_idx <= 15:
        return DEFAULT_REGION_WEIGHTS["early_b"]
    if layer_idx <= 18:
        return DEFAULT_REGION_WEIGHTS["early_c"]
    if layer_idx <= 20:
        return DEFAULT_REGION_WEIGHTS["early_d"]
    if layer_idx <= 24:
        return DEFAULT_REGION_WEIGHTS["pre_key_a"]
    if layer_idx <= 28:
        return DEFAULT_REGION_WEIGHTS["pre_key_b"]
    if layer_idx <= 32:
        return DEFAULT_REGION_WEIGHTS["key_a"]
    if layer_idx <= 35:
        return DEFAULT_REGION_WEIGHTS["key_b"]
    if layer_idx <= 38:
        return DEFAULT_REGION_WEIGHTS["mid_a"]
    if layer_idx <= 41:
        return DEFAULT_REGION_WEIGHTS["mid_b"]
    if layer_idx <= 44:
        return DEFAULT_REGION_WEIGHTS["mid_c"]
    if layer_idx <= 47:
        return DEFAULT_REGION_WEIGHTS["mid_d"]
    if layer_idx <= 51:
        return DEFAULT_REGION_WEIGHTS["late_a"]
    if layer_idx <= 55:
        return DEFAULT_REGION_WEIGHTS["late_b"]
    if layer_idx <= 58:
        return DEFAULT_REGION_WEIGHTS["final_a"]
    return DEFAULT_REGION_WEIGHTS["final_b"]


def make_refusal_ablation_hook(direction: torch.Tensor, weight: float):
    direction = direction.to(dtype=torch.float32)

    def hook_fn(_module, _inputs, output):
        hidden = extract_hidden(output)
        direction_on_device = direction.to(device=hidden.device, dtype=hidden.dtype)
        projection = (
            torch.einsum("...d,d->...", hidden, direction_on_device).unsqueeze(-1)
            * direction_on_device
        )
        return replace_hidden(output, hidden - weight * projection)

    return hook_fn


def load_refusal_directions(path: Path) -> torch.Tensor:
    refusal_dirs = torch.load(path, map_location="cpu")
    if not isinstance(refusal_dirs, torch.Tensor):
        raise TypeError(f"Expected a tensor in {path}, got {type(refusal_dirs).__name__}.")
    if refusal_dirs.ndim != 2:
        raise ValueError(
            "Expected refusal directions with shape (num_layers, hidden_dim), "
            f"got {tuple(refusal_dirs.shape)}."
        )
    return refusal_dirs.to(dtype=torch.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask questions while ablating a cached refusal direction."
    )
    parser.add_argument("--model-id", required=True, help="Hugging Face model id.")
    parser.add_argument(
        "--refusal-directions",
        type=Path,
        required=True,
        help="Path to refusal_directions.pt with shape (num_layers, hidden_dim).",
    )
    parser.add_argument(
        "--questions-json",
        type=Path,
        required=True,
        help=(
            "JSON list of question strings, or objects with a question/prompt/"
            "prompt_text/text field."
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("answers.jsonl"))
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["auto", "float32", "float16", "bfloat16"],
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Passed to from_pretrained. Use 'none' to disable device_map.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be at least 1.")

    questions = load_questions(args.questions_json)
    refusal_dirs = load_refusal_directions(args.refusal_directions)

    tokenizer, model = load_tokenizer_and_model(
        args.model_id,
        dtype=args.dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )

    layers = get_transformer_layers(model)
    if refusal_dirs.shape[0] > len(layers):
        raise ValueError(
            f"Refusal directions cover {refusal_dirs.shape[0]} layers, but the "
            f"model exposes {len(layers)} layers."
        )

    handles = []
    ablation_layers = list(range(refusal_dirs.shape[0]))
    for layer_idx in ablation_layers:
        weight = region_weight_for_layer(layer_idx)
        hook = make_refusal_ablation_hook(refusal_dirs[layer_idx], weight)
        handles.append(layers[layer_idx].register_forward_hook(hook))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("w") as f:
            for index, question in enumerate(questions, start=1):
                prompt = format_chat_prompt(tokenizer, question)
                answer = generate_answer(
                    model,
                    tokenizer,
                    prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                row = {
                    "index": index,
                    "question": question,
                    "answer": answer,
                    "model_id": args.model_id,
                    "refusal_directions": str(args.refusal_directions),
                    "ablation_layers": ablation_layers,
                    "ablation_weight_source": "03d_region_weights",
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                print(f"{index}. {answer}\n")
    finally:
        for handle in handles:
            handle.remove()

    print(f"Saved {len(questions)} answer(s) to {args.output}")


if __name__ == "__main__":
    main()
