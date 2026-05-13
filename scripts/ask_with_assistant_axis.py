#!/usr/bin/env python3
"""
Ask JSON questions with assistant-axis steering.

This script assumes you already have an assistant axis saved as a PyTorch .pt
file. The axis can be either:
  - shape (hidden_dim,), reused for every selected layer
  - shape (num_layers, hidden_dim), indexed by the selected layer

During generation it adds scale * axis_vector to every token position at each
selected layer. The axis is used exactly as saved: no normalization is applied,
so --scale is relative to the provided vector.

Example:
    python scripts/ask_with_assistant_axis.py \
        --model-id google/gemma-3-27b-it \
        --questions-json questions.json \
        --assistant-axis assistant_axis.pt \
        --layers 31 \
        --scale 2.0 \
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
    parse_int_list,
    replace_hidden,
)
from src.io import load_questions
from src.modeling import format_chat_prompt, generate_answer, load_tokenizer_and_model
from src.tensors import extract_tensor_payload


def load_assistant_axis(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    axis = extract_tensor_payload(
        payload,
        path,
        keys=("axis", "assistant_axis", "direction", "vector", "steering_vector"),
    )
    if axis.ndim not in {1, 2}:
        raise ValueError(
            "Expected assistant axis with shape (hidden_dim,) or "
            f"(num_layers, hidden_dim), got {tuple(axis.shape)}."
        )
    return axis.to(dtype=torch.float32)


def axis_vector_for_layer(axis: torch.Tensor, layer_idx: int) -> torch.Tensor:
    if axis.ndim == 1:
        return axis
    if not 0 <= layer_idx < axis.shape[0]:
        raise ValueError(
            f"Layer {layer_idx} is outside assistant-axis rows [0, {axis.shape[0] - 1}]."
        )
    return axis[layer_idx]


def make_axis_steering_hook(vector: torch.Tensor, scale: float):
    delta = (scale * vector).to(dtype=torch.float32)

    def hook_fn(_module, _inputs, output):
        hidden = extract_hidden(output)
        delta_on_device = delta.to(device=hidden.device, dtype=hidden.dtype)
        return replace_hidden(output, hidden + delta_on_device)

    return hook_fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask questions while adding a cached assistant axis."
    )
    parser.add_argument("--model-id", required=True, help="Hugging Face model id.")
    parser.add_argument(
        "--assistant-axis",
        type=Path,
        required=True,
        help="Path to assistant axis .pt file.",
    )
    parser.add_argument(
        "--layers",
        required=True,
        help="Comma-separated transformer layer index or indices, e.g. 31 or 29,31.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Multiplier for the provided axis vector. No normalization is applied.",
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

    selected_layers = parse_int_list(args.layers, name="layers")
    questions = load_questions(args.questions_json)
    assistant_axis = load_assistant_axis(args.assistant_axis)

    tokenizer, model = load_tokenizer_and_model(
        args.model_id,
        dtype=args.dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )

    layers = get_transformer_layers(model)
    hidden_dim = assistant_axis.shape[-1]
    model_hidden_dim = getattr(model.config, "hidden_size", hidden_dim)
    if hidden_dim != model_hidden_dim:
        raise ValueError(
            f"Assistant axis hidden_dim={hidden_dim} does not match "
            f"model hidden_size={model_hidden_dim}."
        )

    handles = []
    for layer_idx in selected_layers:
        if not 0 <= layer_idx < len(layers):
            raise ValueError(f"Layer {layer_idx} is outside model layers [0, {len(layers) - 1}].")
        vector = axis_vector_for_layer(assistant_axis, layer_idx)
        hook = make_axis_steering_hook(vector, args.scale)
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
                    "assistant_axis": str(args.assistant_axis),
                    "assistant_axis_shape": list(assistant_axis.shape),
                    "layers": selected_layers,
                    "scale": args.scale,
                    "positions": "all",
                    "normalization": "none",
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
