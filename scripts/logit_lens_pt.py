#!/usr/bin/env python3
"""Run a logit lens on a hidden-state vector saved in a .pt file.

Examples:
    python scripts/logit_lens_pt.py emotion_vectors.pt \
        --model-id google/gemma-3-27b-it \
        --key emotion_vectors.self-preservation \
        --layer 31

    python scripts/logit_lens_pt.py data/assistant_axis.pt \
        --model-id google/gemma-3-27b-it \
        --layer 31 \
        --top-k 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.logit_lens import project_vector_to_logits, top_tokens_from_logits
from src.modeling import load_tokenizer_and_model
from src.tensors import iter_tensor_paths, load_tensor_by_key


def select_vector(tensor: torch.Tensor, layer: int | None) -> torch.Tensor:
    if tensor.ndim == 1:
        if layer is not None:
            print("--layer ignored because selected tensor is already 1D.", file=sys.stderr)
        return tensor

    if tensor.ndim == 2:
        if layer is None:
            raise ValueError(f"Selected tensor has shape {tuple(tensor.shape)}; pass --layer.")
        if not 0 <= layer < tensor.shape[0]:
            raise ValueError(f"--layer {layer} is outside tensor rows [0, {tensor.shape[0] - 1}].")
        return tensor[layer]

    raise ValueError(f"Expected selected tensor to be 1D or 2D, got shape {tuple(tensor.shape)}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project a saved vector through a model logit lens.")
    parser.add_argument("pt_file", type=Path, help="Path to a .pt tensor or tensor payload.")
    parser.add_argument(
        "--model-id",
        help="Hugging Face model id. Required unless --list-keys is set.",
    )
    parser.add_argument(
        "--key",
        "--tensor-key",
        dest="key",
        help=(
            "Tensor key path inside a .pt dict, e.g. "
            "emotion_vectors.self-preservation or emotion_mean_activations.happy."
        ),
    )
    parser.add_argument("--layer", type=int, help="Row to use if the selected tensor is 2D.")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
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
    parser.add_argument("--no-final-norm", action="store_true")
    parser.add_argument(
        "--list-keys",
        action="store_true",
        help="List tensor key paths in pt_file and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1.")

    if args.list_keys:
        payload = torch.load(args.pt_file, map_location="cpu")
        for key in iter_tensor_paths(payload):
            print(key)
        return
    if args.model_id is None:
        raise ValueError("--model-id is required unless --list-keys is set.")

    tensor = load_tensor_by_key(args.pt_file, args.key)
    vector = select_vector(tensor, args.layer).to(dtype=torch.float32)

    tokenizer, model = load_tokenizer_and_model(
        args.model_id,
        dtype=args.dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )

    logits = project_vector_to_logits(
        model=model,
        vector=vector,
        apply_final_norm=not args.no_final_norm,
    )
    top_tokens = top_tokens_from_logits(tokenizer, logits, args.top_k)

    result: dict[str, Any] = {
        "model_id": args.model_id,
        "pt_file": str(args.pt_file),
        "key": args.key,
        "layer": args.layer,
        "selected_tensor_shape": list(tensor.shape),
        "vector_norm": float(vector.norm().item()),
        "final_norm": not args.no_final_norm,
        "top_tokens": top_tokens,
    }

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    print(f"model_id: {args.model_id}")
    print(f"pt_file: {args.pt_file}")
    if args.key is not None:
        print(f"key: {args.key}")
    if args.layer is not None:
        print(f"layer: {args.layer}")
    print(f"vector_norm: {result['vector_norm']:.6f}")
    print(f"final_norm: {result['final_norm']}")
    print()
    print("rank\ttoken_id\tlogit\tprob\ttoken")
    for row in top_tokens:
        token = row["token"].replace("\n", "\\n")
        print(
            f"{row['rank']}\t{row['token_id']}\t{row['logit']:.6f}\t"
            f"{row['prob']:.6e}\t{token!r}"
        )


if __name__ == "__main__":
    main()
