#!/usr/bin/env python3
"""Generate neutral dialogues and extract neutral activation vectors.

For extraction, input is a JSON list or JSONL file with records containing
`dialogue`, `text`, or `prompt_text`. The script captures transformer-layer
hidden states, averages token positions from --start-token onward, then averages
those activations across all neutral dialogues.

Examples:
    python scripts/extract_neutral_vectors.py generate \
        --model-id google/gemma-3-27b-it \
        --topics data/topics.txt \
        --prompt-template data/neutral_prompts.txt \
        --dialogues-json neutral_dialogues.jsonl

    python scripts/extract_neutral_vectors.py extract \
        --model-id google/gemma-3-27b-it \
        --dialogues-json neutral_dialogues.jsonl \
        --layers all \
        --output neutral_vectors.pt

    python scripts/extract_neutral_vectors.py all \
        --model-id google/gemma-3-27b-it \
        --topics data/topics.txt \
        --prompt-template data/neutral_prompts.txt \
        --dialogues-json neutral_dialogues.jsonl \
        --output neutral_vectors.pt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.activations import (  # noqa: E402
    extract_layer_means_for_text,
    extract_layer_means_for_texts,
    get_transformer_layers,
    parse_layer_spec,
)
from src.io import (  # noqa: E402
    iter_json_or_jsonl,
    load_string_list,
    load_template,
    write_jsonl,
)
from src.modeling import (  # noqa: E402
    format_chat_prompt,
    generate_answer,
    generate_answers_batch,
    load_tokenizer_and_model,
)


DEFAULT_NEUTRAL_TEMPLATE = (
    "Write {n_stories} different neutral Person/AI dialogues based on this topic:\n\n"
    "Topic: {topic}\n\n"
    "Format them as [dialogue 1], [dialogue 2], etc. Use only matter-of-fact "
    "language with no emotional content, no pleasantries, and no emotionally "
    "charged topics."
)


def build_dialogue_prompt(template: str, topic: str, n_stories: int) -> str:
    try:
        return template.format(topic=topic, n_stories=n_stories)
    except KeyError as exc:
        raise KeyError(
            f"Unknown placeholder {{{exc.args[0]}}} in neutral prompt template. "
            "Supported placeholders are {topic} and {n_stories}."
        ) from exc


def split_generated_dialogues(text: str, expected_count: int) -> list[str]:
    marker_pattern = re.compile(r"(?im)^\s*\[\s*dialogue\s+\d+\s*\]\s*$")
    marker_matches = list(marker_pattern.finditer(text))
    if marker_matches:
        dialogues = []
        for index, match in enumerate(marker_matches):
            start = match.end()
            end = marker_matches[index + 1].start() if index + 1 < len(marker_matches) else len(text)
            dialogue = text[start:end].strip()
            if dialogue:
                dialogues.append(dialogue)
        return dialogues

    numbered_pattern = re.compile(r"(?im)^\s*(?:dialogue\s+)?\d+[\).\]:-]\s+")
    pieces = numbered_pattern.split(text)
    if len(pieces) > 1:
        dialogues = [piece.strip() for piece in pieces[1:] if piece.strip()]
        if dialogues:
            return dialogues

    blocks = [block.strip() for block in re.split(r"\n\s*\n\s*\n", text) if block.strip()]
    if len(blocks) >= expected_count:
        return blocks[:expected_count]
    return [text.strip()] if text.strip() else []


def generate_dialogues(
    *,
    model,
    tokenizer,
    model_id: str,
    topics_path: Path,
    template_path: Path | None,
    dialogues_path: Path,
    dialogues_per_topic: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    batch_size: int,
    overwrite: bool,
) -> list[dict[str, str]]:
    if dialogues_per_topic < 1:
        raise ValueError("--dialogues-per-topic must be at least 1.")
    if batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if dialogues_path.exists() and not overwrite:
        print(f"Using existing dialogues at {dialogues_path}. Pass --overwrite to regenerate.")
        return load_dialogues(dialogues_path)

    topics = load_string_list(topics_path, ("topic", "text", "name"))
    template = load_template(template_path, DEFAULT_NEUTRAL_TEMPLATE)

    records: list[dict[str, Any]] = []
    tasks = [
        {
            "topic": topic,
            "topic_idx": topic_idx,
            "prompt": build_dialogue_prompt(template, topic, dialogues_per_topic),
        }
        for topic_idx, topic in enumerate(topics)
    ]

    for batch_start in range(0, len(tasks), batch_size):
        batch_tasks = tasks[batch_start : batch_start + batch_size]
        prompts = [
            format_chat_prompt(tokenizer, task["prompt"])
            for task in batch_tasks
        ]
        for offset, task in enumerate(batch_tasks, start=1):
            current = batch_start + offset
            print(f"[{current}/{len(tasks)}] topic={task['topic']!r}")

        if batch_size == 1:
            outputs = [
                generate_answer(
                    model,
                    tokenizer,
                    prompts[0],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
            ]
        else:
            outputs = generate_answers_batch(
                model,
                tokenizer,
                prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

        for task, output in zip(batch_tasks, outputs):
            dialogues = split_generated_dialogues(output, expected_count=dialogues_per_topic)
            if len(dialogues) != dialogues_per_topic:
                print(
                    f"Warning: parsed {len(dialogues)} dialogues for "
                    f"topic_idx={task['topic_idx']}; expected {dialogues_per_topic}."
                )

            for dialogue_index, dialogue in enumerate(dialogues[:dialogues_per_topic], start=1):
                records.append(
                    {
                        "topic": task["topic"],
                        "topic_idx": task["topic_idx"],
                        "dialogue_index": dialogue_index,
                        "prompt": task["prompt"],
                        "dialogue": dialogue,
                        "raw_generation": output,
                        "model_id": model_id,
                    }
                )

    write_jsonl(dialogues_path, records)
    print(f"Saved {len(records)} generated dialogues to {dialogues_path}")
    return load_dialogues(dialogues_path)


def load_dialogues(path: Path) -> list[dict[str, str]]:
    dialogues: list[dict[str, str]] = []
    for index, item in enumerate(iter_json_or_jsonl(path), start=1):
        if not isinstance(item, dict):
            raise TypeError(f"Dialogue record {index} must be an object.")
        text = item.get("dialogue") or item.get("text") or item.get("prompt_text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"Dialogue record {index} must contain non-empty dialogue, text, or prompt_text."
            )
        dialogues.append({"dialogue": text.strip()})

    if not dialogues:
        raise ValueError(f"No dialogues found in {path}.")
    return dialogues


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate neutral dialogues and extract neutral activation vectors."
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="extract",
        choices=["generate", "extract", "all"],
        help="Run dialogue generation, vector extraction, or both. Defaults to extract.",
    )
    parser.add_argument("--model-id", required=True, help="Hugging Face model id.")
    parser.add_argument(
        "--topics",
        dest="topics_path",
        type=Path,
        default=None,
        help="Topics for generate/all. Supports .txt, JSON, or JSONL.",
    )
    parser.add_argument(
        "--prompt-template",
        type=Path,
        default=None,
        help="Optional text or JSON prompt template with {topic} and {n_stories} placeholders.",
    )
    parser.add_argument(
        "--dialogues-json",
        type=Path,
        required=True,
        help="Generated or existing JSON/JSONL neutral dialogues file.",
    )
    parser.add_argument("--output", type=Path, default=Path("neutral_vectors.pt"))
    parser.add_argument("--dialogues-per-topic", type=int, default=3)
    parser.add_argument("--generation-max-new-tokens", type=int, default=2048)
    parser.add_argument("--generation-temperature", type=float, default=0.8)
    parser.add_argument("--generation-top-p", type=float, default=0.95)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of prompts/dialogues to process in each model call.",
    )
    parser.add_argument(
        "--layers",
        default="all",
        help="Layers to extract: all, comma-separated indices, or ranges like 20-35.",
    )
    parser.add_argument(
        "--start-token",
        type=int,
        default=50,
        help="One-indexed token number where averaging starts.",
    )
    parser.add_argument("--max-length", type=int, default=None)
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
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.start_token < 1:
        raise ValueError("--start-token must be at least 1.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")

    if args.command in {"generate", "all"} and args.topics_path is None:
        raise ValueError("--topics is required for generate/all.")

    tokenizer, model = load_tokenizer_and_model(
        args.model_id,
        dtype=args.dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )

    if args.command in {"generate", "all"}:
        dialogues = generate_dialogues(
            model=model,
            tokenizer=tokenizer,
            model_id=args.model_id,
            topics_path=args.topics_path,
            template_path=args.prompt_template,
            dialogues_path=args.dialogues_json,
            dialogues_per_topic=args.dialogues_per_topic,
            max_new_tokens=args.generation_max_new_tokens,
            temperature=args.generation_temperature,
            top_p=args.generation_top_p,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
        )
    else:
        dialogues = load_dialogues(args.dialogues_json)

    if args.command == "generate":
        return

    layers = get_transformer_layers(model)
    layer_indices = parse_layer_spec(args.layers, len(layers))

    neutral_sum: torch.Tensor | None = None
    for batch_start in range(0, len(dialogues), args.batch_size):
        batch_records = dialogues[batch_start : batch_start + args.batch_size]
        for offset, _record in enumerate(batch_records, start=1):
            index = batch_start + offset
            print(f"[{index}/{len(dialogues)}] neutral dialogue")

        if args.batch_size == 1:
            batch_layer_means = extract_layer_means_for_text(
                model=model,
                tokenizer=tokenizer,
                text=batch_records[0]["dialogue"],
                layer_indices=layer_indices,
                layers=layers,
                start_token=args.start_token,
                max_length=args.max_length,
            ).unsqueeze(0)
        else:
            batch_layer_means = extract_layer_means_for_texts(
                model=model,
                tokenizer=tokenizer,
                texts=[record["dialogue"] for record in batch_records],
                layer_indices=layer_indices,
                layers=layers,
                start_token=args.start_token,
                max_length=args.max_length,
            )

        batch_sum = batch_layer_means.sum(dim=0)
        neutral_sum = batch_sum if neutral_sum is None else neutral_sum + batch_sum

    if neutral_sum is None:
        raise ValueError("No dialogues were available for extraction.")
    neutral_vector = neutral_sum / len(dialogues)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "neutral_vector": neutral_vector,
        "neutral_count": len(dialogues),
        "selected_layers": layer_indices,
        "start_token": args.start_token,
        "batch_size": args.batch_size,
        "model_id": args.model_id,
        "dialogues_json": str(args.dialogues_json),
    }
    torch.save(payload, args.output)

    metadata_path = args.output.with_suffix(".json")
    metadata = {
        "model_id": args.model_id,
        "dialogues_json": str(args.dialogues_json),
        "output": str(args.output),
        "selected_layers": layer_indices,
        "start_token": args.start_token,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "neutral_count": len(dialogues),
        "vector_shape": list(neutral_vector.shape),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved neutral vector to {args.output}")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
