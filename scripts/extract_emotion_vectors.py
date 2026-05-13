#!/usr/bin/env python3
"""
Generate emotion-conditioned stories and extract emotion vectors.

For extraction, input is a JSON list or JSONL file with records containing:
  - emotion: emotion label
  - story, text, or prompt_text: text to run through the model

For each story, the script captures transformer-layer hidden states, averages
token positions from --start-token onward, averages those activations by
emotion, then subtracts the cross-emotion mean. The result is one vector per
emotion per selected layer.

Examples:
    python scripts/extract_emotion_vectors.py generate \
        --model-id google/gemma-3-27b-it \
        --topics data/topics.txt \
        --emotions data/emotions.txt \
        --stories-json stories.jsonl

    python scripts/extract_emotion_vectors.py \
        extract \
        --model-id google/gemma-3-27b-it \
        --stories-json stories.jsonl \
        --layers all \
        --output emotion_vectors.pt

    python scripts/extract_emotion_vectors.py all \
        --model-id google/gemma-3-27b-it \
        --topics data/topics.txt \
        --emotions data/emotions.txt \
        --stories-json stories.jsonl \
        --output emotion_vectors.pt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.activations import (
    extract_layer_means_for_text,
    get_transformer_layers,
    parse_layer_spec,
)
from src.io import (
    iter_json_or_jsonl,
    load_template,
    read_text_items,
    write_jsonl,
)
from src.modeling import format_chat_prompt, generate_answer, load_tokenizer_and_model


def build_story_prompt(
    template: str,
    topic: str,
    emotion: str,
    story_index: int,
    n_stories: int,
) -> str:
    values = {
        "topic": topic,
        "emotion": emotion,
        "story_index": story_index,
        "n_stories": n_stories,
    }
    try:
        return template.format(**values)
    except KeyError as exc:
        raise KeyError(
            f"Unknown placeholder {{{exc.args[0]}}} in story template. "
            "Supported placeholders are {topic}, {emotion}, {story_index}, and {n_stories}."
        ) from exc


def split_generated_stories(text: str, expected_count: int) -> list[str]:
    marker_pattern = re.compile(r"(?im)^\s*\[\s*story\s+\d+\s*\]\s*$")
    marker_matches = list(marker_pattern.finditer(text))
    if marker_matches:
        stories = []
        for idx, match in enumerate(marker_matches):
            start = match.end()
            end = marker_matches[idx + 1].start() if idx + 1 < len(marker_matches) else len(text)
            story = text[start:end].strip()
            if story:
                stories.append(story)
        return stories

    numbered_pattern = re.compile(r"(?im)^\s*(?:story\s+)?\d+[\).\]:-]\s+")
    pieces = numbered_pattern.split(text)
    if len(pieces) > 1:
        stories = [piece.strip() for piece in pieces[1:] if piece.strip()]
        if stories:
            return stories

    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]
    if len(paragraphs) >= expected_count:
        return paragraphs[:expected_count]
    return [text.strip()] if text.strip() else []


def generate_stories(
    *,
    model,
    tokenizer,
    topics_path: Path,
    emotions_path: Path,
    template_path: Path,
    stories_path: Path,
    stories_per_pair: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    overwrite: bool,
) -> list[dict[str, Any]]:
    if stories_per_pair < 1:
        raise ValueError("--stories-per-pair must be at least 1.")
    if stories_path.exists() and not overwrite:
        print(f"Using existing stories at {stories_path}. Pass --overwrite to regenerate.")
        return load_stories(stories_path)

    topics = read_text_items(topics_path)
    emotions = read_text_items(emotions_path)
    template = load_template(template_path)

    records: list[dict[str, Any]] = []
    total = len(topics) * len(emotions)
    current = 0
    for topic_idx, topic in enumerate(topics):
        for emotion_idx, emotion in enumerate(emotions):
            current += 1
            user_prompt = build_story_prompt(
                template,
                topic,
                emotion,
                story_index=1,
                n_stories=stories_per_pair,
            )
            prompt = format_chat_prompt(tokenizer, user_prompt)
            print(f"[{current}/{total}] topic={topic!r} emotion={emotion!r}")
            output = generate_answer(
                model,
                tokenizer,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            stories = split_generated_stories(output, expected_count=stories_per_pair)
            if len(stories) != stories_per_pair:
                print(
                    f"Warning: parsed {len(stories)} stories for topic_idx={topic_idx}, "
                    f"emotion_idx={emotion_idx}; expected {stories_per_pair}."
                )
            for story_index, story in enumerate(stories[:stories_per_pair], start=1):
                records.append(
                    {
                        "topic": topic,
                        "topic_idx": topic_idx,
                        "emotion": emotion,
                        "emotion_idx": emotion_idx,
                        "story_index": story_index,
                        "prompt": user_prompt,
                        "story": story,
                        "raw_generation": output,
                    }
                )

    write_jsonl(stories_path, records)
    print(f"Saved {len(records)} generated stories to {stories_path}")
    return records


def load_stories(path: Path) -> list[dict[str, str]]:
    stories: list[dict[str, str]] = []
    for index, item in enumerate(iter_json_or_jsonl(path), start=1):
        if not isinstance(item, dict):
            raise TypeError(f"Story record {index} must be an object.")
        emotion = item.get("emotion")
        story = item.get("story") or item.get("text") or item.get("prompt_text")
        if not isinstance(emotion, str) or not emotion.strip():
            raise ValueError(f"Story record {index} is missing a non-empty emotion field.")
        if not isinstance(story, str) or not story.strip():
            raise ValueError(
                f"Story record {index} must contain non-empty story, text, or prompt_text."
            )
        stories.append({"emotion": emotion.strip(), "story": story.strip()})

    if not stories:
        raise ValueError(f"No stories found in {path}.")
    return stories


def compute_emotion_vectors(
    emotion_sums: dict[str, torch.Tensor],
    emotion_counts: dict[str, int],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    emotion_means = {
        emotion: emotion_sums[emotion] / emotion_counts[emotion]
        for emotion in sorted(emotion_sums)
        if emotion_counts[emotion] > 0
    }
    if len(emotion_means) < 2:
        raise ValueError("Need at least two emotions to compute emotion-minus-global vectors.")

    global_mean = torch.stack(list(emotion_means.values()), dim=0).mean(dim=0)
    emotion_vectors = {
        emotion: mean_activation - global_mean
        for emotion, mean_activation in emotion_means.items()
    }
    return emotion_vectors, emotion_means


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate emotion stories and extract per-layer emotion vectors."
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="extract",
        choices=["generate", "extract", "all"],
        help="Run story generation, vector extraction, or both. Defaults to extract.",
    )
    parser.add_argument("--model-id", required=True, help="Hugging Face model id.")
    parser.add_argument(
        "--topics",
        dest="topics_path",
        type=Path,
        default=Path("data/topics.txt"),
        help="Plain text topics file for generate/all.",
    )
    parser.add_argument(
        "--emotions",
        dest="emotions_path",
        type=Path,
        default=Path("data/emotions.txt"),
        help="Plain text emotions file for generate/all.",
    )
    parser.add_argument(
        "--prompt-template",
        type=Path,
        default=Path("data/prompts.txt"),
        help=(
            "Text prompt template with {topic}, {emotion}, "
            "{story_index}, and {n_stories} placeholders."
        ),
    )
    parser.add_argument(
        "--stories-json",
        type=Path,
        required=True,
        help="Generated or existing JSON/JSONL stories file.",
    )
    parser.add_argument("--output", type=Path, default=Path("emotion_vectors.pt"))
    parser.add_argument("--stories-per-pair", type=int, default=1)
    parser.add_argument("--generation-max-new-tokens", type=int, default=800)
    parser.add_argument("--generation-temperature", type=float, default=0.9)
    parser.add_argument("--generation-top-p", type=float, default=0.95)
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

    tokenizer, model = load_tokenizer_and_model(
        args.model_id,
        dtype=args.dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )

    if args.command in {"generate", "all"}:
        stories = generate_stories(
            model=model,
            tokenizer=tokenizer,
            topics_path=args.topics_path,
            emotions_path=args.emotions_path,
            template_path=args.prompt_template,
            stories_path=args.stories_json,
            stories_per_pair=args.stories_per_pair,
            max_new_tokens=args.generation_max_new_tokens,
            temperature=args.generation_temperature,
            top_p=args.generation_top_p,
            overwrite=args.overwrite,
        )
    else:
        stories = load_stories(args.stories_json)

    if args.command == "generate":
        return

    layers = get_transformer_layers(model)
    layer_indices = parse_layer_spec(args.layers, len(layers))

    emotion_sums: dict[str, torch.Tensor] = {}
    emotion_counts: dict[str, int] = defaultdict(int)

    for index, record in enumerate(stories, start=1):
        emotion = record["emotion"]
        print(f"[{index}/{len(stories)}] {emotion}")
        layer_means = extract_layer_means_for_text(
            model=model,
            tokenizer=tokenizer,
            text=record["story"],
            layer_indices=layer_indices,
            layers=layers,
            start_token=args.start_token,
            max_length=args.max_length,
        )
        if emotion not in emotion_sums:
            emotion_sums[emotion] = torch.zeros_like(layer_means)
        emotion_sums[emotion] += layer_means
        emotion_counts[emotion] += 1

    emotion_vectors, emotion_means = compute_emotion_vectors(emotion_sums, emotion_counts)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "emotion_vectors": emotion_vectors,
        "emotion_mean_activations": emotion_means,
        "emotion_counts": dict(emotion_counts),
        "selected_layers": layer_indices,
        "start_token": args.start_token,
        "model_id": args.model_id,
        "stories_json": str(args.stories_json),
    }
    torch.save(payload, args.output)

    metadata_path = args.output.with_suffix(".json")
    metadata = {
        "model_id": args.model_id,
        "stories_json": str(args.stories_json),
        "output": str(args.output),
        "selected_layers": layer_indices,
        "start_token": args.start_token,
        "max_length": args.max_length,
        "emotion_counts": dict(emotion_counts),
        "vector_shapes": {
            emotion: list(vector.shape)
            for emotion, vector in emotion_vectors.items()
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved emotion vectors to {args.output}")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
