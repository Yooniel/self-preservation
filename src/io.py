"""Input/output helpers for JSON, JSONL, text lists, and prompt templates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def iter_json_or_jsonl(path: Path) -> Iterable[Any]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Input file is empty: {path}")

    if path.suffix == ".jsonl":
        for line_number, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_number} of {path}.") from exc
        return

    payload = json.loads(text)
    if not isinstance(payload, list):
        raise ValueError(f"Expected {path} to contain a JSON list.")
    yield from payload


def read_text_items(path: Path) -> list[str]:
    items: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        items.extend(part.strip() for part in stripped.split(",") if part.strip())
    if not items:
        raise ValueError(f"No non-empty values found in {path}.")
    return items


def load_string_list(path: Path, field_names: tuple[str, ...]) -> list[str]:
    if path.suffix.lower() not in {".json", ".jsonl"}:
        return read_text_items(path)

    values: list[str] = []
    for index, item in enumerate(iter_json_or_jsonl(path), start=1):
        if isinstance(item, str):
            value = item
        elif isinstance(item, dict):
            value = None
            for field_name in field_names:
                field_value = item.get(field_name)
                if isinstance(field_value, str):
                    value = field_value
                    break
            if value is None:
                raise ValueError(
                    f"Record {index} in {path} must contain one of: {', '.join(field_names)}."
                )
        else:
            raise TypeError(f"Record {index} in {path} must be a string or object.")

        value = value.strip()
        if value:
            values.append(value)

    if not values:
        raise ValueError(f"No non-empty values found in {path}.")
    return values


def load_template(path: Path | None, default_template: str | None = None) -> str:
    if path is None:
        if default_template is None:
            raise ValueError("A prompt template path is required.")
        return default_template
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, str):
            template = payload
        elif isinstance(payload, dict):
            template = (
                payload.get("template")
                or payload.get("prompt_template")
                or payload.get("prompt")
            )
            if not isinstance(template, str):
                raise ValueError(
                    f"{path} must contain a string or one of: template, prompt_template, prompt."
                )
        else:
            raise TypeError(f"{path} must contain a JSON string or object.")
    else:
        template = path.read_text(encoding="utf-8")
    template = template.strip()
    if not template:
        raise ValueError(f"Prompt template is empty: {path}")
    return template


def load_questions(path: Path) -> list[str]:
    questions: list[str] = []
    for index, item in enumerate(iter_json_or_jsonl(path), start=1):
        if isinstance(item, str):
            question = item
        elif isinstance(item, dict):
            question = (
                item.get("question")
                or item.get("prompt")
                or item.get("prompt_text")
                or item.get("text")
            )
            if question is None:
                raise ValueError(
                    f"Question record {index} must contain one of: "
                    "question, prompt, prompt_text, text."
                )
        else:
            raise TypeError(
                f"Question record {index} must be a string or object, "
                f"got {type(item).__name__}."
            )

        question = question.strip()
        if question:
            questions.append(question)

    if not questions:
        raise ValueError(f"No non-empty questions found in {path}.")
    return questions


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
