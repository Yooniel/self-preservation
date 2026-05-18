#!/usr/bin/env python3
"""
LoRA SFT finetuning on a provided JSON/JSONL dataset.

This is a minimal standalone adaptation of the SFT path in
experiments/03f_dpo_mechanism_ablation.py. It trains with standard
cross-entropy on assistant response tokens only: prompt tokens are masked with
-100 in the labels.

Supported dataset formats:
  JSONL or JSON list of {"formatted_prompt": "...", "output": "..."}

Example:
    python scripts/finetune_sft_lora.py \
        --model-id google/gemma-3-27b-it \
        --dataset sft_data.jsonl \
        --output-dir adapters/sft_external \
        --epochs 1
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling import (
    first_model_device,
    load_tokenizer_and_model,
    set_seed,
)


def iter_json_or_jsonl(path: Path) -> Iterable[Any]:
    text = path.read_text().strip()
    if not text:
        raise ValueError(f"Dataset is empty: {path}")

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
    if isinstance(payload, list):
        yield from payload
        return
    raise ValueError(
        f"Expected {path} to be a JSON list or JSONL file."
    )


def normalize_example(record: dict[str, Any]) -> dict[str, Any]:
    if "formatted_prompt" in record and "output" in record:
        prompt = str(record["formatted_prompt"]).strip()
        response = str(record["output"]).strip()
        if prompt and response:
            return {"formatted_prompt": prompt, "response": response}

    raise ValueError('Expected record with non-empty "formatted_prompt" and "output" fields.')


def load_dataset(path: Path, val_fraction: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload = list(iter_json_or_jsonl(path))
    examples = [normalize_example(record) for record in payload]
    rng = random.Random(seed)
    rng.shuffle(examples)
    split = int(len(examples) * (1.0 - val_fraction))
    split = min(max(split, 1), len(examples) - 1)
    train = examples[:split]
    val = examples[split:]

    if not train:
        raise ValueError("No training examples found.")
    if not val:
        raise ValueError("No validation examples found. Provide test data or raise --val-fraction.")
    return train, val


def format_prompt(tokenizer, example: dict[str, Any]) -> str:
    return example["formatted_prompt"]


class SFTDataset(Dataset):
    def __init__(self, examples: list[dict[str, Any]], tokenizer, max_length: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        example = self.examples[idx]
        prompt = format_prompt(self.tokenizer, example)
        response = example["response"]
        eos = self.tokenizer.eos_token or ""

        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]
        full_ids = self.tokenizer(
            prompt + response + eos,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]

        if len(full_ids) <= len(prompt_ids):
            raise ValueError(
                "Tokenized example has no response tokens after truncation. "
                "Increase --max-seq-len or shorten prompts."
            )

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[: len(prompt_ids)] = -100
        attention_mask = torch.ones_like(input_ids)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


def make_collate_fn(tokenizer):
    def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        max_len = max(item["input_ids"].shape[0] for item in batch)
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for row_idx, item in enumerate(batch):
            seq_len = item["input_ids"].shape[0]
            input_ids[row_idx, :seq_len] = item["input_ids"]
            labels[row_idx, :seq_len] = item["labels"]
            attention_mask[row_idx, :seq_len] = item["attention_mask"]
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    return collate_fn


def evaluate_loss(model, loader: DataLoader, max_batches: int) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    device = first_model_device(model)
    with torch.inference_mode():
        for batch in loader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                labels=batch["labels"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            total_loss += float(outputs.loss.item())
            n_batches += 1
            if n_batches >= max_batches:
                break
    model.train()
    return total_loss / max(n_batches, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal LoRA SFT finetuning.")
    parser.add_argument("--model-id", required=True, help="Hugging Face model id.")
    parser.add_argument("--dataset", type=Path, required=True, help="SFT JSON or JSONL dataset.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save LoRA adapter.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max-val-batches", type=int, default=500)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
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
    parser.add_argument(
        "--quantization",
        choices=["8bit", "4bit"],
        default=None,
        help="Optional BitsAndBytes quantization for loading the base model.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and (args.output_dir / "adapter_config.json").exists() and not args.overwrite:
        raise FileExistsError(f"Adapter already exists at {args.output_dir}. Use --overwrite.")
    if args.gradient_accumulation < 1:
        raise ValueError("--gradient-accumulation must be at least 1.")
    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("--val-fraction must be between 0 and 1.")

    set_seed(args.seed)
    train_examples, val_examples = load_dataset(args.dataset, args.val_fraction, args.seed)
    print(f"Train examples: {len(train_examples)}")
    print(f"Validation examples: {len(val_examples)}")

    tokenizer, model = load_tokenizer_and_model(
        args.model_id,
        dtype=args.dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
        quantization=args.quantization,
    )

    from peft import LoraConfig, TaskType, get_peft_model

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules="all-linear",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if hasattr(model, "gradient_checkpointing_enable"):
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    collate_fn = make_collate_fn(tokenizer)
    train_loader = DataLoader(
        SFTDataset(train_examples, tokenizer, args.max_seq_len),
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_batch_size = min(8, max(1, len(val_examples)))
    val_loader = DataLoader(
        SFTDataset(val_examples, tokenizer, args.max_seq_len),
        batch_size=val_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    total_steps = max(1, (len(train_loader) * args.epochs) // args.gradient_accumulation)
    warmup_steps = min(50, total_steps // 10)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = args.output_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    best_val_loss = evaluate_loss(model, val_loader, args.max_val_batches)
    best_step = 0
    patience_counter = 0
    train_loss_history: list[tuple[int, float]] = []
    val_loss_history: list[tuple[int, float]] = [(0, best_val_loss)]
    running_loss = 0.0
    global_step = 0
    stopped_early = False

    print(f"Initial validation loss: {best_val_loss:.4f}")
    model.train()
    optimizer.zero_grad()

    for epoch in range(args.epochs):
        for batch_idx, batch in enumerate(train_loader):
            device = first_model_device(model)
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                labels=batch["labels"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            loss = outputs.loss / args.gradient_accumulation
            loss.backward()
            running_loss += float(loss.item())

            if (batch_idx + 1) % args.gradient_accumulation != 0:
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % args.eval_every != 0 and global_step != total_steps:
                continue

            avg_train_loss = running_loss / max(1, args.eval_every)
            running_loss = 0.0
            val_loss = evaluate_loss(model, val_loader, args.max_val_batches)
            train_loss_history.append((global_step, avg_train_loss))
            val_loss_history.append((global_step, val_loss))

            improved = val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss
                best_step = global_step
                patience_counter = 0
                model.save_pretrained(args.output_dir)
                tokenizer.save_pretrained(args.output_dir)
            else:
                patience_counter += 1

            marker = " *" if improved else f" (patience {patience_counter}/{args.patience})"
            print(
                f"Step {global_step}/{total_steps}, epoch {epoch + 1}/{args.epochs}, "
                f"train {avg_train_loss:.4f}, val {val_loss:.4f}{marker}"
            )

            if patience_counter >= args.patience:
                stopped_early = True
                break

            model.train()

        if stopped_early:
            break

    if best_step == 0:
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

    training_summary = {
        "model_id": args.model_id,
        "dataset": str(args.dataset),
        "train_examples": len(train_examples),
        "validation_examples": len(val_examples),
        "epochs": args.epochs,
        "lr": args.lr,
        "train_batch_size": args.train_batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "max_seq_len": args.max_seq_len,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "quantization": args.quantization,
        "best_val_loss": best_val_loss,
        "best_step": best_step,
        "total_steps": total_steps,
        "stopped_early": stopped_early,
        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history,
    }
    (args.output_dir / "training_config.json").write_text(
        json.dumps(training_summary, indent=2) + "\n"
    )
    print(f"Saved LoRA adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
