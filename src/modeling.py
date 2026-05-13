"""Model loading, prompt formatting, and generation helpers."""

from __future__ import annotations

import random
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_dtype(dtype_name: str) -> torch.dtype | str:
    if dtype_name == "auto":
        return "auto"
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_name]


def load_tokenizer_and_model(
    model_id: str,
    *,
    dtype: str = "bfloat16",
    device_map: str = "auto",
    trust_remote_code: bool = False,
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "torch_dtype": resolve_dtype(dtype),
        "trust_remote_code": trust_remote_code,
    }
    if device_map.lower() not in {"", "none", "null"}:
        model_kwargs["device_map"] = device_map

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.eval()
    return tokenizer, model


def first_model_device(model) -> torch.device:
    return next(model.parameters()).device


def format_chat_prompt(tokenizer, user_text: str) -> str:
    messages = [{"role": "user", "content": user_text}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"User: {user_text}\nAssistant:"


@torch.inference_mode()
def generate_answer(
    model,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False)
    inputs = {key: value.to(first_model_device(model)) for key, value in inputs.items()}
    input_len = inputs["input_ids"].shape[-1]

    do_sample = temperature > 0.0
    generate_kwargs: dict[str, Any] = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p

    output = model.generate(**generate_kwargs)
    new_tokens = output[0, input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
