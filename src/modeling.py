"""Model loading, prompt formatting, and generation helpers."""

from __future__ import annotations

import random
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


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


def resolve_quantization_config(
    quantization: str | None,
    dtype_name: str,
) -> BitsAndBytesConfig | None:
    if quantization is None:
        return None

    compute_dtype = resolve_dtype(dtype_name)
    if compute_dtype == "auto":
        compute_dtype = torch.bfloat16

    if quantization == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    if quantization == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    raise ValueError(f"Unsupported quantization mode: {quantization}")


def _get_text_config(config: Any) -> Any:
    """Return the nested text config used by multimodal text models like Gemma 3."""
    return getattr(config, "text_config", config)


def get_config_value(model_or_config: Any, *names: str) -> Any:
    """Look up config fields on both the top-level and nested text config."""
    config = getattr(model_or_config, "config", model_or_config)
    text_config = _get_text_config(config)
    for name in names:
        if hasattr(config, name):
            return getattr(config, name)
        if hasattr(text_config, name):
            return getattr(text_config, name)
    raise AttributeError(f"Could not find any of these config fields: {', '.join(names)}")


def get_model_hidden_size(model) -> int:
    return int(get_config_value(model, "hidden_size", "d_model", "dim", "n_embd"))


def _looks_like_gemma(model_id: str, model=None) -> bool:
    if "gemma" in model_id.lower():
        return True
    if model is None:
        return False
    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", "")
    text_model_type = getattr(_get_text_config(config), "model_type", "") if config is not None else ""
    return "gemma" in str(model_type).lower() or "gemma" in str(text_model_type).lower()


def _apply_gemma_rotary_patch(model_id: str, model=None) -> None:
    """Mirror the experiment-side Gemma rotary fix for older Transformers stacks."""
    if not _looks_like_gemma(model_id, model):
        return

    config = getattr(model, "config", None)
    model_type = str(getattr(config, "model_type", "")).lower()
    text_model_type = str(getattr(_get_text_config(config), "model_type", "")).lower() if config is not None else ""
    is_gemma3 = (
        "gemma-3" in model_id.lower()
        or "gemma3" in model_id.lower()
        or "gemma3" in model_type
        or "gemma3" in text_model_type
    )

    if is_gemma3:
        from transformers.models.gemma3 import modeling_gemma3 as gemma_module
    else:
        from transformers.models.gemma2 import modeling_gemma2 as gemma_module

    if getattr(gemma_module.apply_rotary_pos_emb, "_self_preservation_fixed", False):
        return

    def fixed_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        if cos.shape[-1] != q.shape[-1]:
            cos = cos[..., : q.shape[-1]]
            sin = sin[..., : q.shape[-1]]
        q_embed = (q * cos) + (gemma_module.rotate_half(q) * sin)
        k_embed = (k * cos) + (gemma_module.rotate_half(k) * sin)
        return q_embed, k_embed

    fixed_apply_rotary_pos_emb._self_preservation_fixed = True
    gemma_module.apply_rotary_pos_emb = fixed_apply_rotary_pos_emb


def load_tokenizer_and_model(
    model_id: str,
    *,
    dtype: str = "bfloat16",
    device_map: str = "auto",
    trust_remote_code: bool = False,
    quantization: str | None = None,
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
    }
    quantization_config = resolve_quantization_config(quantization, dtype)
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    else:
        model_kwargs["dtype"] = resolve_dtype(dtype)
    if device_map.lower() not in {"", "none", "null"}:
        model_kwargs["device_map"] = device_map

    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    except TypeError:
        if "dtype" in model_kwargs:
            model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

    model.eval()
    _apply_gemma_rotary_patch(model_id, model)
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


@torch.inference_mode()
def generate_answers_batch(
    model,
    tokenizer,
    prompts: list[str],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[str]:
    if not prompts:
        return []

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        return_token_type_ids=False,
    )
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
    answers: list[str] = []
    for row in output:
        new_tokens = row[input_len:]
        answers.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return answers
