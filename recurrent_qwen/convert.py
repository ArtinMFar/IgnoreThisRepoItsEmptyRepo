"""Build a RecurrentQwen model from a stock Qwen3.5 checkpoint."""

import torch
from transformers import AutoConfig

from .configuration_recurrent_qwen import RecurrentQwenConfig
from .modeling_recurrent_qwen import RecurrentQwenForConditionalGeneration

NEW_PARAM_NAMES = ("skip_proj", "skip_redundancy", "skip_bias", "phase_proj")


def recurrent_config_from_base(base: str, **text_overrides) -> RecurrentQwenConfig:
    raw = AutoConfig.from_pretrained(base).to_dict()
    raw.pop("model_type", None)
    raw.pop("architectures", None)
    raw.pop("transformers_version", None)
    text = dict(raw["text_config"])
    text.pop("model_type", None)
    text.update(text_overrides)
    raw["text_config"] = text
    cfg = RecurrentQwenConfig(**raw)
    cfg.architectures = ["RecurrentQwenForConditionalGeneration"]
    return cfg


def load_from_base(base: str, dtype=torch.bfloat16, attn_implementation="sdpa", **text_overrides):
    cfg = recurrent_config_from_base(base, **text_overrides)
    model, info = RecurrentQwenForConditionalGeneration.from_pretrained(
        base, config=cfg, dtype=dtype, attn_implementation=attn_implementation, output_loading_info=True
    )
    unexpected_new = [k for k in info["missing_keys"] if not any(n in k for n in NEW_PARAM_NAMES)]
    if unexpected_new:
        raise RuntimeError(f"pretrained weights missing for: {unexpected_new[:10]}")
    return model
