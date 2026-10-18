# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""Configuration for RecurrentQwen: Qwen3.5 converted to a RecurrentGemma-style
"local attention + recurrence" hybrid with bounded memory."""

from huggingface_hub.dataclasses import strict
from transformers.configuration_utils import SubConfigSpec
from transformers.models.qwen3_5.configuration_qwen3_5 import (
    Qwen3_5Config,
    Qwen3_5TextConfig,
    Qwen3_5VisionConfig,
)


@strict
class RecurrentQwenTextConfig(Qwen3_5TextConfig):
    r"""
    sliding_window (`int`, defaults to 2048):
        Every softmax-attention layer attends to at most this many tokens (itself included). Older context is only
        reachable through the Gated DeltaNet recurrent state, so the cache stops growing after `sliding_window` tokens.
    number_of_conv_states (`int`, defaults to 2):
        Cache slots per recurrent layer. Slot 0 holds the original conv/DeltaNet state, slot 1 holds the rolling key
        buffer used by the redundancy detector and the accumulated rotary phase.
    rq_redundancy_window (`int`, defaults to 128):
        How many previous keys the redundancy detector compares each token against.
    rq_rotary_pairs (`int`, defaults to 16):
        Number of (key, query) dimension pairs per head that get a data-dependent rotation (complex-valued state).
    rq_skip_redundancy_init / rq_skip_bias_init (`float`):
        Initial values of the skip gate `sigmoid(W x + u * r + c)` with `r = -log(1 - max cos sim)`. The defaults keep
        the gate near 0 on ordinary text (r ~ 2 -> skip ~ 0.001) and make exact repeats (r ~ 9.2) about half skipped.
    """

    model_type = "recurrent_qwen_text"

    sliding_window: int = 2048
    number_of_conv_states: int = 2
    rq_redundancy_window: int = 128
    rq_rotary_pairs: int = 16
    rq_skip_redundancy_init: float = 1.0
    rq_skip_bias_init: float = -9.0

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        # The global-attention layers of Qwen3.5 become local (sliding window) layers.
        self.layer_types = ["sliding_attention" if t == "full_attention" else t for t in self.layer_types]


@strict
class RecurrentQwenConfig(Qwen3_5Config):
    model_type = "recurrent_qwen"
    sub_configs_defaults = {
        "vision_config": SubConfigSpec(config_class=Qwen3_5VisionConfig),
        "text_config": SubConfigSpec(config_class=RecurrentQwenTextConfig),
    }


__all__ = ["RecurrentQwenConfig", "RecurrentQwenTextConfig"]
