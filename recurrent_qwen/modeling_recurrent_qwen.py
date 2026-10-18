# Copyright 2026. Licensed under the Apache License, Version 2.0.
# Built on top of the Qwen3.5 implementation in `transformers` (Apache 2.0, The Qwen Team / HuggingFace).
"""RecurrentQwen: Qwen3.5 with RecurrentGemma-style bounded memory.

Changes compared with Qwen3.5:

1. Local attention. The 8 softmax-attention layers only see the last `sliding_window` (2048) tokens, and their KV cache
   is a rolling window. Everything older lives in the fixed-size Gated DeltaNet state of the 24 recurrent layers, so
   memory and per-token decode cost stop growing once the window is full.

2. Selective skip gate (Mamba-style selectivity with an explicit novelty signal). Every recurrent layer computes, per
   head, how redundant the current key is with the last `rq_redundancy_window` keys: `r = -log(1 - max cos sim)`,
   which is about 2 on ordinary prose and 6-9 on repetitive streams. A gate
   `skip = sigmoid(W x + u * r + c)` scales the decay, the write strength and the rotary phase step by
   `keep = 1 - skip`. A skipped token leaves the state untouched (no forgetting, no writing), like a Mamba token with
   a tiny timestep. Repetitive streams (NPC movement commands, logs, ticks) then stop eroding older conversation
   memory, while the local attention window still sees them verbatim.

3. Complex-valued state through data-dependent rotary embeddings (Mamba-3's "RoPE trick"). A learned per-head angle
   is accumulated over time and applied as a rotation to `rq_rotary_pairs` dimension pairs of the keys and queries.
   This is equivalent to a complex-valued transition and gives the recurrence rotational dynamics that help it track
   state (parity, counters, positions).

All new parameters are initialised so that the model starts out (almost exactly) as the pretrained Qwen3.5.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn
from transformers import initialization as init
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import (
    create_causal_mask,
    create_recurrent_attention_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention,
    Qwen3_5DecoderLayer,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5GatedDeltaNet,
    Qwen3_5MLP,
    Qwen3_5Model,
    Qwen3_5ModelOutputWithPast,
    Qwen3_5PreTrainedModel,
    Qwen3_5RMSNorm,
    Qwen3_5TextModel,
    Qwen3_5TextRotaryEmbedding,
    apply_mask_to_padding_states,
    apply_rotary_pos_emb,
    causal_conv1d_fn,
    causal_conv1d_update,
    eager_attention_forward,
    l2norm,
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

from .configuration_recurrent_qwen import RecurrentQwenConfig, RecurrentQwenTextConfig

# Cache slot used for the extra recurrent state (key buffer in the conv slot, rotary phase in the recurrent slot).
_EXTRA_STATE = 1

# Opt-in fast path for training: block-sparse FlexAttention over the sliding window (set to "flex" by the trainer).
# Inference uses the regular attention implementation chosen with `attn_implementation`.
TRAIN_ATTENTION = None
_flex_fn = None
_block_masks = {}


def flex_sliding_window(query, key, value, window, scaling):
    """Causal sliding-window attention computed only on the band of `window` keys. (B, H, T, D) in, (B, T, H, D) out."""
    global _flex_fn
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    if _flex_fn is None:
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
        _flex_fn = torch.compile(flex_attention, dynamic=False)
    T = query.shape[2]
    key_ = (T, window, query.device)
    if key_ not in _block_masks:
        def sliding(b, h, q_idx, kv_idx):
            return (q_idx >= kv_idx) & (q_idx - kv_idx < window)

        _block_masks[key_] = create_block_mask(sliding, B=None, H=None, Q_LEN=T, KV_LEN=T, device=query.device)
    out = _flex_fn(query, key, value, block_mask=_block_masks[key_], scale=scaling, enable_gqa=True)
    return out.transpose(1, 2).contiguous()


def cumulative_phase(steps: torch.Tensor, chunk: int = 64) -> torch.Tensor:
    """cumsum over dim 1 modulo 2*pi, in two levels so float32 stays accurate on long sequences."""
    batch, seq_len = steps.shape[:2]
    pad = (-seq_len) % chunk
    x = F.pad(steps, (0, 0, 0, 0, 0, pad)) if pad else steps
    x = x.view(batch, -1, chunk, *steps.shape[2:])
    inner = x.cumsum(2)
    totals = inner[:, :, -1].double()
    offsets = torch.remainder(totals.cumsum(1) - totals, 2 * math.pi).float()
    phase = torch.remainder(inner + offsets[:, :, None], 2 * math.pi)
    return phase.reshape(batch, -1, *steps.shape[2:])[:, :seq_len]


def local_redundancy(keys: torch.Tensor, prev_keys: torch.Tensor | None, window: int, chunk: int = 512) -> torch.Tensor:
    """Max cosine similarity of each key with the `window` keys before it.

    Args:
        keys: `(batch, seq, heads, dim)` unit-norm keys of the current tokens.
        prev_keys: `(batch, past, heads, dim)` unit-norm keys that precede them (from the cache), or None.
    Returns:
        `(batch, seq, heads)` max cosine similarity in [0, 1]. Tokens with no history get 0.
    """
    batch, seq_len, heads, _ = keys.shape
    all_keys = keys if prev_keys is None else torch.cat([prev_keys.to(keys.dtype), keys], dim=1)
    past = all_keys.shape[1] - seq_len
    out = keys.new_zeros(batch, seq_len, heads)
    for start in range(0, seq_len, chunk):
        end = min(start + chunk, seq_len)
        lo = max(past + start - window, 0)
        hi = past + end - 1  # the last query only looks at keys strictly before it
        if hi <= lo:
            continue
        sim = torch.einsum("bqhd,bkhd->bhqk", keys[:, start:end], all_keys[:, lo:hi])
        q_pos = torch.arange(past + start, past + end, device=keys.device)
        k_pos = torch.arange(lo, hi, device=keys.device)
        dist = q_pos[:, None] - k_pos[None, :]
        valid = (dist >= 1) & (dist <= window)
        sim = sim.masked_fill(~valid, -1.0)
        out[:, start:end] = sim.amax(dim=-1).clamp_min(0.0).transpose(1, 2)
    return out


def rotate_pairs(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate the first `2 * pairs` dims of `x` (pairs are (i, i + pairs)) by the given angles."""
    pairs = cos.shape[-1]
    x1, x2, rest = x[..., :pairs], x[..., pairs : 2 * pairs], x[..., 2 * pairs :]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos, rest], dim=-1)


class RecurrentQwenGatedDeltaNet(Qwen3_5GatedDeltaNet):
    def __init__(self, config: RecurrentQwenTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.redundancy_window = config.rq_redundancy_window
        self.rotary_pairs = config.rq_rotary_pairs
        self.skip_redundancy_init = config.rq_skip_redundancy_init
        self.skip_bias_init = config.rq_skip_bias_init
        # Skip gate: data-dependent part, weight on the redundancy signal and bias, per value head.
        self.skip_proj = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.skip_redundancy = nn.Parameter(torch.full((self.num_v_heads,), self.skip_redundancy_init))
        self.skip_bias = nn.Parameter(torch.full((self.num_v_heads,), self.skip_bias_init))
        # Data-dependent rotation angles (complex-valued state).
        self.phase_proj = nn.Linear(self.hidden_size, self.num_v_heads * self.rotary_pairs, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)

        batch_size, seq_len, _ = hidden_states.shape
        use_precomputed_states = cache_params is not None and cache_params.has_previous_state(
            self.layer_idx, state_idx=0
        )

        mixed_qkv = self.in_proj_qkv(hidden_states)
        mixed_qkv = mixed_qkv.transpose(1, 2)

        z = self.in_proj_z(hidden_states)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)
        skip_logits = self.skip_proj(hidden_states)
        theta = self.phase_proj(hidden_states)

        if use_precomputed_states and seq_len == 1 and not cache_params.layers[self.layer_idx].record_past:
            conv_state = cache_params.layers[self.layer_idx].conv_states[0]
            mixed_qkv = causal_conv1d_update(
                mixed_qkv,
                conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            )
        else:
            if cache_params is not None:
                mixed_qkv = cache_params.update_conv_state(
                    mixed_qkv, self.layer_idx, conv_kernel_size=self.conv_kernel_size
                )

            mixed_qkv = causal_conv1d_fn(
                mixed_qkv,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                activation=self.activation,
                **kwargs,
            )

            if cache_params is not None:
                mixed_qkv = mixed_qkv[:, :, -seq_len:]

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)

        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        # Novelty signal: how much does this key repeat one of the recent keys? (a feature, so no gradient)
        with torch.no_grad():
            unit_keys = l2norm(key.float())
            prev_keys = None
            if cache_params is not None:
                key_buffer = cache_params.update_conv_state(
                    unit_keys.reshape(batch_size, seq_len, -1).transpose(1, 2).to(mixed_qkv.dtype),
                    self.layer_idx,
                    state_idx=_EXTRA_STATE,
                    conv_kernel_size=self.redundancy_window,
                )
                all_keys = key_buffer.transpose(1, 2).reshape(batch_size, -1, self.num_k_heads, self.head_k_dim)
                prev_keys = all_keys[:, : all_keys.shape[1] - seq_len]
            redundancy = local_redundancy(unit_keys, prev_keys, self.redundancy_window)

            # log-distance makes near-exact repeats (sim 0.999 vs 0.99) separable; caps at -log(1e-4) ~ 9.2
            redundancy = -torch.log1p(-redundancy.clamp(max=1.0 - 1e-4))

        heads_per_key = self.num_v_heads // self.num_k_heads
        redundancy = redundancy.repeat_interleave(heads_per_key, dim=-1)
        skip = torch.sigmoid(skip_logits.float() + self.skip_redundancy.float() * redundancy + self.skip_bias.float())
        keep = 1.0 - skip
        if attention_mask is not None and attention_mask.dim() == 2:
            keep = keep * attention_mask[:, -seq_len:, None].to(keep.dtype)

        # Gates: a skipped token neither decays nor writes the state.
        beta = (b.float().sigmoid() * keep).to(b.dtype)
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias) * keep

        if heads_per_key > 1:
            query = query.repeat_interleave(heads_per_key, dim=2)
            key = key.repeat_interleave(heads_per_key, dim=2)

        # Data-dependent rotation: phase_t = sum_{s<=t} keep_s * pi * tanh(theta_s), kept modulo 2*pi.
        step = math.pi * torch.tanh(theta.float().view(batch_size, seq_len, self.num_v_heads, self.rotary_pairs))
        phase = cumulative_phase(keep.unsqueeze(-1) * step)
        if use_precomputed_states:
            phase = torch.remainder(
                phase + cache_params.layers[self.layer_idx].recurrent_states[_EXTRA_STATE][:, None], 2 * math.pi
            )
        if cache_params is not None:
            cache_params.update_recurrent_state(phase[:, -1].contiguous(), self.layer_idx, state_idx=_EXTRA_STATE)
        cos, sin = phase.cos(), phase.sin()
        query = rotate_pairs(query.float(), cos, sin).to(value.dtype)
        key = rotate_pairs(key.float(), cos, sin).to(value.dtype)

        recurrent_state = cache_params.layers[self.layer_idx].recurrent_states[0] if use_precomputed_states else None
        delta_rule = (
            torch_recurrent_gated_delta_rule if use_precomputed_states and seq_len == 1 else torch_chunk_gated_delta_rule
        )
        core_attn_out, last_recurrent_state = delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=recurrent_state,
            output_final_state=cache_params is not None,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=kwargs.pop("cu_seq_lens_q", None),
            **kwargs,
        )

        if cache_params is not None:
            cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        return self.out_proj(core_attn_out)


class RecurrentQwenAttention(Qwen3_5Attention):
    """Qwen3.5 gated attention restricted to a sliding window."""

    def __init__(self, config: RecurrentQwenTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.sliding_window = config.sliding_window

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        if TRAIN_ATTENTION == "flex" and past_key_values is None and query_states.is_cuda:
            attn_output = flex_sliding_window(query_states, key_states, value_states, self.sliding_window,
                                              self.scaling)
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = attn_output * torch.sigmoid(gate)
            return self.o_proj(attn_output), None

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        return self.o_proj(attn_output), attn_weights


class RecurrentQwenDecoderLayer(Qwen3_5DecoderLayer):
    def __init__(self, config: RecurrentQwenTextConfig, layer_idx: int):
        GradientCheckpointingLayer.__init__(self)
        self.hidden_size = config.hidden_size
        self.block_type = config.layer_types[layer_idx]
        if self.block_type == "linear_attention":
            self.linear_attn = RecurrentQwenGatedDeltaNet(config, layer_idx)
        else:
            self.self_attn = RecurrentQwenAttention(config, layer_idx)
        self.mlp = Qwen3_5MLP(config, config.intermediate_size)
        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.block_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                attention_mask=attention_mask,
                **kwargs,
            )
        else:
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class RecurrentQwenPreTrainedModel(Qwen3_5PreTrainedModel):
    config_class = RecurrentQwenConfig
    _no_split_modules = ["RecurrentQwenDecoderLayer", "Qwen3_5VisionBlock"]

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, RecurrentQwenGatedDeltaNet):
            init.zeros_(module.skip_proj.weight)
            init.zeros_(module.phase_proj.weight)
            init.constant_(module.skip_redundancy, module.skip_redundancy_init)
            init.constant_(module.skip_bias, module.skip_bias_init)


class RecurrentQwenTextModel(RecurrentQwenPreTrainedModel, Qwen3_5TextModel):
    config_class = RecurrentQwenTextConfig

    def __init__(self, config: RecurrentQwenTextConfig):
        PreTrainedModel.__init__(self, config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [RecurrentQwenDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ) -> Qwen3_5ModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = None

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": text_position_ids,
            }
            causal_mask_mapping = {
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
                "linear_attention": create_recurrent_attention_mask(**mask_kwargs),
            }
            if "full_attention" in self.config.layer_types:
                causal_mask_mapping["full_attention"] = create_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return Qwen3_5ModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)


class RecurrentQwenModel(RecurrentQwenPreTrainedModel, Qwen3_5Model):
    def __init__(self, config: RecurrentQwenConfig):
        PreTrainedModel.__init__(self, config)
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

        self.visual = Qwen3_5VisionModel._from_config(config.vision_config)
        self.language_model = RecurrentQwenTextModel._from_config(config.text_config)
        self.rope_deltas = None
        self.post_init()


class RecurrentQwenForConditionalGeneration(RecurrentQwenPreTrainedModel, Qwen3_5ForConditionalGeneration):
    def __init__(self, config: RecurrentQwenConfig):
        PreTrainedModel.__init__(self, config)
        self.model = RecurrentQwenModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()


__all__ = [
    "RecurrentQwenPreTrainedModel",
    "RecurrentQwenTextModel",
    "RecurrentQwenModel",
    "RecurrentQwenForConditionalGeneration",
]
