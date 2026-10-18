"""CPU checks on a tiny random RecurrentQwen: forward, cached decoding == full forward, bounded cache."""

import torch

from recurrent_qwen.configuration_recurrent_qwen import RecurrentQwenConfig
from recurrent_qwen.modeling_recurrent_qwen import RecurrentQwenForConditionalGeneration, local_redundancy


def tiny_config(window=16):
    text = dict(
        vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, linear_conv_kernel_dim=4, linear_key_head_dim=16,
        linear_value_head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4, sliding_window=window,
        rq_redundancy_window=8, rq_rotary_pairs=4, rope_parameters=dict(
            rope_type="default", rope_theta=10000, partial_rotary_factor=0.5, mrope_section=[1, 1, 0],
            mrope_interleaved=True),
    )
    vision = dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64, patch_size=16)
    return RecurrentQwenConfig(text_config=text, vision_config=vision)


def test_redundancy():
    keys = torch.nn.functional.normalize(torch.randn(1, 10, 2, 8), dim=-1)
    keys[:, 7] = keys[:, 3]
    red = local_redundancy(keys, None, window=8, chunk=3)
    assert red[0, 0].abs().max() == 0
    assert torch.allclose(red[0, 7], torch.ones(2))
    # same result when part of the history comes from a cache
    red2 = local_redundancy(keys[:, 5:], keys[:, :5], window=8, chunk=4)
    assert torch.allclose(red[:, 5:], red2, atol=1e-6)


@torch.no_grad()
def test_incremental_matches_full():
    torch.manual_seed(0)
    cfg = tiny_config(window=16)
    model = RecurrentQwenForConditionalGeneration(cfg).eval()
    # make the new gates non-trivial so the test exercises them
    for layer in model.model.language_model.layers:
        if layer.block_type == "linear_attention":
            layer.linear_attn.phase_proj.weight.normal_(0, 0.3)
            layer.linear_attn.skip_proj.weight.normal_(0, 0.3)
            layer.linear_attn.skip_bias.fill_(-1.0)
    T = 45  # longer than the window so the rolling KV cache is exercised
    ids = torch.randint(0, 128, (2, T))
    full = model(input_ids=ids).logits

    out = model(input_ids=ids[:, :20], use_cache=True)
    cache = out.past_key_values
    steps = [out.logits]
    for t in range(20, 30):
        out = model(input_ids=ids[:, t : t + 1], past_key_values=cache, use_cache=True)
        steps.append(out.logits)
    out = model(input_ids=ids[:, 30:], past_key_values=cache, use_cache=True)  # chunked prefill after decoding
    steps.append(out.logits)
    inc = torch.cat(steps, dim=1)
    err = (inc - full).abs().max().item()
    print("max |incremental - full| =", err)
    assert err < 1e-4, err
    # attention KV cache is bounded by the window
    for i, layer in enumerate(cache.layers):
        if cfg.text_config.layer_types[i] == "sliding_attention":
            assert layer.keys.shape[-2] <= cfg.text_config.sliding_window - 1, layer.keys.shape


@torch.no_grad()
def test_window_limits_attention():
    """Tokens further back than the window can only influence attention layers through the recurrence."""
    torch.manual_seed(0)
    cfg = tiny_config(window=8)
    model = RecurrentQwenForConditionalGeneration(cfg).eval()
    attn = model.model.language_model.layers[3].self_attn
    x = torch.randn(1, 20, 64)
    pos = torch.arange(20)[None, None].expand(3, 1, -1)
    pe = model.model.language_model.rotary_emb(x, pos)
    from transformers.masking_utils import create_sliding_window_causal_mask
    mask = create_sliding_window_causal_mask(config=cfg.text_config, inputs_embeds=x, attention_mask=None,
                                             past_key_values=None, position_ids=torch.arange(20)[None])
    y1, _ = attn(x, pe, mask)
    x2 = x.clone()
    x2[:, :5] = torch.randn(1, 5, 64)
    y2, _ = attn(x2, pe, mask)
    assert torch.allclose(y1[:, 13:], y2[:, 13:], atol=1e-5)  # positions >= 5 + 8 can't see tokens 0..4
    assert not torch.allclose(y1[:, 5:12], y2[:, 5:12], atol=1e-5)


def test_backward():
    torch.manual_seed(0)
    model = RecurrentQwenForConditionalGeneration(tiny_config()).train()
    ids = torch.randint(0, 128, (1, 40))
    loss = model(input_ids=ids, labels=ids).loss
    loss.backward()
    gdn = model.model.language_model.layers[0].linear_attn
    assert gdn.phase_proj.weight.grad.abs().sum() > 0
    assert gdn.skip_proj.weight.grad.abs().sum() > 0
    assert gdn.skip_bias.grad.abs().sum() > 0


@torch.no_grad()
def test_generate_matches_uncached_greedy():
    torch.manual_seed(1)
    cfg = tiny_config(window=8)
    model = RecurrentQwenForConditionalGeneration(cfg).eval()
    for layer in model.model.language_model.layers:
        if layer.block_type == "linear_attention":
            layer.linear_attn.phase_proj.weight.normal_(0, 0.3)
    ids = torch.randint(0, 128, (1, 12))
    gen = model.generate(input_ids=ids, max_new_tokens=20, do_sample=False, eos_token_id=None, pad_token_id=0)
    seq = ids
    for _ in range(20):
        nxt = model(input_ids=seq).logits[:, -1].argmax(-1, keepdim=True)
        seq = torch.cat([seq, nxt], dim=1)
    assert torch.equal(gen, seq), (gen, seq)


if __name__ == "__main__":
    test_redundancy()
    test_incremental_matches_full()
    test_window_limits_attention()
    test_backward()
    test_generate_matches_uncached_greedy()
    print("all tiny tests passed")
