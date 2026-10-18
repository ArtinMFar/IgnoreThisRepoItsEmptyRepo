"""Compare RecurrentQwen (fresh from a Qwen3.5 checkpoint) against the original model on CPU.

usage: python -m tests.test_equivalence /path/to/Qwen3.5-0.8B
"""

import sys

import torch
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from recurrent_qwen.convert import load_from_base
from recurrent_qwen.modeling_recurrent_qwen import local_redundancy

TEXT = (
    "The Treaty of Westphalia in 1648 ended the Thirty Years' War and established a framework of sovereign states "
    "that still shapes international relations. Historians debate how novel it really was: some argue that the "
    "principle of non-interference had older roots, while others see the treaty as the moment Europe accepted "
    "religious pluralism as a political fact. Its negotiations took place in two cities, Osnabrück and Münster, "
    "and involved nearly every major power of the continent. "
) * 3
NPC = "".join(f"<npc id=7 action=move dx=1 dy=0 t={i}>\n" for i in range(60))


@torch.no_grad()
def main(base):
    proc = AutoProcessor.from_pretrained(base)
    ids = proc.tokenizer(TEXT, return_tensors="pt").input_ids
    print("tokens:", ids.shape[1])
    ref = Qwen3_5ForConditionalGeneration.from_pretrained(base, dtype=torch.float32).eval()
    ref_logits = ref(input_ids=ids).logits

    exact = load_from_base(base, dtype=torch.float32, rq_skip_redundancy_init=0.0, rq_skip_bias_init=-60.0).eval()
    err = (exact(input_ids=ids).logits - ref_logits).abs().max().item()
    print(f"identity-gate init vs original: max |dlogit| = {err:.2e}")
    assert err < 1e-3, err

    model = load_from_base(base, dtype=torch.float32).eval()
    logits = model(input_ids=ids).logits
    kl = torch.nn.functional.kl_div(
        logits.log_softmax(-1), ref_logits.log_softmax(-1), log_target=True, reduction="none"
    ).sum(-1).mean().item()
    print(f"default init vs original: mean KL = {kl:.2e} nats/token")

    # redundancy statistics of the pretrained keys on prose vs a repetitive NPC stream
    for name, text in [("prose", TEXT), ("npc", NPC)]:
        x = proc.tokenizer(text, return_tensors="pt").input_ids
        stats = []

        def hook(mod, args, kwargs, out):
            hs = kwargs["hidden_states"]
            qkv = mod.in_proj_qkv(hs)
            qkv = torch.nn.functional.silu(
                torch.nn.functional.conv1d(qkv.transpose(1, 2), mod.conv1d.weight, padding=3, groups=mod.conv_dim)
            )[..., : hs.shape[1]].transpose(1, 2)
            k = qkv[..., mod.key_dim : 2 * mod.key_dim].reshape(1, hs.shape[1], mod.num_k_heads, -1)
            stats.append(local_redundancy(torch.nn.functional.normalize(k, dim=-1), None, mod.redundancy_window))

        hooks = [l.linear_attn.register_forward_hook(hook, with_kwargs=True)
                 for l in model.model.language_model.layers if l.block_type == "linear_attention"]
        model(input_ids=x)
        for h in hooks:
            h.remove()
        r = torch.stack(stats)  # (layers, 1, T, heads)
        print(f"redundancy[{name}] tokens={x.shape[1]} mean={r.mean():.3f} p90={r.flatten().quantile(0.9):.3f} "
              f"frac>0.9={(r > 0.9).float().mean():.3f}")


if __name__ == "__main__":
    main(sys.argv[1])
