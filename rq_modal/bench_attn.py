"""Micro-benchmark of sliding-window attention implementations on B200 (fwd + bwd).

  modal run -m rq_modal.bench_attn
"""

import modal

from rq_modal.common import gpu_image

app = modal.App("rq-bench-attn")


@app.function(image=gpu_image, gpu="B200", timeout=1800)
def bench(T: int = 16384, W: int = 2048):
    import math
    import time

    import torch
    import torch.nn.functional as F
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    torch.manual_seed(0)
    H, Hkv, D = 16, 4, 256
    q = torch.randn(1, H, T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, Hkv, T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(1, Hkv, T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    scale = D**-0.5

    def rep(x):
        return x.repeat_interleave(H // Hkv, dim=1)

    def dense():
        i = torch.arange(T, device="cuda")
        d = i[:, None] - i[None, :]
        return F.scaled_dot_product_attention(q, rep(k), rep(v), attn_mask=(d >= 0) & (d < W), scale=scale)

    def banded():
        outs = []
        for s in range(0, T, W):
            e, ks = min(s + W, T), max(0, s - W + 1)
            d = torch.arange(s, e, device="cuda")[:, None] - torch.arange(ks, e, device="cuda")[None, :]
            outs.append(F.scaled_dot_product_attention(q[:, :, s:e], rep(k[:, :, ks:e]), rep(v[:, :, ks:e]),
                                                       attn_mask=(d >= 0) & (d < W), scale=scale))
        return torch.cat(outs, dim=2)

    def sw(b, h, qi, ki):
        return (qi >= ki) & (qi - ki < W)

    bm = create_block_mask(sw, B=None, H=None, Q_LEN=T, KV_LEN=T, device="cuda")
    flex = torch.compile(flex_attention, dynamic=False)

    def flexed():
        return flex(q, k, v, block_mask=bm, scale=scale, enable_gqa=True)

    def full_flash():  # reference: full causal flash (what the teacher uses)
        return F.scaled_dot_product_attention(q, rep(k), rep(v), is_causal=True, scale=scale)

    ref = dense().detach()
    g = torch.randn_like(ref)
    for name, fn in [("dense mask (current)", dense), ("banded sdpa", banded), ("flex block-sparse", flexed),
                     ("full causal flash (teacher)", full_flash)]:
        try:
            out = fn()
            err = (out.detach() - ref).abs().max().item() if name != "full causal flash (teacher)" else float("nan")
            out.backward(g)
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(3):
                q.grad = k.grad = v.grad = None
                fn().backward(g)
            torch.cuda.synchronize()
            print(f"{name:30s} fwd+bwd {1000 * (time.time() - t0) / 3:8.1f} ms/layer   max|diff|={err:.3g}",
                  flush=True)
        except Exception as e:
            print(f"{name:30s} FAILED: {repr(e)[:300]}", flush=True)

    # phase cumsum: float64 over the sequence vs two-level float32
    x = torch.rand(1, T, 32, 16, device="cuda") * math.pi

    def cs64():
        return torch.remainder(torch.cumsum(x.double(), 1), 2 * math.pi).float()

    def cs2(C=256):
        xx = x.view(1, T // C, C, 32, 16)
        inner = xx.cumsum(2)
        tot = inner[:, :, -1].double()
        off = torch.remainder(tot.cumsum(1) - tot, 2 * math.pi).float()
        return torch.remainder(inner + off[:, :, None], 2 * math.pi).reshape(1, T, 32, 16)

    a, b = cs64(), cs2()
    diff = torch.remainder(a - b + math.pi, 2 * math.pi) - math.pi
    for name, fn in [("cumsum float64", cs64), ("cumsum two-level fp32", cs2)]:
        fn()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        print(f"{name:30s} {1000 * (time.time() - t0) / 10:.2f} ms   max angle err {diff.abs().max().item():.2e}",
              flush=True)


@app.local_entrypoint()
def main():
    bench.remote()
