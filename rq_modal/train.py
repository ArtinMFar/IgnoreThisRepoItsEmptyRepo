"""Full-parameter distillation of Qwen3.5-4B (teacher, full attention) into RecurrentQwen-4B (student, 2048-token
sliding-window attention + gated recurrent memory) on one B200.

Loss per token = KL(teacher || student) over the whole vocabulary (all tokens, text and around images)
               + ce_weight * cross-entropy on assistant tokens.

  modal run -m rq_modal.train::main --smoke             # a few steps, prints memory and throughput
  modal run --detach -m rq_modal.train::main            # full, time-budgeted run (resumes from /vol/ckpt/latest)
"""

import glob
import json
import math
import os
import queue
import random
import threading
import time

import modal

from rq_modal.common import BASE_MODEL, CKPT_DIR, DATA_DIR, VOL_PATH, gpu_image, hf_secret, volume

app = modal.App("rq-train")


def lr_factor(progress, step, warmup=20, decay_start=0.8, final=0.1):
    """Warmup, then constant, then linear decay to `final` over the last 20% of the (time/data) budget."""
    w = min(1.0, (step + 1) / warmup)
    if progress < decay_start:
        return w
    return w * (1.0 - (1.0 - final) * min(1.0, (progress - decay_start) / (1.0 - decay_start)))


def vision_pool(rng):
    import pyarrow.parquet as pq

    pool = []
    for path in sorted(glob.glob(f"{DATA_DIR}/vision/*.parquet")):
        if not os.path.basename(path).startswith("prompts_"):
            pool += pq.read_table(path).to_pylist()
    gen = f"{DATA_DIR}/gen_outputs.jsonl"
    if os.path.exists(gen):
        pool += [r for r in map(json.loads, open(gen)) if r["kind"] == "image"]
    rng.shuffle(pool)
    return pool


def vision_worker(processor, pool, out_q, target_tokens, seed):
    """Background thread: pack vision samples into ~target_tokens sequences."""
    from rq_modal.batches import vision_batch

    rng = random.Random(seed)
    while True:
        group, est = [], 0
        while est < target_tokens:
            s = pool[rng.randrange(len(pool))]
            group.append(s)
            est += 700  # rough tokens per sample (images are at most ~576 tokens)
        try:
            out_q.put(vision_batch(processor, group))
        except Exception as e:  # a broken image should not kill training
            print("vision batch skipped:", repr(e)[:200], flush=True)


@app.function(image=gpu_image, gpu="B200", volumes={VOL_PATH: volume}, secrets=[hf_secret], timeout=6 * 3600,
              cpu=16, memory=196608, retries=modal.Retries(max_retries=2, initial_delay=10.0))
def train(budget_minutes: float = 150.0, smoke: bool = False, tokens_per_step: int = 65536,
          lr: float = 1e-5, ce_weight: float = 0.2, vision_tokens: int = 6000, ckpt_minutes: float = 30.0,
          chunk: int = 4096, attn: str = "sdpa"):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    from recurrent_qwen.convert import load_from_base
    from recurrent_qwen.modeling_recurrent_qwen import RecurrentQwenForConditionalGeneration
    from rq_modal.batches import text_batch

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = torch.device("cuda")
    t_start = time.time()
    run_dir = f"{CKPT_DIR}/smoke" if smoke else f"{CKPT_DIR}/latest"
    state = dict(step=0, seq_index=0, elapsed=0.0, tokens=0)
    if not smoke and os.path.exists(f"{run_dir}/train_state.json"):
        state = json.load(open(f"{run_dir}/train_state.json"))
        print("resuming from", state, flush=True)

    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    teacher = Qwen3_5ForConditionalGeneration.from_pretrained(BASE_MODEL, dtype=torch.bfloat16,
                                                              attn_implementation="sdpa").to(dev).eval()
    teacher.requires_grad_(False)
    if state["step"] > 0:
        student = RecurrentQwenForConditionalGeneration.from_pretrained(run_dir, dtype=torch.float32,
                                                                        attn_implementation=attn)
    else:
        student = load_from_base(BASE_MODEL, dtype=torch.float32, attn_implementation=attn)
    student.to(dev).train()
    student.config.use_cache = False
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    groups = {"gates": [], "gate_scalars": [], "vision": [], "base": []}
    for n, p in student.named_parameters():
        if n.endswith(("skip_redundancy", "skip_bias")):
            groups["gate_scalars"].append(p)
        elif "skip_proj" in n or "phase_proj" in n:
            groups["gates"].append(p)
        elif n.startswith("model.visual."):
            groups["vision"].append(p)
        else:
            groups["base"].append(p)
    base_lrs = {"gates": 3e-4, "gate_scalars": 3e-3, "vision": lr * 0.2, "base": lr}
    opt = torch.optim.AdamW([dict(params=v, lr=base_lrs[k], name=k) for k, v in groups.items()],
                            betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0, fused=True)
    print({k: sum(p.numel() for p in v) for k, v in groups.items()}, flush=True)

    d = f"{DATA_DIR}/train"
    ids_all = np.load(f"{d}/ids.npy", mmap_mode="r")
    mask_all = np.load(f"{d}/mask.npy", mmap_mode="r")
    lens = np.load(f"{d}/lens.npy")
    offs = np.concatenate([[0], np.cumsum(lens)])
    total_tokens = int(offs[-1])
    print(json.load(open(f"{d}/meta.json")), flush=True)

    vq = queue.Queue(maxsize=4)
    pool = vision_pool(random.Random(0))
    print("vision samples:", len(pool), flush=True)
    for w in range(3):
        threading.Thread(target=vision_worker, args=(processor, pool, vq, vision_tokens, w + state["step"]),
                         daemon=True).start()

    def micro_loss(batch, norm):
        """Forward teacher + student on one sequence, backprop the chunked KL/CE loss. Returns (kl, ce, n_ce)."""
        inputs = {k: v.to(dev) for k, v in batch["inputs"].items()}
        ids = inputs["input_ids"]
        kl_mask = batch["kl_mask"].to(dev)[0]
        ce_mask = batch["ce_mask"].to(dev)[0]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            t_h = teacher.model(**inputs, use_cache=False).last_hidden_state[0]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            s_h = student.model(**inputs, use_cache=False).last_hidden_state[0]
        s_leaf = s_h.detach().requires_grad_()
        T = ids.shape[1]
        kl_sum = ce_sum = 0.0
        n_ce = 0
        for a in range(0, T, chunk):
            b = min(a + chunk, T)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                s_logits = student.lm_head(s_leaf[a:b])
                with torch.no_grad():
                    t_logits = teacher.lm_head(t_h[a:b])
            s_logp = F.log_softmax(s_logits.float(), dim=-1)
            with torch.no_grad():
                t_logp = F.log_softmax(t_logits.float(), dim=-1)
                t_p = t_logp.exp()
            kl = (t_p * (t_logp - s_logp)).sum(-1)
            kl = (kl * kl_mask[a:b]).sum()
            # cross-entropy on assistant tokens: position i predicts token i + 1
            tgt_mask = ce_mask[a + 1 : b + 1]
            tgt = ids[0, a + 1 : b + 1]
            n = tgt.shape[0]
            ce = -(s_logp[:n].gather(-1, tgt[:, None])[:, 0] * tgt_mask).sum()
            ((kl + ce_weight * ce) / norm).backward()
            kl_sum += kl.item()
            ce_sum += ce.item()
            n_ce += int(tgt_mask.sum())
            del s_logits, t_logits, s_logp, t_logp, t_p
        s_h.backward(s_leaf.grad)
        return kl_sum, ce_sum, n_ce, int(kl_mask.sum())

    def save(path, final=False):
        t0 = time.time()
        os.makedirs(path, exist_ok=True)
        sd = {k: v.detach().to(torch.bfloat16) for k, v in student.state_dict().items()}
        student.save_pretrained(path, state_dict=sd, safe_serialization=True)
        if not final:
            json.dump(state, open(f"{path}/train_state.json", "w"))
        volume.commit()
        print(f"saved {path} in {time.time() - t0:.0f}s", flush=True)

    budget_s = budget_minutes * 60
    seq_index = state["seq_index"]
    last_ckpt = time.time()
    log = []
    while True:
        elapsed = state["elapsed"] + (time.time() - t_start)
        progress = max(elapsed / budget_s, state["tokens"] / max(total_tokens, 1))
        if progress >= 1.0 or seq_index >= len(lens) or (smoke and state["step"] >= 6):
            break
        f = lr_factor(progress, state["step"])
        for g in opt.param_groups:
            g["lr"] = base_lrs[g["name"]] * f

        batches, n = [], 0
        while n < tokens_per_step - vision_tokens and seq_index < len(lens):
            a, b = offs[seq_index], offs[seq_index + 1]
            batches.append(text_batch(np.asarray(ids_all[a:b]), np.asarray(mask_all[a:b])))
            n += b - a
            seq_index += 1
        text_tokens = n
        try:
            vb = vq.get(timeout=120)
            batches.append(vb)
            n += vb["inputs"]["input_ids"].shape[1]
        except queue.Empty:
            print("no vision batch ready", flush=True)

        t0 = time.time()
        kl_t = ce_t = 0.0
        n_ce = n_kl = 0
        for batch in batches:
            k, c, nc, nk = micro_loss(batch, n)
            kl_t, ce_t, n_ce, n_kl = kl_t + k, ce_t + c, n_ce + nc, n_kl + nk
        gnorm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0).item()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        dt = time.time() - t0
        state["step"] += 1
        state["tokens"] += text_tokens
        state["seq_index"] = seq_index
        rec = dict(step=state["step"], kl=kl_t / max(n_kl, 1), ce=ce_t / max(n_ce, 1), gnorm=gnorm, lr_f=f,
                   tok_s=n / dt, progress=progress, mem_gb=torch.cuda.max_memory_allocated() / 2**30)
        log.append(rec)
        if state["step"] % 5 == 0 or smoke or state["step"] <= 3:
            gdn = [l.linear_attn for l in student.model.language_model.layers if l.block_type == "linear_attention"]
            rec["skip_bias"] = float(torch.stack([g.skip_bias.mean() for g in gdn]).mean())
            rec["skip_red"] = float(torch.stack([g.skip_redundancy.mean() for g in gdn]).mean())
            rec["phase_w"] = float(torch.stack([g.phase_proj.weight.norm() for g in gdn]).mean())
            print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in rec.items()}), flush=True)
        if not smoke and time.time() - last_ckpt > ckpt_minutes * 60:
            state["elapsed"] = state["elapsed"] + (time.time() - t_start)
            t_start = time.time()
            save(run_dir)
            last_ckpt = time.time()

    state["elapsed"] = state["elapsed"] + (time.time() - t_start)
    os.makedirs(f"{CKPT_DIR}/logs", exist_ok=True)
    with open(f"{CKPT_DIR}/logs/{'smoke' if smoke else 'train'}_{int(time.time())}.jsonl", "w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in log)
    if smoke:
        save(f"{CKPT_DIR}/smoke", final=True)
    else:
        save(run_dir)
        save(f"{CKPT_DIR}/final", final=True)
    print("done", state, flush=True)
    return state


@app.local_entrypoint()
def main(smoke: bool = False, budget_minutes: float = 150.0, attn: str = "sdpa"):
    print(train.remote(budget_minutes=budget_minutes, smoke=smoke, attn=attn))
