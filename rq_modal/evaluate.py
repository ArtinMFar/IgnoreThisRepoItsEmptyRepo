"""Evaluate RecurrentQwen-4B against the original Qwen3.5-4B (and the untrained conversion).

  modal run -m rq_modal.evaluate::main                       # all models in parallel containers
  modal run -m rq_modal.evaluate::main --models trained --quick

Results go to /vol/out/eval_<model>.json.
"""

import ast
import json
import os
import re
import string
import time
from collections import Counter

import modal

from rq_modal.common import BASE_MODEL, CKPT_DIR, OUT_DIR, VOL_PATH, gpu_image, hf_secret, volume

app = modal.App("rq-eval")

LONGBENCH = ["hotpotqa_e", "2wikimqa_e", "multifieldqa_en_e", "qasper_e", "passage_retrieval_en_e"]


def load_model(name):
    import torch
    from transformers import Qwen3_5ForConditionalGeneration

    from recurrent_qwen.convert import load_from_base
    from recurrent_qwen.modeling_recurrent_qwen import RecurrentQwenForConditionalGeneration

    if name == "original":
        m = Qwen3_5ForConditionalGeneration.from_pretrained(BASE_MODEL, dtype=torch.bfloat16,
                                                            attn_implementation="sdpa")
    elif name == "naive":  # sliding window + new gates at init, no training
        m = load_from_base(BASE_MODEL, dtype=torch.bfloat16)
    else:
        m = RecurrentQwenForConditionalGeneration.from_pretrained(f"{CKPT_DIR}/{name}", dtype=torch.bfloat16,
                                                                  attn_implementation="sdpa")
    return m.cuda().eval()


def cache_bytes(cache):
    import torch

    total = 0
    for layer in cache.layers:
        for attr in ("keys", "values"):
            t = getattr(layer, attr, None)
            if isinstance(t, torch.Tensor):
                total += t.numel() * t.element_size()
        for states in (getattr(layer, "conv_states", {}), getattr(layer, "recurrent_states", {})):
            for t in states.values():
                if isinstance(t, torch.Tensor):
                    total += t.numel() * t.element_size()
    return total


def chunked_nll(model, ids, chunk=4096):
    """Per-token NLL of a long sequence via chunked prefill with the model's own cache."""
    import torch
    from transformers import DynamicCache

    cache = DynamicCache(config=model.config)
    nll = []
    for a in range(0, ids.shape[1], chunk):
        x = ids[:, a : a + chunk]
        pos = torch.arange(a, a + x.shape[1], device=ids.device)[None]
        h = model.model(input_ids=x, position_ids=pos, past_key_values=cache, use_cache=True).last_hidden_state
        logits = model.lm_head(h[0]).float()
        nxt = ids[0, a + 1 : a + 1 + x.shape[1]]
        lp = logits[: nxt.shape[0]].log_softmax(-1).gather(-1, nxt[:, None])[:, 0]
        nll.append(-lp)
    return torch.cat(nll)


def eval_ppl(model, tok, n_books, length):
    import torch
    from datasets import load_dataset

    buckets = [(0, 2048), (2048, 4096), (4096, 8192), (8192, 16384), (16384, 32768), (32768, 65536)]
    buckets = [b for b in buckets if b[0] < length]
    sums, counts, used = Counter(), Counter(), 0
    for row in load_dataset("emozilla/pg19", split="test"):
        ids = tok(row["text"], add_special_tokens=False).input_ids
        if len(ids) < length + 1:
            continue
        x = torch.tensor(ids[: length + 1], device="cuda")[None]
        with torch.no_grad():
            nll = chunked_nll(model, x)
        nll = nll[:length]
        for lo, hi in buckets:
            sums[(lo, hi)] += nll[lo:hi].sum().item()
            counts[(lo, hi)] += nll[lo:hi].numel()
        used += 1
        if used >= n_books:
            break
    import math

    return {f"{lo // 1024}k-{hi // 1024}k": round(math.exp(sums[(lo, hi)] / counts[(lo, hi)]), 3)
            for lo, hi in buckets}


def normalize(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1(pred, gold):
    p, g = normalize(pred).split(), normalize(gold).split()
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def strip_think(text):
    return text.split("</think>")[-1].strip()


def generate(model, tok, messages, thinking, max_new_tokens):
    import torch

    ids = tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=thinking,
                                  return_tensors="pt", return_dict=True)["input_ids"].cuda()
    with torch.no_grad():
        out = model.generate(input_ids=ids, max_new_tokens=max_new_tokens, do_sample=False, temperature=None,
                             top_p=None, top_k=None)
    return tok.decode(out[0, ids.shape[1] :], skip_special_tokens=True), ids.shape[1]


def generate_batch(model, tok, conversations, thinking, max_new_tokens):
    """Greedy generation for many short prompts at once (left padding)."""
    import torch

    texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
             for m in conversations]
    tok.padding_side = "left"
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None,
                             top_k=None, pad_token_id=tok.pad_token_id)
    return [tok.decode(o[enc["input_ids"].shape[1] :], skip_special_tokens=True) for o in out]


def eval_longbench(model, tok, per_bucket, max_prompt_tokens=24000):
    from datasets import load_dataset

    res = {}
    for task in LONGBENCH:
        ds = load_dataset("Xnhyacinth/LongBench", task, split="test")
        rows = {"0-4k": [], "4-8k": [], "8k+": []}
        for r in ds:
            n = int(r["length"])
            b = "0-4k" if n < 4000 else "4-8k" if n < 8000 else "8k+"
            if len(rows[b]) < per_bucket:
                rows[b].append(r)
        res[task] = {}
        for b, rs in rows.items():
            scores = []
            for r in rs:
                ctx_ids = tok(r["context"], add_special_tokens=False).input_ids
                if len(ctx_ids) > max_prompt_tokens:  # LongBench-style middle truncation
                    half = max_prompt_tokens // 2
                    r = dict(r, context=tok.decode(ctx_ids[:half]) + tok.decode(ctx_ids[-half:]))
                prompt = r["context"].rstrip() + "\n\n" + r["question"].strip() + "\n" + (r["answer_prefix"] or "")
                pred, _ = generate(model, tok, [{"role": "user", "content": prompt}], False,
                                   int(r["max_new_tokens"] or 64))
                golds = ast.literal_eval(r["answers"]) if isinstance(r["answers"], str) else r["answers"]
                if task.startswith("passage_retrieval"):
                    nums = re.findall(r"\d+", pred)
                    score = float(bool(nums) and any(nums[0] == re.findall(r"\d+", g)[0] for g in golds))
                else:
                    score = max(f1(strip_think(pred), g) for g in golds)
                scores.append(score)
            res[task][b] = round(100 * sum(scores) / max(len(scores), 1), 1)
        print(task, res[task], flush=True)
    return res


def last_number(text):
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text.replace("$", ""))
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def eval_gsm8k(model, tok, n, thinking):
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test").select(range(n))
    correct, lengths, long_traces = 0, [], 0
    convs = [[{"role": "user", "content": r["question"] + "\nPut the final answer after 'Answer:'."}] for r in ds]
    outs = []
    for i in range(0, n, 64):
        outs += generate_batch(model, tok, convs[i : i + 64], thinking, 6144 if thinking else 1024)
    for r, out in zip(ds, outs):
        n_tok = len(tok(out).input_ids)
        lengths.append(n_tok)
        long_traces += n_tok > 2048
        gold = r["answer"].split("####")[-1].strip().replace(",", "")
        pred = last_number(strip_think(out))
        try:
            correct += pred is not None and abs(float(pred) - float(gold)) < 1e-6
        except ValueError:
            pass
    return dict(acc=round(100 * correct / n, 1), mean_tokens=round(sum(lengths) / n), over_2048=long_traces)


def eval_arc(model, tok, n):
    import torch
    from datasets import load_dataset

    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test").select(range(n))
    correct = 0
    for r in ds:
        labels = r["choices"]["label"]
        opts = "\n".join(f"{l}. {t}" for l, t in zip(labels, r["choices"]["text"]))
        msg = [{"role": "user", "content": f"{r['question']}\n{opts}\nAnswer with the letter only."}]
        ids = tok.apply_chat_template(msg, add_generation_prompt=True, enable_thinking=False, return_tensors="pt",
                                      return_dict=True)["input_ids"].cuda()
        with torch.no_grad():
            logits = model(input_ids=ids, logits_to_keep=1).logits[0, -1]
        cand = [tok.encode(l, add_special_tokens=False)[0] for l in labels]
        correct += labels[int(torch.stack([logits[c] for c in cand]).argmax())] == r["answerKey"]
    return round(100 * correct / n, 1)


def eval_chartqa(model, processor, n):
    import torch
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceM4/ChartQA", split="test").select(range(n))
    correct = 0
    for r in ds:
        msg = [{"role": "user", "content": [{"type": "image"},
                                            {"type": "text", "text": r["query"] + "\nAnswer with a single word "
                                                                                  "or number."}]}]
        text = processor.apply_chat_template(msg, add_generation_prompt=True, enable_thinking=False, tokenize=False)
        enc = processor(text=[text], images=[r["image"].convert("RGB")], return_tensors="pt",
                        return_mm_token_type_ids=True).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=24, do_sample=False, temperature=None, top_p=None,
                                 top_k=None)
        pred = processor.decode(out[0, enc["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        gold = str(r["label"][0] if isinstance(r["label"], list) else r["label"]).strip()
        try:
            ok = abs(float(pred.rstrip("%")) - float(gold.rstrip("%"))) <= 0.05 * abs(float(gold.rstrip("%")))
        except ValueError:
            ok = normalize(pred) == normalize(gold)
        correct += ok
    return round(100 * correct / n, 1)


def bench(model, tok, lengths):
    """Cache memory, decode latency and new-turn TTFT as a function of context length."""
    import torch
    from datasets import load_dataset
    from transformers import DynamicCache

    text = ""
    for row in load_dataset("emozilla/pg19", split="test"):
        text += row["text"]
        if len(text) > 5 * max(lengths):
            break
    all_ids = torch.tensor(tok(text, add_special_tokens=False).input_ids[: max(lengths)], device="cuda")[None]
    turn = torch.tensor(tok("<|im_start|>user\nWhat happened in the last chapter?<|im_end|>\n<|im_start|>"
                            "assistant\n").input_ids, device="cuda")[None]
    res = {}
    for L in lengths:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        cache = DynamicCache(config=model.config)
        with torch.no_grad():
            torch.cuda.synchronize()
            t0 = time.time()
            for a in range(0, L, 8192):
                x = all_ids[:, a : min(a + 8192, L)]
                pos = torch.arange(a, a + x.shape[1], device="cuda")[None]
                model.model(input_ids=x, position_ids=pos, past_key_values=cache, use_cache=True)
            torch.cuda.synchronize()
            prefill_s = time.time() - t0
            cb = cache_bytes(cache)
            # new turn on top of the cached history: TTFT = prefill the turn + first token
            pos = torch.arange(L, L + turn.shape[1], device="cuda")[None]
            torch.cuda.synchronize()
            t0 = time.time()
            out = model(input_ids=turn, position_ids=pos, past_key_values=cache, use_cache=True, logits_to_keep=1)
            torch.cuda.synchronize()
            ttft_ms = (time.time() - t0) * 1000
            nxt = out.logits[:, -1:].argmax(-1)
            p = L + turn.shape[1]
            times = []
            for i in range(32):
                torch.cuda.synchronize()
                t0 = time.time()
                out = model(input_ids=nxt, position_ids=torch.tensor([[p + i]], device="cuda"),
                            past_key_values=cache, use_cache=True, logits_to_keep=1)
                nxt = out.logits[:, -1:].argmax(-1)
                torch.cuda.synchronize()
                times.append(time.time() - t0)
        res[L] = dict(cache_mb=round(cb / 2**20, 1), prefill_s=round(prefill_s, 2), ttft_new_turn_ms=round(ttft_ms, 1),
                      decode_ms_per_token=round(1000 * sorted(times)[16], 2),
                      peak_mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2))
        print(L, res[L], flush=True)
        del cache
    return res


def eval_cache_consistency(model, tok, n_turns=4, chars_per_turn=9000):
    """README usage pattern: keep the cache across turns and feed only new tokens. Compare the last answer and its
    time-to-first-token with recomputing the whole (window-exceeding) history from scratch."""
    import torch
    from datasets import load_dataset
    from transformers import DynamicCache

    book = next(r["text"] for r in load_dataset("emozilla/pg19", split="test") if len(r["text"]) > 60000)
    cache = DynamicCache(config=model.config)
    seen = torch.empty(1, 0, dtype=torch.long, device="cuda")
    answers, ttft = [], {}
    for i in range(n_turns):
        part = book[i * chars_per_turn : (i + 1) * chars_per_turn]
        msg = f"Here is part {i + 1} of a book:\n{part}\nSummarise this part in one sentence."
        turn = ("\n" if seen.numel() else "") + f"<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n" \
               "<think>\n\n</think>\n\n"
        new = tok(turn, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
        ids = torch.cat([seen, new], dim=1)
        with torch.no_grad():
            torch.cuda.synchronize()
            t0 = time.time()
            out = model.generate(input_ids=ids, past_key_values=cache, max_new_tokens=1, do_sample=False,
                                 temperature=None, top_p=None, top_k=None)
            torch.cuda.synchronize()
            ttft["cached"] = round(1000 * (time.time() - t0), 1)
            out = model.generate(input_ids=out, past_key_values=cache, max_new_tokens=40, do_sample=False,
                                 temperature=None, top_p=None, top_k=None)
        answers.append(out[0, ids.shape[1] :])
        last_ids, seen = ids, out
    with torch.no_grad():
        torch.cuda.synchronize()
        t0 = time.time()
        fresh = model.generate(input_ids=last_ids, max_new_tokens=1, do_sample=False, temperature=None, top_p=None,
                               top_k=None)
        torch.cuda.synchronize()
        ttft["full_recompute"] = round(1000 * (time.time() - t0), 1)
        fresh = model.generate(input_ids=last_ids, max_new_tokens=41, do_sample=False, temperature=None,
                               top_p=None, top_k=None)[0, last_ids.shape[1] :]
    a = answers[-1]
    n = min(len(a), len(fresh))
    return dict(history_tokens=int(last_ids.shape[1]), cache_mb=round(cache_bytes(cache) / 2**20, 1),
                ttft_ms=ttft, same_tokens=f"{int((a[:n] == fresh[:n]).sum())}/{n}",
                cached_answer=tok.decode(a, skip_special_tokens=True), fresh_answer=tok.decode(fresh, skip_special_tokens=True))


@app.function(image=gpu_image, gpu="H100", volumes={VOL_PATH: volume}, secrets=[hf_secret], timeout=3 * 3600,
              cpu=8, memory=65536)
def evaluate(name: str, quick: bool = False):
    import torch
    from transformers import AutoProcessor

    torch.backends.cuda.matmul.allow_tf32 = True
    volume.reload()
    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    tok = processor.tokenizer
    model = load_model(name)
    t0 = time.time()
    res = dict(model=name)
    out_path = f"{OUT_DIR}/eval_{name}{'_quick' if quick else ''}.json"
    os.makedirs(OUT_DIR, exist_ok=True)

    def run(key, fn):
        try:
            res[key] = fn()
        except Exception as e:  # keep the other sections
            import traceback

            traceback.print_exc()
            res[key] = f"failed: {e!r}"[:500]
        print(key, res[key], flush=True)
        res["eval_minutes"] = round((time.time() - t0) / 60, 1)
        json.dump(res, open(out_path, "w"), indent=1)
        volume.commit()

    run("pg19_ppl", lambda: eval_ppl(model, tok, n_books=4 if quick else 16, length=32768))
    run("longbench_e", lambda: eval_longbench(model, tok, per_bucket=2 if quick else 6))
    if name != "naive":
        run("arc_challenge", lambda: eval_arc(model, tok, 50 if quick else 400))
        run("chartqa", lambda: eval_chartqa(model, processor, 20 if quick else 150))
        run("gsm8k_nothink", lambda: eval_gsm8k(model, tok, 10 if quick else 120, thinking=False))
        run("gsm8k_think", lambda: eval_gsm8k(model, tok, 10 if quick else 80, thinking=True))
        run("bench", lambda: bench(model, tok, [4096, 16384] if quick else [4096, 16384, 65536, 131072]))
        run("cache_consistency", lambda: eval_cache_consistency(model, tok))
    return res


@app.local_entrypoint()
def main(models: str = "final,original,naive", quick: bool = False):
    for r in evaluate.starmap([(m, quick) for m in models.split(",")], return_exceptions=True):
        print(json.dumps(r, indent=1, default=str))


@app.function(image=gpu_image, gpu="H100", volumes={VOL_PATH: volume}, secrets=[hf_secret], timeout=1800,
              cpu=4, memory=65536)
def consistency(name: str = "final", n_turns: int = 4, chars_per_turn: int = 9000):
    """Logit-level check: cached turn-by-turn processing vs one full pass over a window-exceeding history."""
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer, DynamicCache

    from recurrent_qwen.modeling_recurrent_qwen import RecurrentQwenForConditionalGeneration

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    book = next(r["text"] for r in load_dataset("emozilla/pg19", split="test") if len(r["text"]) > 60000)
    pieces = []
    for i in range(n_turns):
        part = book[i * chars_per_turn : (i + 1) * chars_per_turn]
        s = ("\n" if i else "") + f"<|im_start|>user\nHere is part {i + 1} of a book:\n{part}\nSummarise this part " \
            "in one sentence.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        pieces.append(tok(s, return_tensors="pt", add_special_tokens=False).input_ids.cuda())
    res = {}
    for dtype in (torch.float32, torch.bfloat16):
        model = RecurrentQwenForConditionalGeneration.from_pretrained(f"{CKPT_DIR}/{name}", dtype=dtype).cuda().eval()
        with torch.no_grad():
            cache = DynamicCache(config=model.config)
            for p in pieces:
                cached = model(input_ids=p, past_key_values=cache, use_cache=True, logits_to_keep=1).logits[0, -1]
            full = model(input_ids=torch.cat(pieces, 1), logits_to_keep=1).logits[0, -1]
        top2 = full.float().topk(2).values
        res[str(dtype)] = dict(history_tokens=int(sum(p.shape[1] for p in pieces)),
                               max_abs_logit_diff=round((cached.float() - full.float()).abs().max().item(), 4),
                               kl=float(torch.nn.functional.kl_div(cached.float().log_softmax(-1),
                                                                   full.float().log_softmax(-1), log_target=True,
                                                                   reduction="sum")),
                               same_argmax=bool(cached.argmax() == full.argmax()),
                               top1_top2_margin=round((top2[0] - top2[1]).item(), 3))
        print(dtype, res[str(dtype)], flush=True)
        del model
        torch.cuda.empty_cache()
    json.dump(res, open(f"{OUT_DIR}/consistency_{name}.json", "w"), indent=1)
    volume.commit()
    return res


@app.local_entrypoint()
def check_consistency(name: str = "final"):
    print(consistency.remote(name))
