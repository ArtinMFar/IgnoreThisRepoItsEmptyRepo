"""Build the training data on CPU containers (real data only).

  modal run -m rq_modal.data::build_all          # base weights + all parts (parallel)
  modal run -m rq_modal.data::build_onpolicy     # after rq_modal.generate has run
  modal run -m rq_modal.data::pack               # merge parts into the shuffled training set

Parts are stored as flat arrays under /vol/data/parts/<name>/: ids.npy (int32), mask.npy (uint8, 1 = assistant
token, used for the cross-entropy term), lens.npy (int32 length of each packed sequence).
"""

import io
import json
import os
import random

import modal

from rq_modal.common import (
    ASSISTANT, BASE_MODEL, DATA_DIR, IM_END, IM_START, NEWLINE, cpu_image, hf_secret, volume, VOL_PATH,
)

app = modal.App("rq-data")
FN = dict(image=cpu_image, volumes={VOL_PATH: volume}, secrets=[hf_secret], cpu=8, memory=32768, timeout=3 * 3600)

SMOLTALK = "HuggingFaceTB/smoltalk2"
# token budgets per SmolTalk2 SFT split
CHAT_NOTHINK = {
    "smoltalk_smollm3_smol_magpie_ultra_no_think": 5_000_000,
    "OpenHermes_2.5_no_think": 2_000_000,
    "tulu_3_sft_personas_instruction_following_no_think": 1_500_000,
    "smoltalk_smollm3_systemchats_30k_no_think": 1_000_000,
    "smoltalk_smollm3_everyday_conversations_no_think": 300_000,
    "smoltalk_smollm3_smol_summarize_no_think": 1_000_000,
    "smoltalk_smollm3_smol_rewrite_no_think": 500_000,
    "smoltalk_smollm3_explore_instruct_rewriting_no_think": 500_000,
    "Mixture_of_Thoughts_science_no_think": 1_000_000,
    "OpenThoughts3_1.2M_no_think_no_think": 1_500_000,
    "smoltalk_multilingual_8languages_lang_5_no_think": 1_000_000,
    "table_gpt_no_think": 200_000,
}
CHAT_THINK = {
    "OpenThoughts3_1.2M_think": 6_000_000,
    "multi_turn_reasoning_if_think": 3_000_000,
    "smoltalk_systemchats_Qwen3_32B_think": 1_500_000,
    "smoltalk_everyday_convs_reasoning_Qwen3_32B_think": 500_000,
    "aya_dataset_Qwen3_32B_think": 1_000_000,
    "smoltalk_multilingual8_Qwen3_32B_think": 1_500_000,
    "table_gpt_Qwen3_32B_think": 500_000,
    "s1k_1.1_think": 1_000_000,
}
LONG = {
    "LongAlign_64k_Qwen3_32B_yarn_131k_think": 7_000_000,
    "LongAlign_64k_context_lang_annotated_lang_6_no_think": 7_000_000,
}
PG19_TOKENS, PG19_CHUNK = 8_000_000, 16384
MAX_LEN = 16384
VISION = {  # FineVision subset -> number of samples
    "LLaVA_Instruct_150K": 1200, "sharegpt4o": 800, "vision_flan(filtered)": 600, "lnqa": 500, "vqav2": 700,
    "a_okvqa": 500, "chartqa": 700, "docvqa": 700, "textvqa": 600, "ai2d_merged": 600, "infographic_vqa": 500,
    "ocrvqa": 400, "scienceqa": 400, "CoSyn_400k_chart": 500, "CoSyn_400k_document": 500, "tallyqa": 400,
    "textcaps": 400, "screenqa": 400, "geo170k(qa)": 400, "iconqa": 300,
}
VISION_PROMPTS = 1200  # extra held-out vision samples whose questions are answered by the teacher (rq_modal.generate)


def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(BASE_MODEL)


def assistant_mask(ids):
    """1 for tokens inside assistant turns (content and the closing <|im_end|>), else 0."""
    mask = [0] * len(ids)
    inside = False
    for i, t in enumerate(ids):
        if inside:
            mask[i] = 1
            if t == IM_END:
                inside = False
        elif t == NEWLINE and i >= 2 and ids[i - 1] == ASSISTANT and ids[i - 2] == IM_START:
            inside = True
    return mask


def clean_messages(messages):
    out = []
    for m in messages:
        if m.get("role") not in ("system", "user", "assistant") or not isinstance(m.get("content"), str):
            return None
        if m["role"] == "system" and out:
            return None
        out.append({"role": m["role"], "content": m["content"]})
    if not out or out[-1]["role"] != "assistant":
        return None
    return out


def render(tok, messages, thinking):
    text = tok.apply_chat_template(messages, tokenize=False, enable_thinking=thinking)
    ids = tok(text, add_special_tokens=False).input_ids
    return ids, assistant_mask(ids)


def pack_items(items, max_len, seed=0):
    """Greedy packing of (ids, mask) items into sequences of at most max_len tokens."""
    random.Random(seed).shuffle(items)
    seqs, cur_ids, cur_mask = [], [], []
    for ids, mask in items:
        if len(ids) > max_len:
            continue
        if len(cur_ids) + len(ids) > max_len:
            seqs.append((cur_ids, cur_mask))
            cur_ids, cur_mask = [], []
        cur_ids, cur_mask = cur_ids + ids, cur_mask + mask
    if cur_ids:
        seqs.append((cur_ids, cur_mask))
    return seqs


def save_part(name, seqs, meta):
    import numpy as np

    d = f"{DATA_DIR}/parts/{name}"
    os.makedirs(d, exist_ok=True)
    np.save(f"{d}/ids.npy", np.fromiter((t for s, _ in seqs for t in s), dtype=np.int32))
    np.save(f"{d}/mask.npy", np.fromiter((t for _, m in seqs for t in m), dtype=np.uint8))
    np.save(f"{d}/lens.npy", np.array([len(s) for s, _ in seqs], dtype=np.int32))
    meta = dict(meta, sequences=len(seqs), tokens=int(sum(len(s) for s, _ in seqs)))
    json.dump(meta, open(f"{d}/meta.json", "w"), indent=1)
    volume.commit()
    print(name, meta)
    return meta


def stream_split(split, seed=0):
    from datasets import load_dataset

    ds = load_dataset(SMOLTALK, "SFT", split=split, streaming=True)
    return ds.shuffle(seed=seed, buffer_size=5_000)


@app.function(**FN)
def download_base():
    from huggingface_hub import snapshot_download

    path = snapshot_download(BASE_MODEL)
    volume.commit()
    return path


@app.function(**FN)
def build_chat(name: str, budgets: dict, thinking: bool, max_len: int):
    tok = tokenizer()
    items, stats = [], {}
    for split, budget in budgets.items():
        n_tok = n = 0
        for row in stream_split(split):
            msgs = clean_messages(row["messages"])
            if msgs is None:
                continue
            try:
                ids, mask = render(tok, msgs, thinking)
            except Exception:
                continue
            if len(ids) > max_len:
                continue
            items.append((ids, mask))
            n_tok += len(ids)
            n += 1
            if n_tok >= budget:
                break
        stats[split] = dict(conversations=n, tokens=n_tok)
        print(split, stats[split], flush=True)
    return save_part(name, pack_items(items, max_len), dict(thinking=thinking, splits=stats))


@app.function(**FN)
def build_long():
    """LongAlign (real long-document QA, thinking on and off) + PG19 book chunks."""
    from datasets import load_dataset

    tok = tokenizer()
    for split, budget in LONG.items():
        thinking = split.endswith("_think") and not split.endswith("no_think")
        items, n_tok = [], 0
        for row in stream_split(split):
            msgs = clean_messages(row["messages"])
            if msgs is None or sum(len(m["content"]) for m in msgs) > 6 * MAX_LEN:
                continue
            ids, mask = render(tok, msgs, thinking)
            if len(ids) > MAX_LEN or len(ids) < 4096:
                continue
            items.append((ids, mask))
            n_tok += len(ids)
            if n_tok >= budget:
                break
        save_part("long_think" if thinking else "long_nothink", items, dict(thinking=thinking, source=split))

    items, n_tok = [], 0
    rng = random.Random(0)
    for row in load_dataset("emozilla/pg19", split="train", streaming=True).shuffle(seed=0, buffer_size=200):
        ids = tok(row["text"], add_special_tokens=False).input_ids
        starts = list(range(0, max(len(ids) - PG19_CHUNK, 1), PG19_CHUNK))
        for s in rng.sample(starts, k=min(2, len(starts))):
            chunk = ids[s : s + PG19_CHUNK]
            if len(chunk) == PG19_CHUNK:
                items.append((chunk, [1] * len(chunk)))
                n_tok += len(chunk)
        if n_tok >= PG19_TOKENS:
            break
    save_part("pg19", items, dict(thinking=None, source="emozilla/pg19"))


def _jpeg(img, max_side=768):
    img = img.convert("RGB")
    scale = max_side / max(img.size)
    if scale < 1:
        img = img.resize((max(32, round(img.width * scale)), max(32, round(img.height * scale))))
    if min(img.size) < 32:
        return None
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


@app.function(**FN)
def build_vision_subset(subset: str, n: int, n_prompts: int):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceM4/FineVision", subset, split="train", streaming=True).shuffle(seed=0,
                                                                                              buffer_size=200)
    rows, prompts = [], []
    for r in ds:
        if (r.get("relevance_min") or 5) < 3 or (r.get("image_correspondence_min") or 5) < 3:
            continue
        if not r["images"] or len(r["images"]) > 2 or not r["texts"]:
            continue
        imgs = [_jpeg(im) for im in r["images"]]
        if any(b is None for b in imgs):
            continue
        turns = r["texts"][:6]
        if len(prompts) < n_prompts:
            prompts.append(dict(images=imgs, question=turns[0]["user"], subset=subset))
        else:
            messages = []
            for i, t in enumerate(turns):
                content = [{"type": "image"}] * len(imgs) + [{"type": "text", "text": t["user"]}] if i == 0 else \
                    [{"type": "text", "text": t["user"]}]
                messages += [{"role": "user", "content": content},
                             {"role": "assistant", "content": [{"type": "text", "text": t["assistant"]}]}]
            rows.append(dict(images=imgs, messages=json.dumps(messages), subset=subset))
        if len(rows) >= n:
            break
    d = f"{DATA_DIR}/vision"
    os.makedirs(d, exist_ok=True)
    safe = subset.replace("(", "_").replace(")", "").replace("/", "_")
    pq.write_table(pa.Table.from_pylist(rows), f"{d}/{safe}.parquet")
    if prompts:
        pq.write_table(pa.Table.from_pylist(prompts), f"{d}/prompts_{safe}.parquet")
    volume.commit()
    print(subset, len(rows), "samples,", len(prompts), "prompts")
    return subset, len(rows), len(prompts)


@app.function(**FN)
def build_prompts(n_think: int = 2400, n_nothink: int = 1500):
    """Real prompts that the original model answers itself (on-policy data), in both thinking modes."""
    tok = tokenizer()
    sources = {
        True: {"OpenThoughts3_1.2M_think": 0.45, "multi_turn_reasoning_if_think": 0.2, "s1k_1.1_think": 0.05,
               "aya_dataset_Qwen3_32B_think": 0.1, "smoltalk_systemchats_Qwen3_32B_think": 0.2},
        False: {"smoltalk_smollm3_smol_magpie_ultra_no_think": 0.5, "smoltalk_smollm3_systemchats_30k_no_think": 0.2,
                "tulu_3_sft_personas_instruction_following_no_think": 0.2,
                "smoltalk_smollm3_everyday_conversations_no_think": 0.1},
    }
    out = []
    for thinking, mix in sources.items():
        total = n_think if thinking else n_nothink
        for split, frac in mix.items():
            want, got = int(total * frac), 0
            for row in stream_split(split, seed=1234):  # different shuffle from the SFT parts
                msgs = clean_messages(row["messages"])
                if msgs is None:
                    continue
                prompt = []
                for m in msgs:
                    prompt.append(m)
                    if m["role"] == "user":
                        break
                if prompt[-1]["role"] != "user" or len(tok(prompt[-1]["content"]).input_ids) > 3000:
                    continue
                out.append(dict(messages=prompt, thinking=thinking, source=split))
                got += 1
                if got >= want:
                    break
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(f"{DATA_DIR}/gen_prompts.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    volume.commit()
    print(len(out), "prompts")
    return len(out)


@app.local_entrypoint()
def build_all():
    calls = [download_base.spawn(), build_prompts.spawn(), build_long.spawn(),
             build_chat.spawn("chat_nothink", CHAT_NOTHINK, False, 8192),
             build_chat.spawn("chat_think", CHAT_THINK, True, MAX_LEN)]
    per = max(1, VISION_PROMPTS // len(VISION))
    vision = list(build_vision_subset.starmap([(s, n, per) for s, n in VISION.items()], return_exceptions=True))
    for v in vision:
        print("vision:", v)
    for c in calls:
        print(c.get())


@app.function(**FN)
def build_onpolicy_fn():
    """Tokenize the teacher's own generations (exact token ids from vLLM) into packed sequences."""
    out = {True: [], False: []}
    with open(f"{DATA_DIR}/gen_outputs.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r.get("kind") != "text":
                continue
            ids = r["prompt_ids"] + r["output_ids"]
            mask = [0] * len(r["prompt_ids"]) + [1] * len(r["output_ids"])
            if r["finished"] and ids[-1] != IM_END:
                ids, mask = ids + [IM_END], mask + [1]
            ids, mask = ids + [NEWLINE], mask + [0]
            out[r["thinking"]].append((ids, mask))
    save_part("onpolicy_think", pack_items(out[True], MAX_LEN), dict(thinking=True, source="teacher generations"))
    save_part("onpolicy_nothink", pack_items(out[False], 8192), dict(thinking=False, source="teacher generations"))


@app.local_entrypoint()
def build_onpolicy():
    build_onpolicy_fn.remote()


# Share of each part in the training stream (by sequences, after packing). Vision is mixed in by the trainer.
MIX = {"chat_nothink": 1.0, "chat_think": 1.0, "long_think": 1.0, "long_nothink": 1.0, "pg19": 1.0,
       "onpolicy_think": 1.0, "onpolicy_nothink": 1.0}


@app.function(**FN)
def pack_fn():
    """Concatenate all parts into one shuffled training stream (every sequence used once)."""
    import numpy as np

    order, parts = [], {}
    for name in MIX:
        d = f"{DATA_DIR}/parts/{name}"
        if not os.path.exists(f"{d}/lens.npy"):
            print("missing part", name)
            continue
        lens = np.load(f"{d}/lens.npy")
        offs = np.concatenate([[0], np.cumsum(lens)])
        parts[name] = (np.load(f"{d}/ids.npy", mmap_mode="r"), np.load(f"{d}/mask.npy", mmap_mode="r"), offs)
        order += [(name, i) for i in range(len(lens))]
    random.Random(0).shuffle(order)
    ids_out, mask_out, lens_out, kinds = [], [], [], []
    names = list(parts)
    for name, i in order:
        ids, mask, offs = parts[name]
        ids_out.append(np.asarray(ids[offs[i] : offs[i + 1]]))
        mask_out.append(np.asarray(mask[offs[i] : offs[i + 1]]))
        lens_out.append(offs[i + 1] - offs[i])
        kinds.append(names.index(name))
    d = f"{DATA_DIR}/train"
    os.makedirs(d, exist_ok=True)
    np.save(f"{d}/ids.npy", np.concatenate(ids_out).astype(np.int32))
    np.save(f"{d}/mask.npy", np.concatenate(mask_out).astype(np.uint8))
    np.save(f"{d}/lens.npy", np.array(lens_out, dtype=np.int32))
    np.save(f"{d}/kinds.npy", np.array(kinds, dtype=np.int8))
    summary = {n: dict(sequences=int(sum(1 for k in kinds if k == j)),
                       tokens=int(sum(l for l, k in zip(lens_out, kinds) if k == j))) for j, n in enumerate(names)}
    json.dump(dict(names=names, parts=summary, total_tokens=int(sum(lens_out))), open(f"{d}/meta.json", "w"),
              indent=1)
    volume.commit()
    print(json.dumps(summary, indent=1), "total tokens", sum(lens_out))


@app.local_entrypoint()
def pack():
    pack_fn.remote()
