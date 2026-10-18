"""On-policy data: the original Qwen3.5-4B answers real prompts itself, with thinking on and off.

  modal run --detach -m rq_modal.generate::main

Writes /vol/data/gen_outputs.jsonl. Text records keep the exact prompt/output token ids from vLLM; image records keep
the decoded text (the trainer re-renders them with the HF processor so pixel inputs line up).
"""

import base64
import glob
import json
import random

import modal

from rq_modal.common import BASE_MODEL, DATA_DIR, IM_END, VOL_PATH, hf_secret, vllm_image, volume

app = modal.App("rq-generate")
EOT = 248044


@app.function(image=vllm_image, gpu="H100", volumes={VOL_PATH: volume}, secrets=[hf_secret], timeout=2 * 3600,
              cpu=8, memory=65536)
def generate(max_think_tokens: int = 7000, max_nothink_tokens: int = 1500, limit: int = 0):
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams, TokensPrompt

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    prompts = [json.loads(l) for l in open(f"{DATA_DIR}/gen_prompts.jsonl")]
    vision = []
    for path in sorted(glob.glob(f"{DATA_DIR}/vision/prompts_*.parquet")):
        vision += pq.read_table(path).to_pylist()
    rng = random.Random(0)
    for v in vision:
        v["thinking"] = rng.random() < 2 / 3
    if limit:
        prompts, vision = prompts[:limit], vision[: max(1, limit // 4)]

    llm = LLM(model=BASE_MODEL, max_model_len=max_think_tokens + 4096, gpu_memory_utilization=0.90,
              limit_mm_per_prompt={"image": 2, "video": 0}, max_num_seqs=384, seed=0)
    stop = [IM_END, EOT]
    think_sp = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=max_think_tokens,
                              stop_token_ids=stop, skip_special_tokens=False, seed=0)
    plain_sp = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, max_tokens=max_nothink_tokens,
                              stop_token_ids=stop, skip_special_tokens=False, seed=0)

    inputs = [TokensPrompt(prompt_token_ids=tok.apply_chat_template(
        p["messages"], tokenize=True, add_generation_prompt=True, enable_thinking=p["thinking"])) for p in prompts]
    params = [think_sp if p["thinking"] else plain_sp for p in prompts]
    outs = llm.generate(inputs, params)

    n_tok = 0
    with open(f"{DATA_DIR}/gen_outputs.jsonl", "w") as f:
        for p, o in zip(prompts, outs):
            c = o.outputs[0]
            ids = list(c.token_ids)
            while ids and ids[-1] in stop:
                ids.pop()
            n_tok += len(ids)
            f.write(json.dumps(dict(kind="text", thinking=p["thinking"], source=p["source"],
                                    prompt_ids=list(o.prompt_token_ids), output_ids=ids,
                                    finished=c.finish_reason == "stop")) + "\n")
        print(f"text: {len(prompts)} prompts, {n_tok} generated tokens", flush=True)

        conversations, vparams = [], []
        for v in vision:
            content = [{"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64," + base64.b64encode(b).decode()}} for b in v["images"]]
            conversations.append([{"role": "user", "content": content + [{"type": "text", "text": v["question"]}]}])
            vparams.append(think_sp if v["thinking"] else plain_sp)
        if conversations:
            vouts = []
            for i in range(0, len(conversations), 64):
                for th in (True, False):
                    idx = [j for j in range(i, min(i + 64, len(conversations))) if vision[j]["thinking"] == th]
                    if not idx:
                        continue
                    res = llm.chat([conversations[j] for j in idx], think_sp if th else plain_sp,
                                   chat_template_kwargs={"enable_thinking": th}, use_tqdm=False)
                    vouts += list(zip(idx, res))
            n_v = 0
            for j, o in vouts:
                c = o.outputs[0]
                if c.finish_reason != "stop":
                    continue
                text = c.text.replace("<|im_end|>", "").replace("<|endoftext|>", "")
                answer = ("<think>\n" + text) if vision[j]["thinking"] else text
                f.write(json.dumps(dict(kind="image", thinking=vision[j]["thinking"], subset=vision[j]["subset"],
                                        images=[base64.b64encode(b).decode() for b in vision[j]["images"]],
                                        question=vision[j]["question"], answer=answer)) + "\n")
                n_v += 1
            print(f"vision: {n_v}/{len(vision)} finished answers", flush=True)
    volume.commit()


@app.local_entrypoint()
def main(limit: int = 0):
    generate.remote(limit=limit)
