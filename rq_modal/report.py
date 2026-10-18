"""Build the model-card result tables from the eval JSONs (runs locally, no GPU).

  python -m rq_modal.report <dir with eval_final.json, eval_original.json, eval_naive.json> hub/README.md out.md
"""

import json
import os
import sys

NAMES = {"original": "Qwen3.5-4B (original, full attention)", "naive": "Converted, untrained",
         "final": "**RecurrentQwen-4B**"}


def load(d):
    return {k: json.load(open(f"{d}/eval_{k}.json")) for k in NAMES if os.path.exists(f"{d}/eval_{k}.json")}


def ppl_table(res):
    cols = list(next(iter(res.values()))["pg19_ppl"].keys())
    rows = [f"| Model | {' | '.join(c + ' tokens' for c in cols)} |", "|---" * (len(cols) + 1) + "|"]
    for k, r in res.items():
        if isinstance(r.get("pg19_ppl"), dict):
            rows.append(f"| {NAMES[k]} | " + " | ".join(f"{r['pg19_ppl'][c]:.2f}" for c in cols) + " |")
    return "\n".join(rows)


def longbench_table(res):
    tasks = list(next(iter(res.values()))["longbench_e"].keys())
    rows = ["| Model | " + " | ".join(t.removesuffix("_e") for t in tasks) + " | average |",
            "|---" * (len(tasks) + 2) + "|"]
    for k, r in res.items():
        lb = r.get("longbench_e")
        if not isinstance(lb, dict):
            continue
        cells, allv = [], []
        for t in tasks:
            v = lb[t]
            cells.append(" / ".join(f"{v[b]:.0f}" for b in ("0-4k", "4-8k", "8k+")))
            allv += [v[b] for b in ("0-4k", "4-8k", "8k+")]
        rows.append(f"| {NAMES[k]} | " + " | ".join(cells) + f" | {sum(allv) / len(allv):.1f} |")
    return "\n".join(rows)


def task_table(res):
    rows = ["| Model | ARC-Challenge | GSM8K (thinking off) | GSM8K (thinking on) | ChartQA (vision) |",
            "|---|---|---|---|---|"]
    for k, r in res.items():
        if "arc_challenge" not in r:
            continue

        def g(key):
            v = r.get(key)
            return f"{v['acc']:.1f}" if isinstance(v, dict) else "n/a"

        rows.append(f"| {NAMES[k]} | {r['arc_challenge']} | {g('gsm8k_nothink')} | {g('gsm8k_think')} | "
                    f"{r.get('chartqa', 'n/a')} |")
    return "\n".join(rows)


def bench_table(res):
    rows = ["| Context | Cache, original | Cache, RecurrentQwen | Prefill from scratch, original / RecurrentQwen | "
            "New-turn TTFT with cached history, original / RecurrentQwen |", "|---|---|---|---|---|"]
    o, f = res.get("original", {}).get("bench"), res.get("final", {}).get("bench")
    if not isinstance(o, dict) or not isinstance(f, dict):
        return ""
    for L in o:
        a, b = o[L], f.get(L) or f.get(str(L))
        if not b:
            continue
        rows.append(f"| {int(L) // 1024}k | {a['cache_mb']:.0f} MB | {b['cache_mb']:.0f} MB | {a['prefill_s']:.2f} s / "
                    f"{b['prefill_s']:.2f} s | {a['ttft_new_turn_ms']:.0f} ms / {b['ttft_new_turn_ms']:.0f} ms |")
    return "\n".join(rows)


def main(d, card_in, card_out):
    res = load(d)
    tables = "\n\n".join([
        "### Long-text perplexity by token position (PG19 test books, 32k tokens each; lower is better)",
        "Tokens past 2048 can only be predicted from what the recurrent state remembers.",
        ppl_table(res),
        "### LongBench-E (real long-document QA; F1 / accuracy by context length 0-4k / 4-8k / 8k+ words)",
        longbench_table(res),
        "### General ability, reasoning (thinking on/off) and vision",
        task_table(res),
        "### Memory and latency (H100, batch 1, bf16, plain Hugging Face implementation)",
        bench_table(res),
    ])
    card = open(card_in).read().replace("{{RESULTS_TABLES}}", tables)
    open(card_out, "w").write(card)
    print(tables)


if __name__ == "__main__":
    main(*sys.argv[1:4])
