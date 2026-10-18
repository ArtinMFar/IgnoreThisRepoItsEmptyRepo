---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
library_name: transformers
pipeline_tag: image-text-to-text
tags:
  - recurrent
  - hybrid
  - linear-attention
  - gated-deltanet
  - sliding-window-attention
  - long-context
  - distillation
  - vision-language
  - reasoning
---

# RecurrentQwen-4B

**RecurrentQwen-4B** is [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) converted into a
[RecurrentGemma](https://arxiv.org/abs/2404.07839)-style model with **bounded memory**: its attention layers only
look at the last **2048 tokens**, and everything older is carried by the fixed-size state of its recurrent layers.
Once a conversation passes 2048 tokens, the cache stops growing and the cost per generated token stays flat, no
matter how long the history gets. It keeps Qwen3.5's image understanding and both of its modes (thinking on and off).

**At a glance** (compared with the original Qwen3.5-4B; full tables below):

- ✅ **Memory stops growing.** The cache is a constant **126 MB per conversation at any length** (original: 4.1 GB at
  128k tokens, +32 KB per token).
- ✅ **Time to first token stays flat.** Answering a new turn on top of a cached history takes ~75 ms at 16k, 64k and
  128k tokens of history (original: 74 → 178 ms). Prefilling 128k tokens from scratch is 1.6× faster (5.8 s vs 9.3 s).
- ✅ **Long-text modelling is recovered.** Perplexity on book positions 16k-32k is 12.8 (original 11.3; just windowing
  the attention without training: 36.6), and it stays flat out to 32k tokens, twice the training length.
- ⚠️ **Exact retrieval from long documents is much weaker than full attention:** LongBench-E average 35.2 vs 69.0
  (untrained conversion 33.5). The fixed-size state keeps the gist of long histories, but quoting a specific detail from
  far back is unreliable. Use the original model for that.
- ⚠️ **Short-input quality dropped a few points.** Thinking off: GSM8K 90.8 (original 95.0), MATH-500 71.0 (75.0),
  HumanEval 75.0 (84.1), ARC-Challenge 89.2 (91.0), ChartQA 60.7 (70.7). Thinking on: GSM8K 82.5 (93.8). On inputs that
  fit in the window the untrained conversion still behaves like the original, so these losses come from the short
  training run: it traded some short-input quality for long-text ability.

## What changed

Qwen3.5-4B is already a hybrid: 24 of its 32 layers are Gated DeltaNet (linear-attention recurrent) layers and 8 are
full softmax-attention layers whose KV cache grows with every token. RecurrentQwen changes three things.

| | Qwen3.5-4B | RecurrentQwen-4B |
|---|---|---|
| Attention layers (8) | full causal attention, KV cache grows forever | **2048-token sliding window**, rolling KV cache (≤2047 tokens) |
| Recurrent layers (24) | Gated DeltaNet | Gated DeltaNet + **selective skip gate** + **data-dependent rotation** |
| Memory per sequence | grows linearly with context | **constant** after 2048 tokens |
| Decode cost per token | grows with context | **constant** after 2048 tokens |

### Ideas taken from Mamba-3 (and Mamba)

1. **Selective skip gate with a novelty signal.** In Mamba the timestep Δ controls both forgetting and writing: a small Δ
   keeps the state and ignores the token. Gated DeltaNet controls decay and write strength with separate gates, so even an
   uninformative token still decays (erodes) the memory. RecurrentQwen adds a per-head gate
   `skip = sigmoid(W·x + u·r + c)` that scales the decay, the write strength and the rotation step together.
   `r = -log(1 - max cos-sim)` measures how much the current key repeats one of the previous 128 keys: it is about 2 on
   ordinary prose and 6 to 9 on repetitive streams (logs, repeated tool output, boilerplate, game-engine ticks).
   Repetitive content therefore gets low priority in the long-term state and stops drowning out earlier conversation,
   while the local attention window still sees it verbatim. The gate is learned from ordinary data during distillation;
   the redundancy signal costs a 128-key rolling buffer per layer.
2. **Complex-valued state via data-dependent rotary embeddings** (Mamba-3's "RoPE trick", Proposition 3 of the paper). A
   learned per-head angle is accumulated over time and applied as a rotation to 16 dimension pairs of the keys and
   queries. This is equivalent to a complex-valued transition and gives the recurrence rotational dynamics that
   real-valued decays cannot express (Mamba-3 shows this is what lets it solve state-tracking tasks such as parity).
   Unlike Mamba-3, which scales each rotation step by Δ, the step is scaled by the skip gate's `keep`, so skipped tokens do
   not rotate the state either. The accumulated phase is part of the fixed-size state.
3. **QK normalisation** (Mamba-3's BC norm) was already present in Gated DeltaNet and is kept.

Not adopted from Mamba-3:
- *Exponential-trapezoidal discretisation*: it adds a data-dependent two-tap convolution on the state input inside the
  recurrence itself, which needs a different recurrence kernel. Gated DeltaNet keeps its separate short causal
  convolution on the inputs instead.
- *MIMO state updates*: the paper reports better accuracy at the same state size and decode speed, but MIMO needs new
  kernels and new projections that cannot be initialised from Qwen's weights; out of scope for a short adaptation.
- *Learnable B/C (key/query) biases*: the delta rule needs unit-norm keys, which a bias added after normalisation
  would break.

All new parameters (33M, 0.7% of the model) start at values that reproduce the original model almost exactly.

## Training

- **Method:** full-parameter knowledge distillation. The original Qwen3.5-4B (full attention, sees everything) is
  the teacher; RecurrentQwen is the student. Loss per token = KL(teacher ‖ student) over the full 248k-token vocabulary
  + 0.2 × cross-entropy on assistant tokens. Training on sequences of up to 16k tokens forces the student to move
  everything older than 2048 tokens into the recurrent state.
- **Data (real data only, thinking on and off):**
  | Data | Tokens available | Thinking |
  |---|---|---|
  | SmolTalk2 chat and instruction following (12 `no_think` subsets: Magpie-Ultra, OpenHermes-2.5, system chats, Tulu-3 persona IF, summarisation, rewriting, multilingual, science, ...) | 15.5M | off |
  | SmolTalk2 reasoning (8 `think` subsets: OpenThoughts3, multi-turn reasoning IF, system chats, everyday conversations, multilingual, Aya, table reasoning, s1k) | 15.0M | on |
  | SmolTalk2 LongAlign-64k long-document QA (8k-16k tokens) | 7.0M + 7.0M | on + off |
  | PG19 books (16k-token chunks) | 8.0M | - |
  | The original Qwen3.5-4B's own answers to 3,900 real prompts (on-policy, exact sampled tokens) | 14.0M | on + off |
  | FineVision images (20 subsets: LLaVA-Instruct, ShareGPT-4o, ChartQA, DocVQA, TextVQA, AI2D, InfographicVQA, OCR-VQA, ScienceQA, VQAv2, A-OKVQA, CoSyn charts and documents, TallyQA, TextCaps, ScreenQA, Geo170k, IconQA, LNQA, Vision-FLAN) + 1,122 on-policy image answers | 12k samples | on + off |

  Training was time-budgeted and saw a random ~55-58% of each text part (38.1M text tokens) plus ~3.5M image-conversation
  tokens.
- **Compute:** 1x NVIDIA B200 (Modal), 100 minutes, 583 AdamW steps of ~64k tokens; fp32 master weights with bf16 autocast, gradient checkpointing and block-sparse FlexAttention for the window; learning rate 1e-5 (vision tower 2e-6, new gates 3e-4 / 2e-5 / 3e-3) with warmup and a final linear decay; the new gate groups are gradient-clipped separately. Teacher generations: vLLM on 1x H100. Whole project, including data building, generation and evaluation: about $28 of GPU/CPU time.

## Results

Sample sizes are small (LongBench-E: 6 questions per length bucket; GSM8K: 120 / 80 problems; MATH-500: first 100;
ChartQA: 150), so individual cells vary by several points. "Converted, untrained" is the same architecture right after
conversion, before any training: it shows what the distillation run changed.

### Long-text perplexity by token position (PG19 test books, 32k tokens each; lower is better)

Tokens past 2048 can only be predicted from what the recurrent state remembers.

| Model | 0k-2k tokens | 2k-4k tokens | 4k-8k tokens | 8k-16k tokens | 16k-32k tokens |
|---|---|---|---|---|---|
| Qwen3.5-4B (original, full attention) | 12.43 | 12.75 | 12.13 | 11.57 | 11.30 |
| Converted, untrained | 12.42 | 13.41 | 16.91 | 24.97 | 36.58 |
| **RecurrentQwen-4B** | 12.63 | 13.38 | 13.09 | 12.78 | 12.76 |

### LongBench-E (real long-document QA; F1 / accuracy by context length 0-4k / 4-8k / 8k+ words)

| Model | hotpotqa | 2wikimqa | multifieldqa_en | qasper | passage_retrieval_en | average |
|---|---|---|---|---|---|---|
| Qwen3.5-4B (original, full attention) | 69 / 94 / 64 | 65 / 67 / 73 | 50 / 51 / 51 | 64 / 56 / 29 | 100 / 100 / 100 | 69.0 |
| Converted, untrained | 36 / 14 / 16 | 28 / 37 / 17 | 38 / 29 / 24 | 33 / 45 / 52 | 33 / 33 / 67 | 33.5 |
| **RecurrentQwen-4B** | 36 / 44 / 28 | 38 / 19 / 19 | 42 / 29 / 23 | 29 / 52 / 36 | 33 / 33 / 67 | 35.2 |

### General ability, math, coding (thinking on/off) and vision

MATH-500 and HumanEval run with thinking off; HumanEval is pass@1 with the generated code executed against the tests.

| Model | ARC-Challenge | GSM8K (thinking off) | GSM8K (thinking on) | MATH-500 | HumanEval | ChartQA (vision) |
|---|---|---|---|---|---|---|
| Qwen3.5-4B (original, full attention) | 91.0 | 95.0 | 93.8 | 75.0 | 84.1 | 70.7 |
| Converted, untrained | 91.0 | 94.2 | 88.8 | 75.0 | 81.1 | 71.3 |
| **RecurrentQwen-4B** | 89.2 | 90.8 | 82.5 | 71.0 | 75.0 | 60.7 |

### Memory and latency (H100, batch 1, bf16, plain Hugging Face implementation)

Cache = everything that must be kept per conversation (KV cache + recurrent state). The 4k TTFT for RecurrentQwen
includes kernel warm-up. Per-token decode latency is ~55-59 ms for both models at every length here, because
single-stream decoding in the plain Hugging Face implementation is bound by Python overhead, so the constant-cost
benefit shows up as memory (how many conversations fit on a GPU) and long-history prefill/TTFT. At short contexts the
sliding-window attention (masked SDPA) prefill is slower than the original's flash attention.

| Context | Cache, original | Cache, RecurrentQwen | Prefill from scratch, original / RecurrentQwen | New-turn TTFT with cached history, original / RecurrentQwen |
|---|---|---|---|---|
| 4k | 178 MB | 126 MB | 0.13 s / 0.18 s | 78 ms / 124 ms |
| 16k | 562 MB | 126 MB | 0.51 s / 0.77 s | 74 ms / 80 ms |
| 64k | 2098 MB | 126 MB | 3.11 s / 2.93 s | 114 ms / 75 ms |
| 128k | 4146 MB | 126 MB | 9.33 s / 5.83 s | 178 ms / 76 ms |

## Usage

Requires `transformers>=5.20.1` (the model code builds on its Qwen3.5 implementation) and `trust_remote_code=True`.
Install `flash-linear-attention` for fast recurrent kernels on GPU.

```python
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

repo = "ArtinMFar/RecurrentQwen-4b"
model = AutoModelForImageTextToText.from_pretrained(repo, trust_remote_code=True, dtype=torch.bfloat16,
                                                    device_map="cuda")
processor = AutoProcessor.from_pretrained(repo, trust_remote_code=True)

messages = [{"role": "user", "content": [{"type": "text", "text": "Explain why the sky is blue in two sentences."}]}]
inputs = processor.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False, tokenize=True,
                                       return_tensors="pt", return_dict=True).to("cuda")
out = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

Set `enable_thinking=True` for the reasoning mode. Images work as with Qwen3.5: add `{"type": "image", "image": <PIL image,
path or URL>}` to the message content list.

### Keeping a conversation's state (constant TTFT and memory)

The cache (`past_key_values`) holds everything the model remembers: at most 2047 attention tokens per attention layer
plus the fixed recurrent state. Keep it between turns and feed only the new turn, so the time to the first token
depends on the length of the new message, not on the length of the history. (A recurrent state cannot be rolled
back, so append new tokens instead of re-rendering the whole history.)

```python
import torch
from transformers import DynamicCache

cache = DynamicCache(config=model.config)
seen = torch.empty(1, 0, dtype=torch.long, device="cuda")   # every token the cache has consumed (or will)

def chat(user_msg, thinking=False, max_new_tokens=512):
    global seen
    turn = ("\n" if seen.numel() else "") + f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
    turn += "<think>\n" if thinking else "<think>\n\n</think>\n\n"
    new = processor.tokenizer(turn, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
    ids = torch.cat([seen, new], dim=1)
    out = model.generate(input_ids=ids, past_key_values=cache, max_new_tokens=max_new_tokens, do_sample=False)
    seen = out
    return processor.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

print(chat("Hi! My name is Ana and I live in Porto."))
print(chat("What's a good weekend trip from where I live?"))
```

Checked on an 8.5k-token history (4× the window): processing it turn by turn with the cache gives the same next-token
logits as processing it in one pass (max difference 0.005 in fp32, KL 5e-7). In bf16, rounding noise (~0.2 logits)
can flip greedy choices at near-ties, as with any bf16 model. Note that Qwen's chat template drops the thinking text of
earlier turns; appending turns to the cache as above keeps it, which is fine for the model but differs from a
re-rendered history.

## Limitations

- The recurrent state has a fixed size, so recall of exact details from far outside the 2048-token window is lossy
  compared with full attention (see the long-context results above). It is best at keeping the gist and salient
  facts of long histories.
- This is a short adaptation run, not a full pre-training run; quality on some benchmarks is below the original model
  (see the results).
- Custom architecture: works with `transformers` (`trust_remote_code=True`); vLLM, llama.cpp, MLX etc. do not support
  it yet.
- Inherits the biases and limitations of Qwen3.5-4B and of the training data.

## Credits

Base model: [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) by the Qwen team (Apache 2.0). Architecture ideas from
RecurrentGemma/Griffin, Gated DeltaNet, Mamba and Mamba-3. Training data: SmolTalk2 (HuggingFaceTB), PG19,
FineVision (HuggingFaceM4) and the original model's own answers to real prompts.

<details><summary>Raw evaluation results (JSON)</summary>

```json
{
 "original": {
  "model": "original",
  "pg19_ppl": {
   "0k-2k": 12.428,
   "2k-4k": 12.749,
   "4k-8k": 12.131,
   "8k-16k": 11.575,
   "16k-32k": 11.302
  },
  "eval_minutes": 24.3,
  "longbench_e": {
   "hotpotqa_e": {
    "0-4k": 69.4,
    "4-8k": 94.4,
    "8k+": 63.9
   },
   "2wikimqa_e": {
    "0-4k": 65.3,
    "4-8k": 66.7,
    "8k+": 73.3
   },
   "multifieldqa_en_e": {
    "0-4k": 49.6,
    "4-8k": 51.3,
    "8k+": 51.1
   },
   "qasper_e": {
    "0-4k": 64.4,
    "4-8k": 56.3,
    "8k+": 28.8
   },
   "passage_retrieval_en_e": {
    "0-4k": 100.0,
    "4-8k": 100.0,
    "8k+": 100.0
   }
  },
  "arc_challenge": 91.0,
  "chartqa": 70.7,
  "gsm8k_nothink": {
   "acc": 95.0,
   "mean_tokens": 314,
   "over_2048": 0
  },
  "gsm8k_think": {
   "acc": 93.8,
   "mean_tokens": 2114,
   "over_2048": 24
  },
  "bench": {
   "4096": {
    "cache_mb": 177.5,
    "prefill_s": 0.13,
    "ttft_new_turn_ms": 78.0,
    "decode_ms_per_token": 52.33,
    "peak_mem_gb": 9.26
   },
   "16384": {
    "cache_mb": 561.5,
    "prefill_s": 0.51,
    "ttft_new_turn_ms": 74.4,
    "decode_ms_per_token": 59.47,
    "peak_mem_gb": 10.43
   },
   "65536": {
    "cache_mb": 2097.5,
    "prefill_s": 3.11,
    "ttft_new_turn_ms": 114.0,
    "decode_ms_per_token": 55.33,
    "peak_mem_gb": 13.99
   },
   "131072": {
    "cache_mb": 4145.5,
    "prefill_s": 9.33,
    "ttft_new_turn_ms": 178.2,
    "decode_ms_per_token": 55.24,
    "peak_mem_gb": 19.99
   }
  },
  "humaneval": 84.1,
  "math500": 75.0
 },
 "naive": {
  "model": "naive",
  "pg19_ppl": {
   "0k-2k": 12.421,
   "2k-4k": 13.413,
   "4k-8k": 16.907,
   "8k-16k": 24.973,
   "16k-32k": 36.583
  },
  "eval_minutes": 6.2,
  "longbench_e": {
   "hotpotqa_e": {
    "0-4k": 36.1,
    "4-8k": 14.3,
    "8k+": 16.4
   },
   "2wikimqa_e": {
    "0-4k": 27.8,
    "4-8k": 36.7,
    "8k+": 16.7
   },
   "multifieldqa_en_e": {
    "0-4k": 38.3,
    "4-8k": 28.9,
    "8k+": 24.2
   },
   "qasper_e": {
    "0-4k": 33.0,
    "4-8k": 44.9,
    "8k+": 52.2
   },
   "passage_retrieval_en_e": {
    "0-4k": 33.3,
    "4-8k": 33.3,
    "8k+": 66.7
   }
  },
  "arc_challenge": 91.0,
  "chartqa": 71.3,
  "gsm8k_nothink": {
   "acc": 94.2,
   "mean_tokens": 317,
   "over_2048": 0
  },
  "humaneval": 81.1,
  "math500": 75.0,
  "gsm8k_think": {
   "acc": 88.8,
   "mean_tokens": 2719,
   "over_2048": 30
  }
 },
 "final": {
  "model": "final",
  "pg19_ppl": {
   "0k-2k": 12.627,
   "2k-4k": 13.38,
   "4k-8k": 13.089,
   "8k-16k": 12.777,
   "16k-32k": 12.756
  },
  "eval_minutes": 25.9,
  "longbench_e": {
   "hotpotqa_e": {
    "0-4k": 36.1,
    "4-8k": 44.4,
    "8k+": 27.8
   },
   "2wikimqa_e": {
    "0-4k": 37.5,
    "4-8k": 19.4,
    "8k+": 18.9
   },
   "multifieldqa_en_e": {
    "0-4k": 42.2,
    "4-8k": 28.6,
    "8k+": 23.4
   },
   "qasper_e": {
    "0-4k": 29.2,
    "4-8k": 52.2,
    "8k+": 35.6
   },
   "passage_retrieval_en_e": {
    "0-4k": 33.3,
    "4-8k": 33.3,
    "8k+": 66.7
   }
  },
  "arc_challenge": 89.2,
  "chartqa": 60.7,
  "gsm8k_nothink": {
   "acc": 90.8,
   "mean_tokens": 293,
   "over_2048": 0
  },
  "gsm8k_think": {
   "acc": 82.5,
   "mean_tokens": 3263,
   "over_2048": 39
  },
  "bench": {
   "4096": {
    "cache_mb": 125.5,
    "prefill_s": 0.18,
    "ttft_new_turn_ms": 123.6,
    "decode_ms_per_token": 59.18,
    "peak_mem_gb": 9.43
   },
   "16384": {
    "cache_mb": 125.5,
    "prefill_s": 0.77,
    "ttft_new_turn_ms": 80.2,
    "decode_ms_per_token": 59.28,
    "peak_mem_gb": 10.41
   },
   "65536": {
    "cache_mb": 125.5,
    "prefill_s": 2.93,
    "ttft_new_turn_ms": 74.9,
    "decode_ms_per_token": 59.24,
    "peak_mem_gb": 10.42
   },
   "131072": {
    "cache_mb": 125.5,
    "prefill_s": 5.83,
    "ttft_new_turn_ms": 75.5,
    "decode_ms_per_token": 55.68,
    "peak_mem_gb": 10.42
   }
  },
  "cache_consistency": {
   "history_tokens": 8662,
   "cache_mb": 125.5,
   "ttft_ms": {
    "cached": 148.0,
    "full_recompute": 404.4
   },
   "same_tokens": "3/41",
   "cached_answer": "This section details the \"Five Million Loan Election\" of 1858, a controversial constitutional amendment that funded railroad construction, which ultimately failed due to financial mismanagement and political corruption, leading to a",
   "fresh_answer": "This section describes the vibrant atmosphere of St. Paul's levee during the steamboat era, the intense political controversy surrounding the \"Five Million Loan Election\" and its disastrous financial consequences, and the brief,"
  },
  "humaneval": 75.0,
  "math500": 71.0
 }
}
```

</details>
