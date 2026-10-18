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

{{RESULTS_SUMMARY}}

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
{{DATA_TABLE}}
- **Compute:** {{COMPUTE}}

## Results

{{RESULTS_TABLES}}

## Usage

Requires `transformers>=5.20.1` (the model code builds on its Qwen3.5 implementation) and `trust_remote_code=True`.
Install `flash-linear-attention` for fast recurrent kernels on GPU.

```python
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

repo = "{{REPO_ID}}"
model = AutoModelForImageTextToText.from_pretrained(repo, trust_remote_code=True, dtype=torch.bfloat16,
                                                    device_map="cuda")
processor = AutoProcessor.from_pretrained(repo, trust_remote_code=True)

messages = [{"role": "user", "content": "Explain why the sky is blue in two sentences."}]
inputs = processor.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False,
                                       return_tensors="pt", return_dict=True).to("cuda")
out = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

Set `enable_thinking=True` for the reasoning mode. Images work as with Qwen3.5 (`{"type": "image", "image": ...}` in the
message content).

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
{{EVAL_JSON}}
```

</details>
