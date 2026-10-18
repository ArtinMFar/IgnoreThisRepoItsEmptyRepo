"""Package the trained checkpoint (weights + custom code + processor + model card) and push it to the Hub.

  modal run -m rq_modal.upload::main --repo-name RecurrentQwen-4b
"""

import json
import os
import shutil

import modal

from rq_modal.common import BASE_MODEL, CKPT_DIR, OUT_DIR, VOL_PATH, cpu_image, hf_secret, volume

app = modal.App("rq-upload")
image = cpu_image.uv_pip_install("torch==2.13.0", "torchvision==0.28.0",
                                 extra_index_url="https://download.pytorch.org/whl/cpu",
                                 extra_options="--index-strategy unsafe-best-match").add_local_dir(
    "hub", remote_path="/root/hub")

PROCESSOR_FILES = ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "preprocessor_config.json",
                   "video_preprocessor_config.json", "vocab.json", "merges.txt", "LICENSE"]


@app.function(image=image, volumes={VOL_PATH: volume}, secrets=[hf_secret], cpu=8, memory=65536, timeout=3600)
def upload(repo_name: str, ckpt: str = "final", private: bool = False):
    import torch
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoModelForImageTextToText, AutoProcessor

    volume.reload()
    src = f"{CKPT_DIR}/{ckpt}"
    dst = "/tmp/release"
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("train_state.json"))
    base = snapshot_download(BASE_MODEL, allow_patterns=PROCESSOR_FILES)
    for f in PROCESSOR_FILES:
        if os.path.exists(f"{base}/{f}"):
            shutil.copy(f"{base}/{f}", dst)
    import recurrent_qwen

    pkg = os.path.dirname(recurrent_qwen.__file__)
    for f in ("configuration_recurrent_qwen.py", "modeling_recurrent_qwen.py"):
        shutil.copy(f"{pkg}/{f}", dst)

    cfg = json.load(open(f"{dst}/config.json"))
    cfg["auto_map"] = {
        "AutoConfig": "configuration_recurrent_qwen.RecurrentQwenConfig",
        "AutoModel": "modeling_recurrent_qwen.RecurrentQwenModel",
        "AutoModelForImageTextToText": "modeling_recurrent_qwen.RecurrentQwenForConditionalGeneration",
        "AutoModelForCausalLM": "modeling_recurrent_qwen.RecurrentQwenForConditionalGeneration",
    }
    cfg["architectures"] = ["RecurrentQwenForConditionalGeneration"]
    json.dump(cfg, open(f"{dst}/config.json", "w"), indent=2)

    # model card with the evaluation results filled in
    results = {}
    for name in ("final", "original", "naive"):
        p = f"{OUT_DIR}/eval_{name}.json"
        if os.path.exists(p):
            results[name] = json.load(open(p))
    card = open("/root/hub/README.md").read()
    card = card.replace("{{EVAL_JSON}}", json.dumps(results, indent=1))
    open(f"{dst}/README.md", "w").write(card)
    if os.path.exists("/root/hub/eval_tables.md"):
        shutil.copy("/root/hub/eval_tables.md", dst)

    # sanity check: the released folder loads through the Auto classes with remote code and runs
    model = AutoModelForImageTextToText.from_pretrained(dst, trust_remote_code=True, dtype=torch.bfloat16)
    proc = AutoProcessor.from_pretrained(dst, trust_remote_code=True)
    msgs = [{"role": "user", "content": "Say hi in three words."}]
    ids = proc.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, return_tensors="pt",
                                   return_dict=True)["input_ids"]
    out = model.generate(input_ids=ids, max_new_tokens=12, do_sample=False)
    print("sanity generation:", repr(proc.decode(out[0, ids.shape[1]:], skip_special_tokens=True)), flush=True)
    del model

    api = HfApi()
    user = api.whoami()["name"]
    repo_id = f"{user}/{repo_name}"
    api.create_repo(repo_id, private=private, exist_ok=True, repo_type="model")
    api.upload_folder(folder_path=dst, repo_id=repo_id, commit_message="Upload RecurrentQwen-4B")
    print("uploaded to", f"https://huggingface.co/{repo_id}", flush=True)
    return repo_id


@app.local_entrypoint()
def main(repo_name: str = "RecurrentQwen-4b", ckpt: str = "final"):
    print(upload.remote(repo_name, ckpt))
