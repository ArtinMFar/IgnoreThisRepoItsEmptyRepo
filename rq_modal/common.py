"""Shared Modal resources for the RecurrentQwen pipeline."""

import modal

BASE_MODEL = "Qwen/Qwen3.5-4B"
VOL_PATH = "/vol"
HF_HOME = f"{VOL_PATH}/hf"
DATA_DIR = f"{VOL_PATH}/data"
CKPT_DIR = f"{VOL_PATH}/ckpt"
OUT_DIR = f"{VOL_PATH}/out"

volume = modal.Volume.from_name("recurrentqwen", create_if_missing=True)
hf_secret = modal.Secret.from_name("hf-recurrentqwen")

ENV = {"HF_HOME": HF_HOME, "HF_HUB_ENABLE_HF_TRANSFER": "0", "TOKENIZERS_PARALLELISM": "false"}

TRANSFORMERS = "transformers==5.20.1"

cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        TRANSFORMERS, "datasets==5.1.0", "huggingface_hub[hf_xet]", "numpy", "pillow", "pyarrow", "jinja2"
    )
    .env(ENV)
    .add_local_python_source("rq_modal", "recurrent_qwen")
)

gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install(
        "torch==2.13.0", "torchvision==0.28.0", extra_index_url="https://download.pytorch.org/whl/cu130",
        extra_options="--index-strategy unsafe-best-match",
    )
    .uv_pip_install(
        TRANSFORMERS, "flash-linear-attention==0.5.2", "datasets==5.1.0", "huggingface_hub[hf_xet]", "numpy",
        "pillow", "pyarrow", "jinja2", "accelerate", "safetensors",
    )
    .env(ENV)
    .add_local_python_source("rq_modal", "recurrent_qwen")
)

# FlashInfer (used by vLLM) JIT-compiles kernels, so this image needs the CUDA toolkit (13.0 matches the host driver)
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:13.0.2-devel-ubuntu24.04", add_python="3.12")
    .uv_pip_install("vllm==0.31.0", "huggingface_hub[hf_xet]", "pillow", "pyarrow")
    .env(ENV)
    .add_local_python_source("rq_modal")
)

# Special tokens of the Qwen3.5 tokenizer
IM_START, IM_END, ASSISTANT, NEWLINE = 248045, 248046, 74455, 198
IMAGE_PAD, VISION_START, VISION_END = 248056, 248053, 248054
