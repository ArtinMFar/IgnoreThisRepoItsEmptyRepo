"""Turn stored samples into model inputs plus loss masks (shared by the trainer and the evals)."""

import base64
import io
import json

import torch

from rq_modal.common import ASSISTANT, IM_END, IM_START, IMAGE_PAD, NEWLINE


def assistant_mask(ids: torch.Tensor) -> torch.Tensor:
    """1 for tokens inside assistant turns (content and closing <|im_end|>)."""
    ids = ids.tolist()
    mask, inside = [0] * len(ids), False
    for i, t in enumerate(ids):
        if inside:
            mask[i] = 1
            if t == IM_END:
                inside = False
        elif t == NEWLINE and i >= 2 and ids[i - 1] == ASSISTANT and ids[i - 2] == IM_START:
            inside = True
    return torch.tensor(mask, dtype=torch.bool)


def text_batch(ids, mask):
    ids = torch.as_tensor(ids, dtype=torch.long)[None]
    return dict(inputs=dict(input_ids=ids), ce_mask=torch.as_tensor(mask, dtype=torch.bool)[None],
                kl_mask=torch.ones_like(ids, dtype=torch.bool))


def _image(b):
    from PIL import Image

    return Image.open(io.BytesIO(b)).convert("RGB")


def vision_messages(sample):
    """(messages, images, thinking) for a FineVision row or an on-policy image record."""
    if "messages" in sample:
        return json.loads(sample["messages"]), [_image(b) for b in sample["images"]], False
    imgs = [_image(base64.b64decode(b)) for b in sample["images"]]
    content = [{"type": "image"} for _ in imgs] + [{"type": "text", "text": sample["question"]}]
    return ([{"role": "user", "content": content},
             {"role": "assistant", "content": [{"type": "text", "text": sample["answer"]}]}], imgs, sample["thinking"])


def vision_batch(processor, samples):
    """Several image conversations packed into one sequence (images in order of appearance)."""
    texts, images = [], []
    for s in samples:
        messages, imgs, thinking = vision_messages(s)
        texts.append(processor.apply_chat_template(messages, tokenize=False, enable_thinking=thinking))
        images += imgs
    enc = processor(text=["".join(texts)], images=images, return_tensors="pt", return_mm_token_type_ids=True)
    ids = enc["input_ids"]
    inputs = {k: v for k, v in enc.items() if k != "attention_mask"}
    not_image = ids != IMAGE_PAD
    return dict(inputs=inputs, ce_mask=assistant_mask(ids[0])[None] & not_image, kl_mask=not_image)
