"""Render `sft_format.conversation` dicts into model inputs with loss on the trainable assistant turns only.

Rendering is the evaluation's: the model's chat template with the tool list patched to bare function objects (as the
Tinker renderer and `hf_jobs/eval_job.sh` do), thinking off. Qwen's template renders earlier assistant turns without
the empty `<think>\\n\\n</think>\\n\\n` block that the generation prompt puts before the turn being sampled, so every
TRAINED turn gets that block inserted in front of it (as context, not as a target): each trained turn is then
predicted from exactly the tokens it follows at inference. Context turns that are not trained keep their history
form. Targets are the turn's text and tool calls through `<|im_end|>`.
"""

from __future__ import annotations

import io
from typing import Any

import torch
from PIL import Image

THINK = "<think>\n\n</think>\n\n"
ASSISTANT = "<|im_start|>assistant\n"
TOOL_LISTING = "{{- tool | tojson }}"


def patched_template(processor) -> str:
    """The chat template as served in evaluation: tools listed as bare function objects (the Tinker renderer's way)."""
    t = processor.chat_template if isinstance(processor.chat_template, str) else processor.tokenizer.chat_template
    if TOOL_LISTING not in t:
        raise ValueError("chat template changed upstream: re-check the tool listing against the Tinker renderer")
    return t.replace(TOOL_LISTING, "{{- (tool.function if tool.function is defined else tool) | tojson }}")


def render_text(conv: dict, processor, template: str) -> str:
    """The full training text, with the think block inserted before each trained assistant turn."""
    text = processor.apply_chat_template(conv["messages"], tools=conv["tools"], chat_template=template,
                                         tokenize=False, enable_thinking=False)
    trained = [f for m, f in zip(conv["messages"], conv["trainable"], strict=True) if m["role"] == "assistant"]
    out, pos = [], 0
    for k in range(len(trained)):
        i = text.index(ASSISTANT, pos) + len(ASSISTANT)
        out.append(text[pos:i])
        pos = i
        if trained[k] and not text.startswith(THINK, pos):
            out.append(THINK)
    out.append(text[pos:])
    if text.count(ASSISTANT) != len(trained):
        raise ValueError("assistant turns in the rendered text do not match the messages")
    return "".join(out)


class Collator:
    """Batch `sft_format.conversation` dicts into input_ids / attention_mask / labels / pixel_values / image_grid_thw."""

    def __init__(self, processor, max_length: int | None = None):
        self.processor = processor
        self.template = patched_template(processor)
        tok = processor.tokenizer
        self.im_start = tok.convert_tokens_to_ids("<|im_start|>")
        self.im_end = tok.convert_tokens_to_ids("<|im_end|>")
        self.assistant_ids = tok.encode("assistant\n", add_special_tokens=False)
        self.think_ids = tok.encode(THINK, add_special_tokens=False)
        self.max_length = max_length

    def labels_for(self, ids: list[int], trainable: list[bool]) -> list[int]:
        labels = [-100] * len(ids)
        k = 0
        n = len(self.assistant_ids)
        for i, t in enumerate(ids):
            if t != self.im_start or ids[i + 1:i + 1 + n] != self.assistant_ids:
                continue
            start = i + 1 + n
            if ids[start:start + len(self.think_ids)] == self.think_ids:
                start += len(self.think_ids)
            end = ids.index(self.im_end, start)
            if trainable[k]:
                labels[start:end + 1] = ids[start:end + 1]
            k += 1
        if k != len(trainable):
            raise ValueError(f"found {k} assistant turns in the tokens, expected {len(trainable)}")
        return labels

    def __call__(self, examples: list[dict]) -> dict[str, Any]:
        texts, images = [], []
        for ex in examples:
            texts.append(render_text(ex, self.processor, self.template))
            images.append([Image.open(io.BytesIO(b)).convert("RGB") for b in ex["images"]] or None)
        flat = [im for ims in images if ims for im in ims]
        batch = self.processor(text=texts, images=flat or None, return_tensors="pt", padding=True)
        labels = []
        for row, ex in zip(batch["input_ids"].tolist(), examples, strict=True):  # pads hold no <|im_start|>: stay -100
            assistant_flags = [f for m, f in zip(ex["messages"], ex["trainable"], strict=True) if m["role"] == "assistant"]
            labels.append(self.labels_for(row, assistant_flags))
        batch["labels"] = torch.tensor(labels)
        if self.max_length and batch["input_ids"].shape[1] > self.max_length:
            raise ValueError(f"sequence of {batch['input_ids'].shape[1]} tokens exceeds max_length {self.max_length}")
        return dict(batch)
