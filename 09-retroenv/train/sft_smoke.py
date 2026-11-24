#!/usr/bin/env python3
"""Supervised warm start on RetroEnv trajectories from dataset/build_sft.py.

Loss covers assistant turns only. Rows are tokenized here rather than by
SFTTrainer. Qwen3.5's chat template renders tool-call arguments as XML
parameters, so it needs them as objects, but the dataset stores them as JSON
strings: decoding them into an Arrow column would merge different tools'
arguments into one struct and render the null-filled gaps as parameters. Each
row is decoded in Python, and only token IDs and the assistant mask reach
Arrow; SFTTrainer turns that mask into labels.

    uv run --with 'trl>=1.12,<1.13' --with datasets --with peft \\
      python train/sft_smoke.py

Data comes from the Hub (``SFT_DATA=AdithyaSK/RetroEnv-SFT``, ``SFT_CONFIG=chemist`` or
``plain``) or from a local ``dataset/build_sft.py`` export directory. The merged
checkpoint lands in ``$OUTPUT_DIR/merged``; continue with
``MODEL=$OUTPUT_DIR/merged python train/grpo_smoke.py``, and evaluate on the v3 dev split
with ``eval/run_eval.py --provider custom``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from trl.chat_template_utils import get_training_chat_template

MODEL = os.getenv("MODEL", "Qwen/Qwen3.5-4B")
# A Hub dataset ID, or a local build_sft.py export directory.
SFT_DATA = os.getenv("SFT_DATA", "AdithyaSK/RetroEnv-SFT")
SFT_CONFIG = os.getenv("SFT_CONFIG", "chemist")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs/retroenv-sft-smoke")
MAX_STEPS = int(os.getenv("MAX_STEPS", "200"))
MAX_ROWS = int(os.getenv("MAX_ROWS", "0"))  # 0 keeps every row
EVAL_ROWS = int(os.getenv("EVAL_ROWS", "64"))  # a fixed slice keeps periodic evaluation cheap
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "8192"))
SEED = int(os.getenv("SEED", "17"))


def decode(row: dict) -> tuple[list[dict], list[dict]]:
    """Messages and tools as objects. Arrow fills absent message keys with None; drop them."""
    messages = []
    for message in row["messages"]:
        message = {key: value for key, value in message.items() if value is not None}
        if message.get("tool_calls"):
            message["tool_calls"] = [
                {**call, "function": {**call["function"], "arguments": json.loads(call["function"]["arguments"])}}
                for call in message["tool_calls"]
            ]
        messages.append(message)
    return messages, json.loads(row["tools"])


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    # The same template with {% generation %} markers around assistant turns.
    template = get_training_chat_template(tokenizer)

    def tokenize(row: dict) -> dict:
        messages, tools = decode(row)
        encoded = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            chat_template=template,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
        return {"input_ids": encoded["input_ids"], "assistant_masks": encoded["assistant_masks"]}

    local = Path(SFT_DATA)
    if local.is_dir():
        files = {"train": str(local / "train.jsonl")}
        if (local / "validation.jsonl").exists() and (local / "validation.jsonl").stat().st_size:
            files["validation"] = str(local / "validation.jsonl")
        raw = load_dataset("json", data_files=files)
    else:
        raw = load_dataset(SFT_DATA, SFT_CONFIG)
    if MAX_ROWS:
        raw["train"] = raw["train"].shuffle(seed=SEED).select(range(min(MAX_ROWS, len(raw["train"]))))
    if "validation" in raw:
        raw["validation"] = raw["validation"].select(range(min(EVAL_ROWS, len(raw["validation"]))))
    data = {name: split.map(tokenize, remove_columns=split.column_names) for name, split in raw.items()}
    lengths = sorted(len(ids) for ids in data["train"]["input_ids"])
    print(f"{len(lengths)} train rows; tokens p50={lengths[len(lengths) // 2]} max={lengths[-1]}")

    cuda = torch.cuda.is_available()
    # Loaded here rather than by name: SFTTrainer's own loader segfaulted on macOS.
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16 if cuda else torch.float32)
    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=OUTPUT_DIR,
            max_steps=MAX_STEPS,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=int(os.getenv("GRAD_ACCUM", "8")),
            learning_rate=float(os.getenv("LEARNING_RATE", "1e-4")),
            lr_scheduler_type="cosine",
            warmup_steps=0.05,  # a float below 1 is a ratio of total steps
            max_length=MAX_LENGTH,
            bf16=cuda,
            gradient_checkpointing=cuda,
            logging_steps=1,
            eval_strategy="steps" if "validation" in data else "no",
            eval_steps=max(1, MAX_STEPS // 4),
            save_strategy="no",
            report_to=os.getenv("REPORT_TO", "none"),
            seed=SEED,
        ),
        train_dataset=data["train"],
        eval_dataset=data.get("validation"),
        processing_class=tokenizer,
        peft_config=LoraConfig(r=32, lora_alpha=64, target_modules="all-linear", task_type="CAUSAL_LM"),
    )
    trainer.train()
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(f"{OUTPUT_DIR}/merged")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/merged")


if __name__ == "__main__":
    main()
