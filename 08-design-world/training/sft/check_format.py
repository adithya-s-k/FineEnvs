"""Offline format checks for DesignGym SFT: every trained turn must be what evaluation would show and expect.

    .venv/bin/python check_format.py /path/to/data   # a folder with <config>/*.parquet (or data/<config>/...)

For sampled windows of every config it asserts:
1. each trained turn's context, with the think blocks inserted before EARLIER trained turns removed, equals the
   vLLM generation prompt for that turn (same patched template, thinking off), and its labelled tokens equal the
   template's rendering of the turn through <|im_end|>;
2. only trainable assistant turns carry labels (masked slips and context turns carry none);
3. the image placeholders in the text match the images, and the processor yields one grid per image;
4. GUI rows: every coordinate is on the 0-1000 grid and the system prompt states the 0-1000 grid.
"""

from __future__ import annotations

import io
import json
import random
import sys
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image
from transformers import AutoProcessor

import sft_format
from collator import ASSISTANT, THINK, Collator, render_text

MODEL = "Qwen/Qwen3.6-35B-A3B"


def strip_thinks(s: str) -> str:
    return s.replace(ASSISTANT + THINK, ASSISTANT)


def check(conv: dict, processor, collator: Collator) -> dict:
    tmpl = collator.template
    msgs, tools = conv["messages"], conv["tools"]
    text = render_text(conv, processor, tmpl)
    stats = {"trained_turns": 0, "history_think_diffs": 0}
    k = -1
    pos = -1
    for i, (m, flag) in enumerate(zip(msgs, conv["trainable"], strict=True)):
        if m["role"] != "assistant":
            continue
        k += 1
        pos = text.index(ASSISTANT, pos + 1)
        if not flag:
            continue
        prompt = processor.apply_chat_template(msgs[:i], tools=tools, chat_template=tmpl, tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False)
        assert prompt.endswith(ASSISTANT + THINK), "generation prompt does not end with the empty think block"
        after = text[pos + len(ASSISTANT):]
        assert after.startswith(THINK), f"turn {i}: trained turn not preceded by the think block"
        hist_train, hist_infer = text[:pos], prompt[: -len(ASSISTANT + THINK)]
        assert strip_thinks(hist_train) == strip_thinks(hist_infer), f"turn {i}: context differs from the inference prompt"
        stats["history_think_diffs"] += hist_train.count(ASSISTANT + THINK) - hist_infer.count(ASSISTANT + THINK)
        seg = processor.apply_chat_template(msgs[: i + 1], tools=tools, chat_template=tmpl, tokenize=False,
                                            enable_thinking=False).rsplit(ASSISTANT, 1)[1]
        seg = seg.removeprefix(THINK)
        target = seg[: seg.index("<|im_end|>") + len("<|im_end|>")]
        assert after[len(THINK):].startswith(target), f"turn {i}: target differs from the template's rendering"
        stats["trained_turns"] += 1
    batch = collator([conv])
    ids, labels = batch["input_ids"][0].tolist(), batch["labels"][0].tolist()
    tok = processor.tokenizer
    labelled = tok.decode([t for t, l in zip(ids, labels, strict=True) if l != -100])
    expect = "".join(
        processor.apply_chat_template(msgs[: i + 1], tools=tools, chat_template=collator.template, tokenize=False,
                                      enable_thinking=False).rsplit(ASSISTANT, 1)[1].removeprefix(THINK).split("<|im_end|>")[0] + "<|im_end|>"
        for i, (m, f) in enumerate(zip(msgs, conv["trainable"], strict=True)) if m["role"] == "assistant" and f)
    assert labelled == expect, "labels are not exactly the trainable turns"
    n_img = text.count("<|vision_start|>")
    assert n_img == len(conv["images"]), f"{n_img} image slots for {len(conv['images'])} images"
    if conv["images"]:
        assert batch["image_grid_thw"].shape[0] == len(conv["images"])
    if conv["meta"]["surface"] == "gui":
        system = msgs[0]["content"]
        assert sft_format.NORMALIZED in system and "pixels of the" not in system, "GUI system prompt not on the 0-1000 grid"
        for m in msgs:
            for c in m.get("tool_calls") or []:
                for key in ("coordinate", "start_coordinate"):
                    v = c["function"]["arguments"].get(key)
                    if v is not None:
                        assert all(0 <= x <= 1000 for x in v), f"{key} {v} off the grid"
    stats.update(tokens=len(ids), labelled_tokens=sum(1 for x in labels if x != -100), images=len(conv["images"]))
    return stats


def main(root: str, per_config: int = 3) -> None:
    root = Path(root)
    base = root / "data" if (root / "data").exists() else root
    processor = AutoProcessor.from_pretrained(MODEL)
    collator = Collator(processor)
    rng = random.Random(0)
    total = {}
    for cfg_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        files = sorted(cfg_dir.glob("*.parquet"))
        if not files:
            continue
        table = pq.read_table(rng.choice(files))
        rows = table.slice(0, min(per_config, table.num_rows)).to_pylist()
        n = toks = lab = turns = 0
        for row in rows:
            for w in sft_format.windows(row):
                conv = sft_format.conversation(w)
                s = check(conv, processor, collator)
                n += 1
                toks = max(toks, s["tokens"])
                lab += s["labelled_tokens"]
                turns += s["trained_turns"]
        total[cfg_dir.name] = n
        print(f"PASS {cfg_dir.name:13s} {len(rows)} rows -> {n} windows, {turns} trained turns, max {toks} tokens/window, "
              f"{lab} labelled tokens", flush=True)
    print(json.dumps(total))


if __name__ == "__main__":
    main(*sys.argv[1:2], *(int(a) for a in sys.argv[2:3]))
