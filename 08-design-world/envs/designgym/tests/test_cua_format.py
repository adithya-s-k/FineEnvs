"""The cua-v1 dataset format: standardisation, the SFT windows, and the Hub folder (needs a recorded cua.jsonl)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from designgym.cua_format import (
    PLACEHOLDER,
    SCHEMA_VERSION,
    Window,
    export_hf,
    standardize,
    windows,
)

PILOT = Path(__file__).resolve().parents[3] / "artifacts" / "cua-pilot-v2" / "cua.jsonl"
needs_pilot = pytest.mark.skipif(not PILOT.exists(), reason="no recorded pilot trajectories")


def _images(msgs):
    return [p for m in msgs for p in m["content"] if p["type"] == "image"]


@pytest.fixture(scope="module")
def rows():
    lines = [json.loads(line) for line in PILOT.read_text().splitlines()[:4]]
    return [standardize(t, PILOT.parent) for t in lines]


@needs_pilot
def test_images_line_up_with_placeholders_and_masks_survive(rows):
    for row in rows:
        assert row["schema_version"] == SCHEMA_VERSION and row["coordinates"] == "pixels"
        assert len(_images(row["messages"])) == len(row["images"]) > 0
        assert row["screen"] == {"width": 1280, "height": 800}
        trainable = [m for m in row["messages"] if m["trainable"]]
        assert all(m["role"] == "assistant" for m in trainable)
        masked = sum(1 for m in row["messages"] if m["role"] == "assistant" and not m["trainable"])
        assert masked == row["masked_turns"]
        for m in row["messages"]:
            for c in m["tool_calls"]:
                json.loads(c["function"]["arguments"])  # arguments are JSON strings


@needs_pilot
def test_windows_cover_each_trainable_turn_once_and_keep_the_target(rows):
    w = Window(max_images=8, keep_images=3)
    for row in rows:
        target = row["images"][: sum(1 for p in row["messages"][1]["content"] if p["type"] == "image")]
        trained = []
        for win in windows(row, w):
            assert len(_images(win["messages"])) == len(win["images"])
            assert win["images"][: len(target)] == target  # the target is never dropped
            # every trained turn saw at most protected + max_images screenshots
            for j, m in enumerate(win["messages"]):
                if m["trainable"]:
                    seen = len(_images(win["messages"][:j]))
                    assert seen <= len(target) + w.max_images
                    trained.append(sum(1 for x in win["messages"][: j + 1] if x["role"] == "assistant"))
            assert win["messages"][-1]["role"] == "assistant" and win["messages"][-1]["trainable"]
        expected = [k + 1 for k, m in enumerate([m for m in row["messages"] if m["role"] == "assistant"]) if m["trainable"]]
        assert sorted(trained) == expected


@needs_pilot
def test_dropped_screenshots_become_placeholders(rows):
    long = max(rows, key=lambda r: len(r["images"]))
    last = list(windows(long, Window(4, 2)))[-1]
    texts = [p["text"] for m in last["messages"] for p in m["content"] if p["type"] == "text"]
    assert texts.count(PLACEHOLDER) == len(long["images"]) - len(last["images"]) - sum(
        1 for m in long["messages"][len(last["messages"]):] for p in m["content"] if p["type"] == "image")


@needs_pilot
def test_export_writes_a_loadable_hub_folder(tmp_path):
    from datasets import load_dataset

    small = tmp_path / "cua.jsonl"
    small.write_text("\n".join(PILOT.read_text().splitlines()[:2]) + "\n")
    for d in {Path(json.loads(line)["images"][0]).parent for line in small.read_text().splitlines()}:
        (tmp_path / d).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / d).symlink_to(PILOT.parent / d)
    summary = export_hf(small, tmp_path / "hf", with_windows=True)
    assert summary["trajectories"] == 2 and summary["windows"] > 2
    card = (tmp_path / "hf" / "README.md").read_text()
    assert "config_name: trajectories" in card and "config_name: windows" in card
    ds = load_dataset("parquet", data_dir=str(tmp_path / "hf" / "data" / "trajectories"), split="train")
    row = ds[0]
    assert row["schema_version"] == SCHEMA_VERSION and len(row["images"]) == len(_images(row["messages"]))
    assert row["images"][1].size == (1280, 800)  # images[0] is the target; then screenshots, as PIL images
