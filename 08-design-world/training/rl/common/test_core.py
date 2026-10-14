"""Tests for the shared rollout core. Offline tests need only the model's tokenizer/processor files (no weights):

    python -m pytest rl/common/test_core.py -q                      # from training/
    DESIGNGYM_URL=https://fineenvs-designgym.hf.space python -m pytest rl/common/test_core.py -q -k live

The live test plays scripted episodes (a fake policy whose turns are hand-written token ids) against a DesignGym
server and checks the chunks: every chunk's context is exactly what the fake engine was sent, loss covers exactly the
sampled tokens, images match vision spans, and the reward comes from the env's submit.
"""

from __future__ import annotations

import asyncio
import base64
import io
import os

import pytest
from PIL import Image

from .episode import EpisodeConfig, GenOut, Trajectory, run_episode
from .qwen_chat import PLACEHOLDER, QwenChat, Seg, parse_tool_calls

MODELS = os.environ.get("TEST_MODELS", "Qwen/Qwen3.6-35B-A3B,Qwen/Qwen3.8-27B").split(",")
TOOLS = [
    {"name": "left_click", "description": "click", "parameters": {"type": "object", "properties": {
        "coordinate": {"type": "array", "items": {"type": "integer"}},
        "text": {"anyOf": [{"type": "string"}, {"type": "null"}]}}, "required": ["coordinate"]}},
    {"name": "type", "description": "type", "parameters": {"type": "object", "properties": {
        "text": {"type": "string"}}, "required": ["text"]}},
    {"name": "key", "description": "key", "parameters": {"type": "object", "properties": {
        "keys": {"anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]}}}},
    {"name": "screenshot", "description": "s", "parameters": {"type": "object", "properties": {}}},
    {"name": "submit", "description": "s", "parameters": {"type": "object", "properties": {}}},
]


def _png(w=1280, h=800, color=(200, 30, 30)) -> str:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture(scope="module", params=MODELS)
def chat(request):
    return QwenChat(request.param)


def test_parse_tool_calls():
    text = ("I'll click.\n\n<tool_call>\n<function=left_click>\n<parameter=coordinate>\n[120, 340]\n</parameter>\n"
            "</function>\n</tool_call>\n<tool_call>\n<function=type>\n<parameter=text>\n120\n</parameter>\n"
            "</function>\n</tool_call>\n<tool_call>\n<function=key>\n<parameter=keys>\nctrl+z\n</parameter>\n"
            "</function>\n</tool_call>")
    content, calls = parse_tool_calls(text, TOOLS)
    assert content == "I'll click."
    assert [(c.name, c.args) for c in calls] == [("left_click", {"coordinate": [120, 340]}), ("type", {"text": "120"}),
                                                 ("key", {"keys": "ctrl+z"})]
    _, calls = parse_tool_calls("<tool_call>\n<function=nope>\n</function>\n</tool_call>", TOOLS)
    assert calls[0].parse_error
    _, calls = parse_tool_calls("no calls here", TOOLS)
    assert calls == []
    # a call cut off by the token limit still parses what is there
    _, calls = parse_tool_calls("<tool_call>\n<function=left_click>\n<parameter=coordinate>\n[1, 2]\n", TOOLS)
    assert calls[0].args == {"coordinate": [1, 2]}


def test_image_tokens_match_processor(chat):
    """N image pads per image equal what the HF processor expands (what the trainer and the engine compute)."""
    item = chat.image_item("image/png", _png())
    assert item.grid_thw == (1, 50, 80) and item.n_tokens == 1000  # 1280x800 at 32 px per merged token
    target = chat.image_item("image/png", _png(419, 298))
    out = chat.image_processor(images=[target.pil], return_tensors="np")
    assert target.n_tokens == int(out["image_grid_thw"][0].prod()) // 4


def test_segments_match_template(chat):
    """Opening + one turn + observation, assembled from segments, equals the chat template's own rendering of the
    same conversation (with the turn sampled after the think-off generation prompt)."""
    ref, shot, shot2 = (chat.image_item("image/png", _png(color=c)) for c in ((1, 2, 3), (4, 5, 6), (7, 8, 9)))
    segs = chat.opening("SYS", TOOLS, [ref, shot], "This is the target design. Recreate it.")
    turn = "I click.\n\n<tool_call>\n<function=left_click>\n<parameter=coordinate>\n[10, 20]\n</parameter>\n" \
           "</function>\n</tool_call>"
    gen = chat.encode(turn) + [chat.im_end]
    segs.append(Seg("gen", gen, logprobs=[0.0] * len(gen), turn=1))
    segs += chat.observation([("left_click", "clicked")], [shot2], "Image returned by left_click.", turn=1)
    traj = Trajectory(chat, 8, 3)
    traj.extend(segs)
    got = chat.decode(traj.assemble().tokens)
    msgs = [{"role": "system", "content": "SYS"},
            {"role": "user", "content": [{"type": "image"}, {"type": "image"},
                                         {"type": "text", "text": "This is the target design. Recreate it."}]},
            {"role": "assistant", "content": "I click.", "tool_calls": [
                {"type": "function", "function": {"name": "left_click", "arguments": {"coordinate": [10, 20]}}}]},
            {"role": "tool", "content": "clicked"},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Image returned by left_click."}]}]
    want = chat.render(msgs, tools=[{"type": "function", "function": t} for t in TOOLS], generation_prompt=True)
    # Token-in/token-out keeps the sampled turn after its generation prompt's empty think block; the template drops
    # it from turns before the last user query (Qwen3.6) and keeps it (Qwen3.8, preserve_thinking).
    head, tail = want.split("<|im_start|>assistant\n", 1)
    if not tail.startswith("<think>"):
        want = head + "<|im_start|>assistant\n<think>\n\n</think>\n\n" + tail
    pad = "<|image_pad|>"
    expanded = want
    for item in (ref, shot, shot2):
        expanded = expanded.replace("<|vision_start|>" + pad + "<|vision_end|>",
                                    "<|vision_start|>" + "\0" * item.n_tokens + "<|vision_end|>", 1)
    assert got == expanded.replace("\0", pad)
    # ...and the token ids are the tokenizer's own for that text
    assert traj.assemble().tokens == chat.encode(expanded.replace("\0", pad))


def test_window_chunks(chat):
    """9 screenshots with window 8,3: chunk 1 holds turns 1-8 with all images; chunk 2 starts from the windowed
    history (6 placeholders, 3 images + target) and trains only on the turns after the drop."""
    ref = chat.image_item("image/png", _png(419, 298))
    traj = Trajectory(chat, 8, 3)
    traj.extend(chat.opening("SYS", TOOLS, [ref], "go"))
    gen = chat.encode("<tool_call>\n<function=screenshot>\n</function>\n</tool_call>") + [chat.im_end]
    shot = chat.image_item("image/png", _png())
    for turn in range(1, 12):
        traj.extend([Seg("gen", gen, logprobs=[-0.5] * len(gen), turn=turn)])
        traj.extend(chat.observation([("screenshot", "(image)")], [shot], "Image returned by screenshot.", turn))
        traj.apply_window()
    traj.close_chunk()
    assert len(traj.chunks) == 2
    c1, c2 = traj.chunks
    assert len(c1.images) == 1 + 8 and c1.n_trained == 9 * len(gen)
    assert len(c2.images) == 1 + 4 and c2.n_trained == 2 * len(gen)  # 3 kept + turn 10; turn 11 obs is cut
    text2 = chat.decode(c2.tokens)
    assert text2.count(PLACEHOLDER) == 6
    assert text2.count("<|vision_start|>") == len(c2.images)
    for c in (c1, c2):
        assert len(c.tokens) == len(c.loss_mask) == len(c.logprobs)
        assert all(m == 0 for m in c.loss_mask[:c.prompt_len]) and c.loss_mask[c.prompt_len] == 1
        assert sum(c.tokens[i] == chat.image_pad for i in range(len(c.tokens))) == sum(i.n_tokens for i in c.images)
        assert all(lp == -0.5 for lp, mk in zip(c.logprobs, c.loss_mask) if mk)


def test_chunk_cap(chat):
    """max_chunks=2: the drop that would open chunk 3 caps the episode instead; chunk_bound matches the window."""
    from .episode import EpisodeConfig

    assert EpisodeConfig(max_turns=20, max_images=8, keep_images=3).chunk_bound() == 4
    assert EpisodeConfig(max_turns=12, max_images=8, keep_images=3).chunk_bound() == 2
    assert EpisodeConfig(max_turns=8, max_images=8, keep_images=3).chunk_bound() == 1
    traj = Trajectory(chat, 4, 2, max_chunks=2)
    traj.extend(chat.opening("SYS", TOOLS, [], "go"))
    gen = chat.encode("<tool_call>\n<function=screenshot>\n</function>\n</tool_call>") + [chat.im_end]
    shot = chat.image_item("image/png", _png(64, 64))
    turns = 0
    for turn in range(1, 30):
        traj.extend([Seg("gen", gen, logprobs=[0.0] * len(gen), turn=turn)])
        traj.extend(chat.observation([("screenshot", "(image)")], [shot], "Image returned by screenshot.", turn))
        traj.apply_window()
        turns = turn
        if traj.capped:
            break
    traj.close_chunk()
    assert traj.capped and len(traj.chunks) == 2 and turns == 8  # drop at 5 opens chunk 2; turn 8 would open 3


# ----------------------------------------------------------------------------------------------- live


class ScriptedEngine:
    """A fake engine: plays a fixed list of assistant turns and records every prompt it was sent."""

    def __init__(self, chat: QwenChat, turns: list[str]):
        self.chat, self.turns, self.prompts = chat, list(turns), []

    async def __call__(self, tokens, images, max_new_tokens):
        assert sum(t == self.chat.image_pad for t in tokens) == sum(i.n_tokens for i in images)
        self.prompts.append((list(tokens), list(images)))
        text = self.turns.pop(0) if self.turns else "I am done."
        ids = self.chat.encode(text) + [self.chat.im_end]
        return GenOut(ids, [-0.1] * len(ids), "stop")


@pytest.mark.skipif(not os.environ.get("DESIGNGYM_URL"), reason="set DESIGNGYM_URL to a DesignGym server")
def test_live_episode(chat):
    url = os.environ["DESIGNGYM_URL"]
    click = "<tool_call>\n<function=left_click>\n<parameter=coordinate>\n[500, 500]\n</parameter>\n</function>\n" \
            "</tool_call>"
    shot = "<tool_call>\n<function=screenshot>\n</function>\n</tool_call>"
    engine = ScriptedEngine(chat, [click, shot, "thinking out loud", click] * 3)
    task = {"split": "rl", "index": 0, "mode": "reference", "surface": "gui", "skin": "figma",
            "coordinates": "normalized"}
    res = asyncio.run(run_episode(url, chat, task, engine, EpisodeConfig(max_turns=12, max_images=4, keep_images=2, max_nudges=5),
                                  keep_transcript=True))
    assert res.status == "auto_submitted", res.error
    assert res.graded and 0.0 <= res.reward <= 1.0
    assert res.metrics["nudges"] == 3 and res.metrics["turns"] == 12
    assert len(res.chunks) >= 2
    # every chunk's context is exactly a prompt the engine saw, with the same images
    for c in res.chunks:
        sent = [p for p in engine.prompts if p[0][:c.prompt_len] == c.tokens[:c.prompt_len]
                and len(p[0]) == c.prompt_len]
        assert sent, f"chunk {c.index} context was never sent to the engine"
        assert [i.b64 for i in sent[0][1]] == [i.b64 for i in c.images[:len(sent[0][1])]]
    assert sum(c.n_trained for c in res.chunks) == sum(
        len(chat.encode(t)) + 1 for t in ([click, shot, "thinking out loud", click] * 3)[:12])
