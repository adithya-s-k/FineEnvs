"""DesignGym SFT rows (cua-v1, FineEnvs/designworld-sft) as conversations laid out exactly like evaluation prompts.

The eval harness (`envs/designgym/rollout.py` behind vLLM, see training/rl/tinker/hf_jobs/eval_job.sh) and the RL
adapter (`tool_images_as_user`) show a Qwen VLM the episode in one layout; SFT must train in that layout too:

1. episodes are cut with `cua_format.windows()` (8 screenshots, then down to 3; the target never drops; dropped ones
   become `cua_format.PLACEHOLDER`), each window training only its own `trainable` assistant turns;
2. GUI coordinates are on the 0-1000 grid (`coordinates="normalized"`): tool-call `coordinate` / `start_coordinate`
   are rescaled from screen pixels and the system prompt's coordinate sentence is the normalized one;
3. images (and their placeholders) in tool results move to one user turn after them, with "Image returned by
   <tool>." per tool, as OpenAI-compatible chat APIs carry tool images;
4. tool-call arguments are JSON objects (the chat template renders them as Qwen XML).
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path
from typing import Any

EXP = Path(__file__).resolve().parents[2]
PIXELS = "Screen coordinates are pixels of the {w}x{h} screenshot."
NORMALIZED = ("Screen coordinates are on a 0-1000 grid over the screenshot: [0, 0] is its top-left corner "
              "and [1000, 1000] its bottom-right, whatever the image size.")


def cua_format():
    """`designgym/cua_format.py`, loaded by path (no env dependencies); bundled next to this file in jobs. Falls back
    to the installed `designgym` package."""
    if "cua_format" in sys.modules:
        return sys.modules["cua_format"]
    for path in (Path(__file__).with_name("cua_format.py"), EXP / "envs" / "designgym" / "cua_format.py"):
        if path.exists():
            spec = importlib.util.spec_from_file_location("cua_format", path)
            module = importlib.util.module_from_spec(spec)
            sys.modules["cua_format"] = module
            spec.loader.exec_module(module)
            return module
    try:
        import designgym.cua_format as module
    except ImportError as exc:
        raise FileNotFoundError("cua_format.py (envs/designgym, or `pip install designgym`)") from exc
    sys.modules["cua_format"] = module
    return module


def to_grid(value: int | float, size: int) -> int:
    """A screen pixel on the env's 0-1000 grid (the env maps back with round(v * (size - 1) / 1000))."""
    return max(0, min(1000, round(float(value) * 1000 / (size - 1))))


def _normalize_call(name: str, arguments: dict, width: int, height: int) -> dict:
    out = dict(arguments)
    for key in ("coordinate", "start_coordinate"):
        v = out.get(key)
        if isinstance(v, (list, tuple)) and len(v) == 2:
            out[key] = [to_grid(v[0], width), to_grid(v[1], height)]
    return out


def _text(parts: list[dict]) -> str:
    return "".join(p.get("text") or "" for p in parts if p.get("type") == "text")


def conversation(window: dict, *, normalized: bool = True) -> dict:
    """One cua-v1 window (from `cua_format.windows`) as {messages, images, tools, trainable}.

    `images` are encoded bytes in the order their `{"type": "image"}` parts appear in `messages`; `trainable` has one
    flag per message (only assistant messages can be True).
    """
    placeholder = cua_format().PLACEHOLDER
    gui = window.get("surface") == "gui"
    width, height = int(window["screen"]["width"]), int(window["screen"]["height"])
    images_in = iter(window["images"])
    images_out: list[bytes] = []
    out: list[dict] = []
    flags: list[bool] = []
    pending_parts: list[dict] = []   # images / placeholders from the current run of tool messages
    pending_notes: list[str] = []

    def flush() -> None:
        if pending_parts or pending_notes:
            out.append({"role": "user", "content": [*pending_parts, {"type": "text",
                                                                     "text": "\n".join(pending_notes) or "Continue."}]})
            flags.append(False)
            pending_parts.clear()
            pending_notes.clear()

    for m in window["messages"]:
        role, parts = m["role"], m["content"] or []
        if role != "tool":
            flush()
        if role == "system":
            text = _text(parts)
            if gui and normalized:
                text = text.replace(PIXELS.format(w=width, h=height), NORMALIZED)
            out.append({"role": "system", "content": text})
            flags.append(False)
        elif role == "user":
            content = []
            for p in parts:
                if p["type"] == "image":
                    images_out.append(_image_bytes(next(images_in)))
                    content.append({"type": "image"})
                else:
                    content.append({"type": "text", "text": p.get("text") or ""})
            out.append({"role": "user", "content": content})
            flags.append(False)
        elif role == "assistant":
            calls = []
            for c in m.get("tool_calls") or []:
                args = json.loads(c["function"]["arguments"] or "{}")
                if gui and normalized:
                    args = _normalize_call(c["function"]["name"], args, width, height)
                calls.append({"type": "function", "function": {"name": c["function"]["name"], "arguments": args}})
            msg = {"role": "assistant", "content": _text(parts)}
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
            flags.append(bool(m.get("trainable")))
        elif role == "tool":
            text_parts, moved = [], False
            for p in parts:
                if p["type"] == "image":
                    images_out.append(_image_bytes(next(images_in)))
                    pending_parts.append({"type": "image"})
                    moved = True
                elif p.get("text") == placeholder:
                    pending_parts.append({"type": "text", "text": placeholder})
                    moved = True
                else:
                    text_parts.append(p.get("text") or "")
            if moved:
                pending_notes.append(f"Image returned by {m.get('name') or 'the tool'}.")
            out.append({"role": "tool", "content": "".join(text_parts), "name": m.get("name") or ""})
            flags.append(False)
    flush()
    tools = json.loads(window["tools"]) if isinstance(window["tools"], str) else window["tools"]
    return {"messages": out, "images": images_out, "tools": tools, "trainable": flags,
            "meta": {"id": window.get("id"), "surface": window.get("surface"), "skin": (window.get("editor") or {}).get("skin")}}


def _image_bytes(img: Any) -> bytes:
    if isinstance(img, dict):
        return img["bytes"]
    if isinstance(img, (bytes, bytearray)):
        return bytes(img)
    buf = io.BytesIO()  # a PIL image
    img.save(buf, format="PNG")
    return buf.getvalue()


def windows(row: dict, *, max_images: int = 8, keep_images: int = 3) -> list[dict]:
    cf = cua_format()
    return list(cf.windows(row, cf.Window(max_images=max_images, keep_images=keep_images)))
