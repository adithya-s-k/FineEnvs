"""Token-level chat rendering and tool-call parsing for Qwen3.5/3.6/3.8 VLMs, shared by every trainer adapter.

Rendering follows the evaluation (`rl/tinker/hf_jobs/eval_job.sh`, `sft/collator.py`): the model's own chat
template with the tool listing patched to bare function objects, thinking off. Messages are rendered once, as token
segments, and never re-rendered: the rollout engine and the trainer see exactly the same ids (token-in, token-out).
An image is its own segment, `<|vision_start|>` + N x `<|image_pad|>` + `<|vision_end|>`, N from the model's image
processor, so the image window can swap it for the placeholder text without touching any other token.

Tool calls are parsed as vLLM's `qwen3_coder` parser does in evaluation (value types from the tool's JSON schema),
then passed through the Tinker adapter's `coerce_to_schema` repairs.
"""

from __future__ import annotations

import ast
import base64
import io
import json
import re
from dataclasses import dataclass, field
from typing import Any

THINK_OFF = "<think>\n\n</think>\n\n"
TOOL_LISTING = "{{- tool | tojson }}"
VISION = "<|vision_start|><|image_pad|><|vision_end|>"
PLACEHOLDER = "[earlier screenshot omitted]"  # designgym.cua_format.PLACEHOLDER (SFT windows, rollout.py, Tinker)


# ----------------------------------------------------------------------------------------------- images


@dataclass
class ImageItem:
    """One image as the model sees it: the original bytes (sent to the engine) and its token grid."""

    mime: str
    b64: str
    grid_thw: tuple[int, int, int] = (1, 0, 0)
    n_tokens: int = 0
    _pil: Any = field(default=None, repr=False)

    @property
    def data_url(self) -> str:
        return f"data:{self.mime};base64,{self.b64}"

    @property
    def pil(self):
        if self._pil is None:
            from PIL import Image

            self._pil = Image.open(io.BytesIO(base64.b64decode(self.b64))).convert("RGB")
        return self._pil


# ----------------------------------------------------------------------------------------------- tool calls


@dataclass
class ToolCall:
    name: str
    args: dict
    parse_error: str | None = None


_TOOL_CALL = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)", re.S)
_FUNCTION = re.compile(r"<function=(.*?)>(.*?)(?:</function>|$)", re.S)
_PARAM = re.compile(r"<parameter=(.*?)>(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)", re.S)


def _convert(value: str, schema: dict | None) -> Any:
    """vLLM qwen3_coder's `_convert_param_value`: the declared JSON type decides; no declared type means string."""
    if value.lower() == "null":
        return None
    kind = str(schema["type"]).strip().lower() if isinstance(schema, dict) and isinstance(schema.get("type"), str) \
        else "string"
    if kind in ("string", "str", "text", "varchar", "char", "enum"):
        return value
    if kind.startswith(("int", "uint", "long", "short", "unsigned")):
        try:
            return int(value)
        except ValueError:
            return value
    if kind.startswith(("num", "float")):
        try:
            f = float(value)
            return int(f) if f == int(f) else f
        except (ValueError, OverflowError):
            return value
    if kind in ("boolean", "bool", "binary"):
        return value.lower() == "true"
    if kind in ("object", "array", "arr") or kind.startswith(("dict", "list")):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            pass
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return value


def _types(schema: dict) -> set[str]:
    out: set[str] = set()
    t = schema.get("type")
    out.update([t] if isinstance(t, str) else t or [])
    for alt in schema.get("anyOf", []) + schema.get("oneOf", []):
        out |= _types(alt)
    return out


def coerce_to_schema(arguments: dict, schema: dict) -> dict:
    """`rl/tinker/openenv_rl/openenv_tools.coerce_to_schema`: scalars recast to the type the schema declares."""
    props = schema.get("properties") or {}
    out = dict(arguments)
    for key, value in arguments.items():
        allowed = _types(props.get(key) or {})
        if not allowed or isinstance(value, bool):
            continue
        if "string" in allowed and isinstance(value, (int, float)) and not ({"integer", "number"} & allowed):
            out[key] = str(value)
        elif "array" in allowed and isinstance(value, (str, int, float)) and not (
                {"string", "integer", "number"} & allowed):
            item = (props[key].get("items") or {}) if isinstance(props.get(key), dict) else {}
            out[key] = [str(value) if "string" in _types(item) else value]
    return out


def parse_tool_calls(text: str, tools: list[dict]) -> tuple[str, list[ToolCall]]:
    """(content before the first call, calls) from a sampled assistant message in Qwen's XML tool format."""
    schemas = {t["name"]: t.get("parameters") or {} for t in tools}
    first = text.find("<tool_call>")
    content = (text if first < 0 else text[:first]).strip()
    calls: list[ToolCall] = []
    if first < 0:
        return content, calls
    for block in _TOOL_CALL.findall(text[first:]):
        fn = _FUNCTION.search(block)
        if not fn:
            calls.append(ToolCall("", {}, "tool call without a <function=...> block"))
            continue
        name = fn.group(1).strip()
        if name not in schemas:
            calls.append(ToolCall(name, {}, f"unknown tool: {name}"))
            continue
        props = schemas[name].get("properties") or {}
        args = {}
        for key, raw in _PARAM.findall(fn.group(2)):
            key = key.strip()
            value = raw[1:] if raw.startswith("\n") else raw
            value = value[:-1] if value.endswith("\n") else value
            args[key] = _convert(value, props.get(key))
        calls.append(ToolCall(name, coerce_to_schema(args, schemas[name])))
    return content, calls


# ----------------------------------------------------------------------------------------------- rendering


@dataclass
class Seg:
    """A run of tokens in the conversation.

    kind: "text" (rendered context), "image" (one image's vision tokens), or "gen" (sampled by the policy, with the
    engine's logprobs). `droppable` images are the ones the window may replace with the placeholder.
    """

    kind: str
    tokens: list[int]
    image: ImageItem | None = None
    droppable: bool = False
    logprobs: list[float] | None = None
    turn: int = -1


class QwenChat:
    """Renders DesignGym conversations for one Qwen3.5-family VLM as token segments.

    Args:
        model: Hub id or path (tokenizer, chat template and image processor are read from it).
        enable_thinking: False everywhere in DesignGym (the eval runs thinking off).
        tokenizer / image_processor: pass already-loaded objects to skip loading (the trainers have them).
    """

    def __init__(self, model: str, enable_thinking: bool = False, tokenizer=None, image_processor=None):
        from transformers import AutoTokenizer

        self.model = model
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model)
        if image_processor is None:
            from transformers import AutoImageProcessor

            image_processor = AutoImageProcessor.from_pretrained(model)
        self.image_processor = image_processor
        self.enable_thinking = enable_thinking
        template = self.tokenizer.chat_template
        if not isinstance(template, str) or TOOL_LISTING not in template:
            raise ValueError("chat template changed upstream: re-check the tool listing against the Tinker renderer")
        self.template = template.replace(TOOL_LISTING,
                                         "{{- (tool.function if tool.function is defined else tool) | tojson }}")
        tok = self.tokenizer
        self.im_end = tok.convert_tokens_to_ids("<|im_end|>")
        self.vision_start = tok.convert_tokens_to_ids("<|vision_start|>")
        self.vision_end = tok.convert_tokens_to_ids("<|vision_end|>")
        self.image_pad = tok.convert_tokens_to_ids("<|image_pad|>")
        self.merge = int(getattr(image_processor, "merge_size", 2))
        self._placeholder_ids = self.encode(PLACEHOLDER)

    # -- low level

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def render(self, messages: list[dict], tools: list[dict] | None = None, generation_prompt: bool = False) -> str:
        return self.tokenizer.apply_chat_template(
            messages, tools=tools, chat_template=self.template, tokenize=False,
            add_generation_prompt=generation_prompt, enable_thinking=self.enable_thinking)

    def image_item(self, mime: str, b64: str) -> ImageItem:
        item = ImageItem(mime, b64)
        out = self.image_processor(images=[item.pil], return_tensors="np")
        t, h, w = (int(x) for x in out["image_grid_thw"][0])
        item.grid_thw = (t, h, w)
        item.n_tokens = t * h * w // (self.merge ** 2)
        return item

    def image_tokens(self, item: ImageItem) -> list[int]:
        return [self.vision_start] + [self.image_pad] * item.n_tokens + [self.vision_end]

    @property
    def placeholder_tokens(self) -> list[int]:
        return list(self._placeholder_ids)

    def _segments(self, text: str, images: list[ImageItem], droppable: bool, turn: int) -> list[Seg]:
        pieces = text.split(VISION)
        if len(pieces) - 1 != len(images):
            raise ValueError(f"rendered {len(pieces) - 1} image slots for {len(images)} images")
        segs: list[Seg] = []
        for k, piece in enumerate(pieces):
            if piece:
                segs.append(Seg("text", self.encode(piece), turn=turn))
            if k < len(images):
                segs.append(Seg("image", self.image_tokens(images[k]), image=images[k], droppable=droppable,
                                turn=turn))
        return segs

    # -- conversation pieces

    @staticmethod
    def _user(images: list[ImageItem], text: str) -> dict:
        return {"role": "user", "content": [*({"type": "image"} for _ in images), {"type": "text", "text": text}]}

    def opening(self, system: str, tools: list[dict], images: list[ImageItem], text: str) -> list[Seg]:
        """System prompt with the tool list, the opening user turn (target + editor screenshot), generation prompt.

        Opening images are never dropped by the window (the target must stay visible).
        """
        fn_tools = [{"type": "function", "function": t} for t in tools]
        rendered = self.render([{"role": "system", "content": system}, self._user(images, text)], tools=fn_tools,
                               generation_prompt=True)
        return self._segments(rendered, images, droppable=False, turn=0)

    def observation(self, tool_results: list[tuple[str, str]], images: list[ImageItem], note: str | None,
                    turn: int) -> list[Seg]:
        """What follows a sampled assistant turn (which ends at `<|im_end|>`): tool responses, then the images the
        tools returned in one user turn (OpenAI-compatible APIs carry tool images that way, as `rollout.py` does),
        then the generation prompt. `tool_results` is [(tool name, text)]; with no tool results, `note` alone is a
        plain user turn (the nudge)."""
        mark = "⁣ANCHOR⁣"
        anchor = [{"role": "user", "content": "x"}, {"role": "assistant", "content": mark}]
        msgs: list[dict] = [{"role": "tool", "name": n, "content": t} for n, t in tool_results]
        if images or note:
            msgs.append(self._user(images, note or "Continue.") if images else {"role": "user", "content": note})
        full = self.render(anchor + msgs, generation_prompt=True)
        # Everything after the anchor turn's <|im_end|>. The sampled turn ends at <|im_end|> itself, so the "\n" the
        # template puts after it belongs to this delta. (The anchor's own rendering may differ with what follows it:
        # Qwen3.6 drops the think block of turns before the last user query, so the delta is cut at the marker.)
        end = mark + "<|im_end|>"
        k = full.index(end) + len(end)
        return self._segments(full[k:], images, droppable=True, turn=turn)

    def closing_tokens(self) -> list[int]:
        """Appended (as context) after a sampled turn that hit the token limit without `<|im_end|>`."""
        return [self.im_end]
