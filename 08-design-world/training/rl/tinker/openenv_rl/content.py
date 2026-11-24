"""Converting tool results and chat content between OpenEnv, MCP, OpenAI and the Tinker cookbook.

Tool results reach a trainer in many shapes: MCP `CallToolResult` dicts (`{"content": [blocks]}`),
bare block lists, strings, or arbitrary JSON. The cookbook's renderers take a string or a list of
`TextPart` / `ImagePart`, and accept images in any role, including tool results. These helpers map
one onto the other without losing images, which is what screenshot-driven environments need.
"""

from __future__ import annotations

import base64
import json
from typing import Any

IMAGE_MIME_TYPES = ("image/png", "image/jpeg")


def _data_uri(data: str, mime_type: str) -> str:
    return f"data:{mime_type};base64,{data}"


def _image_block_to_part(block: dict[str, Any]) -> dict[str, Any] | None:
    """An MCP / OpenAI / Anthropic image block as a cookbook `ImagePart`, or None if it is not one."""
    kind = block.get("type")
    if kind == "image":
        if isinstance(block.get("data"), str):  # MCP ImageContent
            mime = block.get("mimeType") or block.get("mime_type") or "image/png"
            return {"type": "image", "image": _data_uri(block["data"], mime)}
        if isinstance(block.get("image"), str):  # already a cookbook ImagePart
            return {"type": "image", "image": block["image"]}
        source = block.get("source")
        if isinstance(source, dict) and source.get("type") == "base64":  # Anthropic
            return {
                "type": "image",
                "image": _data_uri(source["data"], source.get("media_type", "image/png")),
            }
    if kind == "image_url":  # OpenAI
        url = block.get("image_url")
        url = url.get("url") if isinstance(url, dict) else url
        if isinstance(url, str):
            return {"type": "image", "image": url}
    return None


def to_content_parts(data: Any) -> str | list[dict[str, Any]]:
    """A tool result (any shape) as cookbook message content: a string, or text and image parts.

    Args:
        data: The tool's result: an MCP `CallToolResult` dict, a list of content blocks, a string,
            bytes (treated as a PNG), or any JSON-serialisable value.

    Returns:
        A plain string when the result has no images, otherwise a list of `TextPart` / `ImagePart`
        dicts in the original order.
    """
    if isinstance(data, (bytes, bytearray)):
        return [
            {
                "type": "image",
                "image": _data_uri(base64.b64encode(data).decode(), "image/png"),
            }
        ]
    blocks: list[Any] | None = None
    if isinstance(data, dict) and isinstance(data.get("content"), list):
        blocks = data["content"]
    elif isinstance(data, list) and all(isinstance(b, dict) and "type" in b for b in data):
        blocks = data
    if blocks is None:
        if isinstance(data, str):
            return data
        image = _image_block_to_part(data) if isinstance(data, dict) else None
        return [image] if image else json.dumps(data, default=str)
    parts: list[dict[str, Any]] = []
    for block in blocks:
        image = _image_block_to_part(block)
        if image is not None:
            parts.append(image)
        elif block.get("type") == "text":
            parts.append({"type": "text", "text": str(block.get("text", ""))})
        else:
            parts.append({"type": "text", "text": json.dumps(block, default=str)})
    if not any(p["type"] == "image" for p in parts):
        return "\n".join(p["text"] for p in parts)
    return parts


def count_images(content: Any) -> int:
    """How many images a message's content holds."""
    if not isinstance(content, list):
        return 0
    return sum(1 for p in content if isinstance(p, dict) and p.get("type") == "image")


def openai_content_to_parts(content: Any) -> str | list[dict[str, Any]]:
    """OpenAI chat `content` (string or `text` / `image_url` parts) as cookbook content."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[dict[str, Any]] = []
    for part in content:
        image = _image_block_to_part(part) if isinstance(part, dict) else None
        if image is not None:
            parts.append(image)
        elif isinstance(part, dict) and part.get("type") in ("text", "input_text"):
            parts.append({"type": "text", "text": part.get("text", "")})
    if not any(p["type"] == "image" for p in parts):
        return "".join(p["text"] for p in parts)
    return parts
