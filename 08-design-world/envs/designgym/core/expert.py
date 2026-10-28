"""Expert trajectories: the ground truth replayed as the actions that build it.

Every design template decomposes into one `add_*` call per layer, back to
front, with exact arguments. Replaying that sequence reconstructs the ground
truth, and scores 1.0, which makes it the strongest test the environment has:
if the expert does not score 1.0, the reward or the tools are wrong.

The same trajectory is emitted for the HTML surface as a single page.
"""

from __future__ import annotations

from .doc import DesignDoc, Element
from .html_profile import doc_to_html


def _round(v: float) -> float:
    r = round(float(v), 2)
    return int(r) if r == int(r) else r


def element_call(el: Element) -> tuple[str, dict]:
    box = {"x": _round(el.x), "y": _round(el.y)}
    extra = {}
    if el.angle:
        extra["angle"] = _round(el.angle)
    if el.opacity < 1:
        extra["opacity"] = _round(el.opacity)
    if el.kind == "image":
        return "add_image", {
            "asset_id": el.asset_id, **box, "width": _round(el.width),
            "height": _round(el.height), **extra,
        }
    if el.kind == "shape":
        args = {"shape": el.shape, **box, "width": _round(el.width), "height": _round(el.height),
                "fill": el.fill or "none", **extra}
        if el.stroke and el.stroke_width:
            args.update(stroke=el.stroke, stroke_width=_round(el.stroke_width))
        if el.radius:
            args["radius"] = _round(el.radius)
        return "add_shape", args
    args = {"text": el.text, **box}
    if not el.auto_width:
        args["width"] = _round(el.width)
    args.update(font_family=el.font_family, font_size=_round(el.font_size),
                color=el.color or "#000000")
    if el.align != "left":
        args["align"] = el.align
    for key in ("bold", "italic", "underline", "uppercase"):
        if getattr(el, key):
            args[key] = True
    if el.line_height != 1.0:
        args["line_height"] = _round(el.line_height)
    if el.letter_spacing:
        args["letter_spacing"] = _round(el.letter_spacing)
    if el.runs:
        args["runs"] = [r.model_dump(exclude_none=True) for r in el.runs]
    args.update(extra)
    return "add_text", args


def tool_calls(doc: DesignDoc) -> list[dict]:
    calls = []
    if doc.background.lower() != "#ffffff":
        calls.append({"name": "set_background", "arguments": {"color": doc.background}})
    for el in doc.elements:
        if not el.visible:
            continue
        name, args = element_call(el)
        calls.append({"name": name, "arguments": args})
    calls.append({"name": "submit", "arguments": {}})
    return calls


def html_page(doc: DesignDoc) -> str:
    return doc_to_html(doc)
