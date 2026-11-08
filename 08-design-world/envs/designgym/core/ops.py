"""Edit operations: `(doc, op) -> doc`, the only way a document ever changes.

MCP tool calls, the HTML surface, the browser editor and the computer-use agent
all end up here. That is what makes the surfaces comparable: a drag in the
editor and an `update_element` call are the same operation, validated the same
way, recorded in the same trajectory format.

Operations never mutate their input. `apply` copies, edits the copy and returns
it with a small result dict, so undo is just keeping the previous document.
"""

from __future__ import annotations

import difflib
import math
from dataclasses import dataclass, field
from typing import Any

from .colors import parse_color
from .doc import DesignDoc, Element, Run
from .fonts import FontRegistry
from .text import fit_box


class OpError(ValueError):
    """A rejected operation. The message is written for the model to read."""


@dataclass
class OpContext:
    fonts: FontRegistry
    assets: dict[str, dict] = field(default_factory=dict)  # asset_id -> {"aspect", "kind", ...}


COMMON_PROPS = {"name", "x", "y", "width", "height", "angle", "opacity", "visible", "locked",
                "flip_x", "flip_y"}
KIND_PROPS = {
    "image": {"asset_id"},
    "shape": {"shape", "fill", "stroke", "stroke_width", "radius"},
    "text": {"text", "font_family", "font_size", "color", "bold", "italic", "underline", "align",
             "line_height", "letter_spacing", "uppercase", "runs", "auto_width"},
}
TEXT_LAYOUT_PROPS = {"text", "font_family", "font_size", "bold", "italic", "line_height",
                     "letter_spacing", "uppercase", "runs"}


def _num(name: str, value: Any, lo: float | None = None, hi: float | None = None) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise OpError(f"{name} must be a number, got {value!r}") from None
    if not math.isfinite(v):
        raise OpError(f"{name} must be finite")
    if lo is not None and v < lo:
        raise OpError(f"{name} must be >= {lo}, got {v}")
    if hi is not None and v > hi:
        raise OpError(f"{name} must be <= {hi}, got {v}")
    return v


def _check_geometry(doc: DesignDoc, props: dict) -> None:
    span = 5 * max(doc.width, doc.height)
    for key in ("x", "y"):
        if key in props:
            props[key] = _num(key, props[key], -span, span)
    for key in ("width", "height"):
        if key in props:
            props[key] = _num(key, props[key], 1, span)
    if "angle" in props:
        props["angle"] = _num("angle", props["angle"])
    if "opacity" in props:
        props["opacity"] = _num("opacity", props["opacity"], 0, 1)


def _font(ctx: OpContext, family: str | None) -> str:
    if family is None:
        return ctx.fonts.resolve(None)
    if ctx.fonts.available and not ctx.fonts.knows(family):
        close = difflib.get_close_matches(str(family), ctx.fonts.families(), n=3, cutoff=0.5)
        hint = f" Did you mean: {', '.join(close)}?" if close else " Call list_fonts to see them."
        raise OpError(f"font {family!r} is not available.{hint}")
    return ctx.fonts.resolve(family)


def _color(name: str, value) -> str:
    try:
        return parse_color(value)
    except ValueError as exc:
        raise OpError(f"{name}: {exc}") from None


def _runs(value) -> list[Run] | None:
    if value in (None, []):
        return None
    if not isinstance(value, list):
        raise OpError("runs must be a list of {text, color?, bold?, italic?}")
    try:
        return [Run(**r) if isinstance(r, dict) else Run.model_validate(r) for r in value]
    except Exception as exc:  # pydantic's message is fine for a model to read
        raise OpError(f"invalid runs: {exc}") from None


def _refit_text(el: Element, ctx: OpContext) -> None:
    width, height = fit_box(el, ctx.fonts)
    el.width = width
    el.height = height


def _default_box(doc: DesignDoc, aspect: float) -> tuple[float, float]:
    """A box of the asset's proportions, 40% of the canvas's short side, shrunk to fit inside the canvas."""
    side = 0.4 * min(doc.width, doc.height)
    w, h = (side * aspect, side) if aspect >= 1 else (side, side / aspect)
    s = min(1.0, doc.width / w, doc.height / h)
    return max(1.0, w * s), max(1.0, h * s)


# ------------------------------------------------------------------ ops


def add_image(doc: DesignDoc, ctx: OpContext, asset_id: str, x=None, y=None, width=None,
              height=None, angle=0.0, opacity=1.0, name="") -> dict:
    if asset_id not in ctx.assets:
        known = ", ".join(sorted(ctx.assets, key=lambda a: int(a[1:]) if a[1:].isdigit() else 0))
        raise OpError(f"unknown asset {asset_id!r}; available: {known or 'none'}")
    aspect = float(ctx.assets[asset_id].get("aspect") or 1.0)
    props = {k: v for k, v in dict(x=x, y=y, width=width, height=height, angle=angle,
                                   opacity=opacity).items() if v is not None}
    _check_geometry(doc, props)
    w, h = props.get("width"), props.get("height")
    if w is None and h is None:
        w, h = _default_box(doc, aspect)
    elif w is None:
        w = h * aspect
    elif h is None:
        h = w / aspect
    px = props.get("x", (doc.width - w) / 2)
    py = props.get("y", (doc.height - h) / 2)
    el = Element(id=doc.new_id(), kind="image", asset_id=asset_id, x=px, y=py, width=w, height=h,
                 angle=props.get("angle", 0.0), opacity=props.get("opacity", 1.0), name=name or "")
    doc.elements.append(el)
    return {"element_id": el.id, "element": el.public()}


def add_text(doc: DesignDoc, ctx: OpContext, text: str, x=None, y=None, width=None,
             font_family=None, font_size=48.0, color="#000000", align="left", bold=False,
             italic=False, underline=False, line_height=1.0, letter_spacing=0.0,
             uppercase=False, angle=0.0, opacity=1.0, runs=None, name="") -> dict:
    if not isinstance(text, str) or (not text and not runs):
        raise OpError("text must be a non-empty string")
    if align not in ("left", "center", "right"):
        raise OpError("align must be left, center or right")
    props = {k: v for k, v in dict(x=x, y=y, width=width, angle=angle, opacity=opacity).items()
             if v is not None}
    _check_geometry(doc, props)
    el = Element(
        id=doc.new_id(), kind="text", text=text, name=name or "",
        font_family=_font(ctx, font_family),
        font_size=_num("font_size", font_size, 1, 2000),
        color=_color("color", color), align=align, bold=bool(bold), italic=bool(italic),
        underline=bool(underline), line_height=_num("line_height", line_height, 0.1, 10),
        letter_spacing=_num("letter_spacing", letter_spacing, -50, 500),
        uppercase=bool(uppercase), angle=props.get("angle", 0.0),
        opacity=props.get("opacity", 1.0), runs=_runs(runs),
        width=props.get("width", 10.0), auto_width="width" not in props,
    )
    _refit_text(el, ctx)
    el.x = props.get("x", (doc.width - el.width) / 2)
    el.y = props.get("y", (doc.height - el.height) / 2)
    doc.elements.append(el)
    return {"element_id": el.id, "element": el.public()}


def add_shape(doc: DesignDoc, ctx: OpContext, shape="rect", x=None, y=None, width=None,
              height=None, fill="#d9d9d9", stroke=None, stroke_width=0.0, radius=0.0,
              angle=0.0, opacity=1.0, name="") -> dict:
    if shape not in ("rect", "ellipse"):
        raise OpError("shape must be rect or ellipse")
    props = {k: v for k, v in dict(x=x, y=y, width=width, height=height, angle=angle,
                                   opacity=opacity).items() if v is not None}
    _check_geometry(doc, props)
    side = 0.25 * min(doc.width, doc.height)
    w, h = props.get("width", side), props.get("height", side)
    el = Element(
        id=doc.new_id(), kind="shape", shape=shape, name=name or "",
        x=props.get("x", (doc.width - w) / 2), y=props.get("y", (doc.height - h) / 2),
        width=w, height=h, angle=props.get("angle", 0.0), opacity=props.get("opacity", 1.0),
        fill=None if fill in (None, "none", "") else _color("fill", fill),
        stroke=None if stroke in (None, "none", "") else _color("stroke", stroke),
        stroke_width=_num("stroke_width", stroke_width or 0, 0, 1000),
        radius=_num("radius", radius or 0, 0, 10000),
    )
    doc.elements.append(el)
    return {"element_id": el.id, "element": el.public()}


def update_element(doc: DesignDoc, ctx: OpContext, element_id: str, **props) -> dict:
    el = doc.find(element_id)
    props = {k: v for k, v in props.items() if v is not None or k in ("stroke", "fill", "runs")}
    allowed = COMMON_PROPS | KIND_PROPS[el.kind]
    bad = sorted(set(props) - allowed)
    if bad:
        raise OpError(
            f"{el.kind} elements have no {', '.join(bad)}; settable: {', '.join(sorted(allowed))}"
        )
    if not props:
        raise OpError("nothing to update")
    _check_geometry(doc, props)
    if el.kind == "text":
        if "height" in props:
            # Text height follows the lines, as in any editor's auto-height box.
            props.pop("height")
        if "font_family" in props:
            props["font_family"] = _font(ctx, props["font_family"])
        if "font_size" in props:
            props["font_size"] = _num("font_size", props["font_size"], 1, 2000)
        if "line_height" in props:
            props["line_height"] = _num("line_height", props["line_height"], 0.1, 10)
        if "letter_spacing" in props:
            props["letter_spacing"] = _num("letter_spacing", props["letter_spacing"], -50, 500)
        if "color" in props:
            props["color"] = _color("color", props["color"])
        if "align" in props and props["align"] not in ("left", "center", "right"):
            raise OpError("align must be left, center or right")
        if "runs" in props:
            props["runs"] = _runs(props["runs"])
        elif "text" in props:
            props["runs"] = None
        if "width" in props:
            props.setdefault("auto_width", False)
    if el.kind == "image" and "asset_id" in props and props["asset_id"] not in ctx.assets:
        raise OpError(f"unknown asset {props['asset_id']!r}")
    if el.kind == "shape":
        for key in ("fill", "stroke"):
            if key in props:
                props[key] = None if props[key] in (None, "none", "") else _color(key, props[key])
        if "shape" in props and props["shape"] not in ("rect", "ellipse"):
            raise OpError("shape must be rect or ellipse")
    # Anchor an auto-width text box on its alignment edge, so centred text
    # stays centred when it grows, as it does in Figma and Canva.
    anchor = None
    if el.kind == "text" and el.auto_width and "x" not in props:
        anchor = {"left": el.x, "center": el.x + el.width / 2, "right": el.x + el.width}[el.align]
    try:
        for key, value in props.items():
            setattr(el, key, value)
    except Exception as exc:
        raise OpError(f"invalid value: {exc}") from None
    if el.kind == "text" and (TEXT_LAYOUT_PROPS & set(props) or "width" in props or
                              "auto_width" in props):
        _refit_text(el, ctx)
        if anchor is not None and el.auto_width:
            el.x = {"left": anchor, "center": anchor - el.width / 2,
                    "right": anchor - el.width}[el.align]
    return {"element_id": el.id, "element": el.public()}


def delete_element(doc: DesignDoc, ctx: OpContext, element_id: str) -> dict:
    idx = doc.index_of(element_id)
    doc.elements.pop(idx)
    return {"deleted": element_id}


def duplicate_element(doc: DesignDoc, ctx: OpContext, element_id: str, dx=20.0, dy=20.0) -> dict:
    idx = doc.index_of(element_id)
    clone = doc.elements[idx].model_copy(deep=True)
    clone.id = doc.new_id()
    clone.x += _num("dx", dx)
    clone.y += _num("dy", dy)
    doc.elements.insert(idx + 1, clone)
    return {"element_id": clone.id, "element": clone.public()}


def reorder_element(doc: DesignDoc, ctx: OpContext, element_id: str, to="front") -> dict:
    idx = doc.index_of(element_id)
    el = doc.elements.pop(idx)
    n = len(doc.elements)
    if to == "front":
        new = n
    elif to == "back":
        new = 0
    elif to == "forward":
        new = min(n, idx + 1)
    elif to == "backward":
        new = max(0, idx - 1)
    else:
        try:
            new = int(to)
        except (TypeError, ValueError):
            doc.elements.insert(idx, el)
            raise OpError("to must be front, back, forward, backward or a layer index") from None
        new = max(0, min(n, new))
    doc.elements.insert(new, el)
    return {"element_id": element_id, "index": new, "order": [e.id for e in doc.elements]}


def _bounds(el: Element) -> tuple[float, float, float, float]:
    """Axis-aligned bounds of the rotated box."""
    if not el.angle:
        return el.x, el.y, el.x + el.width, el.y + el.height
    cx, cy = el.x + el.width / 2, el.y + el.height / 2
    a = math.radians(el.angle)
    hw = (abs(el.width * math.cos(a)) + abs(el.height * math.sin(a))) / 2
    hh = (abs(el.width * math.sin(a)) + abs(el.height * math.cos(a))) / 2
    return cx - hw, cy - hh, cx + hw, cy + hh


def align_elements(doc: DesignDoc, ctx: OpContext, element_ids: list[str], alignment: str,
                   relative_to: str = "canvas") -> dict:
    if isinstance(element_ids, str):
        element_ids = [element_ids]
    els = [doc.find(e) for e in element_ids]
    if not els:
        raise OpError("element_ids is empty")
    if relative_to == "selection" and len(els) > 1:
        boxes = [_bounds(e) for e in els]
        ref = (min(b[0] for b in boxes), min(b[1] for b in boxes),
               max(b[2] for b in boxes), max(b[3] for b in boxes))
    else:
        ref = (0.0, 0.0, float(doc.width), float(doc.height))
    for el in els:
        x0, y0, x1, y1 = _bounds(el)
        if alignment == "left":
            el.x += ref[0] - x0
        elif alignment == "center":
            el.x += (ref[0] + ref[2]) / 2 - (x0 + x1) / 2
        elif alignment == "right":
            el.x += ref[2] - x1
        elif alignment == "top":
            el.y += ref[1] - y0
        elif alignment == "middle":
            el.y += (ref[1] + ref[3]) / 2 - (y0 + y1) / 2
        elif alignment == "bottom":
            el.y += ref[3] - y1
        else:
            raise OpError("alignment must be left, center, right, top, middle or bottom")
    return {"aligned": [e.id for e in els], "elements": [e.public() for e in els]}


def distribute_elements(doc: DesignDoc, ctx: OpContext, element_ids: list[str],
                        axis: str = "horizontal") -> dict:
    els = [doc.find(e) for e in element_ids]
    if len(els) < 3:
        raise OpError("distribute needs at least three elements")
    if axis not in ("horizontal", "vertical"):
        raise OpError("axis must be horizontal or vertical")
    lo, hi = (0, 2) if axis == "horizontal" else (1, 3)
    els.sort(key=lambda e: _bounds(e)[lo])
    boxes = [_bounds(e) for e in els]
    total = sum(b[hi] - b[lo] for b in boxes)
    gap = (boxes[-1][hi] - boxes[0][lo] - total) / (len(els) - 1)
    cursor = boxes[0][hi] + gap
    for el, box in zip(els[1:-1], boxes[1:-1], strict=True):
        shift = cursor - box[lo]
        if axis == "horizontal":
            el.x += shift
        else:
            el.y += shift
        cursor += box[hi] - box[lo] + gap
    return {"distributed": [e.id for e in els]}


def set_background(doc: DesignDoc, ctx: OpContext, color: str) -> dict:
    doc.background = _color("color", color)
    return {"background": doc.background}


def clear_canvas(doc: DesignDoc, ctx: OpContext) -> dict:
    doc.elements = []
    doc.background = "#ffffff"
    return {"cleared": True}


def replace_document(doc: DesignDoc, ctx: OpContext, document: dict) -> dict:
    """Swap in a whole document (HTML surface, editor import). Size is fixed by the task.

    Elements parsed from HTML may carry `_anchor: [ax, ay]`: the fraction of
    their own final size to shift left/up by (`translate(-50%)`, `right:`,
    `bottom:`). It is applied after text is laid out, since that is when a text
    box first has a width, and never stored.
    """
    raw = [dict(e) for e in document.get("elements", [])]
    anchors = [e.pop("_anchor", None) for e in raw]
    try:
        new = DesignDoc.model_validate({**document, "elements": raw, "width": doc.width,
                                        "height": doc.height})
    except Exception as exc:
        raise OpError(f"invalid document: {exc}") from None
    for el, anchor in zip(new.elements, anchors, strict=True):
        if el.kind == "image" and el.asset_id not in ctx.assets:
            raise OpError(f"unknown asset {el.asset_id!r}")
        if el.kind == "text":
            el.font_family = _font(ctx, el.font_family)
            _refit_text(el, ctx)
        if anchor:
            el.x -= float(anchor[0]) * el.width
            el.y -= float(anchor[1]) * el.height
    ids = {e.id for e in new.elements}
    nums = [int(i[1:]) for i in ids if i.startswith("e") and i[1:].isdigit()]
    new.next_id = max([doc.next_id - 1, *nums, 0]) + 1
    doc.background = new.background
    doc.elements = new.elements
    doc.next_id = new.next_id
    return {"elements": len(doc.elements)}


OPS = {
    "add_image": add_image,
    "add_text": add_text,
    "add_shape": add_shape,
    "update_element": update_element,
    "delete_element": delete_element,
    "duplicate_element": duplicate_element,
    "reorder_element": reorder_element,
    "align_elements": align_elements,
    "distribute_elements": distribute_elements,
    "set_background": set_background,
    "clear_canvas": clear_canvas,
    "replace_document": replace_document,
}

# Ops that also exist as editor gestures but must not be offered as agent tools.
INTERNAL_OPS = {"replace_document"}


def apply(doc: DesignDoc, op: str, args: dict | None, ctx: OpContext) -> tuple[DesignDoc, dict]:
    fn = OPS.get(op)
    if fn is None:
        raise OpError(f"unknown operation {op!r}")
    new = doc.copy_deep()
    try:
        result = fn(new, ctx, **(args or {}))
    except OpError:
        raise
    except KeyError as exc:
        raise OpError(str(exc).strip("'\"")) from None
    except TypeError as exc:
        raise OpError(f"{op}: {exc}") from None
    return new, result
