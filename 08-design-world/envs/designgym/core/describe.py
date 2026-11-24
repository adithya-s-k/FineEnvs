"""Task 2's prompt: the design described in words instead of shown as an image.

The description is generated from the ground-truth document, so it is exactly
as informative as intended and never hallucinates: every asset and every word
of copy is named, colours and fonts are given, and placement is coarse, in the
vocabulary a designer would use in a brief ("top left", "centred", "large").
The model has to turn that into coordinates, which is the skill the task tests.
"""

from __future__ import annotations

from .colors import color_name
from .doc import DesignDoc, Element


def _where(el: Element, doc: DesignDoc) -> str:
    x0, y0 = max(0.0, el.x), max(0.0, el.y)
    x1, y1 = min(doc.width, el.x + el.width), min(doc.height, el.y + el.height)
    if x1 <= x0 or y1 <= y0:
        return "mostly off the canvas"
    cover_w = (x1 - x0) / doc.width
    cover_h = (y1 - y0) / doc.height
    if cover_w > 0.9 and cover_h > 0.9:
        return "covering the whole canvas"
    cx = (x0 + x1) / 2 / doc.width
    cy = (y0 + y1) / 2 / doc.height
    col = "left" if cx < 0.36 else "right" if cx > 0.64 else "center"
    row = "top" if cy < 0.36 else "bottom" if cy > 0.64 else "middle"
    if cover_w > 0.9:
        return {"top": "across the top", "middle": "across the middle",
                "bottom": "across the bottom"}[row]
    if cover_h > 0.9:
        return {"left": "down the left side", "center": "down the middle",
                "right": "down the right side"}[col]
    if row == "middle" and col == "center":
        return "in the center"
    if col == "center":
        return f"at the {row} center"
    if row == "middle":
        return f"at the middle {col}"
    return f"at the {row} {col}"


def _size(el: Element, doc: DesignDoc) -> str:
    frac = (el.width * el.height) / (doc.width * doc.height)
    if frac > 0.6:
        return "full-bleed"
    if frac > 0.25:
        return "large"
    if frac > 0.07:
        return "medium"
    if frac > 0.015:
        return "small"
    return "tiny"


def _text_size(el: Element, doc: DesignDoc) -> str:
    rel = el.font_size / doc.height
    if rel > 0.12:
        return "huge"
    if rel > 0.06:
        return "large"
    if rel > 0.03:
        return "medium"
    return "small"


def describe(doc: DesignDoc, meta: dict, asset_meta: dict[str, dict]) -> str:
    fmt = meta.get("format") or "design"
    title = meta.get("title") or ""
    head = f"A {fmt} design, {doc.width}x{doc.height}px"
    if title:
        head += f', titled "{title}"'
    lines = [head + "."]
    lines.append(
        f"Background colour: {color_name(doc.background)} ({doc.background})."
    )
    lines.append("Layers from back to front:")
    for n, el in enumerate(doc.elements, 1):
        where = _where(el, doc)
        if el.kind == "image":
            kind = asset_meta.get(el.asset_id, {}).get("kind", "image")
            noun = {"photo": "photo", "graphic": "graphic", "background": "background texture"}.get(
                kind, "image"
            )
            desc = f"{n}. A {_size(el, doc)} {noun} (asset {el.asset_id}) {where}"
        elif el.kind == "shape":
            desc = f"{n}. A {el.shape} in {color_name(el.fill or '#000000')} {where}"
        else:
            text = el.display_text().replace("\n", " / ")
            style = []
            if el.bold:
                style.append("bold")
            if el.italic:
                style.append("italic")
            color = el.color or "#000000"
            desc = (
                f'{n}. Text "{text}" in {" ".join(style + [el.font_family or ""]).strip()}, '
                f"{_text_size(el, doc)} ({el.font_size:.0f}px), {color_name(color)} ({color}), "
                f"{el.align}-aligned, {where}"
            )
            if el.letter_spacing and abs(el.letter_spacing) >= 0.5:
                desc += f", letter-spacing {el.letter_spacing:.0f}px"
            if len(el.lines) > 1:
                desc += f", {len(el.lines)} lines"
            if el.runs:
                colors = sorted({color_name(r.color) for r in el.styled_runs() if r.text.strip()})
                if len(colors) > 1:
                    desc += f" (mixed colours: {', '.join(colors)})"
        if el.angle:
            desc += f", rotated {el.angle:.0f} degrees"
        if el.opacity < 0.95:
            desc += f", {round(el.opacity * 100)}% opacity"
        lines.append(desc + ".")
    keywords = [k for k in (meta.get("keywords") or [])][:8]
    if keywords:
        lines.append(f"Mood / keywords: {', '.join(keywords)}.")
    return "\n".join(lines)
