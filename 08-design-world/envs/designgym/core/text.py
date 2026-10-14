"""Text layout, written to match what a browser does with the editor's CSS.

The editor draws a text element as one box with `white-space: pre`,
`line-height: <font_size * line_height>px`, `letter-spacing` and `text-align`.
This module reproduces that box model for skia, so the server render and the
browser render put glyphs in the same place:

- Lines are the explicit `\\n` breaks. Nothing wraps.
- Every line box is `font_size * line_height` tall.
- The baseline sits where CSS puts it: half of the leading above the font's
  ascent, i.e. `top + (L - (ascent + descent)) / 2 + ascent`.
- `letter-spacing` is added after every character, trailing one included, which
  is Chromium's behaviour and why right-aligned spaced text keeps a gap.
"""

from __future__ import annotations

from dataclasses import dataclass

from .doc import Element
from .fonts import FontRegistry


@dataclass
class Piece:
    text: str
    color: str
    bold: bool
    italic: bool
    width: float


@dataclass
class Line:
    pieces: list[Piece]
    width: float
    x: float  # offset inside the box, from alignment
    top: float  # offset of the line box inside the element
    baseline: float  # offset of the baseline inside the element


@dataclass
class TextLayout:
    lines: list[Line]
    width: float  # widest line
    height: float  # line count * line box height
    line_box: float


def _split_lines(el: Element) -> list[list[tuple[str, object]]]:
    """Runs cut at newline boundaries, one list of (text, run) per line."""
    lines: list[list[tuple[str, object]]] = [[]]
    for run in el.styled_runs():
        text = run.text.upper() if el.uppercase else run.text
        parts = text.split("\n")
        for i, part in enumerate(parts):
            if i > 0:
                lines.append([])
            if part:
                lines[-1].append((part, run))
    return lines


def layout(el: Element, fonts: FontRegistry) -> TextLayout:
    size = el.font_size
    line_box = size * el.line_height
    ascent, descent = fonts.metrics(el.font_family or "", size, el.bold, el.italic)
    half_leading = (line_box - (ascent + descent)) / 2
    lines = []
    for i, raw in enumerate(_split_lines(el)):
        pieces = []
        for text, run in raw:
            w = fonts.advance(
                text, el.font_family or "", size, run.bold, run.italic, el.letter_spacing
            )
            pieces.append(Piece(text, run.color, bool(run.bold), bool(run.italic), w))
        width = sum(p.width for p in pieces)
        if el.align == "center":
            x = (el.width - width) / 2
        elif el.align == "right":
            x = el.width - width
        else:
            x = 0.0
        top = i * line_box
        lines.append(Line(pieces, width, x, top, top + half_leading + ascent))
    widest = max((ln.width for ln in lines), default=0.0)
    return TextLayout(lines, widest, max(1.0, len(lines) * line_box), line_box)


def fit_box(el: Element, fonts: FontRegistry) -> tuple[float, float]:
    """(width, height) the element should have after a text or style change."""
    lay = layout(el, fonts)
    width = max(1.0, round(lay.width + 1.0, 2)) if el.auto_width else el.width
    return width, round(lay.height, 2)
