"""The design document: the one piece of state every surface edits.

A document is a canvas size, a background colour and a flat, back-to-front list
of elements. That is the whole of the source designs' model once their pre-rendered layers
are treated as image assets, and it is deliberately all this environment
supports: no groups, no constraints, no auto-layout. Every field here maps to a
control in the editor and to a parameter of a tool, and nothing else exists.

Three element kinds:

- `image`: a placed asset (photo, illustration, decorative shape), stretched to
  its box.
- `text`: a text box. Lines are explicit (`\\n`), never wrapped automatically,
  so the renderer and the browser cannot disagree about where a line breaks.
  Height is derived from the line count, like Figma's auto-height text.
- `shape`: a rectangle or ellipse with fill and stroke. The designs have none of these
  (its shapes arrive as rendered assets); they exist so the editor has a shape
  tool, as every real editor does.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .colors import parse_color

Kind = Literal["image", "text", "shape"]
Align = Literal["left", "center", "right"]
ShapeType = Literal["rect", "ellipse"]


class Run(BaseModel):
    """A styled stretch of a text element. Runs concatenate to the element's text."""

    model_config = ConfigDict(extra="forbid")

    text: str
    color: str | None = None
    bold: bool | None = None
    italic: bool | None = None

    @field_validator("color")
    @classmethod
    def _color(cls, v):
        return None if v is None else parse_color(v)


class Element(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    id: str
    kind: Kind
    name: str = ""
    x: float = 0.0
    y: float = 0.0
    width: float = Field(default=100.0, gt=0)
    height: float = Field(default=100.0, gt=0)
    angle: float = 0.0
    opacity: float = Field(default=1.0, ge=0.0, le=1.0)
    visible: bool = True
    locked: bool = False
    flip_x: bool = False
    flip_y: bool = False

    # image
    asset_id: str | None = None
    # the part of the picture shown, as the fractions cut off its left, top, right and bottom edges (an app's crop);
    # None shows all of it. The box (x, y, width, height) is the part shown.
    crop: tuple[float, float, float, float] | None = None

    # shape
    shape: ShapeType | None = None
    fill: str | None = None
    stroke: str | None = None
    stroke_width: float = Field(default=0.0, ge=0.0)
    radius: float = Field(default=0.0, ge=0.0)

    # text
    text: str = ""
    font_family: str | None = None
    font_size: float = Field(default=48.0, gt=0)
    color: str | None = None
    bold: bool = False
    italic: bool = False
    underline: bool = False
    align: Align = "left"
    line_height: float = Field(default=1.0, gt=0)
    letter_spacing: float = 0.0
    uppercase: bool = False
    auto_width: bool = False
    runs: list[Run] | None = None

    @field_validator("fill", "stroke", "color")
    @classmethod
    def _colors(cls, v):
        return None if v is None else parse_color(v)

    @field_validator("angle")
    @classmethod
    def _angle(cls, v):
        a = float(v) % 360.0
        return a - 360.0 if a > 180.0 else a

    @field_validator("crop")
    @classmethod
    def _crop(cls, v):
        if v is None:
            return None
        l, t, r, b = (float(x) for x in v)
        if min(l, t, r, b) < 0 or l + r >= 0.999 or t + b >= 0.999:
            raise ValueError("crop fractions are >= 0 and leave some of the picture (left + right, top + bottom < 1)")
        return None if max(l, t, r, b) < 1e-6 else (l, t, r, b)

    @model_validator(mode="after")
    def _kind_fields(self):
        if self.kind == "image" and not self.asset_id:
            raise ValueError("an image element needs an asset_id")
        if self.kind != "image" and self.crop is not None:
            raise ValueError("only an image can be cropped")
        if self.kind == "shape" and self.shape is None:
            object.__setattr__(self, "shape", "rect")
        if self.kind == "text" and self.runs:
            joined = "".join(r.text for r in self.runs)
            if joined != self.text:
                object.__setattr__(self, "text", joined)
        return self

    # ---------------------------------------------------------- helpers

    def uncropped(self) -> "Element":
        """A cropped picture as the whole picture it was cut from: the same pixels show where the box was, the rest
        reaches past it (and is clipped by the canvas like any part off it). Cropping a picture to the part on the
        canvas and placing it whole, hanging off the canvas, are the same design."""
        if self.kind != "image" or not self.crop:
            return self
        import math

        l, t, r, b = self.crop
        w, h = self.width / (1 - l - r), self.height / (1 - t - b)
        dx, dy = w * (r - l) / 2, h * (b - t) / 2  # the whole picture's centre, from the part's, in its own frame
        if self.flip_x:
            dx = -dx
        if self.flip_y:
            dy = -dy
        a = math.radians(self.angle)
        cx = self.x + self.width / 2 + dx * math.cos(a) - dy * math.sin(a)
        cy = self.y + self.height / 2 + dx * math.sin(a) + dy * math.cos(a)
        return self.model_copy(update={"x": cx - w / 2, "y": cy - h / 2, "width": w, "height": h, "crop": None})

    @property
    def lines(self) -> list[str]:
        return self.text.split("\n")

    def display_text(self) -> str:
        return self.text.upper() if self.uppercase else self.text

    def styled_runs(self) -> list[Run]:
        """Runs with every style field filled from the element defaults."""
        base_color = self.color or "#000000"
        if not self.runs:
            return [Run(text=self.text, color=base_color, bold=self.bold, italic=self.italic)]
        return [
            Run(
                text=r.text,
                color=r.color or base_color,
                bold=self.bold if r.bold is None else r.bold,
                italic=self.italic if r.italic is None else r.italic,
            )
            for r in self.runs
        ]

    def public(self) -> dict:
        """The fields that mean something for this kind, for tool output and HTML."""
        common = ["id", "kind", "name", "x", "y", "width", "height", "angle", "opacity"]
        flags = ["visible", "locked", "flip_x", "flip_y"]
        per_kind = {
            "image": ["asset_id"],
            "shape": ["shape", "fill", "stroke", "stroke_width", "radius"],
            "text": [
                "text",
                "font_family",
                "font_size",
                "color",
                "bold",
                "italic",
                "underline",
                "align",
                "line_height",
                "letter_spacing",
                "uppercase",
                "runs",
            ],
        }[self.kind]
        out = {}
        for key in common + per_kind:
            value = getattr(self, key)
            if key == "runs":
                if value:
                    out[key] = [r.model_dump(exclude_none=True) for r in value]
                continue
            if isinstance(value, float):
                value = round(value, 2)
            out[key] = value
        for key in flags:
            value = getattr(self, key)
            default = key == "visible"
            if value != default:
                out[key] = value
        return out


class DesignDoc(BaseModel):
    model_config = ConfigDict(extra="forbid")

    width: int = Field(gt=0, le=10000)
    height: int = Field(gt=0, le=10000)
    background: str = "#ffffff"
    elements: list[Element] = Field(default_factory=list)
    next_id: int = 1

    @field_validator("background")
    @classmethod
    def _bg(cls, v):
        return parse_color(v, "#ffffff")

    # ---------------------------------------------------------- helpers

    def find(self, element_id: str) -> Element:
        for el in self.elements:
            if el.id == element_id:
                return el
        known = ", ".join(e.id for e in self.elements) or "none"
        raise KeyError(f"no element {element_id!r} (elements: {known})")

    def index_of(self, element_id: str) -> int:
        for i, el in enumerate(self.elements):
            if el.id == element_id:
                return i
        raise KeyError(f"no element {element_id!r}")

    def new_id(self) -> str:
        eid = f"e{self.next_id}"
        self.next_id += 1
        return eid

    def copy_deep(self) -> DesignDoc:
        return self.model_copy(deep=True)

    def fingerprint(self) -> str:
        """Stable hash of what the document looks like, for render caching."""
        payload = self.model_dump(exclude={"next_id"})
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(blob.encode()).hexdigest()[:16]

    def public(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "background": self.background,
            "elements": [e.public() for e in self.elements],
        }

    def summary(self) -> str:
        """One line per element, the text a model reads after every edit."""
        lines = [f"canvas {self.width}x{self.height}, background {self.background}"]
        if not self.elements:
            lines.append("(no elements yet)")
        for i, el in enumerate(self.elements):
            box = f"x={el.x:.0f} y={el.y:.0f} w={el.width:.0f} h={el.height:.0f}"
            extra = []
            if el.angle:
                extra.append(f"angle={el.angle:.0f}")
            if el.opacity < 1:
                extra.append(f"opacity={el.opacity:.2f}")
            if not el.visible:
                extra.append("hidden")
            if el.kind == "image":
                what = f"image {el.asset_id}"
            elif el.kind == "shape":
                what = f"{el.shape} fill={el.fill}"
            else:
                snippet = el.text.replace("\n", "\\n")
                if len(snippet) > 40:
                    snippet = snippet[:37] + "..."
                what = (
                    f'text "{snippet}" {el.font_family} {el.font_size:.0f}px {el.color} {el.align}'
                )
            tail = (" " + " ".join(extra)) if extra else ""
            lines.append(f"[{i}] {el.id}: {what} @ {box}{tail}")
        return "\n".join(lines)
