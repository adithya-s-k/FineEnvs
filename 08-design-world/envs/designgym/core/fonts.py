"""The font registry: one set of font files for the renderer and the browser.

The task pack carries the fonts its designs use (263 families, TTF and OTF)
as plain files plus `fonts/manifest.json`. That buys two things: the server can
hand the exact same bytes to the browser editor through `@font-face`, and skia
loads exactly what the browser draws.

Style fallback is deliberately the *browser's* behaviour with
`font-synthesis: none`: a family without a bold face renders bold text in its
regular face. The editor sets that CSS property, so the two renderers agree
instead of the browser inventing a fake bold that skia would not draw.
"""

from __future__ import annotations

import json
import re
import threading
from functools import lru_cache
from pathlib import Path

import skia

STYLES = ("regular", "bold", "italic", "bolditalic")
DEFAULT_FAMILY = "Montserrat"

# Family spellings in the source designs that do not match the font files' own names.
_ALIASES = {
    "arkana script": "Arkana",
    "blogger": "Blogger Sans",
    "delius swash": "Delius Swash Caps",
    "elsie swash": "Elsie Swash Caps",
    "gluk glametrix": "Gluk Foglihtenno06",
    "gluk znikomitno25": "Gluk Foglihtenno06",
    "im fell": "Im Fell Dw Pica Sc",
    "medieval sharp": "Medievalsharp",
    "playlist caps": "Playlist",
    "rissa typeface": "Rissatypeface",
    "selima": "Selima Script",
    "six": "Six Caps",
    "v t323": "Vt323",
}


def style_key(bold: bool, italic: bool) -> str:
    if bold and italic:
        return "bolditalic"
    if bold:
        return "bold"
    if italic:
        return "italic"
    return "regular"


def slug(family: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", family.lower()).strip("-")


class FontRegistry:
    """Family -> style -> font file, plus cached skia typefaces."""

    def __init__(self, root: Path | str | None):
        self.root = Path(root) if root else None
        self._files: dict[str, dict[str, str]] = {}
        self._lower: dict[str, str] = {}
        self._lock = threading.Lock()
        self._typefaces: dict[tuple[str, str], skia.Typeface] = {}
        manifest = self.root / "manifest.json" if self.root else None
        if manifest and manifest.exists():
            self._files = json.loads(manifest.read_text())["families"]
        self._lower = {name.lower(): name for name in self._files}

    # ------------------------------------------------------------- lookup

    @property
    def available(self) -> bool:
        return bool(self._files)

    def families(self) -> list[str]:
        return sorted(self._files)

    def resolve(self, family: str | None) -> str:
        """Canonical family name, or the default when the family is unknown."""
        if not family:
            return DEFAULT_FAMILY if DEFAULT_FAMILY in self._files or not self._files else self.families()[0]
        name = str(family).strip().strip("'\"")
        low = name.replace("_", " ").lower()
        for suffix in (" bold", " regular", " light", " italic", " medium"):
            if low.endswith(suffix):
                low = low[: -len(suffix)]
        low = _ALIASES.get(low, low).lower()
        if low in self._lower:
            return self._lower[low]
        if not self._files:
            return name
        return DEFAULT_FAMILY if DEFAULT_FAMILY in self._files else self.families()[0]

    def knows(self, family: str | None) -> bool:
        if not family:
            return False
        low = str(family).strip().replace("_", " ").lower()
        low = _ALIASES.get(low, low).lower()
        return low in self._lower

    def styles(self, family: str) -> list[str]:
        return [s for s in STYLES if s in self._files.get(self.resolve(family), {})]

    def effective_style(self, family: str, bold: bool, italic: bool) -> str:
        """The face actually drawn, following `font-synthesis: none` semantics."""
        have = self._files.get(self.resolve(family), {})
        want = style_key(bold, italic)
        if want in have or not have:
            return want if want in have else "regular"
        # Browsers keep the closest face on the axis that exists.
        if want == "bolditalic":
            for alt in ("bold", "italic"):
                if alt in have:
                    return alt
        return "regular"

    def file_path(self, family: str, style: str) -> Path | None:
        if not self.root:
            return None
        rel = self._files.get(self.resolve(family), {}).get(style)
        return self.root / rel if rel else None

    def typeface(self, family: str, bold: bool = False, italic: bool = False) -> skia.Typeface:
        fam = self.resolve(family)
        style = self.effective_style(fam, bold, italic)
        key = (fam, style)
        with self._lock:
            cached = self._typefaces.get(key)
            if cached is not None:
                return cached
            path = self.file_path(fam, style)
            if path is not None and path.exists():
                tf = skia.Typeface.MakeFromData(skia.Data.MakeWithCopy(path.read_bytes()))
            else:
                tf = None
            if tf is None:
                # No pack (unit tests, a bare checkout): skia's default face keeps
                # layout deterministic, it just is not the right typeface.
                tf = skia.Typeface.MakeDefault()
            self._typefaces[key] = tf
            return tf

    # --------------------------------------------------------- measurement

    def font(self, family: str, size: float, bold: bool = False, italic: bool = False) -> skia.Font:
        font = skia.Font(self.typeface(family, bold, italic), max(1.0, float(size)))
        font.setSubpixel(True)
        font.setEdging(skia.Font.Edging.kAntiAlias)
        font.setHinting(skia.FontHinting.kNone)
        return font

    def advance(
        self,
        text: str,
        family: str,
        size: float,
        bold: bool = False,
        italic: bool = False,
        letter_spacing: float = 0.0,
    ) -> float:
        """Advance width of one line, with CSS letter-spacing (after every glyph)."""
        if not text:
            return 0.0
        font = self.font(family, size, bold, italic)
        glyphs = font.textToGlyphs(text)
        return float(sum(font.getWidths(glyphs))) + letter_spacing * len(text)

    def metrics(self, family: str, size: float, bold: bool = False, italic: bool = False):
        """(ascent, descent) as positive numbers, the values CSS `line-height: normal` uses."""
        m = self.font(family, size, bold, italic).getMetrics()
        return -m.fAscent, m.fDescent


@lru_cache(maxsize=4)
def registry_for(root: str | None) -> FontRegistry:
    return FontRegistry(root)
