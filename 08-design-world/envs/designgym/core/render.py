"""Server-side renderer: one document in, pixels out, in about ten milliseconds.

This is the renderer every reward is computed from. The target a model is asked
to reproduce is `render(ground_truth)`, never the source's shipped preview, so a
perfect reconstruction scores exactly 1.0 and renderer quirks cancel out.

The browser editor draws the same document with CSS. The two agree because the
text rules in `text.py` are the CSS rules, and because the editor turns off the
browser features skia does not have (kerning, ligatures, font synthesis). They
are not byte-identical; `scripts/parity.py` measures by how much.
"""

from __future__ import annotations

import io
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
import skia
from PIL import Image as PILImage

from .colors import to_skia
from .doc import DesignDoc, Element
from .fonts import FontRegistry
from .text import layout

_SAMPLING = skia.SamplingOptions(skia.FilterMode.kLinear, skia.MipmapMode.kLinear)


class ImageCache:
    """Decoded assets, shared across sessions: every episode of a task reuses them."""

    def __init__(self, capacity: int = 512):
        self._items: OrderedDict[str, skia.Image] = OrderedDict()
        self._lock = threading.Lock()
        self.capacity = capacity

    def get(self, path: Path | str) -> skia.Image:
        key = str(path)
        with self._lock:
            img = self._items.get(key)
            if img is not None:
                self._items.move_to_end(key)
                return img
        with PILImage.open(key) as im:
            arr = np.asarray(im.convert("RGBA"))
        img = skia.Image.fromarray(
            np.ascontiguousarray(arr), colorType=skia.ColorType.kRGBA_8888_ColorType
        )
        with self._lock:
            self._items[key] = img
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)
        return img


_IMAGES = ImageCache()


def scale_for(doc: DesignDoc, max_size: int | None) -> float:
    if not max_size:
        return 1.0
    return min(1.0, max_size / max(doc.width, doc.height))


class Renderer:
    def __init__(self, fonts: FontRegistry, images: ImageCache | None = None):
        self.fonts = fonts
        self.images = images or _IMAGES

    # ------------------------------------------------------------ public

    def render_image(
        self, doc: DesignDoc, assets: dict[str, Path | str], max_size: int | None = 512
    ) -> skia.Image:
        s = scale_for(doc, max_size)
        w, h = max(1, round(doc.width * s)), max(1, round(doc.height * s))
        surface = skia.Surface(w, h)
        canvas = surface.getCanvas()
        canvas.clear(skia.ColorWHITE)
        canvas.scale(s, s)
        canvas.drawColor(to_skia(doc.background))
        for el in doc.elements:
            if el.visible and el.opacity > 0:
                self._draw(canvas, el, assets)
        return surface.makeImageSnapshot()

    def render_png(self, doc: DesignDoc, assets, max_size: int | None = 512) -> bytes:
        return bytes(self.render_image(doc, assets, max_size).encodeToData(skia.kPNG, 100))

    def render_jpeg(self, doc: DesignDoc, assets, max_size: int | None = 512, quality=88) -> bytes:
        img = self.render_image(doc, assets, max_size)
        return bytes(img.encodeToData(skia.kJPEG, quality))

    def render_array(self, doc: DesignDoc, assets, max_size: int | None = 256) -> np.ndarray:
        """HxWx3 uint8 RGB, the scorer's input."""
        img = self.render_image(doc, assets, max_size)
        arr = img.toarray(colorType=skia.ColorType.kRGBA_8888_ColorType)
        return np.ascontiguousarray(arr[:, :, :3])

    # ----------------------------------------------------------- drawing

    def _draw(self, canvas: skia.Canvas, el: Element, assets) -> None:
        canvas.save()
        canvas.translate(el.x, el.y)
        cx, cy = el.width / 2, el.height / 2
        if el.angle:
            canvas.rotate(el.angle, cx, cy)
        if el.flip_x or el.flip_y:
            canvas.translate(cx, cy)
            canvas.scale(-1 if el.flip_x else 1, -1 if el.flip_y else 1)
            canvas.translate(-cx, -cy)
        group = el.opacity < 1.0 and el.kind != "image"
        if group:
            # CSS opacity composites the element as one layer; a per-paint alpha
            # would show a shape's fill through its own stroke.
            canvas.saveLayerAlpha(None, round(el.opacity * 255))
        if el.kind == "image":
            self._draw_image(canvas, el, assets)
        elif el.kind == "shape":
            self._draw_shape(canvas, el)
        else:
            self._draw_text(canvas, el)
        if group:
            canvas.restore()
        canvas.restore()

    def _draw_image(self, canvas, el: Element, assets) -> None:
        path = assets.get(el.asset_id)
        if path is None:
            # A missing asset is drawn as a hatched placeholder rather than
            # silently skipped, so a model sees that its reference was wrong.
            paint = skia.Paint(Color=0x55FF00FF, AntiAlias=True)
            canvas.drawRect(skia.Rect.MakeWH(el.width, el.height), paint)
            return
        img = self.images.get(path)
        paint = skia.Paint(AntiAlias=True)
        paint.setAlphaf(el.opacity)
        iw, ih = img.width(), img.height()
        l, t, r, b = el.crop or (0.0, 0.0, 0.0, 0.0)  # (a crop shows only that part of the picture)
        canvas.drawImageRect(
            img,
            skia.Rect.MakeLTRB(l * iw, t * ih, (1 - r) * iw, (1 - b) * ih),
            skia.Rect.MakeWH(el.width, el.height),
            _SAMPLING,
            paint,
        )

    def _draw_shape(self, canvas, el: Element) -> None:
        rect = skia.Rect.MakeWH(el.width, el.height)
        sw = el.stroke_width if el.stroke else 0.0

        def outline(r: skia.Rect):
            if el.shape == "ellipse":
                return ("oval", r)
            radius = max(0.0, min(el.radius, r.width() / 2, r.height() / 2))
            return ("rrect", skia.RRect.MakeRectXY(r, radius, radius))

        if el.fill:
            paint = skia.Paint(AntiAlias=True, Color=to_skia(el.fill))
            kind, geom = outline(rect)
            canvas.drawOval(geom, paint) if kind == "oval" else canvas.drawRRect(geom, paint)
        if sw > 0:
            # CSS borders sit inside a border-box; inset so the stroke does too.
            inner = skia.Rect.MakeLTRB(sw / 2, sw / 2, el.width - sw / 2, el.height - sw / 2)
            paint = skia.Paint(
                AntiAlias=True,
                Color=to_skia(el.stroke),
                Style=skia.Paint.kStroke_Style,
                StrokeWidth=sw,
            )
            kind, geom = outline(inner)
            if kind == "rrect":
                radius = max(0.0, min(el.radius - sw / 2, inner.width() / 2, inner.height() / 2))
                geom = skia.RRect.MakeRectXY(inner, radius, radius)
            canvas.drawOval(geom, paint) if kind == "oval" else canvas.drawRRect(geom, paint)

    def _draw_text(self, canvas, el: Element) -> None:
        lay = layout(el, self.fonts)
        family = el.font_family or ""
        for line in lay.lines:
            x = line.x
            for piece in line.pieces:
                font = self.fonts.font(family, el.font_size, piece.bold, piece.italic)
                glyphs = font.textToGlyphs(piece.text)
                widths = font.getWidths(glyphs)
                positions = []
                cursor = 0.0
                for w in widths:
                    positions.append(cursor)
                    cursor += w + el.letter_spacing
                builder = skia.TextBlobBuilder()
                builder.allocRunPosH(font, glyphs, positions, 0)
                blob = builder.make()
                paint = skia.Paint(AntiAlias=True, Color=to_skia(piece.color))
                if blob is not None:
                    canvas.drawTextBlob(blob, x, line.baseline, paint)
                if el.underline:
                    thick = max(1.0, el.font_size / 15)
                    y = line.baseline + el.font_size * 0.1
                    canvas.drawRect(skia.Rect.MakeXYWH(x, y, piece.width, thick), paint)
                x += piece.width


def png_to_pil(data: bytes) -> PILImage.Image:
    return PILImage.open(io.BytesIO(data))
