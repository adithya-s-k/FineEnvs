"""Contact sheet: every asset of a task in one labelled image.

A vision model on the tools surface cannot open files, so it needs to *see*
which asset is which before it can place `a3` on the canvas. One image with a
labelled tile per asset is far cheaper than one tool call per asset, and the
checkerboard shows which assets are cut-outs and which are opaque photos.
"""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

TILE = 180
LABEL = 26
PAD = 10


def _checker(size: tuple[int, int], cell: int = 10) -> Image.Image:
    img = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for y in range(0, size[1], cell):
        for x in range(0, size[0], cell):
            if (x // cell + y // cell) % 2:
                draw.rectangle([x, y, x + cell - 1, y + cell - 1], fill=(228, 228, 228))
    return img


def contact_sheet(assets: list[dict], paths: dict[str, Path], font_path: Path | None = None,
                  cols: int = 4) -> bytes:
    n = len(assets)
    cols = max(1, min(cols, n))
    rows = (n + cols - 1) // cols
    width = cols * (TILE + PAD) + PAD
    height = rows * (TILE + LABEL + PAD) + PAD
    sheet = Image.new("RGB", (width, max(height, TILE)), (246, 246, 248))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype(str(font_path), 16) if font_path else ImageFont.load_default(16)
    except Exception:
        font = ImageFont.load_default()
    for i, asset in enumerate(assets):
        r, c = divmod(i, cols)
        x0 = PAD + c * (TILE + PAD)
        y0 = PAD + r * (TILE + LABEL + PAD)
        tile = _checker((TILE, TILE))
        with Image.open(paths[asset["id"]]) as im:
            im = im.convert("RGBA")
            im.thumbnail((TILE - 8, TILE - 8))
            tile.paste(im, ((TILE - im.width) // 2, (TILE - im.height) // 2), im)
        sheet.paste(tile, (x0, y0))
        draw.rectangle([x0, y0, x0 + TILE - 1, y0 + TILE - 1], outline=(200, 200, 205))
        label = f"{asset['id']}  {asset['kind']}  {asset['aspect']:.2f}:1"
        draw.text((x0 + 2, y0 + TILE + 4), label, fill=(30, 30, 35), font=font)
    buf = io.BytesIO()
    sheet.save(buf, "PNG", optimize=True)
    return buf.getvalue()
