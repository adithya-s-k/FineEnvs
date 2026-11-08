"""The pack's fonts as the desktop apps should see them: each file named for the family the designs use.

The pack maps a design's family (e.g. "Playlist Script") to font files, but a file's own name table can say
something else ("Playlist", style "Script"; "Brusher Free Font" is "Brusher"; Oswald's bold file is "Oswald
SemiBold"). The env's renderer reads files by path, so it never notices; an app asked for "Playlist Script" finds no
such family and draws a stand-in, while the saved document still names the right family. The apps get a copy whose
name tables say exactly the pack's family and the slot's style (Regular / Bold / Italic / Bold Italic), so every app
offers, matches and draws the face the env scores with.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

VERSION = "2"  # bump when the renaming changes: the copy is rebuilt
STYLE = {"regular": ("Regular", 400, False, False), "bold": ("Bold", 700, True, False),
         "italic": ("Italic", 400, False, True), "bolditalic": ("Bold Italic", 700, True, True)}


def _ps(s: str) -> str:
    return "".join(c for c in s if c.isalnum() or c == "-")[:63] or "Font"


def _rename(src: Path, dst: Path, family: str, style: str) -> None:
    from fontTools.ttLib import TTFont

    name, weight, bold, italic = STYLE[style]
    f = TTFont(src)
    table = f["name"]
    for nid in (16, 17, 21, 22):  # typographic / WWS names would let an app regroup the faces its own way
        table.removeNames(nameID=nid)
    full = family if style == "regular" else f"{family} {name}"
    for nid, value in ((1, family), (2, name), (4, full), (6, _ps(f"{family}-{name.replace(' ', '')}"))):
        table.setName(value, nid, 3, 1, 0x409)  # Windows, Unicode BMP, en-US
        table.setName(value, nid, 1, 0, 0)  # Mac Roman: some readers look here first
    if "OS/2" in f:
        os2 = f["OS/2"]
        os2.usWeightClass = weight
        sel = os2.fsSelection & ~0b1100001  # clear italic (0), bold (5), regular (6)
        if os2.version < 4:
            sel &= 0x7F  # bits 7-9 are not defined before OS/2 version 4
        os2.fsSelection = sel | (1 if italic else 0) | (1 << 5 if bold else 0) | (1 << 6 if not (bold or italic) else 0)
    f["head"].macStyle = (f["head"].macStyle & ~0b11) | (1 if bold else 0) | (2 if italic else 0)
    dst.parent.mkdir(parents=True, exist_ok=True)
    f.save(dst)


def app_fonts(src: Path) -> Path:
    """`src` (a pack fonts folder with manifest.json) -> its renamed copy beside it (built once per version)."""
    src = Path(src)
    manifest = src / "manifest.json"
    if not manifest.exists():
        return src
    dst = src.parent / f"{src.name}-apps"
    stamp = dst / ".stamp"
    key = f"{VERSION}:{manifest.stat().st_mtime_ns}:{manifest.stat().st_size}"
    if stamp.exists() and stamp.read_text() == key:
        return dst
    tmp = src.parent / f"{src.name}-apps.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    families = json.loads(manifest.read_text())["families"]
    for family, styles in families.items():
        for style, rel in styles.items():
            if style in STYLE:
                _rename(src / rel, tmp / rel, family, style)
    shutil.copy2(manifest, tmp / "manifest.json")
    (tmp / ".stamp").write_text(key)
    shutil.rmtree(dst, ignore_errors=True)
    tmp.rename(dst)
    return dst
