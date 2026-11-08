"""The desktop-app skins (impress, draw, inkscape-desktop): converters that run without the desktop container.

SVG parsing (Inkscape's documents) and the baseline/top conversions that place text exactly through our own layout
are pure Python; the apps themselves are exercised by realapps/desktop round trips against a running agent.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from designgym.core.doc import Element
from designgym.realapps.desktop.client import from_anchor, from_top, text_anchor, text_top

AGENT = Path(__file__).resolve().parents[1] / "realapps" / "desktop" / "agent.py"
spec = importlib.util.spec_from_file_location("desktop_agent", AGENT)
agent = importlib.util.module_from_spec(spec)
sys.modules["desktop_agent"] = agent
spec.loader.exec_module(agent)


class _Matcher:
    sig = {"a1": None, "a2": None}

    def best(self, img):
        return "a1"


def test_svg_images_shapes_and_styled_text_parse():
    svg = agent._svg(400, 300, "#123456", [
        {"kind": "image", "asset_id": "a2", "x": 10, "y": 20, "width": 100, "height": 50, "angle": 30},
        {"kind": "shape", "shape": "ellipse", "x": 200, "y": 100, "width": 80, "height": 40, "fill": "#ff0000"},
        {"kind": "text", "text": "eco\nlain", "x": 50, "y": 150, "width": 120, "height": 60, "font_size": 20,
         "font_family": "Prompt", "align": "center", "line_height": 1.5,
         "runs": [{"text": "eco\n", "color": "#000001"}, {"text": "lain", "color": "#91b533"}],
         "_baseline": 170.0, "_anchor_x": 110.0, "_step": 30.0},
    ], {"a2": Path("/home/user/Pictures/a2.png")})
    out = agent.parse_svg(svg, (400, 300), _Matcher())
    assert out["background"] == "#123456" and out["skipped"] == 0
    img, ell, txt = out["elements"]
    assert img["asset_id"] == "a2" and img["angle"] == pytest.approx(30)
    assert (img["x"], img["y"], img["width"], img["height"]) == pytest.approx((10, 20, 100, 50))
    assert ell["shape"] == "ellipse" and ell["fill"] == "#ff0000"
    assert (ell["x"], ell["y"], ell["width"], ell["height"]) == pytest.approx((200, 100, 80, 40))
    assert txt["text"] == "eco\nlain" and txt["align"] == "center"
    assert [r["color"] for r in txt["runs"] if r["text"].strip()] == ["#000001", "#91b533"]
    assert (txt["_anchor_x"], txt["_baseline"], txt["line_height"]) == pytest.approx((110, 170, 1.5))


def test_svg_transforms_compose():
    m = agent._mul(agent._matrix("translate(100,50)"), agent._matrix("rotate(90)"))
    placed = agent._place(m, 0, 0, 20, 10)
    assert placed["angle"] == pytest.approx(90)
    assert (placed["width"], placed["height"]) == pytest.approx((20, 10))
    # the box centre (10, 5) maps to (100 - 5, 50 + 10)
    assert (placed["x"] + 10, placed["y"] + 5) == pytest.approx((95, 60))


@pytest.mark.parametrize("align", ["left", "center", "right"])
def test_text_by_baseline_and_by_top_invert_exactly(pack, align):
    e = Element(id="t", kind="text", text="Two lines\nof copy", x=37.5, y=80.25, width=300, height=10,
                font_family="Montserrat", font_size=28, line_height=1.4, align=align, auto_width=True)
    from designgym.core.text import fit_box

    e.width, e.height = fit_box(e, pack.fonts)
    raw = {k: v for k, v in e.model_dump().items() if k not in ("id", "x", "y", "width", "height")}
    back = from_anchor({**raw, **{k: v for k, v in text_anchor(e, pack.fonts).items()}}, pack.fonts)
    assert (back["x"], back["y"], back["width"]) == pytest.approx((e.x, e.y, e.width), abs=0.01)
    top = text_top(e, pack.fonts)
    back2 = from_top({**raw, "x": e.x, "width": e.width, "height": 1.0, "_top": top["_top"]}, pack.fonts)
    assert back2["y"] == pytest.approx(e.y, abs=0.01)


@pytest.mark.parametrize("surface", ["tools", "html"])
def test_real_apps_are_computer_use_only(pack, surface):
    from designgym.server.environment import DesignCanvasEnvironment
    from designgym.server.episodes import STORE

    before = len(STORE.recent(10**6))
    with pytest.raises(ValueError, match="real app"):
        DesignCanvasEnvironment(pack).reset(split=pack.splits()[0], index=0, surface=surface, skin="draw")
    assert len(STORE.recent(10**6)) == before  # rejected before any episode (or app) was set up


def test_libreoffice_automatic_text_colour_reads_as_drawn():
    """Text left on LibreOffice's Automatic colour (CharColor -1) is drawn black on a light page and white on a
    dark one: the read-back says that colour, not black regardless, so a model that leaves the colour on Automatic
    over a dark page is not marked wrong for it. The boundary is the page's relative luminance at ~0.345 (measured
    on LibreOffice 25.2.3 in Impress and Draw; the 8-bit luminance does not predict it: pure green draws black)."""
    from designgym.realapps.desktop import agent

    auto = lambda bg: agent._hexcolor(-1, agent._auto_color(bg))  # noqa: E731
    assert auto("#ffffff") == "#000000" and auto("#000000") == "#ffffff"
    assert auto("#9e9e9e") == "#ffffff" and auto("#9f9f9f") == "#000000"  # the measured boundary, greys
    assert auto("#00b800") == "#ffffff" and auto("#00bc00") == "#000000"  # ... and greens
    assert auto("#ff0000") == "#ffffff" and auto("#00ff00") == "#000000" and auto("#ff00ff") == "#ffffff"
    assert auto("#443e40") == "#ffffff"  # a dark brown page (train-01868): white, as seen on screen
    assert agent._hexcolor(0xFF0000, agent._auto_color("#000000")) == "#ff0000"  # an explicit colour is itself
    assert abs(agent._luminance("#ffffff") - 1.0) < 1e-9 and agent._luminance("#000000") == 0.0
