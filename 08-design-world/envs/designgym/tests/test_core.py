"""Document model, operations, HTML profile and colour parsing: no pack needed."""

from __future__ import annotations

import pytest

from designgym.core import ops
from designgym.core.colors import delta_e, parse_color
from designgym.core.doc import DesignDoc, Element
from designgym.core.fonts import FontRegistry
from designgym.core.html_profile import HTMLProfileError, doc_to_html, html_to_doc


@pytest.fixture
def ctx():
    return ops.OpContext(FontRegistry(None), {"a1": {"aspect": 2.0, "kind": "photo"},
                                             "a2": {"aspect": 0.5, "kind": "graphic"}})


@pytest.fixture
def doc():
    return DesignDoc(width=1000, height=800)


def test_colors():
    assert parse_color("rgba(255, 102, 51, 1)") == "#ff6633"
    assert parse_color("rgba(0,0,0,0.5)") == "#00000080"
    assert parse_color("#FFF") == "#ffffff"
    assert parse_color("white") == "#ffffff"
    assert parse_color("#11223344") == "#11223344"
    assert parse_color("#112233ff") == "#112233"
    with pytest.raises(ValueError):
        parse_color("notacolor")
    assert delta_e("#000000", "#000000") == 0
    assert delta_e("#000000", "#ffffff") > 99


def test_ops_do_not_mutate_input(doc, ctx):
    new, res = ops.apply(doc, "add_image", {"asset_id": "a1", "x": 10, "y": 20, "width": 200}, ctx)
    assert doc.elements == []
    el = new.find(res["element_id"])
    assert (el.width, el.height) == (200, 100)  # aspect kept


def test_add_image_defaults_and_errors(doc, ctx):
    new, res = ops.apply(doc, "add_image", {"asset_id": "a2"}, ctx)
    el = new.find(res["element_id"])
    assert el.width / el.height == pytest.approx(0.5)
    assert el.x == pytest.approx((1000 - el.width) / 2)
    with pytest.raises(ops.OpError, match="unknown asset"):
        ops.apply(doc, "add_image", {"asset_id": "zz"}, ctx)
    with pytest.raises(ops.OpError, match="width"):
        ops.apply(doc, "add_image", {"asset_id": "a1", "width": -5}, ctx)


def test_text_height_follows_lines(doc, ctx):
    new, res = ops.apply(doc, "add_text", {"text": "a\nb\nc", "font_size": 40, "line_height": 1.5}, ctx)
    el = new.find(res["element_id"])
    assert el.height == pytest.approx(3 * 40 * 1.5)
    assert el.auto_width
    new2, _ = ops.apply(new, "update_element", {"element_id": el.id, "text": "one line"}, ctx)
    assert new2.find(el.id).height == pytest.approx(60)


def test_update_rejects_wrong_props(doc, ctx):
    new, res = ops.apply(doc, "add_image", {"asset_id": "a1"}, ctx)
    with pytest.raises(ops.OpError, match="image elements have no font_size"):
        ops.apply(new, "update_element", {"element_id": res["element_id"], "font_size": 3}, ctx)
    with pytest.raises(ops.OpError, match="no element"):
        ops.apply(new, "update_element", {"element_id": "e99", "x": 3}, ctx)


def test_reorder_align_duplicate_delete(doc, ctx):
    d = doc
    ids = []
    for x in (0, 100, 300):
        d, r = ops.apply(d, "add_shape", {"x": x, "y": 0, "width": 50, "height": 50}, ctx)
        ids.append(r["element_id"])
    d, _ = ops.apply(d, "reorder_element", {"element_id": ids[0], "to": "front"}, ctx)
    assert [e.id for e in d.elements] == [ids[1], ids[2], ids[0]]
    d, _ = ops.apply(d, "reorder_element", {"element_id": ids[0], "to": "0"}, ctx)
    assert d.elements[0].id == ids[0]
    d, _ = ops.apply(d, "align_elements", {"element_ids": [ids[1]], "alignment": "center"}, ctx)
    assert d.find(ids[1]).x == pytest.approx(475)
    d, _ = ops.apply(d, "align_elements", {"element_ids": ids, "alignment": "top",
                                           "relative_to": "selection"}, ctx)
    d, r = ops.apply(d, "duplicate_element", {"element_id": ids[2]}, ctx)
    assert d.index_of(r["element_id"]) == d.index_of(ids[2]) + 1
    d, _ = ops.apply(d, "delete_element", {"element_id": ids[2]}, ctx)
    assert len(d.elements) == 3


def test_rotated_align_uses_bounds(doc, ctx):
    d, r = ops.apply(doc, "add_shape", {"x": 100, "y": 100, "width": 200, "height": 100, "angle": 90}, ctx)
    d, _ = ops.apply(d, "align_elements", {"element_ids": [r["element_id"]], "alignment": "left"}, ctx)
    el = d.find(r["element_id"])
    # A 200x100 box rotated 90 degrees is 100 wide; its visual left edge is at x+50.
    assert el.x + 50 == pytest.approx(0)


def test_font_validation(doc):
    reg = FontRegistry(None)
    ctx = ops.OpContext(reg, {})
    # Without a font pack any family is accepted (tests run anywhere).
    d, r = ops.apply(doc, "add_text", {"text": "x", "font_family": "Whatever"}, ctx)
    assert d.find(r["element_id"]).font_family == "Whatever"


def test_runs_join_to_text(doc, ctx):
    d, r = ops.apply(doc, "add_text", {"text": "ignored", "runs": [
        {"text": "BUY "}, {"text": "NOW", "color": "#ff0000", "bold": True}]}, ctx)
    el = d.find(r["element_id"])
    assert el.text == "BUY NOW"
    assert el.styled_runs()[1].color == "#ff0000"
    d2, _ = ops.apply(d, "update_element", {"element_id": el.id, "text": "plain"}, ctx)
    assert d2.find(el.id).runs is None


def test_html_round_trip(doc, ctx):
    d = doc
    d, _ = ops.apply(d, "set_background", {"color": "#f4efe6"}, ctx)
    d, _ = ops.apply(d, "add_image", {"asset_id": "a1", "x": 0, "y": 0, "width": 1000, "angle": 5}, ctx)
    d, _ = ops.apply(d, "add_shape", {"shape": "ellipse", "x": 10, "y": 10, "width": 80, "height": 80,
                                      "fill": "#e63946", "stroke": "#000000", "stroke_width": 3}, ctx)
    d, _ = ops.apply(d, "add_text", {"text": " Summer\nSale ", "x": 50, "y": 600, "width": 900,
                                     "font_size": 72, "color": "#1d1d1f", "align": "center",
                                     "letter_spacing": 2, "uppercase": True, "line_height": 1.2,
                                     "runs": [{"text": " Summer\n"}, {"text": "Sale ", "color": "#ff0000"}]}, ctx)
    html = doc_to_html(d)
    parsed, warnings = html_to_doc(html, 1000, 800)
    assert warnings == []
    back, _ = ops.apply(DesignDoc(width=1000, height=800), "replace_document", {"document": parsed}, ctx)
    assert back.public() == d.public()


def test_html_hand_written(ctx):
    page = """
    <html><body>
    <div id="canvas" style="width:1000px;height:800px;background:#123456">
      <img src="asset://a1" style="left:0;top:0;width:500px;height:250px">
      <div style="left:40px;top:300px;width:920px;font-family:'Montserrat';font-size:64px;
                  color:#fff;text-align:center;font-weight:bold">
        Big <span style="color:#ff0">Sale</span><br>today
      </div>
      <div style="left:10px;top:10px;width:30px;height:30px;background:red;border-radius:50%;filter:blur(2px)"></div>
    </div></body></html>"""
    parsed, warnings = html_to_doc(page, 1000, 800)
    kinds = [e["kind"] for e in parsed["elements"]]
    assert kinds == ["image", "text", "shape"]
    text = parsed["elements"][1]
    assert "".join(r["text"] for r in text["runs"]) == "Big Sale\ntoday"
    assert parsed["elements"][2]["shape"] == "ellipse"
    assert any("filter" in w for w in warnings)
    assert parsed["background"] == "#123456"


def test_html_errors():
    with pytest.raises(HTMLProfileError):
        html_to_doc("", 100, 100)
    with pytest.raises(HTMLProfileError, match="asset"):
        html_to_doc('<div id="canvas"><img src="cat.png" style="width:1px;height:1px"></div>', 100, 100)
    with pytest.raises(HTMLProfileError, match="width and height"):
        html_to_doc('<div id="canvas"><div style="left:1px;background:#fff"></div></div>', 100, 100)


def test_element_validation():
    with pytest.raises(ValueError):
        Element(id="e1", kind="image")
    el = Element(id="e1", kind="shape", angle=270)
    assert el.angle == -90


def test_key_tool_accepts_the_shapes_models_send():
    from designgym.server.environment import _key_sequence

    assert _key_sequence("enter") == ["enter"]
    assert _key_sequence(["ctrl+a", "delete"]) == ["ctrl+a", "delete"]
    assert _key_sequence('["enter"]') == ["enter"]          # a JSON list sent as a string
    assert _key_sequence('["ctrl+z", "escape"]') == ["ctrl+z", "escape"]
    assert _key_sequence("") == []


def test_gui_key_names_normalise():
    from designgym.server.gui import normalize_keys

    assert normalize_keys("ctrl+shift+z") == "ControlOrMeta+Shift+z"
    assert normalize_keys("enter") == "Enter"
    assert normalize_keys("esc") == "Escape"
    assert normalize_keys("shift+up") == "Shift+ArrowUp"
    assert normalize_keys("Arrowdown") == "ArrowDown"  # models mangle the case of named keys


def _write(ctx, page, w=1080, h=1080):
    parsed, warnings = html_to_doc(page, w, h)
    doc, _ = ops.apply(DesignDoc(width=w, height=h), "replace_document", {"document": parsed}, ctx)
    return doc, warnings


def test_html_stylesheet_classes_and_specificity(ctx):
    doc, warnings = _write(ctx, """<html><head><style>
        #canvas { width:1080px; height:1080px; background: ivory; }
        .photo { left:0; top:0; width:1080px; height:540px; }
        div { color: #111; }
        .title { left:80px; top:600px; font-size:96px; color: tomato !important; }
        #canvas .tag { left:80px; top:760px; font-size:32px; }
        a:hover { color: red; }
      </style></head><body><div id="canvas">
        <img class="photo" src="asset://a1">
        <div class="title">Summer</div>
        <div class="tag" style="color:#123456">Sale</div>
      </div></body></html>""")
    assert doc.background == "#fffff0"
    img, title, tag = doc.elements
    assert (img.width, img.height) == (1080, 540)
    assert (title.x, title.y, title.font_size, title.color) == (80, 600, 96, "#ff6347")
    assert tag.color == "#123456"  # inline beats the stylesheet
    assert any("a:hover" in w for w in warnings)


def test_html_wrappers_inherit_offset_and_text_style(ctx):
    doc, _ = _write(ctx, """<div id="canvas" style="width:1080px;height:1080px;font-family:Montserrat">
        <section style="left:100px;top:200px;color:navy;font-size:30px">
          <img src="asset://a1" style="left:0;top:0;width:500px;height:250px">
          <p style="left:10px;top:270px">Caption</p>
        </section></div>""")
    img, cap = doc.elements
    assert (img.x, img.y) == (100, 200)
    assert (cap.x, cap.y, cap.color, cap.font_size) == (110, 470, "#000080", 30)


def test_html_centering_anchoring_and_zindex(ctx):
    doc, _ = _write(ctx, """<div id="canvas" style="width:1000px;height:1000px">
        <div style="left:50%;top:50%;width:200px;height:100px;transform:translate(-50%,-50%);background:red;z-index:3"></div>
        <div style="right:40px;bottom:40px;width:100px;height:50px;background:blue;z-index:1"></div>
        <div style="left:50%;top:10px;transform:translateX(-50%);font-size:40px">Title</div></div>""", 1000, 1000)
    title, blue, red = doc.elements  # z-index order: title (auto), blue (1), red (3)
    assert (red.x, red.y) == (400, 450)
    assert (blue.x, blue.y) == (860, 910)
    assert title.x + title.width / 2 == pytest.approx(500, abs=0.6)
    assert doc.elements[-1].fill == "#ff0000"  # highest z-index is drawn last


def test_html_whitespace_follows_css(ctx):
    normal, _ = _write(ctx, """<div id="canvas" style="width:1000px;height:1000px">
        <div style="left:0;top:0">
          Big summer
          sale<br>today
        </div></div>""")
    assert normal.elements[0].text == "Big summer sale\ntoday"
    pre, _ = _write(ctx, """<div id="canvas" style="width:1000px;height:1000px">
        <div style="left:0;top:0;white-space:pre">Big
sale</div></div>""")
    assert pre.elements[0].text == "Big\nsale"


def test_html_units_colours_and_warnings(ctx):
    doc, warnings = _write(ctx, """<div id="canvas" style="width:1000px;height:500px;background:hsl(210,50%,20%);display:flex">
        <div style="left:10vw;top:2rem;width:25%;height:10vh;background:linear-gradient(90deg, gold, red)"></div>
        <img src="asset://a1" style="left:1em;top:0;width:12pt;height:12pt;object-fit:cover;filter:blur(2px)"></div>""", 1000, 500)
    shape, img = doc.elements
    assert (shape.x, shape.y, shape.width, shape.height) == (100, 32, 250, 50)
    assert shape.fill == "#ffd700"
    assert img.width == pytest.approx(16)
    joined = " ".join(warnings)
    for needle in ("display:flex", "gradients", "object-fit", "filter"):
        assert needle in joined, needle
