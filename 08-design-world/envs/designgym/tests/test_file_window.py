"""The computer's file window: how images get into every layout.

Each layout opens it the way the real app does (File > Place Embedded..., Insert >
Pictures > This Device..., ...). It looks like macOS, Windows or Linux, can be
moved by its title bar and resized from its edges, opens files by button,
double-click or Enter, and files can be dragged out of it onto the canvas.
"""

from __future__ import annotations

import pytest

from .conftest import needs_pack
from .test_editor_e2e import FILES_OPENER, api, doc, idle, new_episode, open_editor, open_files

pytestmark = needs_pack

OSES = ("mac", "windows", "linux")


def _open(browser, server, skin, os_, theme="light", index=2):
    eid = new_episode(server, index=index, skin=skin)
    page, errors = open_editor(browser, server, eid, skin, theme=theme, size=(1280, 800))
    page.goto(page.url + f"&agent=1&os={os_}")
    page.wait_for_function("window.designEditor && window.designEditor.ready === true")
    idle(page)
    return eid, page, errors


def _box(page, sel=".fc"):
    b = page.locator(sel).first.bounding_box()
    return b["x"], b["y"], b["width"], b["height"]


def _grab(page):
    """An empty spot on the window's title bar (not a button)."""
    return page.evaluate("""() => {
        for (const bar of document.querySelectorAll('.fc .fc-drag')) {
            const r = bar.getBoundingClientRect(), y = r.top + Math.min(r.height / 2, 18);
            for (let x = r.left + r.width * 0.35; x < r.right - 8; x += 12) {
                const at = document.elementFromPoint(x, y);
                if (at && bar.contains(at) && !at.closest('button, input, select')) return [x, y];
            }
        }
        return null; }""")


def _drag(page, a, b, steps=10):
    page.mouse.move(*a)
    page.mouse.down()
    page.mouse.move(*b, steps=steps)
    page.mouse.up()


def _stage_center(page):
    r = page.locator(".stage").bounding_box()
    return r["x"] + r["width"] / 2, r["y"] + r["height"] / 2


@pytest.mark.parametrize("os_", OSES)
@pytest.mark.parametrize("theme", ["light", "dark"])
def test_file_window_moves_resizes_and_drops_files(browser, server, os_, theme):
    eid, page, errors = _open(browser, server, "slides", os_, theme)
    try:
        assets = api(server, f"/api/episodes/{eid}")["assets"]
        open_files(page, "slides")
        assert page.locator(f".fc.fc-{os_}").count() == 1
        names = page.locator(".fc .fc-item[data-type=file]").evaluate_all("els => els.map((e) => e.title)")
        assert sorted(n.split(".")[0] for n in names) == sorted(a["id"] for a in assets)

        # Move by the title bar.
        x, y, w, h = _box(page)
        g = _grab(page)
        assert g, "the title bar needs an empty spot to grab"
        _drag(page, g, (g[0] - 90, g[1] + 40))
        x2, y2, w2, h2 = _box(page)
        assert (round(x2 - x), round(y2 - y), w2, h2) == (-90, 40, w, h)

        # Resize from the bottom-right corner, and not below the minimum size.
        c = page.locator(".fc .fc-rz-se").bounding_box()
        corner = (c["x"] + c["width"] / 2, c["y"] + c["height"] / 2)
        _drag(page, corner, (corner[0] - 120, corner[1] - 70))
        _, _, w3, h3 = _box(page)
        assert (round(w - w3), round(h - h3)) == (120, 70)
        _drag(page, (corner[0] - 120, corner[1] - 70), (corner[0] - 900, corner[1] - 900))
        _, _, w4, h4 = _box(page)
        assert (w4, h4) == (460, 300)

        # Drag a file out onto the canvas: it lands centred on the drop point; the window stays open.
        aid = assets[0]["id"]
        page.evaluate("() => { const f = window.__app.fileChooser; f.box.x = 700; f.box.y = 380; f.place_(); }")
        item = page.locator(f'.fc .fc-item[data-file="{aid}"]').bounding_box()
        drop = (360, 300)
        _drag(page, (item["x"] + item["width"] / 2, item["y"] + item["height"] / 2), drop)
        idle(page)
        els = doc(server, eid)["elements"]
        assert [e["asset_id"] for e in els] == [aid]
        cx, cy = page.evaluate("([x, y]) => window.__app.stage.dropPoint(x, y)", list(drop))
        assert els[0]["x"] + els[0]["width"] / 2 == pytest.approx(cx, abs=1.5)
        assert els[0]["y"] + els[0]["height"] / 2 == pytest.approx(cy, abs=1.5)
        assert page.locator(".fc").count() == 1

        # Dropping back into the window does nothing.
        item = page.locator(f'.fc .fc-item[data-file="{aid}"]').bounding_box()
        _drag(page, (item["x"] + 20, item["y"] + 20), (item["x"] + 120, item["y"] + 40))
        idle(page)
        assert len(doc(server, eid)["elements"]) == 1

        page.keyboard.press("Escape")
        assert page.locator(".fc").count() == 0
        assert not errors, errors
    finally:
        page.close()


@pytest.mark.parametrize("skin", list(FILES_OPENER))
def test_every_layout_opens_images_through_its_own_route(browser, server, skin):
    eid, page, errors = _open(browser, server, skin, "mac")
    try:
        aid = api(server, f"/api/episodes/{eid}")["assets"][0]["id"]
        open_files(page, skin)
        page.locator(f'.fc .fc-item[data-file="{aid}"]').click()
        page.locator('.fc [data-fc="open"]').click()
        assert page.locator(".fc").count() == 0
        idle(page)
        if skin == "figma":
            assert doc(server, eid)["elements"] == [] and page.locator(".stage.placing").count() == 1
            # Drag out the box: the image keeps its proportions.
            cx, cy = _stage_center(page)
            _drag(page, (cx - 100, cy - 80), (cx + 60, cy + 200))
            idle(page)
            el = doc(server, eid)["elements"][0]
            zoom = page.evaluate("window.__app.stage.zoom")
            aspect = api(server, f"/api/episodes/{eid}")["assets"][0]["aspect"]
            assert el["width"] == pytest.approx(160 / zoom, abs=2)
            assert el["width"] / el["height"] == pytest.approx(aspect, rel=0.02)
            assert page.locator(".stage.placing").count() == 0
        elif skin == "inkscape":
            assert page.locator(".ik-import").count() == 1 and doc(server, eid)["elements"] == []
            page.click(".ik-import .btn-primary")
            idle(page)
            assert [e["asset_id"] for e in doc(server, eid)["elements"]] == [aid]
        elif skin == "canva":
            assert doc(server, eid)["elements"] == []  # uploading does not place it
            page.click(f'.asset-tile[data-asset="{aid}"]')
            idle(page)
            assert [e["asset_id"] for e in doc(server, eid)["elements"]] == [aid]
        else:
            assert [e["asset_id"] for e in doc(server, eid)["elements"]] == [aid]
        assert not errors, errors
    finally:
        page.close()


def test_import_can_be_cancelled_and_windows_accepts_a_typed_name(browser, server):
    eid, page, errors = _open(browser, server, "inkscape", "windows")
    try:
        aid = api(server, f"/api/episodes/{eid}")["assets"][0]["id"]
        open_files(page, "inkscape")
        page.fill(".fc .fc-name", f"{aid}")
        page.keyboard.press("Enter")
        page.locator(".ik-import").wait_for()
        page.click('.ik-import .btn:not(.btn-primary)')
        idle(page)
        assert doc(server, eid)["elements"] == []
        assert not errors, errors
    finally:
        page.close()


def test_template_uploads_many_at_once(browser, server):
    eid, page, errors = _open(browser, server, "canva", "linux")
    try:
        ids = [a["id"] for a in api(server, f"/api/episodes/{eid}")["assets"]]
        page.evaluate("() => window.__app.uploaded.clear()")
        open_files(page, "canva")
        page.locator(".fc .fc-item[data-type=file]").first.click()
        page.keyboard.press("Control+a")
        page.locator('.fc [data-fc="open"]').click()
        idle(page)
        tiles = page.locator(".asset-tile").evaluate_all("els => els.map((e) => e.dataset.asset)")
        assert sorted(tiles) == sorted(ids)
        assert doc(server, eid)["elements"] == []
        assert not errors, errors
    finally:
        page.close()
