"""Legibility audit: every piece of text and every icon, in every UI state, in every look.

A computer-use agent only sees pixels. Text drawn in its own background colour
(blue on blue, white on near-white) is a control the agent cannot find, and it
fails silently: the page works, the screenshot is simply missing a label. This
walks each layout x theme through the states an agent reaches (selections,
popovers, menus, panels, dialogs) and measures the contrast of every visible
glyph against the background actually behind it.
"""

from __future__ import annotations

import pytest

from .conftest import needs_pack
from .test_editor_e2e import SKINS, api, idle, new_episode, open_editor

pytestmark = needs_pack
pytest.importorskip("playwright.sync_api")

MIN_CONTRAST = 2.5  # catches invisible text (~1:1) while allowing muted labels and brand blues

AUDIT_JS = r"""
(label) => {
  const parse = (c) => { const m = c.match(/[\d.]+/g); if (!m) return null;
    return { r: +m[0], g: +m[1], b: +m[2], a: m.length > 3 ? +m[3] : 1 }; };
  const lum = (c) => { const f = (v) => { v /= 255; return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; };
    return 0.2126 * f(c.r) + 0.7152 * f(c.g) + 0.0722 * f(c.b); };
  const ratio = (a, b) => { const x = lum(a), y = lum(b); return (Math.max(x, y) + 0.05) / (Math.min(x, y) + 0.05); };
  const over = (top, under) => ({ r: top.r * top.a + under.r * (1 - top.a), g: top.g * top.a + under.g * (1 - top.a),
                                   b: top.b * top.a + under.b * (1 - top.a), a: 1 });
  // The background actually behind an element: composite ancestors' fills, bottom up.
  const backdrop = (el) => {
    const chain = [];
    for (let p = el; p; p = p.parentElement) {
      const cs = getComputedStyle(p);
      const img = cs.backgroundImage;
      if (img !== "none") {
        // A gradient counts as the average of its stops; an image or checkerboard can't be judged.
        const stops = /gradient/.test(img) ? (img.match(/rgba?\([^)]*\)/g) || []).map(parse) : [];
        if (!stops.length || /conic|url\(/.test(img)) return null;
        const avg = { r: 0, g: 0, b: 0, a: 1 };
        for (const c of stops) { avg.r += c.r / stops.length; avg.g += c.g / stops.length; avg.b += c.b / stops.length; }
        chain.push(avg); break;
      }
      const c = parse(cs.backgroundColor);
      if (c && c.a > 0) { chain.push(c); if (c.a >= 0.99) break; }
    }
    let bg = { r: 255, g: 255, b: 255, a: 1 };
    for (const c of chain.reverse()) bg = over(c, bg);
    return bg;
  };
  const shown = (el) => {
    const r = el.getBoundingClientRect();
    if (r.width < 2 || r.height < 2 || r.bottom < 0 || r.right < 0 || r.top > innerHeight || r.left > innerWidth) return false;
    for (let p = el; p; p = p.parentElement) {
      const cs = getComputedStyle(p);
      if (cs.display === "none" || cs.visibility === "hidden" || +cs.opacity < 0.2) return false;
    }
    return true;
  };
  const skip = (el) => el.closest("[disabled], .disabled, [aria-disabled=true], .stage, .artboard, .asset-img, .ref-wrap, .layer-thumb, .cp-sv, .cp-hue, .color-chip, .cp-swatch, .avatar");
  const fails = [];
  const check = (el, what) => {
    if (skip(el) || !shown(el)) return;
    const fg = parse(getComputedStyle(el).color), bg = backdrop(el);
    if (!fg || !bg) return;
    const k = ratio(over(fg, bg), bg);
    if (k < %MIN%) {
      const path = []; for (let p = el; p && path.length < 4; p = p.parentElement) path.unshift(p.tagName.toLowerCase() + (p.className && typeof p.className === "string" ? "." + p.className.trim().split(/\s+/).slice(0, 2).join(".") : ""));
      fails.push(`${label}: ${what} "${(el.textContent || "").trim().slice(0, 30)}" ${k.toFixed(2)}:1 at ${path.join(" > ")}`);
    }
  };
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  const seen = new Set();
  for (let n = walker.nextNode(); n; n = walker.nextNode()) {
    const el = n.parentElement;
    if (!el || seen.has(el) || !n.textContent.trim() || ["SCRIPT", "STYLE", "OPTION"].includes(el.tagName)) continue;
    seen.add(el); check(el, "text");
  }
  for (const svg of document.querySelectorAll("button svg, .icon-btn svg, [role=button] svg")) {
    const host = svg.closest("button, [role=button]") || svg.parentElement;
    if (!seen.has(host)) { seen.add(host); check(host, "icon"); }
  }
  for (const inp of document.querySelectorAll("input:not([type=range]):not([type=checkbox]), select")) {
    if (inp.value) check(inp, "field");
  }
  return fails;
}
""".replace("%MIN%", str(MIN_CONTRAST))

OPENERS = {  # panels and tabs an agent visits, per skin, clicked in order
    "figma": ['.fg-nav [data-nav="file"]', '.fg-tabs [data-tab="design"]',
              '.fg-tabs [data-tab="reference"]'],
    "canva": ['.cv-rail-btn[data-panel="design"]', '.cv-rail-btn[data-panel="elements"]', '.cv-rail-btn[data-panel="text"]',
              '.cv-rail-btn[data-panel="uploads"]', '.cv-rail-btn[data-panel="background"]',
              '.cv-context [data-open="position"]',
              '.cv-pos-tabs [data-tab="layers"]', '.cv-pos-tabs [data-tab="arrange"]'],
    "photoshop": ['.ps-tabs [data-tab="color"]', '.ps-tabs [data-tab="swatches"]', '.ps-tabs [data-tab="properties"]',
                  '.ps-tabs [data-tab="reference"]', '.ps-tabs [data-tab="layers"]'],
    "slides": ['.sl-toolbar [data-open="images"]', '.sl-toolbar [data-open="format"]'],
    "powerpoint": ['.pp-tabs [data-tab="insert"]', '.pp-tabs [data-tab="design"]', '.pp-tabs [data-tab="view"]',
                   '.pp-tabs [data-tab="format"]', '.pp-ribbon [data-open="format"]', '.pp-tabs [data-tab="home"]'],
    "inkscape": ['[data-command="dlg_text"]', '[data-command="dlg_transform"]', '[data-command="dlg_document"]',
                 '[data-command="dlg_export"]', '.ik-docktab[data-tab="layers"]', '[data-command="dlg_fill"]'],
}
FONT_BUTTON = {"figma": ".inspector .font-btn", "canva": ".cv-context .cv-font", "photoshop": ".ps-options .ps-font",
               "slides": ".sl-toolbar .sl-font", "powerpoint": ".pp-ribbon .pp-font", "inkscape": ".ik-controls .ik-font"}
MENU_BUTTON = {"figma": ".fg-logo", "canva": ".cv-header .cv-hbtn:not(.cv-brand)", "photoshop": ".ps-menu",
               "slides": ".sl-menu", "powerpoint": '.pp-tabs [data-tab="file"]', "inkscape": ".ik-menu"}


def _audit(page, label):
    return page.evaluate(AUDIT_JS, label)


@pytest.mark.parametrize("skin", SKINS)
@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize("agent", [True, False], ids=["agent", "person"])
def test_everything_on_screen_is_legible(browser, server, skin, theme, agent):
    eid = new_episode(server, index=5, skin=skin)
    info = api(server, f"/api/episodes/{eid}")
    fonts = info.get("task", {}).get("fonts_used") or ["Montserrat"]
    page, errors = open_editor(browser, server, eid, skin, theme=theme, size=(1280, 800))
    if agent:
        page.goto(page.url + "&agent=1")
        page.wait_for_function("window.designEditor && window.designEditor.ready")
    fails = []
    fails += _audit(page, "start")
    # A design with an image and a styled, selected text box (bold on: active toggles show).
    aid = info["assets"][0]["id"]
    page.evaluate("""async ([aid, font]) => { const a = window.__app;
        await a.store.op('add_image', {asset_id: aid, x: 20, y: 20, width: 200});
        const r = await a.store.op('add_text', {text: 'Legible?', x: 60, y: 300, font_family: font, bold: true});
        a.store.select(r.element_id); await a.store.whenIdle(3000); }""", [aid, fonts[0]])
    idle(page)
    for sel in OPENERS[skin]:
        if page.locator(sel).count():
            page.click(sel)
            idle(page)
            fails += _audit(page, f"panel {sel}")
    # The computer's file window, in each OS look, grid and list.
    for os_ in ("mac", "windows", "linux"):
        page.evaluate("(os) => { window.__app.params.os = os; window.__app.openFiles(); }", os_)
        page.wait_for_timeout(150)
        fails += _audit(page, f"file window ({os_})")
        page.locator('.fc [data-view="list"]').first.click()
        fails += _audit(page, f"file window ({os_}, list)")
        page.keyboard.press("Escape")
        assert page.locator(".fc").count() == 0, f"{skin}: Escape left the file window open"
    # Popovers: the font picker (current font is a doc font, so its row is highlighted) and colour.
    if skin == "inkscape":  # text controls belong to the Text tool
        page.click('[data-command="toolText"]')
        idle(page)
    for sel in (FONT_BUTTON[skin],):
        if page.locator(sel).first.is_visible():
            page.locator(sel).first.click()
            page.wait_for_timeout(400)
            fails += _audit(page, "font picker")
            page.keyboard.press("Escape")
            assert page.locator(".pop-font").count() == 0, f"{skin}: Escape left the font picker open"
            page.locator(".stage").click(position={"x": 6, "y": 6})
            page.evaluate("(id) => window.__app.store.select(id)", page.evaluate("window.__app.store.doc.elements.at(-1).id"))
            idle(page)
    chip = page.locator(".inspector .color-chip:visible, .cv-context .cv-textcolor:visible, .ps-options .ps-color .color-chip:visible, "
                        ".sl-toolbar .sl-textcolor:visible, .pp-ribbon .pp-textcolor-drop:visible")
    if chip.count():
        chip.first.click()
        page.wait_for_timeout(300)
        fails += _audit(page, "colour picker")
        page.locator(".stage").click(position={"x": 6, "y": 6})
    page.evaluate("(id) => window.__app.store.select(id)", page.evaluate("window.__app.store.doc.elements.at(-1).id"))
    # Menus: the app menu / menubar, and the right-click menu on the element.
    if page.locator(MENU_BUTTON[skin]).count():
        page.locator(MENU_BUTTON[skin]).first.click()
        page.wait_for_timeout(250)
        fails += _audit(page, "app menu")
        page.keyboard.press("Escape")
    box = page.evaluate("""() => { const s = window.__app.stage, r = s.el.getBoundingClientRect();
        const e = window.__app.store.doc.elements.at(-1), [x, y] = s.toScreen(e.x + e.width / 2, e.y + e.height / 2);
        return [r.left + x, r.top + y]; }""")
    page.mouse.click(*box, button="right")
    page.wait_for_timeout(250)
    fails += _audit(page, "context menu")
    page.keyboard.press("Escape")
    # Dialogs: export, keyboard shortcuts.
    if skin == "inkscape":
        page.click('[data-command="dlg_export"]')
    page.click('[data-command="submit"]')
    page.wait_for_timeout(250)
    fails += _audit(page, "export dialog")
    page.keyboard.press("Escape")
    page.evaluate("window.__app.showShortcuts()")
    page.wait_for_timeout(250)
    fails += _audit(page, "shortcuts dialog")
    page.close()
    assert not errors, errors
    assert not fails, f"{skin}/{theme}/{'agent' if agent else 'person'}: {len(fails)} illegible:\n" + "\n".join(fails[:40])
