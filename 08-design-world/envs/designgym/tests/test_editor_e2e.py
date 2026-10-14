"""The browser editor, end to end, in real Chromium.

Two kinds of check:

- **Layout**, for every skin x theme x viewport: the page loads without console
  errors, nothing overflows the window, every control an agent needs is
  visible, unobstructed and on-screen, and the artboard fits the stage.
- **Function**, for every skin: a full editing session done only with the mouse
  and keyboard (add an asset, drag it, resize it, type exact numbers, add and
  edit text, change font and colour, undo, redo, reorder, delete, edit as HTML,
  submit), asserting after each gesture that the *server's* document changed
  exactly as intended. The editor is only correct if what the server scores is
  what the gesture meant.

Needs Playwright's Chromium (`playwright install chromium`). Set E2E_BROWSER=firefox or
webkit (after `playwright install firefox webkit`) to run the same checks in those engines.
"""

from __future__ import annotations

import json
import urllib.request

import pytest

from .conftest import needs_pack

pytestmark = needs_pack
playwright = pytest.importorskip("playwright.sync_api")

from designgym.server.episodes import SKINS as _SKINS  # noqa: E402

SKINS = list(_SKINS)
VIEWPORTS = [(1920, 1080), (1440, 900), (1280, 800), (1024, 768), (900, 700), (800, 600)]

# How to reach each panel: a selector, or a list of clicks (PowerPoint: Insert tab, then Pictures).
# Images come from the computer's file window, opened the way each real app does it.
PLACE = '.menu [data-command="placeImage"]'
FILES_OPENER = {
    "figma": ['.fg-caret[title="Shape tools"]', PLACE],
    "canva": ['.cv-rail-btn[data-panel="uploads"]', ".cv-upload"],
    "photoshop": ['.ps-menu:text-is("File")', PLACE],
    "slides": ['.sl-toolbar [data-open="images"]', PLACE],
    "powerpoint": ['.pp-tabs [data-tab="insert"]', '.pp-ribbon [data-open="pictures"]', PLACE],
    "inkscape": ['.ik-commands [data-open="files"]'],
}
INSPECTOR_OPENER = {
    "figma": '.fg-tabs [data-tab="design"]',
    "canva": '.cv-context [data-open="position"]',
    "photoshop": '.ps-tabs [data-tab="properties"]',
    "slides": '.sl-side[data-panel="format"], .sl-toolbar [data-open="format"]',
    "powerpoint": ['.pp-tabs [data-tab="format"]', '.pp-ribbon [data-open="format"]'],  # Shape Format / Picture > Format Pane
    "inkscape": '[data-command="toolMove"]',  # the Selector's controls hold X / Y / W / H
}
LAYERS_OPENER = {  # Slides has no layers panel; order lives in Arrange
    "figma": '.fg-nav [data-nav="file"]',
    "canva": ['.cv-context [data-open="position"]', '.cv-pos-tabs [data-tab="layers"]'],
    "photoshop": '.ps-tabs [data-tab="layers"]',
    "slides": None,
    "powerpoint": ['.pp-tabs [data-tab="home"]', ".pp-ribbon .pp-select", '.menu-item:has-text("Selection Pane")'],
    "inkscape": '.ik-docktab[data-tab="layers"]',
}
FONT_BUTTON = {"figma": ".inspector .font-btn", "canva": ".cv-context .font-btn", "photoshop": ".ps-options .font-btn",
               "slides": ".sl-toolbar .font-btn", "powerpoint": ".pp-ribbon .font-btn", "inkscape": ".ik-controls .font-btn"}  # (Inkscape: with the Text tool active)
# Each layout's own text controls (the recorder uses them): on screen whenever text is selected.
TEXT_REQUIRED = {
    "figma": [".inspector .font-btn"],
    "canva": [".cv-context .cv-font", ".cv-context .cv-size input", ".cv-context .cv-textcolor"],
    "photoshop": [".ps-options .ps-font", '.ps-options input[aria-label="Font size"]', ".ps-options .ps-color .color-chip"],
    "slides": [".sl-toolbar .sl-font", ".sl-toolbar .sl-size input", ".sl-toolbar .sl-textcolor",
               ".sl-toolbar .sl-align", ".sl-toolbar .sl-spacing"],
    "powerpoint": [".pp-ribbon .pp-font", ".pp-ribbon .pp-size input", ".pp-ribbon .pp-textcolor-drop",
                   ".pp-ribbon .pp-align", ".pp-ribbon .pp-linespacing"],
    "inkscape": [".ik-controls .ik-font", ".ik-controls .ik-size input", ".ik-controls .ik-align", ".ik-controls .ik-style"],
}


def _first(opener):
    return opener[0] if isinstance(opener, list) else opener


def open_panel(page, opener):
    for sel in (opener if isinstance(opener, list) else [opener]):
        page.locator(sel).first.click()
        page.wait_for_timeout(60)
# Controls that must be reachable in each skin without scrolling or hunting.
REQUIRED = {
    "figma": ['[data-command="submit"]', '[data-command="toolText"]', '[data-command="toolRect"]',
              _first(FILES_OPENER["figma"]), LAYERS_OPENER["figma"], '.fg-logo', '.zoom-label'],
    "canva": ['[data-command="submit"]', '[data-command="undo"]', _first(FILES_OPENER["canva"]),
              '.cv-rail-btn[data-panel="text"]', '.cv-rail-btn[data-panel="uploads"]', '.cv-zoom-range'],
    "photoshop": ['[data-command="submit"]', '.ps-menu', '[data-command="toolText"]',
                  '[data-command="toolMove"]', LAYERS_OPENER["photoshop"]],
    "slides": ['[data-command="submit"]', '.sl-menu', '[data-command="toolText"]', '[data-command="toolMove"]',
               _first(FILES_OPENER["slides"]), '.sl-toolbar [data-open="format"]', '.sl-toolbar .sl-zoom',
               ".sl-menusearchbox"],
    "powerpoint": ['[data-command="submit"]', '.pp-tabs [data-tab="home"]', '.pp-tabs [data-tab="insert"]',
                   '.pp-tabs [data-tab="design"]', ".pp-ribbon .pp-select", ".pp-searchbox"],
    "inkscape": ['[data-command="dlg_export"]', '.ik-menu', '[data-command="toolText"]', '[data-command="toolMove"]',
                 _first(FILES_OPENER["inkscape"]), INSPECTOR_OPENER["inkscape"], LAYERS_OPENER["inkscape"]],
}


def api(server, path, body=None):
    if body is None:
        with urllib.request.urlopen(server + path, timeout=10) as r:
            return json.loads(r.read())
    data = json.dumps(body).encode()
    req = urllib.request.Request(server + path, data=data, headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def new_episode(server, index=0, skin="figma", split="test"):
    return api(server, "/api/episodes", {"split": split, "index": index, "skin": skin})["episode_id"]


def doc(server, eid):
    return api(server, f"/api/episodes/{eid}")["doc"]


def open_editor(browser, server, eid, skin, theme=None, size=(1440, 900)):
    page = browser.new_page(viewport={"width": size[0], "height": size[1]})
    errors = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: errors.append(str(e)))
    url = f"{server}/editor/?episode={eid}&skin={skin}" + (f"&theme={theme}" if theme else "")
    page.goto(url)
    page.wait_for_function("window.designEditor && window.designEditor.ready === true", timeout=15000)
    idle(page)
    return page, errors


def idle(page):
    page.evaluate("window.designEditor.whenIdle(3000)")
    page.wait_for_timeout(60)


def stage_point(page, x, y):
    """Canvas pixel -> page pixel, through the stage's current zoom and pan."""
    return page.evaluate("""([x, y]) => {
        const s = window.__app.stage, r = s.el.getBoundingClientRect();
        const [sx, sy] = s.toScreen(x, y);
        return [r.left + sx, r.top + sy];
    }""", [x, y])


def center_of(page, el):
    return stage_point(page, el["x"] + el["width"] / 2, el["y"] + el["height"] / 2)


LONG_LABEL = {"W": "Width", "H": "Height", "Rotation": "Angle", "X": "Horizontal position", "Y": "Vertical position"}  # Canva spells the labels out; Slides says Angle


def field(page, label):
    """The visible inspector number field with this label (X, Y, W, H) or title."""
    loc = page.locator(".inspector .num-field:visible").filter(
        has=page.locator(f'.num-label:text-is("{label}")'))
    if loc.count() == 0 and label in LONG_LABEL:
        loc = page.locator(".inspector .num-field:visible").filter(
            has=page.locator(f'.num-label:text-is("{LONG_LABEL[label]}")'))
    if loc.count() == 0:
        loc = page.locator(f'.inspector .num-field:visible:has(.num-label[title="{label}"])')
    return loc.first.locator("input")


def type_into(page, input_loc, value):
    input_loc.click()
    page.keyboard.press("ControlOrMeta+A")
    page.keyboard.type(str(value))
    page.keyboard.press("Enter")
    idle(page)


# ------------------------------------------------------------------ layout


def _layout_problems(page, skin, required=None):
    problems = page.evaluate("""(required) => {
        const out = [];
        const se = document.scrollingElement;
        if (se.scrollWidth > innerWidth + 1) out.push(`page overflows horizontally (${se.scrollWidth} > ${innerWidth})`);
        if (se.scrollHeight > innerHeight + 1) out.push(`page overflows vertically (${se.scrollHeight} > ${innerHeight})`);
        for (const sel of required) {
            const el = [...document.querySelectorAll(sel)].find((e) => e.offsetParent !== null);
            if (!el) { out.push(`missing: ${sel}`); continue; }
            const r = el.getBoundingClientRect();
            if (r.width < 8 || r.height < 8) { out.push(`too small: ${sel}`); continue; }
            if (r.left < 0 || r.top < 0 || r.right > innerWidth + 1 || r.bottom > innerHeight + 1) {
                out.push(`off-screen: ${sel} ${JSON.stringify(r)}`); continue;
            }
            const hit = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
            if (!hit || !(el === hit || el.contains(hit) || hit.contains(el))) out.push(`obstructed: ${sel} by ${hit && hit.className}`);
        }
        const stage = document.querySelector(".stage").getBoundingClientRect();
        if (stage.width < 360 || stage.height < 260) out.push(`stage too small ${stage.width}x${stage.height}`);
        const art = document.querySelector(".artboard").getBoundingClientRect();
        if (art.left < stage.left - 1 || art.top < stage.top - 1 || art.right > stage.right + 1 || art.bottom > stage.bottom + 1)
            out.push("artboard not fitted inside the stage");
        if (art.width < 100 && art.height < 100) out.push("artboard is tiny");
        return out;
    }""", required if required is not None else REQUIRED[skin])
    return problems


@pytest.mark.parametrize("size", VIEWPORTS, ids=[f"{w}x{h}" for w, h in VIEWPORTS])
@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize("skin", SKINS)
def test_layout(browser, server, skin, theme, size):
    eid = new_episode(server, index=1, skin=skin)
    page, errors = open_editor(browser, server, eid, skin, theme, size)
    try:
        assert not errors, errors
        assert page.evaluate("document.documentElement.dataset.theme") == theme
        problems = _layout_problems(page, skin)
        # Selecting something swaps in the element controls: they must fit too.
        page.evaluate("window.__app.commands.addRect.run()")
        idle(page)
        problems += _layout_problems(page, skin)
        page.evaluate("""async () => { const a = window.__app; const r = await a.store.op('add_text', {text: 'Hi', x: 20, y: 20});
            a.store.select(r.element_id); }""")
        idle(page)
        if skin == "figma":
            page.click(INSPECTOR_OPENER["figma"])
            idle(page)
        if skin == "inkscape":  # the Text tool's controls style the selected text
            page.click('[data-command="toolText"]')
            idle(page)
        problems += _layout_problems(page, skin, TEXT_REQUIRED[skin])
        if skin == "canva":
            problems += page.evaluate("""() => {
                const bar = document.querySelector('.cv-context');
                return bar.scrollWidth > bar.clientWidth + 1 ? [`context bar clipped (${bar.scrollWidth} > ${bar.clientWidth})`] : [];
            }""")
        assert not problems, problems
    finally:
        page.close()


# The panel that should move to the other side under layout=swap, per skin.
SIDE_PANEL = {"figma": ".fg-left", "canva": ".cv-rail", "photoshop": ".ps-dock", "slides": ".sl-filmstrip",
              "powerpoint": ".pp-thumbs", "inkscape": ".ik-dock"}


@pytest.mark.parametrize("skin", SKINS)
def test_layout_swap_mirrors_panels(browser, server, skin):
    """Mirrored panels move to the other side, and the page stays whole: same controls, on screen."""
    eid = new_episode(server, index=1, skin=skin)
    boxes = {}
    for layout in ("default", "swap"):
        page, errors = open_editor(browser, server, eid, skin, size=(1280, 800))
        if layout == "swap":
            page.goto(page.url + "&layout=swap")
            page.wait_for_function("window.designEditor && window.designEditor.ready")
            idle(page)
        stage = page.locator(".stage").bounding_box()
        assert stage["x"] >= 0 and stage["y"] >= 0 and stage["x"] + stage["width"] <= 1280 + 1
        assert stage["width"] > 400 and stage["height"] > 300, f"{skin}/{layout}: stage squashed {stage}"
        for sel in REQUIRED[skin]:
            box = page.locator(sel).first.bounding_box()
            assert box and box["x"] >= 0 and box["x"] + box["width"] <= 1281 and box["y"] >= 0, (skin, layout, sel)
        boxes[layout] = page.locator(SIDE_PANEL[skin]).bounding_box()
        assert not errors, errors
        page.close()
    centre = lambda b: b["x"] + b["width"] / 2  # noqa: E731
    assert (centre(boxes["default"]) < 640) != (centre(boxes["swap"]) < 640), f"{skin}: swap did not mirror"


@pytest.mark.parametrize("skin", SKINS)
def test_editor_never_shows_the_target(browser, server, skin):
    """Neither agents nor people see the target in the editor (no reference panel, no overlay):
    it comes with the task, and the playground shows it beside the editor. It is served only
    after export, for the result screen."""
    eid = new_episode(server, index=3, skin=skin)
    for agent in (True, False):
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        page.goto(f"{server}/editor/?episode={eid}&skin={skin}" + ("&agent=1" if agent else ""))
        page.wait_for_function("window.designEditor && window.designEditor.ready")
        idle(page)
        assert page.locator('[data-tab="reference"], [data-open="reference"], [data-panel="design"], .ref-img, .ref-overlay').count() == 0, (skin, agent)
        assert page.evaluate("window.__app.store.task.reference_url") is None
        page.keyboard.press("Shift+R")  # the old overlay shortcut does nothing
        idle(page)
        assert page.locator(".ref-overlay, .ref-img").count() == 0
        page.close()


PROBE_JS = """async () => {
  // Click the next visible, enabled control not tried yet, and report whether anything reacted.
  const skip = '[data-command="submit"], [data-command="delete"], [data-command="cut"], .stage *, [data-probed]';
  const b = [...document.querySelectorAll('#app button')].find((x) => !x.disabled && x.offsetParent !== null
    && !x.matches(skip) && x.getBoundingClientRect().width > 0);
  if (!b) return null;
  b.dataset.probed = "1";
  // already the active tool or tab: clicking it again rightly does nothing
  if (b.matches('.on, [aria-pressed="true"], [aria-selected="true"]')) return { label: "", changed: true };
  const a = window.__app;
  const before = JSON.stringify([a.store.version, a.stage.zoom, a.store.tool]);
  let n = 0;
  const mo = new MutationObserver((m) => { n += m.filter((r) => r.attributeName !== "data-probed").length; });
  mo.observe(document.body, { subtree: true, childList: true, attributes: true, characterData: true });
  b.click();
  await new Promise((r) => setTimeout(r, 120));
  mo.disconnect();
  const label = b.getAttribute('aria-label') || b.title || b.textContent.trim().slice(0, 30) || b.className;
  return { label, changed: n > 0 || before !== JSON.stringify([a.store.version, a.stage.zoom, a.store.tool]) };
}"""


@pytest.mark.parametrize("skin", SKINS)
def test_every_control_on_the_agent_page_does_something(browser, server, skin):
    """A button that does nothing only teaches an agent to waste clicks: every one must react."""
    eid = new_episode(server, index=3, skin=skin)
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    page.goto(f"{server}/editor/?episode={eid}&skin={skin}&agent=1")
    page.wait_for_function("window.designEditor && window.designEditor.ready")
    probe_id = page.evaluate("""async () => { const a = window.__app;
        const r = await a.store.op('add_text', {text: 'Probe', x: 40, y: 40, font_size: 40});
        a.store.select(r.element_id); await a.store.whenIdle(3000); return r.element_id; }""")
    idle(page)
    dead, tried = [], 0
    revisit = page.locator("#app :is([data-tab], [data-open], [data-panel], [data-nav])[data-probed]")
    reopened = 0
    for _ in range(400):
        res = page.evaluate(PROBE_JS)
        if res is None:  # out of controls: reopen tabs and panels already tried, to reach what they hold
            while reopened < revisit.count() and not revisit.nth(reopened).is_visible():
                reopened += 1
            if reopened >= revisit.count():
                break
            revisit.nth(reopened).click()
            reopened += 1
            idle(page)
            continue
        tried += 1
        if not res["changed"]:
            dead.append(res["label"])
        page.keyboard.press("Escape")
        page.keyboard.press("Escape")
        page.evaluate("""(id) => { document.querySelectorAll('.modal-back, .modal, .popover').forEach((m) => m.remove());
            const a = window.__app; if (a.store.tool !== "move") a.commands.toolMove.run();
            if (a.store.doc.elements.some((e) => e.id === id)) a.store.select(id); }""", probe_id)
    page.close()
    assert tried >= 20, f"{skin}: only {tried} controls probed"
    assert not dead, f"{skin}: controls that do nothing: {dead}"


def _contrast(fg, bg):
    def lum(rgb):
        c = [v / 255 for v in rgb]
        c = [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
        return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    a, b = sorted((lum(fg), lum(bg)), reverse=True)
    return (a + 0.05) / (b + 0.05)


@pytest.mark.parametrize("skin", SKINS)
@pytest.mark.parametrize("theme", ["light", "dark"])
def test_primary_buttons_are_legible(browser, server, skin, theme):
    """Export and the dialog's confirm button must be readable in every look (text vs background)."""
    eid = new_episode(server, index=4, skin=skin)
    page, _ = open_editor(browser, server, eid, skin, theme=theme, size=(1280, 800))
    open_export(page, skin)
    page.click('[data-command="submit"]')
    page.wait_for_selector(".modal-foot .btn-primary")
    for sel in ('[data-command="submit"]', ".modal-foot .btn-primary"):
        fg, bg = page.evaluate("""(sel) => {
            const cs = getComputedStyle(document.querySelector(sel));
            const rgb = (v) => v.match(/\\d+(\\.\\d+)?/g).slice(0, 3).map(Number);
            return [rgb(cs.color), rgb(cs.backgroundColor)]; }""", sel)
        # 2.5:1 catches invisible buttons (the Pixel light bug was ~1:1) while allowing
        # the white-on-bright-blue that real editors use (Studio's #0d99ff is 2.99:1).
        assert _contrast(fg, bg) >= 2.5, f"{skin}/{theme} {sel}: text {fg} on {bg}"
    page.close()


MENU_OPENER = {"figma": ".fg-logo", "canva": ".cv-header .cv-hbtn:not(.cv-brand)", "photoshop": '.ps-menu:text-is("File")',
               "slides": '.sl-menu:text-is("File")', "powerpoint": '.pp-tabs [data-tab="file"]', "inkscape": '.ik-menu:text-is("File")'}


@pytest.mark.parametrize("skin", SKINS)
def test_agent_page_has_no_code_editor(browser, server, skin):
    """On the GUI surface the design must be built with the mouse and keyboard, not typed in as HTML."""
    eid = new_episode(server, index=3, skin=skin)
    for agent in (True, False):
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        page.goto(f"{server}/editor/?episode={eid}&skin={skin}" + ("&agent=1" if agent else ""))
        page.wait_for_function("window.designEditor && window.designEditor.ready")
        idle(page)
        buttons = page.locator('[data-command="html"]:visible').count()
        page.locator(MENU_OPENER[skin]).first.click()
        page.wait_for_timeout(200)
        in_menu = page.locator('.pop-menu :text("Edit as HTML")').count()
        page.keyboard.press("Escape")
        page.keyboard.press("Control+Alt+KeyH")
        page.keyboard.press("Meta+Alt+KeyH")
        page.wait_for_timeout(300)
        opened = page.locator(".modal textarea, .modal .html-editor, .html-modal").count()
        if agent:
            assert (buttons, in_menu, opened) == (0, 0, 0), (skin, buttons, in_menu, opened)
        else:
            assert buttons + in_menu > 0, f"{skin}: people should still have the code editor"
        page.close()


# ------------------------------------------------------------------ function


def open_files(page, skin):
    """Open the file window the app's way (Template's Uploads may already be open)."""
    steps = FILES_OPENER[skin]
    if skin == "canva" and page.locator(".cv-upload").first.is_visible():
        steps = steps[1:]
    open_panel(page, steps)
    page.locator(".fc").wait_for()


def open_export(page, skin):
    """Inkscape exports from its Export dialog (File > Export); elsewhere Export is on screen."""
    if skin == "inkscape" and not page.locator('.ik-exportdlg [data-command="submit"]').count():
        page.click('[data-command="dlg_export"]')
        page.wait_for_selector('.ik-exportdlg [data-command="submit"]')


def add_from_files(page, skin, aid):
    """Open the file window the app's way, open the file, and finish placing it."""
    open_files(page, skin)
    page.locator(f'.fc .fc-item[data-file="{aid}"]').dblclick()
    page.wait_for_timeout(100)
    assert page.locator(".fc").count() == 0
    if skin == "figma":  # the image follows the pointer until a click places it
        r = page.locator(".stage").bounding_box()
        page.mouse.click(r["x"] + r["width"] / 2, r["y"] + r["height"] / 2)
    elif skin == "inkscape":  # the bitmap import question
        page.click(".ik-import .btn-primary")
    elif skin == "canva":  # uploaded: now it is in Uploads
        page.click(f'.asset-tile[data-asset="{aid}"]')
    idle(page)


def run_editing_session(browser, server, skin, size=(1440, 900)):
    eid = new_episode(server, index=2, skin=skin)
    page, errors = open_editor(browser, server, eid, skin, size=size)
    try:
        assets = api(server, f"/api/episodes/{eid}")["assets"]
        aid = assets[0]["id"]

        # 1. add an image from the file window
        add_from_files(page, skin, aid)
        d = doc(server, eid)
        assert [e["asset_id"] for e in d["elements"]] == [aid]
        el = d["elements"][0]
        assert page.evaluate("window.__app.store.selection") == [el["id"]]

        # 2. drag it (Alt disables snapping so the delta is exact)
        cx, cy = center_of(page, el)
        zoom = page.evaluate("window.__app.stage.zoom")
        page.mouse.move(cx, cy)
        page.keyboard.down("Alt")
        page.mouse.down()
        page.mouse.move(cx + 40, cy + 25, steps=8)
        page.mouse.up()
        page.keyboard.up("Alt")
        idle(page)
        moved = doc(server, eid)["elements"][0]
        assert moved["x"] == pytest.approx(el["x"] + 40 / zoom, abs=1.0)
        assert moved["y"] == pytest.approx(el["y"] + 25 / zoom, abs=1.0)

        # 3. resize from the bottom-right handle: an image keeps its aspect ratio
        if skin == "photoshop":  # handles come with Free Transform, as in Photoshop
            assert page.locator(".sel-box .h-se").count() == 0
            page.keyboard.press("Control+t")
            idle(page)
        handle = page.locator(".sel-box .h-se").bounding_box()
        hx, hy = handle["x"] + handle["width"] / 2, handle["y"] + handle["height"] / 2
        page.mouse.move(hx, hy)
        page.keyboard.down("Alt")
        page.mouse.down()
        page.mouse.move(hx + 60, hy + 60, steps=8)
        page.mouse.up()
        page.keyboard.up("Alt")
        idle(page)
        resized = doc(server, eid)["elements"][0]
        assert resized["width"] > moved["width"]
        assert resized["width"] / resized["height"] == pytest.approx(moved["width"] / moved["height"], rel=0.01)
        assert resized["x"] == pytest.approx(moved["x"], abs=1.0)  # top-left anchored
        if skin == "photoshop":
            page.keyboard.press("Enter")  # commit the transform
            idle(page)
            assert page.locator(".sel-box .h-se").count() == 0

        # 4. exact numbers through the inspector
        already = {"slides": '.sl-side[data-panel="format"]', "powerpoint": '.pp-pane[data-panel="format"]'}.get(skin)
        if not (already and page.locator(already).count()):
            open_panel(page, INSPECTOR_OPENER[skin] if skin != "slides" else '.sl-toolbar [data-open="format"]')
        type_into(page, field(page, "X"), 123)
        type_into(page, field(page, "Y"), 45)
        type_into(page, field(page, "W"), 300)
        if skin == "inkscape":  # Object > Transform, Rotate, clockwise, Apply
            page.click('[data-command="dlg_transform"]')
            page.click('.ik-tr .ik-subtab[data-tab="rotate"]')
            page.click('.ik-tr .ik-tr-dir button[data-value="cw"]')
            type_into(page, page.locator('.ik-tr [data-field="rotate_angle"] input'), 15)
            page.click(".ik-tr .ik-apply")
            idle(page)
        else:
            type_into(page, field(page, "Rotation"), 15)
        e = doc(server, eid)["elements"][0]
        assert (e["x"], e["y"], e["width"], e["angle"]) == (123, 45, 300, 15)

        # 5. add text with the text tool, type, commit
        page.keyboard.press("Escape")
        if skin == "canva":  # Canva drops a text box on the page, already editing
            page.click('.cv-rail-btn[data-panel="text"]')
            page.click(".text-presets .btn-primary")
        else:
            if skin == "powerpoint":  # Text Box is on the Insert tab
                page.click('.pp-tabs [data-tab="insert"]')
            page.click('[data-command="toolText"]')
            canvas_w = d["width"]
            tx, ty = stage_point(page, canvas_w * 0.2, d["height"] * 0.8)
            page.mouse.click(tx, ty)
        page.wait_for_selector(".txt[contenteditable]", timeout=5000)
        page.keyboard.type("Summer")
        page.keyboard.press("Enter")
        page.keyboard.type("Sale")
        page.keyboard.press("Escape")
        idle(page)
        texts = [x for x in doc(server, eid)["elements"] if x["kind"] == "text"]
        assert len(texts) == 1 and texts[0]["text"] == "Summer\nSale", texts
        t = texts[0]
        assert t["height"] == pytest.approx(2 * t["font_size"] * t["line_height"], abs=0.5)

        # 6. double-click to edit existing text
        tcx, tcy = center_of(page, t)
        page.mouse.dblclick(tcx, tcy)
        page.wait_for_selector(".txt[contenteditable]", timeout=5000)
        page.keyboard.press("ControlOrMeta+A")
        page.keyboard.type("Big Sale")
        page.keyboard.press("Escape")
        idle(page)
        t = next(x for x in doc(server, eid)["elements"] if x["kind"] == "text")
        assert t["text"] == "Big Sale"

        # 7. font, size and colour from the text controls
        page.evaluate(f"window.__app.store.select('{t['id']}')")
        idle(page)
        if skin == "powerpoint":
            page.click('.pp-tabs [data-tab="home"]')
        if skin == "inkscape":
            page.click('[data-command="toolText"]')
        page.locator(FONT_BUTTON[skin]).first.click()
        if skin == "slides":  # the font menu, then More fonts: search, tick, Done
            page.locator('.sl-fontmenu .menu-item:has-text("More fonts")').click()
            page.fill(".sl-fonts .font-search", "Raleway")
            page.locator('.sl-fonts .font-row[data-family="Raleway"]').click()
            page.click(".sl-fonts .modal-foot .btn-primary")
        else:
            page.fill(".font-search", "Raleway")
            page.keyboard.press("Enter")
        idle(page)
        t = next(x for x in doc(server, eid)["elements"] if x["kind"] == "text")
        assert t["font_family"] == "Raleway"
        if skin != "canva":
            size_field = page.locator('.num-field:visible:has(.num-label[title="Font size"]) input').first
        else:
            size_field = page.locator(".cv-size input")
        type_into(page, size_field, 64)
        t = next(x for x in doc(server, eid)["elements"] if x["kind"] == "text")
        assert t["font_size"] == 64
        if skin == "canva":
            page.click(".cv-textcolor")  # Text color opens in the side panel; its search takes a hex
            page.fill(".cv-colorpanel .cv-colsearch", "#ff0000")
            page.locator(".cv-colorpanel .cv-colsearch").press("Enter")
        elif skin == "powerpoint":  # Font Color arrow > More Colors..., the Hex field (Enter is OK)
            page.click(".pp-ribbon .pp-textcolor-drop")
            page.click(".pp-pop .pp-morecolors")
            page.keyboard.type("FF0000")
            page.keyboard.press("Enter")
        elif skin == "slides":  # the text colour menu: CUSTOM +, then the Hex field (Enter is OK)
            page.click(".sl-toolbar .sl-textcolor")
            page.click(".sl-pop .slc-add")
            page.keyboard.type("#ff0000")
            page.keyboard.press("Enter")
        elif skin == "inkscape":  # Fill and Stroke > Fill > Flat color, the RGBA entry
            page.click('[data-command="dlg_fill"]')
            page.fill(".ik-fs .ik-rgba", "ff0000ff")
            page.locator(".ik-fs .ik-rgba").press("Enter")
        else:
            hexin = page.locator(".inspector .color-field:visible .hex-input").first
            hexin.click()
            page.keyboard.press("ControlOrMeta+A")
            page.keyboard.type("FF0000")
            page.keyboard.press("Enter")
        idle(page)
        page.keyboard.press("Escape")
        t = next(x for x in doc(server, eid)["elements"] if x["kind"] == "text")
        assert t["color"] == "#ff0000"

        # 8. undo / redo from the keyboard
        if skin == "inkscape":  # back to the Selector: a click with the Text tool would start new text
            page.click('[data-command="toolMove"]')
        page.click(".stage", position={"x": 5, "y": 5})
        page.keyboard.press("Control+Z")
        idle(page)
        assert next(x for x in doc(server, eid)["elements"] if x["kind"] == "text")["color"] != "#ff0000"
        page.keyboard.press("Control+Shift+Z")
        idle(page)
        assert next(x for x in doc(server, eid)["elements"] if x["kind"] == "text")["color"] == "#ff0000"

        # 9. stacking order: send the text to the back with the shortcut
        page.evaluate(f"window.__app.store.select('{t['id']}')")
        page.keyboard.press({"figma": "BracketLeft", "inkscape": "End"}.get(skin, "Control+["))  # each app's Send to back
        idle(page)
        assert doc(server, eid)["elements"][0]["id"] == t["id"]

        # 10. the layers panel reflects the document, front first
        if LAYERS_OPENER[skin]:
            open_panel(page, LAYERS_OPENER[skin])
            idle(page)
            rows = page.locator(".layer-row:not(.background-row):not(.fg-frame-row):visible").evaluate_all("rs => rs.map(r => r.dataset.id)")
            assert rows == [x["id"] for x in reversed(doc(server, eid)["elements"])]

        # 11. delete with the keyboard
        img_id = next(x["id"] for x in doc(server, eid)["elements"] if x["kind"] == "image")
        page.evaluate(f"window.__app.store.select('{img_id}')")
        page.keyboard.press("Delete")
        idle(page)
        assert [x["kind"] for x in doc(server, eid)["elements"]] == ["text"]

        # 12. edit as HTML and apply
        page.evaluate("window.__app.commands.html.run()")
        page.wait_for_selector(".code-area")
        page.wait_for_function("document.querySelector('.code-area').value.includes('Big Sale')")
        src = page.eval_on_selector(".code-area", "a => a.value")
        page.fill(".code-area", src.replace("Big Sale", "Mega Sale"))
        page.click(".modal-foot .btn-primary")
        page.wait_for_selector(".code-area", state="detached")
        idle(page)
        assert next(x for x in doc(server, eid)["elements"] if x["kind"] == "text")["text"] == "Mega Sale"

        # 13. what the editor draws is where the server thinks things are
        mismatch = page.evaluate("""() => {
            const s = window.__app.stage, out = [];
            for (const el of window.__app.store.doc.elements) {
                if (el.angle) continue;
                const n = s.nodes.get(el.id).getBoundingClientRect();
                const a = s.artboard.getBoundingClientRect();
                const x = (n.left - a.left) / s.zoom, y = (n.top - a.top) / s.zoom;
                const w = n.width / s.zoom, h = n.height / s.zoom;
                if (Math.abs(x - el.x) > 0.6 || Math.abs(y - el.y) > 0.6 || Math.abs(w - el.width) > 0.6 || Math.abs(h - el.height) > 0.6)
                    out.push({id: el.id, dom: [x, y, w, h], doc: [el.x, el.y, el.width, el.height]});
            }
            return out;
        }""")
        assert not mismatch, mismatch

        # 14. submit through the UI
        open_export(page, skin)
        page.click('[data-command="submit"]')
        page.click(".modal-foot .btn-primary")
        page.wait_for_selector(".modal-result", timeout=10000)
        state = api(server, f"/api/episodes/{eid}")
        assert state["done"] and state["result"] is not None
        assert not errors, errors
    finally:
        page.close()


@pytest.mark.parametrize("skin", SKINS)
def test_editing_session(browser, server, skin):
    run_editing_session(browser, server, skin)


@pytest.mark.parametrize("skin", SKINS)
def test_editing_session_small_screen(browser, server, skin):
    run_editing_session(browser, server, skin, size=(1024, 768))


def test_agent_edits_appear_live(browser, server):
    """An MCP/API edit shows up in an open editor without a reload."""
    import asyncio

    import websockets

    eid = new_episode(server, index=3)
    page, errors = open_editor(browser, server, eid, "figma")
    try:
        async def edit():
            async with websockets.connect(server.replace("http", "ws") + f"/api/episodes/{eid}/ws?source=agent") as ws:
                await ws.recv()
                await ws.send(json.dumps({"type": "op", "op": "add_shape",
                                          "args": {"x": 10, "y": 10, "width": 50, "height": 50}, "id": 1}))
                while json.loads(await ws.recv()).get("type") != "ack":
                    pass

        # Playwright's sync API owns this thread's event loop.
        import threading
        t = threading.Thread(target=lambda: asyncio.run(edit()))
        t.start()
        t.join(10)
        page.wait_for_function("document.querySelectorAll('.artboard .el').length === 1", timeout=3000)
        assert page.locator(".presence").is_visible()
        assert not errors
    finally:
        page.close()


def test_no_design_browser_anywhere(browser, server):
    """No way to browse other designs from the editor: their previews are their answers.

    /editor/ without an episode says where episodes come from; no skin has an Open design
    command, and every menu, for people and agents, is free of it."""
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    page.goto(f"{server}/editor/")
    page.wait_for_selector(".fatal")
    assert page.locator(".task-card, .task-picker").count() == 0
    page.close()
    eid = new_episode(server, index=1)
    for skin in SKINS:
        for agent in (False, True):
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            page.goto(f"{server}/editor/?episode={eid}&skin={skin}" + ("&agent=1" if agent else ""))
            page.wait_for_function("window.designEditor && window.designEditor.ready")
            assert page.evaluate("!('openTask' in window.__app.commands) && !window.__app.openTaskPicker"), (skin, agent)
            page.keyboard.press("ControlOrMeta+O")
            page.wait_for_timeout(150)
            assert page.locator(".task-card, .task-picker").count() == 0, (skin, agent)
            page.close()


def test_gradio_playground_flow(browser, server):
    """The /web playground: load a task, replay the expert, submit, see 1.000."""
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(f"{server}/web/")
    page.get_by_role("button", name="Load index").click()
    frame = page.locator('iframe[title="DesignGym editor"]')
    frame.wait_for(timeout=20000)
    box = frame.bounding_box()
    assert box["width"] > 0.85 * 1440 and box["height"] >= 640, box
    page.get_by_role("tab", name="Expert").click()
    page.get_by_role("button", name="Replay expert trajectory").click()
    page.wait_for_function(
        "() => [...document.querySelectorAll('textarea')].some(t => t.value.includes('add_'))",
        timeout=30000)
    page.get_by_role("button", name="✅ Submit & score").click()
    page.get_by_text("Reward 1.000").wait_for(timeout=20000)
    # The embedded editor shows the same finished document.
    page.get_by_role("tab", name="Editor").click()
    editor = page.frame_locator('iframe[title="DesignGym editor"]')
    editor.locator(".artboard .el").first.wait_for(timeout=10000)
    assert editor.locator(".artboard .el").count() > 1
    assert not errors, errors
    page.close()
