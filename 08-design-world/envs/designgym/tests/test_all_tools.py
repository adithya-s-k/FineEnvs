"""Every MCP tool, on every surface, called over a real session and checked for its effect.

Other tests cover the tools an expert or a scripted agent happens to use. This
one walks the whole advertised surface, so a tool that is listed but broken
cannot hide.
"""

from __future__ import annotations

import json
import urllib.request

import pytest
from openenv.core.env_server.mcp_types import CallToolAction
from openenv.core.mcp_client import MCPToolClient

from .conftest import needs_pack

pytestmark = needs_pack


def call(env, name, **args):
    out = env.step(CallToolAction(tool_name=name, arguments=args))
    obs = out.observation
    assert obs.error is None, f"{name}: {obs.error.message}"
    res = obs.result or {}
    assert not res.get("is_error"), f"{name}: {res}"
    return out, res


def text(res):
    return "".join(c.get("text", "") for c in res.get("content", []))


def doc(server, eid):
    with urllib.request.urlopen(f"{server}/api/episodes/{eid}") as r:
        return json.loads(r.read())["doc"]


def test_every_tools_surface_tool(server, pack):
    with MCPToolClient(base_url=server).sync() as env:
        r = env.reset(split="test", index=5, surface="tools")
        eid = r.observation.metadata["episode_id"]
        advertised = {t.name for t in env.list_tools(use_cache=False)}
        seen = set()

        def use(name, **args):
            seen.add(name)
            return call(env, name, **args)

        info = json.loads(text(use("get_task")[1]))
        assert info["task_id"] == "test-00005" and info["assets"]
        fonts = json.loads(text(use("list_fonts", query="mont")[1]))
        assert "Montserrat" in fonts["fonts"]
        for name in ("view_reference", "view_canvas", "view_assets"):
            block = use(name)[1]["content"][0]
            assert block["type"] == "image" and block["data"], name

        aid = info["assets"][0]["id"]
        use("set_background", color="#f4efe6")
        use("add_image", asset_id=aid, x=10, y=20, width=300)
        use("add_text", text="Hello\nWorld", x=50, y=400, font_family="Montserrat", font_size=40,
            color="#112233", align="center", width=500, bold=True, letter_spacing=2, uppercase=True)
        use("add_shape", shape="ellipse", x=600, y=100, width=120, height=80, fill="#e63946",
            stroke="#000000", stroke_width=3)
        d = doc(server, eid)
        assert d["background"] == "#f4efe6"
        assert [e["kind"] for e in d["elements"]] == ["image", "text", "shape"]
        img, txt, shp = d["elements"]
        assert img["height"] == pytest.approx(300 / info["assets"][0]["aspect_ratio"], rel=1e-3)
        assert txt["height"] == pytest.approx(80) and txt["bold"] and txt["uppercase"]
        assert shp["shape"] == "ellipse" and shp["stroke_width"] == 3

        use("update_element", element_id=txt["id"], properties={"x": 70, "angle": 12, "opacity": 0.5})
        use("duplicate_element", element_id=shp["id"], dx=30, dy=40)
        use("reorder_element", element_id=img["id"], to="front")
        use("align_elements", element_ids=[shp["id"]], alignment="right")
        d = doc(server, eid)
        t = next(e for e in d["elements"] if e["id"] == txt["id"])
        assert (t["x"], t["angle"], t["opacity"]) == (70, 12, 0.5)
        assert len(d["elements"]) == 4 and d["elements"][-1]["id"] == img["id"]
        s = next(e for e in d["elements"] if e["id"] == shp["id"])
        assert s["x"] + s["width"] == pytest.approx(d["width"])

        listing = text(use("list_elements")[1])
        assert all(e["id"] in listing for e in d["elements"])
        use("delete_element", element_id=img["id"])
        assert len(doc(server, eid)["elements"]) == 3
        assert text(use("undo")[1]) == "undone"
        assert len(doc(server, eid)["elements"]) == 4
        assert text(use("redo")[1]) == "redone"
        out, res = use("submit")
        assert out.done and 0 < out.reward < 1 and "reward" in text(res)

        assert seen | {"start_task"} == advertised, advertised - seen


def test_every_html_surface_tool(server, pack):
    with MCPToolClient(base_url=server).sync() as env:
        env.reset(split="test", index=5, surface="html")
        advertised = {t.name for t in env.list_tools(use_cache=False)}
        seen = {"start_task"}

        def use(name, **args):
            seen.add(name)
            return call(env, name, **args)

        use("get_task")
        use("list_fonts")
        for name in ("view_reference", "view_canvas", "view_assets"):
            assert use(name)[1]["content"][0]["type"] == "image"
        page = ('<div id="canvas" style="width:1080px;height:1080px;background:#fff">'
                '<div style="left:10px;top:10px;width:100px;height:50px;background:#f00"></div></div>')
        body = json.loads(text(use("write_html", html=page)[1]))
        assert body["elements"] == 1
        assert "background:#ff0000" in text(use("read_html")[1]).replace(" ", "")
        out, _ = use("submit")
        assert out.done
        assert seen == advertised, advertised - seen


def test_every_gui_surface_tool(server, pack):
    """Each computer-use tool, with its effect checked on the agent's own page."""

    def page_js(eid, js):
        req = urllib.request.Request(f"{server}/api/debug/gui/{eid}/eval", data=json.dumps({"js": js}).encode(),
                                     headers={"content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())

    def locate(eid, selector):
        req = urllib.request.Request(f"{server}/api/debug/gui/{eid}/locate",
                                     data=json.dumps({"selector": selector}).encode(),
                                     headers={"content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            hit = json.loads(r.read())[0]
        return [round(hit["x"]), round(hit["y"])]

    with MCPToolClient(base_url=server, message_timeout_s=120).sync() as env:
        r = env.reset(split="test", index=5, surface="gui", skin="figma")
        eid = r.observation.metadata["episode_id"]
        advertised = {t.name for t in env.list_tools(use_cache=False)}
        seen = {"start_task", "view_reference", "get_task"}
        call(env, "get_task")
        call(env, "view_reference")

        def use(name, **args):
            seen.add(name)
            return call(env, name, **args)

        shot = use("screenshot")[1]["content"][0]
        assert shot["type"] == "image" and shot["mimeType"] == "image/png"

        # left_click: Shape tools > Image/video..., open a file, click to place it
        use("left_click", coordinate=locate(eid, '.fg-caret[title="Shape tools"]'))
        use("left_click", coordinate=locate(eid, '.menu [data-command="placeImage"]'))
        use("double_click", coordinate=locate(eid, ".fc .fc-item[data-type=file]"))
        use("left_click", coordinate=locate(eid, ".stage"))
        d = doc(server, eid)
        assert len(d["elements"]) == 1
        el = d["elements"][0]

        # left_click_drag: move it by its centre (Alt off; snapping allowed, so just "moved")
        c = page_js(eid, f"(() => {{ const s = window.__app.stage, r = s.el.getBoundingClientRect(); "
                         f"const p = s.toScreen({el['x'] + el['width'] / 2}, {el['y'] + el['height'] / 2}); "
                         f"return [Math.round(r.left + p[0]), Math.round(r.top + p[1])]; }})()")
        use("left_click_drag", start_coordinate=c, coordinate=[c[0] + 60, c[1] + 40])
        moved = doc(server, eid)["elements"][0]
        assert (moved["x"], moved["y"]) != (el["x"], el["y"])

        # mouse_move: hover shows an outline on the element under the pointer
        use("key", keys="escape")
        use("mouse_move", coordinate=[c[0] + 60, c[1] + 40])
        assert page_js(eid, "document.querySelectorAll('.overlay .hover-box').length") == 1

        # right_click: opens the context menu
        use("right_click", coordinate=[c[0] + 60, c[1] + 40])
        assert page_js(eid, "document.querySelectorAll('.pop-menu .menu.context').length") == 1
        use("key", keys="escape")

        # triple_click + type + key(list): set X exactly through the inspector
        use("left_click", coordinate=[c[0] + 60, c[1] + 40])
        use("triple_click", coordinate=locate(eid, '.inspector [data-field="x"] input'))
        use("type", text="123")
        use("key", keys=["enter"])
        assert doc(server, eid)["elements"][0]["x"] == 123

        # double_click: add text with the text tool, then double-click it to edit
        use("left_click", coordinate=locate(eid, '[data-command="toolText"]'))
        use("left_click", coordinate=locate(eid, ".stage"))
        use("type", text="Hi")
        use("key", keys="escape")
        texts = [e for e in doc(server, eid)["elements"] if e["kind"] == "text"]
        assert [t["text"] for t in texts] == ["Hi"]
        t = texts[0]
        tc = page_js(eid, f"(() => {{ const s = window.__app.stage, r = s.el.getBoundingClientRect(); "
                          f"const p = s.toScreen({t['x'] + t['width'] / 2}, {t['y'] + t['height'] / 2}); "
                          f"return [Math.round(r.left + p[0]), Math.round(r.top + p[1])]; }})()")
        use("double_click", coordinate=tc)
        assert page_js(eid, "!!document.querySelector('.txt[contenteditable]')")
        use("key", keys="ctrl+a")
        use("type", text="Hello")
        use("key", keys="escape")
        assert [e["text"] for e in doc(server, eid)["elements"] if e["kind"] == "text"] == ["Hello"]

        # key chords: undo restores the previous text
        use("key", keys="escape")
        use("key", keys="ctrl+z")
        assert [e["text"] for e in doc(server, eid)["elements"] if e["kind"] == "text"] == ["Hi"]

        # scroll: plain scroll pans, ctrl+scroll zooms
        z0 = page_js(eid, "window.__app.stage.zoom")
        use("scroll", coordinate=locate(eid, ".stage"), scroll_direction="up", scroll_amount=2, text="ctrl")
        assert page_js(eid, "window.__app.stage.zoom") > z0
        p0 = page_js(eid, "window.__app.stage.panY")
        use("scroll", coordinate=locate(eid, ".stage"), scroll_direction="down", scroll_amount=2)
        assert page_js(eid, "window.__app.stage.panY") < p0

        use("wait", duration=0.2)
        out, res = use("submit")
        assert out.done and out.reward is not None
        assert seen == advertised, advertised - seen


def test_gui_normalized_coordinates_hit_the_same_controls(server, pack):
    """coordinates="normalized": a 0-1000 grid click lands on the control a pixel click would."""

    def locate(eid, selector):
        req = urllib.request.Request(f"{server}/api/debug/gui/{eid}/locate",
                                     data=json.dumps({"selector": selector}).encode(),
                                     headers={"content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            hit = json.loads(r.read())[0]
        return hit["x"], hit["y"]

    with MCPToolClient(base_url=server, message_timeout_s=120).sync() as env:
        r = env.reset(split="test", index=5, surface="gui", skin="canva", coordinates="normalized")
        meta = r.observation.metadata
        assert meta["coordinates"] == "normalized"
        assert "0-1000 grid" in meta["instructions"] and "pixels of the" not in meta["instructions"]
        eid = meta["episode_id"]
        grid = lambda x, y: [round(x * 1000 / 1279), round(y * 1000 / 799)]  # noqa: E731  (how a grid model points)
        # Uploads is open on the agent's page: upload a file, then click it in
        call(env, "left_click", coordinate=grid(*locate(eid, ".cv-upload")))
        call(env, "double_click", coordinate=grid(*locate(eid, ".fc .fc-item[data-type=file]")))
        call(env, "left_click", coordinate=grid(*locate(eid, ".asset-tile")))
        assert len(doc(server, eid)["elements"]) == 1
        call(env, "left_click", coordinate=grid(*locate(eid, '.cv-rail-btn[data-panel="text"]')))
        assert locate(eid, '.cv-rail-btn[data-panel="text"].on')  # the Text panel is now the open one
    with MCPToolClient(base_url=server, message_timeout_s=120).sync() as env:
        r = env.reset(split="test", index=5, surface="gui", skin="canva")
        assert r.observation.metadata["coordinates"] == "pixels"
        assert "pixels of the 1280x800 screenshot" in r.observation.metadata["instructions"]
