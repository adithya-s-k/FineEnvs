"""The OpenEnv contract, over a real server: Task API, MCP tools, rewards, editor API."""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request

import pytest
from openenv.core.env_server.mcp_types import CallToolAction
from openenv.core.mcp_client import MCPToolClient

from designgym.core import expert

from .conftest import needs_pack

pytestmark = needs_pack

SURFACE_TOOLS = {
    "tools": {"add_image", "add_text", "add_shape", "update_element", "delete_element",
              "duplicate_element", "reorder_element", "align_elements", "set_background", "undo",
              "redo", "list_elements", "view_canvas", "view_assets", "list_fonts"},
    "html": {"write_html", "read_html", "view_canvas", "view_assets", "list_fonts"},
    "gui": {"screenshot", "left_click", "double_click", "triple_click", "right_click",
            "mouse_move", "left_click_drag", "scroll", "type", "key", "wait"},
}
COMMON = {"start_task", "get_task", "view_reference", "submit"}


def get(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return r.status, r.read()


def post(url, body, ctype="application/json"):
    data = body if isinstance(body, bytes) else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"content-type": ctype}, method="POST")
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.status, json.loads(r.read())


def call(env, name, **args):
    out = env.step(CallToolAction(tool_name=name, arguments=args))
    return out


def text_of(out):
    res = out.observation.result
    if out.observation.error is not None:
        return "ERROR " + out.observation.error.message
    return "".join(c.get("text", "") for c in (res or {}).get("content", []))


def test_health_and_metadata(server):
    assert get(f"{server}/health")[0] == 200
    assert json.loads(get(f"{server}/healthz")[1])["tasks"]["test"] > 0
    assert json.loads(get(f"{server}/list_environments")[1]) == ["design_canvas"]
    schema = json.loads(get(f"{server}/schema")[1])
    assert "action" in schema and "observation" in schema


def test_task_api(server, pack):
    splits = json.loads(get(f"{server}/design_canvas/splits")[1])
    assert {s["name"] for s in splits} == set(pack.splits())
    n = post(f"{server}/design_canvas/num_tasks", {"split": "test"})[1]["num_tasks"]
    assert n == pack.count("test")
    task = post(f"{server}/design_canvas/task", {"split": "test", "index": 3})[1]["task"]
    assert task["task_id"] == "test-00003" and "doc" not in task and "description" not in task
    rng = post(f"{server}/design_canvas/task_range", {"split": "test", "start": 2, "stop": 5})[1]
    assert [t["index"] for t in rng["tasks"]] == [2, 3, 4]
    req = urllib.request.Request(f"{server}/design_canvas/task", data=json.dumps(
        {"split": "test", "index": 10**6}).encode(), headers={"content-type": "application/json"})
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req)
    assert exc.value.code == 400


@pytest.mark.parametrize("surface", ["tools", "html"])
def test_tool_lists_follow_surface(server, surface):
    with MCPToolClient(base_url=server).sync() as env:
        env.reset(split="validation", index=0, surface=surface)
        names = {t.name for t in env.list_tools(use_cache=False)}
        assert names == COMMON | SURFACE_TOOLS[surface]


@pytest.mark.parametrize("mode", ["reference", "description"])
def test_expert_over_mcp_scores_one(server, pack, mode):
    task = pack.at("test", 7)
    with MCPToolClient(base_url=server).sync() as env:
        r = env.reset(split="test", index=7, surface="tools", mode=mode)
        meta = r.observation.metadata
        assert meta["task_id"] == "test-00007"
        assert ("reference_jpeg_b64" in meta) == (mode == "reference")
        assert ("description" in meta) == (mode == "description")
        rewards = []
        for c in expert.tool_calls(pack.gt_doc(task)):
            out = call(env, c["name"], **c["arguments"])
            assert out.observation.error is None, text_of(out)
            rewards.append(out.reward)
        assert out.done and out.reward == pytest.approx(1.0)
        assert all(r == 0.0 for r in rewards[:-1])
        # MCPToolClient keeps only episode_id/step_count from state; the grade
        # rides on the final observation.
        assert out.observation.metadata["grade"]["total"] == pytest.approx(1.0)
        assert env.state().step_count == len(rewards)


def test_html_surface_over_mcp(server, pack):
    task = pack.at("test", 4)
    with MCPToolClient(base_url=server).sync() as env:
        env.reset(split="test", index=4, surface="html")
        out = call(env, "write_html", html=expert.html_page(pack.gt_doc(task)))
        body = json.loads(text_of(out))
        assert body["elements"] == len(pack.gt_doc(task).elements)
        page = text_of(call(env, "read_html"))
        assert page.startswith('<div id="canvas"')
        bad = call(env, "write_html", html="<p>nothing</p>")
        assert "HTML rejected" in text_of(bad) or bad.observation.result.get("is_error")
        out = call(env, "submit")
        assert out.done and out.reward == pytest.approx(1.0)


def test_errors_are_reported_not_fatal(server):
    with MCPToolClient(base_url=server).sync() as env:
        env.reset(split="validation", index=1)
        for name, args in [("add_image", {"asset_id": "nope"}),
                           ("add_text", {"text": "x", "font_family": "Comic Sans MS"}),
                           ("update_element", {"element_id": "e404", "properties": {"x": 1}}),
                           ("reorder_element", {"element_id": "e1", "to": "sideways"})]:
            out = call(env, name, **args)
            assert "ERROR" in text_of(out) or out.observation.result.get("is_error"), name
        ok = call(env, "add_shape", x=1, y=1, width=10, height=10)
        assert ok.observation.error is None
        assert not ok.done


def test_images_are_content_blocks(server):
    with MCPToolClient(base_url=server).sync() as env:
        env.reset(split="validation", index=2)
        for name in ("view_reference", "view_canvas", "view_assets"):
            res = call(env, name).observation.result
            block = res["content"][0]
            assert block["type"] == "image" and len(block["data"]) > 1000, name


def test_step_limit_submits(server):
    with MCPToolClient(base_url=server).sync() as env:
        env.reset(split="validation", index=3, max_steps=3)
        call(env, "add_shape")
        call(env, "add_shape")
        out = call(env, "add_shape")
        assert out.done and out.reward is not None and 0 <= out.reward < 1
        after = call(env, "add_shape")
        assert "finished" in text_of(after)


def test_looking_does_not_spend_the_action_budget(server):
    """Screenshots, renders and listings are free; only actions count toward max_steps."""
    with MCPToolClient(base_url=server).sync() as env:
        env.reset(split="validation", index=3, max_steps=2)
        for name in ("view_canvas", "view_reference", "list_elements", "view_canvas", "list_fonts"):
            assert not call(env, name).done
        assert not call(env, "add_shape").done
        assert call(env, "add_shape").done  # the second action is the last
    with MCPToolClient(base_url=server).sync() as env:
        env.reset(split="validation", index=3, max_steps=2)
        outs = [call(env, "view_canvas") for _ in range(6)]  # 3x the budget in looking ends it
        assert outs[-1].done and not outs[-2].done


def test_dense_reward_sums_to_final(server, pack):
    task = pack.at("validation", 5)
    with MCPToolClient(base_url=server).sync() as env:
        env.reset(split="validation", index=5, reward_mode="dense")
        total = 0.0
        for c in expert.tool_calls(pack.gt_doc(task)):
            out = call(env, c["name"], **c["arguments"])
            total += out.reward or 0.0
        assert out.done
        assert total == pytest.approx(1.0, abs=1e-6)


def test_undo_redo(server):
    with MCPToolClient(base_url=server).sync() as env:
        env.reset(split="validation", index=0)
        call(env, "add_shape", x=0, y=0, width=10, height=10)
        assert "e1" in text_of(call(env, "list_elements"))
        assert text_of(call(env, "undo")) == "undone"
        assert "no elements" in text_of(call(env, "list_elements"))
        assert text_of(call(env, "redo")) == "redone"
        assert "e1" in text_of(call(env, "list_elements"))


def test_start_task_for_mcp_clients(server):
    with MCPToolClient(base_url=server).sync() as env:
        out = call(env, "start_task", split="test", index=1, mode="description")
        body = json.loads(text_of(out))
        assert body["task_id"] == "test-00001" and "BRIEF" in body["instructions"]
        assert "write_html" not in body["tools"]


def test_editor_api_and_live_sync(server, pack):
    import websockets

    status, ep = post(f"{server}/api/episodes", {"split": "test", "index": 0, "skin": "canva"})
    eid = ep["episode_id"]
    st = json.loads(get(f"{server}/api/episodes/{eid}")[1])
    aid = st["assets"][0]["id"]
    assert get(f"{server}{st['assets'][0]['url']}")[0] == 200
    with pytest.raises(urllib.error.HTTPError) as err:  # the target is served only after export
        get(f"{server}/api/episodes/{eid}/reference.jpg")
    assert err.value.code == 403
    # No design browser: other tasks' previews (their answers) are not served.
    for url in (f"{server}/api/tasks/{st['task']['task_id']}/thumb.jpg", f"{server}/api/tasks"):
        with pytest.raises(urllib.error.HTTPError) as err:
            get(url)
        assert err.value.code in (404, 405)  # nothing served there
    css = get(f"{server}/fonts.css?families=Montserrat")[1].decode()
    assert "@font-face" in css and "/fonts/montserrat/" in css
    font_url = css.split("url('")[1].split("'")[0]
    assert get(f"{server}{font_url}")[0] == 200
    assert get(f"{server}/editor/")[0] == 200
    assert get(f"{server}/web/")[0] == 200

    async def session():
        ws_url = server.replace("http", "ws") + f"/api/episodes/{eid}/ws"
        async with websockets.connect(ws_url) as a, websockets.connect(ws_url) as b:
            first_a = json.loads(await a.recv())
            json.loads(await b.recv())
            assert first_a["type"] == "state" and first_a["doc"]["elements"] == []
            await a.send(json.dumps({"type": "op", "op": "add_image", "args": {"asset_id": aid}, "id": 1}))
            msgs = []
            while not any(m.get("type") == "ack" for m in msgs):
                msgs.append(json.loads(await a.recv()))
            ack = next(m for m in msgs if m["type"] == "ack")
            assert ack["ok"] and ack["result"]["element_id"] == "e1"
            pushed = json.loads(await asyncio.wait_for(b.recv(), 5))
            assert pushed["type"] == "state" and len(pushed["doc"]["elements"]) == 1
            await a.send(json.dumps({"type": "op", "op": "update_element",
                                     "args": {"element_id": "e1", "font_size": 3}, "id": 2}))
            while True:
                m = json.loads(await a.recv())
                if m.get("type") == "ack":
                    assert not m["ok"] and "no font_size" in m["error"]
                    break

    asyncio.run(session())
    log = json.loads(get(f"{server}/api/episodes/{eid}/log")[1])["log"]
    assert [e["op"] for e in log] == ["add_image", "update_element"]
    assert log[0]["source"] == "editor" and log[1]["ok"] is False
    status, grade = post(f"{server}/api/episodes/{eid}/submit", b"", "text/plain")
    assert 0 <= grade["total"] < 1


def test_html_import_endpoint(server, pack):
    _, ep = post(f"{server}/api/episodes", {"split": "test", "index": 2})
    eid = ep["episode_id"]
    page = expert.html_page(pack.gt_doc(pack.at("test", 2)))
    status, body = post(f"{server}/api/episodes/{eid}/html", page.encode(), "text/plain")
    assert body["elements"] == len(pack.gt_doc(pack.at("test", 2)).elements)
    _, grade = post(f"{server}/api/episodes/{eid}/submit", b"", "text/plain")
    assert grade["total"] == pytest.approx(1.0)
