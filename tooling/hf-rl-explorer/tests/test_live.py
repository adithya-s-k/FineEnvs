"""Live Spaces and the MCP bridge (no network: the Hub and the Spaces are stubbed).

What is checked: the explorer only ever talks to a Space's own hf.space host (never a redirect elsewhere, never a
private Space); waking and restarting do only what they should; playground sessions belong to whoever started them
and are capped; answers stay out of the Task API browser; the bridge speaks MCP (handshake, notifications, errors,
sessions), refuses cross-origin pages, and turns servers without tools into reset/step/state tools.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from app import catalog, main, mcp_bridge, spaces_live as live

client = TestClient(main.app, base_url="https://testserver")


def space_info(stage="RUNNING", sub="org-env", private=False, base_path=None):
    card = SimpleNamespace(to_dict=lambda: ({"base_path": base_path} if base_path else {}))
    return SimpleNamespace(private=private, card_data=card, subdomain=sub, sdk="docker", tags=["openenv"],
                           runtime=SimpleNamespace(stage=stage, hardware="cpu-basic"))


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    catalog._memo.clear()
    live._sessions.clear()
    live._woken.clear()
    mcp_bridge._hits.clear()
    mcp_bridge._clients.clear()
    yield
    for s in list(live._sessions.values()):
        s.ws = None
    live._sessions.clear()


def stub_hub(monkeypatch, **kw):
    monkeypatch.setattr(catalog, "_api", lambda token=None: SimpleNamespace(space_info=lambda spec, expand=None: space_info(**kw)))


# ── the Space's record ───────────────────────────────────────────────────────
def test_only_the_spaces_own_hf_space_host_is_used(monkeypatch):
    stub_hub(monkeypatch, sub="evil.example.com/x")
    assert live.record("org/env")["host"] == ""
    catalog._memo.clear()
    stub_hub(monkeypatch, sub="org-env")
    assert live.record("org/env")["host"] == "https://org-env.hf.space"


def test_private_spaces_are_refused(monkeypatch):
    stub_hub(monkeypatch, private=True)
    with pytest.raises(PermissionError):
        live.record("org/env")
    assert client.get("/api/spaces/org/env/live").status_code == 404


def test_base_path_must_be_a_plain_path(monkeypatch):
    stub_hub(monkeypatch, base_path="//evil.com")
    assert live.record("org/env")["base_path"] is None
    catalog._memo.clear()
    stub_hub(monkeypatch, base_path="/web")
    assert live.record("org/env")["base_path"] == "/web"


def test_bad_space_ids_are_refused():
    for bad in ("../x/y", "a/b/c"):
        assert client.get(f"/api/spaces/{bad}/live").status_code in (400, 404)


def test_redirects_off_the_space_are_never_followed(monkeypatch):
    stub_hub(monkeypatch)
    rec = live.record("org/env")
    seen = []

    def fake_get(url, **kw):
        seen.append(url)
        if url.endswith("/web"):
            return httpx.Response(302, headers={"location": "http://169.254.169.254/latest/meta-data"}, request=httpx.Request("GET", url))
        if url.endswith("/start-page"):
            return httpx.Response(302, headers={"location": "/web/"}, request=httpx.Request("GET", url))
        return httpx.Response(200, headers={"content-type": "text/html"}, text="<html>", request=httpx.Request("GET", url))

    monkeypatch.setattr(live._client, "get", fake_get)
    assert live._get_here(rec, "/web") is None                       # off-host: dropped, not fetched
    assert not any("169.254" in u for u in seen)
    r = live._get_here(rec, "/start-page")                            # on-host: followed
    assert r is not None and r.url.path == "/web/"


def test_hub_dataset_names_map_back_safely():
    assert live.hub_dataset("/data/FineEnvs__MiMo-V2.6-RL-harbor-terminal") == "FineEnvs/MiMo-V2.6-RL-harbor-terminal"
    assert live.hub_dataset("/app/prepared/datasets/train") is None
    assert live.hub_dataset("/data/a__b__c") is None
    assert live.hub_dataset("/data/__x") is None


# ── wake and restart ─────────────────────────────────────────────────────────
def test_wake_only_sleeping_spaces_and_at_most_once_a_minute(monkeypatch):
    calls = []
    monkeypatch.setattr(live.httpx, "post", lambda url, **kw: calls.append(url) or httpx.Response(302))
    stub_hub(monkeypatch, stage="SLEEPING")
    assert live.wake("org/env")["woken"] is True
    assert live.wake("org/env")["woken"] is True        # again within a minute: not sent again
    assert calls == ["https://huggingface.co/spaces/org/env/start"]
    catalog._memo.clear()
    stub_hub(monkeypatch, stage="RUNTIME_ERROR")
    with pytest.raises(live.SpaceError):
        live.wake("org/env")
    catalog._memo.clear()
    stub_hub(monkeypatch, stage="RUNNING")
    assert live.wake("org/env") == {"stage": "RUNNING", "woken": False}
    assert len(calls) == 1


def test_restart_needs_a_signed_in_visitor():
    assert client.post("/api/spaces/org/env/restart", json={}).status_code == 401


def test_restart_without_write_access_says_so(monkeypatch):
    from huggingface_hub.errors import HfHubHTTPError

    def refuse(_spec):
        raise HfHubHTTPError("no", response=httpx.Response(403, request=httpx.Request("POST", "https://huggingface.co")))

    space = space_info(stage="RUNTIME_ERROR")
    monkeypatch.setattr(catalog, "_api", lambda token=None: SimpleNamespace(space_info=lambda spec, expand=None: space, restart_space=refuse))
    with pytest.raises(live.SpaceError) as e:
        live.restart("org/env", "hf_x")
    assert e.value.status == 403


def test_wake_and_play_refuse_cross_site_posts():
    for path, body in (("/api/spaces/org/env/wake", {}), ("/api/spaces/org/env/play", {"op": "start"})):
        r = client.post(path, json=body, headers={"Origin": "https://evil.example"})
        assert r.status_code == 403


# ── the Task API browser ─────────────────────────────────────────────────────
def test_task_api_leaves_answers_out(monkeypatch):
    stub_hub(monkeypatch)
    monkeypatch.setattr(live, "probe", lambda spec, fresh=False: {"running": True, "task_api": {"env": "e", "splits": []}})

    def fake_post(url, json=None, **kw):
        task = {"task_id": "t1", "prompt": "find x", "ground_truth": "42", "meta": {"expected_output": "x", "level": 2},
                "target_latex": "a^2", "target_language": "fr", "options": [{"answer": "b", "text": "B"}]}
        body = {"tasks": [task]} if url.endswith("/task_range") else {"task": task}
        return httpx.Response(200, json=body, request=httpx.Request("POST", url))

    monkeypatch.setattr(live._client, "post", fake_post)
    for out in (live.tasks("org/env", "train", 0, 999)["tasks"][0], live.task("org/env", "train", 0)["task"]):
        text = json.dumps(out)
        assert "42" not in text and "expected_output" not in text and "a^2" not in text and '"answer"' not in text
        assert out["prompt"] == "find x" and out["target_language"] == "fr"
    assert live.tasks("org/env", "train", 0, 999)["stop"] == 50                      # a page is at most 50 tasks


# ── playground sessions ──────────────────────────────────────────────────────
def running(monkeypatch):
    stub_hub(monkeypatch)
    monkeypatch.setattr(live, "probe", lambda spec, fresh=False: {"running": True, "openenv": True, "step_api": True, "mcp": None})


def test_sessions_belong_to_who_started_them(monkeypatch):
    running(monkeypatch)
    sid = live.start("org/env", "web:1.1.1.1")["session"]
    with pytest.raises(live.SpaceError) as e:
        live.act(sid, "web:2.2.2.2", "state")
    assert e.value.status == 410


def test_sessions_are_capped_per_owner_and_in_total(monkeypatch):
    running(monkeypatch)
    ids = [live.start("org/env", "web:a")["session"] for _ in range(live.PER_OWNER + 2)]
    mine = [s for s in live._sessions.values() if s.owner == "web:a"]
    assert len(mine) == live.PER_OWNER and ids[-1] in live._sessions and ids[0] not in live._sessions
    monkeypatch.setattr(live, "TOTAL", len(live._sessions))
    with pytest.raises(live.SpaceError) as e:
        live.start("org/env", "web:b")
    assert e.value.status == 429


def test_idle_sessions_end(monkeypatch):
    running(monkeypatch)
    sid = live.start("org/env", "web:a")["session"]
    live._sessions[sid].used = time.time() - live.IDLE - 1
    with pytest.raises(live.SpaceError):
        live.act(sid, "web:a", "state")
    assert sid not in live._sessions


def test_play_rejects_bad_operations_and_tool_names(monkeypatch):
    running(monkeypatch)
    sid = live.start("org/env", "web:a")["session"]
    with pytest.raises(live.SpaceError):
        live.act(sid, "web:a", "rm -rf")
    with pytest.raises(live.SpaceError):
        live.act(sid, "web:a", "call", {"name": "../../x y"})
    r = client.post("/api/spaces/org/env/play", json={"op": "nope"})
    assert r.status_code == 422


def test_play_sessions_do_not_cross_spaces(monkeypatch):
    running(monkeypatch)
    r = client.post("/api/spaces/org/env/play", json={"op": "start"})
    sid = r.json()["session"]
    r = client.post("/api/spaces/org/other/play", json={"op": "state", "session": sid})
    assert r.status_code == 400


def test_websocket_replies_become_results_and_errors(monkeypatch):
    running(monkeypatch)
    sent = []

    class FakeWS:
        def __init__(self, replies):
            self.replies = list(replies)

        def send(self, m):
            sent.append(json.loads(m))

        def recv(self, timeout=None):
            return json.dumps(self.replies.pop(0))

        def close(self):
            pass

    sid = live.start("org/env", "web:a")["session"]
    live._sessions[sid].ws = FakeWS([{"type": "observation", "data": {"observation": {"x": 1}, "reward": 0.5, "done": True}},
                                     {"type": "error", "data": {"message": "bad action"}},
                                     {"type": "mcp", "data": {"jsonrpc": "2.0", "id": 1, "result": {"content": [{"type": "text", "text": "ok"}]}}}])
    out = live.act(sid, "web:a", "reset", {"seed": 1})
    assert out["result"]["reward"] == 0.5 and sent[0] == {"type": "reset", "data": {"seed": 1}}
    with pytest.raises(live.SpaceError) as e:
        live.act(sid, "web:a", "step", {"a": 1})
    assert "bad action" in str(e.value)
    out = live.act(sid, "web:a", "call", {"name": "look", "arguments": {"h": 1}})
    assert out["result"]["content"][0]["text"] == "ok" and sent[-1]["type"] == "mcp" and sent[-1]["data"]["method"] == "tools/call"


def test_images_in_observations_are_found():
    import base64

    png = base64.b64encode(b"\x89PNG" + b"0" * 400).decode()
    parts = live.image_parts({"observation": {"image_base64": png, "image_format": "png", "text": "hi"}})
    assert parts and parts[0][0] == "image/png"
    assert live.image_parts({"observation": {"text": "x" * 500}}) == []


# ── the MCP bridge ───────────────────────────────────────────────────────────
INFO = {"running": True, "stage": "RUNNING", "openenv": True, "step_api": True, "mcp": None,
        "schema": {"action": {"type": "object", "properties": {"latex": {"type": "string"}, "metadata": {"type": "object"}}}},
        "task_api": {"env": "e", "splits": [{"name": "train", "num_tasks": 3}]}}


def rpc(method, params=None, rid=1, sid=None, **headers):
    h = {"Content-Type": "application/json", **({"Mcp-Session-Id": sid} if sid else {}), **headers}
    body = {"jsonrpc": "2.0", "method": method, **({"id": rid} if rid is not None else {}), **({"params": params} if params is not None else {})}
    return client.post("/mcp/org/env", content=json.dumps(body), headers=h)


@pytest.fixture
def bridge(monkeypatch):
    stub_hub(monkeypatch)
    monkeypatch.setattr(live, "probe", lambda spec, fresh=False: INFO)
    calls = []
    monkeypatch.setattr(live, "act", lambda sid, owner, op, data=None: calls.append((op, data)) or {"result": {"observation": {"ok": 1}, "reward": 1.0}})
    return calls


def test_bridge_handshake_and_session(bridge):
    r = rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}})
    assert r.status_code == 200 and r.headers["mcp-session-id"]
    res = r.json()["result"]
    assert res["protocolVersion"] == "2025-06-18" and res["capabilities"]["tools"] == {"listChanged": False}
    assert rpc("initialize", {"protocolVersion": "1999-01-01"}).json()["result"]["protocolVersion"] == mcp_bridge.PROTOCOLS[0]
    assert rpc("notifications/initialized", rid=None).status_code == 202
    assert rpc("ping").json()["result"] == {}


def test_bridge_offers_episode_and_task_tools_for_servers_without_tools(bridge):
    tools = {t["name"]: t for t in rpc("tools/list").json()["result"]["tools"]}
    assert set(tools) == {"reset", "step", "state", "list_splits", "list_tasks", "get_task"}
    assert set(tools["step"]["inputSchema"]["properties"]) == {"latex"}           # the action schema, metadata dropped
    r = rpc("tools/call", {"name": "step", "arguments": {"latex": "x^2"}}).json()["result"]
    assert bridge[-1] == ("step", {"latex": "x^2"}) and "content" in r and not r.get("isError")


def test_bridge_tool_errors_are_results_the_agent_can_read(bridge, monkeypatch):
    def boom(*a, **k):
        raise live.SpaceError("the Space didn't answer within 180 s", 504)

    monkeypatch.setattr(live, "act", boom)
    r = rpc("tools/call", {"name": "step", "arguments": {}}).json()
    assert r["result"]["isError"] is True and "180 s" in r["result"]["content"][0]["text"]
    assert rpc("tools/call", {"name": "nope", "arguments": {}}).json()["result"]["isError"] is True


def test_bridge_rejects_what_it_should(bridge):
    assert rpc("resources/read").json()["error"]["code"] == -32601
    assert client.post("/mcp/org/env", content="{nope", headers={"Content-Type": "application/json"}).status_code == 400
    assert client.post("/mcp/org/env", content=json.dumps([{"jsonrpc": "2.0", "id": 1, "method": "ping"}])).status_code == 400
    assert client.post("/mcp/org/env", content=json.dumps({"id": 1, "method": "ping"})).status_code == 400
    assert rpc("ping", **{"Origin": "https://evil.example"}).status_code == 403
    assert client.post("/mcp/org/env", content="x" * (mcp_bridge.MAX_BODY + 1)).status_code == 413
    assert client.get("/mcp/org/env").status_code == 405
    assert client.delete("/mcp/org/env").status_code == 204


def test_bridge_is_rate_limited(bridge, monkeypatch):
    monkeypatch.setattr(mcp_bridge, "RATE", (5, 60))
    codes = [rpc("ping").status_code for _ in range(8)]
    assert codes[:5] == [200] * 5 and 429 in codes[5:]


def test_bridge_wakes_a_sleeping_space_instead_of_failing(monkeypatch):
    stub_hub(monkeypatch, stage="SLEEPING")
    monkeypatch.setattr(live, "probe", lambda spec, fresh=False: {"running": False, "stage": "SLEEPING"})
    woke = []
    monkeypatch.setattr(live, "wake", lambda spec: woke.append(spec) or {"stage": "APP_STARTING"})
    r = rpc("tools/list").json()
    assert r["error"]["code"] == -32002 and woke == ["org/env"]


def test_bridge_passes_through_mcp_content_and_drops_output_schemas(monkeypatch):
    stub_hub(monkeypatch)
    info = {**INFO, "mcp": [{"name": "look", "inputSchema": {"type": "object"}, "outputSchema": {"type": "object"}}]}
    monkeypatch.setattr(live, "probe", lambda spec, fresh=False: info)
    monkeypatch.setattr(live, "act", lambda *a, **k: {"result": {"content": [{"type": "text", "text": "hi", "annotations": None, "meta": None}],
                                                                  "structured_content": {"result": "hi"}, "is_error": False}})
    tools = rpc("tools/list").json()["result"]["tools"]
    assert all("outputSchema" not in t for t in tools)
    assert {t["name"] for t in tools} >= {"look", "reset", "state"} and "step" not in {t["name"] for t in tools}
    r = rpc("tools/call", {"name": "look", "arguments": {}}).json()["result"]
    assert r["content"] == [{"type": "text", "text": "hi"}] and r["structuredContent"] == {"result": "hi"} and "isError" not in r


# ── a server's own routes (NeMo Gym, ORS, custom) ────────────────────────────
OPENAPI = {"paths": {
    "/guess": {"post": {"summary": "Guess", "requestBody": {"content": {"application/json": {"schema": {"$ref": "#/components/schemas/G"}}}}}},
    "/tasks/{task_id}": {"get": {"summary": "A task"}},
    "//evil.example/x": {"get": {}}, "/a/../b": {"get": {}}, "/q?x=1": {"get": {}}, "/web/x": {"get": {}}, "/reset": {"post": {}},
    "/self": {"post": {"requestBody": {"content": {"application/json": {"schema": {"$ref": "#/components/schemas/S"}}}}}},
}, "components": {"schemas": {"G": {"type": "object", "properties": {"word": {"type": "string", "title": "Word"}}, "required": ["word"]},
                              "S": {"type": "object", "properties": {"next": {"$ref": "#/components/schemas/S"}}}}}}


def test_routes_are_plain_paths_on_non_openenv_servers_only():
    routes = live._api_routes(OPENAPI, openenv=False)
    assert {(r["method"], r["path"]) for r in routes} == {("POST", "/guess"), ("GET", "/tasks/{task_id}"), ("POST", "/self"), ("POST", "/reset")}
    guess = next(r for r in routes if r["path"] == "/guess")
    assert guess["schema"]["properties"]["word"] == {"type": "string", "title": "Word"} and guess["schema"]["required"] == ["word"]
    assert next(r for r in routes if r["path"] == "/tasks/{task_id}")["params"] == ["task_id"]
    assert live._api_routes(OPENAPI, openenv=True) == []          # an OpenEnv server is used through reset/step and tools
    json.dumps(routes)                                             # a self-referencing schema is cut, not looped


def test_route_parameters_cannot_leave_the_route():
    info = {"api": live._api_routes(OPENAPI, openenv=False)}
    assert live.route_url(info, "GET", "/tasks/{task_id}", {"task_id": "t-1.2"}) == "/tasks/t-1.2"
    for bad in ("../admin", "a/b", "x y", "", "%2e%2e"):
        with pytest.raises(live.SpaceError):
            live.route_url(info, "GET", "/tasks/{task_id}", {"task_id": bad})
    with pytest.raises(live.SpaceError):
        live.route_url(info, "POST", "/not/published", {})
    with pytest.raises(live.SpaceError):
        live.route_url(info, "DELETE", "/guess", {})


def test_route_calls_keep_their_cookies_and_withhold_answers_from_gets(monkeypatch):
    stub_hub(monkeypatch)
    monkeypatch.setattr(live, "probe", lambda spec, fresh=False: {"running": True, "openenv": False, "api": live._api_routes(OPENAPI, openenv=False)})
    seen = []

    def handler(request: httpx.Request):
        seen.append((request.method, request.url.path, request.headers.get("cookie")))
        if request.url.path == "/guess":
            return httpx.Response(200, json={"output": "ok"}, headers={"set-cookie": "session=abc; path=/"})
        return httpx.Response(200, json={"task_id": "t1", "prompt": "p", "answer": "SECRET", "meta": {"ground_truth": "TRUTHVAL"}})

    real = live.SpaceClient
    monkeypatch.setattr(live, "SpaceClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    sid = live.start("org/env", "web:a")["session"]
    out = live.act(sid, "web:a", "http", {"method": "POST", "path": "/guess", "body": {"word": "crane"}})["result"]
    assert out["status"] == 200 and out["json"] == {"output": "ok"}
    out = live.act(sid, "web:a", "http", {"method": "GET", "path": "/tasks/{task_id}", "params": {"task_id": "t1"}})["result"]
    assert "SECRET" not in json.dumps(out) and "TRUTHVAL" not in json.dumps(out) and set(out["withheld"]) == {"answer", "ground_truth"}
    assert seen[0][0] == "POST" and seen[1][2] == "session=abc"   # the session's own cookie, kept server-side
    with pytest.raises(live.SpaceError):
        live.act(sid, "web:a", "http", {"method": "PUT", "path": "/guess"})


def test_a_reset_step_server_still_lists_its_tools(monkeypatch):
    """No MCP tools of its own (OpenEnv answers "Environment does not support MCP"): the page gets exactly what the MCP
    bridge offers an agent, step taking the server's own action fields. Tool servers use the same descriptor."""
    info = {"running": True, "stage": "RUNNING", "step_api": True, "mcp": None, "task_api": None,
            "schema": {"action": {"type": "object", "properties": {"latex": {"type": "string", "description": "Predicted LaTeX"}}, "required": ["latex"]}}}
    monkeypatch.setattr(live, "probe", lambda spec, fresh=False: info)
    tools = {t["name"]: t for t in client.get("/api/spaces/o/latex/live").json()["bridge_tools"]}
    assert set(tools) == {"reset", "step", "state"}
    assert "latex" in json.dumps(tools["step"]["inputSchema"])
    assert tools == {t["name"]: t for t in mcp_bridge._tools(info)}   # the page and the bridge never disagree
    monkeypatch.setattr(live, "probe", lambda spec, fresh=False: {**info, "mcp": [{"name": "look", "inputSchema": {"type": "object"}}]})
    assert {t["name"] for t in client.get("/api/spaces/o/geo/live").json()["bridge_tools"]} == {"look", "reset", "state"}
