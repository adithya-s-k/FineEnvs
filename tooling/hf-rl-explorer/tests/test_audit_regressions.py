"""Security and lifecycle regressions found during PR #33's explorer audit."""
import gzip

import httpx
import pytest
from fastapi.testclient import TestClient

from app import config, main, runner, spaces_live as live


def test_forwarded_addresses_require_a_trusted_proxy(monkeypatch):
    from starlette.requests import Request
    from app.http import client_ip
    request = Request({"type": "http", "headers": [(b"x-forwarded-for", b"forged, proxy-appended")], "client": ("peer", 123)})
    monkeypatch.setattr(config, "TRUST_PROXY", False)
    assert client_ip(request) == "peer"
    monkeypatch.setattr(config, "TRUST_PROXY", True)
    assert client_ip(request) == "proxy-appended"


def test_streamed_gzip_has_a_decompression_output_cap(monkeypatch):
    monkeypatch.setattr(live, "MAX_REPLY", 1024)
    class Stream(httpx.SyncByteStream):
        def __iter__(self):
            yield gzip.compress(b"x" * 10_000_000)
    with live.SpaceClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, headers={"content-encoding": "gzip"}, stream=Stream()))) as c:
        with pytest.raises(live.SpaceError, match="too large"):
            c.get("https://fixture.hf.space/")


@pytest.mark.parametrize("headers", [
    {"Origin": "https://evil.example", "X-Forwarded-Host": "evil.example"},
    {"Origin": "http://testserver"},
    {"Origin": "null"},
    {"Origin": "https://testserver/path"},
    {"Origin": "https://testserver@evil.example"},
])
def test_origin_must_match_scheme_and_authority(monkeypatch, headers):
    monkeypatch.setattr(config, "PUBLIC_URL", "")
    with TestClient(main.app, base_url="https://testserver") as c:
        assert c.post("/api/logout", headers=headers).status_code == 403


def test_valid_origin_and_explicit_public_origin(monkeypatch):
    with TestClient(main.app, base_url="https://testserver") as c:
        monkeypatch.setattr(config, "PUBLIC_URL", "")
        assert c.post("/api/logout", headers={"Origin": "https://testserver:443"}).status_code == 200
        monkeypatch.setattr(config, "PUBLIC_URL", "https://public.hf.space")
        assert c.post("/api/logout", headers={"Origin": "https://public.hf.space"}).status_code == 200


def test_personal_api_responses_are_not_cached():
    with TestClient(main.app, base_url="https://testserver") as c:
        assert c.get("/api/runs").headers["cache-control"] == "private, no-store"
        assert c.get("/api/me").headers["cache-control"] == "private, no-store"


@pytest.mark.parametrize("bad", [".", "..", "x\n", "x/y", "%2e%2e", "//elsewhere"])
def test_path_parameters_cannot_change_the_published_route(bad):
    info = {"api": [{"method": "GET", "path": "/tasks/{id}", "params": ["id"]}]}
    with pytest.raises(live.SpaceError):
        live.route_url(info, "GET", "/tasks/{id}", {"id": bad})


def test_upstream_body_is_stopped_before_it_is_fully_buffered(monkeypatch):
    monkeypatch.setattr(live, "MAX_REPLY", 64 * 1024)
    reads = []

    class Stream(httpx.SyncByteStream):
        def __iter__(self):
            for i in range(100):
                reads.append(i)
                yield b"x" * 64 * 1024

    with live.SpaceClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, stream=Stream()))) as c:
        with pytest.raises(live.SpaceError, match="too large"):
            c.get("https://fixture.hf.space/")
    assert len(reads) <= 3


def test_gzip_reply_is_decoded_once_and_limited(monkeypatch):
    monkeypatch.setattr(live, "MAX_REPLY", 1024)
    def response(r):
        return httpx.Response(200, content=gzip.compress(b'{"ok":true}'), headers={"Content-Encoding": "gzip"})
    with live.SpaceClient(transport=httpx.MockTransport(response)) as c:
        assert c.get("https://fixture.hf.space/").json() == {"ok": True}
    with live.SpaceClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, content=gzip.compress(b"x" * 2048), headers={"Content-Encoding": "gzip"}))) as c:
        with pytest.raises(live.SpaceError, match="too large"):
            c.get("https://fixture.hf.space/")


def test_disconnected_episode_is_not_silently_replaced(monkeypatch):
    s = live.Session("fixture/env", "owner")
    class Socket:
        def send(self, message): pass
        def recv(self, timeout): raise TimeoutError
        def close(self): pass
    s.ws = Socket()
    with pytest.raises(live.SpaceError) as error:
        live._ws_send(s, {"type": "step", "data": {}}, 1)
    assert error.value.status == 504
    monkeypatch.setattr(live, "_open", lambda s: pytest.fail("must not create a replacement episode"))
    with pytest.raises(live.SpaceError) as error:
        live._ws_send(s, {"type": "step", "data": {}}, 1)
    assert error.value.status == 410


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 1.5])
def test_rollout_limits_reject_non_finite_and_fractional_values(value):
    with pytest.raises(ValueError):
        runner._check_fields([{"key": "steps", "type": "number", "step": 1, "min": 1, "max": 100}], {"steps": value})


def test_raw_file_names_are_unicode_safe_and_cannot_inject_headers(monkeypatch):
    monkeypatch.setattr(main.registry, "raw", lambda *a: (b"hello", "text/plain"))
    c = TestClient(main.app, base_url="https://testserver")
    for name in ['文件.txt', 'x\r\nInjected: true.txt']:
        r = c.get('/api/env/org/ds/raw', params={"ref": "train/0", "f": name, "download": "true"})
        assert r.status_code == 200
        assert r.headers['content-disposition'].startswith("attachment; filename*=UTF-8''")
        assert '\r' not in r.headers['content-disposition'] and '\n' not in r.headers['content-disposition']
        assert 'Injected' not in r.headers
        assert r.headers['content-security-policy'] == 'sandbox'


def test_query_parameters_survive_route_discovery_and_are_encoded():
    api = {"paths": {"/tasks/{id}": {"parameters": [{"name": "split", "in": "query", "required": True, "schema": {"type": "string"}}],
          "get": {"parameters": [{"name": "limit", "in": "query", "schema": {"type": "integer", "default": 10}}]}}}}
    info = {"api": live._api_routes(api, False)}
    schema = info["api"][0]["query_schema"]
    assert schema["required"] == ["split"] and schema["properties"]["limit"]["default"] == 10
    assert live.route_url(info, "GET", "/tasks/{id}", {"id": "a"}, {"split": "x&admin=true"}) == '/tasks/a?split=x%26admin%3Dtrue'
    with pytest.raises(live.SpaceError, match='required'):
        live.route_url(info, "GET", "/tasks/{id}", {"id": "a"})
    with pytest.raises(live.SpaceError, match='published'):
        live.route_url(info, "GET", "/tasks/{id}", {"id": "a"}, {"url": "https://evil.example"})


def test_mcp_routes_separate_body_path_and_query_and_keep_grading_routes(monkeypatch):
    from app import mcp_bridge
    api = {"paths": {"/verify/{id}": {"post": {
        "parameters": [{"name": "id", "in": "query", "schema": {"type": "string"}}],
        "requestBody": {"required": True, "content": {"application/json": {"schema": {"type": "object", "properties": {"id": {"type": "integer"}}, "required": ["id"]}}}},
    }}}}
    info = {"api": live._api_routes(api, False), "mcp": [{"name": "look", "inputSchema": {"type": "object"}}]}
    tools = {t['name']: t for t in mcp_bridge._tools(info)}
    assert {'look', 'verify_id'} == set(tools)
    assert set(tools['verify_id']['inputSchema']['properties']) == {'path_params', 'query', 'body'}
    captured = []
    monkeypatch.setattr(mcp_bridge, '_play', lambda *a: 'session')
    monkeypatch.setattr(live, 'act', lambda *args: captured.append(args[-1]) or {'result': {'json': {'reward': 1}, 'status': 200}})
    args = {"path_params": {"id": "path-id"}, "query": {"id": "query-id"}, "body": {"id": 42}}
    mcp_bridge._call('o/s', info, {"owner": "owner"}, 'verify_id', args)
    assert captured[0]['params'] == args['path_params'] and captured[0]['body'] == args['body'] and captured[0]['query'] == args['query']


def test_mcp_rate_limiter_does_not_accumulate_rejected_requests():
    from app import mcp_bridge
    owner = 'rate-regression'
    for _ in range(mcp_bridge.RATE[0] * 3):
        mcp_bridge._limited(owner)
    assert len(mcp_bridge._hits[owner]) == mcp_bridge.RATE[0]
    mcp_bridge._hits.pop(owner)
