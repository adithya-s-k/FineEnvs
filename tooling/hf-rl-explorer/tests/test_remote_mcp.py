"""Standard MCP sessions preserve cookies, negotiated headers and reply identity."""
import json

import httpx
import pytest

from app import spaces_live as live


def test_standard_mcp_initializes_isolates_and_closes_sessions():
    seen, next_id = [], []
    def handle(request):
        seen.append(request)
        if request.method == "DELETE":
            return httpx.Response(204)
        message = json.loads(request.content)
        if message["method"] == "initialize":
            sid = str(len(next_id) + 1)
            next_id.append(sid)
            return httpx.Response(200, headers={"Mcp-Session-Id": sid, "Set-Cookie": f"episode={sid}; Path=/"},
                                  json={"jsonrpc": "2.0", "id": message["id"], "result": {"protocolVersion": "2025-06-18"}})
        sid = request.headers["mcp-session-id"]
        assert request.headers["mcp-protocol-version"] == "2025-06-18"
        assert request.headers["cookie"] == f"episode={sid}"
        if message["method"] == "notifications/initialized":
            assert "id" not in message
            return httpx.Response(202)
        # A progress notification is not the result; the actual event has multi-line data.
        event = json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": {"episode": sid}}, indent=2)
        text = 'data: {"jsonrpc":"2.0","method":"notifications/progress"}\n\n' + '\n'.join('data: '+line for line in event.splitlines())+'\n\n'
        return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, text=text)
    with live.SpaceClient(transport=httpx.MockTransport(handle)) as a, live.SpaceClient(transport=httpx.MockTransport(handle)) as b:
        first = live.RemoteMCP({"host": "https://fixture.hf.space"}, a)
        second = live.RemoteMCP({"host": "https://fixture.hf.space"}, b)
        assert first.call("tools/call", {}, 2) == {"episode": "1"}
        assert second.call("tools/call", {}, 2) == {"episode": "2"}
        assert first.call("tools/call", {}, 2) == {"episode": "1"}
        first.close()
        second.close()
    assert len(next_id) == 2
    assert [r.headers["mcp-session-id"] for r in seen if r.method == "DELETE"] == ["1", "2"]


def test_expired_mcp_session_does_not_replay_or_reinitialize():
    calls = []
    def handle(request):
        calls.append(request)
        return httpx.Response(404)
    with live.SpaceClient(transport=httpx.MockTransport(handle)) as c:
        remote = live.RemoteMCP({"host": "https://fixture.hf.space"}, c)
        remote.initialized, remote.session = True, "existing"
        for _ in range(2):
            with pytest.raises(live.SpaceError) as error:
                remote.call("tools/call", {"name": "step"}, 2)
            assert error.value.status == 410
        assert len(calls) == 1


def test_mcp_rejects_an_unrelated_rpc_response():
    with live.SpaceClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"jsonrpc": "2.0", "id": "other", "result": {}}))) as c:
        with pytest.raises(live.SpaceError, match="matching"):
            live.RemoteMCP({"host": "https://fixture.hf.space"}, c).call("tools/list", {}, 2)
