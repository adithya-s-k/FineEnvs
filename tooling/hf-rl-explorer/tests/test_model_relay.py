from fastapi.testclient import TestClient

from app import main, runner, runtime
from app.mimo.runner import core


def test_local_relay_precedes_capture_catchall_and_keeps_capability_auth(monkeypatch):
    monkeypatch.setattr(runner, "_service", None)
    client = TestClient(runner.service().capture.app)
    monkeypatch.setattr(core, "by_cap", lambda cap: None)
    assert client.get("/rlx/llm/invalid/health").status_code == 404
    assert client.post("/rlx/llm/invalid/v1/chat/completions", json={}).status_code == 404
    assert client.post("/v1/chat/completions", json={"model": "test", "messages": []}).status_code == 401
    async def relay(cap, request):
        return {"route": "scoped-relay"}
    monkeypatch.setattr(main, "llm_proxy", relay)
    assert client.post("/rlx/llm/test/v1/chat/completions", json={}).json() == {"route": "scoped-relay"}


def test_local_and_deployed_capabilities_are_redacted_from_logs():
    for prefix in ("api", "rlx"):
        msg = f"POST /{prefix}/llm/SECRET-CAP/v1/chat/completions"
        assert "SECRET-CAP" not in runtime.SECRET_PATH.sub(lambda m: (m.group(1) or m.group(2)) + "[hidden]", msg)
