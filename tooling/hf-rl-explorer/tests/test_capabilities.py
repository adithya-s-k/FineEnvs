"""Support claims reflect runtime evidence, not a source's framework tag."""
import pytest

from app import spaces_live as live
from app.envs.capabilities import dataset_support, space_support


def state(support, feature):
    return next(c["state"] for c in support["capabilities"] if c["id"] == feature)


@pytest.mark.parametrize("framework,runner", [("Verifiers", "verifiers"), ("NeMo Gym", "nemo-gym"), ("verl", "verl"), ("SkyRL", "skyrl")])
def test_native_framework_dataset_does_not_claim_local_execution(framework, runner):
    support = dataset_support(framework, options=[{"runner": runner, "label": framework, "ok": False, "why": "Use its native harness"}])
    assert state(support, "browse") == "available"
    assert state(support, "run") == "external"
    assert support["framework"]["docs"].startswith("https://")


def test_task_support_distinguishes_blocked_from_runnable():
    options = [{"runner": "harbor", "label": "Harbor", "ok": False, "why": "GPU required"}]
    assert state(dataset_support("Harbor", options=options), "run") == "unavailable"
    options[0]["ok"] = True
    assert state(dataset_support("Harbor", options=options), "run") == "available"
    assert state(dataset_support("Harbor"), "run") == "unknown"


def test_sleeping_server_never_claims_live_capabilities():
    s = space_support({"running": False, "stage": "SLEEPING", "last_seen": {"step_api": True}})
    assert state(s, "play") == "unknown"
    assert s["transport"] is None


@pytest.mark.parametrize("protocol", ["openenv", "gymnasium", "nemo-gym", "mcp", "api"])
def test_probe_uses_published_protocol_not_reset_route_or_tag(monkeypatch, protocol):
    from app import catalog
    catalog._memo.clear()
    rec = {"id": "fixture/env", "stage": "RUNNING", "host": "https://fixture-env.hf.space", "base_path": None, "tags": ["openenv"]}
    monkeypatch.setattr(live, "record", lambda *a, **k: rec)
    paths = {"/reset": {"post": {}}, "/step": {"post": {}}} if protocol in {"openenv", "gymnasium"} else (
        {"/seed_session": {"post": {}}, "/verify": {"post": {}}} if protocol == "nemo-gym" else {"/mcp": {"post": {}}} if protocol == "mcp" else {"/guess": {"post": {}}})
    docs = {"/openapi.json": {"info": {"version": "1"}, "paths": paths}}
    if protocol == "openenv":
        docs["/schema"] = {k: {"type": "object", "properties": {}} for k in ("action", "observation", "state")}
    monkeypatch.setattr(live, "_get_json", lambda rec, path: docs.get(path))
    monkeypatch.setattr(live, "_ui_path", lambda rec: None)
    monkeypatch.setattr(live, "_tools", lambda rec: ([{"name": "guess"}] if protocol == "mcp" else [], None, {}))
    monkeypatch.setattr(live, "remember", lambda *a: None)
    info = live.probe("fixture/env")
    assert info["framework"] == protocol
    assert info["openenv"] is (protocol == "openenv")
    assert info["step_api"] is (protocol == "openenv")
    if protocol == "gymnasium":
        assert {r["path"] for r in info["api"]} == {"/reset", "/step"}
        assert info["support"]["transport"] == "http"
    assert state(info["support"], "play") == "available"


def test_private_dataset_does_not_offer_public_mcp():
    assert state(dataset_support("Harbor", restricted=True), "mcp") == "unavailable"


def test_packed_harbor_rows_keep_the_harbor_lifecycle():
    assert dataset_support("Harbor (packed)")["framework"]["id"] == "harbor"


def test_unknown_runner_is_not_claimed_as_hosted():
    assert state(dataset_support("Custom", options=[{"runner": "custom", "label": "Custom", "ok": True}]), "run") == "unavailable"
