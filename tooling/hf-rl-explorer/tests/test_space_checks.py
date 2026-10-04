import json
import time

import pytest

from app import space_checks as checks, spaces_live as live


def facts():
    api = {"info": {"version": "1.0.0"}, "paths": {
        "/reset": {"post": {}}, "/step": {"post": {}}, "/state": {"get": {}}, "/mcp": {"post": {}}}}
    return [api, {"status": "healthy"}, {"name": "test", "description": "test", "version": "9.9.9"},
            {"action": {}, "observation": {}, "state": {}}, None]


def test_typed_actions_do_not_require_mcp_but_production_does():
    f = facts()
    assert checks.assess(*f)["status"] == checks.PASS
    f[0]["paths"] = {"/mcp": {"post": {}}}
    assert checks.assess(*f)["status"] == "Checks failed"
    f[-1] = [{"name": "echo"}]
    assert checks.assess(*f)["mode"] == "MCP service"


@pytest.mark.parametrize("change", ["schema", "health", "route", "method", "html"])
def test_failures_cannot_become_api_checked(change):
    f = facts()
    if change == "schema":
        f[3].pop("state")
    elif change == "health":
        f[1]["status"] = "ok"
    elif change == "route":
        f[0]["paths"].pop("/step")
    elif change == "method":
        f[0]["paths"]["/step"] = {"get": {}}
    else:
        f[0] = "<html>Space UI</html>"
    assert checks.assess(*f)["status"] == "Checks failed"


@pytest.mark.parametrize("dependency,value,source", [
    ("openenv-core==0.2.3", "0.2.3", "Dependency pin"),
    ("openenv==0.7.0", "0.7.0", "Dependency pin"),
    ("openenv_core[core]>=0.2.0,<0.3", "<0.3,>=0.2.0", "Dependency constraint"),
    ("openenv-core==0.2.*", "==0.2.*", "Dependency constraint"),
    ("openenv-core", "Unpinned", "Dependency constraint"),
    ("openenv-core @ git+https://github.com/meta-pytorch/OpenEnv.git@abc123", "git:abc123", "Source reference"),
    ('openenv-core==0.2.3; python_version < "3.12"', "Conditional", "Dependency constraint"),
])
def test_declared_versions_keep_constraints_and_source_refs(dependency, value, source):
    v = checks.requirement_version(dependency)
    assert (v["value"], v["source"]) == (value, source)


def test_version_does_not_use_environment_or_api_version():
    assert checks.parse_version('[project]\nname="env"\nversion="9.9.9"\ndependencies=["fastapi"]', "pyproject.toml") is None
    assert checks.requirement_version("gymnasium==1.0.0") is None
    assert checks.tag_version(["openenv", "openenv-nonsense"])["value"] == "Unknown"
    assert checks.tag_version(["openenv-0.2.3"])["value"] == "0.2.3"


def test_lock_and_uv_source_precedence():
    doc = '[[package]]\nname="openenv-core"\nversion="0.7.1.dev0"\nsource={git="https://example.org/env?rev=main#abcdef"}'
    assert checks.parse_version(doc, "uv.lock") == {"value": "git:abcdef", "source": "Lockfile"}
    doc = '[project]\ndependencies=["openenv-core>=0.2"]\n[tool.uv.sources]\nopenenv-core={git="https://example.org/env",rev="v0.2.3"}'
    assert checks.parse_version(doc, "pyproject.toml")["value"] == "git:v0.2.3"
    assert checks.parse_version(doc.replace("openenv-core", "openenv"), "pyproject.toml")["value"] == "git:v0.2.3"


def test_expiry_future_dates_and_newer_failures(tmp_path, monkeypatch):
    monkeypatch.setattr(checks.config, "STORAGE_DIR", tmp_path)
    now = time.time()
    passing = {"schema": 1, "id": "owner/env", "checked_at": now - 10, "status": checks.PASS}
    checks.save(passing)
    assert checks.status(checks.inventory()["owner/env"], now) == checks.PASS
    assert checks.status(passing, now + checks.FRESH) == "Check expired"
    assert checks.status(passing, now - 20) == "Check expired"
    checks.save({**passing, "checked_at": now, "status": "Checks failed"})
    checks.save(passing)  # late completion cannot overwrite a newer failure
    assert checks.status(checks.inventory()["owner/env"], now) == "Checks failed"


def test_sleeping_space_is_not_contacted_or_woken(tmp_path, monkeypatch):
    monkeypatch.setattr(checks.config, "STORAGE_DIR", tmp_path)
    monkeypatch.setattr(live, "record", lambda *a, **kw: {"stage": "SLEEPING"})
    monkeypatch.setattr(live, "SpaceClient", lambda *a, **kw: pytest.fail("contacted sleeping server"))
    r = checks.check("owner/sleeping")
    assert r["status"] == "Not running"


def test_network_failure_invalidates_previous_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(checks.config, "STORAGE_DIR", tmp_path)
    checks.save({"schema": 1, "id": "owner/env", "checked_at": time.time() - 10, "status": checks.PASS})
    def offline(*a, **kw):
        raise ConnectionError("offline")
    monkeypatch.setattr(live, "record", offline)
    assert checks.check("owner/env")["status"] == "Check unavailable"
    assert checks.status(checks.inventory()["owner/env"]) != checks.PASS


def test_concurrent_refresh_skips_locked_queue(tmp_path, monkeypatch):
    fcntl = pytest.importorskip("fcntl")
    monkeypatch.setattr(checks.config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(checks, "_scan", lambda: pytest.fail("started duplicate queue"))
    with (tmp_path / "openenv-checks.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert checks.scan() == []


def test_automatic_queue_includes_inactive_and_untagged_without_starvation():
    candidates = [(f"org/active{i}", "RUNNING") for i in range(120)] + [(f"org/sleep{i}", "SLEEPING") for i in range(40)]
    candidates += [("FineEnvs/recent", "RUNNING"), ("FineEnvs/new", "RUNNING")]
    now = time.time()
    old = {"FineEnvs/recent": {"stage": "RUNNING", "checked_at": now, "status": checks.PASS, "task_catalog": {}}}
    pending = checks.pending(candidates, old, now)
    assert len(pending) == 128 and pending[0] == "FineEnvs/new"
    assert sum("sleep" in spec for spec in pending) == 32
    assert "FineEnvs/recent" not in pending
    for spec in pending:
        old[spec] = {"checked_at": now, "stage": "SLEEPING", "status": "Not running"}
    next_batch = checks.pending(candidates, old, now)
    assert next_batch and not set(next_batch) & set(pending)


def test_other_environment_interfaces_need_published_methods_and_fresh_running_checks():
    api = {"info": {"version": "1"}, "paths": {"/seed_session": {"post": {}}, "/verify": {"post": {}}}}
    assert checks.environment_interface(api) == "nemo-gym"
    rec = {"interface": "nemo-gym", "status": "Checks failed", "stage": "RUNNING", "checked_at": time.time()}
    assert checks.browseable(rec) and not checks.verified(rec)
    assert not checks.browseable({**rec, "checked_at": time.time() - checks.FRESH})
    assert not checks.browseable({**rec, "stage": "SLEEPING"})
    assert not checks.browseable({**rec, "status": "Check unavailable"})
    api["paths"]["/verify"] = {"get": {}}
    assert checks.environment_interface(api) is None
    assert checks.environment_interface(facts()[0]) is None  # generic reset/step alone is insufficient
