"""An environment Space read as files, and what a live probe leaves behind (no network: the Hub is stubbed).

What is checked: openenv.yaml in its variants (canonical, legacy class names, hackathon task lists, the `validation:`
block, a manifest in a subfolder, broken or explosive YAML); models.py, server/app.py and the environment read with
`ast`, never imported or run; vendored OpenEnv and build output left out of the tree; the file endpoint serves only
listed text files (no traversal, no binaries, no answers); private Spaces are refused and no token is ever sent;
the last good probe is kept and shown while the Space sleeps; `openenv validate`'s six criteria are judged right.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import catalog, config
from app import space_files as sf
from app import spaces_live as live

api = FastAPI()
api.include_router(sf.router)
client = TestClient(api)

SHA = "a" * 40


@pytest.fixture(autouse=True)
def fresh():
    catalog._memo.clear()
    yield
    catalog._memo.clear()


def sibling(name, size=100, lfs=False):
    return SimpleNamespace(rfilename=name, size=size, lfs={"sha256": "x"} if lfs else None)


def stub_space(monkeypatch, tmp_path, files: dict[str, str | bytes], private=False, extra: list[str] = (), seen_tokens=None):
    """A public Space whose repository holds `files` (path -> content) and `extra` (listed only)."""
    sibs = [sibling(p, len(c.encode() if isinstance(c, str) else c)) for p, c in files.items()] + [sibling(p) for p in extra]
    sp = SimpleNamespace(sha=SHA, private=private, gated=False, siblings=sibs, subdomain="org-env", sdk="docker", tags=["openenv"],
                         card_data=SimpleNamespace(to_dict=lambda: {"title": "Env", "base_path": "/web", "app_port": 8000}),
                         runtime=SimpleNamespace(stage="SLEEPING", hardware=None, requested_hardware="cpu-basic"),
                         last_modified=None)

    def space_info(spec, files_metadata=False, expand=None):
        return sp

    def _api(token=None):
        if seen_tokens is not None:
            seen_tokens.append(("api", token))
        return SimpleNamespace(space_info=space_info)

    calls = []

    def download(repo_id, filename, repo_type=None, revision=None, token=None, cache_dir=None, **kw):
        calls.append({"repo": repo_id, "file": filename, "type": repo_type, "revision": revision, "token": token})
        p = tmp_path / "dl" / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        data = files[filename]
        p.write_bytes(data.encode() if isinstance(data, str) else data)
        return str(p)

    import huggingface_hub

    monkeypatch.setattr(catalog, "_api", _api)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    return calls


# ── openenv.yaml ─────────────────────────────────────────────────────────────
CANONICAL = """spec_version: 1
name: echo_env
type: space
runtime: fastapi
app: server.app:app
port: 8000
"""


def test_canonical_manifest_keys():
    m = sf.parse_manifest(CANONICAL, "openenv.yaml")
    assert m["keys"] == {"spec_version": 1, "name": "echo_env", "type": "space", "runtime": "fastapi", "app": "server.app:app", "port": 8000}
    assert "validation" not in m and "tasks" not in m and "error" not in m


def test_legacy_manifest_names_its_classes():
    m = sf.parse_manifest('name: browsergym_env\nversion: "0.1.0"\naction: BrowserGymAction\nobservation: BrowserGymObservation\n', "openenv.yaml")
    assert m["legacy"] == {"action": "BrowserGymAction", "observation": "BrowserGymObservation"}
    assert m["keys"]["version"] == "0.1.0"


def test_hackathon_task_lists_in_every_shape_and_without_answers():
    listed = sf.parse_manifest("""name: triage
tasks:
  - id: easy
    name: Direct injection
    difficulty: easy
    description: Detect it.
    expected_score_range: [0.2, 0.5]
    gold_answer: "yes"
  - id: hard
    level: hard
reward:
  min: 0.0
  max: 1.0
  expected_output: 42
""", "openenv.yaml")
    assert [t["id"] for t in listed["tasks"]] == ["easy", "hard"]
    assert listed["tasks"][0]["difficulty"] == "easy" and listed["tasks"][1]["difficulty"] == "hard"
    flat = json.dumps(listed)
    assert "gold_answer" not in flat and "expected_score_range" not in flat and "42" not in flat
    assert "expected_output" in listed["withheld"]
    strings = sf.parse_manifest("name: x\ntasks: [easy, medium, hard]\n", "openenv.yaml")
    assert [t["id"] for t in strings["tasks"]] == ["easy", "medium", "hard"]
    keyed = sf.parse_manifest("name: x\ntasks:\n  easy: {description: one}\n  hard: two\n", "openenv.yaml")
    assert keyed["tasks"] == [{"id": "easy", "description": "one"}, {"id": "hard", "description": "two"}]


def test_validation_block():
    m = sf.parse_manifest(CANONICAL + """validation:
  reward: {range: [0.0, 1.0], oracle_tolerance: 0.01, floor_margin: 0.5, variance_tolerance: 0.05}
  judge: {model: Qwen/Qwen3-8B, version: "2025-05", params: {temperature: 0}}
  resources: {cpu: 2, memory_mb: 4096, disk_mb: 2048, episode_timeout_s: 600}
  network: {mode: allowlist, allowed_hosts: [api.example.com]}
  capabilities:
    verifier: {kind: reward_channel}
    llm_judged: true
    task_api: true
    declared_tools: [look, submit_guess]
    declared_task_count: {train: 1000, eval: 200}
  types: {tags: [geo]}
""", "openenv.yaml")
    v = m["validation"]
    assert v["reward"]["range"] == [0.0, 1.0] and v["reward"]["variance_tolerance"] == 0.05
    assert v["judge"]["model"] == "Qwen/Qwen3-8B"
    assert v["capabilities"]["llm_judged"] is True and v["capabilities"]["verifier"]["kind"] == "reward_channel"
    assert v["capabilities"]["declared_tools"] == ["look", "submit_guess"]
    assert v["capabilities"]["declared_task_count"] == {"train": 1000, "eval": 200}
    assert v["network"] == {"mode": "allowlist", "allowed_hosts": ["api.example.com"]}
    assert v["resources"]["memory_mb"] == 4096 and v["types"] == ["geo"]


def test_broken_and_explosive_yaml_is_said_so_quickly():
    assert sf.parse_manifest("name: [unclosed\n", "openenv.yaml")["error"].startswith("not valid YAML")
    assert sf.parse_manifest("- just\n- a list\n", "openenv.yaml")["error"] == "not a YAML mapping"
    laughs = "a: &a [x, x, x, x, x, x, x, x, x]\n" + "".join(f"{c}: &{c} [*{p}, *{p}, *{p}, *{p}, *{p}, *{p}, *{p}, *{p}, *{p}]\n"
                                                         for p, c in zip("abcdefgh", "bcdefghi")) + "rubric: *i\n"
    t = time.time()
    m = sf.parse_manifest(laughs, "openenv.yaml")
    assert time.time() - t < 2 and len(json.dumps(m)) < 2_000_000


def test_the_shallowest_manifest_wins_and_ties_follow_the_dockerfile():
    paths = ["envs/deep/x/openenv.yaml", "env/openenv.yaml", "geoguesser_env/openenv.yaml", "src/openenv/cli/templates/openenv_env/openenv.yaml"]
    assert sf.manifest_path(paths, "FROM python:3.12\nCOPY geoguesser_env /app/geoguesser_env\n", "FineEnvs/geoguesser-env") == "geoguesser_env/openenv.yaml"
    assert sf.manifest_path(paths, None, "FineEnvs/geoguesser-env") == "geoguesser_env/openenv.yaml"   # named like the Space
    assert sf.manifest_path(paths, None, "someone/other") == "env/openenv.yaml"
    assert sf.manifest_path(["a/openenv.yaml", "openenv.yaml"], None, "o/n") == "openenv.yaml"
    assert sf.manifest_path(["src/openenv/cli/templates/openenv_env/openenv.yaml"], None, "o/n") is None


def test_a_subfolder_manifest_roots_the_environment(monkeypatch, tmp_path):
    files = {"Dockerfile": "FROM python:3.12\nCOPY my_env /app/my_env\nENV ENABLE_WEB_INTERFACE=true\n",
             "my_env/openenv.yaml": CANONICAL.replace("echo_env", "my_env"),
             "my_env/models.py": "from openenv.core.env_server.types import Action, Observation\nclass A(Action):\n    x: int = 1\n",
             "my_env/server/app.py": "app = create_app(E, A, O, env_name='my_env')\n",
             "my_env/server/my_environment.py": "class E(Environment):\n    pass\n",
             "my_env/pyproject.toml": '[project]\nname = "openenv-my-env"\ndependencies = ["openenv-core>=0.2.1"]\n'}
    stub_space(monkeypatch, tmp_path, files)
    d = sf.declared("org/env")
    assert d["root"] == "my_env" and d["sha"] == SHA and d["short"] == SHA[:7]
    assert d["keys"]["manifest"] == "my_env/openenv.yaml" and d["keys"]["models"] == "my_env/models.py"
    assert d["declared"]["manifest"]["keys"]["name"] == "my_env"
    assert d["declared"]["app"]["env_name"] == "my_env"
    assert d["declared"]["dockerfile"]["web_interface"] is True
    assert d["declared"]["pyproject"]["openenv_version"] == ">=0.2.1"
    assert d["declared"]["card"]["base_path"] == "/web"


# ── Python, parsed and never run ─────────────────────────────────────────────
MODELS = '''
import os
os.system("touch {marker}")
raise SystemExit("models.py must never run here")

from enum import Enum
from pydantic import Field
from openenv.core.env_server.types import Action, Observation, State


class Mode(str, Enum):
    EASY = "easy"
    HARD = "hard"


class Base(Action):
    """Every action."""


class MoveAction(Base):
    """Walk along the road.

    More detail."""
    direction: str = Field(..., description="forward or backward", pattern="^(forward|backward)$")
    meters: float = Field(default=10.0, gt=0, description="how far")
    tags: list[str] = Field(default_factory=list)
    _private: int = 0


class GeoObservation(Observation):
    image_b64: str = Field(default="", description="the view, as base64 PNG")
    feedback: str = ""
    model_config = {{"extra": "forbid"}}


class GeoState(State):
    step_count: int = 0
'''


def test_models_are_parsed_and_never_executed(tmp_path):
    marker = tmp_path / "pwned"
    m = sf.parse_models(MODELS.format(marker=marker), "models.py")   # would raise SystemExit, or touch the marker, if run
    assert not marker.exists()
    by = {c["name"]: c for c in m["classes"]}
    assert by["MoveAction"]["role"] == "action" and by["Base"]["role"] == "action"     # through Base, within the file
    assert by["GeoObservation"]["role"] == "observation" and by["GeoState"]["role"] == "state"
    assert by["Mode"]["kind"] == "enum" and by["Mode"]["values"] == ["'easy'", "'hard'"] and by["Mode"]["role"] is None
    assert by["MoveAction"]["doc"] == "Walk along the road."
    f = {x["name"]: x for x in by["MoveAction"]["fields"]}
    assert set(f) == {"direction", "meters", "tags"}                                      # private fields left out
    assert f["direction"]["required"] is True and f["direction"]["description"] == "forward or backward"
    assert f["meters"]["default"] == "10.0" and not f["meters"]["required"] and "gt 0" in f["meters"]["limits"]
    assert f["tags"]["default"] == "list()" and f["tags"]["type"] == "list[str]"
    o = {x["name"]: x for x in by["GeoObservation"]["fields"]}
    assert o["image_b64"].get("image") is True and "model_config" not in o
    assert sf.parse_models("def broken(:\n", "models.py")["error"]


def test_server_app_and_environment_are_read_not_run(tmp_path):
    marker = tmp_path / "ran"
    app_py = f'''
import os
open("{marker}", "w").write("x")
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT_ENVS", "4"))
app = create_app(GeoEnvironment, GeoAction, GeoObservation, env_name="geo", max_concurrent_envs=MAX_CONCURRENT,
                 gradio_builder=build_ui, custom_tab_name="Play")

@app.get("/geo/task/{{i}}")
def task(i: int): ...
'''
    a = sf.parse_app(app_py, "server/app.py")
    assert not marker.exists()
    assert (a["env"], a["action"], a["observation"], a["env_name"]) == ("GeoEnvironment", "GeoAction", "GeoObservation", "geo")
    assert a["max_concurrent_envs"] == 4 and a["gradio_builder"] == "build_ui" and a["custom_tab_name"] == "Play"
    assert a["routes"] == ["GET /geo/task/{i}"] and not a["mcp"]
    assert sf.parse_app("app = create_app(E, CallToolAction, CallToolObservation, env_name='echo')", "server/app.py")["mcp"] is True

    env = sf.parse_environment('''
from openai import OpenAI
from .scoring import compute_reward
from openenv.core.env_server.mcp_environment import MCPEnvironment

class GeoEnvironment(MCPEnvironment):
    def _register(self, mcp):
        @mcp.tool
        def look(heading_deg: float, fov_deg: float = 90):
            """Render a view."""
        @mcp.tool()
        def submit_guess(lat: float, lon: float):
            """Commit the answer."""

    def _grade(self, guess):
        return compute_reward(guess)
''', "server/geo_environment.py")
    assert [t["name"] for t in env["tools"]] == ["look", "submit_guess"]
    assert env["tools"][0]["params"] == ["heading_deg", "fov_deg"] and env["tools"][0]["description"] == "Render a view."
    assert env["llm_judge"] is True and env["mcp"] is True and "_grade" in env["functions"]
    assert sf.reward_files("server/geo_environment.py", env["reward_imports"], {"server/scoring.py"}, "") == ["server/scoring.py"]
    assert sf.parse_environment("from openenv.core.rubrics import Rubric\nclass R(Rubric): pass\n", "e.py")["rubric"] is True


def test_dockerfile_and_pyproject_facts():
    d = sf.parse_dockerfile("ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest\nFROM ${BASE_IMAGE} AS builder\n"
                            "FROM ${BASE_IMAGE}\nENV ENABLE_WEB_INTERFACE=true\nEXPOSE 7860\n"
                            'CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]\n', "Dockerfile")
    assert d["from"] == ["ghcr.io/meta-pytorch/openenv-base:latest"] and d["web_interface"] is True
    assert d["expose"] == 7860 and d["port"] == 8000
    assert sf.parse_dockerfile("FROM python:3.12\nENV ENABLE_WEB_INTERFACE false\n", "Dockerfile")["web_interface"] is False
    git = sf.parse_pyproject('[project]\nname = "openenv-echo-env"\ndependencies = ["fastapi", '
                             '"openenv-core[core] @ git+https://github.com/meta-pytorch/OpenEnv.git@v0.2.3"]\n'
                             '[project.scripts]\nserver = "echo_env.server.app:main"\n', "pyproject.toml")
    assert git["package"] == "openenv-echo-env" and git["openenv_version"] == "0.2.3" and git["server_script"] == "echo_env.server.app:main"
    assert sf.parse_pyproject('[project]\ndependencies = ["openenv==0.3.1"]\n', "p")["openenv_version"] == "0.3.1"
    assert "openenv" not in sf.parse_pyproject('[project]\ndependencies = ["openenv-echo-env>=1"]\n', "p")   # another package
    assert sf.parse_pyproject("not = [toml", "p")["error"]


# ── the tree ─────────────────────────────────────────────────────────────────
def test_junk_is_left_out_of_the_tree():
    info = {"files": [(p, 10, False) for p in (
        ".gitattributes", "README.md", "openenv.yaml", "models.py", "server/app.py", "uv.lock",
        "src/openenv/core/env_server/http_server.py", "src/core/env_client.py", "src/openenv_core/__init__.py",
        "build/lib/server/app.py", "envs/echo_env/build/lib/echo_env/client.py", "openenv_echo_env.egg-info/PKG-INFO",
        "server/__pycache__/app.cpython-312.pyc", "envs/echo_env/server/app.py", "solution/solve.py",
        "tasks/eval.jsonl", "data/geo/countries.geojson")] + [("assets/map.png", 10, True)]}
    files, dropped, truncated = sf.listing(info)
    paths = [f["path"] for f in files]
    assert paths == sorted(["README.md", "openenv.yaml", "models.py", "server/app.py", "uv.lock", "envs/echo_env/server/app.py",
                            "solution/solve.py", "tasks/eval.jsonl", "data/geo/countries.geojson", "assets/map.png"])
    assert dropped == 8 and not truncated
    by = {f["path"]: f for f in files}
    assert by["uv.lock"]["collapsed"] and by["assets/map.png"]["binary"]
    assert by["solution/solve.py"]["withheld"] and by["tasks/eval.jsonl"]["withheld"]
    assert not by["data/geo/countries.geojson"].get("withheld") and not by["openenv.yaml"].get("withheld")


# ── one file ─────────────────────────────────────────────────────────────────
def test_the_file_endpoint_serves_only_listed_text(monkeypatch, tmp_path):
    big = "x" * (sf.MAX_TEXT + 10)
    calls = stub_space(monkeypatch, tmp_path, {
        "openenv.yaml": CANONICAL, "models.py": "class A: pass\n", "big.txt": big,
        "blob.dat": b"\x00\x01binary", "picture.png": b"\x89PNG", "config/answers.json": '{"x": 1}',
        "data/manifest.json": '{"steps": [{"original_bash": "ls"}]}',
        "uv.lock": 'version = 1\n[[package]]\nname = "fastapi"\nversion = "0.115.0"\n[[package]]\nname = "openenv-core"\nversion = "0.2.3"\n',
    }, extra=["src/openenv/core/secret.py"])
    r = client.get("/api/spaces/org/env/file", params={"f": "models.py"})
    assert r.status_code == 200 and r.json()["text"] == "class A: pass\n" and r.json()["sha"] == SHA
    for bad in ("../etc/passwd", "/etc/passwd", "server/../models.py", "a\\b", "./models.py", ""):
        assert client.get("/api/spaces/org/env/file", params={"f": bad}).status_code in (400, 404), bad
    assert client.get("/api/spaces/org/env/file", params={"f": "src/openenv/core/secret.py"}).status_code == 404   # junk: not served
    assert client.get("/api/spaces/org/env/file", params={"f": "nothere.py"}).status_code == 404
    assert client.get("/api/spaces/org/env/file", params={"f": "picture.png"}).status_code == 415   # binary by name
    assert client.get("/api/spaces/org/env/file", params={"f": "blob.dat"}).status_code == 415      # binary by content
    long = client.get("/api/spaces/org/env/file", params={"f": "big.txt"}).json()
    assert long["truncated"] and len(long["text"]) == sf.MAX_TEXT
    for answers in ("config/answers.json", "data/manifest.json"):                                  # by name; by its keys
        w = client.get("/api/spaces/org/env/file", params={"f": answers}).json()
        assert w["withheld"] and "text" not in w, answers
    lock = client.get("/api/spaces/org/env/file", params={"f": "uv.lock"}).json()
    assert lock["collapsed"] and "openenv-core  0.2.3" in lock["text"] and "[[package]]" not in lock["text"]
    assert {c["file"] for c in calls} <= {"models.py", "big.txt", "blob.dat", "data/manifest.json", "uv.lock"}   # never the junk or the png
    assert all(c["token"] is False and c["revision"] == SHA and c["type"] == "space" for c in calls)


def test_files_endpoint_shape_and_no_token(monkeypatch, tmp_path):
    tokens = []
    stub_space(monkeypatch, tmp_path, {"openenv.yaml": CANONICAL, "README.md": "# hi\n"}, extra=["src/core/x.py", "build/lib/a.py"], seen_tokens=tokens)
    r = client.get("/api/spaces/org/env/files", headers={"Authorization": "Bearer hf_visitor_token"})
    assert r.status_code == 200
    d = r.json()
    assert [f["path"] for f in d["files"]] == ["README.md", "openenv.yaml"] and d["dropped"] == 2
    assert d["subdomain"] == "org-env" and d["hardware"] == "cpu-basic" and d["declared"]["manifest"]["keys"]["name"] == "echo_env"
    assert tokens and all(t is None for _, t in tokens)                                             # catalog._api(): anonymous


def test_private_spaces_are_refused(monkeypatch, tmp_path):
    stub_space(monkeypatch, tmp_path, {"openenv.yaml": CANONICAL}, private=True)
    assert client.get("/api/spaces/org/env/files").status_code == 404
    assert client.get("/api/spaces/org/env/file", params={"f": "openenv.yaml"}).status_code == 404
    from huggingface_hub.errors import RepositoryNotFoundError

    def missing(spec, **kw):
        raise RepositoryNotFoundError("nope", response=SimpleNamespace(status_code=404, headers={}, request=None))

    catalog._memo.clear()
    monkeypatch.setattr(catalog, "_api", lambda token=None: SimpleNamespace(space_info=missing))
    assert client.get("/api/spaces/org/env/files").status_code == 404
    assert client.get("/api/spaces/../x/files").status_code in (400, 404)


# ── last seen, and openenv validate's criteria ───────────────────────────────
SCHEMA = {"action": {"properties": {"message": {"type": "string"}}}, "observation": {"properties": {"reward": {}}}, "state": {"properties": {}}}
OPENAPI = {"info": {"version": "1.0.0"}, "paths": {p: {} for p in ("/reset", "/step", "/state", "/metadata", "/health", "/schema", "/mcp")}}


def stub_running(monkeypatch, stage="RUNNING"):
    rec = {"id": "org/env", "host": "https://org-env.hf.space", "stage": stage, "base_path": "/web"}
    monkeypatch.setattr(live, "record", lambda spec, fresh=False: rec)
    answers = {"/openapi.json": OPENAPI, "/metadata": {"name": "env", "description": "an env", "readme_content": "x" * 5000},
               "/schema": SCHEMA, "/health": {"status": "healthy"}, "/list_environments": None}
    monkeypatch.setattr(live, "_get_json", lambda rec, path: answers.get(path))
    monkeypatch.setattr(live, "_ui_path", lambda rec: "/web/")
    monkeypatch.setattr(live, "_tools", lambda rec: ([{"name": "echo", "inputSchema": {"type": "object"}}], None, {"ok": True, "detail": "JSON-RPC 2.0"}))
    return rec


def test_a_good_probe_is_kept_and_shown_while_the_space_sleeps(monkeypatch):
    rec = stub_running(monkeypatch)
    info = live.probe("org/env")
    assert info["running"] and info["mode"] == "simulation" and info["openapi_version"] == "1.0.0"
    assert [c["status"] for c in info["conformance"]] == ["pass"] * 6
    p = Path(config.STORAGE_DIR) / "space-probes" / "org__env.json"
    assert p.exists() and not [x for x in p.parent.iterdir() if x.name.endswith(".tmp")]          # written whole
    kept = json.loads(p.read_text())
    assert kept["mcp"][0]["name"] == "echo" and kept["schema"] == SCHEMA and kept["ui"] == "/web/"
    assert kept["metadata"] == {"name": "env", "description": "an env"}                            # no README copy
    rec["stage"] = "SLEEPING"
    catalog._memo.clear()
    asleep = live.probe("org/env")
    assert not asleep["running"] and asleep["last_seen"]["mcp"][0]["name"] == "echo"
    assert asleep["last_seen"]["checked_at"] <= time.time() and len(asleep["last_seen"]["conformance"]) == 6


def test_a_probe_that_got_nothing_does_not_replace_the_last_good_one(monkeypatch):
    stub_running(monkeypatch)
    live.probe("org/env")
    catalog._memo.clear()
    monkeypatch.setattr(live, "_get_json", lambda rec, path: None)
    monkeypatch.setattr(live, "_tools", lambda rec: (None, "the Space's /mcp answered HTTP 503", {"ok": False, "detail": "HTTP 503"}))
    bad = live.probe("org/env")
    assert bad["running"] and bad["conformance"][0]["status"] == "fail"
    assert live.last_seen("org/env")["mcp"][0]["name"] == "echo"


def test_the_last_seen_record_is_trimmed_to_fit():
    huge = [{"name": f"t{i}", "inputSchema": {"description": "x" * 4000}, "outputSchema": {"d": "y" * 4000}} for i in range(120)]
    kept = live._seen_record("org/env", {"mcp": huge, "schema": SCHEMA, "checked": 1.0})
    assert len(json.dumps(kept)) <= live.SEEN_MAX and [t["name"] for t in kept["mcp"]] == [f"t{i}" for i in range(120)]


def test_conformance_criteria():
    allgood = {"openapi": OPENAPI, "health": {"status": "healthy"}, "metadata": {"name": "a", "description": "b"}, "schema": SCHEMA,
               "mcp": {"ok": True, "detail": "JSON-RPC 2.0"}, "mcp_tried": True, "paths": list(OPENAPI["paths"])}
    ids = [c["id"] for c in live.conformance(allgood)]
    assert ids == ["openapi_version_available", "health_endpoint", "metadata_endpoint", "schema_endpoint", "mcp_endpoint", "mode_endpoint_consistency"]
    assert all(c["status"] == "pass" for c in live.conformance(allgood))

    def status(facts):
        return {c["id"]: c["status"] for c in live.conformance({**allgood, **facts})}

    assert status({"openapi": {"paths": {}}})["openapi_version_available"] == "fail"
    assert status({"health": {"status": "ok"}})["health_endpoint"] == "fail"
    assert status({"metadata": {"name": "a"}})["metadata_endpoint"] == "fail"
    assert status({"schema": {"action": {}, "observation": {}}})["schema_endpoint"] == "fail"
    assert status({"mcp": {"ok": False, "detail": "HTTP 404"}})["mcp_endpoint"] == "fail"
    assert status({"mcp": None})["mcp_endpoint"] == "fail"                       # not among its routes
    assert status({"mcp": None, "paths": None})["mcp_endpoint"] == "unknown"
    assert status({"paths": ["/mcp", "/health"]})["mode_endpoint_consistency"] == "pass"           # production
    assert status({"paths": ["/mcp", "/step"]})["mode_endpoint_consistency"] == "fail"             # production with /step
    assert status({"paths": ["/reset", "/step"]})["mode_endpoint_consistency"] == "fail"           # simulation without /state
    assert status({"paths": None})["mode_endpoint_consistency"] == "fail"
    assert {c["status"] for c in live.conformance(None)} == {"unknown"}
