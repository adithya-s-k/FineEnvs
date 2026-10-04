"""The environment contract (app/envs/contract.py): a new format plugs in by registering an adapter, and then the
pages' API, the runner, rollouts across a task's forms and the MCP server all work for it, with nothing else changed
(no network: the Hub is stubbed).

What is checked: the surest adapter that confirms reads a source, and one that hands a source over isn't kept;
views come back without an adapter's private keys and with run options and aliases; files and folders go through the
adapter; a run is refused unless the task's chosen run option says it can run, with its own inputs checked and its
agents limited; rollouts of the same task elsewhere are listed with it; linkers add links; the dataset MCP server
lists, reads and withholds through the same adapter; row refs round-trip.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import auth, main, models, runner, settings, store
from app.envs import contract as c, registry, rows

client = TestClient(main.app, base_url="https://testserver")
TOKEN = "hf_contract_secret_token_0003"


def cookie() -> dict[str, str]:
    session = {"token": TOKEN, "name": "carol", "avatar": None, "via": "token", "exp": time.time() + 3600}
    return {"Cookie": f"{auth.COOKIE}={auth._box.encrypt(json.dumps(session).encode()).decode()}"}


class Toy(c.Adapter):
    """A made-up format: tasks are numbered puzzles; puzzle 2 holds its answer in a file; "3" can't run."""
    id, name, framework = "toy", "Toy puzzles", "Toy"
    about = "Each task is a numbered puzzle."

    def detect(self, kind, meta):
        return 0.95 if meta["id"].startswith("toy/") else 0.0

    def summary(self, env, subset=None):
        return {"state": "ready", "total": 3, "inline": True, "search": True, "subsets": [], "how": {"framework": "Toy", "name": self.name, "about": self.about},
                "facets": [c.facet("level", "Level", "level")], "overview": [c.section("graded", "How it's graded", [c.kv([("Reward", "1 if solved")])])]}

    def tasks(self, env, *, subset=None, q="", filters=None, offset=0, everything=False):
        cards = [c.card(f"p/{i}", f"Puzzle {i}", facets={"level": ["easy" if i < 2 else "hard"]}) for i in (1, 2, 3)]
        return {"cards": cards, "total": 3, "offset": 0, "page": 3}

    def task(self, env, ref):
        if ref not in ("p/1", "p/2", "p/3"):
            raise LookupError("no such puzzle")
        return {"ref": ref, "title": f"Puzzle {ref[2:]}", "sections": [c.section("task", "The task", [c.markdown("Solve it."),
                c.files([{"path": "README.md", "size": 3}, {"path": "answer.txt", "size": 2, "withheld": True}])])],
                "glance": [["Level", "easy"]], "withheld": ["answer"], "links": [], "_secret": "never shown"}

    def file(self, env, ref, path):
        return {"path": path, "withheld": True} if path == "answer.txt" else {"path": path, "text": "hi\n", "size": 3}

    def random(self, env, subset=None):
        return "p/1"

    def run_options(self, env, ref, view):
        return [c.run_option("harbor", "Harbor", ok=ref != "p/3", why="no image", default=True,
                             fields=[c.field("judge", "Judge", "select", default="a", options=["a", "b"]), c.field("hint", "Hint", "bool", default=False)],
                             harnesses=["opencode"])]

    def run_task(self, env, ref):
        return {"title": f"Puzzle {ref[2:]}", "restricted": False, "sha": "abc", "bytes": 10, "runnable": {"ok": True, "how": "image"}, "image": "python:3.12"}

    def aliases(self, env, ref):
        return [("toy/converted", f"tasks/{ref[2:]}")]


class Unsure(c.Adapter):
    """Sure of everything at first, then hands every source over (a Harbor tag on a dataset of rows)."""
    id = "unsure"

    def detect(self, kind, meta):
        return 0.99 if meta["id"].startswith("toy/") else 0.0

    def confirm(self, env):
        return False


@pytest.fixture(autouse=True)
def toy(monkeypatch):
    monkeypatch.setattr(registry, "_meta", lambda kind, spec, token: {"id": spec, "sha": "abc", "restricted": False})
    monkeypatch.setattr(registry, "ADAPTERS", [Unsure(), Toy(), *registry.ADAPTERS])
    monkeypatch.setattr(registry, "_choice", {})
    monkeypatch.setattr(models, "get", lambda mid: {"id": mid, "tools": True, "provider": "x"})
    monkeypatch.setattr(runner.Rollout, "execute", lambda self: None)
    monkeypatch.setattr(settings, "_org_members", lambda: set())


def test_the_surest_adapter_that_confirms_reads_it():
    env = registry.resolve("dataset", "toy/puzzles")
    assert env.adapter.id == "toy"
    d = client.get("/api/env/toy/puzzles").json()
    assert d["env"]["adapter"] == "toy" and d["env"]["framework"] == "Toy" and d["total"] == 3 and d["inline"]
    assert d["overview"][0]["blocks"][0] == {"type": "kv", "rows": [["Reward", "1 if solved"]]}


def test_task_views_are_public_and_complete():
    v = client.get("/api/env/toy/puzzles/task?ref=p/1").json()
    assert "_secret" not in json.dumps(v)
    assert v["env"]["adapter"] == "toy" and v["run"]["options"][0]["ok"] and v["run"]["note"] == ""
    assert v["aliases"] == [{"env": "toy/converted", "ref": "tasks/1"}]
    assert client.get("/api/env/toy/puzzles/task?ref=nope").status_code == 404
    assert client.get("/api/env/toy/puzzles/file?ref=p/2&f=answer.txt").json()["withheld"] is True
    assert client.get("/api/env/toy/puzzles/random").json() == {"ref": "p/1"}
    assert client.get("/api/env/toy/puzzles/folder?ref=p/1&f=x/").status_code == 404   # lists its folders up front


def test_linkers_add_links_without_either_adapter_knowing(monkeypatch):
    monkeypatch.setattr(registry, "LINKERS", [lambda env, ref, view: [c.link("Raw row", "/t/raw/ds/train/1", rel="same")]])
    v = client.get("/api/env/toy/puzzles/task?ref=p/1").json()
    assert v["links"] == [{"label": "Raw row", "href": "/t/raw/ds/train/1", "note": "", "rel": "same"}]


def test_runs_follow_the_chosen_run_option():
    post = lambda **kw: client.post("/api/runs", json={"dataset": "toy/puzzles", "path": "p/1", "model": "m/x", **kw}, headers=cookie())
    r = post(fields={"judge": "b"})
    assert r.status_code == 200, r.text
    run = r.json()["run"]
    assert run["adapter"] == "toy" and run["runner"] == "harbor" and run["fields"] == {"judge": "b", "hint": False} and run["env"] == "toy/puzzles"
    assert TOKEN not in r.text
    assert post(fields={"judge": "zzz"}).status_code == 400              # not one of its choices
    assert post(fields={"other": 1}).status_code == 400                  # an input it doesn't take
    assert post(fields={"hint": "yes"}).status_code == 400               # not a bool
    assert post(runner="mimo").status_code == 400                        # no such runner for this task
    assert post(runner="Bad Runner!").status_code == 422
    assert post(harness="terminus-2").status_code == 400                 # this runner drives only opencode
    r = client.post("/api/runs", json={"dataset": "toy/puzzles", "path": "p/3", "model": "m/x"}, headers=cookie())
    assert r.status_code == 400 and "no image" in r.json()["detail"]


def test_rollouts_of_the_same_task_elsewhere_are_listed_with_it():
    base = {"user": "carol", "title": "T", "model": "m/x", "harness": "opencode", "status": "done", "reward": 1.0, "visibility": "public"}
    here = store.create({**base, "id": "20260102-000000-aaaaaa", "dataset": "toy/puzzles", "path": "p/2", "task_id": "toy/puzzles:p/2"})
    there = store.create({**base, "id": "20260102-000000-bbbbbb", "dataset": "toy/converted", "path": "tasks/2", "task_id": "toy/converted:tasks/2", "user": "dave"})
    other = store.create({**base, "id": "20260102-000000-cccccc", "dataset": "toy/converted", "path": "tasks/3", "task_id": "toy/converted:tasks/3"})
    d = client.get("/api/env/toy/puzzles/runs?ref=p/2", headers=cookie()).json()
    assert [r["id"] for r in d["mine"]] == [here["id"]]
    assert [r["id"] for r in d["public"]] == [there["id"]] and "user" not in d["public"][0]
    assert other["id"] not in json.dumps(d)
    assert d["elsewhere"] == [{"env": "toy/converted", "ref": "tasks/2"}]


def test_the_dataset_mcp_server_uses_the_same_adapter():
    def rpc(method, params=None, **headers):
        return client.post("/mcp/d/toy/puzzles", json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}, headers=headers)

    init = rpc("initialize", {"protocolVersion": "2025-06-18"})
    assert init.status_code == 200 and init.headers.get("mcp-session-id") and "Toy" in init.json()["result"]["serverInfo"]["name"]
    names = [t["name"] for t in rpc("tools/list").json()["result"]["tools"]]
    assert names == ["describe", "list_tasks", "get_task", "read_file", "random_task"]
    call = lambda name, args: rpc("tools/call", {"name": name, "arguments": args}).json()["result"]
    assert json.loads(call("list_tasks", {"filters": {"level": ["hard"]}})["content"][0]["text"])["total"] == 2
    assert json.loads(call("list_tasks", {"query": "puzzle 3"})["content"][0]["text"])["tasks"][0]["ref"] == "p/3"
    text = call("get_task", {"ref": "p/2"})["content"][0]["text"]
    assert "Solve it." in text and "answer.txt (withheld)" in text and "never shown" not in text
    assert "withheld" in call("read_file", {"ref": "p/2", "path": "answer.txt"})["content"][0]["text"]
    assert call("get_task", {"ref": "nope"})["isError"] is True
    assert rpc("tools/list", Origin="https://evil.example").status_code == 403


def test_row_refs_round_trip():
    for config, split, i in (("default", "train", 0), ("code", "train", 6), ("a/b", "test", 12)):
        ref = rows.ref_of(config, split, i)
        assert rows.parse_ref(ref) == (config, split, i)
    assert rows.ref_of("default", "train", 3) == "train/3" and rows.ref_of("code", "train", 6) == "code/train/6"
    assert rows.subset_id("default", "train") == "train" and rows.parse_subset("code/train") == ("code", "train")
    with pytest.raises(LookupError):   # a 404, not a 500
        rows.parse_ref("train/x")
