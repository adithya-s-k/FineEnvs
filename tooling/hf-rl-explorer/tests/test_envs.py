"""Rows datasets (app/envs) and big Harbor datasets walked folder by folder (no network: rows and the Hub are stubbed).

What is checked: each processor claims the datasets it should; answers never come back (named columns, nested
keys, a packed task's solution and answer files, MiMo's patches); archives can't escape their folder or blow up;
filters can't inject; the direct reader pages JSON Lines right; the walker finds tasks without listing whole repos,
and a task's unlisted folders can't be used to reach files outside it.
"""

from __future__ import annotations

import base64
import io
import json
import tarfile
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import catalog, envs, main
from app.envs import base, direct, processors

client = TestClient(main.app, base_url="https://testserver")


def ds(columns, rows, tags=(), spec="org/ds"):
    return base.Dataset(spec=spec, tags=list(tags), card={}, splits=[{"config": "default", "split": "train"}],
                        features=[{"name": c, "type": {}} for c in columns], sample=rows)


def tarball(files: dict[str, bytes], prefix="task_0/") -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for name, data in files.items():
            info = tarfile.TarInfo(prefix + name if not name.startswith(("..", "/")) else name)
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))
    return base64.b64encode(buf.getvalue()).decode()


TASK = {
    "task.toml": b'[metadata]\ntitle = "Count the words"\ndifficulty = "easy"\ngold_answer = "42"\n[environment]\ndocker_image = "python:3.12"\n',
    "instruction.md": b"# Count the words\n\nCount the words in /app/a.txt.\n",
    "tests/test.sh": b"#!/bin/bash\nexpected=42\necho 1 > /logs/verifier/reward.txt\n",
    "tests/expected_output.json": b'{"answer": 42}',
    "solution/solve.sh": b"echo 42\n",
    "task_manifest.json": b'{"original_bash": "wc -w /app/a.txt"}',
    "environment/Dockerfile": b"FROM python:3.12\n",
    "../escape.txt": b"nope",
}


# ── which processor reads what ───────────────────────────────────────────────
def test_processors_claim_their_formats():
    pick = lambda d: envs.pick(d).id  # noqa: E731
    assert pick(ds(["path", "task_binary"], [{"path": "t", "task_binary": "x"}], tags=["library:harbor"])) == "harbor-packed"
    assert pick(ds(["data_source", "prompt", "ability", "reward_model", "extra_info", "agent_name"], [{}], spec="XiaomiMiMo/MiMo-V2.6-RL-oss")) == "mimo"
    assert pick(ds(["id", "responses_create_params", "kwargs"], [{}])) == "nemo-gym"
    assert pick(ds(["data_source", "prompt", "reward_model"], [{}])) == "verl"
    conv = [{"role": "user", "content": "do it"}, {"role": "assistant", "content": "done"}]
    assert pick(ds(["conversations", "agent", "result"], [{"conversations": conv, "agent": "a", "result": "pass"}])) == "traces"
    assert pick(ds(["question", "answer"], [{"question": "q", "answer": "a"}], tags=["library:verifiers"])) == "verifiers"
    assert pick(ds(["question", "answer"], [{"question": "q", "answer": "a"}])) == "generic"


def test_generic_roles_and_answers():
    rows = [{"task_id": f"t{i}", "instruction": f"You are an agent.\nWork carefully.\n\nQuestion {i}: what is {i}?", "question": f"what is {i}?",
             "answer": str(i), "reward_mode": "exact", "atol": 0.0, "files": ["a.csv"], "difficulty": "easy"} for i in range(5)]
    d = ds(list(rows[0]), rows)
    p = envs.pick(d)
    roles = p.roles(d)
    assert roles["task"] == "instruction" and roles["id"] == "task_id" and roles["answer"] == ["answer"]
    assert roles["title"] == "question"                       # the instructions share a preamble: titled by the question
    assert set(roles["grading"]) == {"reward_mode", "atol"} and roles["environment"] == ["files"]
    view = p.view(rows[3], 3, d, roles)
    assert '"3"' not in json.dumps([s["body"] for s in view["sections"] if s["id"] != "task"]) and view["withheld"] == ["answer"]
    card = p.card(rows[3], 3, roles)
    assert card["title"] == "what is 3?" and card["id"] == "t3"


def test_nested_answers_are_withheld_everywhere():
    clean, gone = base.withhold({"meta": {"oracle_commands": ["rm"], "level": 1}, "items": [{"gold": 1, "x": 2}], "prompt": "p"})
    assert clean == {"meta": {"level": 1}, "items": [{"x": 2}], "prompt": "p"}
    assert set(gone) == {"meta.oracle_commands", "items.gold"}
    clean, _ = base.withhold({"label": "tw_1", "x": 1}, keep=("label",))   # a label that is the row's id stays
    assert clean["label"] == "tw_1"


def test_retitle_drops_a_shared_preamble():
    cards = [{"title": "Setup", "snippet": "", "_text": f"# Setup\nrun setup.sh\n\n# Goal\nTask number {i} is hard\nmore"} for i in range(4)]
    base.retitle(cards)
    assert [c["title"] for c in cards] == [f"Task number {i} is hard" for i in range(4)]
    assert all("_text" not in c for c in cards)


# ── Harbor tasks packed into rows ────────────────────────────────────────────
def test_packed_tasks_unpack_without_escaping_and_withhold_answers():
    files = processors.unpack(tarball(TASK))
    assert "task.toml" in files and "tests/test.sh" in files        # the folder around them stripped
    assert not any(".." in n for n in files)
    p, d = processors.PackedHarbor(), ds(["path", "task_binary"], [], tags=["library:harbor"])
    row = {"path": "task_0", "task_binary": tarball(TASK)}
    view = p.view(row, 0, d, p.roles(d))
    text = json.dumps(view)
    assert "gold_answer" not in json.dumps([s for s in view["sections"] if s["id"] == "metadata"])
    assert "solution/solve.sh" in view["withheld"] and "tests/expected_output.json" in view["withheld"] and "task_manifest.json" in view["withheld"]
    assert p.file(row, d, "task_manifest.json")["withheld"] is True
    assert view["title"] == "Count the words" and "Count the words in /app/a.txt" in text
    assert p.file(row, d, "solution/solve.sh") == {"path": "solution/solve.sh", "size": 8, "withheld": True}
    assert p.file(row, d, "tests/expected_output.json")["withheld"] is True
    assert "42" not in p.file(row, d, "task.toml")["text"]
    assert "42" not in p.file(row, d, "tests/test.sh")["text"]           # expected=42 masked
    assert p.file(row, d, "../escape.txt")["error"]
    card = p.card(row, 0, {})
    assert card["title"] == "Count the words" and "easy" in card["chips"]


def test_data_files_naming_the_answer_are_withheld():
    assert catalog._withheld("task_manifest.json", json.dumps({"original_nl": "do x", "mutated_bash": "grep -c x f", "captured_output_path": "tests/o"}))
    assert catalog._withheld("meta.yaml", "task: x\nexpected_output: 42\n")
    assert catalog._withheld("data.jsonl", '{"q": 1, "answer": 2}\n')
    assert not catalog._withheld("config.json", json.dumps({"captured_output_path": "x", "timeout": 3, "name": "t"}))
    assert not catalog._withheld("instruction.md", "the expected_output is printed")      # only data files by key


def test_bad_archives_are_empty_not_errors():
    assert processors.unpack("not base64 !!!") == {}
    assert processors.unpack(base64.b64encode(b"\x00" * 64).decode()) == {}
    assert processors.unpack(12) == {}
    big = base64.b64encode(b"x" * (processors.MAX_TAR + 10)).decode()
    assert processors.unpack(big) == {}


# ── MiMo's raw release ───────────────────────────────────────────────────────
def test_mimo_rows_withhold_patches_and_link_their_harbor_twin(monkeypatch):
    inst = {"instance_id": "Repo_Task-001", "docker_image": "img:1", "problem_statement": "fix it", "test_patch": "diff --git a/t b/t",
            "patch": "diff --git a/src b/src SECRET", "poc": "crash bytes"}
    row = {"data_source": "x", "ability": "swe", "agent_name": "a", "prompt": [{"role": "user", "content": "fix it"}],
           "reward_model": {"style": "rule", "ground_truth": "SECRET2"}, "extra_info": {"instance_id": "Repo_Task-001", "instance_json": json.dumps(inst)}}
    monkeypatch.setattr(catalog, "_read_index", lambda spec, sha=None: {"tasks": [{"path": "tasks/repo_task-001"}]} if spec.endswith("-code") else None)
    p = processors.MiMo()
    d = ds(list(row), [row], spec="XiaomiMiMo/MiMo-V2.6-RL-oss")
    roles = {**p.roles(d), "_config": "code"}
    view = p.view(row, 0, d, roles)
    text = json.dumps(view)
    assert "SECRET" not in text and "crash bytes" not in text
    assert view["run"]["dataset"] == "FineEnvs/MiMo-V2.6-RL-harbor-code" and view["run"]["path"] == "tasks/repo_task-001"
    assert any(s["id"] == "tests" for s in view["sections"])          # hidden tests are shown: they're how it's graded


# ── filters and the direct reader ────────────────────────────────────────────
def test_filters_are_quoted_and_limited_to_known_columns():
    w = envs._where({"difficulty": "x' OR '1'='1", "nope": "1", 'a"b': "v"}, ["difficulty", 'a"b'])
    assert w == "\"difficulty\"='x'' OR ''1''=''1' AND \"a\"\"b\"='v'"


def test_env_api_rejects_bad_filter_json():
    assert client.get("/api/env/org/ds/tasks?f=nope").status_code == 400
    assert client.get("/api/env/org/ds/tasks?f=[1]").status_code == 400
    assert client.get("/api/env/org/ds/tasks?f=" + json.dumps({f"k{i}": ["v"] for i in range(20)})).status_code == 400


def test_jsonl_rows_page_by_line(tmp_path, monkeypatch):
    p = tmp_path / "data.jsonl"
    p.write_text("\n".join(json.dumps({"id": i, "prompt": f"p{i}"}) for i in range(10)) + "\n\n")
    monkeypatch.setattr(direct, "_background", lambda *a, **k: {"state": "done", "local": str(p)})
    monkeypatch.setattr(direct.Source, "_size", lambda self, path: 100)
    src = direct.Source("org/ds", {"sha": "abc"}, ["data.jsonl"], None)
    assert src.total() == 10
    assert [i for i, _ in src.read(8, 5)] == [8, 9] and src.read(8, 5)[0][1]["prompt"] == "p8"
    hits, cut = direct.scan(src, lambda r: r["id"] % 3 == 0)
    assert [i for i, _ in hits] == [0, 3, 6, 9] and not cut


def test_a_file_still_coming_serves_its_first_rows(monkeypatch):
    monkeypatch.setattr(direct, "_background", lambda *a, **k: {"state": "running"})
    monkeypatch.setattr(direct, "_head", lambda *a, **k: ([{"i": n} for n in range(30)], False))
    monkeypatch.setattr(direct.Source, "_size", lambda self, path: 10**9)
    src = direct.Source("org/ds", {"sha": "abc"}, ["big.jsonl"], None)
    assert src.total() is None
    assert [r["i"] for _, r in src.read(0, 24)] == list(range(24))
    with pytest.raises(direct.DirectError) as e:
        src.read(100, 24)
    assert e.value.status == 409


def test_data_files_follow_the_card_configs(monkeypatch):
    files = [("data/train-00000.parquet", 10), ("data/test-00000.parquet", 10), ("extra/notes.json", 5)]
    monkeypatch.setattr(catalog, "_cached", lambda key, ttl, fn: files if key[0] == "data-files" else {
        "configs": [{"config_name": "default", "data_files": [{"split": "train", "path": "data/train-*"}, {"split": "test", "path": "data/test-*"}]}]})
    got = direct.data_files("org/ds", {"sha": "abc"}, None)
    assert got == {"default": {"train": ["data/train-00000.parquet"], "test": ["data/test-00000.parquet"]}}


# ── big Harbor datasets, walked folder by folder ─────────────────────────────
class FakeHub:
    """A repo of 3 splits x 40 task folders, each with a big app folder that must never be listed."""

    def __init__(self):
        self.listed, self.info_calls = [], 0
        self.dirs = {"": ["tasks"], "tasks": ["tasks/a", "tasks/b", "tasks/c"]}
        self.files = {}
        for s in ("a", "b", "c"):
            self.dirs[f"tasks/{s}"] = [f"tasks/{s}/t{i}" for i in range(40)]
            for i in range(40):
                d = f"tasks/{s}/t{i}"
                self.files.update({f"{d}/task.toml": 10, f"{d}/instruction.md": 20, f"{d}/tests/test.sh": 5})
                self.dirs[d] = [f"{d}/app", f"{d}/tests"]

    def list_repo_tree(self, spec, repo_type=None, revision=None, path_in_repo=None, recursive=False):
        from huggingface_hub.hf_api import RepoFile, RepoFolder

        p = path_in_repo or ""
        self.listed.append(p)
        assert not p.endswith("/app"), "a task's app folder was listed"
        out = [RepoFolder(path=d, oid="x", tree_id="x") for d in self.dirs.get(p, [])]
        out += [RepoFile(path=f, size=n, oid="x") for f, n in self.files.items() if f.rsplit("/", 1)[0] == p]
        return iter(out)

    def get_paths_info(self, spec, paths, repo_type=None, revision=None):
        from huggingface_hub.hf_api import RepoFile

        self.info_calls += 1
        assert len(paths) <= 500
        return [RepoFile(path=p, size=self.files[p], oid="x") for p in paths if p in self.files]


def test_walker_finds_every_task_with_few_calls(monkeypatch):
    hub = FakeHub()
    monkeypatch.setattr(catalog, "_api", lambda token=None: hub)
    job = {}
    out = catalog._walk("org/big", "abc", None, job)
    assert len(out["tasks"]) == 120 and out["tasks"][0] == "tasks/a/t0"
    assert out["files"]["tasks/a/t0"] == {"task.toml": 10, "instruction.md": 20, "tests/test.sh": 5}
    assert len(hub.listed) <= 6                        # root, tasks/, three splits (+0): never a task folder
    assert hub.info_calls <= 6 and job["tasks"] == 120


def test_a_task_folder_listing_stays_inside_the_task(monkeypatch):
    monkeypatch.setattr(catalog, "_task_row", lambda spec, path, token=None: ({"path": path}, {"sha": "abc", "info": {}}))
    for bad in ("../other", "..", "", "/", "a/../../b"):
        with pytest.raises(ValueError):
            catalog.task_folder("org/ds", "tasks/t1", bad)


def test_listing_falls_back_to_walking_when_too_slow(monkeypatch):
    hub = FakeHub()

    def slow(*a, recursive=False, **k):
        if recursive:
            for n in range(10_000):
                yield SimpleNamespace(path=f"f{n}")
        else:
            yield from hub.list_repo_tree(*a, recursive=False, **k)

    monkeypatch.setattr(catalog, "_api", lambda token=None: SimpleNamespace(list_repo_tree=slow, get_paths_info=hub.get_paths_info))
    monkeypatch.setattr(catalog, "LIST_BUDGET", -1)
    files, walked = catalog._listing("org/big", "abc", None, {})
    assert files == {} and walked and len(walked["tasks"]) == 120


# ── when the viewer can't filter or search ───────────────────────────────────
def test_filter_and_search_fall_back_to_reading_a_small_split(monkeypatch):
    rows = [(i, {"id": f"t{i}", "q": f"question {i}", "tier": "hard" if i % 3 == 0 else "easy", "answer": "zebra" if i == 4 else "x"}) for i in range(50)]

    def no(*a, **k):
        raise envs.ViewerError("the dataset index is loading")

    monkeypatch.setattr(envs.viewer, "filter_rows", no)
    monkeypatch.setattr(envs.viewer, "search", no)
    monkeypatch.setattr(envs.rows, "_all_rows", lambda spec, config, split, tok: rows)
    envs._broken.clear()
    v = envs.Viewer("org/ds", None)
    page, _ = v.filter("default", "train", {"tier": "hard"}, ["tier"], 0)
    assert page["total"] == 17 and all(r["tier"] == "hard" for _, r, _ in page["rows"])
    page, _ = v.search("default", "train", "question 4", 0)
    assert {r["id"] for _, r, _ in page["rows"]} >= {"t4", "t40", "t41"}
    page, _ = v.search("default", "train", "zebra", 0)
    assert page["total"] == 0                                    # searching an answer finds nothing
    assert ("org/ds", "filter") in envs._broken                  # next time straight to the fallback


def test_one_slow_viewer_answer_does_not_send_a_dataset_to_its_files(monkeypatch):
    calls = {"n": 0}

    def flaky(spec, tok=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise envs.ViewerError("timeout", 502)
        return [{"config": "default", "split": "train"}]

    monkeypatch.setattr(catalog, "info", lambda spec, token=None: {"sha": "abc", "restricted": False})
    monkeypatch.setattr(envs.viewer, "splits", flaky)
    monkeypatch.setattr(envs.viewer, "rows", lambda *a, **k: {"rows": [], "total": 0, "features": []})
    envs._choice.clear()
    _, be = envs._backend("org/ds", None)
    assert be.kind == "viewer" and calls["n"] == 2


def test_harbor_layouts_are_found_whatever_the_tags(monkeypatch):
    hub = FakeHub()
    monkeypatch.setattr(catalog, "_api", lambda token=None: hub)
    catalog._memo.clear()
    assert catalog.looks_harbor("org/big", "abc") is True
    hub.files = {}
    catalog._memo.clear()
    assert catalog.looks_harbor("org/big", "abc") is False
