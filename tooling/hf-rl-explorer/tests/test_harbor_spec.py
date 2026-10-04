"""Harbor's task.toml as Harbor 0.23 reads it (app/catalog.py), and what a task page shows of it (app/envs/harbor.py).
No network: each task is a folder written to a temp dir, indexed by catalog._row, and served through stubs.

What is checked: multi-step tasks read their instructions from steps/<name>/instruction.md; a grader in a container of
its own is said to run there, built from tests/Dockerfile; legacy `memory = "4G"` is converted; tasks that ask for a
GPU, a Windows container or no network for one phase can't run here (and say why); nested metadata and other tables are
kept; a model judge is a real model call, not a mention; healthchecks and MCP servers are shown; a step's solution and
the answers in its tests stay out.
"""

from __future__ import annotations

import json

import pytest

from app import catalog
from app.envs import contract as c
from app.envs.harbor import HarborAdapter

TB = """schema_version = "1.4"
[task]
name = "org/{name}"
[metadata]
difficulty = "easy"
[verifier]
timeout_sec = 300
[agent]
timeout_sec = 600
[environment]
docker_image = "python:3.12-slim"
"""
TEST_SH = "#!/bin/bash\npytest /tests/test_outputs.py && echo 1 > /logs/verifier/reward.txt\n"


def dump(x) -> str:
    return json.dumps(x, ensure_ascii=False)


def build(tmp_path, files: dict[str, str], name: str = "t1"):
    """A task folder on disk, indexed the way build_index indexes it: (row, the pack's texts, file sizes, picks)."""
    root = tmp_path / name
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    sizes = {rel: len(text.encode()) for rel, text in files.items()}
    picks = catalog._pick(sizes)
    row, kept = catalog._row(0, name, tmp_path, sizes, picks, {})
    return row, kept, sizes, picks


def view(monkeypatch, tmp_path, files: dict[str, str]):
    """The task page's view of a task folder, through HarborAdapter.task, with the index and the pack stubbed."""
    row, kept, sizes, picks = build(tmp_path, files)
    monkeypatch.setattr(catalog, "_task_row", lambda s, p, t=None: (row, {"sha": "abc", "info": {}}))
    monkeypatch.setattr(catalog, "_task_texts", lambda s, sha, p, t: (sizes, kept, set(picks)))
    monkeypatch.setattr(catalog, "_walked_root", lambda s, idx, p, t: ({}, []))
    monkeypatch.setattr(catalog, "collection", lambda s: None)
    adapter = HarborAdapter()
    return adapter.task(c.Env("dataset", "org/ds", {"sha": "abc"}, adapter), "t1"), row, kept


def section(v, sid: str) -> dict:
    return next(s for s in v["sections"] if s["id"] == sid)


def rows_of(sec: dict) -> dict:
    """A section's kv rows as {label: value} (the first of a repeated label)."""
    out: dict = {}
    for b in sec["blocks"]:
        if b["type"] == "kv":
            for k, v in b["rows"]:
                out.setdefault(k, v)
    return out


# ── multi-step tasks ─────────────────────────────────────────────────────────
MULTI = """version = "1.0"
multi_step_reward_strategy = "final"
[environment]
docker_image = "python:3.12"
[[steps]]
name = "write"
min_reward = 0.5
[steps.agent]
timeout_sec = 120
[[steps]]
name = "extend"
"""


def test_multi_step_instructions_come_from_each_steps_folder(tmp_path, monkeypatch):
    files = {"task.toml": MULTI, "tests/test.sh": TEST_SH,
             "steps/write/instruction.md": "Implement greet(name) in /app/greeting.py.\n\nReturn a greeting.",
             "steps/write/tests/test.sh": "echo 1 > /logs/verifier/reward.txt\n",
             "steps/write/workdir/setup.sh": "mkdir -p /app\n",
             "steps/extend/instruction.md": "Extend greet with an uppercase flag.",
             "steps/extend/solution/solve.sh": "echo SECRET-STEP-SOLUTION\n"}
    assert catalog._wanted("steps/write/instruction.md", 100) and catalog._wanted("steps/write/tests/test.sh", 100)
    v, row, kept = view(monkeypatch, tmp_path, files)
    assert row["steps"] == 2 and row["spec"]["step_names"] == ["write", "extend"]
    assert row["spec"]["multi"] == {"strategy": "final", "min_reward": {"write": 0.5}}
    assert row["title"].startswith("Implement greet(name)")   # no instruction.md: the first step's
    assert "steps/write/instruction.md" in kept and "steps/write/tests/test.sh" in kept
    assert row["run"] == "image" and row["solution"]   # a step's solution counts as the task's reference solution
    steps = section(v, "steps")
    text = dump(steps)
    assert "Implement greet(name)" in text and "Extend greet with an uppercase flag." in text
    assert "`tests/test.sh`" in text and "`setup.sh`" in text and "reward ≥ 0.5" in text
    assert section(v, "task")["title"] == "The task, in 2 steps"
    assert "the last step's reward" in dump(section(v, "config"))
    assert "SECRET-STEP-SOLUTION" not in dump(v)


def test_an_inline_step_instruction_is_flagged(tmp_path, monkeypatch):
    toml = 'version = "1.0"\n[environment]\ndocker_image = "x"\n[[steps]]\nname = "only"\ninstruction = "Do the thing."\n'
    v, row, _ = view(monkeypatch, tmp_path, {"task.toml": toml, "tests/test.sh": TEST_SH})
    text = dump(section(v, "steps"))
    assert "Do the thing." in text and "no longer reads it" in text


# ── a grader in a container of its own ───────────────────────────────────────
SEPARATE = TB + """[verifier.environment]
cpus = 2
memory = "4G"
"""


def test_separate_verifier_is_said_to_run_in_its_own_container(tmp_path, monkeypatch):
    toml = SEPARATE.format(name="sep").replace("[verifier]\n", '[verifier]\nenvironment_mode = "separate"\n')
    files = {"task.toml": toml, "instruction.md": "Do it.", "tests/test.sh": TEST_SH,
             "tests/Dockerfile": "FROM python:3.12-slim\nRUN pip install pytest\nCOPY . /tests\n"}
    v, row, _ = view(monkeypatch, tmp_path, files)
    sep = row["verifier"]["separate"]
    assert sep["own"] and sep["dockerfile"] and sep["ok"] and not sep.get("image")
    runs = rows_of(section(v, "grading"))["Runs"]
    assert "separate container" in runs and "tests/Dockerfile" in runs and "same sandbox" not in runs
    assert "4 GB memory" in rows_of(section(v, "config"))["Grader's container"]
    assert ["Grader", "in its own container"] in v["glance"]
    assert catalog.runnable(row, {})["ok"]


def test_separate_verifier_without_an_image_or_dockerfile_cant_run(tmp_path):
    toml = TB.format(name="sep2").replace('docker_image = "python:3.12-slim"\n', "") + '[verifier.environment]\ncpus = 1\n'
    row, *_ = build(tmp_path, {"task.toml": toml, "instruction.md": "x", "tests/test.sh": TEST_SH,
                               "environment/Dockerfile": "FROM python:3.12\n"})
    r = catalog.runnable(row, {"environment/Dockerfile": "FROM python:3.12\n"})
    assert not r["ok"] and "container of its own" in r["why"]


# ── what Harbor converts, and what an HF Sandbox can't do ────────────────────
def test_legacy_memory_and_storage_strings_are_converted(tmp_path, monkeypatch):
    toml = TB.format(name="m").replace("[environment]\n", '[environment]\ncpus = 1\nmemory = "4G"\nstorage = "512M"\n')
    v, row, _ = view(monkeypatch, tmp_path, {"task.toml": toml, "instruction.md": "x", "tests/test.sh": TEST_SH})
    assert row["env"]["memory_mb"] == 4096 and row["env"]["storage_mb"] == 512
    assert rows_of(section(v, "config"))["Resources"] == "1 CPU · 4 GB memory · 512 MB storage"


@pytest.mark.parametrize("patch,why", [
    ('[environment]\ngpus = 1\ngpu_types = ["L4"]\n', "GPU"),
    ('[environment]\nos = "Windows"\n', "Windows"),
    ('[environment]\ntpu = { type = "v6e", topology = "2x4" }\n', "TPU"),
])
def test_what_hf_sandboxes_cant_give_is_not_runnable(tmp_path, patch, why):
    toml = TB.format(name="g").replace("[environment]\n", patch)
    row, *_ = build(tmp_path, {"task.toml": toml, "instruction.md": "x", "tests/test.sh": TEST_SH, "tests/test.bat": "@echo off\n"})
    r = catalog.runnable(row, {})
    assert row["run"] is None and not r["ok"] and why in r["why"]


@pytest.mark.parametrize("section_,label", [("agent", "agent"), ("verifier", "grader")])
def test_a_phase_without_internet_is_not_runnable(tmp_path, monkeypatch, section_, label):
    toml = TB.format(name="n").replace(f"[{section_}]\n", f'[{section_}]\nnetwork_mode = "no-network"\n')
    v, row, _ = view(monkeypatch, tmp_path, {"task.toml": toml, "instruction.md": "x", "tests/test.sh": TEST_SH})
    assert row["env"]["network"] is None and row["env"]["phases"] == {section_: "no-network"}
    r = catalog.runnable(row, {})
    assert not r["ok"] and f"the {label} asks for no internet access" in r["why"]
    assert f"{label}: no internet" in rows_of(section(v, "config"))["Network"]
    assert not v["_runnable"]["ok"]


def test_an_allowlist_lists_its_hosts(tmp_path, monkeypatch):
    toml = TB.format(name="a").replace("[environment]\n", '[environment]\nnetwork_mode = "allowlist"\nallowed_hosts = ["pypi.org", "*.github.com"]\n')
    v, row, _ = view(monkeypatch, tmp_path, {"task.toml": toml, "instruction.md": "x", "tests/test.sh": TEST_SH})
    assert row["env"]["hosts"] == ["pypi.org", "*.github.com"] and not catalog.runnable(row, {})["ok"]
    assert "`pypi.org`" in rows_of(section(v, "config"))["Allowed hosts"]


def test_a_task_toml_harbor_rejects_is_read_as_written_and_cant_run(tmp_path):
    toml = '[task]\nname = "no-org-slash"\n[environment]\ndocker_image = "x"\nmemory = "2G"\n'
    row, *_ = build(tmp_path, {"task.toml": toml, "instruction.md": "x", "tests/test.sh": TEST_SH})
    assert "task.name" in row["spec"]["harbor_error"] and row["env"]["memory_mb"] == 2048
    r = catalog.runnable(row, {})
    assert not r["ok"] and "Harbor itself rejects" in r["why"]


# ── metadata ─────────────────────────────────────────────────────────────────
NESTED = TB + """[metadata.repo2env]
recipe = "tasksmith"
fail_to_pass_count = 14
gold_patch = "diff --git SECRET-PATCH"
[metadata.repo2env.evaluation]
status = "verified"
[metadata.expected_result]
value = "SECRET-NESTED"
[scoring]
difficulty_label = "hard"
"""


def test_nested_metadata_and_other_tables_are_kept_answers_left_out(tmp_path, monkeypatch):
    v, row, kept = view(monkeypatch, tmp_path, {"task.toml": NESTED.format(name="nm"), "instruction.md": "x", "tests/test.sh": TEST_SH})
    assert row["meta"]["repo2env.recipe"] == "tasksmith" and row["meta"]["repo2env.evaluation.status"] == "verified"
    assert row["meta"]["scoring.difficulty_label"] == "hard"
    assert "repo2env.gold_patch" in row["withheld"] and "expected_result" in row["withheld"]
    assert not any("gold" in k or "expected" in k for k in row["meta"])
    meta = dump(section(v, "metadata"))
    assert "tasksmith" in meta and "verified" in meta and "hard" in meta
    assert "SECRET" not in dump(v) and "SECRET" not in kept["task.toml"]   # nor in the raw task.toml shown


def test_masked_toml_hides_multi_line_values_and_answer_tables():
    out = catalog._masked_toml('[metadata]\nexpected = [\n  "a1",\n  "a2",\n]\ntags = [\n  "x",\n]\n[metadata.gold]\nv = 42\n'
                               '[solution.env]\nFOO = "${FOO}"\n')
    assert "a1" not in out and "42" not in out and '"x"' in out and 'FOO = "${FOO}"' in out
    import tomllib

    assert tomllib.loads(out)["metadata"]["gold"] == {"v": "‹withheld›"}


# ── a model judge is a model call ────────────────────────────────────────────
GRADER_CALLS = """import os
from openai import OpenAI

client = OpenAI(base_url=os.environ["JUDGE_URL"])
resp = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "grade"}])
"""
GRADER_MENTIONS = """# compare with an OpenAI-style judge later, maybe
def grade(answer: str, judge: bool = False) -> float:
    llm_judge = None
    return float(answer.strip() == open('/app/out.txt').read().strip())
"""


@pytest.mark.parametrize("grader,judge", [(GRADER_CALLS, True), (GRADER_MENTIONS, False)])
def test_judge_is_a_real_model_call_not_a_mention(tmp_path, grader, judge):
    row, *_ = build(tmp_path, {"task.toml": TB.format(name="j"), "instruction.md": "x",
                               "tests/test.sh": "python3 /tests/grade.py > /logs/verifier/reward.txt\n", "tests/grade.py": grader})
    assert row["verifier"]["judge"] is judge


def test_judge_from_the_graders_model_key():
    toml = TB.format(name="k") + '[verifier.env]\nOPENAI_API_KEY = "${OPENAI_API_KEY}"\n'
    assert catalog.verifier_kind("", "", toml_text=toml, sources={})["judge"]
    assert not catalog.verifier_kind("", "", ["SOME_API_KEY"], toml_text=TB.format(name="k"), sources={})["judge"]   # a key's name isn't a judge
    js = "import OpenAI from 'openai';\nconst r = await client.responses.create({model: 'x', input: 'y'});\n"
    assert catalog._calls_model(js) and not catalog._calls_model("import openai  # unused\nprint('judge')\n")


# ── what the agent gets besides the instruction ──────────────────────────────
def test_healthcheck_mcp_servers_and_settings_are_shown(tmp_path, monkeypatch):
    toml = TB.format(name="h").replace("[agent]\n", '[agent]\nuser = "learner"\n') + """workdir = "/testbed"
skills_dir = "/skills"
[environment.healthcheck]
command = "test -f /ready"
interval_sec = 2
retries = 5
[[environment.mcp_servers]]
name = "docs"
transport = "streamable-http"
url = "http://docs:8000/mcp"
[[verifier.collect]]
command = "pg_dump > /logs/db.sql"
service = "db"
[[artifacts]]
source = "/app/out"
exclude = ["*.pyc"]
"""
    v, row, _ = view(monkeypatch, tmp_path, {"task.toml": toml, "instruction.md": "x", "tests/test.sh": TEST_SH, "trajectory.json": "{}"})
    assert row["env"]["healthcheck"] and row["env"]["mcp"] == ["docs"] and row["env"]["workdir"] == "/testbed"
    assert row["verifier"]["collect"] == 1 and row["spec"]["agent_user"] == "learner" and row["spec"]["artifacts"] == ["/app/out"]
    cfg = rows_of(section(v, "config"))
    assert "`test -f /ready`" in cfg["Healthcheck"] and "every 2 s" in cfg["Healthcheck"] and "5 failed tries" in cfg["Healthcheck"]
    assert cfg["MCP server"].startswith("`docs` · streamable-http") and "`http://docs:8000/mcp`" in cfg["MCP server"]
    assert cfg["Users"] == "agent as `learner`" and cfg["Working directory"] == "`/testbed`" and "`/skills`" in cfg["Skills"]
    assert "`pg_dump > /logs/db.sql` in `db`" in cfg["Collect hooks"] and "`/app/out`" in cfg["Artifacts"]
    assert cfg["Name"] == "`org/h`" and "Prior context" in cfg
    assert any(b["type"] == "disclose" and b["label"].startswith("task.toml") for b in section(v, "config")["blocks"])
    assert "trajectory.json" in dump(section(v, "task"))


def test_summary_facets_name_what_sets_tasks_apart(tmp_path, monkeypatch):
    plain, *_ = build(tmp_path, {"task.toml": TB.format(name="p"), "instruction.md": "x", "tests/test.sh": TEST_SH}, "p")
    multi, *_ = build(tmp_path, {"task.toml": MULTI, "tests/test.sh": TEST_SH, "steps/write/instruction.md": "a",
                                 "steps/extend/instruction.md": "b"}, "m")
    rows = [plain, multi]
    catalog._group(rows)
    for r in rows:
        r.pop("_paras", None)
        r["brief"] = ""
    idx = {"tasks": rows, "summary": catalog.summarize(rows), "info": {}, "sha": "abc"}
    monkeypatch.setattr(catalog, "index_status", lambda spec, token=None: {"state": "done", "index": idx})
    adapter = HarborAdapter()
    env = c.Env("dataset", "org/ds", {"sha": "abc"}, adapter)
    assert "setup" in [f["key"] for f in adapter.summary(env)["facets"]]
    cards = {x["ref"]: x for x in adapter.tasks(env, everything=True)["cards"]}
    assert cards["m"]["facets"]["setup"] == ["Multi-step"] and cards["p"]["facets"]["setup"] == []
    assert "2 steps" in cards["m"]["chips"]


# ── answers stay out of steps ────────────────────────────────────────────────
@pytest.mark.parametrize("rel,text,withheld", [
    ("steps/s1/solution/solve.sh", None, True),
    ("steps/s1/solution/notes.md", "anything", True),
    ("steps/s1/tests/test.sh", "pytest", False),
    ("steps/s1/tests/grade.py", "import json", False),
    ("steps/s1/tests/expected.json", None, True),
    ("steps/s1/tests/data.csv", None, True),
    ("steps/s1/tests/check.py", "GOLD = {'gold_answer': 42}", True),
    ("steps/s1/instruction.md", "Write it.", False),
])
def test_step_answers_are_withheld(rel, text, withheld):
    assert catalog._withheld(rel, text) is withheld


def test_a_step_solution_is_never_served(tmp_path, monkeypatch):
    files = {"task.toml": MULTI, "tests/test.sh": TEST_SH, "steps/write/instruction.md": "a", "steps/extend/instruction.md": "b",
             "steps/write/tests/test.sh": 'EXPECTED="SECRET-42"\n[ "$(cat /app/x)" = "$EXPECTED" ] && echo 1 > /logs/verifier/reward.txt\n',
             "steps/write/solution/solve.sh": "echo SECRET-SOLVE\n"}
    row, kept, sizes, picks = build(tmp_path, files)
    assert "steps/write/solution/solve.sh" not in kept and "SECRET-42" not in kept["steps/write/tests/test.sh"]
    monkeypatch.setattr(catalog, "_task_row", lambda s, p, t=None: (row, {"sha": "abc", "info": {}}))
    monkeypatch.setattr(catalog, "_task_texts", lambda s, sha, p, t: (sizes, kept, set(picks)))
    monkeypatch.setattr(catalog, "_walked_root", lambda s, idx, p, t: ({}, []))
    monkeypatch.setattr(catalog, "_fetch", lambda *a, **k: pytest.fail("a withheld file was fetched"))
    got = catalog.task_file("org/ds", "t1", "steps/write/solution/solve.sh")
    assert got.get("withheld") and "text" not in got
    tree = {f["path"]: f["withheld"] for f in catalog.task("org/ds", "t1")["tree"]}
    assert tree["steps/write/solution/solve.sh"] is True and tree["steps/write/instruction.md"] is False
