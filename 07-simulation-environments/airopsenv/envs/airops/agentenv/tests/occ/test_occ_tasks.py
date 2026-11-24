"""The ops control center as a bundle: each day's steps (the desks' roles and tools, one trigger per watch with
advisories on the end_turn that opens it, the agents, the clock, the plays at once, end-day after both, the grade), the
rules each desk reads, the pack the env loads, and the verifier on a day's data/get."""

import json
import types

import pytest
from agent_env.task_step.task_steps.verifiers.scoring import ScoreAggregator, aggregate_score
from airops_core.occ.pack import Pack
from occ_helpers import play, scripted, storm, write_pack
from paths import BUNDLES, OCC_PACK

from airops_agentenv.occ.tasks import RULES, VERIFIERS, generate, steps

pytestmark = pytest.mark.anyio


def test_a_joint_day_deploys_two_desks_in_their_roles_and_triggers_each_watch_with_news():
    task = storm().task
    s = steps(task, "joint", cap=3.0)
    assert [x["id"] for x in s] == ["deploy", "load-task", "crew-only", "crew-tools", "ops-only", "ops-tools",
                                    "watches", "deploy-crew-desk", "deploy-ops-desk", "clock", "play-crew-desk",
                                    "play-ops-desk", "end-day", "grade"]
    by = {x["id"]: x for x in s}
    assert by["load-task"]["directives"][0]["args"] == {"task_id": task.task_id, "mode": "joint"}
    assert by["crew-only"]["tools"] == ["*"] and "crew_confirm" in by["crew-tools"]["tools"] and \
        "ops_confirm" not in by["crew-tools"]["tools"] and "wait_turn" in by["ops-tools"]["tools"]
    assert by["watches"]["watch_roles"] == ["crew", "ops"] and by["watches"]["triggers"] == [{
        "id": "watch-2", "when": {"type": "action", "tool": "end_turn", "where": {"result.opened": {"equals": 2}}},
        "barrier": {"at": "provoking_call", "timeout_seconds": 60},
        "actions": [{"type": "tool", "tool": "airops_notice", "args": {"news_id": "adv-b"}}]}]
    assert by["deploy-ops-desk"] | {"env_vars": None} == {
        "id": "deploy-ops-desk", "type": "deploy_agent", "a2a_agent_id": "airops-desk", "agent_name": "ops-desk",
        "role": "ops", "env_ids": ["airops-occ"], "env_vars": None}
    assert by["deploy-ops-desk"]["env_vars"] == {"AIROPS_MAX_COST_USD": "3"}
    play = by["play-ops-desk"]
    assert (play["max_turns"], play["depends_on"], play["fail_task_on_error"], play["effort"]) == (
        6 * 25 + 10, ["clock"], False, "high")
    assert by["end-day"]["depends_on"] == ["play-crew-desk", "play-ops-desk"]


def test_one_desk_runs_both_without_the_channel_and_no_desk_is_the_crew_system_alone():
    task = storm().task
    by = {x["id"]: x for x in steps(task, "one")}
    tools = by["both-tools"]["tools"]
    assert "ops_confirm" in tools and "crew_confirm" in tools and not any(t.endswith(("request", "reply"))
                                                                         for t in tools)
    assert by["play-occ-desk"]["max_turns"] == 8 * 25 + 10 and by["deploy-occ-desk"]["role"] == "both"
    idle = steps(task, "idle")
    assert [x["id"] for x in idle] == ["deploy", "load-task", "watches", "clock", "end-day", "grade"]
    assert all(len(text) <= 5000 for text in RULES.values())


def test_the_bundle_and_the_pack(tmp_path):
    w = storm()
    pack = write_pack(tmp_path / "pack", w)
    names = generate(tmp_path / "bundle", split="eval", pack=pack, noplay=True)
    assert names == [f"{w.task.task_id}-joint", f"{w.task.task_id}-one", "wiring-noplay"]
    readme = (tmp_path / "bundle" / "README.md").read_text()
    assert "v2" not in readme and (tmp_path / "bundle" / "artifacts" / "airops-occ-verifier" / "verify.py").is_file()
    again = Pack(pack).world(w.task.task_id)
    assert again.task == w.task and again.units == w.units and again.trips == w.trips and again.partner == {0: 1, 1: 0}
    assert again.local("JFK", 300) == 300


def load_verifier() -> types.ModuleType:
    path = VERIFIERS / "airops-occ-verifier" / "verify.py"
    module = types.ModuleType("occ_verifier")
    exec(compile(path.read_text(), str(path), "exec"), module.__dict__)
    return module


async def test_the_verifier_scores_an_audited_day_and_refuses_one_that_fails(monkeypatch):
    verifier = load_verifier()
    data = json.loads(json.dumps(play(storm(), policies=scripted("crew", "ops")).data(), default=str))

    async def get_data(url):
        return types.SimpleNamespace(parts=[types.SimpleNamespace(data=data)])
    monkeypatch.setattr(verifier.client, "get_data", get_data)
    results = await verifier.verify("http://env/mcp")
    assert results[0]["score"] == data["reward"] and aggregate_score(results, ScoreAggregator.WEIGHTED_AVERAGE) == \
        data["reward"]
    data["audit"] = {"ok": False, "problems": {"A7": ["crew: crash"]}}
    with pytest.raises(RuntimeError, match="can't be scored"):
        await verifier.verify("http://env/mcp")
    data["audit"] = {"ok": False, "problems": {"A5": ["flown without a legal crew: [3]"], "A7": ["crew: crash"]}}
    with pytest.raises(RuntimeError, match="can't be scored"):
        await verifier.verify("http://env/mcp")


def test_generate_writes_the_task_files_the_published_runs_played(tmp_path):
    for mode, cap in (("joint", 5.0), ("one", 8.0)):
        generate(tmp_path, split="all", modes=(mode,), pack=OCC_PACK, cap=cap)
    written = {p.name: p.read_bytes() for p in sorted((tmp_path / "tasks").iterdir())}
    assert written == {p.name: p.read_bytes() for p in sorted((BUNDLES / "airops-occ" / "tasks").iterdir())}
    assert (tmp_path / "artifacts/airops-occ-verifier/verify.py").read_bytes() == (
        VERIFIERS / "airops-occ-verifier" / "verify.py").read_bytes()
