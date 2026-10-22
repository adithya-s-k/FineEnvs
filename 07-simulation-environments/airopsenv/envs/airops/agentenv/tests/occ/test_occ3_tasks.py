"""The ops control center with passengers as a bundle: each day's steps (three desks in their roles, the customer desk
with its three tools, or one agent at all three, each capped; no desk at all), the rules each desk reads, the pack the
env loads, the verifier on a day's data/get (the reward under P scored, reward_L beside it at no weight), and the task
files the board played."""

import json
import types

import pytest
from agent_env.task_step.task_steps.verifiers.scoring import ScoreAggregator, aggregate_score
from airops_core.occ.day3 import OccDay3
from airops_core.occ.pack import Pack3
from occ_helpers import play, scripted, storm3, write_pack3
from paths import BUNDLES, OCC3_PACK

from airops_agentenv.occ import tasks3
from airops_agentenv.occ.tasks import VERIFIERS

pytestmark = pytest.mark.anyio


def test_a_bundle_day_deploys_three_desks_or_one_agent_capped():
    task = storm3().task
    joint = tasks3.steps(task, "joint", policy="script")
    agents = [s for s in joint if s["type"] == "deploy_agent"]
    assert [(s["role"], s["agent_name"]) for s in agents] == [("crew", "crew-desk"), ("customer", "customer-desk"),
                                                               ("ops", "ops-desk")]
    assert all(s["env_vars"] == {"AIROPS_MAX_COST_USD": "5", "AIROPS_DESK_POLICY": "script"} for s in agents)
    tools = {s["role"]: s["tools"] for s in joint if s["type"] == "modify_env_tool_access" and s["action"] == "enable"}
    assert tools["customer"] == ["cust_flights", "cust_flight", "cust_request", "end_turn", "wait_turn", "resign"]
    one = tasks3.steps(task, "one")
    [agent] = [s for s in one if s["type"] == "deploy_agent"]
    assert agent["role"] == "both" and agent["env_vars"] == {"AIROPS_MAX_COST_USD": "12"}
    [play_one] = [s for s in one if s["type"] == "prompt_agent"]
    assert "cust_flight" in play_one["system_prompt"] and play_one["max_turns"] == 10 * 25 + 10
    assert play_one["timeout_seconds"] == tasks3.PLAY_SECONDS == 21600
    idle = tasks3.steps(task, "idle")
    assert [s["type"] for s in idle] == ["deploy_env", "apply_server_config", "register_env_triggers",
                                         "sync_env_clock", "apply_server_config", "env_outcome_verifier"]
    assert all(s.get("env_id", "airops-occ3") == "airops-occ3" for s in joint + one + idle)
    assert "rebook" in tasks3.RULES["customer"] and "PROTECT" in tasks3.RULES["ops"]


def test_the_bundle_and_the_pack(tmp_path):
    w = storm3()
    pack = write_pack3(tmp_path / "pack", w)
    names = tasks3.generate(tmp_path / "bundle", split="eval", pack=pack, noplay=True)
    assert names == [f"{w.task.task_id}-joint", f"{w.task.task_id}-one", "wiring-noplay"]
    readme = (tmp_path / "bundle" / "README.md").read_text()
    assert "envs/airops-occ3, built from its Dockerfile" in readme and "cost P" in readme
    assert (tmp_path / "bundle" / "artifacts" / "airops-occ3-verifier" / "verify.py").is_file()
    again = Pack3(pack).world(w.task.task_id)
    assert again.task == w.task and again.passengers == w.passengers and again.refs["best"] == w.refs["best"]


def load_verifier() -> types.ModuleType:
    path = VERIFIERS / "airops-occ3-verifier" / "verify.py"
    module = types.ModuleType("occ3_verifier")
    exec(compile(path.read_text(), str(path), "exec"), module.__dict__)
    return module


async def test_the_verifier_scores_the_reward_under_p_with_reward_l_beside_it_and_refuses_a_failed_audit(monkeypatch):
    verifier = load_verifier()
    day = play(storm3(), "joint", scripted("crew", "customer", "ops"), True, OccDay3)
    data = json.loads(json.dumps(day.data(), default=str))

    async def get_data(url):
        return types.SimpleNamespace(parts=[types.SimpleNamespace(data=data)])
    monkeypatch.setattr(verifier.client, "get_data", get_data)
    results = await verifier.verify("http://env/mcp")
    assert [r.get("weight", 1) for r in results] == [1, 0, 0]
    assert results[0]["score"] == data["grade"]["reward"] and results[1]["score"] == data["grade"]["reward_L"]
    assert aggregate_score(results, ScoreAggregator.WEIGHTED_AVERAGE) == data["grade"]["reward"]
    data["grade"]["reward_L"] = None
    assert len(await verifier.verify("http://env/mcp")) == 2
    data["audit"] = {"ok": False, "problems": {"A7": ["customer: crash"]}}
    with pytest.raises(RuntimeError, match="can't be scored"):
        await verifier.verify("http://env/mcp")


def test_generate_writes_the_task_files_the_board_played(tmp_path):
    names = tasks3.generate(tmp_path, split="all", pack=OCC3_PACK, caps={"joint": 3.0, "one": 6.0})
    assert len(names) == 4
    written = {p.name: p.read_bytes() for p in sorted((tmp_path / "tasks").iterdir())}
    assert written == {p.name: p.read_bytes() for p in sorted((BUNDLES / "airops-occ3" / "tasks").iterdir())}
    assert (tmp_path / "artifacts/airops-occ3-verifier/verify.py").read_bytes() == (
        VERIFIERS / "airops-occ3-verifier" / "verify.py").read_bytes()
