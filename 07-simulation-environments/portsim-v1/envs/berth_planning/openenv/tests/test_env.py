import json
import os
from pathlib import Path

# These tests exercise the first pack (berth-v1); the default served packs are the dock-v1 ones.
os.environ.setdefault("BERTH_TASKS_DIR", str(Path(__file__).resolve().parents[2] / "tasks" / "berth-v1"))

from fastapi.testclient import TestClient  # noqa: E402
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

from berth_core import load_pack
from berth_openenv.environment import MAX_CHECKS, MAX_TOOL_CALLS, BerthPlanningEnvironment
from berth_openenv.server import create_server

PACK = load_pack()


def _call(env, tool, **args):
    return env.step(CallToolAction(tool_name=tool, arguments=args))


def _text(obs):
    res = obs.result
    if hasattr(res, "content"):
        return res.content[0].text
    if isinstance(res, dict):
        return res["content"][0]["text"]
    return str(res)


def test_tools_and_optimal_submit():
    env = BerthPlanningEnvironment(PACK)
    task = PACK.tasks[3]
    obs = env.reset(task_id=task.task_id)
    assert obs.metadata["task_id"] == task.task_id and "Cost" in obs.metadata["instructions"]
    names = {t.name for t in env.step(ListToolsAction()).tools}
    assert names == {"get_situation", "check_plan", "submit_plan"}
    assert task.terminal in _text(_call(env, "get_situation"))
    naive = task.reference["naive_plan"]
    out = _call(env, "check_plan", plan=naive)
    assert out.reward == 0.0 and not out.done
    res = json.loads(_text(out))
    assert res["feasible"] and res["cost"] == task.reference["naive_cost"] and res["checks_left"] == MAX_CHECKS - 1
    assert "optimal" not in _text(out)
    done = _call(env, "submit_plan", plan=task.reference["optimal_plan"])
    assert done.done and done.reward == 1.0
    assert done.metadata["grade"]["cost"] == task.reference["optimal_cost"]
    assert done.metadata["rubric"]["ships_clean"] == 1.0 and done.metadata["rubric"]["cost_quality"] == 1.0
    after = _call(env, "check_plan", plan=naive)
    assert after.error is not None or "over" in _text(after)


def test_infeasible_and_string_plans():
    env = BerthPlanningEnvironment(PACK)
    task = PACK.tasks[10]
    env.reset(task_id=task.task_id)
    published = [{"ship": s.id, "berth_hour": s.planned_hour if s.planned_hour is not None else s.arrival,
                  "section": s.planned_section if s.planned_section is not None else task.first_section}
                 for s in task.ships]
    res = json.loads(_text(_call(env, "check_plan", plan=json.dumps(published))))
    assert not res["feasible"] and res["violations"]
    done = _call(env, "submit_plan", plan=published)
    assert done.done and 0.0 <= done.reward < 0.2
    assert done.metadata["grade"]["feasible"] is False


def test_tool_call_limit_ends_with_zero():
    env = BerthPlanningEnvironment(PACK)
    env.reset(task_id=PACK.tasks[0].task_id)
    obs = None
    for _ in range(MAX_TOOL_CALLS):
        obs = _call(env, "get_situation")
    assert obs.done and obs.reward == 0.0 and obs.metadata["end_reason"] == "tool_call_limit"


def test_http_api_hides_references():
    client = TestClient(create_server())
    assert client.get("/health").status_code == 200
    tasks = client.get("/api/tasks").json()
    assert len(tasks) == 100
    t = client.get(f"/api/tasks/{tasks[0]['task_id']}").json()
    assert "reference" not in t and t["ships"]
    ref = client.get(f"/api/tasks/{tasks[0]['task_id']}/reference").json()
    assert ref["optimal_cost"] < ref["naive_cost"]
    splits = client.get("/berth_planning/splits") if False else None
    task_api = client.post("/berth_planning/num_tasks", json={"split": "test"})
    assert task_api.status_code == 200 and task_api.json()["num_tasks"] == 20
