"""The OpenEnv environment in-process: tools, terminal reward, budgets, toolsets, Task API."""

from __future__ import annotations

import json

import pytest
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction
from openenv_helpers import resources, settings  # noqa: F401  (resources is a fixture)
from retroenv.tools import tool_names
from retroenv.verifier import known_routes_submission
from retroenv_openenv.config import Resources
from retroenv_openenv.environment import RetroRouteEnvironment

SPLIT = "test_id"


def _call(env, name, **arguments):
    return env.step(CallToolAction(tool_name=name, arguments=arguments))


def _standard(resources, split=SPLIT):
    return [i for i, task in enumerate(resources.store.tasks(split)) if task.variant == "standard"]


def _oracle(env, split, index):
    task = env.resources.store.task(split, index)
    return known_routes_submission(task, env.resources.store.stock(task.stock_id))


def test_reset_reveals_no_hidden_routes(resources):
    env = RetroRouteEnvironment(resources)
    for index, task in enumerate(resources.store.tasks(SPLIT)):
        opening = env.reset(split=SPLIT, index=index)
        text = json.dumps(opening.model_dump())
        assert opening.done is False and opening.reward is None
        for route in task.reference_routes:
            assert route.route_id not in text
            assert not any(source["patent"] in text for source in route.source)
            assert not any(step.product in text for step in route.steps if step.product != task.target_smiles)


def test_tools_match_the_core_toolset_and_schemas(resources):
    env = RetroRouteEnvironment(resources)
    env.reset(split=SPLIT, index=0)
    tools = env.step(ListToolsAction()).tools
    assert [tool.name for tool in tools] == list(tool_names("full"))
    emit = next(tool for tool in tools if tool.name == "emit_routes")
    assert "$defs" in emit.input_schema and "mol" in emit.input_schema["$defs"]


def test_emit_routes_ends_the_episode_and_reports_reward_once(resources):
    env = RetroRouteEnvironment(resources)
    index = _standard(resources)[0]
    env.reset(split=SPLIT, index=index)
    first = _call(env, "emit_routes", submission=_oracle(env, SPLIT, index))
    assert first.done is True and first.reward == 1.0
    assert env.state.done is True and env.state.reward == 1.0
    again = _call(env, "inspect_molecule", smiles="CCO")
    assert again.done is True and again.reward is None


def test_every_fixture_task_is_solved_by_its_known_routes(resources):
    env = RetroRouteEnvironment(resources)
    for split in resources.store.splits():
        for index, task in enumerate(resources.store.tasks(split)):
            env.reset(split=split, index=index)
            final = _call(env, "emit_routes", submission=_oracle(env, split, index))
            assert final.result.data["score"]["valid"], (task.task_id, final.result.data["score"]["hard_failures"])


def test_malformed_submission_is_scored_not_rejected(resources):
    env = RetroRouteEnvironment(resources)
    env.reset(split=SPLIT, index=0)
    observation = _call(env, "emit_routes", submission={"routes": "not a list"})
    assert observation.error is None
    assert observation.done is True and observation.reward == 0.0


def test_budget_exhaustion_keeps_emit_available(resources):
    small = Resources(settings=settings(max_tool_calls=2), benchmark=resources.benchmark)
    env = RetroRouteEnvironment(small)
    index = _standard(resources)[1]
    env.reset(split=SPLIT, index=index)
    for _ in range(2):
        assert "error" not in _call(env, "inspect_molecule", smiles="CCO").result.data
    blocked = _call(env, "inspect_molecule", smiles="CCO").result.data
    assert "budget exhausted" in blocked["error"]
    assert _call(env, "emit_routes", submission=_oracle(env, SPLIT, index)).reward == 1.0


def test_unaided_toolset_drops_the_step_checker(resources):
    unaided = Resources(settings=settings(toolset="unaided"), benchmark=resources.benchmark)
    env = RetroRouteEnvironment(unaided)
    env.reset(split=SPLIT, index=0)
    names = [tool.name for tool in env.step(ListToolsAction()).tools]
    assert "validate_disconnection" not in names and "reaction_class_lookup" in names
    step = resources.store.task(SPLIT, 0).reference_routes[0].steps[0]
    result = _call(env, "reaction_conditions_search", product_smiles=step.product, reactants=list(step.reactants))
    assert result.result.data["source"] == "train-visible analogues"


def test_task_api_serves_public_rows(resources):
    env = RetroRouteEnvironment(resources)
    splits = {row["name"]: row for row in env.list_splits()}
    assert set(splits) == {"train", "dev", "test_id", "test_hard"}
    assert splits[SPLIT]["num_tasks"] == env.num_tasks(SPLIT)
    row = env.get_task(SPLIT, 1)
    assert row["index"] == 1 and "reference_routes" not in row and "difficulty" not in row
    assert len(env.get_task_range(SPLIT, 0, 3)) == 3
    with pytest.raises(IndexError):
        env.get_task(SPLIT, 999)


def test_reset_by_task_id_and_seed(resources):
    env = RetroRouteEnvironment(resources)
    task = resources.store.task("dev", 2)
    assert env.reset(split="dev", task_id=task.task_id).metadata["index"] == 2
    first = env.reset(split="dev", seed=7).metadata["task_id"]
    assert env.reset(split="dev", seed=7).metadata["task_id"] == first
    with pytest.raises(KeyError):
        env.reset(split="dev", task_id="retro_missing")
