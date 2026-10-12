"""The scripted reference expert that generates SFT trajectories."""

from __future__ import annotations

import json

from openenv_helpers import BENCHMARK, _serve, resources, server_url  # noqa: F401  (fixtures)
from retroenv.chemistry import canonicalize_smiles
from retroenv.models import ReactionStep, ReferenceRoute, RetroTask
from retroenv_openenv import agent
from retroenv_openenv.agent_reference import ReferenceExpert, ReferenceTasks, run_episode
from retroenv_openenv.client import RetroEnvClient
from test_server import STORE

TARGET = canonicalize_smiles("CC(=O)Nc1ccc(O)cc1")
MIDDLE = canonicalize_smiles("Nc1ccc(O)cc1")
NITRO = canonicalize_smiles("O=[N+]([O-])c1ccc(O)cc1")
STOCK = frozenset({canonicalize_smiles("CC(=O)Cl"), MIDDLE, NITRO})


def _task() -> RetroTask:
    steps = (
        ReactionStep(product=TARGET, reactants=(canonicalize_smiles("CC(=O)Cl"), MIDDLE)),
        ReactionStep(product=MIDDLE, reactants=(NITRO,)),
    )
    return RetroTask(
        task_id="retro_test",
        target_smiles=TARGET,
        max_depth=3,
        stock_id="s",
        split="train",
        reference_routes=(ReferenceRoute(route_id="r1", steps=steps), ReferenceRoute(route_id="r2", steps=steps)),
    )


def _expert(resources, **kwargs):
    return ReferenceExpert(_task(), resources.benchmark.tool_library, STOCK, **kwargs)


def _turn(expert, messages, tools=("validate_disconnection", "stock_retrieve", "emit_routes")):
    response = expert.create(messages=messages, tools=[{"function": {"name": n}} for n in tools])
    message = response.choices[0].message
    return message.content, [(c.function.name, json.loads(c.function.arguments)) for c in message.tool_calls]


def _answer(messages, content, calls, results):
    turn = sum(m["role"] == "assistant" for m in messages) + 1
    ids = [f"call_{turn}_{k}" for k in range(len(calls))]
    messages.append(
        {
            "role": "assistant",
            "content": content,
            "tool_calls": [
                {"id": i, "type": "function", "function": {"name": n, "arguments": json.dumps(a)}}
                for i, (n, a) in zip(ids, calls)
            ],
        }
    )
    messages.extend(
        {"role": "tool", "tool_call_id": i, "name": n, "content": json.dumps(r)}
        for i, (n, _), r in zip(ids, calls, results)
    )


def test_a_duplicate_first_cut_is_submitted_once(resources):
    assert len(_expert(resources).routes) == 1


def test_an_intermediate_found_in_stock_stays_a_leaf(resources):
    expert = _expert(resources, research=False, alternatives=0)
    messages: list = []
    _, calls = _turn(expert, messages)
    assert [n for n, _ in calls] == ["validate_disconnection", "stock_retrieve", "stock_retrieve"]
    _answer(messages, "", calls, [{"supported": True}, {"results": [{"smiles": "x"}]}, {"results": [{"smiles": "y"}]}])
    content, calls = _turn(expert, messages)
    # The patent makes the aminophenol, but the stock has it, so the expert stops there.
    assert [n for n, _ in calls] == ["emit_routes"] and "Both pieces are in stock" in content
    leaves = calls[0][1]["submission"]["routes"][0]["children"][0]["children"]
    assert all(leaf["in_stock"] and not leaf["children"] for leaf in leaves)


def test_a_missing_intermediate_is_disconnected_next_without_the_validator(resources):
    expert = _expert(resources, research=False, alternatives=0)
    messages: list = []
    tools = ("stock_retrieve", "emit_routes")
    _, calls = _turn(expert, messages, tools)
    assert [n for n, _ in calls] == ["stock_retrieve", "stock_retrieve"]
    _answer(messages, "", calls, [{"results": [{"smiles": "x"}]}, {"results": []}])
    content, calls = _turn(expert, messages, tools)
    assert f"Not in stock: {MIDDLE}." in content and "reduction of the nitro group" in content
    assert calls == [("stock_retrieve", {"query": NITRO, "mode": "exact", "limit": 1})]


def test_expert_solves_every_fixture_train_task_without_writing_its_own_patent(server_url):
    tasks = ReferenceTasks(BENCHMARK)
    config = agent.AgentConfig(model="reference-expert", max_turns=16, temperature=None)
    for index, task in enumerate(STORE.tasks("train")):
        with RetroEnvClient(server_url) as env:
            result = run_episode(tasks, env, env.reset("train", index=index), config)
        assert result["valid"] and not result["errors"] and not result["auto_emitted"], task.task_id
        own = {s["patent"] for route in task.reference_routes for s in route.source}
        written = json.dumps([m for m in result["transcript"] if m["role"] == "assistant"])
        assert not any(patent in written for patent in own), task.task_id


def test_expert_passes_without_the_step_checker():
    process, url = _serve({"RETROENV_TOOLSET": "unaided"})
    try:
        config = agent.AgentConfig(model="reference-expert", max_turns=16, temperature=None)
        with RetroEnvClient(url) as env:
            result = run_episode(ReferenceTasks(BENCHMARK), env, env.reset("train", index=0), config)
    finally:
        process.terminate()
        process.wait(timeout=20)
    names = {c["function"]["name"] for m in result["transcript"] for c in m.get("tool_calls") or []}
    assert result["valid"] and "validate_disconnection" not in names and {"stock_retrieve", "emit_routes"} <= names
