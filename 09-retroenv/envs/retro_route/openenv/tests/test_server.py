"""The running server: HTTP Task API, MCP over WebSocket, concurrent sessions, the UI mount."""

from __future__ import annotations

import threading

import httpx
from openenv_helpers import BENCHMARK, server_url  # noqa: F401  (server_url is a fixture)
from retroenv.store import TaskStore
from retroenv.verifier import known_routes_submission
from retroenv_openenv.client import RetroEnvClient

STORE = TaskStore(BENCHMARK / "tasks-private", BENCHMARK / "stocks")
SPLIT = "test_id"
STANDARD = [i for i, task in enumerate(STORE.tasks(SPLIT)) if task.variant == "standard"]


def _oracle(split, index):
    task = STORE.task(split, index)
    return known_routes_submission(task, STORE.stock(task.stock_id))


def test_health_metadata_and_task_api(server_url):
    assert httpx.get(f"{server_url}/health").json() == {"status": "healthy"}
    assert httpx.get(f"{server_url}/metadata").json()["name"] == "retro_route"
    client = RetroEnvClient(server_url)
    assert {row["name"] for row in client.splits()} == {"train", "dev", "test_id", "test_hard"}
    assert client.num_tasks(SPLIT) == len(STORE.tasks(SPLIT))
    row = client.task(SPLIT, 0)
    assert row["task_id"] == STORE.task(SPLIT, 0).task_id and "reference_routes" not in row
    client.close()


def test_full_episode_over_mcp(server_url):
    with RetroEnvClient(server_url) as env:
        index = STANDARD[0]
        opening = env.reset(SPLIT, index=index)
        assert opening["task_id"] == STORE.task(SPLIT, index).task_id
        assert {tool["function"]["name"] for tool in env.openai_tools()} >= {"stock_retrieve", "emit_routes"}
        lookup = env.call("stock_retrieve", {"query": opening["target_smiles"], "mode": "exact"})
        assert lookup.done is False and lookup.reward is None and "results" in lookup.result
        unknown = env.call("no_such_tool", {})
        assert unknown.error is not None
        final = env.call("emit_routes", {"submission": _oracle(SPLIT, index)})
        assert final.done is True and final.reward == 1.0 and final.result["score"]["valid"] is True


def test_concurrent_sessions_are_isolated(server_url):
    outcomes = {}

    def episode(index):
        with RetroEnvClient(server_url) as env:
            opening = env.reset(SPLIT, index=index)
            reward = env.call("emit_routes", {"submission": _oracle(SPLIT, index)}).reward
            outcomes[index] = (opening["task_id"] == STORE.task(SPLIT, index).task_id, reward)

    threads = [threading.Thread(target=episode, args=(i,)) for i in STANDARD]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert outcomes == {i: (True, 1.0) for i in STANDARD}


def test_playground_is_mounted(server_url):
    response = httpx.get(f"{server_url}/web/", follow_redirects=True, timeout=30)
    assert response.status_code == 200 and "RetroEnv" in response.text
