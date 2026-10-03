from __future__ import annotations

from fastapi.testclient import TestClient

from retroenv.server.app import app
from retroenv.server.models import RetroRouteAction
from retroenv.server.retro_environment import RetroRouteEnvironment
from retroenv.store import TaskStore


def test_openenv_structured_rollout_reaches_terminal_reward():
    env = RetroRouteEnvironment(
        TaskStore("tasks", "stocks"), default_split="train", max_tool_calls=4
    )
    opening = env.reset(index=0)
    assert opening.done is False
    assert "reference_routes" not in opening.model_dump_json()
    task = env.store.task("train", 0)
    reference = task.reference_routes[0]
    route = {"route": [step.to_dict(include_evidence=False) for step in reference.steps]}
    terminal = env.step(RetroRouteAction(op="submit", route=route))
    assert terminal.done is True
    assert terminal.reward == 1.0


def test_openenv_http_health_discovery_and_reset():
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "healthy"}
        assert "retro_route_env" in client.get("/list_environments").json()
        splits = client.get("/retro_route_env/splits")
        assert splits.status_code == 200
        assert any(item["name"] == "train" for item in splits.json())
        reset = client.post("/reset", json={"seed": 0, "split": "train", "index": 0})
        assert reset.status_code == 200
        payload = reset.json()
        assert payload["done"] is False
        assert "reference_routes" not in reset.text
