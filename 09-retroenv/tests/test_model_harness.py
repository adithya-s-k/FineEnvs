from __future__ import annotations

import json
import importlib.util
from pathlib import Path
from types import SimpleNamespace


_SPEC = importlib.util.spec_from_file_location(
    "retroenv_eval_run_model", Path(__file__).parents[1] / "eval" / "run_model.py"
)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
rollout = _MODULE.rollout


class _Call:
    id = "call-1"
    function = SimpleNamespace(
        name="emit_routes",
        arguments=json.dumps({"submission": {"routes": []}}),
    )

    def model_dump(self):
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.function.name,
                "arguments": self.function.arguments,
            },
        }


class _Completions:
    def __init__(self):
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        calls = [] if len(self.requests) == 1 else [_Call()]
        message = SimpleNamespace(content="", tool_calls=calls)
        return SimpleNamespace(
            usage=None,
            choices=[SimpleNamespace(message=message)],
            model="mock/model",
        )


class _EmptyCompletions(_Completions):
    def create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(
            usage=None,
            choices=[SimpleNamespace(message=SimpleNamespace(content="", tool_calls=[]))],
            model="mock/empty",
        )


class _Session:
    TOOL_NAMES = ("emit_routes",)

    def __init__(self):
        self.tool_calls = 0
        self.validations = []
        self.final_score = None
        self.done = False

    def emit_routes(self, submission):
        self.done = True
        self.final_score = {"valid": False, "submission": submission}
        return {"done": True, "score": self.final_score}


def test_rollout_recovers_from_empty_turn_and_forces_final_emit():
    completions = _Completions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    result = rollout(
        client,
        "mock/model",
        _Session(),
        {"prompt": "test"},
        max_turns=2,
        temperature=0,
    )

    assert len(completions.requests) == 2
    assert [tool["function"]["name"] for tool in completions.requests[1]["tools"]] == [
        "emit_routes"
    ]
    assert any(
        item.get("role") == "user" and "No tool call" in item.get("content", "")
        for item in result["transcript"]
    )
    assert result["submission"] == {"routes": []}
    assert result["resolved_models"] == ["mock/model"]


def test_rollout_bounds_empty_turn_recovery():
    completions = _EmptyCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    result = rollout(
        client,
        "mock/empty",
        _Session(),
        {"prompt": "test"},
        max_turns=16,
        temperature=0,
        max_empty_turns=2,
    )

    assert len(completions.requests) == 3
    assert [tool["function"]["name"] for tool in completions.requests[-1]["tools"]] == [
        "emit_routes"
    ]
    assert result["submission"] == {"routes": []}
    assert len(result["errors"]) == 3
