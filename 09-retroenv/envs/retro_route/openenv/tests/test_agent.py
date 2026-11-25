"""The OpenAI-compatible agent loop against a live server, with a scripted model."""

from __future__ import annotations

import json
from types import SimpleNamespace

from openenv_helpers import server_url  # noqa: F401  (fixture)
from retroenv_openenv import agent
from retroenv_openenv.agent import normalize_arguments
from retroenv_openenv.client import RetroEnvClient
from test_server import SPLIT, STANDARD, STORE, _oracle


class ScriptedLLM:
    """Replays tool calls; records which tools each request exposed."""

    def __init__(self, calls):
        self.calls = list(calls)
        self.exposed = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **request):
        self.exposed.append([tool["function"]["name"] for tool in request["tools"]])
        name, arguments = self.calls.pop(0) if self.calls else (None, None)
        tool_calls = []
        if name:
            call = SimpleNamespace(
                id=f"call{len(self.exposed)}", function=SimpleNamespace(name=name, arguments=arguments)
            )
            call.model_dump = lambda exclude_none=True, c=call: {
                "id": c.id,
                "type": "function",
                "function": {"name": c.function.name, "arguments": c.function.arguments},
            }
            tool_calls = [call]
        message = SimpleNamespace(content="", tool_calls=tool_calls)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            model="scripted",
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, model_extra={}),
        )


def test_normalize_arguments_repairs_only_a_stringified_submission():
    good = {"routes": []}
    leaf = {"type": "mol", "smiles": "CCO", "in_stock": True, "children": []}
    assert normalize_arguments("emit_routes", {"submission": json.dumps(good) + "}"}) == ({"submission": good}, True)
    broken = json.dumps({"routes": [leaf]})[:-2] + "]]}"
    assert normalize_arguments("emit_routes", {"submission": broken}) == ({"submission": {"routes": [leaf]}}, True)
    assert normalize_arguments("emit_routes", {"submission": good}) == ({"submission": good}, False)
    assert normalize_arguments("emit_routes", {"submission": "no routes here"})[1] is False
    assert normalize_arguments("stock_retrieve", {"query": "{}"})[1] is False


def test_thinking_is_recorded_beside_the_turn_but_not_sent_back():
    entry = {"role": "assistant", "content": "ok"}
    assert agent.recorded(entry, SimpleNamespace(reasoning="because")) == {**entry, "reasoning": "because"}
    assert agent.recorded(entry, SimpleNamespace(reasoning_content="vllm style"))["reasoning"] == "vllm style"
    assert agent.recorded(entry, SimpleNamespace(reasoning=None)) is entry
    assert agent.recorded(entry, SimpleNamespace()) is entry
    assert "reasoning" not in entry


def test_rate_limits_sent_as_bad_requests_are_retried():
    class BadRequestError(Exception):
        def __init__(self, code):
            self.code = code

    assert agent._is_transient(BadRequestError("-20048"))
    assert not agent._is_transient(BadRequestError("invalid_value"))


def test_episode_scores_through_the_server(server_url):
    target = STORE.task(SPLIT, STANDARD[1]).target_smiles
    llm = ScriptedLLM(
        [
            ("stock_retrieve", json.dumps({"query": target, "mode": "exact"})),
            ("emit_routes", json.dumps({"submission": json.dumps(_oracle(SPLIT, STANDARD[1])) + "}"})),
        ]
    )
    with RetroEnvClient(server_url) as env:
        opening = env.reset(SPLIT, index=STANDARD[1])
        result = agent.run_episode(llm, env, opening, agent.AgentConfig(model="scripted", max_turns=4))
    assert result["reward"] == 1.0 and result["valid"] and result["submission_coerced"]
    assert result["tool_calls"] == 2 and not result["auto_emitted"]


def test_final_turn_exposes_only_emit_and_unsubmitted_episodes_score_the_floor(server_url):
    llm = ScriptedLLM([("inspect_molecule", json.dumps({"smiles": "CCO"}))] * 3)
    with RetroEnvClient(server_url) as env:
        opening = env.reset(SPLIT, index=STANDARD[2])
        result = agent.run_episode(llm, env, opening, agent.AgentConfig(model="scripted", max_turns=3))
    assert llm.exposed[-1] == ["emit_routes"]
    assert result["auto_emitted"] and result["reward"] < 0.1 and not result["valid"]


def test_close_episode_keeps_a_score_the_server_already_recorded(server_url):
    """A transport error after a successful emit must not become a floor score."""
    from retroenv_openenv.agent import close_episode

    with RetroEnvClient(server_url) as env:
        env.reset(SPLIT, index=STANDARD[3])
        scored = env.call("emit_routes", {"submission": _oracle(SPLIT, STANDARD[3])})
        assert scored.reward == 1.0
        # The loop missed that reward (its call raised), so it closes the episode.
        final, submission = close_episode(env)
    assert final["reward"] == 1.0
    assert final["recovered_after_transport_error"] is True
    assert submission is None


def test_close_episode_scores_the_floor_when_nothing_was_submitted(server_url):
    from retroenv_openenv.agent import close_episode

    with RetroEnvClient(server_url) as env:
        env.reset(SPLIT, index=STANDARD[3])
        final, submission = close_episode(env)
    assert final["reward"] < 0.1 and submission == {"routes": []}
    assert "recovered_after_transport_error" not in final
