"""Offline tests for the openenv_rl recipe (no Tinker API calls, no model downloads).

The cookbook's real env stack (`EnvFromMessageEnv` -> `AgentToolMessageEnv`) is exercised end to
end; only the renderer and the samplers are fakes.
"""

from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import dataclass, field
from typing import Any

import pytest

pytest.importorskip("tinker_cookbook")

import tinker
from openenv.core.env_server.mcp_types import Tool
from openenv.core.harness import (
    ResourceSession,
    ResourceSessionFactory,
    ToolResult,
    VerifyResult,
)
from openenv.core.harness.capture.validate import check_upstream_response
from openenv.core.harness.training import TrainingTrace, TrainingTurn
from tinker_cookbook.renderers.base import ParseTermination, ToolCall

import openenv_rl as ot
from openenv_rl import openenv_env as env_mod

PNG = base64.b64encode(b"\x89PNG fake").decode()


# --------------------------------------------------------------------------- fakes


class FakeRenderer:
    """Token i of an action selects scripted assistant message i; prompts render to 1 token/message."""

    def __init__(self, script: list[dict[str, Any]]):
        self.script = script
        self.rendered: list[list[dict[str, Any]]] = []
        self.tokenizer = type(
            "Tok",
            (),
            {"decode": staticmethod(lambda ids: "".join(f"<{i}>" for i in ids))},
        )()

    def create_conversation_prefix_with_tools(self, tools, system_prompt=""):
        return [
            {
                "role": "system",
                "content": system_prompt + "\nTOOLS:" + ",".join(t["name"] for t in tools),
            }
        ]

    def build_generation_prompt(self, messages, **kwargs):
        self.rendered.append(messages)
        return tinker.ModelInput.from_ints(list(range(len(messages))))

    def get_stop_sequences(self):
        return [0]

    def parse_response(self, tokens):
        return self.script[tokens[0]], ParseTermination.STOP_SEQUENCE


def call(name: str, args: dict | None = None, i: int = 0) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            ToolCall(
                id=f"c{i}",
                function=ToolCall.FunctionBody(name=name, arguments=json.dumps(args or {})),
            )
        ],
    }


@dataclass
class FakeSession(ResourceSession):
    """A tiny environment: `look` returns a screenshot, `submit` ends the episode with a reward."""

    reward: float | None = 0.75
    verify_reward: float | None = "same"  # type: ignore[assignment]
    calls: list[str] = field(default_factory=list)
    closed: bool = False

    def initial_messages(self):
        return [
            {"role": "system", "content": "Be careful."},
            {"role": "user", "content": "Recreate the design."},
        ]

    def list_tools(self):
        return [
            Tool(
                name="look",
                description="screenshot",
                input_schema={"type": "object", "properties": {}},
            ),
            Tool(
                name="submit",
                description="finish",
                input_schema={"type": "object", "properties": {}},
            ),
        ]

    def call_tool(self, name, arguments):
        self.calls.append(name)
        if name == "look":
            return ToolResult(
                data={
                    "content": [
                        {"type": "text", "text": "screen"},
                        {"type": "image", "data": PNG, "mimeType": "image/png"},
                    ]
                }
            )
        return ToolResult(
            data={"content": [{"type": "text", "text": "graded"}]},
            done=True,
            metadata={"reward": self.reward},
        )

    def verify(self, transcript, final_state=None):
        reward = self.reward if self.verify_reward == "same" else self.verify_reward
        if "submit" not in self.calls:
            reward = None
        return VerifyResult(env_reward=reward, done=True, metrics={"score_detail": 0.5})

    def close(self):
        self.closed = True


class FakeFactory(ResourceSessionFactory):
    def __init__(self, **kw):
        self.kw = kw
        self.sessions: list[FakeSession] = []

    def create(self, task, seed=None, episode_id=None):
        s = FakeSession(**self.kw)
        self.sessions.append(s)
        return s


def build(script, factory=None, **kw):
    renderer = FakeRenderer(script)
    factory = factory or FakeFactory()
    builder = ot.OpenEnvEnvGroupBuilder(factory, task={"i": 0}, group_size=2, model_name="fake/model", **kw)
    return builder, factory, renderer


@pytest.fixture
def patch_renderer(monkeypatch):
    def apply(renderer):
        monkeypatch.setattr(env_mod, "make_renderer", lambda *a, **k: renderer)

    return apply


def run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------- content


def test_mcp_result_with_image_becomes_parts():
    parts = ot.to_content_parts(
        {
            "content": [
                {"type": "text", "text": "hi"},
                {"type": "image", "data": PNG, "mimeType": "image/png"},
            ]
        }
    )
    assert parts[0] == {"type": "text", "text": "hi"}
    assert parts[1]["type"] == "image" and parts[1]["image"].startswith("data:image/png;base64,")


def test_text_only_results_stay_strings():
    assert ot.to_content_parts({"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}) == "a\nb"
    assert ot.to_content_parts("plain") == "plain"
    assert json.loads(ot.to_content_parts({"x": 1})) == {"x": 1}


# --------------------------------------------------------------------------- white-box


def test_episode_runs_tools_and_takes_the_environment_reward(patch_renderer):
    script = [call("look", i=0), call("submit", i=1)]
    builder, factory, renderer = build(script)
    patch_renderer(renderer)

    async def go():
        envs = await builder.make_envs()
        assert len(envs) == 2 and len(factory.sessions) == 2
        env = envs[0]
        await env.initial_observation()
        first = renderer.rendered[-1]
        assert (
            first[0]["role"] == "system"
            and "Be careful." in first[0]["content"]
            and "TOOLS:look,submit" in first[0]["content"]
        )
        assert first[-1] == {"role": "user", "content": "Recreate the design."}
        r1 = await env.step([0])
        assert not r1.episode_done and r1.reward == 0
        tool_msg = renderer.rendered[-1][-1]
        assert tool_msg["role"] == "tool" and any(p["type"] == "image" for p in tool_msg["content"])
        r2 = await env.step([1])
        assert r2.episode_done and r2.reward == pytest.approx(0.75)
        assert r2.metrics["openenv/score_detail"] == 0.5
        assert factory.sessions[0].calls == ["look", "submit"]
        await builder.cleanup()
        assert all(s.closed for s in factory.sessions)

    run(go())


def test_ungraded_episode_raises_instead_of_scoring_zero(patch_renderer):
    builder, _, renderer = build([{"role": "assistant", "content": "I am done."}])
    patch_renderer(renderer)

    async def go():
        env = (await builder.make_envs())[0]
        await env.initial_observation()
        with pytest.raises(ot.MissingRewardError):
            await env.step([0])

    run(go())


def test_finalize_tool_grades_a_cut_off_episode(patch_renderer):
    builder, factory, renderer = build([{"role": "assistant", "content": "done"}], finalize_tool="submit")
    patch_renderer(renderer)

    async def go():
        env = (await builder.make_envs())[0]
        await env.initial_observation()
        result = await env.step([0])
        assert result.episode_done and result.reward == pytest.approx(0.75)
        assert factory.sessions[0].calls == ["submit"]

    run(go())


def test_verify_and_tool_reward_must_agree(patch_renderer):
    builder, _, renderer = build([call("submit")], factory=FakeFactory(verify_reward=0.1))
    patch_renderer(renderer)

    async def go():
        env = (await builder.make_envs())[0]
        await env.initial_observation()
        with pytest.raises(ValueError, match="disagrees"):
            await env.step([0])

    run(go())


def test_sum_mode_pays_tool_rewards_per_step(patch_renderer):
    builder, _, renderer = build([call("submit")], reward_source="sum")
    patch_renderer(renderer)

    async def go():
        env = (await builder.make_envs())[0]
        await env.initial_observation()
        result = await env.step([0])
        assert result.episode_done and result.reward == pytest.approx(0.75)

    run(go())


def test_image_window_drops_old_screenshots_in_chunks(patch_renderer):
    script = [call("look", i=i) for i in range(6)]
    builder, _, renderer = build(script, image_window=ot.ImageWindow(max_images=3, keep_images=1))
    patch_renderer(renderer)

    def images(messages):
        return sum(
            1 for m in messages if isinstance(m.get("content"), list) for p in m["content"] if p.get("type") == "image"
        )

    async def go():
        env = (await builder.make_envs())[0]
        await env.initial_observation()
        seen = []
        for i in range(5):
            await env.step([i])
            seen.append(images(renderer.rendered[-1]))
        assert seen == [1, 2, 3, 1, 2]  # grows to max, drops to keep, grows again
        placeholders = [
            p
            for m in renderer.rendered[-1]
            if isinstance(m.get("content"), list)
            for p in m["content"]
            if p.get("type") == "text" and "omitted" in p.get("text", "")
        ]
        assert len(placeholders) == 3

    run(go())


class TargetSession(FakeSession):
    def initial_messages(self):
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Recreate this."},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + PNG}},
                ],
            }
        ]


class TargetFactory(FakeFactory):
    def create(self, task, seed=None, episode_id=None):
        s = TargetSession(**self.kw)
        self.sessions.append(s)
        return s


def test_image_window_never_drops_the_target(patch_renderer):
    script = [call("look", i=i) for i in range(6)]
    builder, _, renderer = build(
        script, factory=TargetFactory(), image_window=ot.ImageWindow(max_images=2, keep_images=1)
    )
    patch_renderer(renderer)

    async def go():
        env = (await builder.make_envs())[0]
        await env.initial_observation()
        for i in range(5):
            await env.step([i])
        user = next(m for m in renderer.rendered[-1] if m["role"] == "user")
        assert any(p.get("type") == "image" for p in user["content"])  # the target survives every drop

    run(go())


def test_builder_pickles_without_live_sessions(patch_renderer):
    import pickle

    builder, _, renderer = build([call("submit")])
    patch_renderer(renderer)
    run(builder.make_envs())
    clone = pickle.loads(pickle.dumps(builder))
    assert clone._sessions == [] and clone.group_size == 2


def test_dataset_builder_batches_groups():
    ds_builder = ot.OpenEnvRLDatasetBuilder(
        session_factory=FakeFactory(),
        tasks=list(range(5)),
        model_name="fake/model",
        batch_size=2,
        group_size=4,
        test_tasks=[9],
    )
    train, test = run(ds_builder())
    assert len(train) == 3 and [len(train.get_batch(i)) for i in range(3)] == [2, 2, 1]
    assert train.get_batch(0)[0].group_size == 4 and test.get_batch(0)[0].group_size == 1


# --------------------------------------------------------------------------- black-box


def turn(node, prompt, completion, mask=None):
    return TrainingTurn(
        node_id=node,
        prompt_token_ids=prompt,
        completion_token_ids=completion,
        per_token_logps=[-0.5] * len(completion),
        loss_mask=[0] * len(prompt) + (mask or [1] * len(completion)),
    )


def test_extending_turns_merge_into_one_datum_with_masks():
    trace = TrainingTrace(
        turns=[
            turn("a", [1, 2], [3, 4]),
            turn("b", [1, 2, 3, 4, 5], [6, 7], mask=[1, 0]),
        ]
    )
    (datum,) = ot.trace_to_datums(trace, advantage=2.0)
    assert datum.model_input.to_ints() == [1, 2, 3, 4, 5, 6]
    assert datum.loss_fn_inputs["target_tokens"].to_torch().tolist() == [
        2,
        3,
        4,
        5,
        6,
        7,
    ]
    assert datum.loss_fn_inputs["advantages"].to_torch().tolist() == [0, 2, 2, 0, 2, 0]
    assert datum.loss_fn_inputs["logprobs"].to_torch().tolist() == [
        0,
        -0.5,
        -0.5,
        0,
        -0.5,
        -0.5,
    ]


def test_non_extending_turn_starts_a_new_datum():
    trace = TrainingTrace(turns=[turn("a", [1, 2], [3]), turn("b", [9, 9], [4])])
    assert len(ot.trace_to_datums(trace, 1.0)) == 2


def test_constant_reward_groups_are_skipped():
    trace = TrainingTrace(turns=[turn("a", [1], [2])])
    same = ot.BlackBoxGroup(
        task=0,
        rollouts=[ot.BlackBoxRollout(trace, 1.0), ot.BlackBoxRollout(trace, 1.0)],
    )
    diff = ot.BlackBoxGroup(
        task=1,
        rollouts=[ot.BlackBoxRollout(trace, 1.0), ot.BlackBoxRollout(trace, 0.0)],
    )
    assert ot.groups_to_datums([same]) == []
    assert len(ot.groups_to_datums([diff])) == 2
    assert ot.group_advantages([1.0, 0.0]) == [0.5, -0.5]


class LoopSession:
    def __init__(self, code, reward):
        self.code, self.reward, self.closed = code, reward, False

    def wait_for_completion(self, timeout_s=None):
        return self.code

    def fetch_training_trace(self):
        return TrainingTrace(turns=[turn("a", [1], [2])])

    def verify(self, transcript, final_state=None):
        return VerifyResult(env_reward=self.reward, done=True)

    def close(self):
        self.closed = True


class LoopFactory:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)

    def create(self, task, seed=None, episode_id=None):
        return LoopSession(*self.outcomes.pop(0))


def test_black_box_group_drops_failed_and_ungraded_rollouts():
    group = run(ot.run_black_box_group(LoopFactory([(0, 1.0), (1, 0.0), (0, None), (0, 0.0)]), task=0, group_size=4))
    assert sorted(group.rewards) == [0.0, 1.0] and group.failures == 2


# --------------------------------------------------------------------------- server


class FakeSampler:
    async def sample_async(self, prompt, num_samples, sampling_params):
        seq = type(
            "Seq",
            (),
            {
                "tokens": [5, 6, 7],
                "logprobs": [-0.1, -0.2, -0.3],
                "stop_reason": "stop",
            },
        )()
        return type("Res", (), {"sequences": [seq]})()


def test_chat_server_returns_what_the_capture_proxy_needs():
    renderer = FakeRenderer([call("bash", {"cmd": "ls"})] * 8)
    server = ot.TinkerChatServer(FakeSampler(), renderer, model_name="m")
    body = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {"name": "bash", "parameters": {"type": "object"}},
            }
        ],
        "logprobs": True,
        "return_token_ids": True,
        "top_logprobs": 0,
    }
    out = run(server.complete(body))
    assert check_upstream_response(out).ok
    choice = out["choices"][0]
    assert choice["token_ids"] == [5, 6, 7] and len(choice["logprobs"]["content"]) == 3
    assert choice["finish_reason"] == "tool_calls" and choice["message"]["tool_calls"][0]["function"]["name"] == "bash"
    assert out["prompt_token_ids"] and renderer.rendered[-1][0]["content"].startswith("sys")


def test_chat_server_rejects_images_and_streaming():
    server = ot.TinkerChatServer(FakeSampler(), FakeRenderer([]))
    img = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AA"},
                    }
                ],
            }
        ]
    }
    with pytest.raises(Exception, match="image"):
        run(server.complete(img))
    with pytest.raises(Exception, match="stream"):
        run(server.complete({"messages": [], "stream": True}))


def test_history_renders_text_before_a_tool_call_as_sampled():
    """A sampled assistant message parses to content parts ending in the blank line the model wrote before its
    tool call; the history must not add a second one (that broke every multi-turn sequence on Qwen3.5/3.6/3.8)."""
    import pytest

    try:
        from openenv_rl.openenv_env import make_renderer
        renderer = make_renderer("Qwen/Qwen3.6-35B-A3B", "qwen3_5_disable_thinking")
    except Exception as e:  # noqa: BLE001 - needs the tokenizer (downloaded once)
        pytest.skip(f"tokenizer unavailable: {e}")
    from tinker_cookbook.renderers.base import ToolCall

    call = ToolCall(function=ToolCall.FunctionBody(name="screenshot", arguments="{}"))
    msg = {"role": "assistant", "content": [{"type": "text", "text": "Let me look.\n\n"}], "tool_calls": [call]}
    convo = [{"role": "user", "content": "go"}, msg, {"role": "user", "content": "next"}]
    text = renderer.tokenizer.decode(renderer.build_generation_prompt(convo).to_ints())
    assert "Let me look.\n\n<tool_call>" in text and "\n\n\n\n<tool_call>" not in text


def test_tool_images_move_to_a_user_turn_and_keep_prefixes():
    from openenv_rl.openenv_env import _WindowedMessageEnv

    img = {"type": "image", "image": "data:image/png;base64,AAAA"}
    turn1 = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": [img, {"type": "text", "text": "target"}]},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "a"}]},
        {"role": "tool", "name": "left_click", "tool_call_id": "a", "content": [{"type": "text", "text": "clicked"}, img]},
    ]
    view1 = _WindowedMessageEnv._images_to_user(turn1)
    assert view1[3] == {"role": "tool", "name": "left_click", "tool_call_id": "a", "content": "clicked"}
    assert view1[4] == {"role": "user", "content": [img, {"type": "text", "text": "Image returned by left_click."}]}
    turn2 = turn1 + [{"role": "assistant", "content": "", "tool_calls": [{"id": "b"}]},
                     {"role": "tool", "name": "type", "tool_call_id": "b", "content": "typed"}]
    view2 = _WindowedMessageEnv._images_to_user(turn2)
    assert view2[: len(view1)] == view1  # the next prompt extends the last one
    assert view2[-1] == {"role": "tool", "name": "type", "tool_call_id": "b", "content": "typed"}


def test_arguments_are_coerced_to_the_tool_schema():
    from openenv_rl.openenv_tools import coerce_to_schema

    schema = {"type": "object", "properties": {
        "text": {"type": "string"},
        "keys": {"anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]},
        "coordinate": {"type": "array", "items": {"type": "integer"}},
        "duration": {"type": "number"},
    }}
    got = coerce_to_schema({"text": 120, "keys": 5, "coordinate": [10, 20], "duration": 1}, schema)
    assert got == {"text": "120", "keys": "5", "coordinate": [10, 20], "duration": 1}
