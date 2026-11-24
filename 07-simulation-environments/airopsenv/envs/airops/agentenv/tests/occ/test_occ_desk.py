"""airops-desk on scripted model turns from the fake LiteLLM against a served airops-occ day, or an airops-occ3 day
with its passengers (no gateway, so every tool is listed): the model sees its page as the result of end_turn, never
sees wait_turn or resign, and the day goes on
without a desk that stops at its cost cap, which it leaves as played; a request that fails is sent again and costs
nothing; a reply cut off at the output limit leaves no reasoning without its answer in the history, a call after
end_turn in the same reply isn't run, an error from wait_turn stops the desk unscored, and a time limit within the
margin fails any desk, the customer desk too, before it plays. No model is called."""

import json
import sys
import types
from pathlib import Path

import httpx
import pytest
from agentenv_protocol.a2a_agent import TaskOutcome, TaskRequest, TextPart
from airops_core.occ.day import OccDay
from occ_helpers import desk_agent, serve, storm, storm3, write_pack, write_pack3

from airops_agentenv.occ.live import OccEnv
from airops_agentenv.occ.live3 import OccEnv3

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fake_litellm import FakeLiteLLM  # noqa: E402

pytestmark = pytest.mark.anyio
SOL = "openai/gpt-6.1-sol"


def end(i: int, big: bool = False) -> dict:
    usage = {"input": 100000, "output": 10000} if big else {"input": 1000, "output": 100}
    return {"content": "", "reasoning": "", "stop": "completed", "usage": usage,
            "tool_calls": [{"id": f"call_{i}", "name": "end_turn", "arguments": json.dumps(
                {"note": None, "stand_down": False})}]}


async def run(tmp_path, monkeypatch, cap: str | None = None, turns: list[dict] | None = None,
              turn_timeout: float | None = None, v3: bool = False):
    """The one-desk day served (airops-occ3's with its passengers when v3), its advisories delivered by the env as each
    watch opens (no gateway, no trigger)."""
    world = storm3() if v3 else storm()
    monkeypatch.setenv("AIROPS_OCC3_PACK" if v3 else "AIROPS_OCC_PACK",
                       str((write_pack3 if v3 else write_pack)(tmp_path / "pack", world)))
    opened = OccDay._open
    monkeypatch.setattr(OccDay, "_open", lambda self, by: opened(self, "env"))
    env = OccEnv3() if v3 else OccEnv()
    mcp = env.create_app()
    env.occ_load(world.task.task_id, "one", *([turn_timeout] if turn_timeout is not None else []))
    turns = turns or [end(i, cap is not None) for i in range(60)]
    with serve(mcp.http_app()) as url, FakeLiteLLM(turns) as fake:
        monkeypatch.setenv("LITELLM_BASE_URL", fake.url)
        monkeypatch.setenv("LITELLM_API_KEY", "sk-fake")
        if cap:
            monkeypatch.setenv("AIROPS_MAX_COST_USD", cap)
        result = await desk_agent.AirOpsDesk().run(TaskRequest(
            task_id="t", context_id="c", parts=(TextPart(text="You run both desks."),),
            config=desk_agent.DeskConfig(model=SOL, system_prompt="The rules.", role="both", max_turns=200),
            mcp_servers={mcp.name: {"url": f"{url}/mcp", "headers": {}}}))
    return result, fake, env


async def test_the_model_reads_each_page_as_end_turns_result(tmp_path, monkeypatch):
    result, fake, env = await run(tmp_path, monkeypatch)
    s = result.parts[1].data["structured_output"]
    day = env.day
    assert result.outcome is TaskOutcome.SUCCEEDED and day.done and day.audit()["ok"]
    assert (s["end_reason"], s["desk"], s["policy"], s["desk_turns"]) == ("done", "one", "llm", len(day.turns))
    assert s["turns"] == len(day.turns) and s["context_max"] == 1000 and s["cost_usd"] == pytest.approx(
        len(day.turns) * (1000 * 2.0 + 100 * 10.0) / 1e6)
    first = fake.requests[0][2]
    assert first["input"][0]["content"][0]["text"].startswith("You run both desks.\n\n# Ops control, JetBlue")
    names = {t["name"] for t in first["tools"]}
    assert {"end_turn", "ops_confirm", "crew_confirm"} <= names and not {"wait_turn", "resign"} & names
    out = json.loads(fake.requests[1][2]["input"][-1]["output"])
    assert out["ended"] == "one" and out["page"].startswith("# Ops control") and out["watch"] == 1
    assert day.absent["one"]["reason"] == "done"


async def test_a_desk_at_its_cost_cap_leaves_and_the_day_goes_on_without_it(tmp_path, monkeypatch):
    result, fake, env = await run(tmp_path, monkeypatch, cap="1.0")
    s = result.parts[1].data["structured_output"]
    assert s["end_reason"] == "cost_cap" and result.outcome is TaskOutcome.SUCCEEDED and len(fake.requests) == 3
    assert env.day.absent["one"]["reason"] == "cost_cap" and env.day.done and env.day.audit()["ok"]
    assert env.day.grade["end_reason"] == "done" and env.day.grade["cost"] == env.day.w.refs["idle"]


async def test_one_desk_plays_a_day_with_passengers_with_the_customer_desks_lookups_too(tmp_path, monkeypatch):
    result, fake, env = await run(tmp_path, monkeypatch, v3=True)
    s = result.parts[1].data["structured_output"]
    day = env.day
    assert result.outcome is TaskOutcome.SUCCEEDED and day.done and day.audit()["ok"] and day.grade["reward_L"]
    assert (s["end_reason"], s["desk"], s["turns"]) == ("done", "one", len(day.turns))
    first = fake.requests[0][2]
    assert "## Connections at risk" in first["input"][0]["content"][0]["text"]
    assert {"cust_flights", "cust_flight", "ops_confirm", "crew_confirm"} <= {t["name"] for t in first["tools"]}


async def test_a_request_that_fails_mid_stream_is_sent_again_and_costs_nothing(tmp_path, monkeypatch):
    result, fake, env = await run(tmp_path, monkeypatch, turns=[{"stream_error": "overloaded"}] +
                                  [end(i) for i in range(60)])
    s = result.parts[1].data["structured_output"]
    assert (s["end_reason"], s["failed_requests"], s["turns"]) == ("done", 1, len(env.day.turns))
    assert s["cost_usd"] == pytest.approx(len(env.day.turns) * (1000 * 2.0 + 100 * 10.0) / 1e6)


async def test_a_cut_off_reply_leaves_no_dangling_reasoning_and_a_call_after_end_turn_is_not_run(tmp_path, monkeypatch):
    cut = {"content": "", "reasoning": "weighing the holds", "stop": "incomplete", "usage": {"input": 1000,
                                                                                                "output": 32000}}
    late = {**end(1), "tool_calls": end(1)["tool_calls"] + [{"id": "call_x", "name": "ops_confirm",
                                                              "arguments": json.dumps({"changes": []})}]}
    result, fake, env = await run(tmp_path, monkeypatch, turns=[cut, late] + [end(i) for i in range(2, 60)])
    second = fake.requests[1][2]["input"]
    assert not any(x.get("type") == "reasoning" for x in second) and "ran out of output tokens" in json.dumps(second)
    third = fake.requests[2][2]["input"]
    assert "not run" in next(x["output"] for x in third if x.get("type") == "function_call_output" and
                             x["call_id"] == "call_x")
    assert not any(c["tool"] == "ops_confirm" for c in env.day.calls) and env.day.done
    assert result.parts[1].data["structured_output"]["effort"] == "medium"


async def test_a_desk_the_env_puts_off_stops_calling_the_model_and_the_day_ends_unscored(tmp_path, monkeypatch):
    result, fake, env = await run(tmp_path, monkeypatch, turn_timeout=0.0)
    s = result.parts[1].data["structured_output"]
    assert (s["end_reason"], s["turns"], len(fake.requests)) == ("off", 1, 1)
    assert env.day.done and env.day.audit()["problems"]["A7"] == ["one: timeout"]


class Replies:
    def __init__(self, *results: dict):
        self.results = list(results)

    async def call(self, name: str, args: dict) -> str:
        return json.dumps(self.results.pop(0))


async def test_wait_turn_polls_until_the_page_and_an_error_without_done_stops_the_desk_unscored():
    episode = desk_agent.Episode(desk_agent.DeskConfig(role="crew"), {})
    page = {"watch": 0, "turn": "crew", "page": "# Crew"}
    assert await episode.wait(Replies({"waiting_for": "ops"}, page)) == page
    assert await episode.wait(Replies({"done": True, "reward": 0.5})) is None and episode.reward == 0.5
    assert episode.record["end_reason"] is None and episode.failure is None
    assert await episode.wait(Replies({"error": "no day loaded"})) is None
    assert episode.record["end_reason"] == "error" and episode.failure[0] == "error"


@pytest.mark.parametrize("role", ["ops", "crew", "customer", "both"])
async def test_a_time_limit_within_the_margin_fails_before_the_desk_plays(role):
    config = desk_agent.DeskConfig(role=role, model=SOL, timeout_seconds=desk_agent.TIME_MARGIN)
    episode = desk_agent.Episode(config, {"LITELLM_BASE_URL": "http://127.0.0.1:9", "LITELLM_API_KEY": "sk-none"})
    await episode.run({"url": "http://127.0.0.1:9/mcp"}, "")
    assert episode.failure[0] == "timeout" and "leaves no turn" in episode.failure[1]
    assert episode.record["end_reason"] == "timeout" and episode.tool_calls == 0


class Dropping:
    def __init__(self):
        self.calls: list[tuple] = []

    async def call(self, name: str, args: dict) -> str:
        self.calls.append((name, args))
        if name == "wait_turn":
            raise httpx.RemoteProtocolError("Server disconnected without sending a response")
        return json.dumps({})


async def test_a_dropped_connection_resigns_the_desk_as_an_error_not_done():
    episode = desk_agent.Episode(desk_agent.DeskConfig(role="crew"), {})
    env = Dropping()
    with pytest.raises(httpx.RemoteProtocolError):
        await episode.play(env, "")
    assert env.calls[-1] == ("resign", {"reason": "error"}) and episode.record["end_reason"] == "error"


def test_the_cap_reads_the_last_request_from_the_cache_only_while_it_is_warm():
    agent = types.SimpleNamespace(route="responses")
    warm = desk_agent.estimate(agent, 35_000, (100_000, 0), 2_000, (2.0, 10.0, 0.1), True)
    cold = desk_agent.estimate(agent, 35_000, (100_000, 0), 2_000, (2.0, 10.0, 0.1), False)
    assert warm == pytest.approx((0.01 + 0.02, 0.04)) and cold == pytest.approx((0.2 + 0.02, 0.04))
