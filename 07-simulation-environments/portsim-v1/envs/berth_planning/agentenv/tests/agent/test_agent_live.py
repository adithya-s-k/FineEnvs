"""portsim-llm in live mode, on scripted turns from the fake LiteLLM against a stub of portsim-live's four agent tools:
it plays a week in watches past v1's 24-call cap, ends on advance's done with its reward, nudges in live terms, and
puts one rolling cache breakpoint on Anthropic requests only. No model is called."""

import json
import threading
import time
from contextlib import contextmanager

import portsim_llm
import pytest
import uvicorn
from agentenv_protocol.a2a_agent import TaskOutcome, TaskRequest, TextPart
from fake_litellm import FakeLiteLLM
from fastmcp import FastMCP

pytestmark = pytest.mark.anyio

KEY = "sk-fake-live-key"
SONNET, GPT, GLM = "anthropic/claude-sonnet-5-5", "openai/gpt-6.1-sol", "fireworks_ai/glm-5p3-flash"
MODELS = [SONNET, GPT, GLM]
WATCHES = [(0, "Mon 00:00"), (30, "Tue 06:00"), (42, "Tue 18:00"), (54, "Wed 06:00"), (84, "Thu 12:00"),
           (90, "Thu 18:00"), (120, "Sat 00:00")]
MAX_TURNS = 5 * len(WATCHES) + 2
WINDOW = {"ship": 7, "berth_hour": 95, "section": 4, "cranes": 2}
DONE = {"done": True, "feasible": True, "cost": 226, "reward": 1.0, "messages": []}
NOTE = "This watch's bulletin arrives with your next tool result."


def bulletin(watch: int) -> dict:
    return {"hour": WATCHES[watch][0], "from": "Harbour master", "text": f"Watch {watch}'s bulletin."}


class StubWeek:
    """portsim-live's agent tools on a scripted week of len(WATCHES) watches, in the env's result shapes: advance
    queues the watch's bulletin and the next result carries it. `calls` lists the tools that were called."""

    def __init__(self):
        self.watch, self.planning, self.queued, self.calls = 0, 0, [], []
        mcp = FastMCP("portsim-live")
        for fn in (self.get_situation, self.check_plan, self.confirm_berths, self.advance):
            mcp.tool(fn)
        self.app = mcp.http_app()

    def answer(self, name: str, result: dict) -> str:
        self.calls.append(name)
        messages, self.queued = self.queued, []
        return json.dumps({**result, "messages": messages})

    def get_situation(self) -> str:
        hour, label = WATCHES[self.watch]
        return self.answer("get_situation", {
            "watch": self.watch, "hour": hour, "time": label, "frozen_before": hour + 6 if self.watch else 0,
            "situation": "The week as the port knows it.", "windows": [{**WINDOW, "departure": 141, "status": "open"}],
            "unconfirmed": [], "planning_calls_left": 3 - self.planning})

    def check_plan(self, plan: list[dict] | str) -> str:
        self.planning += 1
        return self.answer("check_plan", {
            "feasible": True, "cost": 226, "delay_cost": 226, "moves": 0,
            "ships": [{**WINDOW, "departure": 141, "delay_h": 20, "moved": False, "cost": 20, "problems": []}],
            "refused": [], "entry_problems": [], "planning_calls_left": 3 - self.planning})

    def confirm_berths(self, plan: list[dict] | str) -> str:
        self.planning += 1
        return self.answer("confirm_berths", {"confirmed": [7], "unchanged": [], "refused": [], "entry_problems": [],
                                              "planning_calls_left": 3 - self.planning})

    def advance(self) -> str:
        if self.watch == len(WATCHES) - 1:
            return self.answer("advance", {k: v for k, v in DONE.items() if k != "messages"})
        self.watch, self.planning = self.watch + 1, 0
        hour, label = WATCHES[self.watch]
        out = self.answer("advance", {"watch": self.watch, "hour": hour, "time": label, "berthed": [], "departed": [],
                                      "frozen_before": hour + 6, "unconfirmed": [], "note": NOTE})
        self.queued = [bulletin(self.watch)]
        return out


@contextmanager
def serve(app):
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.01)
    yield f"http://127.0.0.1:{server.servers[0].sockets[0].getsockname()[1]}"
    server.should_exit = True
    thread.join()


def call(name: str, arguments: dict, i: int) -> dict:
    return {"id": f"call_{name}_{i}", "name": name, "arguments": json.dumps(arguments)}


def turn(*calls: dict, stop: str = "stop", content: str = "", reasoning: str = "") -> dict:
    return {"content": content, "reasoning": reasoning, "stop": stop, "tool_calls": list(calls)}


def watch_turn(k: int) -> dict:
    return turn(call("check_plan", {"plan": [WINDOW]}, 4 * k), call("confirm_berths", {"plan": [WINDOW]}, 4 * k + 1),
                call("advance", {}, 4 * k + 2), call("get_situation", {}, 4 * k + 3))


WEEK = [watch_turn(k) for k in range(len(WATCHES))]


def summary(result) -> dict:
    return result.parts[1].data["structured_output"]


def tool_outputs(path: str, body: dict) -> list[str]:
    """Every tool result a request hands back to the model, in call order."""
    if path == "/v1/messages":
        return [b["content"][0]["text"] for m in body["messages"] if m["role"] == "user" for b in m["content"]
                if b["type"] == "tool_result"]
    if path == "/v1/responses":
        return [i["output"] for i in body["input"] if i.get("type") == "function_call_output"]
    return [m["content"] for m in body["messages"] if m["role"] == "tool"]


def newest_text(path: str, body: dict) -> str:
    if path == "/v1/messages":
        return body["messages"][-1]["content"][-1]["text"]
    if path == "/v1/responses":
        return body["input"][-1]["content"][0]["text"]
    return body["messages"][-1]["content"]


@pytest.fixture(autouse=True)
def environ(monkeypatch):
    for var in ["LITELLM_BASE_URL", "LITELLM_API_KEY", "PORTSIM_MAX_COST_USD", "ANTHROPIC_API_KEY",
                "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY"]:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def play(monkeypatch):
    """Runs portsim-llm on a fresh stub week against the fake, with the live task's turn limit: the result, the fake
    and the week."""

    async def play(turns: list[dict], model: str, max_turns: int = MAX_TURNS):
        week = StubWeek()
        with serve(week.app) as url, FakeLiteLLM(turns) as fake:
            monkeypatch.setenv("LITELLM_BASE_URL", fake.url)
            monkeypatch.setenv("LITELLM_API_KEY", KEY)
            result = await portsim_llm.PortSimLLM().run(TaskRequest(
                task_id="t", context_id="c", parts=(TextPart(text="The week as known at hour 0.\n"),),
                config=portsim_llm.PortSimConfig(model=model, system_prompt="The live rules.", max_turns=max_turns),
                mcp_servers={"portsim-live": {"url": f"{url}/mcp"}}))
        return result, fake, week

    return play


async def test_an_env_that_lists_advance_is_a_live_week(harness):
    with serve(StubWeek().app) as url, FakeLiteLLM([]) as fake:
        environ = {"LITELLM_BASE_URL": fake.url, "LITELLM_API_KEY": KEY}
        config = portsim_llm.PortSimConfig(max_turns=0)
        live, v1 = portsim_llm.Episode(config, environ), portsim_llm.Episode(config, environ)
        await live.run({"url": f"{url}/mcp"}, "go")
        await v1.run({"url": f"{harness.base_url}/mcp"}, "go")
    assert (live.live, v1.live, fake.requests) == (True, False, [])


@pytest.mark.parametrize("model", MODELS)
async def test_a_week_runs_past_24_calls_and_ends_on_done_with_its_reward(play, model):
    result, fake, week = await play(WEEK, model)
    s, record = summary(result), result.native_trajectory.payload
    assert result.outcome is TaskOutcome.SUCCEEDED
    assert week.calls == ["check_plan", "confirm_berths", "advance", "get_situation"] * 6 + \
        ["check_plan", "confirm_berths", "advance"]
    assert len(fake.requests) == record["turns"] == 7
    assert {k: s[k] for k in ["end_reason", "submitted", "reward", "turns", "tool_calls", "checks"]} == {
        "end_reason": "done", "submitted": False, "reward": 1.0, "turns": 7, "tool_calls": 27, "checks": 7}
    assert (record["end_reason"], record["reward"]) == ("done", 1.0)
    assert record["final"] == {"submitted": False, "plan": None}
    assert result.parts[0].text.startswith("done: reward 1.0, 7 turns, 27 tool calls, $")
    assert [(st["turn"], st["tool"], st["plan"]) for st in record["steps"]] == [
        (k + 1, tool, plan) for k in range(7)
        for tool, plan in [("check_plan", [WINDOW]), ("confirm_berths", [WINDOW]), ("advance", None)]]
    assert record["steps"][-1]["result"] == DONE
    assert [m["content"] for m in record["messages"][-2:]] == [json.dumps(DONE), '{"error": "the episode is over"}']


@pytest.mark.parametrize("model", MODELS)
async def test_each_watch_tells_the_turns_left_and_the_bulletin_arrives_with_the_next_result(play, model):
    _, fake, _ = await play(WEEK, model)
    outs = tool_outputs(*fake.requests[-1][::2])
    assert [out.rpartition("\n")[2] for out in outs] == [f"(turns left: {MAX_TURNS - 1 - k // 4})" for k in range(24)]
    results = [json.loads(out.rpartition("\n")[0]) for out in outs]
    assert [r["watch"] for r in results[2::4]] == [r["watch"] for r in results[3::4]] == [1, 2, 3, 4, 5, 6]
    assert [r["messages"] for r in results[2::4]] == [[]] * 6
    assert [r["messages"] for r in results[3::4]] == [[bulletin(k)] for k in range(1, 7)]


@pytest.mark.parametrize("model", MODELS)
async def test_the_last_turn_note_asks_for_windows(play, model):
    result, fake, _ = await play([WEEK[0], turn(call("check_plan", {"plan": [WINDOW]}, 9))], model, max_turns=2)
    record = result.native_trajectory.payload
    assert newest_text(*fake.requests[1][::2]) == portsim_llm.LIVE.last_turn == (
        "This is your last turn: confirm_berths now for every ship that still needs a window; the rest of the week "
        "then runs on your confirmed windows.")
    assert record["messages"][-3] == {"role": "user", "content": portsim_llm.LIVE.last_turn}
    assert (record["end_reason"], record["turns"], summary(result)["reward"]) == ("turn_limit", 2, None)


@pytest.mark.parametrize("model", MODELS)
async def test_turns_without_a_tool_call_are_nudged_toward_the_live_tools(play, model):
    result, fake, _ = await play([turn(content="Thinking."), WEEK[0], turn(stop="length", reasoning="the draft"),
                                  WEEK[1], WEEK[2]], model, max_turns=5)
    record = result.native_trajectory.payload
    assert newest_text(*fake.requests[1][::2]) == portsim_llm.LIVE.no_call == (
        "No tool was called. Windows only count through the tools: confirm_berths to commit windows, advance to move "
        "to the next watch.")
    assert newest_text(*fake.requests[3][::2]).endswith(portsim_llm.LIVE.cut_off)
    assert portsim_llm.LIVE.cut_off == (
        "Your last turn ran out of output tokens before calling a tool. Your notes from it are above. Keep reasoning "
        "short now and call check_plan with your current draft, or confirm_berths.")
    assert [m["content"] for m in record["messages"] if m["role"] == "user"][1:] == [
        portsim_llm.LIVE.no_call, portsim_llm.LIVE.cut_off, portsim_llm.LIVE.last_turn]


async def test_live_anthropic_requests_carry_one_breakpoint_on_the_newest_block(play):
    _, fake, _ = await play(WEEK, SONNET)
    for _, _, body in fake.requests:
        assert json.dumps(body).count('"cache_control"') == 1
        assert body["messages"][-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert [body["messages"][-1]["content"][-1]["type"] for _, _, body in fake.requests] == \
        ["text"] + ["tool_result"] * 6


@pytest.mark.parametrize("model", [GPT, GLM])
async def test_live_requests_on_the_other_routes_carry_no_breakpoint(play, model):
    _, fake, _ = await play(WEEK, model)
    assert not any("cache_control" in json.dumps(body) for _, _, body in fake.requests)


async def test_v1_anthropic_requests_carry_no_breakpoint(harness, monkeypatch):
    await harness.load_task("dock-24B-w07x1-busy-0")
    with FakeLiteLLM([turn(call("check_plan", {"plan": []}, 0)), turn(call("submit_plan", {"plan": []}, 1))]) as fake:
        monkeypatch.setenv("LITELLM_BASE_URL", fake.url)
        monkeypatch.setenv("LITELLM_API_KEY", KEY)
        result = await portsim_llm.PortSimLLM().run(TaskRequest(
            task_id="t", context_id="c", parts=(TextPart(text="The situation.\n"),),
            config=portsim_llm.PortSimConfig(model=SONNET, system_prompt="The rules."),
            mcp_servers={"portsim": {"url": f"{harness.base_url}/mcp"}}))
    assert (summary(result)["end_reason"], len(fake.requests)) == ("submitted", 2)
    assert not any("cache_control" in json.dumps(body) for _, _, body in fake.requests)
