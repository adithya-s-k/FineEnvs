"""airops-llm on scripted turns from the fake LiteLLM against a real airops-day env serving a fixture day: its card,
its three routes, the day ending on advance's done with its reward, the turn and time limits and their notes, its cost
and cap, its failures, the cache breakpoint and what it returns. No model is called."""

import json
import logging

import airops_llm
import pytest
from agent_env.a2a_agent.a2a_agent import A2AAgent
from agentenv_protocol.a2a_agent import TaskOutcome, TaskRequest, TextPart
from fake_litellm import FakeLiteLLM
from served import free_port, served_day
from starlette.testclient import TestClient

pytestmark = pytest.mark.anyio

KEY = "sk-fake-agent-key"
SONNET, GPT, GLM = "anthropic/claude-sonnet-5-5", "openai/gpt-6.1-sol", "fireworks_ai/glm-5p3-flash"
MODELS = [SONNET, GPT, GLM]
FORWARDED = {"model", "system_prompt", "effort", "max_turns", "timeout_seconds", "model_params"}
WATCHES = 13
SUMMARY_KEYS = {"agent", "model", "route", "end_reason", "reward", "turns", "tool_calls", "checks", "confirms",
                "input_tokens", "output_tokens", "cached_tokens", "cache_write_tokens", "cost_usd", "max_cost_usd",
                "priced_cost_usd", "failed_requests", "seconds", "error"}
CANCEL = [{"flight": "f3", "action": "cancel", "not_before": None}]


def call(name: str, arguments: dict | str, i: int = 0) -> dict:
    return {"id": f"call_{name}_{i}", "name": name,
            "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments)}


def turn(*calls: dict, stop: str = "stop", content: str = "", reasoning: str = "", **usage) -> dict:
    return {"content": content, "reasoning": reasoning, "stop": stop, "tool_calls": list(calls), "usage": usage}


def watch_turn(k: int) -> dict:
    return turn(call("check_plan", {"changes": CANCEL}, 4 * k), call("confirm_changes", {"changes": CANCEL}, 4 * k + 1),
                call("advance", {}, 4 * k + 2), call("get_situation", {}, 4 * k + 3))


DAY = [watch_turn(k) for k in range(WATCHES)]
RUSH = turn(*[call("advance", {}, i) for i in range(WATCHES)])
MAX_TURNS = 5 * WATCHES + 4


def summary(result) -> dict:
    return result.parts[1].data["structured_output"]


@pytest.fixture(autouse=True)
def environ(monkeypatch):
    for var in ["LITELLM_BASE_URL", "LITELLM_API_KEY", "AIROPS_MAX_COST_USD", "ANTHROPIC_API_KEY",
                "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY", "HF_BILL_TO"]:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def play(monkeypatch):
    """Runs airops-llm on a fresh day against the fake: the result, the fake and the env."""

    async def play(turns: list[dict], model: str = SONNET, *, suffix: str = "", mcp_url: str | None = None,
                   **config):
        with served_day(monkeypatch) as (url, env), FakeLiteLLM(turns) as fake:
            monkeypatch.setenv("LITELLM_BASE_URL", fake.url + suffix)
            monkeypatch.setenv("LITELLM_API_KEY", KEY)
            result = await airops_llm.AirOpsLLM().run(TaskRequest(
                task_id="t", context_id="c", parts=(TextPart(text="The day at watch 0.\n"),),
                config=airops_llm.AirOpsConfig(model=model, system_prompt="The rules.", **config),
                mcp_servers={"airops-day": {"url": mcp_url or f"{url}/mcp", "headers": {"x-env": "1"}}}))
        return result, fake, env

    return play


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


def test_the_card_takes_every_field_prompt_agent_forwards():
    with TestClient(airops_llm.AirOpsLLM().create_app()) as client:
        card = client.get("/.well-known/agent-card.json").json()
    uris = {e["uri"] for e in card["capabilities"]["extensions"]}
    desired = {"model": SONNET, "system_prompt": "s", "effort": "high", "harness": "h", "max_turns": 69,
               "max_thinking_tokens": 1, "output_format": {}, "timeout_seconds": 7200, "task_id": "t",
               "agentenv_tools": [], "model_params": {"max_tokens": 32000}}
    assert (card["name"], card["version"]) == ("airops-llm", "0.1.0")
    assert {"urn:agentenv:mcp-config/v1", "urn:agentenv:trajectory/v1"} <= uris and not any("peer" in u for u in uris)
    assert A2AAgent.negotiate_agent_config(card, desired).fields.keys() == FORWARDED


@pytest.mark.parametrize("suffix", ["", "/v1/"])
@pytest.mark.parametrize(("model", "path", "fields"), [
    (SONNET, "/v1/messages", {"max_tokens": 32000, "stream": True}),
    (GPT, "/v1/responses", {"max_output_tokens": 32000, "store": False, "include": ["reasoning.encrypted_content"],
                            "reasoning": {"effort": "medium", "summary": "auto"}, "stream": True}),
    (GLM, "/v1/chat/completions", {"max_tokens": 32000, "stream": True, "stream_options": {"include_usage": True}}),
])
async def test_each_route_sends_its_request(play, model, path, fields, suffix):
    result, fake, _ = await play([RUSH], model, suffix=suffix)
    [(sent, headers, body)] = fake.requests
    assert result.outcome is TaskOutcome.SUCCEEDED
    assert (sent, body["model"], headers["authorization"]) == (path, model, f"Bearer {KEY}")
    assert "x-api-key" not in headers
    assert {k: body[k] for k in fields} == fields
    assert not {"output_config", "reasoning_effort", "temperature"} & body.keys()
    assert summary(result)["route"] == {"/v1/messages": "messages", "/v1/responses": "responses"}.get(path, "chat")


@pytest.mark.parametrize(("model", "effort"), [
    (SONNET, {"output_config": {"effort": "high"}}),
    (GPT, {"reasoning": {"effort": "high", "summary": "auto"}}),
    (GLM, {"reasoning_effort": "high"}),
])
async def test_effort_high_goes_where_each_api_takes_it(play, model, effort):
    _, fake, _ = await play([RUSH], model, effort="high")
    [(_, _, body)] = fake.requests
    assert {k: body[k] for k in effort} == effort


@pytest.mark.parametrize("model", MODELS)
async def test_a_day_runs_its_watches_and_ends_on_done_with_its_reward(play, model):
    result, fake, env = await play(DAY, model, max_turns=MAX_TURNS)
    s, record = summary(result), result.native_trajectory.payload
    data = env.day.data()
    assert result.outcome is TaskOutcome.SUCCEEDED
    assert len(fake.requests) == record["turns"] == WATCHES
    assert {k: s[k] for k in ["end_reason", "reward", "turns", "tool_calls", "checks", "confirms"]} == {
        "end_reason": "done", "reward": data["reward"], "turns": WATCHES, "tool_calls": 4 * WATCHES - 1,
        "checks": WATCHES, "confirms": WATCHES}
    assert data["done"] and data["end_reason"] == "done" and data["calls_used"] == 4 * WATCHES - 1
    assert data["plan"] == [{"flight": 3, "cancel": True, "not_before": None}]
    assert (record["end_reason"], record["reward"]) == ("done", data["reward"])
    assert [(st["turn"], st["tool"], st["changes"]) for st in record["steps"]] == [
        (k + 1, tool, changes) for k in range(WATCHES)
        for tool, changes in [("check_plan", CANCEL), ("confirm_changes", CANCEL), ("advance", None)]]
    done = {"done": True, "cost": data["grade"]["cost"]["total"], "reward": data["reward"], "messages": []}
    assert record["steps"][-1]["result"] == done
    assert [m["content"] for m in record["messages"][-2:]] == [json.dumps(done, separators=(",", ":")),
                                                              '{"error": "the episode is over"}']
    outs = tool_outputs(*fake.requests[-1][::2])
    assert [out.rpartition("\n")[2] for out in outs] == [f"(turns left: {MAX_TURNS - 1 - k // 4})"
                                                          for k in range(4 * (WATCHES - 1))]
    assert [json.loads(out.rpartition("\n")[0])["watch"] for out in outs[2::4]] == list(range(1, WATCHES))


async def test_calls_after_done_in_the_same_turn_are_not_sent(play):
    result, fake, env = await play([turn(*RUSH["tool_calls"], call("get_situation", {}, 99)), RUSH])
    record = result.native_trajectory.payload
    assert len(fake.requests) == 1 and env.day.calls == WATCHES
    assert record["messages"][-1]["content"] == '{"error": "the episode is over"}'


@pytest.mark.parametrize("model", MODELS)
async def test_turns_left_and_the_last_turn_note_until_the_turn_limit(play, model):
    result, fake, _ = await play([DAY[0], turn(call("check_plan", {"changes": CANCEL}, 9))], model, max_turns=2)
    record = result.native_trajectory.payload
    assert newest_text(*fake.requests[1][::2]) == airops_llm.LAST_TURN == (
        "This is your last turn: confirm_changes now with any change you still want; the rest of the day then runs "
        "on your plan.")
    assert record["messages"][-3] == {"role": "user", "content": airops_llm.LAST_TURN}
    assert (record["end_reason"], record["turns"], summary(result)["reward"]) == ("turn_limit", 2, None)


@pytest.mark.parametrize("model", MODELS)
async def test_turns_without_a_tool_call_are_nudged_and_two_in_a_row_end_the_day(play, model):
    result, fake, _ = await play([turn(content="Thinking."), DAY[0], turn(stop="length", reasoning="the draft"),
                                  DAY[1], turn(content="Done?"), turn(content="Done.")], model, max_turns=8)
    record = result.native_trajectory.payload
    assert newest_text(*fake.requests[1][::2]) == airops_llm.NO_CALL == (
        "No tool was called. Changes only count through the tools: confirm_changes to commit changes, advance to "
        "move to the next watch.")
    assert newest_text(*fake.requests[3][::2]).endswith(airops_llm.CUT_OFF)
    assert airops_llm.CUT_OFF == (
        "Your last turn ran out of output tokens before calling a tool. Your notes from it are above. Keep reasoning "
        "short now and call check_plan with your current changes, or confirm_changes.")
    assert (record["end_reason"], record["turns"], len(fake.requests)) == ("no_tool_call", 6, 6)


async def test_a_chat_turn_cut_off_while_reasoning_gets_its_notes_back(play):
    _, fake, _ = await play([turn(stop="length", reasoning="x" * 9000 + "the tail"), RUSH], GLM)
    cut = fake.requests[1][2]["messages"]
    assert cut[-2] == {"role": "assistant", "content": "(notes so far, cut off by the output limit)\n"
                                                       + ("x" * 9000 + "the tail")[-8000:]}
    assert cut[-1]["content"].startswith("Your last turn ran out of output tokens")


async def test_arguments_that_are_not_json_are_answered_by_the_harness(play):
    result, fake, env = await play([turn(call("check_plan", "{")), RUSH], GLM)
    record = result.native_trajectory.payload
    assert record["messages"][3]["content"] == json.dumps(
        {"error": "arguments are not valid JSON: Expecting property name enclosed in double quotes"})
    [sent] = [m for m in fake.requests[1][2]["messages"] if m.get("tool_calls")]
    assert sent["tool_calls"][0]["function"]["arguments"] == "{}"
    assert [c["arguments"] for m in record["messages"] for c in m.get("tool_calls") or []][0] == "{"
    assert env.day.planning_calls[0] == 0 and summary(result)["end_reason"] == "done"


@pytest.mark.parametrize(("model", "cost"), [
    (SONNET, (500 * 2.00 + 300 * 0.20 + 200 * 1.25 * 2.00 + 400 * 10.00) / 1e6),
    (GPT, (700 * 2.00 + 300 * 0.10 + 400 * 10.00) / 1e6),
    (GLM, (700 * 0.15 + 300 * 0.03 + 400 * 0.50) / 1e6),
])
async def test_cost_prices_fresh_cached_and_written_input(play, model, cost):
    written = 200 if model == SONNET else 0
    result, _, _ = await play([{**RUSH, "usage": {"input": 1000, "output": 400, "cached": 300,
                                                  "cache_write": written}}], model)
    s = summary(result)
    assert (s["input_tokens"], s["output_tokens"], s["cached_tokens"], s["cache_write_tokens"]) == \
        (1000, 400, 300, written)
    assert s["cost_usd"] == round(cost, 6) == result.usage.cost_usd
    assert result.usage.provider_details == {"cached_tokens": 300, "cache_write_tokens": written}


async def test_the_cap_stops_before_a_request_that_could_pass_it(play, monkeypatch):
    monkeypatch.setenv("AIROPS_MAX_COST_USD", "0.03")
    result, fake, _ = await play([turn(call("get_situation", {}), input=1000, output=2000), RUSH],
                                 model_params={"max_tokens": 1000})
    s = summary(result)
    assert (result.outcome, result.error.code, result.error.error_type) == (TaskOutcome.FAILED, "cost_cap",
                                                                           "infra_error")
    assert len(fake.requests) == 1
    assert (s["end_reason"], s["cost_usd"], s["max_cost_usd"]) == ("cost_cap", 0.022, 0.03)


async def test_a_cap_below_the_first_request_sends_nothing(play, monkeypatch):
    monkeypatch.setenv("AIROPS_MAX_COST_USD", "0.001")
    result, fake, _ = await play([RUSH])
    assert (result.error.code, fake.requests, summary(result)["cost_usd"]) == ("cost_cap", [], 0.0)


async def test_a_chat_stream_the_harness_cuts_is_charged_at_its_bound(play):
    result, _, _ = await play([turn(content="x" * 6000, input=100_000, output=100_000),
                               {**RUSH, "usage": {"input": 10, "output": 10}}], GLM, model_params={"max_tokens": 1000})
    s, record = summary(result), result.native_trajectory.payload
    assert [m["stop"] for m in record["messages"] if m["role"] == "assistant"] == ["length (harness cap)", "stop"]
    assert (s["end_reason"], s["input_tokens"], s["output_tokens"]) == ("done", 10, 10)
    assert s["cost_usd"] > 5 * 1000 * 0.50 / 1e6


async def test_a_reply_that_fails_mid_stream_is_sent_again(play):
    result, fake, _ = await play([{"stream_error": "payload is not completed"}, RUSH], GLM)
    record, s = result.native_trajectory.payload, summary(result)
    assert (s["end_reason"], len(fake.requests)) == ("done", 2)
    assert fake.requests[0][2]["messages"] == fake.requests[1][2]["messages"]
    assert record["errors"] == ["APIError: payload is not completed (turn 1 sent again)"]
    assert s["cost_usd"] > 32000 * 5 * airops_llm.PRICES[GLM][1] / 1e6
    assert s["failed_requests"] == 1 and 0 <= s["priced_cost_usd"] < 0.001 < s["cost_usd"]


async def test_a_reply_that_keeps_failing_mid_stream_is_a_provider_error(play):
    result, fake, _ = await play([{"stream_error": "cut"}] * 3, GLM)
    assert (result.error.code, len(fake.requests)) == ("provider_error", 3)
    assert len(result.native_trajectory.payload["errors"]) == 3


@pytest.mark.parametrize(("model", "requests"), [(SONNET, 5), (GPT, 4), (GLM, 9)])
async def test_a_failing_provider_is_a_provider_error_after_the_sdk_retries(play, model, requests):
    result, fake, _ = await play([{"status": 500}], model)
    s = summary(result)
    assert (result.error.code, len(fake.requests)) == ("provider_error", requests)
    assert s["error"].startswith("InternalServerError: ")
    assert result.native_trajectory.payload["errors"] == [s["error"]]
    assert 32000 * (5 if model == GLM else 1) * airops_llm.PRICES[model][1] / 1e6 < s["cost_usd"] <= 5
    assert (s["failed_requests"], s["priced_cost_usd"]) == (1, 0)


@pytest.mark.parametrize("bill_to", [None, "ScaleAI"])
async def test_hf_bill_to_bills_the_chat_requests_to_that_organization(play, monkeypatch, bill_to):
    if bill_to:
        monkeypatch.setenv("HF_BILL_TO", bill_to)
    result, fake, _ = await play([RUSH], GLM)
    assert summary(result)["end_reason"] == "done"
    assert [headers.get("x-hf-bill-to") for _, headers, _ in fake.requests] == [bill_to]


@pytest.mark.parametrize("model", MODELS)
async def test_a_provider_that_refuses_the_account_ends_the_run_at_no_cost(play, model):
    result, fake, _ = await play([{"status": 402}], model)
    s = summary(result)
    assert (result.error.code, len(fake.requests), s["cost_usd"]) == ("provider_refused", 1, 0)
    assert "402" in s["error"]


async def test_an_env_that_cannot_be_reached_is_an_env_error(play):
    result, fake, _ = await play([RUSH], mcp_url=f"http://127.0.0.1:{free_port()}/mcp")
    assert (result.error.code, fake.requests) == ("env_error", [])
    assert summary(result)["error"] == "ConnectError: All connection attempts failed"


async def test_no_model_endpoint_or_no_price_fails_before_any_request(play, monkeypatch):
    with served_day(monkeypatch) as (url, _):
        result = await airops_llm.AirOpsLLM().run(TaskRequest(
            task_id="t", context_id="c", parts=(TextPart(text="go"),), config=airops_llm.AirOpsConfig(),
            mcp_servers={"airops-day": {"url": f"{url}/mcp"}}))
    assert (result.error.code, result.error.error_type, summary(result)["cost_usd"]) == ("model_endpoint",
                                                                                        "infra_error", 0.0)
    result, fake, _ = await play([RUSH], "fireworks_ai/minimax-m3")
    assert (result.error.code, fake.requests) == ("unpriced_model", [])


async def test_a_time_limit_reached_before_the_first_turn_fails_the_task(play, monkeypatch):
    init = airops_llm.Episode.__init__

    def started_a_minute_ago(self, *args):
        init(self, *args)
        self.started -= 61

    monkeypatch.setattr(airops_llm.Episode, "__init__", started_a_minute_ago)
    result, fake, env = await play([RUSH], timeout_seconds=airops_llm.TIME_MARGIN + 60)
    assert (result.outcome, result.error.code, fake.requests, env.day.calls) == (TaskOutcome.FAILED, "timeout", [], 0)
    assert "left no turn" in result.error.message


async def test_a_time_limit_within_the_margin_fails_before_any_request(play):
    result, fake, env = await play([RUSH], timeout_seconds=airops_llm.TIME_MARGIN)
    assert (result.outcome, result.error.code, fake.requests, env.day.calls) == (
        TaskOutcome.FAILED, "timeout", [], 0)
    assert "leaves no turn" in result.error.message


async def test_the_result_carries_the_text_summary_usage_and_rollout(play):
    result, _, _ = await play([turn(call("check_plan", {"changes": CANCEL}), input=3000, output=1000), RUSH])
    s, record = summary(result), result.native_trajectory.payload
    assert result.parts[0].text == f"done: reward {json.dumps(s['reward'])}, 2 turns, {WATCHES + 1} tool calls, " \
                                   f"${s['cost_usd']:.4f} ({SONNET})"
    assert s.keys() == SUMMARY_KEYS
    assert {k: s[k] for k in ["agent", "model", "route", "end_reason", "turns", "tool_calls", "checks", "confirms",
                              "input_tokens", "output_tokens", "max_cost_usd", "error"]} == {
        "agent": "airops-llm", "model": SONNET, "route": "messages", "end_reason": "done", "turns": 2,
        "tool_calls": WATCHES + 1, "checks": 1, "confirms": 0, "input_tokens": 3000, "output_tokens": 1000,
        "max_cost_usd": None, "error": None}
    assert (result.usage.tool_call_count, result.usage.input_tokens, result.usage.output_tokens,
            result.usage.total_tokens) == (WATCHES + 1, 3000, 1000, 4000)
    assert result.native_trajectory.format == "airops-rollout"
    assert record.keys() == {"model", "messages", "steps", "reward", "usage", "turns", "end_reason", "errors",
                             "seconds"}
    assert record["messages"][:2] == [{"role": "system", "content": "The rules."},
                                      {"role": "user", "content": "The day at watch 0.\n"}]
    assert [step["tool"] for step in record["steps"]] == ["check_plan"] + ["advance"] * WATCHES


async def test_the_key_stays_out_of_the_result_and_the_logs(play, caplog):
    caplog.set_level(logging.DEBUG)
    result, _, _ = await play([turn(call("get_situation", {})), {"status": 401, "message": f"bad key {KEY}"}])
    assert result.error.code == "provider_refused"
    assert "<LITELLM_API_KEY>" in summary(result)["error"]
    assert KEY not in repr(result)
    assert KEY not in caplog.text
    assert "turn 1: 1 tool calls, stop stop, $0.0000 spent" in caplog.text


def test_an_error_is_scrubbed_of_the_key_before_it_is_cut_to_800_characters():
    episode = airops_llm.Episode(airops_llm.AirOpsConfig(), {"LITELLM_API_KEY": KEY})
    episode.fail("provider_error", error=RuntimeError("x" * 795 + KEY))
    assert episode.error == "RuntimeError: " + "x" * 795 + "<LITE"


async def test_anthropic_requests_carry_one_breakpoint_on_the_newest_block(play):
    _, fake, _ = await play(DAY, SONNET, max_turns=MAX_TURNS)
    for _, _, body in fake.requests:
        assert json.dumps(body).count('"cache_control"') == 1
        assert body["messages"][-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert [body["messages"][-1]["content"][-1]["type"] for _, _, body in fake.requests] == \
        ["text"] + ["tool_result"] * (WATCHES - 1)


@pytest.mark.parametrize("model", [GPT, GLM])
async def test_the_other_routes_carry_no_breakpoint(play, model):
    _, fake, _ = await play(DAY, model, max_turns=MAX_TURNS)
    assert not any("cache_control" in json.dumps(body) for _, _, body in fake.requests)


async def test_the_tools_offered_are_the_envs(play):
    _, fake, _ = await play([RUSH], GPT)
    [(_, _, body)] = fake.requests
    assert [t["name"] for t in body["tools"]] == ["get_situation", "check_plan", "confirm_changes", "advance",
                                                  "airops_notice"]
    assert body["tools"][2]["parameters"]["properties"]["changes"]["anyOf"][0]["items"]["required"] == [
        "flight", "action", "not_before"]
