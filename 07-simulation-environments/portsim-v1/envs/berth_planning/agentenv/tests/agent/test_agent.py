"""portsim-llm on scripted turns from the fake LiteLLM against this env: its card, its routes, upstream's stop rules,
its cost and cap, its failures and what it returns. No model is called."""

import json
import logging
import socket

import portsim_llm
import pytest
from agent_env.a2a_agent.a2a_agent import A2AAgent
from agentenv_protocol.a2a_agent import TaskOutcome, TaskRequest, TextPart
from berth_core import load_pack
from fake_litellm import FakeLiteLLM
from starlette.testclient import TestClient

pytestmark = pytest.mark.anyio

TASK_ID = "dock-24B-w07x1-busy-0"
OPTIMAL = load_pack().get(TASK_ID).reference["optimal_plan"]
KEY = "sk-fake-agent-key"
SONNET, GPT, GLM = "anthropic/claude-sonnet-5-5", "openai/gpt-6.1-sol", "fireworks_ai/glm-5p3-flash"
FORWARDED = {"model", "system_prompt", "effort", "max_turns", "timeout_seconds", "model_params"}
LAST_TURN = "This is your last turn: call submit_plan now with your best plan."
SUMMARY_KEYS = {"agent", "model", "route", "end_reason", "submitted", "reward", "turns", "tool_calls", "checks",
                "input_tokens", "output_tokens", "cached_tokens", "cache_write_tokens", "cost_usd", "max_cost_usd",
                "priced_cost_usd", "failed_requests", "seconds", "error"}


def call(name: str, arguments: dict | str, i: int = 0) -> dict:
    return {"id": f"call_{name}_{i}", "name": name,
            "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments)}


def turn(*calls: dict, stop: str = "stop", content: str = "", reasoning: str = "", **usage) -> dict:
    return {"content": content, "reasoning": reasoning, "stop": stop, "tool_calls": list(calls), "usage": usage}


SUBMIT = turn(call("submit_plan", {"plan": OPTIMAL}))


def summary(result) -> dict:
    return result.parts[1].data["structured_output"]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(autouse=True)
def environ(monkeypatch):
    for var in ["LITELLM_BASE_URL", "LITELLM_API_KEY", "PORTSIM_MAX_COST_USD", "ANTHROPIC_API_KEY",
                "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY", "HF_BILL_TO"]:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def play(harness, monkeypatch):
    """Runs portsim-llm on a fresh episode of TASK_ID against the fake: the result and the fake."""

    async def play(turns: list[dict], model: str = SONNET, *, suffix: str = "", mcp_url: str | None = None,
                   **config):
        await harness.load_task(TASK_ID)
        with FakeLiteLLM(turns) as fake:
            monkeypatch.setenv("LITELLM_BASE_URL", fake.url + suffix)
            monkeypatch.setenv("LITELLM_API_KEY", KEY)
            result = await portsim_llm.PortSimLLM().run(TaskRequest(
                task_id="t", context_id="c", parts=(TextPart(text="The situation.\n"),),
                config=portsim_llm.PortSimConfig(model=model, system_prompt="The rules.", **config),
                mcp_servers={"portsim": {"url": mcp_url or f"{harness.base_url}/mcp", "headers": {"x-env": "1"}}}))
        return result, fake

    return play


@pytest.fixture
def adapters(monkeypatch) -> list:
    """The model adapter each episode makes, to read what it would send next."""
    made, make = [], portsim_llm.make_agent
    monkeypatch.setattr(portsim_llm, "make_agent", lambda *args: made.append(make(*args)) or made[-1])
    return made


def test_the_card_takes_every_field_prompt_agent_forwards():
    with TestClient(portsim_llm.PortSimLLM().create_app()) as client:
        card = client.get("/.well-known/agent-card.json").json()
    uris = {e["uri"] for e in card["capabilities"]["extensions"]}
    desired = {"model": SONNET, "system_prompt": "s", "effort": "high", "harness": "h", "max_turns": 12,
               "max_thinking_tokens": 1, "output_format": {}, "timeout_seconds": 7200, "task_id": "t",
               "agentenv_tools": [], "model_params": {"max_tokens": 32000}}
    assert card["name"] == "portsim-llm"
    assert {"urn:agentenv:mcp-config/v1", "urn:agentenv:trajectory/v1"} <= uris
    assert A2AAgent.negotiate_agent_config(card, desired).fields.keys() == FORWARDED


@pytest.mark.parametrize("suffix", ["", "/v1/"])
@pytest.mark.parametrize(("model", "path", "fields"), [
    (SONNET, "/v1/messages", {"max_tokens": 32000, "stream": True}),
    (GPT, "/v1/responses", {"max_output_tokens": 32000, "store": False, "include": ["reasoning.encrypted_content"],
                            "reasoning": {"effort": "medium", "summary": "auto"}, "stream": True}),
    (GLM, "/v1/chat/completions", {"max_tokens": 32000, "stream": True, "stream_options": {"include_usage": True}}),
])
async def test_each_route_sends_upstreams_request(play, model, path, fields, suffix):
    result, fake = await play([SUBMIT], model, suffix=suffix)
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
    _, fake = await play([SUBMIT], model, effort="high")
    [(_, _, body)] = fake.requests
    assert {k: body[k] for k in effort} == effort


async def test_a_submit_ends_the_episode_with_its_reward_and_later_calls_are_not_sent(play, harness):
    result, fake = await play([turn(call("submit_plan", {"plan": OPTIMAL}), call("check_plan", {"plan": []}, 1)),
                               SUBMIT])
    record, data = result.native_trajectory.payload, await harness.data()
    assert len(fake.requests) == 1
    assert (record["end_reason"], record["reward"], summary(result)["reward"]) == ("submitted", 1.0, 1.0)
    assert record["final"] == {"submitted": True, "plan": OPTIMAL}
    assert record["messages"][-2]["content"] == '{"submitted":true,"feasible":true,"reward":1.0,' \
        f'"cost":{data["grade"]["cost"]},"delay_cost":{data["grade"]["delay_cost"]},"moves":{data["grade"]["moves"]}}}'
    assert record["messages"][-1]["content"] == '{"error": "the episode is over"}'
    assert (data["calls_used"], data["reward"]) == (1, 1.0)


async def test_the_24th_call_ends_the_episode(play, harness, adapters):
    result, fake = await play([turn(*[call("get_situation", {}, i) for i in range(25)]), SUBMIT], GLM)
    record, data = result.native_trajectory.payload, await harness.data()
    told = [m["content"] for m in adapters[0].messages if m["role"] == "tool"]
    assert len(fake.requests) == 1
    assert (record["end_reason"], record["reward"], summary(result)["reward"]) == ("tool_call_limit", 0.0, 0.0)
    assert (summary(result)["tool_calls"], data["calls_used"], data["end_reason"]) == (24, 24, "tool_call_limit")
    assert all(text.endswith("\n(turns left: 11)") for text in told[:23])
    assert told[23] == record["messages"][-2]["content"] and told[23].startswith("# ")
    assert told[24] == '{"error": "the episode is over"}'


async def test_turns_left_and_the_last_turn_note_until_the_turn_limit(play):
    checks = [turn(call("check_plan", {"plan": []}, i)) for i in range(3)]
    result, fake = await play(checks, GLM, max_turns=3)
    record = result.native_trajectory.payload
    sent = [body["messages"] for _, _, body in fake.requests]
    assert (record["end_reason"], record["turns"], summary(result)["checks"]) == ("turn_limit", 3, 3)
    assert [m["content"].rpartition("\n")[2] for m in (sent[1][-1], sent[2][-2])] == ["(turns left: 2)",
                                                                                    "(turns left: 1)"]
    assert sent[2][-1] == {"role": "user", "content": LAST_TURN}
    assert record["messages"][-3] == {"role": "user", "content": LAST_TURN}
    assert summary(result)["reward"] is None


async def test_two_turns_without_a_tool_call_end_the_episode(play):
    result, fake = await play([turn(content="Thinking."), turn(content="Still thinking."), SUBMIT])
    record = result.native_trajectory.payload
    assert result.outcome is TaskOutcome.SUCCEEDED
    assert (record["end_reason"], record["turns"], len(fake.requests)) == ("no_tool_call", 2, 2)
    assert record["messages"][3]["content"].startswith("No tool was called.")
    assert (summary(result)["submitted"], summary(result)["reward"]) == (False, None)


async def test_a_turn_cut_off_while_reasoning_gets_its_notes_back(play):
    _, fake = await play([turn(stop="length", reasoning="x" * 9000 + "the tail"), SUBMIT], GLM)
    cut = fake.requests[1][2]["messages"]
    assert cut[-2] == {"role": "assistant", "content": "(notes so far, cut off by the output limit)\n"
                                                       + ("x" * 9000 + "the tail")[-8000:]}
    assert cut[-1]["content"].startswith("Your last turn ran out of output tokens")


async def test_arguments_that_are_not_json_are_answered_by_the_harness(play, harness):
    result, _ = await play([turn(call("check_plan", "{"))], GLM, max_turns=1)
    record, data = result.native_trajectory.payload, await harness.data()
    assert record["messages"][-1]["content"] == json.dumps(
        {"error": "arguments are not valid JSON: Expecting property name enclosed in double quotes"})
    assert (data["calls_used"], summary(result)["tool_calls"], record["steps"]) == (0, 0, [])


async def test_a_call_that_is_not_json_goes_back_into_the_history_as_an_empty_object(play):
    result, fake = await play([turn(call("check_plan", "{")), SUBMIT], GLM)
    [sent] = [m for m in fake.requests[1][2]["messages"] if m.get("tool_calls")]
    assert sent["tool_calls"][0]["function"]["arguments"] == "{}"
    record = result.native_trajectory.payload
    assert [c["arguments"] for m in record["messages"] for c in m.get("tool_calls") or []][0] == "{"
    assert summary(result)["submitted"] is True


@pytest.mark.parametrize(("model", "cost"), [
    (SONNET, (500 * 2.00 + 300 * 0.20 + 200 * 1.25 * 2.00 + 400 * 10.00) / 1e6),
    (GPT, (700 * 2.00 + 300 * 0.10 + 400 * 10.00) / 1e6),
    (GLM, (700 * 0.15 + 300 * 0.03 + 400 * 0.50) / 1e6),
])
async def test_cost_prices_fresh_cached_and_written_input(play, model, cost):
    written = 200 if model == SONNET else 0
    result, _ = await play([{**SUBMIT, "usage": {"input": 1000, "output": 400, "cached": 300,
                                                 "cache_write": written}}], model)
    s = summary(result)
    assert (s["input_tokens"], s["output_tokens"], s["cached_tokens"], s["cache_write_tokens"]) == \
        (1000, 400, 300, written)
    assert s["cost_usd"] == round(cost, 6) == result.usage.cost_usd
    assert result.usage.provider_details == {"cached_tokens": 300, "cache_write_tokens": written}


async def test_the_cap_stops_before_a_request_that_could_pass_it(play, monkeypatch):
    monkeypatch.setenv("PORTSIM_MAX_COST_USD", "0.03")
    result, fake = await play([turn(call("check_plan", {"plan": []}), input=1000, output=2000), SUBMIT],
                              model_params={"max_tokens": 1000})
    s = summary(result)
    assert (result.outcome, result.error.code, result.error.error_type) == (TaskOutcome.FAILED, "cost_cap",
                                                                           "infra_error")
    assert len(fake.requests) == 1
    assert (s["end_reason"], s["cost_usd"], s["max_cost_usd"]) == ("cost_cap", 0.022, 0.03)


async def test_a_cap_below_the_first_request_sends_nothing(play, monkeypatch):
    monkeypatch.setenv("PORTSIM_MAX_COST_USD", "0.001")
    result, fake = await play([SUBMIT])
    assert (result.error.code, fake.requests, summary(result)["cost_usd"]) == ("cost_cap", [], 0.0)


async def test_a_chat_stream_the_harness_cuts_is_charged_at_its_bound(play):
    result, _ = await play([turn(content="x" * 6000, input=100_000, output=100_000),
                            {**SUBMIT, "usage": {"input": 10, "output": 10}}], GLM, model_params={"max_tokens": 1000})
    s, record = summary(result), result.native_trajectory.payload
    assert [m["stop"] for m in record["messages"] if m["role"] == "assistant"] == ["length (harness cap)", "stop"]
    assert (s["end_reason"], s["input_tokens"], s["output_tokens"]) == ("submitted", 10, 10)
    assert s["cost_usd"] > 5 * 1000 * 0.50 / 1e6


async def test_a_reply_that_fails_mid_stream_is_sent_again(play):
    result, fake = await play([{"stream_error": "payload is not completed"}, SUBMIT], GLM)
    record = result.native_trajectory.payload
    assert (summary(result)["end_reason"], len(fake.requests)) == ("submitted", 2)
    assert fake.requests[0][2]["messages"] == fake.requests[1][2]["messages"]
    assert record["errors"] == ["APIError: payload is not completed (turn 1 sent again)"]
    s = summary(result)
    assert s["cost_usd"] > 32000 * 5 * portsim_llm.PRICES[GLM][1] / 1e6
    assert s["failed_requests"] == 1 and 0 <= s["priced_cost_usd"] < 0.001 < s["cost_usd"]


async def test_a_reply_that_keeps_failing_mid_stream_is_a_provider_error(play):
    result, fake = await play([{"stream_error": "cut"}] * 3, GLM)
    assert (result.error.code, len(fake.requests)) == ("provider_error", 3)
    assert len(result.native_trajectory.payload["errors"]) == 3


@pytest.mark.parametrize(("model", "requests"), [(SONNET, 5), (GPT, 4), (GLM, 9)])
async def test_a_failing_provider_is_a_provider_error_after_the_sdk_retries(play, model, requests):
    result, fake = await play([{"status": 500}], model)
    s = summary(result)
    assert (result.error.code, len(fake.requests)) == ("provider_error", requests)
    assert s["error"].startswith("InternalServerError: ")
    assert result.native_trajectory.payload["errors"] == [s["error"]]
    assert 32000 * (5 if model == GLM else 1) * portsim_llm.PRICES[model][1] / 1e6 < s["cost_usd"] <= 5
    assert (s["failed_requests"], s["priced_cost_usd"]) == (1, 0)


@pytest.mark.parametrize("bill_to", [None, "ScaleAI"])
async def test_hf_bill_to_bills_the_chat_requests_to_that_organization(play, monkeypatch, bill_to):
    if bill_to:
        monkeypatch.setenv("HF_BILL_TO", bill_to)
    result, fake = await play([SUBMIT], GLM)
    assert summary(result)["end_reason"] == "submitted"
    assert [headers.get("x-hf-bill-to") for _, headers, _ in fake.requests] == [bill_to]


@pytest.mark.parametrize("model", [SONNET, GPT, GLM])
async def test_a_provider_that_refuses_the_account_ends_the_run_at_no_cost(play, model):
    result, fake = await play([{"status": 402}], model)
    s = summary(result)
    assert (result.error.code, len(fake.requests), s["cost_usd"]) == ("provider_refused", 1, 0)
    assert "402" in s["error"]


async def test_an_env_that_cannot_be_reached_is_an_env_error(play):
    result, fake = await play([SUBMIT], mcp_url=f"http://127.0.0.1:{free_port()}/mcp")
    assert (result.error.code, fake.requests) == ("env_error", [])
    assert summary(result)["error"] == "ConnectError: All connection attempts failed"


async def test_no_model_endpoint_fails_before_any_request(harness):
    result = await portsim_llm.PortSimLLM().run(TaskRequest(
        task_id="t", context_id="c", parts=(TextPart(text="go"),), config=portsim_llm.PortSimConfig(),
        mcp_servers={"portsim": {"url": f"{harness.base_url}/mcp"}}))
    assert (result.error.code, result.error.error_type) == ("model_endpoint", "infra_error")
    assert summary(result)["cost_usd"] == 0.0


async def test_a_model_without_a_price_fails_before_any_request(play):
    result, fake = await play([SUBMIT], "fireworks_ai/minimax-m3")
    assert (result.error.code, fake.requests) == ("unpriced_model", [])


async def test_a_turn_due_after_the_time_limit_fails_on_the_wall_clock(play):
    result, fake = await play([SUBMIT], timeout_seconds=0)
    assert (result.error.code, fake.requests, summary(result)["end_reason"]) == ("wall_clock", [], "wall_clock")


async def test_the_result_carries_the_text_summary_usage_and_rollout(play):
    result, _ = await play([turn(call("check_plan", {"plan": OPTIMAL}), input=3000, output=1000), SUBMIT])
    s, record = summary(result), result.native_trajectory.payload
    assert result.parts[0].text == f"submitted: reward 1.0, 2 turns, 2 tool calls, ${s['cost_usd']:.4f} ({SONNET})"
    assert s.keys() == SUMMARY_KEYS
    assert {k: s[k] for k in ["agent", "model", "route", "end_reason", "submitted", "reward", "turns", "tool_calls",
                              "checks", "input_tokens", "output_tokens", "max_cost_usd", "error"]} == {
        "agent": "portsim-llm", "model": SONNET, "route": "messages", "end_reason": "submitted", "submitted": True,
        "reward": 1.0, "turns": 2, "tool_calls": 2, "checks": 1, "input_tokens": 3000, "output_tokens": 1000,
        "max_cost_usd": None, "error": None}
    assert (result.usage.tool_call_count, result.usage.input_tokens, result.usage.output_tokens,
            result.usage.total_tokens) == (2, 3000, 1000, 4000)
    assert result.native_trajectory.format == "portsim-rollout"
    assert record.keys() == {"model", "messages", "steps", "final", "reward", "usage", "turns", "end_reason", "errors",
                             "seconds"}
    assert [m["role"] for m in record["messages"]] == ["system", "user", "assistant", "tool", "assistant", "tool"]
    assert record["messages"][:2] == [{"role": "system", "content": "The rules."},
                                      {"role": "user", "content": "The situation.\n"}]
    assert [step["tool"] for step in record["steps"]] == ["check_plan", "submit_plan"]


async def test_the_key_stays_out_of_the_result_and_the_logs(play, caplog):
    caplog.set_level(logging.DEBUG)
    result, _ = await play([turn(call("check_plan", {"plan": []})), {"status": 401, "message": f"bad key {KEY}"}])
    assert result.error.code == "provider_refused"
    assert "<LITELLM_API_KEY>" in summary(result)["error"]
    assert KEY not in repr(result)
    assert KEY not in caplog.text
    assert "turn 1: 1 tool calls, stop stop, $0.0000 spent" in caplog.text


def test_an_error_is_scrubbed_of_the_key_before_it_is_cut_to_800_characters():
    episode = portsim_llm.Episode(portsim_llm.PortSimConfig(), {"LITELLM_API_KEY": KEY})
    episode.fail("provider_error", error=RuntimeError("x" * 795 + KEY))
    assert episode.error == "RuntimeError: " + "x" * 795 + "<LITE"


def test_the_message_schema_lets_a_strict_provider_leave_either_half_out():
    request = portsim_llm.message_schema(["globalliner"])["properties"]["requests"]["items"]
    assert set(request["required"]) == set(request["properties"]) and request["additionalProperties"] is False
    assert request["properties"]["ask"]["enum"] == ["eta", None]
    assert request["properties"]["request_arrival"]["type"] == ["integer", "null"]
