"""The 830 tool calls of the 300 published episodes, replayed in order over MCP, each on its episode's task, return
the recorded outputs byte for byte, except the one in ALLOWED, whose recorded output differs and whose output here is
pinned; each submitted episode then reports its published reward and grade in data/get."""

import json

import pytest

pytestmark = [pytest.mark.anyio, pytest.mark.network]

ALLOWED = {("openai:gpt-6.1-sol", "dock-24B-w35x2-storm-1", 2):
           '{"submitted":true,"feasible":true,"reward":1.0,"cost":312,"delay_cost":282,"moves":6}'}


def tool_calls(episode: dict):
    calls = {}
    for message in json.loads(episode["messages"]):
        if message["role"] == "assistant":
            calls.update((call["id"], call) for call in message.get("tool_calls") or [])
        elif message["role"] == "tool":
            call = calls[message["tool_call_id"]]
            yield call["name"], json.loads(call["arguments"]), message["content"]


async def test_recorded_outputs(harness, episodes):
    replayed, differs, graded = 0, {}, {}
    for episode in episodes:
        await harness.load_task(episode["task_id"])
        async with harness.session() as session:
            for i, (name, arguments, recorded) in enumerate(tool_calls(episode)):
                output = await harness.call(session, name, arguments)
                replayed += 1
                if output != recorded:
                    differs[episode["model"], episode["task_id"], i] = recorded, output
        if episode["submitted"]:
            data = await harness.data()
            graded[episode["model"], episode["task_id"]] = (
                (data["reward"], data["grade"]), (episode["reward"], json.loads(episode["grade"])))
    assert replayed == 830
    assert {key: v for key, v in differs.items() if ALLOWED.get(key) != v[1]} == {}
    assert differs.keys() == ALLOWED.keys()
    assert len(graded) == 202
    assert {key: v for key, v in graded.items() if v[0] != v[1]} == {}
