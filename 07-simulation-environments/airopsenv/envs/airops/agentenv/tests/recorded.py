"""Recorded airops-day runs played back through the env over MCP, as airops-llm and the gateway played them: load the
day, sync the clock, make every recorded call to the env's tools with its recorded arguments, deliver each watch's news
through airops_notice after the advance that opens it, end the day; then each output and the day's data/get against
the recording and the day its verifier read.

A call the env never saw is skipped: one to a gateway tool (get_time), one with arguments that aren't a JSON object
(the agent answers it itself) and one after the day is done. A check_plan or confirm_changes whose changes the tool's
schema refuses was refused by the gateway before the env saw it, in the gateway's words: here only its being an error
is compared."""

import json
from dataclasses import dataclass
from pathlib import Path

import httpx
from agentenv_protocol import client
from airops_core.core.model import Task
from airops_core.schedule import CLOCK_URI, END_DAY_URI, LIVE_LOAD_URI, TOOLS, timetable, triggers
from fastmcp import Client


@dataclass(frozen=True)
class Episode:
    run: str
    row: dict
    transcript: dict
    verified: dict


@dataclass
class Replayed:
    outputs: int = 0
    refusals: int = 0
    notices: int = 0
    skipped: int = 0


def episodes(root: Path) -> list[Episode]:
    """The scored runs of the sweeps under ``root``, each with the day its verifier read (verified/<task id>.json)."""
    out = []
    for path in sorted(root.glob("*/sweep.json")):
        for line in (path.parent / "results.jsonl").read_text().splitlines():
            row = json.loads(line)
            if row["outcome"] == "scored":
                out.append(Episode(path.parent.name, row, json.loads((path.parent / row["transcript"]).read_text()),
                                   json.loads((path.parent / "verified" / f"{row['task_id']}.json").read_text())))
    return out


def text(result) -> str:
    """The tool's output as the agent records it: its text, or {"error": text}."""
    out = "\n".join(block.text for block in result.content if block.type == "text")
    return json.dumps({"error": out}) if result.isError else out


def arguments(raw: str) -> dict | None:
    try:
        args = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return None
    return args if isinstance(args, dict) else None


async def invoke(http: httpx.AsyncClient, uri: str, **params) -> dict:
    card = (await http.get("/.well-known/agent-env.json")).json()
    response = await http.post(client.extension_params(card, uri)["endpoint"], json=params)
    response.raise_for_status()
    return response.json()


async def data(http: httpx.AsyncClient) -> dict:
    response = await http.post("/agentenv", json={"jsonrpc": "2.0", "id": 1, "method": "data/get"})
    return response.json()["result"]["parts"][0]["data"]


async def replay(mcp: Client, http: httpx.AsyncClient, task: Task, episode: Episode, clock_url: str) -> Replayed:
    """Plays ``episode`` back on a fresh env; raises AssertionError at the first output or outcome that differs."""
    counts = Replayed()
    watches = timetable(task)
    assert await invoke(http, LIVE_LOAD_URI, task_id=task.task_id) == {"task_id": task.task_id,
                                                                        "watches": len(watches)}
    await invoke(http, CLOCK_URI, env_get_time_url=clock_url)
    actions = {t["when"]["where"]["result.watch"]["equals"]: t["actions"] for t in triggers(watches)}
    calls, done = {}, False
    for m in episode.transcript["messages"]:
        if m["role"] == "assistant":
            calls |= {c["id"]: c for c in m.get("tool_calls") or []}
            continue
        if m["role"] != "tool":
            continue
        call = calls[m["tool_call_id"]]
        args = arguments(call["arguments"])
        if call["name"] not in TOOLS or done or args is None:
            counts.skipped += 1
            continue
        result = await mcp.call_tool_mcp(call["name"], args)
        where = f"{episode.run} {task.task_id} {call['name']} {call['id']}"
        if result.isError:
            assert set(json.loads(m["content"])) == {"error"}, f"{where}: replayed {text(result)}, recorded ok"
            counts.refusals += 1
            continue
        assert text(result) == m["content"], f"{where}: replayed {text(result)[:300]}, recorded {m['content'][:300]}"
        counts.outputs += 1
        if call["name"] == "advance":
            out = json.loads(text(result))
            if out.get("done"):
                done = True
                continue
            for action in actions.get(out["watch"], []):
                delivered = await mcp.call_tool_mcp(action["tool"], action["args"])
                assert json.loads(text(delivered)) == {"delivered": action["args"]["news_id"], "watch": out["watch"]}
                counts.notices += 1
    if not done:
        await invoke(http, END_DAY_URI)
    day = await data(http)
    assert day["done"] and day["audit"] == {"ok": True, "problems": []}, episode.run
    assert day["reward"] == episode.row["reward"] and day["grade"] == episode.verified["grade"], episode.run
    return counts
