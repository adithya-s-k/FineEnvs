"""Recorded live, marine and wind runs played back through the env over MCP, as the agent and the gateway played them:
load the week, sync the clock, make every recorded call to the env's tools with its recorded arguments, deliver each
watch's notices after the advance that opens it, end the week; then the outputs and the week's data/get against the
recording.

A call the env never saw is skipped: one to a gateway tool (get_time), one with arguments that aren't a JSON object
(the agent answers it itself) and one after the week is done (the agent answers "the episode is over"). A check_plan or
confirm_berths without a plan was refused by the gateway's schema check before the env saw it, in the gateway's words:
here the env refuses it, only its being an error is compared, and the env's call count is one more than recorded."""

import json
from dataclasses import dataclass
from pathlib import Path

import httpx
from agentenv_protocol import client
from berth_core import Task
from fastmcp import Client
from port_week_core.schedule import schedule

from port_week_agentenv.contract import (
    CLOCK_URI,
    END_WEEK_URI,
    LIVE_ENV,
    LIVE_LOAD_URI,
    MARINE_ENV,
    TOOLS,
    WIND_ENV,
    triggers,
)

PLANNING = ("check_plan", "confirm_berths")


@dataclass(frozen=True)
class Episode:
    run: str
    env: str
    row: dict
    transcript: dict


@dataclass
class Replayed:
    outputs: int = 0
    refusals: int = 0
    notices: int = 0
    skipped: int = 0


def env_of(sweep: dict) -> str:
    return WIND_ENV if sweep.get("wind") else MARINE_ENV if sweep.get("marine") else LIVE_ENV


def episodes(root: Path) -> list[Episode]:
    """The scored attempts of the live, marine and wind sweeps under ``root`` (each sweep's folder: sweep.json,
    results.jsonl and transcripts/)."""
    out = []
    for path in sorted(root.glob("*/sweep.json")):
        sweep = json.loads(path.read_text())
        if not sweep.get("live"):
            continue
        for line in (path.parent / "results.jsonl").read_text().splitlines():
            row = json.loads(line)
            if row["outcome"] == "scored":
                transcript = json.loads((path.parent / row["transcript"]).read_text())
                out.append(Episode(path.parent.name, env_of(sweep), row, transcript))
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
    assert await invoke(http, LIVE_LOAD_URI, task_id=task.task_id) == {"task_id": task.task_id,
                                                                        "watches": len(schedule(task))}
    await invoke(http, CLOCK_URI, env_get_time_url=clock_url)
    actions = {t["when"]["where"]["result.watch"]["equals"]: t["actions"] for t in triggers(schedule(task))}
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
        if call["name"] in PLANNING and "plan" not in args:
            assert result.isError and set(json.loads(m["content"])) == {"error"}, where
            counts.refusals += 1
            continue
        assert text(result) == m["content"], f"{where}: replayed {text(result)}, recorded {m['content']}"
        counts.outputs += 1
        if call["name"] == "advance" and not result.isError:
            out = json.loads(text(result))
            if out.get("done"):
                done = True
                continue
            for action in actions[out["watch"]]:
                delivered = await mcp.call_tool_mcp(action["tool"], action["args"])
                assert json.loads(text(delivered)) == {"applied": action["args"]["event_id"], "watch": out["watch"]}
                counts.notices += 1
    await invoke(http, END_WEEK_URI)
    week, row = await data(http), episode.row
    assert week["done"] and week["audit"] == {"ok": True, "problems": []}, episode.run
    grade = week["grade"]
    assert {"reward": week["reward"], "feasible": grade["feasible"], "plan_cost": grade["cost"],
            "optimal_cost": grade["optimal_cost"], "excused_cost": grade["excused_cost"], "regret": grade["regret"],
            "calls": week["calls_used"] - counts.refusals, "checks": sum(week["planning_calls"]),
            "watches": week["watches"]} == {k: row[k] for k in (
                "reward", "feasible", "plan_cost", "optimal_cost", "excused_cost", "regret", "calls", "checks",
                "watches")}, episode.run
    return counts
