"""Recorded live, marine, wind and lines runs played back through the env over MCP, as the agent and the gateway played
them: load the week, sync the clock, make every recorded call to the env's tools with its recorded arguments, deliver
each watch's notices after the advance that opens it, end the week; then the outputs and the week's data/get against
the recording. On a lines week each message the planner sent a ship's agent is asked of the env as portsim-line asks
it, a request at a time through ship_status or agree_arrival, and each reply compared with the recorded one (an
error by its presence, its wording being the agent's or the gateway's).

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
from port_week_core import lines
from port_week_core.schedule import schedule

from port_week_agentenv.contract import (
    CLOCK_URI,
    END_WEEK_URI,
    LINES_ENV,
    LIVE_ENV,
    LIVE_LOAD_URI,
    MARINE_ENV,
    TOOLS,
    WIND_ENV,
    triggers,
)

PLANNING = ("check_plan", "confirm_berths")
MESSAGE_TOOL = "message_agent"
MAX_REQUESTS = 20
"""portsim-line's cap on the requests in one message."""


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
    messages: int = 0


def env_of(sweep: dict) -> str:
    return (LINES_ENV if sweep.get("lines") else WIND_ENV if sweep.get("wind") else MARINE_ENV if sweep.get("marine")
            else LIVE_ENV)


def timetable(env: str):
    return lines.timetable if env == LINES_ENV else schedule


def episodes(root: Path) -> list[Episode]:
    """The scored attempts of the live sweeps at or under ``root`` (each sweep's folder: sweep.json, results.jsonl and
    transcripts/)."""
    out = []
    for path in [root / "sweep.json"] if (root / "sweep.json").is_file() else sorted(root.glob("*/sweep.json")):
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


def shape(reply) -> object:
    """A reply as replay compares it: an error by its presence alone."""
    if not isinstance(reply, dict) or "error" not in reply:
        return reply
    return {k: v for k, v in reply.items() if k != "error"} | {"error": True}


async def ask(mcp: Client, firm: str, entry) -> dict:
    """One request as portsim-line answers it: through ship_status or agree_arrival, under the firm's name."""
    if not isinstance(entry, dict) or "ship" not in entry:
        return {"error": "each request names a ship"}
    ask, arrival = entry.get("ask"), entry.get("request_arrival")
    if ask == "eta" and arrival is None:
        out = json.loads(text(await mcp.call_tool_mcp("ship_status", {"agent": firm, "ship": entry["ship"]})))
        return out | ({"agreed_before": True} if out.get("agreed") else {})
    if ask is None and arrival is not None:
        result = await mcp.call_tool_mcp("agree_arrival", {"agent": firm, "ship": entry["ship"], "arrival": arrival})
        return json.loads(text(result)) | {"asked": arrival}
    return {"ship": entry["ship"], "error": "ask for the ETA or request an arrival, one per request"}


async def message(mcp: Client, task: Task, call: dict, content: str) -> bool:
    """Plays a message_agent call on the env, each reply checked against the recorded one; False when the message
    never reached the env: the planner's harness refused it, or the agent didn't answer. A message whose recorded
    replies are empty is the agent failing after it asked the env, so its requests are played unchecked."""
    recorded, args = arguments(content), arguments(call["arguments"])
    if not isinstance(recorded, dict) or not isinstance(recorded.get("replies"), list) or args is None:
        return False
    firm, requests = args.get("agent"), args.get("requests")
    assert firm in task.rules["lines"]["firms"] and isinstance(requests, list) and requests, call["id"]
    replies = [] if len(requests) > MAX_REQUESTS else [await ask(mcp, firm, entry) for entry in requests]
    if recorded["replies"]:
        assert [shape(r) for r in replies] == [shape(r) for r in recorded["replies"]], (
            f"{MESSAGE_TOOL} {call['id']}: replayed {replies}, recorded {recorded['replies']}")
    return True


async def replay(mcp: Client, http: httpx.AsyncClient, task: Task, episode: Episode, clock_url: str) -> Replayed:
    """Plays ``episode`` back on a fresh env; raises AssertionError at the first output or outcome that differs."""
    counts = Replayed()
    watches = timetable(episode.env)(task)
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
        if call["name"] == MESSAGE_TOOL and episode.env == LINES_ENV and not done:
            if await message(mcp, task, call, m["content"]):
                counts.messages += 1
            else:
                counts.skipped += 1
            continue
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
    if episode.env == LINES_ENV:
        assert (grade["anchor_hours"], len(week["lines"]["exchanges"])) == (row["anchor_hours"], row["messages"]), (
            episode.run)
    return counts
