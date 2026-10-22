"""The AirOps days as a folder bundle, each played live on airops-day by airops-llm and graded by airops-verifier."""

import json
from pathlib import Path

from airops_core.core.model import Task, load_pack
from airops_core.day import Day
from airops_core.schedule import (
    AGENT,
    END_DAY_URI,
    ENV,
    LIVE_LOAD_URI,
    NOTICE_TOOL,
    PLANNING_CALLS,
    VERIFIER,
    max_turns,
    timetable,
    triggers,
    virtual_time,
)

from .live import PACK

VERIFIERS = Path(__file__).resolve().parents[1] / "verifiers"
NOPLAY = "wiring-noplay"
CREDIT = ("Built from public U.S. records: BTS on-time and T-100 data, the FAA registry, FAA Command Center advisories "
          "and NWS METARs, TAFs and forecast discussions via the Iowa Environmental Mesonet.")
AIROPS_RULES = f"""You run the JetBlue operations desk at one of its hubs on a disrupted day: thunderstorms, snow or \
wind, and the FAA ground stops and ground delay programs that follow, cut how many JetBlue flights can land and take \
off each hour. Your job is to decide which flights to cancel and which to hold, watch by watch, so that passengers \
lose as little time as possible over the day.

How the day runs:
- The day is played in watches, one an hour, starting at watch 0, an hour before the disruption. The first message \
gives the day as the desk knows it at watch 0. Time only passes when you call advance.
- What the desk knows: a flight's own departure delay (weather, air traffic control, crews: anything but waiting for \
its aircraft) becomes known 2 hours before its scheduled departure; until then it is assumed on time. How many JetBlue \
arrivals and departures an airport can take in an hour (its slot cap) becomes known 4 hours ahead; until then the \
table shows JetBlue's own schedule for that hour. FAA Command Center advisories, the hub's METARs and TAFs and the \
forecasters' discussion arrive as messages as they are issued.
- The env flies your plan on the day as it happens: each flight leaves at the earliest time its delay, its hold and an \
aircraft allow. It takes the first aircraft of its type ready at its origin (aircraft of one type at one airport are \
interchangeable; an aircraft needs its turn time on the ground between flights), and leaves only when the hour has a \
free slot at its origin and the hour it lands has one at its destination; otherwise it waits for the next hour with \
room. A flight that can't leave within 6 hours of its schedule is cancelled automatically. At the end of the day the \
env ferries each aircraft to the airport it is needed at tomorrow.
- Your decisions: cancel a flight (its aircraft stays on the ground, free for other flights), restore a cancelled \
flight, or hold a flight until a time (not_before) so that its slot or aircraft goes to another flight first. A flight \
you don't touch leaves as early as it can.
- The freeze: at watch 0 nothing is frozen. From watch 1 on, flights your plan has leaving before the freeze line, an \
hour after the watch, are frozen: they can't be changed, and stay frozen. Changes reach no further back than the \
freeze line: the env holds any other flight your plan would have leaving before it to the freeze line.
- You have {PLANNING_CALLS} planning calls (check_plan or confirm_changes) per watch. Call advance and get_situation \
together in one turn: the new watch's news arrives with the first tool result after advance.

Tools:
- get_situation(): the time, the freeze line, the slot caps as known for the next 8 hours, the FAA programs in effect, \
the latest METAR and TAF, every flight (its status, its expected departure once its delay is known, your plan, where \
your plan has it leave and land on the day as known, how late it lands and the aircraft it takes), where the aircraft \
are, and new messages.
- check_plan(changes): your changes over your plan, run on the day as known now: the plan's cost, the flights whose \
plan or times change, the automatic cancellations, the flights the env would hold to the freeze line, and the changes \
confirm_changes would refuse. It confirms nothing.
- confirm_changes(changes): commits the changes; flights you leave out keep their plan.
- advance(): ends the watch. After the last watch it returns "done": the rest of the day runs on your plan.
A change is {{"flight": "f12", "action": "cancel" | "restore" | "hold", "not_before": "HH:MM" | null}}: cancel takes \
not_before null; restore and hold fly the flight, held until not_before (local time, "00:30+1" after midnight), or as \
early as it can with null.

The cost, in passenger-minutes: every passenger's minutes of arrival delay, 523 minutes for each passenger on a \
cancelled flight (rebooking takes most of a day, and some ask for a refund), and the end-of-day ferries (a few \
thousand each). A cancellation frees an aircraft and a slot; it pays when it saves the other passengers more delay \
than it costs its own. The day is graded once, on your final plan flown on the day as it really happened: the closer \
its cost to the best plan possible in hindsight, the higher the score. If you stop before the last watch, the rest of \
the day runs on your plan."""
README = f"""AirOps: a storm day at a JetBlue hub played live in hourly watches on the gateway's virtual clock by the \
airops-llm planner, the FAA advisories and the weather arriving as issued, graded against the hindsight optimum.

Every task deploys the env `{ENV}` behind the AgentEnv gateway (the bundle's envs/{ENV}, built from its Dockerfile)
and loads one day of the b6-v1 pack through the env's `{LIVE_LOAD_URI}` extension. It hides the env's
`{NOTICE_TOOL}` tool from the agent's role, registers one trigger per watch that delivers the watch's news through
`{NOTICE_TOOL}` when the agent's `advance` reaches it, and arms the clock frozen at the first watch. After the play,
`{END_DAY_URI}` runs any watches the agent didn't reach, flies the final plan on the day as it happened and grades it.
The `{VERIFIER}` artifact reads the day from the env's `data/get` and scores the run with its reward; a day that never
finished, or whose news broke the reveal schedule, fails the step, so the run is not scored.

- `<task id>` plays that day with the `{AGENT}` agent.
- `{NOPLAY}` runs the first day's steps without the agent and needs no model: end-day delivers every watch's news
  itself and the day flies with no change, as the naive reference does.

{CREDIT}
"""


def pack_tasks(pack: Path | None = None) -> list[Task]:
    return load_pack(pack or Path(str(PACK)))


def agent_env(episode_cap_usd: float | None, hf_bill_to: str | None) -> dict:
    """The agent's environment: a cost cap only when one is asked for (a sweep's), so a published task carries none."""
    env = ({"AIROPS_MAX_COST_USD": format(episode_cap_usd, "g")} if episode_cap_usd is not None else {}) | (
        {"HF_BILL_TO": hf_bill_to} if hf_bill_to else {})
    return {"env_vars": env} if env else {}


def steps(task: Task, episode_cap_usd: float | None = None, hf_bill_to: str | None = None) -> list[dict]:
    watches = timetable(task)
    return [
        {"id": "deploy", "type": "deploy_env", "env_id": ENV},
        {"id": "load-task", "type": "apply_server_config", "env_id": ENV,
         "directives": [{"service": ENV, "uri": LIVE_LOAD_URI, "args": {"task_id": task.task_id}}]},
        {"id": "hide-notice", "type": "modify_env_tool_access", "env_id": ENV, "action": "disable", "role": "default",
         "tools": [NOTICE_TOOL]},
        {"id": "watches", "type": "register_env_triggers", "env_id": ENV, "watch_roles": ["default"],
         "triggers": triggers(watches)},
        {"id": "deploy-agent", "type": "deploy_agent", "a2a_agent_id": AGENT, "env_ids": [ENV],
         **agent_env(episode_cap_usd, hf_bill_to)},
        {"id": "clock", "type": "sync_env_clock", "env_id": ENV, "virtual_time": virtual_time(task, task.start),
         "virtual_seconds_per_real_second": 0, "tolerate_missing_sync_time": False},
        {"id": "play", "type": "prompt_agent", "prompt_id": task.task_id, "system_prompt": AIROPS_RULES,
         "prompt": Day(task).briefing(), "max_turns": max_turns(watches), "model_params": {"max_tokens": 32000},
         "timeout_seconds": 7200},
        {"id": "end-day", "type": "apply_server_config", "env_id": ENV,
         "directives": [{"service": ENV, "uri": END_DAY_URI, "args": {}}]},
        {"id": "grade", "type": "env_outcome_verifier", "env_id": ENV, "file_artifact_id": VERIFIER,
         "verifier_id": "airops", "score_aggregator": "weighted_average"},
    ]


def noplay_steps(task: Task) -> list[dict]:
    return [s for s in steps(task) if s["id"] not in ("deploy-agent", "play")]


def generate(out: Path, *, task_ids: list[str] | None = None, split: str = "eval", pack: Path | None = None,
             episode_cap_usd: float | None = None, hf_bill_to: str | None = None, noplay: bool = False) -> list[str]:
    """Writes the bundle into ``out``: README.md, the verifier and tasks/<task_id>.json for ``task_ids``, else every
    task of ``split``, in the pack's order; with ``noplay``, the first one's steps without the agent too."""
    chosen = [t for t in pack_tasks(pack)
              if (t.task_id in task_ids if task_ids is not None else split in ("all", t.split))]
    _write(out / "README.md", README.encode())
    _write(out / "artifacts" / VERIFIER / "verify.py", (VERIFIERS / VERIFIER / "verify.py").read_bytes())
    for task in chosen:
        _write(out / "tasks" / f"{task.task_id}.json", _dump(steps(task, episode_cap_usd, hf_bill_to)))
    if noplay and chosen:
        _write(out / "tasks" / f"{NOPLAY}.json", _dump(noplay_steps(chosen[0])))
    return [t.task_id for t in chosen]


def _dump(obj) -> bytes:
    return (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode()


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
