"""The ops control center's days as a folder bundle: two desks (joint) or one desk running both (one), played on the
gateway's clock and graded by airops-occ-verifier."""

import json
from pathlib import Path

from airops_core.occ.day import EXTRA_TURNS, TURNS, timetable
from airops_core.occ.pack import Pack
from airops_core.schedule import virtual_time

from .live import END_DAY_URI, ENV, LOAD_URI, NOTICE_TOOL

AGENT = "airops-desk"
VERIFIER = "airops-occ-verifier"
VERIFIERS = Path(__file__).resolve().parents[2] / "verifiers"
NOPLAY = "wiring-noplay"
BARRIER_SECONDS = 60
PLAY_SECONDS = 21600
MODEL_PARAMS = {"max_tokens": 32000}
EFFORT = "high"
SHARED = ["end_turn", "wait_turn", "resign"]
OPS_TOOLS = ["ops_flights", "ops_program", "ops_aircraft", "ops_check", "ops_confirm", "ops_request", "ops_reply"]
CREW_TOOLS = ["crew_flights", "crew_crews", "crew_reserves", "crew_program", "crew_options", "crew_check",
              "crew_confirm", "crew_request", "crew_reply"]
DESKS = {"joint": [("crew", "crew-desk", CREW_TOOLS), ("ops", "ops-desk", OPS_TOOLS)],
         "one": [("both", "occ-desk", [t for t in OPS_TOOLS + CREW_TOOLS
                                       if not t.endswith(("request", "reply")) and t != "crew_program"])]}
CREDIT = ("Built from public U.S. records: BTS on-time and T-100 data, the FAA registry and FAA Command Center "
          "advisories; crews generated from the published schedule under FAR 117 and 14 CFR 121.467.")

DAY = """The day: JetBlue's whole network on a real storm day, every flight and aircraft, with the FAA's ground stops \
and ground delay programs at JFK and Boston as issued. Watches every hour from 04:00; your turn opens with your page, \
the day as projected now on what is known; act through your tools, then end_turn, whose result is your next page. A \
flight's own delay becomes known 2 hours before its scheduled departure ("?" until then); FAA advisories arrive as \
issued. The env flies the plan: a flight leaves at the earliest its delay, its hold, an aircraft of its type at its \
origin (aircraft of one type at a station are interchangeable, so aircraft swaps happen on their own), its FAA slot or \
the hub's hourly room, and its crews allow; one that can't leave by its deadline is cancelled automatically, late. \
Pilots fly under FAR 117 (duty limits by report time and legs, a 2 h extension with the captain's agreement, 10 h \
rest), flight attendants under 14 CFR 121.467 (14 h scheduled duty, 10 h rest). The crew system crews every flight the \
crew desk leaves alone by standard practice (its own crew, else a reserve, a swap, a deadhead, the extension) and \
can't cancel: where it is stuck the flight waits for a crew or its deadline. Crew flags, pilot/cabin: legal; \
"times out HH:MM" (its crew's last legal departure is near); "crew HH:MM" (it waits for its crew until then); "no \
crew" (no crew can reach it: it will cancel automatically unless something changes). Times are local, "+1" after \
midnight; flights are f<id>."""

COST = """Cost, in passenger-minutes: each passenger's minutes of arrival delay; each cancelled passenger \
(rebooking and refunds, more when decided under 2 hours ahead); the FAA's rates for aircraft time on the ground and in \
the air, and tonight's ferries for tomorrow; crew deadheads (pay and the seats), hotels away from base, and crews out \
of place for tomorrow. The day is graded once, on everything as it really happened, against the best plan found in \
hindsight. Decisions are taken now: none reaches back before this watch."""

OPS = """Ops decisions (ops_check tries them, ops_confirm commits them): cancel; cancel_pair (it and its round-trip \
partner on the planned rotation); hold to a time, no earlier than the freeze line an hour from now (at the first \
watch, from now); release a hold or claim; claim a slot (its arrival time) of a program's pool for a flight the \
program controls; swap_slots between two flights holding slots of one program (FAA CDM refuses a slot outside the \
pool, used or claimed). Cancellations are final. Flights committed at an earlier watch (due out before its freeze \
line) can't be cancelled, held or released, except that accepting the crew desk's CANCEL cancels one: no crew can \
come."""

CREW = """Crew moves (crew_check every move before crew_confirm commits it): assign a unit to fly a flight (the \
flight waits for it, so it can leave late or cancel automatically at its deadline); call_reserve a short-call reserve \
to report on a whole hour (time, or in time for the flight given) and assign it; swap the remaining trips of two \
units of one kind and type at one station; deadhead a unit on a flight; rest a unit to report again at a time (a \
hotel away from base); release a unit you assigned or sent to ride (no flight: its rest, call or swap), or keep it \
off a ride its trip plans. A release undoes a move while nothing it changed has happened yet, as within its turn; \
later it lets the unit go from now and gives back what the move took. Refused: a flight gone or cancelled, another \
type group, a reserve not free to call or a report outside its window, a second period the unit can't start, a swap \
across kinds, types or stations, a release of what isn't yours. A move changes only what it names: a moved unit keeps \
the legs of its trip that still connect; legs it, or the crew it replaced, can no longer make go back to the crew \
system (crew_check: handed_back, and warnings for flights moved over 30 min or cancelled), which doesn't use a unit \
while a flight you named it on is to come. crew_options: the crew system's options for a flight's crew, priced as \
scheduled, the first also on the projection; (a) needs no move. crew_crews, crew_reserves: units and reserves."""

RULES = {
    "ops": f"""You are the ops controller of JetBlue's operations control center. The crew scheduler, another desk, \
runs the crews; you run the flights and aircraft, with the FAA. Keep the day's cost as low as you can.

{DAY}

Each watch the crew desk takes a turn, then you, then the crew desk again if you sent it a request or a text.

{OPS} You never name crews.

The channel: ops_request COVER (crew a flight by a time) or TEXT (400 characters). The crew desk's requests appear on \
your page with ids: HOLD (a flight until a time), CANCEL, CANCEL_PAIR, TEXT. ops_reply accept carries a HOLD or a \
CANCEL out now as your decision; decline leaves it. A request lapses when its flight leaves, is cancelled or its time \
passes, and a HOLD when its flight is committed; a newer request on a flight, from either desk, replaces the older.

Tools: ops_flights, ops_program, ops_aircraft, ops_check, ops_confirm, ops_request, ops_reply, end_turn \
(stand_down: true skips your remaining turns).

{COST}""",
    "crew": f"""You are the crew scheduler of JetBlue's operations control center. The ops controller, another desk, \
runs the flights and aircraft, with the FAA; you run the crews. Keep the day's cost as low as you can.

{DAY}

Each watch you take a turn first, then the ops desk, then you again if it sent you a request or a text. Your page \
lists the FAA's advisories as issued (crew_program shows a hub's programs) and the crew system's words: HOLD (a \
flight's crew is ready only later) or CANCEL (no crew can come), with the fix it chose; only you see those, unless \
you tell ops.

{CREW} You never cancel or hold flights: ask ops.

The channel: crew_request HOLD (a flight until a time), CANCEL, CANCEL_PAIR (a flight and its round-trip partner) or \
TEXT (400 characters); CANCEL_PAIR is refused for a flight with no partner. Ops's requests appear on your page \
with ids: COVER (crew a flight by a time), TEXT; crew_reply accept or decline is your word to ops on a COVER, which \
the env doesn't hold you to (your moves can follow with ref). A request lapses when its flight leaves, is cancelled \
or its time passes, and a HOLD when its flight is committed; a newer request on a flight, from either desk, replaces \
the older.

Tools: crew_flights, crew_crews, crew_reserves, crew_program, crew_options, crew_check, crew_confirm, crew_request, \
crew_reply, end_turn (stand_down: true skips your remaining turns).

{COST}""",
    "both": f"""You run JetBlue's operations control center alone: the flights and aircraft, with the FAA, and the \
crews. Keep the day's cost as low as you can.

{DAY}

You take one turn a watch. Your page has both pictures: the flights that need attention and the crew system's words, \
HOLD (a flight's crew is ready only later) or CANCEL (no crew can come), with the fix it chose.

{OPS}

{CREW}

Tools: ops_flights, ops_program, ops_aircraft, ops_check, ops_confirm, crew_flights, crew_crews, crew_reserves, \
crew_options, crew_check, crew_confirm, end_turn (stand_down: true skips your remaining turns).

{COST}""",
}
OPENING = {"ops": "You are the ops desk on {date}. Your first page follows.",
           "crew": "You are the crew desk on {date}. Your first page follows.",
           "both": "You run both desks on {date}. Your first page follows."}


def triggers(task, roles: list[str]) -> list[dict]:
    return [{"id": f"watch-{k}", "when": {"type": "action", "tool": "end_turn", "where": {"result.opened": {
             "equals": k}}}, "barrier": {"at": "provoking_call", "timeout_seconds": BARRIER_SECONDS},
             "actions": [{"type": "tool", "tool": NOTICE_TOOL, "args": {"news_id": n}} for n in news]}
            for k, news in timetable(task).items() if k > 0 and news]


def steps(task, mode: str, cap: float | None = None, env_vars: dict | None = None, turn_timeout: float | None = None,
          policy: str | None = None) -> list[dict]:
    """A day's steps: two desks (joint), one (one), or none (idle: the crew system alone)."""
    desks = DESKS.get(mode, [])
    load = {"task_id": task.task_id, "mode": mode} | ({"turn_timeout_s": turn_timeout} if turn_timeout else {})
    out = [{"id": "deploy", "type": "deploy_env", "env_id": ENV},
           {"id": "load-task", "type": "apply_server_config", "env_id": ENV,
            "directives": [{"service": ENV, "uri": LOAD_URI, "args": load}]}]
    for role, _, tools in desks:
        out += [{"id": f"{role}-only", "type": "modify_env_tool_access", "env_id": ENV, "action": "disable",
                 "role": role, "tools": ["*"]},
                {"id": f"{role}-tools", "type": "modify_env_tool_access", "env_id": ENV, "action": "enable",
                 "role": role, "tools": tools + SHARED}]
    out.append({"id": "watches", "type": "register_env_triggers", "env_id": ENV,
                "watch_roles": [r for r, _, _ in desks] or ["default"], "triggers": triggers(task, [])})
    watches = len(timetable(task))
    plays = []
    for role, name, _ in desks:
        env = ({"AIROPS_MAX_COST_USD": format(cap, "g")} if cap is not None else {}) | \
            ({"AIROPS_DESK_POLICY": policy} if policy else {}) | (env_vars or {}).get(role, {})
        out.append({"id": f"deploy-{name}", "type": "deploy_agent", "a2a_agent_id": AGENT, "agent_name": name,
                    "role": role, "env_ids": [ENV], **({"env_vars": env} if env else {})})
    out.append({"id": "clock", "type": "sync_env_clock", "env_id": ENV, "virtual_time": virtual_time(task, task.start),
                "virtual_seconds_per_real_second": 0, "tolerate_missing_sync_time": False})
    for role, name, _ in desks:
        plays.append(f"play-{name}")
        out.append({"id": f"play-{name}", "type": "prompt_agent", "agent_name": name,
                    "prompt_id": f"{task.task_id}-{mode}-{role}", "system_prompt": RULES[role],
                    "prompt": OPENING[role].format(date=task.date), "max_turns": TURNS[mode] * watches + EXTRA_TURNS,
                    "model_params": MODEL_PARAMS, "effort": EFFORT, "timeout_seconds": PLAY_SECONDS,
                    "depends_on": ["clock"], "fail_task_on_error": False})
    out += [{"id": "end-day", "type": "apply_server_config", "env_id": ENV,
             "directives": [{"service": ENV, "uri": END_DAY_URI, "args": {}}],
             **({"depends_on": plays} if plays else {})},
            {"id": "grade", "type": "env_outcome_verifier", "env_id": ENV, "file_artifact_id": VERIFIER,
             "verifier_id": "airops-occ", "score_aggregator": "weighted_average"}]
    return out


def readme() -> str:
    return f"""AirOps ops control center: a real storm day across JetBlue's whole network, run by two desks (the ops \
controller and the crew scheduler, each its own `{AGENT}` agent in its role) or by one desk running both, in hourly \
watches on the gateway's virtual clock, and graded against the best plan found in hindsight.

Every task deploys the env `{ENV}` (the bundle's envs/{ENV}, built from its Dockerfile) and loads one day of its pack
through `{LOAD_URI}`. Each desk's role sees only its own tools; one trigger per watch delivers the watch's FAA
advisories through the hidden `{NOTICE_TOOL}` when an `end_turn` opens that watch, and the clock starts frozen at
04:00. The desks play concurrently, taking turns the env grants (crew, ops, then crew again if ops asked it something).
After the play, `{END_DAY_URI}` runs whatever is left of the day and grades it; `{VERIFIER}` scores the run only if its
audit passes (news, turns, causality, roles, crew legality, the bound and each desk's presence).

- `<task id>-joint` and `<task id>-one`: the day with two desks, and with one.
- `{NOPLAY}`: the first day with no desk, the crew system alone; it needs no model.

{CREDIT}
"""


def generate(out: Path, *, task_ids: list[str] | None = None, split: str = "eval", modes: tuple[str, ...] = (
        "joint", "one"), pack: Path | None = None, cap: float | None = None, noplay: bool = False,
             env_vars: dict | None = None, turn_timeout: float | None = None, policy: str | None = None,
             suffix: str = "") -> list[str]:
    """Writes the bundle into `out`: README.md, the verifier and tasks/<task id>-<mode>.json for `task_ids`, else every
    task of `split`, in date order; with `noplay`, the first one with no desk too."""
    p = Pack(pack)
    chosen = [t for t, v in sorted(p.tasks.items(), key=lambda kv: kv[1]["date"])
              if (t in task_ids if task_ids is not None else split in ("all", v["split"]))]
    _write(out / "README.md", readme().encode())
    _write(out / "artifacts" / VERIFIER / "verify.py", (VERIFIERS / VERIFIER / "verify.py").read_bytes())
    names = []
    for task_id in chosen:
        task = p.world(task_id).task
        for mode in modes:
            name = f"{task_id}-{mode}{suffix}"
            _write(out / "tasks" / f"{name}.json", _dump(steps(task, mode, cap, env_vars, turn_timeout, policy)))
            names.append(name)
        if noplay and task_id == chosen[0]:
            _write(out / "tasks" / f"{NOPLAY}.json", _dump(steps(task, "idle")))
            names.append(NOPLAY)
    return names


def _dump(obj) -> bytes:
    return (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode()


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
