"""The ops control center's days with passengers as a folder bundle (airops-occ3): three desks (joint: crew, customer
and ops) or one agent at all three (one), played on the gateway's clock under cost P and graded by
airops-occ3-verifier."""

from pathlib import Path

from airops_core.occ.day import EXTRA_TURNS, timetable
from airops_core.occ.day3 import TURNS
from airops_core.occ.pack import Pack3
from airops_core.schedule import virtual_time

from .live import END_DAY_URI, LOAD_URI, NOTICE_TOOL
from .live3 import ENV3
from .tasks import (
    AGENT,
    BARRIER_SECONDS,
    CREW,
    CREW_TOOLS,
    DAY,
    EFFORT,
    MODEL_PARAMS,
    NOPLAY,
    OPS,
    OPS_TOOLS,
    PLAY_SECONDS,
    SHARED,
    VERIFIERS,
    _dump,
    _write,
)

VERIFIER3 = "airops-occ3-verifier"
CAPS = {"joint": 5.0, "one": 12.0}
CUST_TOOLS = ["cust_flights", "cust_flight", "cust_request"]
DESKS = {"joint": [("crew", "crew-desk", CREW_TOOLS), ("customer", "customer-desk", CUST_TOOLS),
                   ("ops", "ops-desk", OPS_TOOLS)],
         "one": [("both", "occ-desk", [t for t in OPS_TOOLS + CREW_TOOLS + CUST_TOOLS
                                       if not t.endswith(("request", "reply")) and t != "crew_program"])]}
CREDIT = ("Built from public U.S. records: BTS on-time and T-100 data, the DB1B and DB1C passenger surveys, the FAA "
          "registry and FAA Command Center advisories; crews generated from the published schedule under FAR 117 and "
          "14 CFR 121.467; passengers' itineraries generated from the surveys' connecting shares.")

PASSENGERS = """Passengers: each flight carries its booked passengers; some connect at JetBlue's connecting stations \
(JFK, BOS, FLL, MCO, SJU) onto another JetBlue flight, with a minimum connecting time of 45 min at JFK and BOS and 40 \
elsewhere. A passenger whose flight is cancelled, or who misses a connection, is rebooked automatically by the \
passenger system, on JetBlue only: the earliest arrival at the final destination first, first come first served, from \
one connecting time after the cancellation is decided (a flight 3 h or more earlier only when no other seat is left); \
who finds no seat today takes the next five days' first seats, a night away for each night, and who finds none is \
refunded. Crews riding to a duty take their seats first. No desk rebooks passengers; holds, cancellations and their \
timing decide where the passengers go."""

COST = """Cost, in passenger-minutes: each passenger's minutes of delay at the final destination (rebooked or on a \
later day included; a refunded passenger's wait); each cancelled flight's booked passengers 66 each (refunds), more \
when decided under 2 hours ahead; each night a passenger spends away, a lodging night; each refunded passenger's fare; \
the FAA's rates for aircraft time on the ground and in the air, and tonight's ferries for tomorrow; crew deadheads \
(pay and the seats), hotels away from base, and crews out of place for tomorrow. The day is graded once, on \
everything as it really happened, against the best plan found in hindsight. Decisions are taken now: none reaches back \
before this watch."""

CUSTOMER = """Your page lists the connections at risk (departures in the next 4 h whose connecting passengers make \
it with 15 min or less, or miss it: each inbound, its passengers and margin, the hold that makes them, and how many \
have no later JetBlue seat today) and the flights ops might cancel (auto-cancelling, no crew, a cancellation asked, or \
3 h late or more: where their passengers would go if cancelled now, same day, a later day or refund, and how many have \
no later seat today). cust_flights and cust_flight look passengers up."""

RULES = {
    "ops": f"""You are the ops controller of JetBlue's operations control center. The crew scheduler and the \
customer desk, two other desks, run the crews and look after the passengers; you run the flights and aircraft, with \
the FAA. Keep the day's cost as low as you can.

{DAY}

{PASSENGERS}

Each watch the crew desk takes a turn, then the customer desk, then you, then the crew desk again if you sent it a \
request or a text, then the customer desk if you texted it.

{OPS} You never name crews.

The channel: ops_request to the crew desk COVER (crew a flight by a time) or TEXT, to the customer desk TEXT (400 \
characters). Requests to you appear on your page with ids: the crew desk's HOLD (a flight until a time), CANCEL, \
CANCEL_PAIR, TEXT; the customer desk's HOLD (a departure until a time, for its connecting passengers, with how many), \
PROTECT (don't cancel a flight), SUGGEST_CANCEL (cancel one), TEXT. ops_reply accept carries a HOLD out now as your \
hold, and a CANCEL as your cancellation; PROTECT and SUGGEST_CANCEL are advice: accept or decline is your word, and \
ops_confirm acts on it. A request lapses when its flight leaves, is cancelled or its time passes, and a HOLD when its \
flight is committed; a newer request on a flight replaces the older one from the same side (the customer desk's, or \
those between the crew desk and you).

Tools: ops_flights, ops_program, ops_aircraft, ops_check, ops_confirm, ops_request, ops_reply, end_turn (stand_down: \
true skips your remaining turns).

{COST}""",
    "crew": f"""You are the crew scheduler of JetBlue's operations control center. The ops controller runs the \
flights and aircraft, with the FAA, and the customer desk looks after the passengers, two other desks; you run the \
crews. Keep the day's cost as low as you can.

{DAY}

{PASSENGERS}

Each watch you take a turn first, then the customer desk, then the ops desk, then you again if ops sent you a request \
or a text. Your page lists the FAA's advisories as issued (crew_program shows a hub's programs) and the crew system's \
words: HOLD (a flight's crew is ready only later) or CANCEL (no crew can come), with the fix it chose; only you see \
those, unless you tell ops.

{CREW} You never cancel or hold flights: ask ops.

The channel: crew_request HOLD (a flight until a time), CANCEL, CANCEL_PAIR (a flight and its round-trip partner) or \
TEXT (400 characters), to ops; CANCEL_PAIR is refused for a flight with no partner. Ops's requests appear on your \
page with ids: COVER (crew a flight by a time), TEXT; crew_reply accept or decline is your word to ops on a COVER, \
which the env doesn't hold you to (your moves can follow with ref). A request lapses when its flight leaves, is \
cancelled or its time passes, and a HOLD when its flight is committed; a newer request on a flight between ops and \
you replaces the older.

Tools: crew_flights, crew_crews, crew_reserves, crew_program, crew_options, crew_check, crew_confirm, crew_request, \
crew_reply, end_turn (stand_down: true skips your remaining turns).

{COST}""",
    "customer": f"""You are the customer desk of JetBlue's operations control center. The ops controller runs the \
flights and aircraft, with the FAA, and the crew scheduler runs the crews, two other desks; you look after the \
passengers. Keep the day's cost as low as you can.

{DAY}

{PASSENGERS}

Each watch the crew desk takes a turn, then you, then the ops desk, then the crew desk again if ops sent it a request \
or a text, then you again if ops texted you.

{CUSTOMER}

You never cancel, hold, rebook or name crews: ask ops. The channel: cust_request to ops HOLD (a departure until a \
time, for its connecting passengers; ops holds no earlier than the freeze line an hour from now, and a flight \
committed at an earlier watch can't be held), PROTECT (don't cancel a flight), SUGGEST_CANCEL (cancel one: its \
passengers are better off rebooked now) or TEXT (400 characters), each with a note if you like. Ops's answers and \
texts appear on your page with ids. A request lapses when its flight leaves, is cancelled or its time passes, and a \
HOLD when its flight is committed; your newer request on a flight replaces your older one.

Tools: cust_flights, cust_flight, cust_request, end_turn (stand_down: true skips your remaining turns).

{COST}""",
    "both": f"""You run JetBlue's operations control center alone: the flights and aircraft, with the FAA, the crews \
and the passengers. Keep the day's cost as low as you can.

{DAY}

{PASSENGERS}

You take one turn a watch. Your page has all three pictures: the flights that need attention; the crew system's words, \
HOLD (a flight's crew is ready only later) or CANCEL (no crew can come), with the fix it chose; and the passengers. \
{CUSTOMER}

{OPS}

{CREW}

Tools: ops_flights, ops_program, ops_aircraft, ops_check, ops_confirm, crew_flights, crew_crews, crew_reserves, \
crew_options, crew_check, crew_confirm, cust_flights, cust_flight, end_turn (stand_down: true skips your remaining \
turns).

{COST}""",
}
OPENING = {"ops": "You are the ops desk on {date}. Your first page follows.",
           "crew": "You are the crew desk on {date}. Your first page follows.",
           "customer": "You are the customer desk on {date}. Your first page follows.",
           "both": "You run all three desks on {date}. Your first page follows."}


def triggers(task) -> list[dict]:
    return [{"id": f"watch-{k}", "when": {"type": "action", "tool": "end_turn", "where": {"result.opened": {
             "equals": k}}}, "barrier": {"at": "provoking_call", "timeout_seconds": BARRIER_SECONDS},
             "actions": [{"type": "tool", "tool": NOTICE_TOOL, "args": {"news_id": n}} for n in news]}
            for k, news in timetable(task).items() if k > 0 and news]


def steps(task, mode: str, caps: dict | None = None, env_vars: dict | None = None, turn_timeout: float | None = None,
          policy: str | None = None) -> list[dict]:
    """A day's steps: three desks (joint), one agent at all three (one), or none (idle: the crew and passenger systems
    alone)."""
    desks = DESKS.get(mode, [])
    caps = CAPS if caps is None else caps
    load = {"task_id": task.task_id, "mode": mode} | ({"turn_timeout_s": turn_timeout} if turn_timeout else {})
    out = [{"id": "deploy", "type": "deploy_env", "env_id": ENV3},
           {"id": "load-task", "type": "apply_server_config", "env_id": ENV3,
            "directives": [{"service": ENV3, "uri": LOAD_URI, "args": load}]}]
    for role, _, tools in desks:
        out += [{"id": f"{role}-only", "type": "modify_env_tool_access", "env_id": ENV3, "action": "disable",
                 "role": role, "tools": ["*"]},
                {"id": f"{role}-tools", "type": "modify_env_tool_access", "env_id": ENV3, "action": "enable",
                 "role": role, "tools": tools + SHARED}]
    out.append({"id": "watches", "type": "register_env_triggers", "env_id": ENV3,
                "watch_roles": [r for r, _, _ in desks] or ["default"], "triggers": triggers(task)})
    watches = len(timetable(task))
    plays = []
    for role, name, _ in desks:
        cap = caps.get(mode)
        env = ({"AIROPS_MAX_COST_USD": format(cap, "g")} if cap is not None else {}) | \
            ({"AIROPS_DESK_POLICY": policy} if policy else {}) | (env_vars or {}).get(role, {})
        out.append({"id": f"deploy-{name}", "type": "deploy_agent", "a2a_agent_id": AGENT, "agent_name": name,
                    "role": role, "env_ids": [ENV3], **({"env_vars": env} if env else {})})
    out.append({"id": "clock", "type": "sync_env_clock", "env_id": ENV3, "virtual_time": virtual_time(task, task.start),
                "virtual_seconds_per_real_second": 0, "tolerate_missing_sync_time": False})
    for role, name, _ in desks:
        plays.append(f"play-{name}")
        out.append({"id": f"play-{name}", "type": "prompt_agent", "agent_name": name,
                    "prompt_id": f"{task.task_id}-{mode}-{role}", "system_prompt": RULES[role],
                    "prompt": OPENING[role].format(date=task.date), "max_turns": TURNS[mode] * watches + EXTRA_TURNS,
                    "model_params": MODEL_PARAMS, "effort": EFFORT, "timeout_seconds": PLAY_SECONDS,
                    "depends_on": ["clock"], "fail_task_on_error": False})
    out += [{"id": "end-day", "type": "apply_server_config", "env_id": ENV3,
             "directives": [{"service": ENV3, "uri": END_DAY_URI, "args": {}}],
             **({"depends_on": plays} if plays else {})},
            {"id": "grade", "type": "env_outcome_verifier", "env_id": ENV3, "file_artifact_id": VERIFIER3,
             "verifier_id": "airops-occ3", "score_aggregator": "weighted_average"}]
    return out


def readme() -> str:
    return f"""AirOps ops control center with passengers: a real storm day across JetBlue's whole network with its \
passengers, run by three desks (the crew scheduler, the customer desk and the ops controller, each its own `{AGENT}` \
agent in its role) or by one agent at all three, in hourly watches on the gateway's virtual clock, under cost P (each \
passenger's delay at the final destination, refunds, nights away and refunded fares in place of v2's per-flight delay \
and cancellation price), and graded against the best plan found in hindsight under P.

Every task deploys the env `{ENV3}` (the bundle's envs/{ENV3}, built from its Dockerfile) and loads one day of its pack
(b6-v3) through `{LOAD_URI}`. Each desk's role sees only its own tools; one trigger per watch delivers the watch's FAA
advisories through the hidden `{NOTICE_TOOL}` when an `end_turn` opens that watch, and the clock starts frozen at 04:00.
The desks play concurrently, taking turns the env grants (crew, customer, ops, then the crew desk again if ops asked it
something, then the customer desk if ops texted it). Rebooking is the passenger system's, automatic: no desk rebooks.
After the play, `{END_DAY_URI}` runs whatever is left of the day and grades it; `{VERIFIER3}` scores the run only if its
audit passes (news, turns, causality, roles, crew and passenger legality, the bound and each desk's presence).

- `<task id>-joint` and `<task id>-one`: the day with three desks, and with one agent at all three; each desk's cost
  cap $5 in a three-desk run, $12 for one agent.
- `{NOPLAY}`: the first day with no desk, the crew and passenger systems alone; it needs no model.

{CREDIT}
"""


def generate(out: Path, *, task_ids: list[str] | None = None, split: str = "eval", modes: tuple[str, ...] = (
        "joint", "one"), pack: Path | None = None, caps: dict | None = None, noplay: bool = False,
             env_vars: dict | None = None, turn_timeout: float | None = None, policy: str | None = None,
             suffix: str = "") -> list[str]:
    """Writes the bundle into `out`: README.md, the verifier and tasks/<task id>-<mode>.json for `task_ids`, else every
    task of `split`, in date order; with `noplay`, the first one with no desk too."""
    p = Pack3(pack)
    chosen = [t for t, v in sorted(p.tasks.items(), key=lambda kv: kv[1]["date"])
              if (t in task_ids if task_ids is not None else split in ("all", v["split"]))]
    _write(out / "README.md", readme().encode())
    _write(out / "artifacts" / VERIFIER3 / "verify.py", (VERIFIERS / VERIFIER3 / "verify.py").read_bytes())
    names = []
    for task_id in chosen:
        task = p.world(task_id).task
        for mode in modes:
            name = f"{task_id}-{mode}{suffix}"
            _write(out / "tasks" / f"{name}.json", _dump(steps(task, mode, caps, env_vars, turn_timeout, policy)))
            names.append(name)
        if noplay and task_id == chosen[0]:
            _write(out / "tasks" / f"{NOPLAY}.json", _dump(steps(task, "idle")))
            names.append(NOPLAY)
    return names
