"""Compute the lines pack, tasks/dock-v1-lines/, and tasks/lines/references.jsonl: the wind weeks of
tasks/dock-v1-wind/ with each ship's agent (the firm the port's 2024 record names for the call), its previous port's
distance and its ETA bias, played live through port_week_core.lines.LinesWeek.

Four players per week. The hindsight optimum knows every arrival and slows ships from the first watch it could ask,
proven optimal by CP-SAT. The follower re-plans at every watch on the arrivals as known and requests arrival times
through the agents as the env's planner would: for each ship whose window would freeze before the next watch, each
ship it plans to slow, and each ship due within 12 hours of the next watch, which could otherwise arrive early unseen
and hold up the queue; a refusal is a fact it keeps, and it probes until the ship agrees an hour, then re-plans, four
rounds at most. The silent follower re-plans the same way but never asks. The naive policy is v2's. A week's reference
cost, its anchor, is the lower of the hindsight optimum's cost and the follower's best over its four solver
configurations, as in the wind port; planners that learn arrivals as they come play close to hindsight. A week
qualifies when (a) at least 3 of the follower's
configurations finish feasible, (b) the silent follower scores at most 0.9 in at least 3 of 4, so asking the agents
matters, and (c) it has at most 10 watches.

pack writes the pack from the wind pack and the port's 2024 calls, screen plays every week four ways, and references
sets each week's anchor in the pack, replays the follower's stored plans and requests without solving and writes the
records. tools/build_lines.py runs the three on the built wind weeks, caching each week's screen, and checks the result
against the published pins. Run from envs/port_week; the screen defaults to ~/.cache/portsim-live/lines/, the rest to
packs.tasks_root().

    uv run -p 3.12 --with ortools==9.15.6755 --with-editable core python tools/lines_references.py pack [--raw PATH]
        [--out DIR]
    uv run -p 3.12 --with ortools==9.15.6755 --with-editable core python tools/lines_references.py screen
        [--tasks id,...] [--out PATH]
    uv run -p 3.12 --with ortools==9.15.6755 --with-editable core python tools/lines_references.py references
        [--screen PATH] [--out DIR]
"""

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import ortools
from berth_core import Plan, Task, plan_from_list, plan_to_list
from berth_core.check import unavoidable_cost
from port_week_core import lines, packs, schedule, wind, world
from port_week_core.lines import LinesWeek

from live_references import CONFIGS, DETERMINISTIC_SECONDS, outcome, solve
from marine_references import raw
from wind_references import pools
from wind_weather import cache_dir

SCREEN = cache_dir() / "lines/screen.jsonl"
REACHED = 3
SILENT_BAR = 0.9
MAX_WATCHES = 10
ROUNDS = 4
MARGIN = 12
QUEUE_PENALTY = 100_000
LATE_SHARE = 0.8
BIAS_SCALE = math.sqrt(math.pi / 2)
BIAS_CAP = 4.0
RAW_TIME = "%Y-%m-%d %H:%M:%S"
NGA = "https://maritimesafetyinnovationlab.org/wp-content/uploads/2015/01/pub151-distances-btw-ports.pdf"
DISTANCES = {
    "Valencia": (165, "Valencia"), "Castellon de la Plana": (165, "Valencia, a proxy"),
    "Sagunto": (165, "Valencia, a proxy"), "Alicante": (228, "Alicante"),
    "Fos-sur-Mer": (178, "Port St. Louis du Rhone, a proxy"), "Marseille": (189, "Marseille"),
    "Genoa": (356, "Genova"), "Vado Ligure": (341, "Savona, as Pub. 151 directs for Rada di Vado"),
    "La Spezia": (383, "La Spezia"), "Livorno": (386, "Livorno"), "Alger (Algiers)": (282, "Algiers"),
    "Oran": (365, "Oran"), "Bejaia (ex Bougie)": (282, "Algiers, a proxy"),
    "Skikda (ex Philippeville)": (354, "Skikda"), "Annaba (ex Bone)": (386, "Annaba"),
    "Algeciras": (516, "Gibraltar, a proxy"), "Tanger Med": (515, "Ceuta, a proxy"),
    "Casablanca": (700, "Strait of Gibraltar 534 + 166"), "Lisboa": (820, "Strait of Gibraltar 534 + 286"),
    "Gioia Tauro": (681, "Messina, a proxy"), "Marsaxlokk": (671, "Valletta, a proxy"),
    "Piraeus": (1172, "Piraievs, south of Greece"), "Aliaga": (1332, "Izmir, a proxy"), "Koper": (1296, "Koper"),
    "Damietta": (1600, "Port Said, a proxy"),
    "Anvers": (1893, "Strait of Gibraltar 534 + Ile d'Ouessant 918 + 441"), "Moerdijk": (1893, "Antwerp, a proxy"),
    "Al 'Aqabah": (1980, "Port Said 1,600 + 380, through the Suez Canal")}
"""Sea distances from Barcelona in nautical miles, NGA Pub. 151, Distances Between Ports, 11th edition (2001), no
copyright claimed; a port it doesn't list takes its nearest listed neighbour. Every ship from any other previous port
in these weeks left it before the week began, so its leg never limits the ship's slowing."""
FIRMS = {
    "MEDITERRANEAN SHIPPING COM ESPAÑA SLU": ("msc", "MSC España"),
    "NOATUM MARITIME SPAIN SAU": ("noatum", "Noatum Maritime"),
    "A PEREZ Y CIA SL": ("aperez", "A. Pérez y Cía"),
    "CMA CGM IBERICA SAU": ("cmacgm", "CMA CGM Ibérica"),
    "MARTICO SL": ("martico", "Martico"),
    "HAPAG LLOYD SPAIN SL": ("hapag", "Hapag-Lloyd Spain"),
    "GLOBAL LINER AGENCIAS IBERIA SL": ("globalliner", "Global Liner Agencias"),
    "SOLUCIONES INTEGRALES MARITIMAS SLU": ("sim", "Soluciones Integrales Marítimas"),
    "COSCO SHIPPING LINES SPAIN SA": ("cosco", "COSCO Shipping Lines Spain"),
    "OCEAN NETWORK EXPRESS SPAIN SA": ("one", "Ocean Network Express Spain"),
    "ARKAS SPAIN SA": ("arkas", "Arkas Spain"),
    "WEC LINES ESPAÑA SLU": ("wec", "WEC Lines España"),
    "BOLUDA LINES SA": ("boluda", "Boluda Lines"),
    "MARITIME AGENCY HMM SPAIN SL": ("hmm", "HMM Spain"),
    "ANTILLAS SEA AGENCY SL": ("antillas", "Antillas Sea Agency"),
    "UNIVERSAL MARITIMA SL": ("universal", "Universal Marítima"),
    "ORIENT OVERSEAS CONTAINER LINE SPAIN SL": ("oocl", "OOCL Spain"),
    "BERGE MARITIMA SL": ("berge", "Bergé Marítima")}


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def calls(body: bytes) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for row in csv.DictReader(io.StringIO(body.decode("utf-8"))):
        if row["IMO"] and row["MOLLCODI"] in ("36A", "24B"):
            out[row["IMO"]].append(row)
    return out


def agent_of(by_imo: dict[str, list[dict]], task: Task, sid: int) -> str:
    """The call's shipping agent: the 2024 call of this ship at the task's quays nearest its planned arrival, or for an
    unscheduled call, its nearest call at either quay."""
    s = task.ships[sid]
    when = datetime.strptime(task.week_start_utc, schedule.TIME_FORMAT) + timedelta(
        hours=s.planned_hour if s.planned_hour is not None else s.arrival)
    row = min(by_imo[s.imo], key=lambda r: abs(datetime.strptime(r["ETAUTC"], RAW_TIME) - when))
    return row["CONSIGNATARI"]


def bias(task_id: str, sid: int) -> float:
    """How far off the ship's agent is, as a multiple of the mean error: 80% of ships later than their ETA (a positive
    bias), the size half-normal with mean 1, capped at 4."""
    rng = random.Random(f"lines:{task_id}:{sid}")
    late = rng.random() < LATE_SHARE
    size = min(BIAS_CAP, abs(rng.gauss(0.0, BIAS_SCALE)))
    return round(size if late else -size, 3)


def with_lines(task: Task, by_imo: dict[str, list[dict]]) -> Task:
    firms, ships = {}, []
    for s in task.ships:
        firm_id, name = FIRMS[agent_of(by_imo, task, s.id)]
        firms[firm_id] = name
        leg = DISTANCES.get(s.from_port)
        ships.append({"firm": firm_id, "leg_nm": leg[0] if leg else None, "bias": bias(task.task_id, s.id)})
    return replace(task, rules=task.rules | {"lines": {"firms": dict(sorted(firms.items())), "ships": ships}})


def drift(task: Task) -> Task:
    """Each ship off its published schedule by its agent's error from 72 hours out, so the schedule is the line's
    long-range estimate rather than the truth. A delayed or bunched ship's whole story moves with it, and it arrives
    after its news and later than it would have without it; an unscheduled call keeps v4's arrival, which its notice
    announces."""
    news = lines.first_known(task)
    kinds = {s: e for e in task.disruptions if e["type"] in ("late", "bunching", "extra")
             for s in (e["ships"] if e["type"] == "bunching" else [e["ship"]])}
    metas, ships = [], []
    for s in task.ships:
        m, e = task.rules["lines"]["ships"][s.id], kinds.get(s.id)
        miss = round(lines.ETA_ERROR * m["bias"] * lines.ETA_HORIZON)
        planned = before = None
        arrival = s.arrival
        if e is None:
            planned = s.arrival
            arrival = reachable(task, s.id, max(1, planned + miss), planned)
        elif e["type"] != "extra":
            planned = s.arrival - e["hours"] if e["type"] == "late" else s.planned_hour
            before = reachable(task, s.id, max(planned + miss, news[s.id] + 2), planned)
            arrival = max(s.arrival + miss, before + 1)
        metas.append(m | {"scheduled": planned, "before_news": before, "v4_arrival": s.arrival})
        moved = arrival - s.arrival
        ships.append(replace(s, arrival=arrival, berth_deadline=None if s.berth_deadline is None else
                             s.berth_deadline + moved))
    events, notices = [], []
    for e, text in zip(task.disruptions, task.notices, strict=True):
        if e["type"] == "emergency":
            deadline = ships[e["ship"]].berth_deadline
            text = re.sub(r"docks by hour \d+", f"docks by hour {deadline}", text)
            e = e | {"deadline": deadline}
        events.append(e)
        notices.append(text)
    return replace(task, ships=ships, disruptions=events, notices=notices,
                   rules=task.rules | {"lines": task.rules["lines"] | {"ships": metas}})


def reachable(task: Task, sid: int, arrival: int, planned: int) -> int:
    """The arrival, no earlier than the slowest course from hour 0 could still make the published schedule."""
    while lines.slowest(task, sid, 0, arrival) < planned:
        arrival += 1
    return arrival


def delayed(task: Task, sid: int, hour: int) -> int:
    """A delay notice's ETA: what the ship's agent said before the news, moved later by the delay."""
    before, truth = lines.before_news(task, sid), task.ships[sid].arrival
    eta = lines.estimate(task, sid, hour, before) + truth - before
    return min(max(hour + 1, eta), lines.slowest(task, sid, hour, truth))


def news_etas(task: Task) -> Task:
    """v4's late, bunching and unscheduled-call notices as the agents' ETAs at the watch they reach the planner."""
    at = {n.event_id: w.hour for w in schedule.schedule(lines.as_v4(task)) for n in w.notices}
    events, notices = [], []
    for i, (e, text) in enumerate(zip(task.disruptions, task.notices, strict=True)):
        hour = at.get(f"{e['type']}-{i}")
        if e["type"] == "late":
            eta = delayed(task, e["ship"], hour)
            text = re.sub(r"it now arrives at hour \d+", f"its agent now gives ETA hour {eta}", text).replace(
                "was delayed leaving", "has been held up at")
            e = e | {"eta": eta}
        elif e["type"] == "extra":
            eta = lines.estimate(task, e["ship"], hour)
            e, text = e | {"eta": eta}, re.sub(r"arrives at hour \d+", f"has ETA hour {eta} from its agent", text)
        elif e["type"] == "bunching":
            etas = [delayed(task, s, hour) for s in e["ships"]]
            e, text = e | {"etas": etas}, re.sub(
                r" and now arrive together between hours \d+ and \d+ \(each ship's arrival hour is in the table\)",
                "; their agents' ETAs: "
                + ", ".join(f"ship {s} hour {x}" for s, x in zip(e["ships"], etas, strict=True)), text)
        events.append(e)
        notices.append(text)
    return replace(task, disruptions=events, notices=notices)


def lines_tasks(body: bytes) -> list[Task]:
    """The wind pack's weeks with the agents, their errors and the news as the agents' ETAs, from the 2024 calls."""
    by_imo = calls(body)
    return [news_etas(drift(with_lines(t, by_imo))) for t in wind.pack().tasks]


def write_pack(out: Path, tasks: list[Task]) -> Path:
    body = "".join(json.dumps(t.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n" for t in tasks).encode()
    out = out / packs.LINES_PACK
    out.mkdir(parents=True, exist_ok=True)
    (out / "tasks.jsonl").write_bytes(body)
    (out / "manifest.json").write_text(json.dumps(manifest(tasks, body), indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {len(tasks)} tasks to {out}")
    return out


def pack(args: argparse.Namespace) -> None:
    write_pack(args.out, lines_tasks(raw(args.raw)))


def manifest(tasks: list[Task], body: bytes) -> dict:
    return {"name": packs.LINES_PACK, "split": "eval", "tasks": len(tasks),
            "tiers": dict(sorted(Counter(t.difficulty for t in tasks).items())),
            "sha256": hashlib.sha256(body).hexdigest(), "weeks": sorted({t.week for t in tasks}),
            "quays": dict(sorted(Counter(t.quay for t in tasks).items())),
            "source": f"{packs.WIND_PACK}'s tasks with rules.lines added, each arrival off its schedule by its "
                      "agent's long-range error, and the arrival news as the agents' ETAs",
            "wind": {"pack": packs.WIND_PACK, "sha256": wind.pack().manifest["sha256"]},
            "lines": {"grounding": GROUNDING}}


GROUNDING = {
    "agents": {"value": "the shipping agent the port's 2024 record names for the call (CONSIGNATARI)",
               "basis": "Port de Barcelona open data, CC BY-SA 4.0, the 2024 calls of every ship type, pinned by "
                        "sha256 as for dock-v1-marine; an unscheduled call takes its ship's nearest 2024 call."},
    "reports": {"value": "ETAs 48 and 24 hours out and an arrival call 1 hour out, delivered at the next watch",
                "basis": "Directive 2002/59/EC art. 4 and Orden FOM/1194/2011 art. 9 require a notice at least 24 h "
                         "before arrival; the IMO Just In Time Arrival Guide (2020) puts berth planning at about 48 h; "
                         "the 2025 berthing ordinance (BOE-A-2025-5157) defines arrival by the call 1 h before buoy "
                         "S. The 48 h report and the delivery at the next watch are assumptions."},
    "eta_error": {"value": "mean absolute error 20% of the hours left, as from 72 hours out at most, 80% of ships "
                           "later than their ETA; the published schedule is the estimate from 72 hours out",
                  "basis": "Assumption, inside the published range: Busan New Port 2022-23 (agent ETAs 24-48 h out, "
                           "MAE 5 h 15 min), Antwerp 2019 (10-15 h), MSC's Barcelona-Valencia loop 2017-18 (80% "
                           "later than the leg's ETA, 60% of those within 10 h)."},
    "speeds": {"value": f"cruise {lines.CRUISE_KN} kn, slowest {lines.SLOWEST_KN} kn, slowing only after leaving "
                        "the previous port", "basis": (
        "Assumption: two-stroke engines run continuously down to 10% load (Wärtsilä); cruising speeds of 12-20 kn on "
        "MSC's Barcelona-Valencia loop; the IMO JIT container study's economical speed is 10 kn. Distances from NGA "
        f"Pub. 151 ({NGA}).")},
    "queue": {"value": "first come, first served, a ship slowed on request keeping its turn; the wind's hold is the "
                       "only exception", "basis": (
        "Barcelona's container terminals berthed first come, first served in 2024 (El Mercantil, 2024-11-05; "
        "Transporte XXI, 2024-09-09), and the 2025 ordinance made it the general rule (BOE-A-2025-5157, art. 3). "
        "Keeping the turn of a slowed ship is the virtual-arrival arrangement; the ordinance's other exceptions "
        "(technical incompatibility, occupancy thresholds, BEST's local-cargo priority) are left out.")},
    "anchor": {"value": "an hour at anchor costs the ship's sections, as an hour late does", "basis": (
        "The main-engine fuel a ship saves by slowing instead of waiting is worth 1.3-1.8 hours of its charter "
        "(Third IMO GHG Study 2014 table 14, VLSFO $626/t in 2024, Alphaliner charter rates June 2024), and an hour "
        "late costs more than charter alone, so 1 is the round value inside the evidence.")},
}


def latest(week: LinesWeek, sid: int, arrival: int) -> int:
    """The follower's estimate of the latest the ship can arrive, from what it can know: its known arrival and its
    previous port's distance."""
    leg = lines.leg_hours(week.task, sid)
    start = week.hour if leg is None else max(week.hour, arrival - leg)
    return max(arrival, math.floor(start + (arrival - start) * lines.STRETCH))


def hindsight_bounds(week: LinesWeek) -> dict[int, tuple[int, int]]:
    """Each ship's arrival range for the hindsight player: from cruising speed to the slowest it could arrive if
    asked at the first watch its agent can commit (once the ship has sailed or its ETA is within 48 hours) after its
    news, if any: an agreement made before a ship's news is the plan as it stood."""
    first = lines.first_known(week.task)
    hours = [w.hour for w in week.watches]
    out = {}
    for s in week.task.ships:
        asked = next((h for h in hours if h >= first.get(s.id, 0) and lines.commits(week.task, s.id, h, s.arrival)),
                     None)
        out[s.id] = (s.arrival, s.arrival if asked is None or asked >= s.arrival else
                     week.reach(s.id, asked, s.arrival)[1])
    return out


def extra_for(view: Task, fixed: Plan, bounds: dict[int, tuple[int, int]], turns: dict[int, int], base,
              agreed: dict | None = None):
    """Each open ship's arrival within its bounds and berthing at or after it, the hours between them charged; a ship
    that starts while a ship earlier in the queue has arrived and waits is penalised, so the plans keep the queue. Given
    ``agreed``, the queue protects only the ships that arrive later than at cruising speed, whose arrival must be
    agreed, and ``agreed`` gets each ship's (arrival, agreed) variables."""
    def add(md, ships):
        base(md, ships)
        terms, arrive, guard = [], {}, {}
        for s in view.ships:
            lo, hi = bounds[s.id]
            guard[s.id] = agreed is None
            if s.id in fixed:
                arrive[s.id] = lo
                continue
            t = ships[s.id][0]
            p = md.NewIntVar(lo, hi, f"a{s.id}") if hi > lo else lo
            md.Add(t >= p)
            wait = md.NewIntVar(0, 10_000, f"wait{s.id}")
            md.Add(wait == t - p)
            terms.append(s.sections * wait)
            arrive[s.id] = p
            if agreed is not None and hi > lo:
                guard[s.id] = md.NewBoolVar(f"agreed{s.id}")
                md.Add(p >= lo + 1).OnlyEnforceIf(guard[s.id])
                md.Add(p == lo).OnlyEnforceIf(guard[s.id].Not())
        if agreed is not None:
            agreed.update({s: (arrive[s], guard[s]) for s in arrive})
        ids = [s.id for s in view.ships]
        for x in ids:
            for y in ids:
                if turns[x] < turns[y] and not (x in fixed and y in fixed) and guard[x] is not False:
                    tx, ty = ships[x][0], ships[y][0]
                    jump, there, broken = md.NewBoolVar(""), md.NewBoolVar(""), md.NewBoolVar("")
                    md.Add(ty < tx).OnlyEnforceIf(jump)
                    md.Add(ty >= tx).OnlyEnforceIf(jump.Not())
                    md.Add(ty >= arrive[x]).OnlyEnforceIf(there)
                    md.Add(ty < arrive[x]).OnlyEnforceIf(there.Not())
                    md.AddBoolOr([jump.Not(), there.Not(), broken, *([] if guard[x] is True else [guard[x].Not()])])
                    terms.append(QUEUE_PENALTY * broken)
        return terms
    return add


def lines_solve(view: Task, fixed: Plan, before: int, hint: Plan, workers: int, tiebreak: bool,
                bounds: dict[int, tuple[int, int]], turns: dict[int, int], optimal: bool = False) -> Plan:
    return solve(view, fixed, before, hint, workers, tiebreak,
                 extra=extra_for(view, fixed, bounds, turns, pools(view, fixed)), optimal=optimal,
                 forecast_relief=not optimal)


def hindsight(task: Task) -> tuple[int, Plan, dict[int, int]]:
    """The best week in hindsight, each ship asked at the first watch its agent can commit: the queue protects only the
    ships it slows, as the port's does only the ships whose arrival was agreed."""
    week = LinesWeek(task)
    bounds = hindsight_bounds(week)
    turns = {s.id: s.arrival for s in task.ships}
    truth = lines.with_arrivals(task, turns, turns)
    chosen, values = {}, {}
    plan = solve(truth, {}, 0, plan_from_list(task.reference["optimal_plan"]), 1, False,
                 extra=extra_for(truth, {}, bounds, turns, pools(truth, {}), chosen), optimal=True,
                 solved=lambda solver: values.update({s: (solver.Value(p), solver.Value(g))
                                                      for s, (p, g) in chosen.items()}))
    arrivals = {s.id: values[s.id][0] for s in task.ships}
    graded = lines.evaluate(lines.with_arrivals(task, arrivals, turns, [s for s, (_, g) in values.items() if g]), plan)
    assert graded.feasible, task.task_id
    return graded.cost, plan, arrivals


def planning(week: LinesWeek, low: dict[int, int], high: dict[int, int], ask: bool) -> tuple[Task, dict, dict]:
    """The follower's view: each ship's arrival at least what it has learned, a frozen ship's no later than its
    window; its bounds and its turns as known."""
    view, fixed = week.view, week.fixed
    arrivals, bounds = {}, {}
    for s in view.ships:
        a = max(s.arrival, low.get(s.id, s.arrival))
        if s.id in fixed:
            a = min(a, fixed[s.id][0])
        settled = s.id in week.locks or s.id in week.arrived or s.id in fixed or not ask
        hi = a if settled else min(high.get(s.id, 10**6), latest(week, s.id, a))
        arrivals[s.id], bounds[s.id] = a, (a, max(a, hi))
    turns = {s.id: week.known_turn(view, s.id) for s in view.ships}
    return lines.with_arrivals(view, arrivals, turns), bounds, turns


def lock(week: LinesWeek, sid: int, want: int, low: dict[int, int], high: dict[int, int]) -> int | None:
    """Request ``want``, and on a counter-offer request the hour offered. Each offer is a fact the follower keeps;
    returns the agreed hour, or None, as when the ship hasn't sailed yet."""
    firm = lines.firm(week.task, sid)
    reply = week.request(firm, sid, want)
    if reply["agreed"]:
        return want
    if "earliest" in reply:
        low[sid] = offer = reply["earliest"]
    elif "latest" in reply:
        high[sid] = offer = reply["latest"]
    else:
        if reply["reason"].startswith("already arrived"):
            week.arrived.add(sid)
        return None
    return offer if week.request(firm, sid, offer)["agreed"] else None


def follower(workers: int, tiebreak: bool, plans: list[list[dict]], ask: bool = True) -> world.Policy:
    low: dict[int, int] = {}
    high: dict[int, int] = {}

    heard: set[int] = set()

    def policy(week: LinesWeek) -> Plan:
        for sid in week.news:
            if sid not in heard and not week.pending(sid):
                heard.add(sid)
                low.pop(sid, None)
                high.pop(sid, None)
        upcoming = week.watches[week.watch + 1].hour if not week.last else 10**6
        hint: Plan = week.plan if week.watch else {}
        for attempt in range(ROUNDS + 1):
            view, bounds, turns = planning(week, low, high, ask)
            plan = lines_solve(view, week.fixed, week.before, hint, workers, tiebreak, bounds, turns)
            hint = plan
            if not ask or attempt == ROUNDS:
                break
            moved = False
            for sid, (start, _, _) in sorted(plan.items()):
                if sid in week.fixed or sid in week.locks or sid in week.arrived:
                    continue
                lo, hi = bounds[sid]
                if start < upcoming + schedule.FREEZE_HOURS or start > lo or lo < upcoming + MARGIN:
                    want = min(start, hi)
                    moved |= lock(week, sid, want, low, high) != start
            if not moved:
                break
        plans.append(plan_to_list(plan))
        return plan
    return policy


def played(task: Task, policy: world.Policy) -> LinesWeek:
    week = world.play(task, policy, week=LinesWeek)
    assert week.refusals == [], week.refusals
    return week


def configs(task: Task, ask: bool) -> list[tuple[dict, LinesWeek | None, list[list[dict]]]]:
    out = []
    for workers, tiebreak in CONFIGS:
        row, plans = {"workers": workers, "tiebreak": tiebreak}, []
        try:
            out.append((row, played(task, follower(workers, tiebreak, plans, ask)), plans))
        except RuntimeError as e:
            out.append((row | {"cost": None, "reward": None, "error": str(e)}, None, plans))
    return out


def regraded(week: LinesWeek, task: Task) -> dict:
    final = LinesWeek(task)
    final.__dict__.update({k: v for k, v in week.__dict__.items() if k not in ("task", "grade")})
    final.finish(week.end_reason)
    return outcome(final.grade) | {"anchor_hours": final.grade["anchor_hours"]}


def summary(runs, task: Task) -> tuple[dict | None, list[dict]]:
    first, rows = None, []
    for k, (row, week, plans) in enumerate(runs):
        if week is not None:
            graded = regraded(week, task)
            row = row | {"cost": graded["cost"], "reward": graded["reward"], "anchor_hours": graded["anchor_hours"],
                         "messages": len(week.exchanges),
                         "most_messages_in_a_watch": max(Counter(e["watch"] for e in week.exchanges).values(),
                                                         default=0)}
            if first is None:
                first = {"config": k, **graded, "plans": plans,
                         "exchanges": [{"watch": e["watch"], "ship": e["ship"], "arrival": e.get("arrival"),
                                        "agreed": e["reply"].get("agreed")} for e in week.exchanges]}
        rows.append(row)
    return first, rows


def screened(task: Task) -> dict:
    hours = [w.hour for w in lines.timetable(task)]
    row = {"task_id": task.task_id, "watch_hours": hours, "ships": len(task.ships),
           "firms": len(task.rules["lines"]["firms"])}
    started = time.monotonic()
    try:
        best, plan, arrivals = hindsight(task)
    except RuntimeError as e:
        return row | {"error": str(e), "qualifies": False}
    follow = configs(task, ask=True)
    costs = [w.grade["cost"] for _, w, _ in follow if w is not None and w.grade["cost"] is not None]
    if not costs:
        return row | {"hindsight_cost": best, "error": "no follower configuration finished feasible",
                      "configs": [r for r, _, _ in follow], "qualifies": False}
    anchor = min([best, *costs])
    final = replace(task, reference=task.reference | {"optimal_cost": anchor, "optimal_plan": plan_to_list(plan),
                                                      "lower_bound": best, "proven_optimal": False,
                                                      "hindsight_arrivals": [arrivals[s.id] for s in task.ships]})
    rolled, rows = summary(follow, final)
    silent_first, silent_rows = summary(configs(task, ask=False), final)
    naive = played(final, world.naive)
    clauses = {"a": sum(c["cost"] is not None for c in rows) >= REACHED,
               "b": sum(c["reward"] is not None and c["reward"] <= SILENT_BAR for c in silent_rows) >= REACHED,
               "c": len(hours) <= MAX_WATCHES}
    return row | {"optimal_cost": anchor, "hindsight_cost": best,
                  "hindsight_arrivals": [arrivals[s.id] for s in task.ships],
                  "wind_optimal_cost": task.reference["optimal_cost"],
                  "unavoidable_cost": unavoidable_cost(task), "rolling": rolled, "configs": rows,
                  "silent": (silent_first or {}) | {"configs": silent_rows},
                  "naive": outcome(naive.grade) | {"anchor_hours": naive.grade["anchor_hours"]},
                  "clauses": clauses, "qualifies": all(clauses.values()),
                  "solver": {"ortools": ortools.__version__, "max_deterministic_time": DETERMINISTIC_SECONDS},
                  "hindsight_plan": plan_to_list(plan), "seconds": round(time.monotonic() - started)}


def one(task_id: str) -> dict:
    return screened(lines.pack().get(task_id))


def week(task: dict) -> tuple[dict, float]:
    """The week's screen row and how long it took, from its task as a dict, for a process pool."""
    started = time.monotonic()
    return screened(Task.from_dict(task)), time.monotonic() - started


def screen(args: argparse.Namespace) -> None:
    ids = args.tasks.split(",") if args.tasks else [t.task_id for t in lines.pack().tasks]
    done = {r["task_id"] for r in read(args.out)} if args.out.is_file() else set()
    todo = [i for i in ids if i not in done]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(min(len(todo), os.cpu_count() or 1) or 1) as pool, args.out.open("a") as out:
        for future in as_completed([pool.submit(one, i) for i in todo]):
            row = future.result()
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(f"{row['task_id']}: " + (f"error {row['error']}" if "error" in row else (
                f"hindsight {row['hindsight_cost']}, anchor {row['optimal_cost']}, follow "
                f"{[c['cost'] for c in row['configs']]} msgs {[c.get('messages') for c in row['configs']]}, silent "
                f"{[c['reward'] for c in row['silent']['configs']]}, naive {row['naive']['reward']}, "
                f"clauses {row['clauses']}, {row['seconds']}s")), flush=True)


def references(args: argparse.Namespace) -> None:
    write_references(args.out, lines.pack().tasks, {r["task_id"]: r for r in read(args.screen)})


def write_references(out: Path, tasks: list[Task], rows: dict[str, dict]) -> Path:
    """The pack with each week's anchor, and the references: the screen rows but for the hindsight plan, which the
    pack holds, and the solve's wall time, so that two builds write the same bytes."""
    out_tasks, lines_out = [], []
    for task in tasks:
        row = rows[task.task_id]
        final = replace(task, reference=task.reference | {
            "optimal_cost": row["optimal_cost"], "optimal_plan": row["hindsight_plan"],
            "lower_bound": row["hindsight_cost"], "proven_optimal": False})
        out_tasks.append(final)
        stored = row["rolling"]
        assert replayed(final, stored) == outcome(stored) | {"anchor_hours": stored["anchor_hours"]}, task.task_id
        lines_out.append(json.dumps({k: v for k, v in row.items() if k not in ("hindsight_plan", "seconds")}) + "\n")
        print(f"{task.task_id}: replayed, qualifies {row['qualifies']}")
    body = "".join(json.dumps(t.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n"
                   for t in out_tasks).encode()
    pack_out = out / packs.LINES_PACK
    pack_out.mkdir(parents=True, exist_ok=True)
    (pack_out / "tasks.jsonl").write_bytes(body)
    (pack_out / "manifest.json").write_text(json.dumps(manifest(out_tasks, body), indent=2, ensure_ascii=False) + "\n")
    path = out / "lines/references.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines_out))
    print(f"Wrote {len(lines_out)} references to {path}")
    return path


def replayed(task: Task, stored: dict) -> dict:
    """The follower's stored plans and requests played again, without a solver."""
    exchanges = stored["exchanges"]

    def policy(week: LinesWeek) -> Plan:
        for e in exchanges:
            if e["watch"] == week.watch:
                week.request(lines.firm(task, e["ship"]), e["ship"], e["arrival"])
        return plan_from_list(stored["plans"][week.watch])
    week = played(task, policy)
    return outcome(week.grade) | {"anchor_hours": week.grade["anchor_hours"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    one_cmd = commands.add_parser("pack", help="write the pack from the wind pack and the port's 2024 calls")
    one_cmd.add_argument("--raw", type=Path)
    one_cmd.add_argument("--out", type=Path, default=packs.tasks_root())
    one_cmd.set_defaults(run=pack)
    one_cmd = commands.add_parser("screen", help="play every week four ways")
    one_cmd.add_argument("--tasks", help="comma-separated task ids; default: every one")
    one_cmd.add_argument("--out", type=Path, default=SCREEN)
    one_cmd.set_defaults(run=screen)
    one_cmd = commands.add_parser("references", help="set each week's anchor, replay and write the references")
    one_cmd.add_argument("--screen", type=Path, default=SCREEN)
    one_cmd.add_argument("--out", type=Path, default=packs.tasks_root())
    one_cmd.set_defaults(run=references)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
