"""A date's crews: its trips over the published schedule, its crew units at the start, the crew replay of JetBlue's real
day and the crew-aware calm day.

- Trips (pairings.py) are built once per date; every duty of the date is a unit, based where it starts or ends (a base),
  else at the base nearest its start (labelled), and the reserves follow the units based there (crews.reserves).
- The storm day's units at the start come from a crew replay of the record before the start with the rest of the day as
  scheduled (crewreplay.py): the legs JetBlue flew before the start at their recorded times and blocks, and the task's
  flights at their scheduled departures (no earlier than the start) and blocks, as the airline planned the day from the
  record so far; every duty of the date from its report and the reserves from their periods; the fewest legs flown
  before the start without a legal crew first, then the fewest of the rest of the day (AHEAD each), then crew cost with
  SWAP for each leg flown off a crew's own trip (the airline keeps its trips unless one breaks; a reserve's every leg
  counts). It prefers real layovers: crews sit where the aircraft overnighted. Where the record has fewer aircraft of a
  type group at a station at NIGHT than the planned rotations put there, as many of the crews starting there from NIGHT
  on (the earliest first) may start where the record has the group's aircraft beyond the planned ones instead (the
  nearest, by an assignment), and staying costs LAYOVER, a preference under any leg's crew. Each unit stands at the
  start where that replay has it, and its trip from the start is the replay's plan for it. So a leg cancelled before the
  start strands no crew it need not, a crew a recorded overrun timed out hands its legs to another, and no crew is sent
  where the day goes on that could not go on. A unit is off when it was released before the start with nothing left, and
  so is a reserve whose period or shift is over unused. A short-call reserve the replay plans to call after the start
  for its legs from then reports when it calls it, so those legs keep a crew that comes. At the tasks' 04:00 start
  (network.START) the record before it is the few legs flown by then, red-eyes and the first departures: a crew on one
  carries its duty so far, and every other crew starts rested where its trip reports.
- The calm day's units stand where the schedule assumes them, as network.scheduled stands its aircraft: every leg
  scheduled before the start flown on schedule. A leg scheduled before the start that left after it in the record leaves
  at the start in the calm world; its duty is split there (labelled): a crew flies those legs from their origin at the
  start, its duty so far as scheduled, and the rest of the trip has its crew where the schedule has it, the one crew
  more taken from its base's reserves (short-call first; none left, none taken).
- The crew replay of JetBlue's real day: its flown flights at their recorded departures and blocks (its cancellations
  out), crewed by the units at the start, their reserves and every move, the fewest flights without a legal crew first
  (crewreplay.solve: a dive that always ends in an integer plan, with its LP and Lagrangian bounds).
- The crew-aware calm day is crewplan.calm_day's.
"""

import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from ..build import fleet, pack
from ..core.cost import miles
from ..core.model import Airport
from . import candidates, crewreplay, crews, crewsim, network, pairings
from .crewreplay import DH, OP, Column, Leg, Problem
from .crews import Unit
from .model import Task

AHEAD = crewreplay.UNCREWED / 100
SWAP = 1000
LAYOVER = AHEAD / 2
NIGHT = 240


@dataclass
class World:
    """A date's crew world: the task, its BTS context, its trips, the airports any crew can reach and the scheduled
    legs the trips cover, with their planned rotations."""

    task: Task
    ctx: candidates.Context
    built: dict[tuple[str, str], pairings.Built]
    airports: dict[str, Airport]
    legs: list[pairings.Leg]

    @property
    def duties(self) -> list[pairings.Duty]:
        return [d for b in self.built.values() for d in b.duties]

    def local(self, station: str, minute: int) -> int:
        return network.local(self.task, station, minute)

    def rows(self) -> pd.DataFrame:
        day = self.ctx.day
        return pd.concat([day.prev, day.cur])

    def flight_of(self) -> dict[int, int]:
        """Each BTS row of the task's flights, to its flight."""
        by_key = {(f.number, f.orig, f.sched_dep): f.id for f in self.task.flights}
        out = {}
        for r in self.rows().itertuples():
            k = (int(r.Flight_Number_Reporting_Airline), r.Origin, int(r.sd))
            if k in by_key:
                out[int(r.Index)] = by_key[k]
        return out


def world(task: Task, inputs, derived, time_limit: float = 600.0) -> World:
    ctx = candidates.context(inputs, derived, task.date, task.hub)
    legs = pairings.legs_of(ctx.day, pairings.schedule(ctx, inputs, derived), ctx.families)
    built = pairings.build_all(legs, task.date, time_limit)
    codes = {g.orig for g in legs} | {g.dest for g in legs} | set(task.airports) | \
        {b for k in crews.KINDS for b, _ in crews.BASES[k]}
    return World(task, ctx, built, candidates.airports_of(codes), legs)


def home(d: pairings.Duty, date: str, airports: Mapping[str, Airport]) -> str:
    bases = crews.bases(d.kind, date)
    if d.start in bases:
        return d.start
    if d.end in bases:
        return d.end
    return min(sorted(bases), key=lambda b: miles(airports[d.start], airports[b]))


def day_units(w: World) -> list[Unit]:
    """Every duty of the date as a unit at its report, keyed by BTS rows, free to leave once reported (the record's
    early departures included)."""
    out = []
    for n, d in enumerate(sorted(w.duties, key=lambda d: (d.kind, crews.GROUPS.index(d.group), d.report, d.legs))):
        out.append(Unit(f"{d.kind[0].upper()}-{d.group}-{n:04d}", d.kind, d.group, crews.size(d.kind, d.group),
                        home(d, w.task.date, w.airports), d.end, d.start, d.report, d.report, d.hour,
                        at=d.start, trip=d.legs, deadheads=d.deadheads))
    return out


def reserves(w: World, units: Sequence[Unit], share: str = "middle") -> list[Unit]:
    working = Counter((u.kind, u.group, u.home) for u in units if not u.reserve)
    return crews.reserves(working, w.task.date, w.local, share)


def family_seats(task: Task) -> dict[str, int]:
    out: dict[str, int] = {}
    for t in task.tails:
        out[t.family] = min(out.get(t.family, t.seats), t.seats)
    return out


def recorded_legs(w: World) -> list[Leg]:
    """The legs JetBlue flew before the start (any typed tail), at their recorded times; a leg of a duty of the date
    needs a crew of that duty's kind."""
    task, day = w.task, w.ctx.day
    need = defaultdict(set)
    for d in w.duties:
        for k in d.legs:
            need[k].add(d.kind)
    first = min(d.report for d in w.duties)
    seats = family_seats(task)
    out = []
    for r in w.rows().itertuples():
        if r.Cancelled != 0 or r.Tail_Number not in w.ctx.families or r.Tail_Number not in day.fleet:
            continue
        if not first <= r.ad < task.start:
            continue
        fam = w.ctx.families[r.Tail_Number]
        cap = seats.get(fam, day.fleet[r.Tail_Number].seats)
        pax = min(fleet.pax(day.fleet[r.Tail_Number].seats, day.lf, r.Origin, r.Dest), cap)
        out.append(Leg(int(r.Index), f"{pack.CARRIER} {int(r.Flight_Number_Reporting_Airline)}", r.Origin, r.end,
                       int(r.ad), network.landed(r), int(r.CRSElapsedTime),
                       crews.TYPE[fam], max(0, cap - pax), tuple(k for k in crews.KINDS if k in need[int(r.Index)])))
    return out


def task_legs(task: Task, deps: Mapping[int, int], blocks: Mapping[int, int], ends: Mapping[int, str] | None = None
              ) -> list[Leg]:
    """The task's flights leaving at `deps` with `blocks`, every one needing a crew."""
    seats = family_seats(task)
    return [Leg(f.id, f.flight, f.orig, (ends or {}).get(f.id, f.dest), d, d + blocks[f.id], f.sched_block,
                crews.TYPE[f.family], max(0, seats[f.family] - f.pax)) for f in task.flights
            if (d := deps.get(f.id)) is not None]


def storm_units(w: World, share: str = "middle", time_limit: float = 900.0, polish: float = 300.0,
                log=None) -> tuple[list[Unit], dict]:
    """The units at the start from a crew replay of the record before it with the rest of the day as scheduled, and
    that replay's summary."""
    task = w.task
    flight_of = w.flight_of()
    units = day_units(w)
    units += reserves(w, units, share)
    keyed = [replace(u, trip=tuple(ahead(flight_of[k]) if k in flight_of else k for k in u.trip),
                     deadheads=tuple(ahead(flight_of[k]) if k in flight_of else k for k in u.deadheads))
             for u in units]
    later = [replace(g, id=ahead(g.id), price=AHEAD) for g in
             task_legs(task, {f.id: max(f.sched_dep, task.start) for f in task.flights},
                       {f.id: f.sched_block for f in task.flights})]
    real = layovers(w, keyed)
    p = Problem(recorded_legs(w) + later, keyed, task.date, w.local, task_end(w), foreign=SWAP, starts=real,
                ends=crews.Ends(keyed, w.airports, task.date))
    t0 = time.monotonic()
    sol = crewreplay.solve(p, time_limit, polish=polish, log=log)
    out = [s for u in keyed if (s := at_start(p, u, sol.columns.get(u.id), task.start)) is not None]
    before = [i for i in sol.uncrewed if i >= 0]
    taken = sum(c.start == real[uid][0][0] for uid, c in sol.columns.items() if uid in real)
    return out, {**sol.summary(), "real_layovers": {"offered": len(real), "taken": taken},
                 "legs": sum(g.id >= 0 for g in p.legs),
                 "needed": sum(bool(g.need) for g in p.legs if g.id >= 0), "units": len(units),
                 "uncrewed_before": len(before), "uncrewed_after_as_scheduled": len(sol.uncrewed) - len(before),
                 "seconds_total": round(time.monotonic() - t0, 1),
                 "causes": [c for c in uncrewed_causes(p, sol) if c["id"] >= 0]}


def layovers(w: World, units: Sequence[Unit]) -> dict[str, tuple[tuple[str, float], ...]]:
    """The units that may start where the record has the aircraft instead (see the module doc): each with that station
    first, preferred, and its own at LAYOVER."""
    planned, real = rotations_at(w.legs, NIGHT), aircraft_at(w, NIGHT)
    out = {}
    for kind in crews.KINDS:
        for g in crews.GROUPS:
            short = {s: n for (k, s), n in (planned - real).items() if k == g}
            spare = {s: n for (k, s), n in (real - planned).items() if k == g}
            sent = crews.transport(short, spare, lambda a, b: miles(w.airports[a], w.airports[b]), most=True)
            laid = defaultdict(list)
            for u in sorted(units, key=lambda u: (u.ready, u.id)):
                if u.kind == kind and u.group == g and not u.reserve and u.station in short and \
                        u.report >= NIGHT - crews.REPORT:
                    laid[u.station].append(u)
            for (a, b), n in sorted(sent.items()):
                for u in laid[a][:n]:
                    out[u.id] = ((b, 0.0), (a, LAYOVER))
                laid[a] = laid[a][n:]
    return out


def rotations_at(legs: Sequence[pairings.Leg], t: int) -> Counter:
    """Per (type group, station), the planned rotations flying from minute t on, where each has its aircraft at t: the
    destination of its last leg leaving before t, else its first leg's origin."""
    by = {g.key: g for g in legs}
    turned = {g.after for g in legs if g.after is not None}
    out: Counter = Counter()
    for head in (g for g in legs if g.key not in turned):
        chain = [head]
        while chain[-1].after is not None:
            chain.append(by[chain[-1].after])
        if chain[-1].sd < t:
            continue
        before = [g for g in chain if g.sd < t]
        out[(head.group, before[-1].dest if before else chain[0].orig)] += 1
    return out


def aircraft_at(w: World, t: int) -> Counter:
    """Per (type group, station), the typed aircraft the record has there at minute t, of those with a leg from t on
    (flown or not): where its last leg flown before t ended, else its first flown leg's origin, else its first leg's."""
    rows = w.rows()
    rows = rows[rows.Tail_Number.isin(w.ctx.families) & rows.Tail_Number.isin(w.ctx.day.fleet)]
    out: Counter = Counter()
    for tail, x in rows.groupby("Tail_Number"):
        if not ((x.sd >= t) | (x.ad >= t)).any():
            continue
        flown = x[(x.Cancelled == 0) & x.ad.notna()].sort_values("ad")
        before = flown[flown.ad < t]
        at = before.iloc[-1].end if len(before) else (flown if len(flown) else x.sort_values("sd")).iloc[0].Origin
        out[(crews.TYPE[w.ctx.families[tail]], at)] += 1
    return out


def ahead(i: int) -> int:
    """A task flight's leg id in the replay of the record before the start, apart from the record's rows."""
    return -1 - i


def at_start(p: Problem, u: Unit, c: Column | None, start: int) -> Unit | None:
    """Unit u after the steps of its column before the start, its trip from the start the column's legs from then (the
    replay's plan for the rest of the day), a short-call reserve the plan calls after the start reporting when it calls
    it; None when it is off: released before the start with nothing left, or a reserve whose period or shift ended
    unused."""
    steps = () if c is None else c.steps
    trip = tuple(ahead(i) for i, how in steps if i < 0 and how == OP)
    deadheads = tuple(ahead(i) for i, how in steps if i < 0 and how == DH)
    began = u.station if c is None else p.began(u, c)
    s = {"station": began, "ready": u.ready, "report": u.report, "hour": u.hour, "flown": u.flown,
         "minutes": u.minutes, "landed": u.landed, "at": u.at, "until": u.until}
    if began != u.station:
        s.update(hour=p.hour(began, u.report), at=began)
    moved = False
    for n, (i, how) in enumerate(steps):
        if i < 0:
            break
        if n == 0 and c.report is not None:
            s.update(report=c.report, hour=p.hour(u.home, c.report))
        if c.rest is not None and n == len(c.first):
            s.update(report=c.rest[1], hour=p.hour(*c.rest), flown=0, minutes=0, at=c.rest[0])
        g = p.leg[i]
        s.update(station=g.dest, ready=g.arr + crews.CONNECT, landed=g.arr)
        if how == OP:
            s.update(flown=s["flown"] + 1, minutes=s["minutes"] + g.block)
        moved = True
    if moved and u.reserve == "standby":
        s["until"] = None
    if u.reserve == "short" and not moved and trip and c.report is not None:
        s.update(report=c.report, hour=p.hour(u.home, c.report), ready=c.report + crews.BRIEF, at=u.home)
    if u.reserve and not moved and (u.until is None or u.until < start):
        return None
    if not u.reserve and not trip and not deadheads and s["landed"] is not None and \
            s["landed"] + crews.DEBRIEF < start:
        return None
    return replace(u, trip=trip, deadheads=deadheads, **s)


def calm_units(w: World, share: str = "middle") -> list[Unit]:
    """The calm day's units at the start, as the schedule assumes them (see the module doc)."""
    task = w.task
    units = day_units(w)
    spare = reserves(w, units, share)
    flight_of = w.flight_of()
    rows = w.rows()
    out = []
    for u in units:
        steps = sorted([(k, OP) for k in u.trip] + [(k, DH) for k in u.deadheads], key=lambda s: rows.at[s[0], "sd"]
                       if s[0] in rows.index else -10 ** 6)
        state = {"station": u.station, "ready": u.ready, "flown": 0, "minutes": 0, "landed": None}
        pending: list[int] = []
        before = None
        trip, deadheads = [], []
        for k, how in steps:
            f = task.flight(flight_of[k]) if k in flight_of else None
            if f is not None and f.sched_dep >= task.start:
                (trip if how == OP else deadheads).append(f.id)
                continue
            if f is not None and how == OP:
                if before is None:
                    before = dict(state)
                pending.append(f.id)
            if k in rows.index:
                r = rows.loc[k]
                state = {"station": r.Dest, "ready": int(r.sa) + crews.CONNECT, "landed": int(r.sa),
                         "flown": state["flown"] + (how == OP),
                         "minutes": state["minutes"] + int(r.CRSElapsedTime) * (how == OP)}
        if not trip and not deadheads and not pending:
            continue
        if trip or deadheads:
            out.append(replace(u, trip=tuple(trip), deadheads=tuple(deadheads), **state))
        if pending:
            first = task.flight(pending[0])
            out.append(replace(u, id=u.id + "p", station=first.orig, ready=max(task.start, before["ready"]),
                               flown=before["flown"], minutes=before["minutes"], landed=before["landed"],
                               trip=tuple(pending), deadheads=(), end=task.flight(pending[-1]).dest))
            if trip or deadheads:
                take = [r for r in spare if r.kind == u.kind and r.group == u.group and r.home == u.home]
                take.sort(key=lambda r: (r.reserve != "short", r.id))
                if take:
                    spare.remove(take[0])
    out += [r for r in spare if r.until is not None and r.until >= task.start]
    return out


def task_end(w: World) -> Callable[[Unit, str], float]:
    return lambda u, station: crews.reposition(u, station, w.airports, w.task.date)


def replay_record(w: World, units: Sequence[Unit], time_limit: float = 1800.0, polish: float = 300.0,
                  log=None) -> tuple[Problem, crewreplay.Solved]:
    """JetBlue's real day crewed: its flown flights at their recorded departures and blocks."""
    task = w.task
    deps, blocks, ends = recorded(task)
    p = Problem(task_legs(task, deps, blocks, ends), units, task.date, w.local, task_end(w), call=task.start,
                ends=crews.Ends(units, w.airports, task.date))
    return p, crewreplay.solve(p, time_limit, polish=polish, log=log)


def recorded(task: Task) -> tuple[dict[int, int], dict[int, int], dict[int, str]]:
    """The flown flights' recorded departures, blocks and where they ended (a diversion that never got there, an hour
    after its scheduled block)."""
    deps, blocks, ends = {}, {}, {}
    for f in task.flights:
        a = f.actual
        if a.dep is not None:
            deps[f.id], ends[f.id] = a.dep, a.end
            blocks[f.id] = (a.arr if a.arr is not None else a.dep + f.sched_block + 60) - a.dep
    return deps, blocks, ends


def plan_of(p: Problem, sol: crewreplay.Solved, base: Mapping[int, Mapping] | None = None) -> dict:
    """The crew replay's day as a crew sim plan: each flight's crews and riders on top of `base`, and the reports."""
    out: dict = {i: dict(v) for i, v in (base or {}).items()}
    reports = {}
    for c in sol.columns.values():
        u = p.unit[c.unit]
        for i, how in c.steps:
            entry = out.setdefault(i, {})
            if how == OP:
                entry.setdefault("crew", {})[u.kind] = u.id
            else:
                entry.setdefault("deadhead", []).append(u.id)
        if c.report is not None:
            reports[u.id] = c.report
        if c.rest is not None:
            reports[f"{u.id}#2"] = c.rest[1]
    out["reports"] = reports
    return out


def columns_of(p: Problem, day: crewsim.CrewDay) -> dict[str, Column] | None:
    """A crew sim day's crews as the replay's columns over `p`'s legs (each unit that moved, its duty periods, a
    called reserve's report, its rest), or None if it flew a leg at a minute `p` lacks."""
    out = {}
    for uid, rec in day.units.items():
        periods = [[], []]
        report = rest = None
        for d in rec["duties"]:
            if d["begins"] == "call":
                report = d["report"]
            if d["begins"] == "rest":
                rest = (d["at"], d["report"])
            steps = [(t, i, OP) for i, t, _ in d["legs"]] + [(t, i, DH) for i, t, _ in d["deadheads"]]
            if any(i not in p.leg or p.leg[i].dep != t for t, i, _ in steps):
                return None
            periods[d["begins"] == "rest"] += [(i, how) for _, i, how in sorted(steps)]
        if periods[0] or periods[1] or report is not None:
            out[uid] = Column(uid, tuple(periods[0]), tuple(periods[1]), report, rest, rec["station"])
    return out


def named(day: crewsim.CrewDay) -> dict:
    """A crew sim day as a plan that names everything it did: each flight's minute or cancellation, its crews and
    riders, and the reports."""
    out: dict = {}
    for i, r in day.run.flights.items():
        if r.dep is not None:
            out[i] = {"not_before": r.dep, "crew": dict(day.crews.get(i, {}))}
            if day.riders.get(i):
                out[i]["deadhead"] = list(day.riders[i])
        elif r.status == "cancelled":
            out[i] = {"cancel": True, "decided": r.decided}
        else:
            out[i] = {"not_before": 10 ** 6}
    reports = {}
    for uid, rec in day.units.items():
        for d in rec["duties"]:
            if d["begins"] == "call":
                reports[uid] = d["report"]
            elif d["begins"] == "rest":
                reports[f"{uid}#2"] = d["report"]
    out["reports"] = reports
    return out


def uncrewed_causes(p: Problem, sol: crewreplay.Solved) -> list[dict]:
    """Per leg left without a legal crew: the kinds missing, and for each its planned unit's state as the plan has
    it at the leg's departure (where it is, its duty so far, the rule it fails)."""
    crewed = sol.crews(p)
    planned = defaultdict(dict)
    for u in p.units:
        for i in u.trip:
            planned[i][u.kind] = u
    at = {u.id: whereabouts(p, u, sol.columns.get(u.id)) for u in p.units}
    pricer = crewreplay.Pricer(p)
    out = []
    for i in sol.uncrewed:
        g = p.leg[i]
        row = {"id": g.id, "flight": g.flight, "orig": g.orig, "dest": g.dest, "dep": g.dep, "group": g.group,
               "missing": {}, "could_alone": {}}
        for kind in g.need:
            if kind in crewed.get(i, {}):
                continue
            u = planned[i].get(kind)
            row["missing"][kind] = reason(p, u, at[u.id], g) if u is not None else "no unit's plan has it" + "".join(
                f"; {reason(p, v, at[v.id], g)}" for v in p.units if v.kind == kind and v.group == g.group
                and here(at[v.id], g))
            row["could_alone"][kind] = alone(p, pricer, kind, g)
        out.append(row)
    return out


def alone(p: Problem, pricer: crewreplay.Pricer, kind: str, g: Leg) -> list[str]:
    """The units that, alone in the day, have a legal day flying leg g."""
    pi = {k: np.zeros(len(p.legs)) for k in crews.KINDS}
    pi[kind][p.pos[g.id]] = 10.0 * crewreplay.UNCREWED
    y = crewreplay.Duals(pi, np.zeros(len(p.legs)), {u.id: 0.0 for u in p.units}, {})
    return [u.id for u in p.units if u.kind == kind and u.group == g.group
            and pricer.price(u, y, 1)[0] < -crewreplay.UNCREWED]


def whereabouts(p: Problem, u: Unit, c: Column | None) -> list[tuple[int, str, dict]]:
    """The unit's state after each step of its plan: (minute, station, duty so far)."""
    state = {"report": u.report, "hour": u.hour, "flown": u.flown, "minutes": u.minutes}
    out = [(u.ready, u.station if c is None else p.began(u, c), dict(state))]
    if c is None:
        return out
    if c.report is not None:
        state.update(report=c.report, hour=p.hour(u.home, c.report), flown=0, minutes=0)
    for n, (i, how) in enumerate(c.steps):
        if c.rest is not None and n == len(c.first):
            state.update(report=c.rest[1], hour=p.hour(*c.rest), flown=0, minutes=0)
        g = p.leg[i]
        if how == OP:
            state.update(flown=state["flown"] + 1, minutes=state["minutes"] + g.block)
        out.append((g.arr + crews.CONNECT, g.dest, dict(state)))
    return out


def here(states: Sequence[tuple[int, str, dict]], g: Leg) -> bool:
    return max(((t, st) for t, st, _ in states if t <= g.dep), default=(0, ""))[1] == g.orig


def reason(p: Problem, u: Unit, states: Sequence[tuple[int, str, dict]], g: Leg) -> str:
    ready, station, s = max(((t, st, s) for t, st, s in states if t <= g.dep), default=states[0],
                            key=lambda x: x[0])
    if station != g.orig:
        return f"{u.id} at {station}, not {g.orig}"
    if s["report"] is None:
        return f"{u.id} (short-call) not called in time"
    if u.kind == "pilot":
        limit = crews.pilot_limit(s["report"], s["hour"], s["flown"] + 1, u.rap if u.reserve == "short" else None)
        if g.dep + g.sched_block > limit:
            return (f"{u.id} times out: report {s['report']}, {s['flown']} segments, limit {limit} "
                    f"< expected block-in {g.dep + g.sched_block}")
        if s["minutes"] + g.sched_block > crews.flight_time(s["hour"]):
            return f"{u.id} Table A: {s['minutes']} + {g.sched_block} > {crews.flight_time(s['hour'])}"
    elif g.id not in u.trip and g.dep + g.sched_block + crews.DEBRIEF - s["report"] > crews.CABIN_DUTY:
        return f"{u.id} 14 h duty from report {s['report']}"
    return f"{u.id} legal at {g.orig} but used elsewhere"
