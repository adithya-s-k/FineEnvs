"""The two desks of the control center, scripted: the standard crew desk (crewsim.py, desk on) and an ops script that
plans on the shared picture, with and without the channel between them.

The crew desk is the standard one. With `drop` it cancels where it is stuck, as in the baseline; without it, it can't,
and asks ops to (CANCEL); either way each decision is also its word to ops, CANCEL or HOLD(T) (crewsim.Pool.say).

Ops acts at its watches, every STEP from the start while a decision could still take effect (t + FREEZE by the last
deadline). At watch t it reads the day as known (sim.known: delays within LOOK, the programs as issued), the history
before t from the current crew day (a flight that left keeps its minute, one the crew desk cancelled stays cancelled),
its own plan, and in its mode the crew desk's messages sent by t and the crew flags logged by t; it never reads crews.
In order:
1. answer the messages it reads that are new, each flight's latest only: CANCEL accepted for an open flight (cancelled
   at t), HOLD(T) for an open flight taken as its ETA in the projection only (not binding); the rest declined, an
   earlier message of the same flight superseded;
2. project the day, aircraft only, on that view (each ETA as a not-before);
3. the rule (references.rule): cancel each open flight the projection auto-cancels, not committed, its auto-cancel
   FREEZE or more after t;
4. cut round trips: of the out-and-backs X then Y on the planned rotations (`round_trips`, never BTS tails), both open,
   neither committed nor bound to a Move (cancelling one would move an aircraft before t), X projected to leave late,
   those whose cancellation at t lowers the projected cost (cost.priced), largest saving first, ties by flight, each
   projected again on top of the cuts taken and kept only if it still lowers it;
5. with flags on, cancel each open flight whose latest flag by t is "no crew", or "times out" before t, once its
   scheduled departure has passed;
6. play the crew day again with these decisions; if the crew desk, reacting at t, sends anything new by t, go back to
   1 (each pass only adds cancellations, so it ends);
7. commit as references.rule does: a flight the final projection has leaving before t + FREEZE flies (no later rule
   or cut).
Every decision is a cancellation decided at t; ops never holds, claims slots, swaps aircraft, un-cancels or names
crews. A cancellation reaches the crew desk at its minute (sim.execute), so what happened before t can't change, which
each watch checks (`past`) and records otherwise.

The modes: `baseline` (the desk cancels, ops idle), `quiet-idle` (the desk asks, nobody answers), `talk-idle` (ops only
answers), `quiet` (the script, no channel), `quiet-flags` (the script and the flag rule), `talk` (the script reading
CANCEL and HOLD), `one` (the desk cancels at once, the script reading HOLD: one controller). `at_once` is the check that
the channel can cancel as the desk does: idle ops accepting each CANCEL at the minute it was sent.
"""

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from ..core.known import FREEZE, STEP
from ..core.model import Airport
from ..core.sim import AUTO_CANCELLED, CANCELLED, FLOWN
from . import crewday, crews, crewsim, pairings, references, repairs, sim
from .crews import Unit
from .crewsim import CANCEL, HOLD, NO_CREW, TIMES_OUT
from .model import Task

RULE, CUT, REQUEST, FLAG = "rule", "cut", "request", "flag"


@dataclass(frozen=True)
class Mode:
    drop: bool
    script: bool
    flags: bool
    reads: tuple[str, ...]


MODES = {
    "baseline": Mode(True, False, False, ()),
    "quiet-idle": Mode(False, False, False, ()),
    "talk-idle": Mode(False, False, False, (CANCEL, HOLD)),
    "quiet": Mode(False, True, False, ()),
    "quiet-flags": Mode(False, True, True, ()),
    "talk": Mode(False, True, False, (CANCEL, HOLD)),
    "one": Mode(True, True, False, (HOLD,)),
}


@dataclass(frozen=True)
class Case:
    """A date's two-desk world: the task on the truth, its crew units at the start, the station clock, the airports
    and the round trips on its planned rotations."""

    task: Task
    units: tuple[Unit, ...]
    local: crews.Local
    airports: Mapping[str, Airport]
    trips: tuple[tuple[int, int], ...]
    world: sim.World
    known: dict[int, sim.World] = field(default_factory=dict, compare=False)

    @classmethod
    def of(cls, w: crewday.World, units: Sequence[Unit]) -> "Case":
        return cls(w.task, tuple(units), w.local, w.airports, round_trips(w.legs, w.flight_of()), sim.truth(w.task))

    def day(self, plan: Mapping, drop: bool) -> crewsim.CrewDay:
        return crewsim.execute(self.task, plan, self.units, self.world, self.local, airports=self.airports, desk=True,
                               drop=drop)

    def project(self, view: Mapping[int, Mapping], t: int) -> sim.Execution:
        if t not in self.known:
            self.known[t] = sim.known(self.task, t)
        return sim.execute(self.task, view, self.known[t])


@dataclass
class Run:
    """A mode's day: the crew day under ops's plan, the cause of each of its cancellations, ops's replies by message,
    each watch's decisions, the watches where something before t changed, the crew days played and the seconds."""

    mode: str
    day: crewsim.CrewDay
    plan: dict[int, dict] = field(default_factory=dict)
    causes: dict[int, str] = field(default_factory=dict)
    replies: dict[tuple, dict] = field(default_factory=dict)
    watches: list[dict] = field(default_factory=list)
    acausal: list[dict] = field(default_factory=list)
    days: int = 1
    seconds: float = 0.0

    def cancel(self, i: int, t: int, cause: str) -> None:
        self.plan[i] = {"cancel": True, "decided": t}
        self.causes[i] = cause

    def messages(self) -> list[dict]:
        """The crew desk's messages of the final day with ops's replies."""
        return [{**m, **self.replies.get(key(m), {})} for m in self.day.messages]


def run(case: Case, name: str, mode: Mode | None = None) -> Run:
    """Mode `name` played on the case (`mode` in place of its own)."""
    t0 = time.monotonic()
    mode = MODES[name] if mode is None else mode
    r = Run(name, case.day({}, mode.drop))
    if mode.script or mode.reads:
        committed: set[int] = set()
        eta: dict[int, int] = {}
        for t in watches(case.task):
            watch(case, mode, r, t, committed, eta)
    r.seconds = round(time.monotonic() - t0, 2)
    return r


def watch(case: Case, mode: Mode, r: Run, t: int, committed: set[int], eta: dict[int, int]) -> None:
    """Ops at watch t (see the module doc)."""
    task = case.task
    was = past(r.day, t)
    log: dict = {"t": t, RULE: [], CUT: [], REQUEST: [], FLAG: [], "eta": [], "passes": 0}
    while True:
        log["passes"] += 1
        n = len(r.plan)
        answer(r, mode, t, eta, log)
        if mode.script:
            proj = case.project(view(task, r.plan, r.day, t, eta), t)
            for i in sorted(i for i, x in proj.flights.items() if x.status == AUTO_CANCELLED and i not in committed
                            and x.decided >= t + FREEZE and opened(r.day, r.plan, i, t)):
                r.cancel(i, t, RULE)
                log[RULE].append(i)
            for x, y in cut(case, r, t, committed, eta):
                r.cancel(x, t, CUT)
                r.cancel(y, t, CUT)
                log[CUT].append([x, y])
        if mode.flags:
            for i in flagged(task, r.day, r.plan, t):
                r.cancel(i, t, FLAG)
                log[FLAG].append(i)
        if len(r.plan) == n:
            break
        said = {key(m) for m in r.day.messages}
        r.day = case.day(r.plan, mode.drop)
        r.days += 1
        if not any(m["sent"] <= t and key(m) not in said for m in r.day.messages):
            break
    if mode.script and t > task.start:
        proj = case.project(view(task, r.plan, r.day, t, eta), t)
        committed |= {i for i, x in proj.flights.items() if x.status == FLOWN and x.dep < t + FREEZE}
    now = past(r.day, t)
    if now != was:
        r.acausal.append({"t": t, "parts": [k for k, a, b in zip(PARTS, was, now, strict=True) if a != b]})
    r.watches.append(log)


def answer(r: Run, mode: Mode, t: int, eta: dict[int, int], log: dict) -> None:
    """Ops's replies at t to the new messages it reads: each flight's latest accepted if the flight is open (CANCEL:
    cancelled at t; HOLD: its ETA), else declined; an earlier one superseded."""
    fresh = sorted((m for m in r.day.messages if m["sent"] <= t and m["type"] in mode.reads
                    and key(m) not in r.replies), key=lambda m: m["sent"])
    latest = {m["params"]["flight"]: key(m) for m in fresh}
    for m in fresh:
        i, k = m["params"]["flight"], key(m)
        if latest[i] != k:
            r.replies[k] = {"reply": "superseded", "replied": t, "actions": []}
        elif not opened(r.day, r.plan, i, t):
            r.replies[k] = {"reply": "decline", "replied": t, "actions": []}
        elif m["type"] == CANCEL:
            r.cancel(i, t, REQUEST)
            log[REQUEST].append(i)
            r.replies[k] = {"reply": "accept", "replied": t, "actions": [{"cancel": i, "at": t}]}
        else:
            eta[i] = m["params"]["minute"]
            log["eta"].append([i, eta[i]])
            r.replies[k] = {"reply": "accept", "replied": t, "actions": [{"eta": i, "at": eta[i]}]}


def cut(case: Case, r: Run, t: int, committed: set[int], eta: Mapping[int, int]) -> list[tuple[int, int]]:
    """The round trips to cut at t (step 4 of the module doc)."""
    task = case.task
    base = view(task, r.plan, r.day, t, eta)
    proj = case.project(base, t)
    now = proj.cost.total
    tried = []
    for x, y in case.trips:
        if {x, y} & committed or x in task.moved or y in task.moved or not opened(r.day, r.plan, x, t) or \
                not opened(r.day, r.plan, y, t):
            continue
        f = proj.flights[x]
        if f.status != FLOWN or f.dep <= task.flight(x).sched_dep:
            continue
        cost = case.project(base | cancels((x, y), t), t).cost.total
        if cost < now:
            tried.append((cost - now, x, y))
    out = []
    for _, x, y in sorted(tried):
        nxt = base | cancels((x, y), t)
        cost = case.project(nxt, t).cost.total
        if cost < now:
            out.append((x, y))
            base, now = nxt, cost
    return out


def flagged(task: Task, day: crewsim.CrewDay, plan: Mapping[int, Mapping], t: int) -> list[int]:
    """The open flights whose latest crew flag by t says no crew, or times out before t, their schedule past."""
    out = []
    for i, log in sorted(day.status.items()):
        seen = [e for e in log if e[0] <= t]
        if seen and (seen[-1][1] == NO_CREW or (seen[-1][1] == TIMES_OUT and seen[-1][2] < t)) and \
                task.flight(i).sched_dep <= t and opened(day, plan, i, t):
            out.append(i)
    return out


def view(task: Task, plan: Mapping[int, Mapping], day: crewsim.CrewDay, t: int, eta: Mapping[int, int]
         ) -> dict[int, dict]:
    """The day as ops plans it at t: the history before t from the crew day (references.so_far), what the crew desk
    cancelled before t cancelled, each ETA a not-before."""
    out = references.so_far(task, plan, t, day.run)
    for i, x in day.run.flights.items():
        if x.status == CANCELLED and x.decided < t and not out[i].get("cancel"):
            out[i] = {"cancel": True, "decided": x.decided}
        elif i in eta and "not_before" in out[i]:
            out[i]["not_before"] = max(out[i]["not_before"], eta[i])
    return out


def opened(day: crewsim.CrewDay, plan: Mapping[int, Mapping], i: int, t: int) -> bool:
    """Whether flight i is open at t: not in ops's plan, not gone before t, not cancelled before t."""
    x = day.run.flights[i]
    return not plan.get(i, {}).get("cancel") and (x.dep is None or x.dep >= t) and \
        (x.status == FLOWN or x.decided >= t)


def cancels(ids: Sequence[int], t: int) -> dict[int, dict]:
    return {i: {"cancel": True, "decided": t} for i in ids}


def watches(task: Task) -> list[int]:
    """Every STEP from the start while t + FREEZE is by the last deadline."""
    return list(range(task.start, max(f.deadline for f in task.flights) - FREEZE + 1, STEP))


def round_trips(legs: Sequence[pairings.Leg], flight_of: Mapping[int, int]) -> tuple[tuple[int, int], ...]:
    """The out-and-backs on the planned rotations (pairings.Leg.after) as task flights: consecutive legs A-B, B-A of a
    rotation, both the task's, paired left to right from any station (repairs.pairs)."""
    by = {g.key: g for g in legs}
    turned = {g.after for g in legs if g.after is not None}
    out = []
    for head in sorted((g for g in legs if g.key not in turned), key=lambda g: (g.sd, g.key)):
        chain, runs = [head], [[]]
        while chain[-1].after is not None:
            chain.append(by[chain[-1].after])
        for g in chain:
            if g.key in flight_of:
                runs[-1].append((flight_of[g.key], g.orig, g.dest))
            elif runs[-1]:
                runs.append([])
        for legs_ in runs:
            paired = repairs.pairs(legs_, None)
            out += [(i, paired[i]) for i, _, _ in legs_ if i in paired and paired[i] > i]
    return tuple(sorted(out))


def at_once(case: Case) -> Run:
    """Idle ops accepting each CANCEL at the minute it was sent, the earliest first, until none is new."""
    t0 = time.monotonic()
    r = Run("at once", case.day({}, False))
    while True:
        asks = [m for m in r.day.messages if m["type"] == CANCEL and key(m) not in r.replies]
        if not asks:
            break
        s = min(m["sent"] for m in asks)
        n = len(r.plan)
        for m in asks:
            if m["sent"] == s:
                i = m["params"]["flight"]
                ok = opened(r.day, r.plan, i, s)
                if ok:
                    r.cancel(i, s, REQUEST)
                r.replies[key(m)] = {"reply": "accept" if ok else "decline", "replied": s,
                                     "actions": [{"cancel": i, "at": s}] if ok else []}
        if len(r.plan) > n:
            r.day = case.day(r.plan, False)
            r.days += 1
    r.seconds = round(time.monotonic() - t0, 2)
    return r


def same(a: crewsim.CrewDay, b: crewsim.CrewDay) -> bool:
    """Whether two crew days flew, cancelled and crewed alike at the same cost."""
    return a.run.rows() == b.run.rows() and a.crews == b.crews and a.riders == b.riders and \
        abs(a.total - b.total) < 1e-6


PARTS = ("departures", "cancellations", "flags", "messages")


def past(day: crewsim.CrewDay, t: int) -> tuple:
    """What the day did before t: departures, cancellations, crew flags and the crew desk's messages."""
    flights = day.run.flights.items()
    return (tuple((i, x.dep, x.arr, x.tail) for i, x in flights if x.dep is not None and x.dep < t),
            tuple((i, x.status, x.decided) for i, x in flights if x.status != FLOWN and x.decided < t),
            tuple((i, e) for i, log in sorted(day.status.items()) if (e := tuple(v for v in log if v[0] < t))),
            tuple((m["sent"], m["type"], m["params"]["flight"], m["params"]["minute"]) for m in day.messages
                  if m["sent"] < t))


def key(m: Mapping) -> tuple:
    return m["sent"], m["type"], m["params"]["flight"], m["params"]["minute"]


def roles(r: Run, mode: Mode, task: Task) -> list[str]:
    """What breaks the roles: ops naming anything but a cancellation, or deciding off its watches; the crew desk
    cancelling without `drop`, or a cancellation of its own it didn't log as accepted at once."""
    at = set(watches(task))
    out = [f"ops plan for {i}: {sorted(e)}" for i, e in sorted(r.plan.items()) if set(e) != {"cancel", "decided"}]
    out += [f"ops cancelled {i} at {e['decided']}, not a watch" for i, e in sorted(r.plan.items())
            if e.get("decided") not in at]
    own = sorted(i for i, x in r.day.run.flights.items() if x.status == CANCELLED and i not in r.plan)
    if not mode.drop:
        out += [f"crew desk dropped {i}" for i in sorted(r.day.dropped)]
        out += [f"crew desk cancelled {i}" for i in own]
    else:
        said = {(m["params"]["flight"], m["sent"]) for m in r.day.messages if m["type"] == CANCEL
                and m["reply"] == "accept"}
        out += [f"crew desk cancelled {i} unannounced" for i in own
                if (i, r.day.run.flights[i].decided) not in said]
    return out


def counts(r: Run) -> dict:
    """The run's cancellations by cause, the crew desk's own, auto-cancels and messages sent and accepted."""
    flights = r.day.run.flights
    by = {c: sum(1 for i, v in r.causes.items() if v == c and flights[i].status == CANCELLED) for c in
          (RULE, CUT, REQUEST, FLAG)}
    ms = r.messages()
    return {"cancelled": by, "desk": sum(1 for i, x in flights.items() if x.status == CANCELLED and i not in r.plan),
            "auto": sum(1 for x in flights.values() if x.status == AUTO_CANCELLED),
            "sent": {k: sum(m["type"] == k for m in ms) for k in (CANCEL, HOLD)},
            "accepted": {k: sum(m["type"] == k and m.get("reply") == "accept" for m in ms) for k in (CANCEL, HOLD)}}
