"""JetBlue's slots under a ground delay program, by ration by schedule.

A GDP advisory with rate r in hour h offers arrival slots at h:00 + floor(60j/r) inside its period. Its demand is every
BTS-scheduled arrival in the period, any carrier, in the task or not, plus e unseen arrivals per hour (traffic BTS never
sees and UDP's pop-up reserve), spread evenly and never drawn as aircraft. Exempt flights take the first free slot at
or after their scheduled arrival, before everyone else: the unseen stream, origins outside the program's scope, and
flights that had left when it was signed. The rest are rationed by schedule, ordered by (scheduled arrival, carrier,
flight number, origin). Past the period's slots the list spills on at the last hour's rate, which prices the delay of
the flights the program can't hold (FSM's slot list runs past the end the same way); JetBlue's pool keeps only the
slots inside the period, and a JetBlue flight beyond it arrives after the period at nominal capacity. A cancelled
flight outside the task leaves the list at its scheduled departure less NOTICE (BTS dates no cancellation, so this one
rule serves the truth, the fit of e and the forecast); in the truth the queue then moves up (one-way compression,
applied for every cancellation before the segment is superseded), and JetBlue's task flights always stay, cancelling
them being the desk's call. JetBlue's pool is the slots RBS gives its task flights that are in scope and not exempt.
"Left" means airborne, as FSM exempts active flights: the recorded departure (the scheduled one when cancelled), held
to the end of any ground stop at the hub covering its origin, because BTS dates the pushback and a stop holds the
takeoff. The truth and the forecast read the same rule, so the pool the desk sees is the one it gets, less the
compression still to come.

Each GDP advisory is one segment, governing departures from its signature until the next GDP advisory, a cancellation
or its period's end (advisories.as_ran). The build's tests read the schedule and the recorded day, never lb: a flight is
controlled when its origin is in scope and its scheduled arrival is inside the period of the segment in force at its
recorded departure (the scheduled one for a cancelled flight), and GS-held when it is bound for the hub from an origin
a stop covers while its scheduled departure is inside the stop's as-ran window.

Outside programs the nominal cap applies: JetBlue's arrivals in hour h at hub X may reach
max(J_h, AAR_X - O_h - e_X), with J_h its scheduled task arrivals, O_h every other BTS-scheduled arrival (cancelled
ones too: the cap reads schedules only), e_X the airport's median unseen rate and AAR_X the FAA Airport Capacity
Profile's highest facility-reported arrival rate in visual conditions: JFK 52 in arrival priority (13L/22L arrivals,
JFK profile 2014, p. 3) and BOS 61 in north flow (04L/04R arrivals, BOS profile 2019, p. 5). It holds in the truth and
the view alike, in instrument hours too (a labelled simplification).
"""

import bisect
import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path

from ..build import bts, fetch
from .advisories import Run, Scope, Stop, as_ran, held_until, issued_by
from .model import Arrival, Programs, Segment, Task

NOTICE = 120
AAR = {"JFK": 52, "BOS": 61}
PROFILES = "https://www.faa.gov/sites/faa.gov/files/airports/planning_capacity/profiles/"
PROFILE = {"JFK": "JFK-Airport-Capacity-Profile-2014.pdf", "BOS": "BOS-Airport-Capacity-Profile-2019.pdf"}


Covers = Callable[[Scope, str, str], bool]
"""covers(scope, hub, origin): whether a program at `hub` with `scope` covers departures from `origin` (Task.covers)."""


def segments(programs: Programs, covers: Covers, known: int | None = None) -> tuple[Segment, ...]:
    """The hub's GDP segments as they ran (`known` None) or as issued by watch `known`, each with JetBlue's pool."""
    runs = as_ran(programs.advisories)[0] if known is None else issued_by(programs.advisories, known)[0]
    return tuple(segment(r, programs.demand, covers, programs.e(r.adv.id), programs.stops, known)[0] for r in runs)


def segment(run: Run, demand: Sequence[Arrival], covers: Covers, e: float, stops: Sequence[Stop],
            known: int | None = None) -> tuple[Segment, dict[int, int]]:
    """The segment of `run` and the slot of each demand row in its list, by index (after `end`: spilled); `stops` are
    the hub's ground-stop pieces as they ran."""
    got = assign(run, demand, covers, e, stops, known)
    pool = sorted((s, demand[i].flight) for i, s in got.items()
                  if s <= run.adv.end and demand[i].flight is not None and not exempt(demand[i], run, covers, stops))
    a = run.adv
    seg = Segment(a.id, a.hub, run.since, run.until, a.start, a.end, round(e, 4), tuple(s for s, _ in pool),
                  tuple(f for _, f in pool))
    return seg, got


def assign(run: Run, demand: Sequence[Arrival], covers: Covers, e: float, stops: Sequence[Stop],
           known: int | None = None) -> dict[int, int]:
    """RBS for one run: the demand rows in its period still listed at min(until - 1, known), exempt ones and the unseen
    stream first by scheduled arrival, then the rest in RBS order; past the period's slots the list spills on at the
    last hour's rate."""
    a = run.adv
    t = run.until - 1 if known is None else min(run.until - 1, known)
    rows = [i for i, r in enumerate(demand) if a.start <= r.sched_arr <= a.end and not gone(r, t)]
    out = {i for i in rows if exempt(demand[i], run, covers, stops)}
    first = sorted([(demand[i].sched_arr, 0, demand[i].key, i) for i in out]
                   + [(m, 1, (k,), -1 - k) for k, m in enumerate(unseen(a.start, a.end, e))])
    ranked = sorted((i for i in rows if i not in out), key=lambda i: demand[i].key)
    offered = slots(a.start, a.end, a.rates)
    offered += [a.end + 1 + 60 * k // a.rates[-1] for k in range(len(first) + len(ranked))]
    got = ration([(m, i) for m, _, _, i in first], [(demand[i].sched_arr, i) for i in ranked], offered)
    return {i: s for i, s in got.items() if i >= 0}


def exempt(r: Arrival, run: Run, covers: Covers, stops: Sequence[Stop]) -> bool:
    """Out of the program's scope, or airborne when it was signed."""
    if not covers(run.adv.scope, run.adv.hub, r.orig):
        return True
    return held_until(stops, covers, r.orig, r.departs) < run.since


def gone(r: Arrival, t: int) -> bool:
    """Whether a cancelled flight outside the task has left the demand list by minute t."""
    return r.cancelled and r.flight is None and r.sched_dep - NOTICE <= t


def ration(first: list[tuple[int, int]], ranked: list[tuple[int, int]], offered: list[int]) -> dict[int, int]:
    """Each (eta, key) of `first`, then of `ranked`, takes the first free slot at or after its eta, in list order
    (`offered` must outlast the demand)."""
    free = sorted(offered)
    out: dict[int, int] = {}
    for eta, key in [*first, *ranked]:
        out[key] = free.pop(bisect.bisect_left(free, eta))
    return out


def slots(start: int, end: int, rates: Sequence[int]) -> list[int]:
    """h:00 + floor(60j/r) for each hour's rate r, from the hour holding `start`, kept inside [start, end]."""
    h0 = start // 60
    return [s for k, r in enumerate(rates) for j in range(r) if start <= (s := 60 * (h0 + k) + 60 * j // r) <= end]


def unseen(start: int, end: int, e: float) -> list[int]:
    """e arrivals an hour, evenly from `start` through `end`."""
    if e <= 0:
        return []
    return [m for k in range(math.floor((end - start) * e / 60) + 1) if (m := start + math.floor(k * 60 / e)) <= end]


def governing(runs: Sequence[Run], covers: Covers, orig: str, sched_arr: int, departs: int) -> Run | None:
    """The run controlling an arrival (FSM-style): its origin in scope and its scheduled arrival in the period of
    the GDP in force when it departs."""
    for r in runs:
        if r.since <= departs < r.until:
            a = r.adv
            return r if a.start <= sched_arr <= a.end and covers(a.scope, a.hub, orig) else None
    return None


def gs_held(stops: Sequence[Stop], covers: Covers, orig: str, sched_dep: int) -> str | None:
    """The chain holding a departure for its stop's hub: an origin in scope, scheduled to leave inside the window."""
    for p in stops:
        if p.start <= sched_dep < p.end and covers(p.scope, p.hub, orig):
            return p.chain
    return None


def nominal(task: Task, hub: str) -> dict[int, int]:
    """JetBlue's arrival cap per local hour at `hub` outside programs, from the first watch's hour through the hour
    of the latest arrival a deadline allows."""
    p = task.programs[hub]
    mine = Counter(f.sched_arr // 60 for f in task.flights if f.dest == hub)
    others = Counter(a.sched_arr // 60 for a in p.demand if a.flight is None)
    last = max(f.deadline + f.block for f in task.flights) // 60
    return {h: max(mine[h], math.floor(p.aar - others[h] - p.e_bar)) for h in range(task.start // 60, last + 1)}


def forecast(task: Task, hub: str, t: int, early: int = 0) -> tuple[tuple[Segment, ...], tuple[Stop, ...]]:
    """The hub's programs on the day as known at watch t: segments with JetBlue's forecast pool, and the stop
    pieces, each to its as-issued end. With `early` (a base rate) every GDP segment stops governing `early` minutes
    before its period's end, never before t, its pool unchanged."""
    p = task.programs[hub]
    segs = segments(p, task.covers, t)
    if early:
        segs = tuple(replace(s, until=min(s.until, max(t, s.end + 1 - early))) for s in segs)
    return segs, issued_by(p.advisories, t)[1]


def rescale(task: Task, scale: Mapping[str, float], e_fit: Mapping[str, float],
            e_median: Mapping[str, float]) -> Task:
    """The task with each hub's unseen arrivals at a new scale (each advisory's e and the airport's median at it, as
    unseen.table gives them) and its truth pools recomputed."""
    programs = {}
    for hub, p in task.programs.items():
        q = replace(p, scale=scale[hub], e_median=e_median[hub], e_fit={a: e_fit[a] for a in p.e_fit})
        programs[hub] = replace(q, segments=segments(q, task.covers))
    return replace(task, programs=programs)


def profile_path(inputs: Path, hub: str) -> Path:
    return inputs / "faa" / "capacity" / PROFILE[hub]


def fetch_profiles(inputs: Path) -> list[Path]:
    out = []
    for hub in PROFILE:
        path = profile_path(inputs, hub)
        if not path.exists():
            fetch.save(path, fetch.get(PROFILES + PROFILE[hub], timeout=300))
            fetch.record(inputs, str(path.relative_to(inputs)), PROFILES + PROFILE[hub], bts.today())
        out.append(path)
    return out
