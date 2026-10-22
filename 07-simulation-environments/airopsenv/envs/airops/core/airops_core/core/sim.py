"""Running a plan: the env dispatches the aircraft, keeps the hourly slots, cancels what cannot leave in time and
ferries aircraft to their required stations at the end of the day (DESIGN.md, Execution).

A plan maps flight ids to {"cancel": bool, "not_before": minute | None}; flights left out fly as early as they can.
Flights are taken in order of their next try, ties by their first candidate time and id. A flight leaves when an
aircraft of its family at its origin is ready (the first-ready one, ties by tail) and both its departure and arrival
hours have room; otherwise it is tried again when the next aircraft there is ready (one dispatched there later wakes
it sooner) or at the next minute with room in both hours. One that cannot leave by its deadline is auto-cancelled.
"""

import heapq
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

from .cost import Cost, breakdown, ferry_cost
from .model import Flight, Task

FLOWN, CANCELLED, AUTO_CANCELLED = "flown", "cancelled", "auto_cancelled"
Plan = Mapping[int | str, Mapping]


@dataclass(frozen=True)
class FlightRun:
    id: int
    status: str
    tail: str | None
    dep: int | None
    arr: int | None
    arr_delay: int | None

    def to_dict(self) -> dict:
        return {"id": self.id, "tail": self.tail, "status": self.status, "dep": self.dep, "arr": self.arr,
                "arr_delay": self.arr_delay}


@dataclass(frozen=True)
class Ferry:
    tail: str
    family: str
    orig: str
    dest: str
    cost: int

    def to_dict(self) -> dict:
        return {"tail": self.tail, "family": self.family, "orig": self.orig, "dest": self.dest, "cost": self.cost}


@dataclass(frozen=True)
class Execution:
    flights: dict[int, FlightRun]
    legs: dict[str, tuple[int, ...]]
    ends: dict[str, str]
    ferries: tuple[Ferry, ...]
    cost: Cost

    @property
    def counts(self) -> dict[str, int]:
        c = Counter(r.status for r in self.flights.values())
        return {"flights": len(self.flights), FLOWN: c[FLOWN], CANCELLED: c[CANCELLED],
                AUTO_CANCELLED: c[AUTO_CANCELLED], "ferries": len(self.ferries)}

    def rows(self) -> list[dict]:
        return [self.flights[i].to_dict() for i in sorted(self.flights)]


def execute(task: Task, plan: Plan, lb: Mapping[int, int] | None = None,
            caps: Mapping[str, Mapping[int, int]] | None = None) -> Execution:
    """The day under ``plan``, with the exogenous delays ``lb`` and hourly ``caps`` (the truth by default)."""
    lb = task.lb if lb is None else lb
    caps = task.caps if caps is None else caps
    want = {int(k): v for k, v in plan.items()}
    runs: dict[int, FlightRun] = {}
    when: dict[int, float] = {}
    first: dict[int, int] = {}
    heap = []
    for f in task.flights:
        d = want.get(f.id, {})
        if d.get("cancel"):
            runs[f.id] = FlightRun(f.id, CANCELLED, None, None, None, None)
            continue
        c = max(task.earliest(f, lb[f.id]), d.get("not_before") or 0)
        when[f.id] = first[f.id] = c
        heap.append((c, c, f.id))
    heapq.heapify(heap)
    ready = {t.tail: t.ready for t in task.tails}
    at: dict[tuple[str, str], set[str]] = defaultdict(set)
    for t in task.tails:
        at[(t.family, t.pos)].add(t.tail)
    legs: dict[str, list[int]] = {t.tail: [] for t in task.tails}
    used: Counter = Counter()
    waiting: dict[tuple[str, str], set[int]] = defaultdict(set)
    while heap:
        t, c, i = heapq.heappop(heap)
        if i in runs or when[i] != t:
            continue
        f = task.flight(i)
        if t > task.deadline(f, lb[i]):
            runs[i] = FlightRun(i, AUTO_CANCELLED, None, None, None, None)
            continue
        here = at[(f.family, f.orig)]
        free = [p for p in here if ready[p] <= t]
        if not free:
            waiting[(f.family, f.orig)].add(i)
            when[i] = min((ready[p] for p in here), default=float("inf"))
            if here:
                heapq.heappush(heap, (when[i], c, i))
            continue
        waiting[(f.family, f.orig)].discard(i)
        s = _slot(f, t, caps, used)
        if s != t:
            when[i] = s
            heapq.heappush(heap, (s, c, i))
            continue
        p = min(free, key=lambda p: (ready[p], p))
        here.remove(p)
        arr = t + f.block
        ready[p] = arr + task.turn_for(f.family)
        at[(f.family, f.dest)].add(p)
        used[(f"{f.orig}:dep", t // 60)] += 1
        used[(f"{f.dest}:arr", arr // 60)] += 1
        legs[p].append(i)
        runs[i] = FlightRun(i, FLOWN, p, t, arr, max(0, arr - f.sched_arr))
        for j in waiting[(f.family, f.dest)]:
            if ready[p] < when[j]:
                when[j] = ready[p]
                heapq.heappush(heap, (ready[p], first[j], j))
    for f in task.flights:
        runs.setdefault(f.id, FlightRun(f.id, AUTO_CANCELLED, None, None, None, None))
    ends = {p: station for (_, station), ps in at.items() for p in ps}
    ferries = _ferries(task, ends)
    cost = breakdown(task, {i: r.arr for i, r in runs.items()}, [(x.orig, x.dest) for x in ferries])
    return Execution({i: runs[i] for i in sorted(runs)}, {p: tuple(v) for p, v in legs.items()},
                     dict(sorted(ends.items())), ferries, cost)


def as_plan(stored: Mapping[int | str, Mapping]) -> dict[int, dict]:
    """A stored reference plan ({"cancel": 0|1, "dep": minute | None} per flight) as a plan holding each flight to
    its departure."""
    return {int(i): {"cancel": bool(v["cancel"]), "not_before": None if v["cancel"] else v["dep"]}
            for i, v in stored.items()}


def _slot(f: Flight, t: int, caps: Mapping[str, Mapping[int, int]], used: Counter) -> int:
    """The first minute from ``t`` whose departure hour at the origin and arrival hour at the destination have room."""
    dep, arr = caps.get(f"{f.orig}:dep", {}), caps.get(f"{f.dest}:arr", {})
    while True:
        h = t // 60
        if h in dep and used[(f"{f.orig}:dep", h)] >= dep[h]:
            t = 60 * (h + 1)
            continue
        h = (t + f.block) // 60
        if h in arr and used[(f"{f.dest}:arr", h)] >= arr[h]:
            t = 60 * (h + 1) - f.block
            continue
        return t


def _ferries(task: Task, ends: Mapping[str, str]) -> tuple[Ferry, ...]:
    """The cheapest end-of-day ferries putting each family's aircraft on its required stations. Costs obey the
    triangle inequality, so aircraft already on a required station stay; the rest are matched optimally."""
    out = []
    for family in sorted({t.family for t in task.tails}):
        tails = sorted(t.tail for t in task.tails if t.family == family)
        need = Counter(t.end_req for t in task.tails if t.family == family)
        spare, left = [], Counter(need)
        for p in tails:
            if left[ends[p]] > 0:
                left[ends[p]] -= 1
            else:
                spare.append(p)
        holes = sorted(left.elements())
        if not spare:
            continue
        matrix = [[ferry_cost(task, ends[p], b) for b in holes] for p in spare]
        for k, j in enumerate(_assign(matrix)):
            out.append(Ferry(spare[k], family, ends[spare[k]], holes[j], matrix[k][j]))
    return tuple(out)


def _assign(cost: list[list[int]]) -> list[int]:
    """Min-cost perfect matching of a square matrix (the Hungarian algorithm): the column of each row."""
    n = len(cost)
    u, v, match, way = [0] * (n + 1), [0] * (n + 1), [0] * (n + 1), [0] * (n + 1)
    for i in range(1, n + 1):
        match[0], j0 = i, 0
        low, done = [float("inf")] * (n + 1), [False] * (n + 1)
        while match[j0]:
            done[j0] = True
            i0, delta, j1 = match[j0], float("inf"), 0
            for j in range(1, n + 1):
                if not done[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < low[j]:
                        low[j], way[j] = cur, j0
                    if low[j] < delta:
                        delta, j1 = low[j], j
            for j in range(n + 1):
                if done[j]:
                    u[match[j]] += delta
                    v[j] -= delta
                else:
                    low[j] -= delta
            j0 = j1
        while j0:
            j1 = way[j0]
            match[j0] = match[j1]
            j0 = j1
    col = [0] * n
    for j in range(1, n + 1):
        col[match[j] - 1] = j - 1
    return col
