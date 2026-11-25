"""The time-line network MIP (HiGHS) over the env's decision space: per flight, cancel or a departure minute.

Each flight gets one arc per departure minute from its earliest time to its deadline; each (family, station) has a node
at every event time (departure copies, ready times = arrival + turn, aircraft availability), linked by ground arcs. A
flight is flown by one copy or cancelled, hourly caps count copies, and overnight nodes feed end-of-day ferries into
the required stations. Aircraft of a family at a station are interchangeable, so any solution is a dispatch the env
reproduces. ``floor`` drops the aircraft flow: each flight alone at its earliest time under the caps.
"""

import time
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

import highspy
import numpy as np

from ..core.cost import cancel_cost, ferry_cost
from ..core.model import Task

GAP = 1e-4


@dataclass(frozen=True)
class Solution:
    status: str
    objective: int
    bound: float
    gap_pct: float
    wall_s: float
    deps: dict[int, int | None]
    cols: int
    rows: int

    @property
    def proven(self) -> bool:
        return self.status == "Optimal" and self.gap_pct <= 100 * GAP

    @property
    def plan(self) -> dict[int, dict]:
        """The solution as a plan: cancel, or hold to the departure minute."""
        return {i: {"cancel": d is None, "not_before": d} for i, d in self.deps.items()}

    def stored(self) -> dict[str, dict]:
        return {str(i): {"cancel": int(d is None), "dep": d} for i, d in self.deps.items()}


def solve(task: Task, lb: Mapping[int, int] | None = None, caps: Mapping[str, Mapping[int, int]] | None = None,
          fixed: Mapping[int, int | None] | None = None, earliest: int | None = None, floor: bool = False,
          step: int = 1, time_limit: float = 60.0, gap: float = GAP, threads: int = 1,
          canonical: bool = False) -> Solution:
    """The optimum under ``lb`` and ``caps`` (the truth by default). ``fixed`` pins flights to a departure minute
    (None: cancelled); the others leave at or after ``earliest``. ``canonical`` re-solves among the optimal plans for
    the one with the earliest weighted departures, so the plan, not just its cost, is the same on every run."""
    lb = task.lb if lb is None else lb
    caps = task.caps if caps is None else caps
    fixed = fixed or {}
    m = _Model()
    events: dict[tuple[str, str], set[int]] = defaultdict(set)
    copies = {}
    for f in task.flights:
        if f.id in fixed:
            copies[f.id] = [] if fixed[f.id] is None else [fixed[f.id]]
        else:
            lo = max(task.earliest(f, lb[f.id]), earliest or 0)
            copies[f.id] = list(range(lo, task.deadline(f, lb[f.id]) + 1, step))
        turn = task.turn_for(f.family)
        for d in copies[f.id]:
            events[(f.family, f.orig)].add(d)
            events[(f.family, f.dest)].add(d + f.block + turn)
    for t in task.tails:
        events[(t.family, t.pos)].add(t.ready)
    node = {}
    for key, ts in events.items():
        for x in sorted(ts):
            node[(key, x)] = len(node)
    over = {key: len(node) + n for n, key in enumerate(events)}
    flow: dict[int, dict[int, int]] = defaultdict(dict)

    def arc(u: int | None, v: int | None, c: int) -> None:
        if u is not None:
            flow[u][c] = flow[u].get(c, 0) - 1
        if v is not None:
            flow[v][c] = flow[v].get(c, 0) + 1

    xs = {}
    cover: dict[int, dict[int, int]] = defaultdict(dict)
    capr: dict[tuple[str, int], dict[int, int]] = defaultdict(dict)
    pinned_use: dict[tuple[str, int], int] = defaultdict(int)
    for f in task.flights:
        turn = task.turn_for(f.family)
        for d in copies[f.id]:
            c = m.col(f.pax * max(0, d + f.block - f.sched_arr), 1, True, (d - f.sched_dep + 1) * _weight(task, f.id))
            xs[(f.id, d)] = c
            arc(node[((f.family, f.orig), d)], node[((f.family, f.dest), d + f.block + turn)], c)
            cover[f.id][c] = 1
            for key, tm in ((f"{f.orig}:dep", d), (f"{f.dest}:arr", d + f.block)):
                if tm // 60 in caps.get(key, {}):
                    if f.id in fixed:
                        pinned_use[(key, tm // 60)] += 1
                    else:
                        capr[(key, tm // 60)][c] = 1
        flies = f.id in fixed and fixed[f.id] is not None
        cover[f.id][m.col(cancel_cost(f.pax), 0 if flies else 1, True, (f.dmax + 2) * _weight(task, f.id))] = 1
    if not floor:
        n_tails = len(task.tails)
        for key, ts in events.items():
            ts = sorted(ts)
            for a, b in zip(ts, ts[1:], strict=False):
                arc(node[(key, a)], node[(key, b)], m.col(0, n_tails, False))
            arc(node[(key, ts[-1])], over[key], m.col(0, n_tails, False))
        need: dict[tuple[str, str], int] = defaultdict(int)
        for t in task.tails:
            need[(t.family, t.end_req)] += 1
        into: dict[tuple[str, str], dict[int, int]] = defaultdict(dict)
        for fam, a in events:
            for fam2, b in need:
                if fam2 == fam:
                    c = m.col(ferry_cost(task, a, b), n_tails, True)
                    arc(over[(fam, a)], None, c)
                    into[(fam, b)][c] = 1
        for key, n in need.items():
            m.row(into[key], n, n)
        supply: dict[int, int] = defaultdict(int)
        for t in task.tails:
            supply[node[((t.family, t.pos), t.ready)]] += 1
        for v in range(len(node) + len(over)):
            m.row(flow[v], -supply[v], -supply[v])
    for r in cover.values():
        m.row(r, 1, 1)
    for (key, h), r in capr.items():
        m.row(r, -highspy.kHighsInf, max(0, caps[key][h] - pinned_use[(key, h)]))
    status, objective, bound, mip_gap, wall, x = m.run(time_limit, gap, threads)
    if canonical and status == "Optimal":
        m.row({c: v for c, v in enumerate(m.cost) if v}, -highspy.kHighsInf, round(objective) + 0.5)
        m.cost, main = m.second, m.cost
        status, _, _, _, wall_second, x = m.run(time_limit, 0.0, threads)
        m.cost, wall = main, wall + wall_second
    deps = {}
    for f in task.flights:
        flown = [d for d in copies[f.id] if x[xs[(f.id, d)]] > 0.5]
        deps[f.id] = flown[0] if flown else None
    return Solution(status, round(objective), bound, round(100 * mip_gap, 4), round(wall, 2), deps, len(m.cost),
                    len(m.rows))


def _weight(task: Task, i: int) -> float:
    return 1 + i / (len(task.flights) + 1)


class _Model:
    def __init__(self):
        self.cost, self.second, self.ub, self.integer, self.rows = [], [], [], [], []

    def col(self, cost: int, ub: int, integer: bool, second: float = 0.0) -> int:
        self.cost.append(cost)
        self.second.append(second)
        self.ub.append(ub)
        self.integer.append(integer)
        return len(self.cost) - 1

    def row(self, coefs: Mapping[int, int], lo: float, up: float) -> None:
        self.rows.append((coefs, lo, up))

    def run(self, time_limit: float, gap: float, threads: int) -> tuple[str, float, float, float, float, np.ndarray]:
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("time_limit", float(time_limit))
        h.setOptionValue("threads", threads)
        h.setOptionValue("mip_rel_gap", gap)
        n = len(self.cost)
        h.addVars(n, np.zeros(n), np.array(self.ub, dtype=float))
        h.changeColsCost(n, np.arange(n, dtype=np.int32), np.array(self.cost, dtype=float))
        starts, idx, val = [], [], []
        for coefs, _, _ in self.rows:
            starts.append(len(idx))
            for c, v in coefs.items():
                if v:
                    idx.append(c)
                    val.append(v)
        h.addRows(len(self.rows), np.array([r[1] for r in self.rows], dtype=float),
                  np.array([r[2] for r in self.rows], dtype=float), len(idx), np.array(starts, dtype=np.int32),
                  np.array(idx, dtype=np.int32), np.array(val, dtype=float))
        ints = np.array([c for c in range(n) if self.integer[c]], dtype=np.int32)
        h.changeColsIntegrality(len(ints), ints, np.array([highspy.HighsVarType.kInteger] * len(ints)))
        t0 = time.monotonic()
        h.run()
        wall = time.monotonic() - t0
        status = h.modelStatusToString(h.getModelStatus())
        info = h.getInfo()
        if info.primal_solution_status != 2:
            raise RuntimeError(f"no feasible solution: {status}")
        return (status, info.objective_function_value, info.mip_dual_bound, info.mip_gap, wall,
                np.array(h.getSolution().col_value))
