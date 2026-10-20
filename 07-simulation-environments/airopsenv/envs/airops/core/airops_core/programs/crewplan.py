"""The best day over aircraft and crews found within a time budget, and the aircraft optimum it starts from.

The aircraft MIP (mip.Network) is built once per task and world and kept in HiGHS; a solve may price it flat or price a
cancellation to fly (FLY times its price), and close copies by column bounds: a flight held to a minute, one cancelled
at the start, one kept to the minutes some crew could fly it (crewbound.reach). That is the same as building it again
with those prices and lower bounds. The search (a local search over departures and cancellations, crews optimized at
each):
- an aircraft optimum's departures crewed by the crew sim's default rule (fill on), as are any plans given (the
  standard crew desk's day, the airline's), each then a plan naming everything its day did; and crews optimized at
  those departures
  (crewreplay.solve, started from the default rule's crews where they flew every flight at its minute, and from the
  last crew solve's columns and plan where their legs leave at the same minutes), its plan
  naming each flight's crews, riders and reports, replayed in the crew sim, where a flight flies only with its named
  crews (fill off);
- while flights go without a legal crew there, the crew sim decides them by its default rule (fill on), and the day
  it runs is a plan naming everything it did (and so is the day with them cancelled at the start, crewed by the
  default rule); the aircraft MIP again with those flights held to the minute the crew sim flew them (or cancelled at
  the start where it could not, or where it flew one at the minute it was already held to), crewed again;
- then, one at a time while time remains, each flight the best plan cancels priced to fly (and no longer held
  cancelled), those whose crews cost most first, in case its crews cost more stranded than another flight's
  cancellation does;
- with time left, the best day's departures crewed again, the crew solve taking what remains.
Every plan is replayed in the crew sim and the cheapest replay wins. A plan's claimed cost is the aircraft MIP's plus
the crew solve's, or the day the crew sim ran with fill on; it replays exactly when its replay costs that, with a legal
crew on every flight it flies.

`joint` finds a day's best plan and its lower bound (crewbound.py): the aircraft optimum A* and the restricted A*
(every flight kept to the minutes some crew could fly it; A* itself when it keeps to them) start the search; the
bound is the best of A*'s, the restricted A*'s, the restricted A* plus each unit's least crew cost, and the covering
bound with the best plan's cost.

The crew-aware calm day (`calm_day`) is the search on the calm world from the calm optimum without crews
(references.calm's), the calm units where the schedule assumes them (crewday.calm_units); the calm optimum without
crews bounds it below, and so does `joint`'s bound on the calm world, which may say more.
"""

import time
from collections import defaultdict
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field

import highspy
import numpy as np

from ..core.sim import AUTO_CANCELLED, FLOWN
from ..refs.mip import GAP
from . import crewbound, crewday, crewreplay, crews, crewsim, mip, network, sim
from .cost import flat_price
from .crews import Unit
from .model import Task

FLY = 10
INTEGRAL = 1e-6


class Aircraft:
    """A task's aircraft MIP in a world, built once: its notice and flat costs, and solves under changed costs and
    column bounds."""

    def __init__(self, task: Task, world: sim.World):
        self.task, self.world = task, world
        self.net = net = mip.Network(task, world)
        self.notice = np.array(net.m.cost, dtype=float)
        self.flat = self.notice.copy()
        for f in task.flights:
            self.flat[net.cancel[f.id]] = flat_price(f)
            if f.id in net.lapsed:
                self.flat[net.lapsed[f.id]] = flat_price(f)
        self.ub = np.array(net.m.ub, dtype=float)
        self.integer = np.array(net.m.integer)
        self.h = net.m._highs(600.0, 1)
        self.n, self.rows = len(self.notice), len(net.m.rows)
        net.m.rows = []

    def cost(self, flat: bool = False, fly: Collection[int] = ()) -> np.ndarray:
        out = (self.flat if flat else self.notice).copy()
        for i in fly:
            out[self.net.cancel[i]] *= FLY
            if i in self.net.lapsed:
                out[self.net.lapsed[i]] *= FLY
        return out

    def upper(self, held: Mapping[int, int] | None = None, dropped: Collection[int] = (),
              keep: Mapping[int, Collection[int]] | None = None) -> np.ndarray:
        """Column bounds with each flight in `held` leaving no earlier than its minute, those in `dropped` cancelled at
        the start, and each flight in `keep` leaving only at its minutes there."""
        ub = self.ub.copy()
        held, keep = held or {}, keep or {}
        for (i, d), c in self.net.xs.items():
            if d < held.get(i, -1) or i in dropped or (i in keep and d not in keep[i]):
                ub[c] = 0.0
        for i in dropped:
            if i in self.net.lapsed:
                ub[self.net.lapsed[i]] = 0.0
        return ub

    def solve(self, cost: np.ndarray, upper: np.ndarray, start: sim.Execution | None = None,
              time_limit: float = 600.0, value: np.ndarray | None = None) -> tuple[mip.Solution, np.ndarray]:
        """The MIP at `cost` within `upper`, from `start` when it holds there; the objective is the plan's cost at
        `value` (`cost` by default)."""
        h = self._set(cost, upper, True, time_limit)
        h.setOptionValue("mip_rel_gap", GAP)
        if start is not None and (x0 := self.net.vector(start)) is not None and np.all(x0 <= upper + 1e-9):
            h.setSolution(self.n, np.arange(self.n, dtype=np.int32), x0)
        t0 = time.monotonic()
        h.run()
        wall = time.monotonic() - t0
        info = h.getInfo()
        if info.primal_solution_status != 2:
            raise RuntimeError(f"aircraft MIP: no feasible solution ({h.modelStatusToString(h.getModelStatus())})")
        x = np.array(h.getSolution().col_value)
        return self.solution(x, h.modelStatusToString(h.getModelStatus()), info.mip_dual_bound, info.mip_gap, wall,
                             start is not None, cost if value is None else value), x

    def relax(self, cost: np.ndarray, upper: np.ndarray, time_limit: float = 600.0) -> mip.Relaxed | None:
        """The LP relaxation's optimum within `upper`, or None if it isn't reached."""
        h = self._set(cost, upper, False, time_limit)
        t0 = time.monotonic()
        h.run()
        wall = time.monotonic() - t0
        if h.modelStatusToString(h.getModelStatus()) != "Optimal":
            return None
        sol = h.getSolution()
        x = np.array(sol.col_value)
        integral = bool(np.all(np.abs(x[self.integer] - np.round(x[self.integer])) <= INTEGRAL))
        return mip.Relaxed(h.getInfo().objective_function_value, np.where(self.integer, np.round(x), x),
                           np.array(sol.col_dual), integral, wall)

    def optimum(self, cost: np.ndarray, upper: np.ndarray, start: sim.Execution | None = None,
                time_limit: float = 600.0) -> tuple[mip.Solution, mip.Relaxed | None]:
        """The MIP's optimum within `upper`, its LP relaxation first (an integral LP optimum is the optimum)."""
        lp = self.relax(cost, upper, time_limit)
        if lp is not None and lp.integral:
            return self.solution(lp.x, "Optimal", lp.objective, 0.0, lp.wall, False, cost), lp
        sol, _ = self.solve(cost, upper, start, time_limit)
        return sol, lp

    def solution(self, x: np.ndarray, status: str, bound: float, gap: float, wall: float, started: bool,
                 value: np.ndarray) -> mip.Solution:
        net = self.net
        deps = {}
        for f in self.task.flights:
            flown = [d for d, _ in net.copies[f.id] if x[net.xs[(f.id, d)]] > 0.5]
            deps[f.id] = flown[0] if flown else None
        lapsed = {i: self.task.flight(i).deadline + 1 for i, c in net.lapsed.items() if x[c] > 0.5}
        return mip.Solution(status, round(float(value @ x)), bound, round(100 * gap, 4), round(wall, 2), deps,
                            self.n, self.rows, sum(len(v) for v in net.copies.values()), started, False, lapsed)

    def _set(self, cost: np.ndarray, upper: np.ndarray, integer: bool, time_limit: float) -> highspy.Highs:
        h = self.h
        idx = np.arange(self.n, dtype=np.int32)
        kind = highspy.HighsVarType.kInteger if integer else highspy.HighsVarType.kContinuous
        cols = idx[self.integer]
        h.changeColsIntegrality(len(cols), cols, np.array([kind] * len(cols)))
        h.changeColsCost(self.n, idx, np.asarray(cost, dtype=float))
        h.changeColsBounds(self.n, idx, np.zeros(self.n), np.asarray(upper, dtype=float))
        h.setOptionValue("time_limit", float(time_limit))
        h.clearSolver()
        return h


@dataclass
class Calm:
    """The crew-aware calm day: the calm optimum without crews, the best crew-aware day found, and the pieces of a
    lower bound on the crew-aware calm optimum (`lower`)."""

    task: Task
    optimum: mip.Solution
    found: "Found"
    pieces: dict[str, float | None] = field(default_factory=dict)

    @property
    def bound(self) -> float | None:
        return max((v for v in self.pieces.values() if v is not None), default=None)

    @property
    def rise_pct(self) -> float | None:
        base, day = self.optimum.objective, self.found.day.total
        if abs(day - base) < 1e-6:
            return 0.0
        return None if base <= 0 else 100 * (day - base) / base


def calm_optimum(w: crewday.World, time_limit: float = 1800.0) -> mip.Solution:
    """The calm day's optimum without crews, as references.calm solves it."""
    task = network.scheduled(w.task)
    calm = sim.calm(task)
    return mip.solve(task, calm, time_limit=time_limit, canonical=False, start=sim.execute(task, {}, calm))


def calm_day(w: crewday.World, units: Sequence[Unit], optimum: mip.Solution | None = None,
             budget: float = 3600.0, time_limit: float = 1800.0, crew_limit: float = 900.0, tries: int = 4,
             cover_limit: float = 0.0, width: int = crewbound.WIDTH, log: Callable[[str], None] | None = None) -> Calm:
    """The best crew-aware calm day within `budget` seconds; with `cover_limit`, its lower bound too."""
    task = network.scheduled(w.task)
    calm = sim.calm(task)
    optimum = calm_optimum(w, time_limit) if optimum is None else optimum
    deadline = time.monotonic() + budget
    air = Aircraft(task, calm)
    found = search(task, calm, units, w.local, w.airports, air, [("optimum", optimum)], deadline,
                   crew_limit=crew_limit, air_limit=time_limit, tries=tries, log=log)
    if cover_limit <= 0:
        return Calm(task, optimum, found)
    fl = crewbound.Flights.of(task, calm.block, air.net.copies)
    reached = crewbound.reach(task, fl, units, w.local)
    least = crewbound.least(task, fl, units, w.local, crews.Ends(units, w.airports, task.date).least, reached)
    a, ar, lp, up, _ = optima(task, calm, air, air.notice, reached, time_limit)
    pieces, _ = lower(task, units, w.local, w.airports, air, fl, least, a, ar, lp, up, found, cover_limit, width, log)
    return Calm(task, optimum, found, pieces)


@dataclass
class Found:
    """The best day found: its plan, its replay in the crew sim (fill off), what the plan claimed it costs, and every
    candidate tried."""

    plan: dict
    day: crewsim.CrewDay
    claimed: float
    flat: bool
    attempts: list[dict] = field(default_factory=list)

    @property
    def value(self) -> float:
        return worth(self.day, self.flat)

    @property
    def exact(self) -> bool:
        """It replays at its claimed cost, every flight it flies with a legal crew."""
        return abs(self.value - self.claimed) <= 0.5 and not self.day.uncrewed


def worth(day: crewsim.CrewDay, flat: bool) -> float:
    return (day.run.cost.flat if flat else day.run.cost.total) + day.crew_cost


@dataclass
class Joint:
    """A day's best plan found and its lower bound: A*, the restricted A*, the search's best, the bound's pieces and
    value, and how much `reach` keeps."""

    optimum: mip.Solution
    restricted: mip.Solution
    found: Found
    pieces: dict[str, float | None]
    cover: crewbound.Covered | None
    reach: dict

    @property
    def bound(self) -> float:
        return max(v for v in self.pieces.values() if v is not None)

    @property
    def gap_pct(self) -> float:
        return 100 * (self.found.value - self.bound) / self.found.value if self.found.value else 0.0


def joint(task: Task, world: sim.World, units: Sequence[Unit], local: crews.Local, airports, air: Aircraft,
          fl: crewbound.Flights, reached: Mapping[int, np.ndarray], least: np.ndarray, budget: float,
          cover_limit: float = 300.0, flat: bool = False, crew_limit: float = 900.0, air_limit: float = 900.0,
          tries: int = 4, days: Sequence[tuple[str, Mapping]] = (), width: int = crewbound.WIDTH,
          log: Callable[[str], None] | None = None) -> Joint:
    """The best plan found within `budget` seconds (`days` among its candidates) and the bound, notice-priced or
    `flat`; `least` is each unit's least crew cost over `reached` (crewbound.least)."""
    t0 = time.monotonic()
    cost = air.cost(flat)
    a, ar, lp, up, keep = optima(task, world, air, cost, reached, air_limit)
    found = search(task, world, units, local, airports, air, [("optimum", a), ("restricted optimum", ar)],
                   t0 + budget, keep=keep, flat=flat, crew_limit=crew_limit, air_limit=air_limit, tries=tries,
                   days=days, log=log)
    pieces, covered = lower(task, units, local, airports, air, fl, least, a, ar, lp, up, found, cover_limit, width,
                            log)
    outside = sum(d is not None and d not in keep.get(i, ()) for i, d in a.deps.items())
    stats = {"minutes": sum(len(v) for v in reached.values()), "of": sum(len(v) for v in fl.minutes.values()),
             "flights_without": sum(not len(v) for v in reached.values()), "optimum_outside": outside}
    return Joint(a, ar, found, pieces, covered, stats)


def optima(task: Task, world: sim.World, air: Aircraft, cost: np.ndarray, reached: Mapping[int, np.ndarray],
           air_limit: float) -> tuple:
    """A*, the restricted A* (A* itself where it keeps to `reached`, or where the restricted MIP finds no plan in
    time), the LP whose prices the covering bound reads, its column bounds, and `reached` as sets."""
    a, lp = air.optimum(cost, air.ub, sim.execute(task, {}, world), air_limit)
    keep = {i: set(v.tolist()) for i, v in reached.items()}
    up = air.upper(keep=keep)
    outside = any(d is not None and d not in keep.get(i, ()) for i, d in a.deps.items())
    try:
        ar, lp = air.optimum(cost, up, None, air_limit) if outside else (a, air.relax(cost, up, air_limit))
    except RuntimeError:
        ar, up = a, air.ub
    return a, ar, lp, up, keep


def lower(task: Task, units: Sequence[Unit], local: crews.Local, airports, air: Aircraft, fl: crewbound.Flights,
          least: np.ndarray, a: mip.Solution, ar: mip.Solution, lp: mip.Relaxed | None, up: np.ndarray,
          found: "Found", cover_limit: float, width: int, log: Callable[[str], None] | None = None
          ) -> tuple[dict[str, float | None], crewbound.Covered | None]:
    """The bound's pieces, the covering bound's taken with `found`'s cost and started from its crew days."""
    pieces: dict[str, float | None] = {"aircraft": a.bound, "restricted": ar.bound,
                                       "restricted_least": ar.bound + float(least.sum()), "cover": None}
    covered = None
    if lp is not None and cover_limit > 0:
        z = lp.objective - crewbound.TOL - 1e-7 * abs(lp.objective)
        choices = crewbound.prices(air, lp, up, found.value - z)
        covered = crewbound.cover(task, fl, units, local, crews.Ends(units, airports, task.date), choices, cover_limit,
                                  width, found.day, log)
        pieces["cover"] = min(found.value, z + covered.bound)
    return pieces, covered


def search(task: Task, world: sim.World, units: Sequence[Unit], local: crews.Local, airports, air: Aircraft,
           starts: Sequence[tuple[str, mip.Solution]], deadline: float,
           keep: Mapping[int, Collection[int]] | None = None, flat: bool = False, crew_limit: float = 900.0,
           air_limit: float = 900.0, tries: int = 4, days: Sequence[tuple[str, Mapping]] = (),
           log: Callable[[str], None] | None = None) -> Found:
    """The cheapest day found from the aircraft optima `starts` (label, solution) and the plans `days` (label, plan,
    crewed by the crew sim's default rule) by the time `deadline` (monotonic), the aircraft MIP kept to `keep` once
    flights are held."""
    end = lambda u, s: crews.reposition(u, s, airports, task.date)  # noqa: E731
    ends = crews.Ends(units, airports, task.date)
    best: list[Found] = []
    attempts: list[dict] = []
    pool: list[tuple[crewreplay.Column, tuple[tuple[int, int], ...]]] = []
    last: set[tuple] = set()

    def left() -> float:
        return deadline - time.monotonic()

    def keep_best(plan: dict, day: crewsim.CrewDay, claimed: float) -> None:
        if not best or worth(day, flat) < best[0].value - 1e-6:
            best[:] = [Found(plan, day, claimed, flat, attempts)]

    def default(plan: Mapping, label: str) -> crewsim.CrewDay:
        """The plan crewed by the crew sim's default rule, as a plan naming what it did; its day."""
        filled = crewsim.execute(task, plan, units, world, local, airports=airports)
        named = crewday.named(filled)
        day = crewsim.execute(task, named, units, world, local, fill=False, airports=airports)
        attempts.append({"try": label, "claimed": round(worth(filled, flat), 1), "replayed": round(worth(day, flat), 1),
                         "sim_uncrewed": len(day.uncrewed)})
        keep_best(named, day, worth(filled, flat))
        return filled

    def consider(sol: mip.Solution, label: str, limit: float | None = None) -> tuple[crewreplay.Solved,
                                                                                 crewsim.CrewDay]:
        deps = {i: d for i, d in sol.deps.items() if d is not None}
        p = crewreplay.Problem(crewday.task_legs(task, deps, world.block), units, task.date, local, end,
                               call=task.start, ends=ends)
        filled = default(sol.plan(task.start), f"{label}, default crews")
        kept = [c for c, at in pool if all(i in p.leg and p.leg[i].dep == d for i, d in at)]
        starts = [plan for plan in (crewday.columns_of(p, filled), {c.unit: c for c in kept if c.key in last}) if plan]
        span, polish = (min(crew_limit, left()), min(crew_limit / 2, left() / 6)) if limit is None else \
            (limit / 2, limit / 3)
        crewed = crewreplay.solve(p, max(30.0, span), polish=max(10.0, polish), starts=starts, pool=kept, log=log)
        pool[:] = [(c, tuple((i, p.leg[i].dep) for i, _ in c.steps)) for c in crewed.pool]
        last.clear()
        last.update(c.key for c in crewed.columns.values())
        plan = crewday.plan_of(p, crewed, sol.plan(task.start))
        claimed = sol.objective + crewed.crew_cost
        day = crewsim.execute(task, plan, units, world, local, fill=False, airports=airports)
        exact = worth(day, flat)
        if crewed.uncrewed:
            filled = crewsim.execute(task, plan, units, world, local, airports=airports)
            plan, claimed = crewday.named(filled), worth(filled, flat)
            day = crewsim.execute(task, plan, units, world, local, fill=False, airports=airports)
        attempts.append({"try": label, "aircraft": sol.objective, "aircraft_status": sol.status,
                         "aircraft_s": sol.wall_s, "crew": crewed.summary(), "crew_cost": round(crewed.crew_cost, 1),
                         "planned": round(sol.objective + crewed.crew_cost, 1), "exact": round(exact, 1),
                         "claimed": round(claimed, 1), "replayed": round(worth(day, flat), 1),
                         "sim_uncrewed": len(day.uncrewed), "uncrewed": list(crewed.uncrewed)})
        if log:
            log(f"{label}: aircraft {sol.objective} crew {crewed.crew_cost:.0f} uncrewed {len(crewed.uncrewed)} "
                f"replayed {worth(day, flat):.0f}")
        keep_best(plan, day, claimed)
        if crewed.uncrewed:
            cancel = {i: {"cancel": True, "decided": task.start} for i in crewed.uncrewed}
            default(sol.plan(task.start) | cancel, f"{label}, its uncrewed cancelled, default crews")
        return crewed, day

    for label, plan in days:
        default(plan, label)
    cost = air.cost(flat)
    crewed = day = None
    seen = []
    for label, sol in starts:
        if (sol.deps, sol.lapsed) in seen or (best and left() <= 0):
            continue
        seen.append((sol.deps, sol.lapsed))
        crewed, day = consider(sol, label)
    held: dict[int, int] = {}
    dropped: set[int] = set()
    for n in range(tries):
        if not crewed.uncrewed or left() <= 0:
            break
        for i in crewed.uncrewed:
            r = day.run.flights[i]
            if r.dep is None or held.get(i) == r.dep:
                dropped.add(i)
                held.pop(i, None)
            else:
                held[i] = max(held.get(i, 0), r.dep)
        sol = held_solve(task, world, air, cost, held, dropped, set(), keep, min(air_limit, max(30.0, left())))
        crewed, day = consider(sol, f"held {n + 1}")
    fly: set[int] = set()
    tried: set[int] = set()
    while left() > 0:
        out = stranding(units, best[0].day, tried)
        if not out:
            break
        i = out[0]
        tried.add(i)
        before = best[0].value
        sol = held_solve(task, world, air, air.cost(flat, fly | {i}), held, dropped - {i}, fly | {i}, keep,
                         min(air_limit, max(30.0, left())), cost)
        consider(sol, f"fly {task.flight(i).flight}")
        if best[0].value < before - 1e-6:
            fly.add(i)
            dropped.discard(i)
    if left() > 60:
        consider(as_solution(task, best[0].day, flat), "best day, crews again", left())
    return best[0]


def as_solution(task: Task, day: crewsim.CrewDay, flat: bool) -> mip.Solution:
    """A crew sim day's departures, cancellations and auto-cancels as an aircraft solution at its aircraft cost."""
    run = day.run
    deps = {i: r.dep if r.status == FLOWN else None for i, r in run.flights.items()}
    lapsed = {i: task.flight(i).deadline + 1 for i, r in run.flights.items() if r.status == AUTO_CANCELLED}
    cost = run.cost.flat if flat else run.cost.total
    return mip.Solution("day", cost, cost, 0.0, 0.0, deps, 0, 0, 0, False, False, lapsed)


def held_solve(task: Task, world: sim.World, air: Aircraft, cost: np.ndarray, held: Mapping[int, int],
               dropped: Collection[int], fly: Collection[int], keep: Mapping[int, Collection[int]] | None,
               time_limit: float, value: np.ndarray | None = None) -> mip.Solution:
    """The aircraft optimum with flights held, dropped, kept to their minutes in `keep` and priced to fly, started
    from the day those holds give where it keeps to them; without `keep` if no plan is found in time with it (the
    start then holds)."""
    plan = {i: {"not_before": d} for i, d in held.items()} | {i: {"cancel": True} for i in dropped}
    start = sim.execute(task, plan, world)
    try:
        sol, _ = air.solve(cost, air.upper(held, dropped, keep), start, time_limit, value)
    except RuntimeError:
        sol, _ = air.solve(cost, air.upper(held, dropped), start, time_limit, value)
    return sol


def stranding(units: Sequence[Unit], day: crewsim.CrewDay, skip: Collection[int]) -> list[int]:
    """The flights the day cancels (not in `skip`), those whose planned crews cost the day most first."""
    spent = {uid: sum(r["cost"].values()) for uid, r in day.units.items()}
    weight: dict[int, float] = defaultdict(float)
    for u in units:
        for i in u.trip:
            weight[i] += spent.get(u.id, 0.0)
    out = [i for i, r in day.run.flights.items() if r.status != FLOWN and i not in skip]
    return sorted(out, key=lambda i: (-weight[i], i))
