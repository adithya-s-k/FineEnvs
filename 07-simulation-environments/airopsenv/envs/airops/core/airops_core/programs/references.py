"""The references: plans played on the truth through sim.execute and priced by cost.py (passengers by notice, the
airline's aircraft-minutes at the FAA's rates).

- anchor: the hindsight MIP on the truth, every cancellation priced at the first watch, proven and canonical;
- follower: at every watch the MIP on the day as known (unknown delays 0; the programs as issued, a proposed GDP
  included), its plan frozen as the env freezes: a flight the plan, run on that day, has leaving before t + FREEZE
  commits to fly; a projected auto-cancel stays open until FREEZE before the minute it is decided (its deadline, or its
  crew deadline when its crew's rest would end past that, Task.auto_at), and is priced and pinned there as an
  auto-cancel, which moves no aircraft through a break; a cancellation is final from the watch that confirms it, priced
  there and fixed in every later solve. A flight the solve lets auto-cancel is held past its deadline, not cancelled,
  so the sim decides it as the solve did. Its plan holds a flight only where its last solve has it leaving later than it
  would with no hold on the day as known (`deliberate`): an EDCT or a stop the forecast expects is the FAA's, not the
  desk's, so on the truth a flight the desk did not hold leaves as soon as its aircraft, its crew, the stops and its
  slot allow. Every watch solve is canonical (the earliest optimal departures, v1's) and starts from the plan's own
  projection, so no truth reaches the search;
- program-blind follower: the follower with every program removed from its model (the nominal caps stay);
- base-rate follower (`early`, off by default): the follower expecting every GDP to end `early` minutes before its
  as-issued end, never before the watch, as GDPs on other days did by the median; its stops run as issued;
- rule: at every watch, cancel what the plan run on the day as known auto-cancels while it is still open (FREEZE or
  more before its auto-cancel is decided) and not committed (a flight an earlier watch had leaving before its freeze
  line commits to fly, as in the follower);
- naive: the empty plan;
- airline replay: JetBlue's cancellations decided at sched - NOTICE, its recorded departures as holds and its realised
  arrival order as slot claims (each claims the slot of its recorded departure plus its scheduled block); auto-cancels
  are counted over the flights it flew by their deadline, which covers dmax and the hub cutoff, less those whose
  aircraft a leg it flew past its deadline brought (`counted`); a flight it flew past its crew deadline still counts,
  which is what tells a crew model too strict for JetBlue's day;
- recorded day: JetBlue's real outcome priced directly, its aircraft's minutes on its recorded departures and blocks,
  cancellations decided at sched - NOTICE, plus end-of-day ferries from its recorded end positions to tomorrow's needs.
  A diverted flight that never reached its destination counts as cancelled when it left, the moment the desk knew;
- calm day: from the positions the schedule assumes, every aircraft ready at the start and tomorrow's needs following
  them (network.scheduled), no delay, scheduled blocks, no programs: the hindsight optimum's forced cancellations and
  ferries; with `record`, from the record's positions and the task's needs, as the storm day starts.
Like for like, nothing is decided before the first watch: a JetBlue cancellation due earlier than the start counts as
decided there, as the anchor's and every desk's do. At a watch the day so far is history, as the env shows it: a
flight that has left on the truth leaves when it did in the follower's and the rule's projection and is pinned there in
the follower's MIP, and every other flight leaves no earlier than the watch. No reference may cost less than the
anchor (`below_anchor`).
"""

from collections.abc import Mapping
from dataclasses import dataclass

from ..core.known import FREEZE, LOOK, STEP
from ..core.sim import AUTO_CANCELLED, CANCELLED, FLOWN
from . import mip, network, sim
from .advisories import HUBS
from .cost import NOTICE, Cost, cancel_price, priced
from .model import Flight, Task

TIME_LIMIT = 2400.0


@dataclass(frozen=True)
class Followed:
    plan: dict[int, dict]
    run: sim.Execution
    watches: tuple[dict, ...]

    @property
    def cost(self) -> int:
        return self.run.cost.total


@dataclass(frozen=True)
class Replay:
    run: sim.Execution
    flew: int
    auto: int


@dataclass(frozen=True)
class Calm:
    solution: mip.Solution
    run: sim.Execution
    task: Task

    @property
    def cancelled(self) -> int:
        return sum(d is None for d in self.solution.deps.values())

    @property
    def ferries(self) -> int:
        return len(self.run.ferries)


def anchor(task: Task, time_limit: float = TIME_LIMIT) -> tuple[mip.Solution, sim.Execution]:
    sol = mip.solve(task, time_limit=time_limit, start=naive(task))
    return sol, sim.execute(task, sol.plan(task.start))


def flat_anchor(task: Task, time_limit: float = TIME_LIMIT, start: sim.Execution | None = None) -> mip.Solution:
    """The hindsight optimum with every cancellation, the desk's or an auto-cancel, at the flat 523 per passenger,
    started from `start` (a day on the truth, naive by default)."""
    return mip.solve(task, flat=True, time_limit=time_limit, canonical=False,
                     start=naive(task) if start is None else start)


def follow(task: Task, programs: bool = True, time_limit: float = TIME_LIMIT, canonical: bool = True,
           early: int = 0) -> Followed:
    plan: dict[int, dict] = {}
    pinned: set[int] = set()
    log = []
    for t in watch_times(task):
        world = sim.known(task, t, programs, early)
        run = sim.execute(task, so_far(task, plan, t), world)
        if t > task.start:
            pinned |= frozen(run, t)
        fixed = {i: run.flights[i].dep if run.flights[i].status == FLOWN else None for i in pinned}
        lapsed = {i for i in pinned if run.flights[i].status == AUTO_CANCELLED}
        price = {f.id: _price(f, plan.get(f.id, {}), task.auto_at(f) if f.id in pinned else t) for f in task.flights}
        sol = mip.solve(task, world, price, fixed, t + FREEZE if t > task.start else None, time_limit=time_limit,
                        canonical=canonical, start=run, lapsed=lapsed)
        if not sol.proven:
            raise RuntimeError(f"{task.task_id}: watch {t} solve {sol.status} at gap {sol.gap_pct}%")
        cancelled = sorted(i for i, d in sol.deps.items() if i not in pinned and d is None and i not in sol.lapsed)
        plan = {i: v for i, v in plan.items() if i in pinned} | {i: {"cancel": True, "decided": t} for i in cancelled}
        pinned |= set(cancelled)
        deps = {i: sol.lapsed.get(i, d) for i, d in sol.deps.items() if i not in pinned}
        held = deliberate(task, plan, deps, world, t)
        plan |= {i: {"not_before": d} for i, d in held.items()}
        log.append({"t": t, "frozen": len(pinned), "cancelled": cancelled, "held": len(held),
                    "planned": sol.objective, "status": sol.status, "gap_pct": sol.gap_pct,
                    "canonical": sol.canonical, "wall_s": sol.wall_s, "cols": sol.cols, "rows": sol.rows})
    return Followed(plan, sim.execute(task, plan), tuple(log))


def deliberate(task: Task, plan: Mapping[int, Mapping], deps: Mapping[int, int], world: sim.World,
               t: int) -> dict[int, int]:
    """The holds of a solve at watch t: each open flight's departure in `deps` that is later than the one `plan`
    (without a hold on any open flight) gives it on the day as known."""
    free = sim.execute(task, so_far(task, plan, t), world).flights
    return {i: d for i, d in deps.items() if free[i].status != FLOWN or d > free[i].dep}


def rule(task: Task) -> Followed:
    plan: dict[int, dict] = {}
    committed: set[int] = set()
    log = []
    for t in watch_times(task):
        run = sim.execute(task, so_far(task, plan, t), sim.known(task, t))
        doomed = [i for i, r in run.flights.items()
                  if r.status == AUTO_CANCELLED and i not in committed and r.decided >= t + FREEZE]
        for i in doomed:
            plan[i] = {"cancel": True, "decided": t}
        if t > task.start:
            committed |= {i for i, r in run.flights.items() if r.status == FLOWN and r.dep < t + FREEZE}
        log.append({"t": t, "cancelled": doomed})
    return Followed(plan, sim.execute(task, plan), tuple(log))


def naive(task: Task) -> sim.Execution:
    return sim.execute(task, {})


def airline_plan(task: Task) -> dict[int, dict]:
    return {f.id: {"cancel": True, "decided": max(f.sched_dep - NOTICE, task.start)} if f.actual.dep is None
            else {"not_before": f.actual.dep, "claim": f.actual.dep + f.sched_block} for f in task.flights}


def airline(task: Task) -> Replay:
    run = sim.execute(task, airline_plan(task))
    flew = counted(task)
    return Replay(run, len(flew), sum(run.flights[i].status == AUTO_CANCELLED for i in flew))


def counted(task: Task) -> list[int]:
    """The flights JetBlue flew by their deadline, less every later leg of a tail once it flew one past it."""
    out, late = [], set()
    for f in sorted((f for f in task.flights if f.actual.dep is not None), key=lambda f: (f.actual.dep, f.id)):
        if f.actual.tail in late:
            continue
        if f.actual.dep > f.deadline:
            late.add(f.actual.tail)
        else:
            out.append(f.id)
    return sorted(out)


def recorded(task: Task) -> tuple[Cost, int]:
    """JetBlue's recorded day like for like, and its end-of-day ferries."""
    dep, arr, decided = {}, {}, {}
    for f in task.flights:
        a = f.actual
        dep[f.id], arr[f.id] = a.dep, a.arr if a.dep is not None and a.arr is not None else None
        if arr[f.id] is None:
            decided[f.id] = max(f.sched_dep - NOTICE, task.start) if a.dep is None else a.dep
    out = sim.ferries(task, {t.tail: t.actual_end for t in task.tails}, {t.tail: t.family for t in task.tails})
    return priced(task, dep, arr, decided, [(x.family, x.orig, x.dest) for x in out]), len(out)


def calm(task: Task, time_limit: float = TIME_LIMIT, record: bool = False) -> Calm:
    task = task if record else network.scheduled(task)
    world = sim.calm(task)
    sol = mip.solve(task, world, time_limit=time_limit, canonical=False, start=sim.execute(task, {}, world))
    return Calm(sol, sim.execute(task, sol.plan(task.start), world), task)


def watch_times(task: Task) -> list[int]:
    """Every STEP minutes from the start, through the first watch that knows every delay, while some flight could
    still leave unfrozen (t + FREEZE <= its deadline)."""
    reveal = max(f.sched_dep - LOOK for f in task.flights)
    last = max(f.deadline for f in task.flights) - FREEZE
    ts = [task.start]
    while ts[-1] < reveal and ts[-1] + STEP <= last:
        ts.append(ts[-1] + STEP)
    return ts


def frozen(run: sim.Execution, t: int) -> set[int]:
    """What `run` (the plan on the day as known at watch t) commits: flights leaving before t + FREEZE, every
    cancellation, and projected auto-cancels decided within FREEZE."""
    return {i for i, r in run.flights.items()
            if (r.status == FLOWN and r.dep < t + FREEZE) or r.status == CANCELLED
            or (r.status == AUTO_CANCELLED and r.decided < t + FREEZE)}


def so_far(task: Task, plan: Mapping[int, Mapping], t: int, run: sim.Execution | None = None) -> dict[int, dict]:
    """`plan` as the env runs it at watch t: a flight that has left on the truth (in `run`, the plan's day by
    default) leaves when it did, every other open one no earlier than t."""
    run = sim.execute(task, plan) if run is None else run
    out = {}
    for f in task.flights:
        entry, dep = plan.get(f.id, {}), run.flights[f.id].dep
        if dep is not None and dep < t:
            out[f.id] = {"left": dep}
        elif entry.get("cancel"):
            out[f.id] = dict(entry)
        else:
            out[f.id] = {**entry, "not_before": max(entry.get("not_before") or 0, t)}
    return out


def below_anchor(costs: Mapping[str, int], anchor_cost: int) -> dict[str, int]:
    return {k: v for k, v in costs.items() if v < anchor_cost}


def seats_ok(task: Task, run: sim.Execution) -> bool:
    """pax <= seats of the aircraft that flew it, for every flown flight."""
    seats = {t.tail: t.seats for t in task.tails}
    return all(task.flight(i).pax <= seats[r.tail] for i, r in run.flights.items() if r.status == FLOWN)


def past_midnight(task: Task, run: sim.Execution | None = None) -> int:
    """Hub movements (departures from or arrivals at JFK or BOS) after 24:00: the run's, or JetBlue's recorded."""
    out = 0
    for f in task.flights:
        dep, arr = (f.actual.dep, f.actual.arr) if run is None else (run.flights[f.id].dep, run.flights[f.id].arr)
        out += dep is not None and dep >= 1440 and f.orig in HUBS
        out += arr is not None and arr >= 1440 and f.dest in HUBS
    return out


def _price(f: Flight, entry: Mapping, at: int) -> int:
    return cancel_price(f, entry["decided"] if entry.get("cancel") else at)
