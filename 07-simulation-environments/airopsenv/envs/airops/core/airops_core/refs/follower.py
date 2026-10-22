"""The follower: the MIP re-solved at every watch on the day as known, freezing as the env does, its final plan then
run on the truth.

At watch t the flights frozen so far (core.known.frozen on the plan run on the day as known, as the env freezes) are
pinned where the plan runs on the forecast day; the others may leave from t + FREEZE on (from the start at the first
watch). Unknown delays and caps are forecast by persistence: the mean delay of the flights due in the last hour and
the next LOOK minutes from the same origin (else from anywhere), and the mean of the last three known hours of each
cap. ``blind`` plans on the day as known as it is: no delay, and the nominal schedule for caps.
"""

import math
import statistics
from dataclasses import dataclass

from ..core.known import FREEZE, LOOK, cap_known, frozen, known_caps, known_lb, lb_known, view, watch_times
from ..core.model import Caps, Task
from ..core.sim import FLOWN, Execution, execute
from .mip import solve

PERSIST, BLIND = "persist", "blind"
WATCH_GAP = 0.0


@dataclass(frozen=True)
class Followed:
    plan: dict[int, dict]
    run: Execution
    watches: tuple[dict, ...]

    @property
    def cost(self) -> int:
        return self.run.cost.total

    def stored(self) -> dict[str, dict]:
        return {str(i): {"cancel": int(v["cancel"]), "dep": v["not_before"]} for i, v in sorted(self.plan.items())}


def follow(task: Task, mode: str = PERSIST, time_limit: float = 600.0, gap: float = WATCH_GAP,
           threads: int = 1) -> Followed:
    plan: dict[int, dict] = {}
    pinned: set[int] = set()
    log = []
    for t in watch_times(task):
        pinned |= frozen(task, execute(task, plan, *view(task, t)), t)
        lb, caps = forecast_lb(task, t, mode), forecast_caps(task, t, mode)
        run = execute(task, plan, lb, caps)
        fixed = {i: run.flights[i].dep if run.flights[i].status == FLOWN else None for i in pinned}
        sol = solve(task, lb, caps, fixed, earliest=t + FREEZE if t > task.start else None, time_limit=time_limit,
                    gap=gap, threads=threads, canonical=True)
        plan = {i: plan[i] if i in pinned else d for i, d in sol.plan.items()}
        log.append({"t": t, "frozen": len(pinned), "planned": sol.objective, "status": sol.status,
                    "gap_pct": sol.gap_pct, "wall_s": sol.wall_s})
    return Followed(plan, execute(task, plan), tuple(log))


def forecast_lb(task: Task, t: int, mode: str) -> dict[int, int]:
    lb = known_lb(task, t)
    if mode == BLIND:
        return lb
    recent = [f for f in task.flights if t - 60 <= f.sched_dep <= t + LOOK]
    out = {}
    for f in task.flights:
        if lb_known(f.sched_dep, t):
            out[f.id] = lb[f.id]
            continue
        pool = [g.lb for g in recent if g.orig == f.orig] or [g.lb for g in recent]
        out[f.id] = round(statistics.mean(pool)) if pool else 0
    return out


def forecast_caps(task: Task, t: int, mode: str) -> Caps:
    caps = known_caps(task, t)
    if mode == BLIND:
        return caps
    out = {}
    for key, hours in caps.items():
        truth = task.caps.get(key, {})
        last = [truth[h] for h in sorted(h for h in truth if cap_known(h, t))[-3:]]
        guess = math.ceil(statistics.mean(last)) if last else None
        out[key] = {h: v if cap_known(h, t) or guess is None else guess for h, v in hours.items()}
    return out
