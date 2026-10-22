"""The day as known at a watch (DESIGN.md, Information): delays LOOK minutes before departure, slot caps CAP_LOOK
minutes before the hour (JetBlue's nominal schedule until then), news as issued, and the freeze."""

from .model import Caps, News, Task
from .sim import FLOWN, Execution

LOOK, CAP_LOOK, FREEZE, STEP = 120, 240, 60, 60


def lb_known(sched_dep: int, t: int) -> bool:
    return sched_dep <= t + LOOK


def cap_known(hour: int, t: int) -> bool:
    return 60 * hour <= t + CAP_LOOK


def known_lb(task: Task, t: int) -> dict[int, int]:
    return {f.id: f.lb if lb_known(f.sched_dep, t) else 0 for f in task.flights}


def known_caps(task: Task, t: int) -> Caps:
    out = {}
    for key in dict.fromkeys([*task.caps, *task.nominal]):
        hours = {h: v for h, v in task.caps.get(key, {}).items() if cap_known(h, t)}
        hours |= {h: v for h, v in task.nominal.get(key, {}).items() if not cap_known(h, t)}
        out[key] = dict(sorted(hours.items()))
    return out


def news_until(task: Task, t: int) -> list[News]:
    return [n for n in task.news if n.at <= t]


def watch_times(task: Task) -> list[int]:
    """Every STEP minutes from the start, through the first watch that knows every delay and cap (later ones could
    only freeze), and while some flight could still leave unfrozen (t + FREEZE <= its scheduled departure + dmax)."""
    hours = [h for caps in (task.caps, task.nominal) for c in caps.values() for h in c]
    reveal = max([f.sched_dep - LOOK for f in task.flights] + [60 * h - CAP_LOOK for h in hours])
    last = max(f.sched_dep + f.dmax for f in task.flights) - FREEZE
    ts = [task.start]
    while ts[-1] < reveal and ts[-1] + STEP <= last:
        ts.append(ts[-1] + STEP)
    return ts


def frozen(task: Task, run: Execution, t: int) -> set[int]:
    """Flights ``run`` (the plan on the day as known at watch t) has leaving before t + FREEZE, cancelled ones by
    their schedule; none at the first watch, where the planner sets the day up. Callers keep the union over watches."""
    if t <= task.start:
        return set()
    return {i for i, r in run.flights.items()
            if (r.dep if r.status == FLOWN else task.flight(i).sched_dep) < t + FREEZE}


def view(task: Task, t: int) -> tuple[dict[int, int], Caps]:
    return known_lb(task, t), known_caps(task, t)

