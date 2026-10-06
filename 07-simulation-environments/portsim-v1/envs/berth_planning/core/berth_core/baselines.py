"""Cheap policies a model could imitate without planning. Used to check that a task is hard and that the reward
does not pay for laziness: every task should give these low rewards.

    published     submit the broken published plan as is (usually infeasible)
    naive         the stored quick fix: keep slots that work, push the rest (solve.naive_replan)
    greedy        arrival order, earliest free hour at the nearest section, the crane count that finishes soonest
    serial        every ship in arrival order, one after another on the first sections, far from any conflict
"""

from __future__ import annotations

from .model import Plan, Task
from .solve import _Ledger, naive_replan


def published(task: Task) -> Plan:
    plan = task.planned()
    for s in task.ships:
        if s.id not in plan:
            plan[s.id] = (s.arrival, task.first_section) + ((s.std_cranes,) if task.cranes else ())
    return plan


def greedy(task: Task) -> Plan:
    led = _Ledger(task)
    plan: Plan = {}
    for s in sorted(task.ships, key=lambda s: (s.arrival, s.id)):
        anchor = s.planned_section if s.planned_section is not None else task.first_section
        positions = sorted(range(task.first_section, task.last_section - s.sections + 2), key=lambda k: (abs(k - anchor), k))
        options = range(s.max_cranes, s.min_cranes - 1, -1) if task.cranes else [None]
        h = s.arrival
        while True:
            best = None
            for c in options:
                sec = next((k for k in positions if led.fits(s, h, k, c)), None)
                if sec is not None:
                    best = (sec, c)
                    break
            if best:
                break
            h += 1
        led.add(s, h, best[0], best[1])
        plan[s.id] = (h, best[0]) + ((best[1],) if task.cranes else ())
    return plan


def serial(task: Task) -> Plan:
    """Ships one at a time in arrival order on the first sections: valid by construction, very late."""
    led = _Ledger(task)
    plan: Plan = {}
    t = 0
    for s in sorted(task.ships, key=lambda s: (s.arrival, s.id)):
        c = s.std_cranes if task.cranes else None
        h = max(t, s.arrival)
        while not led.fits(s, h, task.first_section, c):
            h += 1
        led.add(s, h, task.first_section, c)
        plan[s.id] = (h, task.first_section) + ((c,) if task.cranes else ())
        t = h + 1
    return plan


POLICIES = {"published": published, "naive": naive_replan, "greedy": greedy, "serial": serial}
