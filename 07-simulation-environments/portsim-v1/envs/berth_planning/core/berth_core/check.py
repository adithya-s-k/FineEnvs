"""Feasibility and cost of a plan. Pure Python: the server grades with this and needs no solver."""

from __future__ import annotations

from dataclasses import dataclass, field

from .model import MOVE_PENALTY, Plan, Task


@dataclass
class ShipResult:
    ship: int
    berth_hour: int | None
    section: int | None
    departure: int | None
    delay_h: int
    moved: bool
    cost: int
    problems: list[str] = field(default_factory=list)
    cranes: int | None = None


@dataclass
class PlanResult:
    feasible: bool
    ships: list[ShipResult]
    cost: int | None           # None when the plan is infeasible
    delay_cost: int | None
    moves: int | None

    @property
    def clean_fraction(self) -> float:
        return sum(1 for s in self.ships if not s.problems) / max(1, len(self.ships))

    @property
    def violations(self) -> list[dict]:
        return [{"ship": s.ship, "problem": p} for s in self.ships for p in s.problems]


def _span(a0: int, a1: int) -> str:
    return f"{a0}" if a0 == a1 else f"{a0}-{a1}"


def _hours(hs: list[int]) -> str:
    """[3, 4, 5, 9] -> '3-5, 9'"""
    out, run = [], []
    for h in sorted(set(hs)):
        if run and h == run[-1] + 1:
            run.append(h)
        else:
            if run:
                out.append(_span(run[0], run[-1]))
            run = [h]
    if run:
        out.append(_span(run[0], run[-1]))
    return ", ".join(out[:4]) + (" …" if len(out) > 4 else "")


def evaluate(task: Task, plan: Plan) -> PlanResult:
    """Check every ship of `plan` against the task's rules and price it.

    Rules: berth at or after arrival; stay within the quay; never share a section-hour with another ship, a ship
    already alongside, or a closed section. On crane-rule tasks also: cranes within the ship's min-max, cranes in use
    at every hour within the pool (less any outage), and at most max_moves_per_hour berthings + departures per hour.
    Cost: sum over ships of sections x hours late against the planned departure, plus MOVE_PENALTY for each
    scheduled ship placed on different sections than planned."""
    lo, hi = task.first_section, task.last_section
    out: list[ShipResult] = []
    rects: dict[int, tuple[int, int, int, int]] = {}
    for s in task.ships:
        if s.id not in plan:
            out.append(ShipResult(s.id, None, None, None, 0, False, 0, ["missing from the plan"]))
            continue
        entry = plan[s.id]
        h, sec = entry[0], entry[1]
        c = entry[2] if task.cranes and len(entry) > 2 else (s.std_cranes if task.cranes else None)
        bad_cranes = task.cranes and not (s.min_cranes <= c <= s.max_cranes)
        work = s.handling_for(s.std_cranes if bad_cranes else c)   # never divide by a zero or negative count
        dep = task.departure(s, h + work) if task.cranes else h + work
        stay = dep - h
        r = ShipResult(s.id, h, sec, dep, max(0, dep - s.due),
                       s.planned_section is not None and sec != s.planned_section, 0, cranes=c)
        if task.cranes:
            for a, b in task.no_move_windows(s):
                if a <= h < b:
                    r.problems.append(f"berths at hour {h}, inside the no-movement window {a}-{b}")
        if bad_cranes:
            r.problems.append(f"gets {c} cranes but can be worked by {s.min_cranes}-{s.max_cranes}")
        if h < s.arrival:
            r.problems.append(f"berths at hour {h} but cannot arrive before hour {s.arrival}")
        last = sec + s.sections - 1
        if sec < lo or last > hi:
            r.problems.append(f"needs sections {_span(sec, last)} but the quay has sections {lo}-{hi}")
        for b in task.blocks:
            if sec <= b.last and b.first <= last and h < b.end and b.start < h + stay:
                what = f"{b.label} (alongside)" if b.kind == "alongside" else "closed sections"
                r.problems.append(f"overlaps {what} {_span(b.first, b.last)} during hours {b.start}-{b.end}")
        rects[s.id] = (h, h + stay, sec, last, h + work)
        out.append(r)
    ids = sorted(rects)
    for i, a in enumerate(ids):
        t0, t1, m0, m1 = rects[a][:4]
        for b in ids[i + 1:]:
            u0, u1, n0, n1 = rects[b][:4]
            if t0 < u1 and u0 < t1 and m0 <= n1 and n0 <= m1:
                hrs, secs = f"{max(t0, u0)}-{min(t1, u1)}", _span(max(m0, n0), min(m1, n1))
                out[a].problems.append(f"overlaps ship {b} ({task.ships[b].name}) in sections {secs} during hours {hrs}")
                out[b].problems.append(f"overlaps ship {a} ({task.ships[a].name}) in sections {secs} during hours {hrs}")
    if task.cranes:
        _crane_and_move_limits(task, out, rects)
    feasible = not any(r.problems for r in out)
    for r in out:
        if r.berth_hour is not None:
            r.cost = ship_delay_cost(task.ships[r.ship], r.berth_hour, r.delay_h) + (MOVE_PENALTY if r.moved else 0)
    if not feasible:
        return PlanResult(False, out, None, None, None)
    delay = sum(ship_delay_cost(task.ships[r.ship], r.berth_hour, r.delay_h) for r in out)
    moves = sum(r.moved for r in out)
    return PlanResult(True, out, delay + MOVE_PENALTY * moves, delay, moves)


def ship_delay_cost(s, berth_hour: int, delay_h: int) -> int:
    """sections x hours late x weight, plus the emergency penalty for berthing after a genuine deadline."""
    cost = s.sections * delay_h * s.weight
    if s.berth_deadline is not None and berth_hour > s.berth_deadline:
        cost += s.deadline_penalty * (berth_hour - s.berth_deadline)
    return cost


def _crane_and_move_limits(task: Task, out: list[ShipResult], rects: dict) -> None:
    """Crane pool and movement limit, hour by hour; a ship involved in an over-limit hour gets one problem line."""
    use: dict[int, int] = {}
    moves: dict[int, list[int]] = {}
    fixed_moves: dict[int, int] = {}
    for b in task.blocks:
        if b.kind == "alongside":
            for t in range(b.start, b.end):
                use[t] = use.get(t, 0) + b.cranes
            fixed_moves[b.end] = fixed_moves.get(b.end, 0) + 1
    for sid, (t0, t1, _, _, tw) in rects.items():
        c = out[sid].cranes or 0
        for t in range(t0, tw):          # cranes work until the ship finishes, not while it waits for the wind
            use[t] = use.get(t, 0) + c
        moves.setdefault(t0, []).append(sid)
        moves.setdefault(t1, []).append(sid)
    over = {t for t, u in use.items() if u > task.crane_pool_at(t)}
    if over:
        for sid, (t0, t1, _, _, tw) in rects.items():
            hs = [t for t in range(t0, tw) if t in over]
            if hs:
                worst = max(hs, key=lambda t: use[t] - task.crane_pool_at(t))
                out[sid].problems.append(
                    f"cranes over the pool at hours {_hours(hs)} (e.g. hour {worst}: {use[worst]} in use, "
                    f"{task.crane_pool_at(worst)} available)")
    cap = task.rules.get("max_moves_per_hour")
    if cap:
        for t, ids in sorted(moves.items()):
            n = len(ids) + fixed_moves.get(t, 0)
            if n > cap:
                for sid in ids:
                    kind = "berths" if rects[sid][0] == t else "leaves"
                    out[sid].problems.append(f"{kind} at hour {t}, when {n} ships move (limit {cap} per hour)")


def plan_cost(task: Task, plan: Plan) -> int:
    res = evaluate(task, plan)
    if not res.feasible:
        raise ValueError("plan is infeasible: " + "; ".join(v["problem"] for v in res.violations[:3]))
    return res.cost


def unavoidable_cost(task: Task) -> int:
    """A lower bound on any plan's cost: each ship on its own, docking at its earliest legal hour (its arrival, or the
    end of a no-movement window it arrives in) with the most cranes it can take. No plan can do better for that ship,
    so the sum is a floor under the optimum; the reward measures mistakes against the cost above this floor."""
    total = 0
    for s in task.ships:
        h = s.arrival
        if task.cranes:
            for a, b in task.no_move_windows(s):
                if a <= h < b:
                    h = b
            dep = task.departure(s, h + s.handling_for(s.max_cranes))
        else:
            dep = h + s.handling
        total += ship_delay_cost(s, h, max(0, dep - s.due))
    return total
