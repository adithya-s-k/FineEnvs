"""Reference plans: the naive re-plan a hurried planner would make, and the CP-SAT optimum.

Both handle the crane rules when a task has them (crane count per ship, crane pool with outages, movements per hour).
"""

from __future__ import annotations

from .model import MOVE_PENALTY, Plan, Task


class _Ledger:
    """Occupied rectangles, cranes in use per hour and movements per hour, for greedy placement."""

    def __init__(self, task: Task):
        self.task = task
        self.rects = [(b.start, b.end, b.first, b.last) for b in task.blocks]
        self.use: dict[int, int] = {}
        self.moves: dict[int, int] = {}
        for b in task.blocks:
            if b.kind == "alongside":
                for t in range(b.start, b.end):
                    self.use[t] = self.use.get(t, 0) + b.cranes
                self.moves[b.end] = self.moves.get(b.end, 0) + 1

    def _times(self, s, h: int, cranes: int | None) -> tuple[int, int]:
        work = s.handling_for(cranes)
        dep = self.task.departure(s, h + work) if self.task.cranes else h + work
        return h + work, dep

    def fits(self, s, h: int, sec: int, cranes: int | None) -> bool:
        finish, dep = self._times(s, h, cranes)
        last = sec + s.sections - 1
        if any(h < r[1] and r[0] < dep and sec <= r[3] and r[2] <= last for r in self.rects):
            return False
        if self.task.cranes:
            if any(a <= h < b for a, b in self.task.no_move_windows(s)):
                return False
            if any(self.use.get(t, 0) + cranes > self.task.crane_pool_at(t) for t in range(h, finish)):
                return False
            cap = self.task.rules.get("max_moves_per_hour")
            if cap and (self.moves.get(h, 0) + 1 > cap or self.moves.get(dep, 0) + 1 > cap):
                return False
        return True

    def add(self, s, h: int, sec: int, cranes: int | None) -> None:
        finish, dep = self._times(s, h, cranes)
        self.rects.append((h, dep, sec, sec + s.sections - 1))
        if self.task.cranes:
            for t in range(h, finish):
                self.use[t] = self.use.get(t, 0) + cranes
            self.moves[h] = self.moves.get(h, 0) + 1
            self.moves[dep] = self.moves.get(dep, 0) + 1


def naive_replan(task: Task) -> Plan:
    """Keep each ship's published slot if it still works; otherwise give it the earliest hour, then the section
    closest to its planned one, that is free. Ships are taken in published order (unscheduled calls by arrival).
    On crane-rule tasks every ship keeps its standard crane count."""
    led = _Ledger(task)
    plan: Plan = {}
    order = sorted(task.ships, key=lambda s: (s.planned_hour if s.planned_hour is not None else s.arrival, s.id))
    for s in order:
        cranes = s.std_cranes if task.cranes else None
        anchor = s.planned_section if s.planned_section is not None else task.first_section
        h = max(s.arrival, s.planned_hour if s.planned_hour is not None else s.arrival)
        positions = sorted(range(task.first_section, task.last_section - s.sections + 2), key=lambda k: (abs(k - anchor), k))
        while True:
            sec = next((k for k in positions if led.fits(s, h, k, cranes)), None)
            if sec is not None:
                break
            h += 1
        led.add(s, h, sec, cranes)
        plan[s.id] = (h, sec) + ((cranes,) if task.cranes else ())
    return plan


def optimal_plan(task: Task, time_limit: float = 60.0, workers: int = 8, hint: Plan | None = None):
    """Minimum-cost plan by CP-SAT (time x quay rectangles with NoOverlap2D; on crane-rule tasks one optional
    interval per crane count, a cumulative crane pool and a cumulative movement limit). Returns
    (plan, cost, bound, status)."""
    from ortools.sat.python import cp_model

    horizon = max(s.arrival for s in task.ships) + sum(s.handling_for(s.min_cranes if task.cranes else None)
                                                         for s in task.ships) + max([b.end for b in task.blocks] + [0])
    md = cp_model.CpModel()
    xs, ys, terms, T, M, C = [], [], [], {}, {}, {}
    crane_iv, crane_dem, move_iv = [], [], []
    for s in task.ships:
        t = md.NewIntVar(s.arrival, horizon, f"t{s.id}")
        m = md.NewIntVar(task.first_section, task.last_section - s.sections + 1, f"m{s.id}")
        end = md.NewIntVar(s.arrival, horizon * 2, f"e{s.id}")
        pres = {}
        if not task.cranes:
            stay = s.handling
            xs.append(md.NewIntervalVar(t, stay, t + stay, f"x{s.id}"))
            ys.append(md.NewIntervalVar(m, s.sections, m + s.sections, f"y{s.id}"))
            md.Add(end == t + stay)
        else:
            finish = md.NewIntVar(s.arrival, horizon * 2, f"f{s.id}")
            for k in range(s.min_cranes, s.max_cranes + 1):
                p = md.NewBoolVar(f"p{s.id}_{k}")
                pres[k] = p
                work = s.handling_for(k)
                md.Add(finish == t + work).OnlyEnforceIf(p)
                crane_iv.append(md.NewOptionalIntervalVar(t, work, t + work, p, f"c{s.id}_{k}"))
                crane_dem.append(k)
            md.AddExactlyOne(pres.values())
            # departure: the finish hour, or the end of a no-movement window the finish falls in (wait alongside)
            inside = []
            for a, b in task.no_move_windows(s):
                lt, ge = md.NewBoolVar(""), md.NewBoolVar("")      # berth hour outside [a, b)
                md.Add(t <= a - 1).OnlyEnforceIf(lt)
                md.Add(t >= b).OnlyEnforceIf(ge)
                md.AddBoolOr([lt, ge])
                w = md.NewBoolVar(f"w{s.id}_{a}")
                md.Add(finish >= a).OnlyEnforceIf(w)
                md.Add(finish <= b - 1).OnlyEnforceIf(w)
                below, above = md.NewBoolVar(""), md.NewBoolVar("")
                md.Add(finish <= a - 1).OnlyEnforceIf(below)
                md.Add(finish >= b).OnlyEnforceIf(above)
                md.AddBoolOr([w, below, above])
                md.Add(end == b).OnlyEnforceIf(w)
                inside.append(w)
            if inside:
                none = md.NewBoolVar(f"n{s.id}")
                md.AddBoolAnd([x.Not() for x in inside]).OnlyEnforceIf(none)
                md.AddBoolOr(inside + [none])
                md.Add(end == finish).OnlyEnforceIf(none)
            else:
                md.Add(end == finish)
            size = md.NewIntVar(1, horizon * 2, f"sz{s.id}")
            md.Add(size == end - t)
            xs.append(md.NewIntervalVar(t, size, end, f"x{s.id}"))
            ys.append(md.NewIntervalVar(m, s.sections, m + s.sections, f"y{s.id}"))
            move_iv.append(md.NewFixedSizeIntervalVar(t, 1, f"mb{s.id}"))
            move_iv.append(md.NewFixedSizeIntervalVar(end, 1, f"md{s.id}"))
        late = md.NewIntVar(0, horizon * 2, f"late{s.id}")
        md.Add(late >= end - s.due)
        terms.append(s.sections * s.weight * late)
        if s.berth_deadline is not None and s.deadline_penalty:
            over = md.NewIntVar(0, horizon * 2, f"over{s.id}")
            md.Add(over >= t - s.berth_deadline)
            terms.append(s.deadline_penalty * over)
        if s.planned_section is not None:
            moved = md.NewBoolVar(f"moved{s.id}")
            md.Add(m == s.planned_section).OnlyEnforceIf(moved.Not())
            terms.append(MOVE_PENALTY * moved)
        T[s.id], M[s.id], C[s.id] = t, m, pres
    for k, b in enumerate(task.blocks):
        bx = md.NewFixedSizeIntervalVar(b.start, b.end - b.start, f"bx{k}")
        xs.append(bx)
        ys.append(md.NewFixedSizeIntervalVar(b.first, b.last - b.first + 1, f"by{k}"))
        if task.cranes and b.kind == "alongside":
            if b.cranes:
                crane_iv.append(bx)
                crane_dem.append(b.cranes)
            move_iv.append(md.NewFixedSizeIntervalVar(b.end, 1, f"bm{k}"))
    md.AddNoOverlap2D(xs, ys)
    if task.cranes:
        for k, o in enumerate(task.rules.get("crane_outages", [])):
            crane_iv.append(md.NewFixedSizeIntervalVar(o["start"], o["end"] - o["start"], f"out{k}"))
            crane_dem.append(int(o["cranes"]))
        md.AddCumulative(crane_iv, crane_dem, int(task.rules["crane_pool"]))
        cap = task.rules.get("max_moves_per_hour")
        if cap:
            md.AddCumulative(move_iv, [1] * len(move_iv), int(cap))
    md.Minimize(sum(terms))
    if hint:
        for sid, entry in hint.items():
            md.AddHint(T[sid], entry[0])
            md.AddHint(M[sid], entry[1])
            if task.cranes and len(entry) > 2:
                for k, p in C[sid].items():
                    md.AddHint(p, int(k == entry[2]))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = workers
    status = solver.Solve(md)
    name = solver.StatusName(status)
    if name not in ("OPTIMAL", "FEASIBLE"):
        return None, None, None, name
    plan: Plan = {}
    for s in task.ships:
        entry = (solver.Value(T[s.id]), solver.Value(M[s.id]))
        if task.cranes:
            entry += (next(k for k, p in C[s.id].items() if solver.Value(p)),)
        plan[s.id] = entry
    return plan, int(round(solver.ObjectiveValue())), int(round(solver.BestObjectiveBound())), name
