"""Compute tasks/live/references.jsonl: each one-week dock-v1-eval week played live by a rolling CP-SAT re-planner and
by the naive online policy, both through port_week_core.world and graded as the live env grades.

At every watch the re-planner solves the week as known, with the frozen windows pinned and every other ship berthing at
or after the freeze line. It runs under four solver configurations (1 or 8 workers, with or without an earliest-finish
tie-break) on a deterministic time limit, the 8 workers interleaved because CP-SAT's parallel portfolio doesn't
reproduce otherwise; the first one's plans are stored. A week qualifies for the live tasks when at least 3 of the 4
reach the hindsight optimum. The tests replay the stored plans without ortools.

The frozen windows keep every rule, so if news leaves them breaking one by themselves (a state the live grade excuses)
the re-solve is infeasible and the script stops; no one-week dock-v1-eval week gets there.

Run from envs/port_week; --out defaults to live/references.jsonl under packs.tasks_root().

    uv run -p 3.12 --with ortools==9.15.6755 --with-editable core python tools/live_references.py [--tasks id,...]
        [--out PATH]
"""

import argparse
import json
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import ortools
from berth_core import Plan, Task, plan_to_list
from berth_core.model import MOVE_PENALTY
from berth_core.solve import naive_replan
from ortools.sat.python import cp_model
from port_week_core import packs, schedule, world

CONFIGS = [(1, True), (1, False), (8, True), (8, False)]
DETERMINISTIC_SECONDS = 60.0
TIEBREAK_SCALE = 10000


def solve(view: Task, fixed: Plan, before: int, hint: Plan, workers: int, tiebreak: bool,
          extra: Callable[[cp_model.CpModel, dict], None] | None = None, optimal: bool = False,
          forecast_relief: bool = False, solved: Callable[[cp_model.CpSolver], None] | None = None) -> Plan:
    """berth_core.solve.optimal_plan's crane-rule model, with ``fixed`` pinned and the others at or after ``before``;
    ``extra`` adds constraints over each ship's (berthing, leaving) hour variables, and any cost terms it returns to
    the objective; ``optimal`` accepts only a proven optimum. ``forecast_relief`` takes what no one can move as
    given, since a new forecast may put a pinned window inside its no-movement windows: pinned ships may berth
    inside them, the movements of pinned ships (at their stays as held in the view) and of the ships alongside count
    as fixed demand up to the hourly limit, and those stays and the blocks may overlap one another but no other
    ship. ``solved`` reads the solver's other values."""
    horizon = (max([s.arrival for s in view.ships] + [before]) + sum(s.handling_for(s.min_cranes) for s in view.ships)
               + max([b.end for b in view.blocks] + [0]))
    md = cp_model.CpModel()
    xs, ys, terms, ends, T, M, C = [], [], [], [], {}, {}, {}
    crane_iv, crane_dem, move_iv = [], [], []
    rects, fixed_moves = [], Counter()
    for s in view.ships:
        pinned = forecast_relief and s.id in fixed
        t = md.NewIntVar(s.arrival if s.id in fixed else max(s.arrival, before), horizon, f"t{s.id}")
        m = md.NewIntVar(view.first_section, view.last_section - s.sections + 1, f"m{s.id}")
        end = md.NewIntVar(s.arrival, horizon * 2, f"e{s.id}")
        finish = md.NewIntVar(s.arrival, horizon * 2, f"f{s.id}")
        pres = {}
        for k in range(s.min_cranes, s.max_cranes + 1):
            pres[k] = md.NewBoolVar(f"p{s.id}_{k}")
            work = s.handling_for(k)
            md.Add(finish == t + work).OnlyEnforceIf(pres[k])
            crane_iv.append(md.NewOptionalIntervalVar(t, work, t + work, pres[k], f"c{s.id}_{k}"))
            crane_dem.append(k)
        md.AddExactlyOne(pres.values())
        inside = []
        for a, b in view.no_move_windows(s):
            if not pinned:
                lt, ge = md.NewBoolVar(""), md.NewBoolVar("")
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
            md.AddBoolOr([*inside, none])
            md.Add(end == finish).OnlyEnforceIf(none)
        else:
            md.Add(end == finish)
        size = md.NewIntVar(1, horizon * 2, f"sz{s.id}")
        md.Add(size == end - t)
        if pinned:
            h, sec, cranes = fixed[s.id]
            dep = view.departure(s, h + s.handling_for(cranes))
            rects.append((h, dep, sec, sec + s.sections, md.NewFixedSizeIntervalVar(h, dep - h, f"x{s.id}"),
                          md.NewFixedSizeIntervalVar(sec, s.sections, f"y{s.id}")))
            fixed_moves.update([h, dep])
        else:
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
        if s.id in fixed:
            h, sec, cranes = fixed[s.id]
            md.Add(t == h)
            md.Add(m == sec)
            md.Add(pres[cranes] == 1)
        ends.append(end)
        T[s.id], M[s.id], C[s.id] = t, m, pres
    for k, b in enumerate(view.blocks):
        bx = md.NewFixedSizeIntervalVar(b.start, b.end - b.start, f"bx{k}")
        by = md.NewFixedSizeIntervalVar(b.first, b.last - b.first + 1, f"by{k}")
        if forecast_relief:
            rects.append((b.start, b.end, b.first, b.last + 1, bx, by))
        else:
            xs.append(bx)
            ys.append(by)
        if b.kind == "alongside":
            if b.cranes:
                crane_iv.append(bx)
                crane_dem.append(b.cranes)
            if forecast_relief:
                fixed_moves[b.end] += 1
            else:
                move_iv.append(md.NewFixedSizeIntervalVar(b.end, 1, f"bm{k}"))
    if forecast_relief:
        for group in _apart(rects) or [[]]:
            md.AddNoOverlap2D(xs + [r[4] for r in group], ys + [r[5] for r in group])
    else:
        md.AddNoOverlap2D(xs, ys)
    for k, o in enumerate(view.rules.get("crane_outages", [])):
        crane_iv.append(md.NewFixedSizeIntervalVar(o["start"], o["end"] - o["start"], f"out{k}"))
        crane_dem.append(int(o["cranes"]))
    md.AddCumulative(crane_iv, crane_dem, int(view.rules["crane_pool"]))
    cap = int(view.rules["max_moves_per_hour"])
    move_dem = [1] * len(move_iv)
    for h, n in sorted(fixed_moves.items()):
        move_iv.append(md.NewFixedSizeIntervalVar(h, 1, f"fm{h}"))
        move_dem.append(min(n, cap))
    md.AddCumulative(move_iv, move_dem, cap)
    if extra and (more := extra(md, {s.id: (T[s.id], end) for s, end in zip(view.ships, ends, strict=True)})):
        terms += more
    md.Minimize(sum(terms) * TIEBREAK_SCALE + sum(ends) if tiebreak else sum(terms))
    for sid, (h, sec, cranes) in hint.items():
        md.AddHint(T[sid], h)
        md.AddHint(M[sid], sec)
        for k, p in C[sid].items():
            md.AddHint(p, int(k == cranes))
    solver = cp_model.CpSolver()
    solver.parameters.num_workers = workers
    solver.parameters.max_deterministic_time = DETERMINISTIC_SECONDS
    solver.parameters.interleave_search = workers > 1
    status = solver.StatusName(solver.Solve(md))
    if status not in (("OPTIMAL",) if optimal else ("OPTIMAL", "FEASIBLE")):
        raise RuntimeError(f"{view.task_id}: the re-solve at the freeze line {before} is {status}")
    if solved:
        solved(solver)
    return {s.id: (solver.Value(T[s.id]), solver.Value(M[s.id]), next(k for k, p in C[s.id].items() if solver.Value(p)))
            for s in view.ships}


def _apart(rects: list[tuple]) -> list[list[tuple]]:
    """The rectangles in groups that overlap nowhere within a group, each new one in the first group it fits."""
    groups: list[list[tuple]] = []
    for r in rects:
        fits = [g for g in groups if not any(r[0] < q[1] and q[0] < r[1] and r[2] < q[3] and q[2] < r[3] for q in g)]
        if fits:
            fits[0].append(r)
        else:
            groups.append([r])
    return groups


def rolling(workers: int, tiebreak: bool, plans: list[list[dict]]) -> world.Policy:
    def policy(week: world.Week) -> Plan:
        view = week.view
        plan = solve(view, week.fixed, week.before, week.plan if week.watch else naive_replan(view), workers, tiebreak)
        plans.append(plan_to_list(plan))
        return plan
    return policy


def played(task: Task, policy: world.Policy) -> dict:
    week = world.play(task, policy)
    assert week.refusals == []
    return week.grade


def outcome(grade: dict) -> dict:
    return {key: grade[key] for key in ("cost", "reward", "feasible", "excused_cost")}


def reference(task: Task) -> dict:
    runs = []
    for workers, tiebreak in CONFIGS:
        plans = []
        runs.append((played(task, rolling(workers, tiebreak, plans)), plans))
    (grade, plans), costs = runs[0], [g["cost"] for g, _ in runs]
    return {"task_id": task.task_id, "watch_hours": [w.hour for w in schedule.schedule(task)],
            "optimal_cost": grade["optimal_cost"], "unavoidable_cost": grade["unavoidable_cost"],
            "rolling": {**outcome(grade), "plans": plans},
            "configs": [{"workers": w, "tiebreak": tb, "cost": c} for (w, tb), c in zip(CONFIGS, costs, strict=True)],
            "qualifies": sum(c is not None and c <= grade["optimal_cost"] for c in costs) >= 3,
            "naive": outcome(played(task, world.naive)),
            "solver": {"ortools": ortools.__version__, "max_deterministic_time": DETERMINISTIC_SECONDS}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tasks", help="comma-separated task ids; default: every one-week dock-v1-eval task")
    parser.add_argument("--out", type=Path, default=packs.references("live"))
    args = parser.parse_args()
    chosen = [t for t in packs.live_pack().tasks if t.split == "eval" and schedule.ONE_WEEK.match(t.task_id)
              and (args.tasks is None or t.task_id in args.tasks.split(","))]
    lines = []
    for task in chosen:
        started = time.monotonic()
        line = reference(task)
        lines.append(json.dumps(line) + "\n")
        print(f"{task.task_id}: {len(line['watch_hours'])} watches, optimum {line['optimal_cost']}, rolling "
              f"{[c['cost'] for c in line['configs']]}, naive {line['naive']['cost']}, qualifies {line['qualifies']}, "
              f"{time.monotonic() - started:.1f}s", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(lines))
    print(f"Wrote {len(lines)} weeks to {args.out}")


if __name__ == "__main__":
    main()
