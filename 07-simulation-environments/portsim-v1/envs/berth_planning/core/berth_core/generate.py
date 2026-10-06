"""Build tasks: a real week at a Barcelona quay, then a seeded disruption the published plan cannot absorb.

The 2024 record is almost conflict-free (ETA is effectively berthing time), so an undisturbed week has nothing to
optimise. Each task applies one to four disruptions a berth planner meets in practice:

    closure    a run of 5-9 sections closed for 18-48 h (crane maintenance, fender repair, dredging)
    late       1-3 ships arrive 4-20 h after their planned berthing hour
    overrun    one ship needs 30-60 % longer alongside (a quay crane breaks down)
    extra      an unscheduled call (a real ship from another 2024 week at this quay) asks for a berth
    divert     the other container quay closes for 2-3 days and its real calls in that window come here (expert)

and stores the naive re-plan and the CP-SAT optimum as the reward anchors.
"""

from __future__ import annotations

import random
from datetime import date, datetime, timedelta, timezone

from .check import evaluate
from .data import QUAYS, Call, load_calls
from .model import Block, Ship, Task, plan_to_list
from .solve import naive_replan, optimal_plan

CLOSURE_REASONS = ("STS crane maintenance", "fender repair", "dredging along the berth", "quay crane rail repair")
DIFFICULTIES = {
    "easy": [("closure",), ("late",)],
    "medium": [("closure", "late"), ("closure", "overrun"), ("late", "extra")],
    "hard": [("closure", "late", "overrun"), ("closure", "late", "extra"), ("closure", "late", "overrun", "extra")],
    "expert": [("divert", "closure", "late"), ("divert", "late", "overrun"), ("divert", "closure", "overrun")],
}


def week_start(week: int, year: int = 2024) -> datetime:
    return datetime.combine(date.fromisocalendar(year, week, 1), datetime.min.time(), tzinfo=timezone.utc)


def _hour(t0: datetime, ts: datetime) -> int:
    return int(round((ts - t0).total_seconds() / 3600))


def real_week(calls: list[Call], quay: str, week: int) -> tuple[list[Ship], list[Block], int]:
    """Ships berthing in the week, the ships already alongside at hour 0, and how many 1-hour overlaps in the
    published record were trimmed (data noise: the earlier ship's stay is cut to end when the next one berths)."""
    q = QUAYS[quay]
    t0 = week_start(week)
    t1 = t0 + timedelta(days=7)
    rows = sorted((c for c in calls if c.quay == quay and c.eta < t1 and c.etd > t0), key=lambda c: (c.eta, c.call))
    ships: list[Ship] = []
    blocks: list[Block] = []
    for c in rows:
        first, last = max(q.first_section, c.first), min(q.last_section, c.last)
        a, e = _hour(t0, c.eta), _hour(t0, c.etd)
        if e <= 0:
            continue
        if a < 0:
            blocks.append(Block(first, last, 0, e, "alongside", c.ship))
            continue
        ships.append(Ship(id=len(ships), name=c.ship, imo=c.imo, length_m=c.length_m, sections=last - first + 1,
                          arrival=a, handling=max(1, e - a), planned_hour=a, planned_section=first, due=max(a + 1, e),
                          from_port=c.from_port, to_port=c.to_port, beam_m=c.beam_m, draught_m=c.draught_m))
    # Make the published plan consistent: trim stays that run into a later berthing on shared sections.
    trims = 0
    rects = [("b", i, b.start, b.end, b.first, b.last) for i, b in enumerate(blocks)] + \
            [("s", s.id, s.planned_hour, s.planned_hour + s.handling, s.planned_section,
              s.planned_section + s.sections - 1) for s in ships]
    for _ in range(3):
        changed = False
        for i in range(len(rects)):
            for j in range(len(rects)):
                ki, ii, a0, a1, m0, m1 = rects[i]
                kj, jj, b0, b1, n0, n1 = rects[j]
                if i == j or not (a0 <= b0 < a1 and m0 <= n1 and n0 <= m1):
                    continue
                if (a0, i) >= (b0, j):
                    continue
                if b0 > a0:  # cut the earlier stay
                    if ki == "s":
                        ships[ii].handling = b0 - a0
                        ships[ii].due = min(ships[ii].due, b0)
                    else:
                        blocks[ii].end = b0
                    rects[i] = (ki, ii, a0, b0, m0, m1)
                elif kj == "s":  # same hour: the later one waits
                    s = ships[jj]
                    s.planned_hour = s.arrival = a1
                    s.due = max(s.due, a1 + s.handling)
                    rects[j] = (kj, jj, a1, a1 + s.handling, n0, n1)
                trims += 1
                changed = True
        if not changed:
            break
    return ships, blocks, trims


def _inside(blocks, first, last, start, end) -> bool:
    return any(first <= b.last and b.first <= last and start < b.end and b.start < end for b in blocks)


def disrupt(ships: list[Ship], blocks: list[Block], calls: list[Call], quay: str, week: int, kinds, rng: random.Random):
    q = QUAYS[quay]
    notices: list[str] = []
    events: list[dict] = []
    if "divert" in kinds:
        other = "24B" if quay == "36A" else "36A"
        days = rng.randint(2, 3)
        d0 = rng.randint(0, 7 - days)
        t0 = week_start(week)
        lo, hi = t0 + timedelta(days=d0), t0 + timedelta(days=d0 + days)
        moved = []
        for c in sorted((c for c in calls if c.quay == other and lo <= c.eta < hi), key=lambda c: (c.eta, c.call)):
            n = c.last - c.first + 1
            a = _hour(t0, c.eta)
            h = max(1, _hour(t0, c.etd) - a)
            s = Ship(id=len(ships), name=c.ship, imo=c.imo, length_m=c.length_m, sections=min(n, q.last_section - q.first_section + 1),
                     arrival=a, handling=h, planned_hour=None, planned_section=None, due=a + h, from_port=c.from_port,
                     to_port=c.to_port, beam_m=c.beam_m, draught_m=c.draught_m)
            ships.append(s)
            moved.append(s.id)
        name = QUAYS[other].terminal
        notices.append(f"Quay {other} ({name}) is closed from hour {24 * d0} to hour {24 * (d0 + days)}; its {len(moved)} calls "
                       f"in that window are diverted here (ships {', '.join(map(str, moved))}). They have no planned berth on "
                       f"this quay, so placing them costs no move penalty, and they keep their planned departure hours.")
        events.append({"type": "divert", "from_quay": other, "start": 24 * d0, "end": 24 * (d0 + days), "ships": moved})
    if "closure" in kinds:
        scheduled = [s for s in ships if s.planned_section is not None]
        target = rng.choice(sorted(scheduled, key=lambda s: -s.sections * s.handling)[:5])
        span = rng.randint(5, 9)
        first = max(q.first_section, min(q.last_section - span + 1, target.planned_section + rng.randint(-2, 2)))
        last = first + span - 1
        start = max(0, target.planned_hour - rng.randint(0, 12))
        dur = rng.randint(18, 48)
        for b in blocks:  # never close sections under a ship that is already alongside
            if b.kind == "alongside" and first <= b.last and b.first <= last:
                start = max(start, b.end)
        reason = rng.choice(CLOSURE_REASONS)
        blocks.append(Block(first, last, start, start + dur, "closed", reason))
        notices.append(f"Sections {first}-{last} are closed from hour {start} to hour {start + dur} ({reason}).")
        events.append({"type": "closure", "first": first, "last": last, "start": start, "end": start + dur, "reason": reason})
    if "late" in kinds:
        scheduled = [s for s in ships if s.planned_section is not None]
        for s in rng.sample(scheduled, min(len(scheduled), rng.randint(1, 3))):
            d = rng.randint(4, 20)
            s.arrival += d
            where = f" leaving {s.from_port}" if s.from_port else ""
            notices.append(f"Ship {s.id} {s.name} was delayed{where}: it now arrives at hour {s.arrival} "
                           f"(planned berthing hour {s.planned_hour}). Its planned departure stays hour {s.due}.")
            events.append({"type": "late", "ship": s.id, "hours": d})
    if "overrun" in kinds:
        scheduled = [s for s in ships if s.planned_section is not None]
        s = rng.choice([s for s in scheduled if s.handling >= 8] or scheduled)
        extra = max(4, int(round(s.handling * rng.uniform(0.3, 0.6))))
        notices.append(f"A quay crane assigned to ship {s.id} {s.name} broke down: it now needs {s.handling + extra} "
                       f"hours alongside instead of {s.handling}. Its planned departure stays hour {s.due}.")
        events.append({"type": "overrun", "ship": s.id, "hours": extra})
        s.handling += extra
    if "extra" in kinds:
        others = [c for c in calls if c.quay == quay and abs(c.eta.isocalendar()[1] - week) > 2
                  and 4 <= (c.etd - c.eta).total_seconds() / 3600 <= 36]
        c = rng.choice(others)
        first, last = max(q.first_section, c.first), min(q.last_section, c.last)
        handling = max(4, int(round((c.etd - c.eta).total_seconds() / 3600)))
        arrival = rng.randint(12, 140)
        due = arrival + handling + rng.randint(4, 12)
        s = Ship(id=len(ships), name=c.ship, imo=c.imo, length_m=c.length_m, sections=last - first + 1, arrival=arrival,
                 handling=handling, planned_hour=None, planned_section=None, due=due, from_port=c.from_port,
                 to_port=c.to_port, beam_m=c.beam_m, draught_m=c.draught_m)
        ships.append(s)
        notices.append(f"Unscheduled call: ship {s.id} {s.name} ({s.length_m:.0f} m, {s.sections} sections) arrives at "
                       f"hour {arrival}, needs {handling} hours alongside and asks to sail by hour {due}. "
                       f"It has no planned berth, so placing it anywhere costs no move penalty.")
        events.append({"type": "extra", "ship": s.id})
    return notices, events


def build_task(calls: list[Call], quay: str, week: int, seed: int, difficulty: str, split: str,
               time_limit: float = 60.0, workers: int = 8) -> tuple[Task | None, dict]:
    """One candidate task. Returns (task or None, stats) - None when the instance is too small or the solver did not
    prove the optimum."""
    rng = random.Random(f"{quay}-{week}-{seed}-{difficulty}")
    ships, blocks, trims = real_week(calls, quay, week)
    if len(ships) < 6:
        return None, {"reason": "fewer than 6 ships"}
    kinds = rng.choice(DIFFICULTIES[difficulty])
    notices, events = disrupt(ships, blocks, calls, quay, week, kinds, rng)
    q = QUAYS[quay]
    task = Task(task_id=f"bcn-{quay}-w{week:02d}-{difficulty}-{seed}", split=split, quay=quay, terminal=q.terminal,
                first_section=q.first_section, last_section=q.last_section, section_m=q.section_m, week=week,
                week_start_utc=week_start(week).strftime("%Y-%m-%dT%H:%M:%SZ"), ships=ships, blocks=blocks,
                notices=notices, disruptions=events, difficulty=difficulty)
    published = evaluate(task, task.planned())
    broken = sum(1 for r in published.ships if r.problems and r.problems != ["missing from the plan"])
    naive = naive_replan(task)
    naive_cost = evaluate(task, naive).cost
    plan, cost, bound, status = optimal_plan(task, time_limit=time_limit, workers=workers, hint=naive)
    stats = {"ships": len(ships), "kinds": list(kinds), "naive": naive_cost, "optimal": cost, "bound": bound,
             "status": status, "trims": trims, "broken": broken}
    if status != "OPTIMAL":
        return None, {**stats, "reason": f"solver {status}"}
    assert evaluate(task, plan).cost == cost
    task.reference = {"naive_cost": naive_cost, "naive_plan": plan_to_list(naive), "optimal_cost": cost,
                      "optimal_plan": plan_to_list(plan), "proven_optimal": True, "published_conflicts": broken,
                      "record_trims": trims, "solver": "ortools-cpsat"}
    return task, stats


__all__ = ["build_task", "real_week", "disrupt", "load_calls", "week_start", "DIFFICULTIES"]


# ---------------------------------------------------------------- frontier tier: cranes, movements, two weeks

# Quay cranes. Pools are the terminals' 2024 ship-to-shore cranes: BEST (quay 36A) 13; APM Terminals had 14 on a
# 1,515 m quay of which quay 24B is about 880 m, so 24B gets 9. The standard allocation by length (1/2/3/4 cranes
# below 180/260/330 m and above) is calibrated so the 2024 plans fit those pools: peaks of 12-13 cranes at 36A and 8-9
# at 24B across every fortnight. The most cranes a ship can take is one per 50 m of length, at most 7.
CRANE_POOL = {"36A": 13, "24B": 9}
CRANE_RATE = 28                             # gross moves per crane-hour (European benchmarks 25-31)
BASE_MOVES_PER_HOUR = {"36A": 3, "24B": 2}  # the 2024 record never exceeds these at either quay


def crane_class(length_m: float) -> tuple[int, int, int]:
    std = 1 if length_m < 180 else 2 if length_m < 260 else 3 if length_m < 330 else 4
    hi = max(std, min(7, int(length_m // 50)))
    lo = 1 if length_m < 330 else 2
    return lo, hi, std


def _with_cranes(ships: list[Ship], blocks: list[Block]) -> None:
    for s in ships:
        lo, hi, std = crane_class(s.length_m)
        s.min_cranes, s.max_cranes, s.std_cranes = lo, hi, std
        s.workload = s.handling * std
    for b in blocks:
        if b.kind == "alongside":
            b.cranes = 2  # a ship part-way through its call: assume two cranes until it leaves


def _published_peaks(ships: list[Ship], blocks: list[Block]) -> tuple[int, int]:
    use: dict[int, int] = {}
    moves: dict[int, int] = {}
    for b in blocks:
        if b.kind == "alongside":
            for t in range(b.start, b.end):
                use[t] = use.get(t, 0) + b.cranes
            moves[b.end] = moves.get(b.end, 0) + 1
    for s in ships:
        if s.planned_hour is None:
            continue
        for t in range(s.planned_hour, s.planned_hour + s.handling):
            use[t] = use.get(t, 0) + s.std_cranes
        moves[s.planned_hour] = moves.get(s.planned_hour, 0) + 1
        moves[s.planned_hour + s.handling] = moves.get(s.planned_hour + s.handling, 0) + 1
    return max(use.values(), default=0), max(moves.values(), default=0)


def real_fortnight(calls: list[Call], quay: str, week: int) -> tuple[list[Ship], list[Block], int]:
    """Two consecutive ISO weeks as one horizon (hour 0 = Monday of `week`)."""
    a, blocks_a, trims_a = real_week(calls, quay, week)
    b, blocks_b, trims_b = real_week(calls, quay, week + 1)
    seen = {(s.name, s.planned_hour) for s in a}
    for s in b:  # week 2 in week-1 hours; ships that straddle the boundary appear in both and are kept once
        s2 = Ship(**{**s.__dict__})
        s2.arrival += 168
        s2.planned_hour += 168
        s2.due += 168
        if (s2.name, s2.planned_hour) in seen:
            continue
        s2.id = len(a)
        a.append(s2)
    return a, blocks_a, trims_a + trims_b


def build_frontier_task(calls: list[Call], quay: str, week: int, seed: int, split: str, time_limit: float = 180.0,
                        workers: int = 8, max_gap: float = 0.05) -> tuple[Task | None, dict]:
    """A fortnight at the quay's real crane count with a movement limit, hit by a gale (the port's wind rules stop
    ships of 300 m and more above 25 kn and every ship above 30 kn), a closure, late ships, a crane outage and extra
    traffic (diverted from 24B at 36A, an unscheduled call at 24B). Kept when CP-SAT's best plan is within `max_gap`
    of its bound."""
    rng = random.Random(f"frontier-{quay}-{week}-{seed}")
    ships, blocks, trims = real_fortnight(calls, quay, week)
    _with_cranes(ships, blocks)
    peak_cranes, peak_moves = _published_peaks(ships, blocks)
    q = QUAYS[quay]
    kinds = ["closure", "late"] + (["divert"] if quay == "36A" else ["extra"])
    notices, events = disrupt(ships, blocks, calls, quay, week, kinds, rng)
    for s in ships:  # diverted / extra calls get crane data too
        if s.workload is None:
            lo, hi, std = crane_class(s.length_m)
            s.min_cranes, s.max_cranes, s.std_cranes, s.workload = lo, hi, std, s.handling * std
    pool = max(CRANE_POOL[quay], peak_cranes)
    # gale: a 25 kn window for ships of 300 m and more, with a 30 kn core for everyone
    g0 = rng.randint(30, 150)
    g1 = g0 + rng.randint(16, 30)
    c0 = g0 + rng.randint(2, 6)
    c1 = min(g1 - 1, c0 + rng.randint(6, 12))
    no_moves = [{"start": g0, "end": g1, "min_length": 300, "reason": "wind above 25 kn"},
                {"start": c0, "end": c1, "min_length": 0, "reason": "wind above 30 kn"}]
    for b in blocks:  # a ship already alongside that is due out in the gale waits too
        if b.kind == "alongside" and c0 <= b.end < c1:
            b.end = c1
    out_start, out_len, out_n = rng.randint(24, 260), rng.randint(24, 48), rng.randint(2, 3)
    rules = {"crane_pool": pool, "crane_rate": CRANE_RATE,
             "max_moves_per_hour": max(BASE_MOVES_PER_HOUR[quay], peak_moves),
             "crane_outages": [{"start": out_start, "end": out_start + out_len, "cranes": out_n}],
             "no_moves": no_moves, "reward": 2}
    notices.insert(0, f"Gale warning. From hour {g0} to hour {g1} the wind is above 25 kn: under the port's traffic rules "
                      f"ships of 300 m or more may not berth or leave. From hour {c0} to hour {c1} it is above 30 kn: no "
                      f"ship may berth or leave. Cranes keep working; a ship that finishes inside a window waits "
                      f"alongside until the window ends.")
    events.insert(0, {"type": "gale", "start": g0, "end": g1, "core_start": c0, "core_end": c1})
    notices.append(f"{out_n} of the terminal's {pool} quay cranes are out of service from hour {out_start} to hour "
                   f"{out_start + out_len} (gantry repair): {pool - out_n} cranes in that window.")
    events.append({"type": "crane_outage", "start": out_start, "end": out_start + out_len, "cranes": out_n})
    task = Task(task_id=f"bcn-{quay}-w{week:02d}-frontier-{seed}", split=split, quay=quay, terminal=q.terminal,
                first_section=q.first_section, last_section=q.last_section, section_m=q.section_m, week=week,
                week_start_utc=week_start(week).strftime("%Y-%m-%dT%H:%M:%SZ"), ships=ships, blocks=blocks,
                notices=notices, disruptions=events, difficulty="frontier", rules=rules)
    naive = naive_replan(task)
    naive_cost = evaluate(task, naive).cost
    plan, cost, bound, status = optimal_plan(task, time_limit=time_limit, workers=workers, hint=naive)
    published = evaluate(task, task.planned())
    broken = sum(1 for r in published.ships if r.problems and r.problems != ["missing from the plan"])
    stats = {"ships": len(ships), "pool": pool, "moves_cap": rules["max_moves_per_hour"], "naive": naive_cost,
             "optimal": cost, "bound": bound, "status": status, "trims": trims, "broken": broken}
    if plan is None or (cost - bound) > max_gap * max(cost, 1):
        return None, {**stats, "reason": f"solver {status}, gap {cost}-{bound}" if plan else f"solver {status}"}
    assert evaluate(task, plan).cost == cost
    task.reference = {"naive_cost": naive_cost, "naive_plan": plan_to_list(naive), "optimal_cost": cost,
                      "optimal_plan": plan_to_list(plan), "proven_optimal": status == "OPTIMAL", "lower_bound": bound,
                      "published_conflicts": broken, "solver": "ortools-cpsat", "record_trims": trims}
    return task, stats
