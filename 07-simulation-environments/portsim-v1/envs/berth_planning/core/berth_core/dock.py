"""The hard static pack ("dock-v1"): real one-to-three-week windows at a Barcelona container quay, every operating
rule we can ground, and stacked disruptions. All information is given up front; the difficulty is the size and the
coupling of the problem.

Rules on every task (see generate.py for where the numbers come from):
    quay cranes          the terminal's real 2024 fleet (BEST 13, quay 24B 9), 1-7 per ship by length, work in moves
                         at 28 moves per crane-hour
    movements            at most 3 (36A) / 2 (24B) ships berth or leave per hour, the 2024 record's maximum
Disruptions, drawn per tier and seed:
    closure      sections closed for works                 late       ships arrive hours after their slot
    bunching     several delayed ships arrive together     outage     cranes out of service
    gale         the port's wind rules stop manoeuvres     divert     the other quay's calls come here (36A)
    extra        unscheduled calls                         emergency  a reefer failure: dock by a deadline or pay
    priority     a transshipment connection: lateness costs triple

Eval and train never share a week: eval windows sit inside HELD_OUT, train windows avoid it.
"""

from __future__ import annotations

import random
from datetime import timedelta

from .check import evaluate
from .data import QUAYS, Call
from .generate import (BASE_MOVES_PER_HOUR, CLOSURE_REASONS, CRANE_POOL, CRANE_RATE, _published_peaks, _with_cranes,
                       crane_class, real_week, week_start, _hour)
from .model import Block, Ship, Task, plan_to_list
from .solve import naive_replan, optimal_plan

# Weeks held out for evaluation: three 3-week blocks and one single week (10 of the year's usable 50). The pack was
# built with 20 held-out weeks; the eval tasks from weeks 20, 25-27, 30, 40, 45-47 and 50 later moved to train.
HELD_OUT = {5, 6, 7, 10, 15, 16, 17, 35, 36, 37}
FIRST_WEEK, LAST_WEEK = 2, 51
GAP_K = 100          # reward: g = (cost - best) / (best - floor + GAP_K); floor = check.unavoidable_cost
EMERGENCY_RATE = 10  # cost per section per hour a genuine emergency docks after its deadline
MAX_GREEDY_REWARD = 0.85   # a task a simple greedy heuristic nearly solves is not kept
MAX_NAIVE_REWARD = 0.6

TIERS = {
    # weeks, closures, late, bunching, outages, gales, divert, extra, emergencies, priority
    "standard": dict(weeks=(1, 1), closures=(1, 1), late=(1, 2), bunch=(0, 0), outages=(0, 1), gales=(0, 0),
                     divert=(0, 0), extra=(0, 1), emergency=(0, 1), priority=(0, 1)),
    "busy":     dict(weeks=(1, 2), closures=(1, 2), late=(2, 3), bunch=(0, 1), outages=(1, 1), gales=(0, 1),
                     divert=(0, 0), extra=(0, 1), emergency=(0, 1), priority=(1, 2)),
    "storm":    dict(weeks=(2, 2), closures=(1, 2), late=(2, 4), bunch=(0, 1), outages=(1, 1), gales=(1, 1),
                     divert=(0, 1), extra=(0, 1), emergency=(1, 1), priority=(1, 2)),
    "extreme":  dict(weeks=(2, 3), closures=(2, 3), late=(3, 5), bunch=(1, 1), outages=(1, 2), gales=(1, 2),
                     divert=(1, 1), extra=(1, 2), emergency=(1, 2), priority=(2, 3)),
}


def windows(split: str, n_weeks: int) -> list[int]:
    """Start weeks of n_weeks-long windows for a split: eval windows lie inside HELD_OUT, train windows avoid it."""
    out = []
    for w in range(FIRST_WEEK, LAST_WEEK - n_weeks + 2):
        span = set(range(w, w + n_weeks))
        if split == "eval" and span <= HELD_OUT and all(x in HELD_OUT for x in span):
            out.append(w)
        if split == "train" and not (span & HELD_OUT):
            out.append(w)
    return out


def real_window(calls: list[Call], quay: str, week: int, n_weeks: int) -> tuple[list[Ship], list[Block], int]:
    ships, blocks, trims = real_week(calls, quay, week)
    seen = {(s.name, s.planned_hour) for s in ships}
    for k in range(1, n_weeks):
        more, _, t = real_week(calls, quay, week + k)
        trims += t
        for s in more:
            s2 = Ship(**{**s.__dict__})
            s2.arrival += 168 * k
            s2.planned_hour += 168 * k
            s2.due += 168 * k
            if (s2.name, s2.planned_hour) in seen:
                continue
            s2.id = len(ships)
            seen.add((s2.name, s2.planned_hour))
            ships.append(s2)
    return ships, blocks, trims


def _n(rng, lo_hi):
    lo, hi = lo_hi
    return rng.randint(lo, hi)


def build_dock_task(calls: list[Call], quay: str, week: int, n_weeks: int, tier: str, seed: int, split: str,
                    time_limit: float = 300.0, workers: int = 8, max_gap: float = 0.01):
    """One candidate. Returns (task or None, stats); None when the solver does not get within max_gap of its bound."""
    rng = random.Random(f"dock-{quay}-{week}-{n_weeks}-{tier}-{seed}")
    recipe = TIERS[tier]
    q = QUAYS[quay]
    H = 168 * n_weeks
    ships, blocks, trims = real_window(calls, quay, week, n_weeks)
    if len(ships) < 8:
        return None, {"reason": "fewer than 8 ships"}
    _with_cranes(ships, blocks)
    peak_cranes, peak_moves = _published_peaks(ships, blocks)
    notices: list[str] = []
    events: list[dict] = []
    rules = {"crane_pool": max(CRANE_POOL[quay], peak_cranes), "crane_rate": CRANE_RATE,
             "max_moves_per_hour": max(BASE_MOVES_PER_HOUR[quay], peak_moves), "crane_outages": [], "no_moves": [],
             "reward": 3, "gap_k": GAP_K}
    scheduled = lambda: [s for s in ships if s.planned_section is not None]

    # diverted traffic (36A only): the other quay closes for 2-3 days and its real calls come here
    if quay == "36A" and _n(rng, recipe["divert"]):
        other = "24B"
        days = rng.randint(2, 3)
        d0 = rng.randint(0, n_weeks * 7 - days)
        t0 = week_start(week)
        lo, hi = t0 + timedelta(days=d0), t0 + timedelta(days=d0 + days)
        moved = []
        for c in sorted((c for c in calls if c.quay == other and lo <= c.eta < hi), key=lambda c: (c.eta, c.call)):
            a = _hour(t0, c.eta)
            h = max(1, _hour(t0, c.etd) - a)
            lo_c, hi_c, std = crane_class(c.length_m)
            s = Ship(id=len(ships), name=c.ship, imo=c.imo, length_m=c.length_m,
                     sections=min(c.last - c.first + 1, q.last_section - q.first_section + 1), arrival=a, handling=h,
                     planned_hour=None, planned_section=None, due=a + h, from_port=c.from_port, to_port=c.to_port,
                     beam_m=c.beam_m, draught_m=c.draught_m, workload=h * std, min_cranes=lo_c, max_cranes=hi_c,
                     std_cranes=std)
            ships.append(s)
            moved.append(s.id)
        if moved:
            notices.append(f"Quay {other} ({QUAYS[other].terminal}) is closed from hour {24 * d0} to hour "
                           f"{24 * (d0 + days)}; its {len(moved)} calls in that window are diverted here (ships "
                           f"{', '.join(map(str, moved))}). They have no planned slot on this quay, so placing them costs "
                           f"no move penalty, and they keep their planned departure hours.")
            events.append({"type": "divert", "from_quay": other, "start": 24 * d0, "end": 24 * (d0 + days),
                           "ships": moved})

    # unscheduled calls: real ships from far-away weeks at this quay
    for _ in range(_n(rng, recipe["extra"])):
        others = [c for c in calls if c.quay == quay and abs(c.eta.isocalendar()[1] - week) > n_weeks + 2
                  and 4 <= (c.etd - c.eta).total_seconds() / 3600 <= 36]
        c = rng.choice(others)
        handling = max(4, int(round((c.etd - c.eta).total_seconds() / 3600)))
        arrival = rng.randint(12, H - 30)
        due = arrival + handling + rng.randint(4, 12)
        lo_c, hi_c, std = crane_class(c.length_m)
        s = Ship(id=len(ships), name=c.ship, imo=c.imo, length_m=c.length_m,
                 sections=min(c.last - c.first + 1, q.last_section - q.first_section + 1), arrival=arrival,
                 handling=handling, planned_hour=None, planned_section=None, due=due, from_port=c.from_port,
                 to_port=c.to_port, beam_m=c.beam_m, draught_m=c.draught_m, workload=handling * std,
                 min_cranes=lo_c, max_cranes=hi_c, std_cranes=std)
        ships.append(s)
        notices.append(f"Unscheduled call: ship {s.id} {s.name} ({s.length_m:.0f} m, {s.sections} sections) arrives at "
                       f"hour {arrival}, has {s.workload * CRANE_RATE} moves and asks to sail by hour {due}. It has no "
                       f"planned slot, so placing it anywhere costs no move penalty.")
        events.append({"type": "extra", "ship": s.id})

    # closures
    for _ in range(_n(rng, recipe["closures"])):
        target = rng.choice(sorted(scheduled(), key=lambda s: -s.sections * s.handling)[:8])
        span = rng.randint(5, 9)
        first = max(q.first_section, min(q.last_section - span + 1, target.planned_section + rng.randint(-2, 2)))
        last = first + span - 1
        start = max(0, target.planned_hour - rng.randint(0, 12))
        dur = rng.randint(18, 48)
        for b in blocks:
            if b.kind == "alongside" and first <= b.last and b.first <= last:
                start = max(start, b.end)
        reason = rng.choice(CLOSURE_REASONS)
        blocks.append(Block(first, last, start, start + dur, "closed", reason))
        notices.append(f"Sections {first}-{last} are closed from hour {start} to hour {start + dur} ({reason}).")
        events.append({"type": "closure", "first": first, "last": last, "start": start, "end": start + dur,
                       "reason": reason})

    # late ships
    for s in rng.sample(scheduled(), min(len(scheduled()), _n(rng, recipe["late"]))):
        d = rng.randint(4, 24)
        s.arrival += d
        where = f" leaving {s.from_port}" if s.from_port else ""
        notices.append(f"Ship {s.id} {s.name} was delayed{where}: it now arrives at hour {s.arrival} (planned slot hour "
                       f"{s.planned_hour}). Its planned departure stays hour {s.due}.")
        events.append({"type": "late", "ship": s.id, "hours": d})

    # bunching: delayed ships from the same few days all arrive in one 12-hour window
    if _n(rng, recipe["bunch"]):
        pool = [s for s in scheduled() if s.arrival == s.planned_hour]
        anchor = rng.choice(pool)
        group = [s for s in pool if 0 <= s.planned_hour - anchor.planned_hour <= 72][: rng.randint(3, 5)]
        if len(group) >= 3:
            t = max(s.planned_hour for s in group) + rng.randint(0, 12)
            ids = []
            for s in group:
                s.arrival = max(s.arrival, t + rng.randint(0, 11))
                ids.append(s.id)
            notices.append(f"Ships {', '.join(map(str, ids))} were held up by the Red Sea diversions and now arrive "
                           f"together between hours {t} and {t + 12} (each ship's arrival hour is in the table). Their "
                           f"planned departures stay as published.")
            events.append({"type": "bunching", "ships": ids, "start": t, "end": t + 12})

    # crane outages: never overlapping each other, and after the ships already alongside have left
    along_end = max([b.end for b in blocks if b.kind == "alongside"] + [0])
    for _ in range(_n(rng, recipe["outages"])):
        lo_o, hi_o = max(12, along_end), H - 40
        if lo_o >= hi_o:
            break
        for _try in range(20):
            o0, n = rng.randint(lo_o, hi_o), rng.randint(2, 3)
            o1 = o0 + rng.randint(18, 48)
            if all(o1 <= o["start"] or o["end"] <= o0 for o in rules["crane_outages"]):
                break
        else:
            continue
        rules["crane_outages"].append({"start": o0, "end": o1, "cranes": n})
        notices.append(f"{n} of the terminal's {rules['crane_pool']} quay cranes are out of service from hour {o0} to "
                       f"hour {o1} (gantry repair): {rules['crane_pool'] - n} cranes in that window.")
        events.append({"type": "crane_outage", "start": o0, "end": o1, "cranes": n})

    # gales under the port's wind rules
    for _ in range(_n(rng, recipe["gales"])):
        g0 = rng.randint(24, H - 48)
        g1 = g0 + rng.randint(16, 30)
        c0 = g0 + rng.randint(2, 6)
        c1 = min(g1 - 1, c0 + rng.randint(6, 14))
        rules["no_moves"] += [{"start": g0, "end": g1, "min_length": 300, "reason": "wind above 25 kn"},
                              {"start": c0, "end": c1, "min_length": 0, "reason": "wind above 30 kn"}]
        for b in blocks:
            if b.kind == "alongside" and c0 <= b.end < c1:
                b.end = c1
        notices.insert(0, f"Gale warning. From hour {g0} to hour {g1} the wind is above 25 kn: under the port's traffic "
                          f"rules ships of 300 m or more may not berth or leave. From hour {c0} to hour {c1} it is above "
                          f"30 kn: no ship may berth or leave. Cranes keep working; a ship that finishes inside a window "
                          f"waits alongside until the window ends.")
        events.insert(0, {"type": "gale", "start": g0, "end": g1, "core_start": c0, "core_end": c1})

    # genuine emergencies: a reefer power failure means the ship must dock by a deadline
    for s in rng.sample(ships, min(len(ships), _n(rng, recipe["emergency"]))):
        s.berth_deadline = s.arrival + rng.randint(4, 10)
        s.deadline_penalty = EMERGENCY_RATE * s.sections
        notices.append(f"Emergency: ship {s.id} {s.name} reports reefer power failures; its cargo is at risk unless it "
                       f"docks by hour {s.berth_deadline}. Each hour it docks later costs {s.deadline_penalty} on top of "
                       f"any delay.")
        events.append({"type": "emergency", "ship": s.id, "deadline": s.berth_deadline})

    # priority connections: lateness costs triple
    for s in rng.sample(ships, min(len(ships), _n(rng, recipe["priority"]))):
        if s.berth_deadline is not None:
            continue
        s.weight = 3
        notices.append(f"Priority: ship {s.id} {s.name} carries transshipment cargo for a connecting service; each hour "
                       f"it leaves after its planned departure (hour {s.due}) costs three times the usual.")
        events.append({"type": "priority", "ship": s.id})

    # fixed loads alone (ships already alongside, outages) must fit the pool and the movement limit, so the solver's
    # capacity constraints and the checker's per-ship view agree on every hour
    for t in range(0, H + 200):
        fixed = sum(b.cranes for b in blocks if b.kind == "alongside" and b.start <= t < b.end)
        out = sum(o["cranes"] for o in rules["crane_outages"] if o["start"] <= t < o["end"])
        if fixed > rules["crane_pool"] - out:
            return None, {"reason": f"fixed crane load {fixed} over the pool at hour {t}"}
    ends: dict[int, int] = {}
    for b in blocks:
        if b.kind == "alongside":
            ends[b.end] = ends.get(b.end, 0) + 1
    if any(v > rules["max_moves_per_hour"] for v in ends.values()):
        return None, {"reason": "fixed departures over the movement limit"}

    tid = f"dock-{quay}-w{week:02d}x{n_weeks}-{tier}-{seed}"
    task = Task(task_id=tid, split=split, quay=quay, terminal=q.terminal, first_section=q.first_section,
                last_section=q.last_section, section_m=q.section_m, week=week,
                week_start_utc=week_start(week).strftime("%Y-%m-%dT%H:%M:%SZ"), ships=ships, blocks=blocks,
                notices=notices, disruptions=events, difficulty=tier, rules=rules)
    naive = naive_replan(task)
    naive_cost = evaluate(task, naive).cost
    plan, cost, bound, status = optimal_plan(task, time_limit=time_limit, workers=workers, hint=naive)
    published = evaluate(task, task.planned())
    broken = sum(1 for r in published.ships if r.problems and r.problems != ["missing from the plan"])
    stats = {"task_id": tid, "ships": len(ships), "weeks": n_weeks, "naive": naive_cost, "optimal": cost,
             "bound": bound, "status": status, "broken": broken}
    if plan is None or (cost - bound) > max_gap * max(cost, 1):
        return None, {**stats, "reason": f"solver {status}, gap {cost}-{bound}" if plan else f"solver {status}"}
    assert evaluate(task, plan).cost == cost
    from .baselines import greedy
    from .reward import score_v3
    g_res = evaluate(task, greedy(task))
    from .check import unavoidable_cost
    floor = unavoidable_cost(task)
    greedy_reward = score_v3(g_res.cost, g_res.feasible, 1.0, cost, GAP_K, floor)[0]
    naive_reward = score_v3(naive_cost, True, 1.0, cost, GAP_K, floor)[0]
    stats.update(greedy_reward=round(greedy_reward, 3), naive_reward=round(naive_reward, 3))
    if greedy_reward > MAX_GREEDY_REWARD or naive_reward > MAX_NAIVE_REWARD or len(ships) < 12:
        return None, {**stats, "reason": f"too easy (greedy {greedy_reward:.2f}, naive {naive_reward:.2f}, {len(ships)} ships)"}
    task.reference = {"naive_cost": naive_cost, "greedy_cost": g_res.cost, "naive_plan": plan_to_list(naive), "optimal_cost": cost,
                      "optimal_plan": plan_to_list(plan), "proven_optimal": status == "OPTIMAL", "lower_bound": bound,
                      "published_conflicts": broken, "solver": "ortools-cpsat", "record_trims": trims}
    return task, stats
