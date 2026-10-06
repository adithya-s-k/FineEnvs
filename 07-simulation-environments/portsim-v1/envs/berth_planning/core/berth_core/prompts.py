"""What the agent reads: the situation (quay, notices, ships) and the rules, cost and plan format."""

from __future__ import annotations

from datetime import datetime

from .model import MOVE_PENALTY, Task

PROMPT_VERSION = 1
DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def _when(h: int) -> str:
    return f"{DAYS[(h // 24) % 7]} {h % 24:02d}:00" if 0 <= h < 24 * 7 else f"+{h // 24}d {h % 24:02d}:00"


def situation(task: Task) -> str:
    start = datetime.strptime(task.week_start_utc, "%Y-%m-%dT%H:%M:%SZ")
    lines = [
        f"# {task.terminal}, Port of Barcelona - quay {task.quay}, week {task.week} of 2024",
        f"Hour 0 is Monday {start:%d %B %Y} 00:00 UTC. Times are whole hours from then.",
        f"The quay is sections {task.first_section}-{task.last_section} (about {task.section_m:.0f} m each). "
        f"A ship needs its number of consecutive sections for its hours alongside.",
        "",
        "## Notices",
        *[f"- {n}" for n in task.notices],
    ]
    along = [b for b in task.blocks if b.kind == "alongside"]
    closed = [b for b in task.blocks if b.kind == "closed"]
    if along:
        lines += ["", "## Already alongside at hour 0 (cannot move)"]
        lines += [f"- {b.label}: sections {b.first}-{b.last} until hour {b.end}" for b in along]
    if closed:
        lines += ["", "## Closed sections"]
        lines += [f"- sections {b.first}-{b.last}, hours {b.start}-{b.end} ({b.label})" for b in closed]
    if task.cranes:
        r = task.rules
        rate = r.get("crane_rate")
        lines += ["", "## Quay cranes and movements",
                  f"- The terminal has {r['crane_pool']} quay cranes." + "".join(
                      f" From hour {o['start']} to hour {o['end']} only {r['crane_pool'] - o['cranes']} are available."
                      for o in r.get("crane_outages", [])),
                  f"- At most {r['max_moves_per_hour']} ships may berth or leave in any one hour (pilots and tugs)."]
        if rate:
            lines.append(f"- A crane handles {rate} container moves per hour: hours alongside = ceil(moves / ({rate} x cranes)).")
        for w in r.get("no_moves", []):
            who = f"ships of {w['min_length']} m or more" if w.get("min_length") else "all ships"
            lines.append(f"- No berthing or leaving for {who} from hour {w['start']} to hour {w['end']} ({w['reason']}).")
        if along:
            lines.append("- Each ship already alongside keeps 2 cranes until it leaves.")
        lines += ["", "## Ships to berth", "",
                  f"| id | ship | length m | sections | arrives (hour) | workload ({'moves' if rate else 'crane-hours'}) | cranes min-max | planned berth (hour @ section, cranes) | planned departure (hour) | notes |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for s in sorted(task.ships, key=lambda s: (s.planned_hour if s.planned_hour is not None else s.arrival, s.id)):
            planned = ("none (unscheduled)" if s.planned_section is None
                       else f"{s.planned_hour} @ {s.planned_section}, {s.std_cranes} cranes")
            notes = []
            if s.berth_deadline is not None:
                notes.append(f"emergency: dock by hour {s.berth_deadline} (+{s.deadline_penalty} per hour later)")
            if s.weight != 1:
                notes.append(f"priority: lateness x{s.weight}")
            lines.append(f"| {s.id} | {s.name} | {s.length_m:.0f} | {s.sections} | {s.arrival} | {s.workload * (rate or 1)} | "
                         f"{s.min_cranes}-{s.max_cranes} | {planned} | {s.due} | {'; '.join(notes)} |")
        return "\n".join(lines)
    lines += ["", "## Ships to berth", "",
              "| id | ship | length m | sections | arrives (hour) | hours alongside | planned berth (hour @ section) | planned departure (hour) |",
              "|---|---|---|---|---|---|---|---|"]
    for s in sorted(task.ships, key=lambda s: (s.planned_hour if s.planned_hour is not None else s.arrival, s.id)):
        planned = "none (unscheduled)" if s.planned_section is None else f"{s.planned_hour} @ {s.planned_section}"
        lines.append(f"| {s.id} | {s.name} | {s.length_m:.0f} | {s.sections} | {s.arrival} | {s.handling} | {planned} | {s.due} |")
    return "\n".join(lines)


def rules(task: Task, max_checks: int) -> str:
    if task.cranes:
        return crane_rules(task, max_checks)
    return f"""You are the berth planner for this quay. The published plan no longer works. Make a new berth plan for every ship in the table.

Rules (a plan that breaks any of them is infeasible):
- A ship berths at or after its arrival hour and stays for exactly its hours alongside.
- It occupies `sections` consecutive sections starting at the section you give, all within {task.first_section}-{task.last_section}.
- No two ships may use the same section in the same hour. Ships already alongside and closed sections are off limits for the hours listed.
  A ship that leaves at hour h frees its sections for a ship berthing at hour h.

Cost (lower is better):
- Delay: for each ship, (its sections) x (hours it departs after its planned departure). Departure = berth hour + hours alongside. Leaving early earns nothing.
- Moves: {MOVE_PENALTY} for each scheduled ship you put on a different starting section than planned. Changing only the hour costs nothing beyond delay. Unscheduled calls have no planned section.

Plan format - one entry per ship, using the ids from the table:
[{{"ship": 0, "berth_hour": 36, "section": 22}}, {{"ship": 1, "berth_hour": 163, "section": 24}}, ...]

Tools:
- get_situation(): this situation again.
- check_plan(plan): lists rule violations in your plan and its cost. You have {max_checks} checks.
- submit_plan(plan): your final answer. It ends the episode and is graded once: an infeasible plan scores low, a feasible one scores higher the lower its cost.
Submit before you run out of turns; an episode that ends without a submission scores 0."""


def brief(task: Task, max_checks: int) -> str:
    return rules(task, max_checks) + "\n\n" + situation(task)


def crane_rules(task: Task, max_checks: int) -> str:
    r = task.rules
    return f"""You are the berth planner for this quay. The published plan no longer works. Make a new berth plan for every ship in the table: when it berths, where, and how many quay cranes work it.

Rules (a plan that breaks any of them is infeasible):
- A ship berths at or after its arrival hour.
- It gets between its min and max cranes. {"Hours alongside = ceil(moves / (" + str(r["crane_rate"]) + " x cranes))" if r.get("crane_rate") else "Hours alongside = ceil(workload / cranes)"}; it departs when the work is done, unless that hour is inside a no-movement window for that ship, in which case it waits alongside (still occupying its sections, cranes free) until the window ends.
- No ship may berth inside a no-movement window that applies to it.
- It occupies `sections` consecutive sections starting at the section you give, all within {task.first_section}-{task.last_section}.
- No two ships may use the same section in the same hour. Ships already alongside and closed sections are off limits for the hours listed.
  A ship that leaves at hour h frees its sections for a ship berthing at hour h.
- In every hour, the cranes of all ships alongside (including ships already alongside) may not exceed the cranes available that hour.
- At most {r['max_moves_per_hour']} movements per hour: a ship berthing at hour h and a ship leaving at hour h each count as one movement in hour h.

Cost (lower is better):
- Delay: for each ship, (its sections) x (hours it departs after its planned departure). Leaving early earns nothing.
- Moves: {MOVE_PENALTY} for each scheduled ship you put on a different starting section than planned. Changing the hour or the cranes costs nothing beyond delay. Unscheduled calls have no planned section.
- Priority ships (notes column): their delay counts that many times. Emergency ships (notes column): each hour they dock after their deadline adds the stated amount.

Plan format - one entry per ship, using the ids from the table:
[{{"ship": 0, "berth_hour": 36, "section": 22, "cranes": 3}}, {{"ship": 1, "berth_hour": 163, "section": 24, "cranes": 2}}, ...]

Tools:
- get_situation(): this situation again.
- check_plan(plan): lists rule violations in your plan and its cost. You have {max_checks} checks.
- submit_plan(plan): your final answer. It ends the episode and is graded once: an infeasible plan scores low, a feasible one scores higher the closer its cost is to the best possible.
Submit before you run out of turns; an episode that ends without a submission scores 0."""
