"""Tasks and plans.

A task is one week at one Barcelona container quay after something went wrong: the quay is a row of numbered
sections (the port's own numbering, about 42 m each), every ship needs a run of consecutive sections for a fixed
number of hours, and the published plan no longer works. Times are whole hours from Monday 00:00 UTC.

A plan gives every ship a berthing hour and the first (lowest-numbered) section it occupies. Tasks with quay-crane
rules (`task.rules["crane_pool"]`) also take a crane count per ship: hours alongside = ceil(workload / cranes), the
cranes working at any hour may not exceed the pool, and at most `max_moves_per_hour` ships may berth or leave in any
one hour.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any

MOVE_PENALTY = 5  # cost of moving a ship off its planned sections, in section-hours of delay


@dataclass
class Ship:
    id: int
    name: str
    imo: str
    length_m: float
    sections: int              # consecutive sections the ship needs
    arrival: int               # earliest hour it can berth (after any delay notice)
    handling: int              # hours alongside, loading and discharging
    planned_hour: int | None   # berthing hour in the published plan (None: unscheduled call)
    planned_section: int | None
    due: int                   # planned departure hour; leaving later costs sections x hours
    from_port: str = ""
    to_port: str = ""
    beam_m: float | None = None
    draught_m: float | None = None
    workload: int | None = None     # crane-hours of loading and discharging (crane-rule tasks only)
    min_cranes: int | None = None
    max_cranes: int | None = None
    std_cranes: int | None = None   # cranes the published plan assumed; `handling` is the stay at this count
    weight: int = 1                  # delay multiplier (live tasks: a genuine transshipment connection)
    berth_deadline: int | None = None  # live tasks: a genuine emergency must be berthed by this hour ...
    deadline_penalty: int = 0          # ... or pay this much per hour berthed after it

    @property
    def planned_departure(self) -> int:
        return self.due

    def handling_for(self, cranes: int | None) -> int:
        """Hours alongside with `cranes` quay cranes (the published stay when cranes do not apply)."""
        if self.workload is None or cranes is None:
            return self.handling
        return max(1, math.ceil(self.workload / cranes))


@dataclass
class Block:
    """A rectangle of quay that no ship may use: a ship already alongside, or closed sections."""
    first: int
    last: int                  # inclusive
    start: int                 # hour
    end: int                   # hour, exclusive
    kind: str                  # "alongside" | "closed"
    label: str = ""
    cranes: int = 0            # cranes working a ship already alongside (crane-rule tasks)


@dataclass
class Task:
    task_id: str
    split: str
    quay: str
    terminal: str
    first_section: int
    last_section: int
    section_m: float
    week: int
    week_start_utc: str
    ships: list[Ship]
    blocks: list[Block]
    notices: list[str]
    disruptions: list[dict]
    difficulty: str
    reference: dict = field(default_factory=dict)   # naive/optimal costs and plans: never shown to the agent
    rules: dict = field(default_factory=dict)       # crane_pool, crane_outages, max_moves_per_hour, reward

    @property
    def n_sections(self) -> int:
        return self.last_section - self.first_section + 1

    def ship(self, sid: int) -> Ship:
        return self.ships[sid]

    def to_dict(self, public: bool = False) -> dict:
        d = asdict(self)
        if public:
            d.pop("reference")
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Task":
        d = dict(d)
        d["ships"] = [Ship(**s) for s in d["ships"]]
        d["blocks"] = [Block(**b) for b in d["blocks"]]
        d.setdefault("reference", {})
        d.setdefault("rules", {})
        return cls(**d)

    @property
    def cranes(self) -> bool:
        return bool(self.rules.get("crane_pool"))

    def crane_pool_at(self, hour: int) -> int:
        pool = int(self.rules["crane_pool"])
        for o in self.rules.get("crane_outages", []):
            if o["start"] <= hour < o["end"]:
                pool -= int(o["cranes"])
        return pool

    def no_move_windows(self, ship: "Ship") -> list[tuple[int, int]]:
        """Merged hour windows [start, end) in which this ship may not berth or leave (wind limits by length)."""
        ws = sorted((w["start"], w["end"]) for w in self.rules.get("no_moves", [])
                    if ship.length_m >= w.get("min_length", 0))
        out: list[list[int]] = []
        for a, b in ws:
            if out and a <= out[-1][1]:
                out[-1][1] = max(out[-1][1], b)
            else:
                out.append([a, b])
        return [(a, b) for a, b in out]

    def departure(self, ship: "Ship", finish: int) -> int:
        """A ship that finishes inside a no-movement window waits alongside until the window ends."""
        for a, b in self.no_move_windows(ship):
            if a <= finish < b:
                return b
        return finish

    def cranes_of(self, plan_entry: tuple) -> int | None:
        return plan_entry[2] if len(plan_entry) > 2 else None

    def planned(self) -> dict[int, tuple]:
        """The published plan for ships that had one (with the standard crane count on crane-rule tasks)."""
        return {s.id: (s.planned_hour, s.planned_section) + ((s.std_cranes,) if self.cranes else ())
                for s in self.ships if s.planned_section is not None}


Plan = dict[int, tuple]   # ship id -> (berthing hour, first section) or (berthing hour, first section, cranes)


class PlanError(ValueError):
    pass


def parse_plan(task: Task, raw: Any) -> tuple[Plan, list[str]]:
    """Read a plan from what a model sends: a list of {ship, berth_hour, section} objects (or a JSON string of
    one), or a {ship_id: [hour, section]} mapping. Ships may be named by id or by name. Returns the plan and a list
    of problems with entries that could not be read (unknown ship, duplicate, non-integer values)."""
    if isinstance(raw, str):
        text = raw.strip()
        fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
        if fence:
            text = fence.group(1)
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as e:
            raise PlanError(f"plan is not valid JSON: {e.msg} at position {e.pos}") from None
    if isinstance(raw, dict) and "plan" in raw and len(raw) == 1:
        raw = raw["plan"]
    entries: list[tuple[Any, Any, Any, Any]] = []
    if isinstance(raw, dict):
        for k, v in raw.items():
            if isinstance(v, dict):
                entries.append((k, v.get("berth_hour", v.get("hour")), v.get("section"), v.get("cranes")))
            elif isinstance(v, (list, tuple)) and len(v) in (2, 3):
                entries.append((k, v[0], v[1], v[2] if len(v) == 3 else None))
            else:
                entries.append((k, None, None, None))
    elif isinstance(raw, list):
        for e in raw:
            if not isinstance(e, dict):
                raise PlanError("each plan entry must be an object like {\"ship\": 3, \"berth_hour\": 40, \"section\": 12}")
            entries.append((e.get("ship", e.get("id", e.get("name"))), e.get("berth_hour", e.get("hour")), e.get("section"),
                            e.get("cranes")))
    else:
        raise PlanError("plan must be a list of {ship, berth_hour, section} objects")

    if len(entries) > 4 * len(task.ships) + 10:
        raise PlanError(f"plan has {len(entries)} entries for {len(task.ships)} ships")
    horizon = 100 * 24 * 7  # no plan needs hours beyond ~100 weeks; refusing them keeps grading cheap
    by_name: dict[str, list[int]] = {}
    for s in task.ships:
        by_name.setdefault(s.name.upper(), []).append(s.id)
    plan: Plan = {}
    problems: list[str] = []
    for key, hour, section, cranes in entries:
        sid = None
        if isinstance(key, bool):
            pass
        elif isinstance(key, int) or (isinstance(key, str) and key.strip().lstrip("-").isdigit()):
            sid = int(key)
        elif isinstance(key, str) and len(by_name.get(key.strip().upper(), [])) == 1:
            sid = by_name[key.strip().upper()][0]
        if sid is None or not 0 <= sid < len(task.ships):
            problems.append(f"unknown ship {key!r}: use the ship ids from the table")
            continue
        if sid in plan:
            problems.append(f"ship {sid} appears more than once; the first entry is used")
            continue
        try:
            if isinstance(hour, bool) or isinstance(section, bool):
                raise ValueError
            h, sec = int(hour), int(section)
            if h != float(hour) or sec != float(section):
                raise ValueError
        except (TypeError, ValueError, OverflowError):
            problems.append(f"ship {sid}: berth_hour and section must be whole numbers")
            continue
        if not (-horizon <= h <= horizon and -1000 <= sec <= 1000):
            problems.append(f"ship {sid}: berth_hour or section out of range")
            continue
        if task.cranes:
            if cranes is None:
                cranes = task.ships[sid].std_cranes
            try:
                if isinstance(cranes, bool):
                    raise ValueError
                c = int(cranes)
                if c != float(cranes) or not -1000 <= c <= 1000:
                    raise ValueError
            except (TypeError, ValueError, OverflowError):
                problems.append(f"ship {sid}: cranes must be a whole number")
                continue
            plan[sid] = (h, sec, c)
        else:
            plan[sid] = (h, sec)
    return plan, problems


def plan_to_list(plan: Plan) -> list[dict]:
    out = []
    for sid, entry in sorted(plan.items()):
        row = {"ship": sid, "berth_hour": entry[0], "section": entry[1]}
        if len(entry) > 2 and entry[2] is not None:
            row["cranes"] = entry[2]
        out.append(row)
    return out


def plan_from_list(rows: list[dict]) -> Plan:
    return {int(r["ship"]): (int(r["berth_hour"]), int(r["section"])) + ((int(r["cranes"]),) if r.get("cranes") is not None else ())
            for r in rows}
