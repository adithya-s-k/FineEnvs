"""When each party's news reaches the planner in a live week.

A one-week task's disruptions are revealed a fixed lead before the hour they are about, grouped into 6-hour bulletins
(watches); closures are planned works, known at hour 0."""

import re
from dataclasses import dataclass
from datetime import datetime, timedelta

from berth_core import Task
from berth_core.prompts import when

FREEZE_HOURS = 6
BULLETIN_HOURS = 6
PLANNING_CALLS = 3
TURNS_PER_WATCH = 5
MAX_TURNS = TURNS_PER_WATCH * 9 + 2
"""Every week gets the turns of the longest (9 watches), so the turns-left note doesn't reveal how many
bulletins are coming."""
WIND_MAX_TURNS = TURNS_PER_WATCH * 10 + 2
"""Every wind week gets the turns of the longest wind week (10 watches), for the same reason."""
LINES_TURNS_PER_WATCH = 7
LINES_MAX_TURNS = LINES_TURNS_PER_WATCH * 10 + 2
"""The wind week's watches with two more turns each, for messaging the ships' agents."""
LEADS = {"late": 24, "bunching": 24, "extra": 24, "priority": 24, "emergency": 12, "crane_outage": 24, "gale": 24,
         "tug_outage": 24, "pilot_shortage": 24, "forecast": FREEZE_HOURS}
SPEAKERS = {"bunching": "Ship agents", "gale": "Harbour master", "emergency": "Harbour master",
            "closure": "Terminal ops", "crane_outage": "Terminal ops", "priority": "Line desk",
            "tug_outage": "Tug company", "pilot_shortage": "Pilot station", "forecast": "Barcelona Port Control"}
HARBOUR_MASTER = "Harbour master"
ONE_WEEK = re.compile(r"dock-\w+-w\d{2}x1-")
TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


@dataclass(frozen=True)
class Notice:
    event_id: str
    index: int
    kind: str
    name: str
    hour: int
    event_hour: int
    text: str
    event: dict


@dataclass(frozen=True)
class Watch:
    index: int
    hour: int
    notices: tuple[Notice, ...]


def _event_hour(task: Task, e: dict) -> int:
    kind = e["type"]
    if kind == "late":
        return task.ships[e["ship"]].arrival - e["hours"]
    if kind == "bunching":
        return min(task.ships[s].planned_hour for s in e["ships"])
    if kind in ("extra", "priority", "emergency"):
        return task.ships[e["ship"]].arrival
    return e["start"]


def _own_ships(e: dict) -> list[int]:
    return [e["ship"]] if e["type"] in ("late", "extra") else e["ships"] if e["type"] == "bunching" else []


def schedule(task: Task) -> list[Watch]:
    if not ONE_WEEK.match(task.task_id):
        raise ValueError(f"{task.task_id} is not a one-week task: live weeks are one week")
    events = task.disruptions
    hours = [0 if e["type"] == "closure" else max(0, _event_hour(task, e) - LEADS[e["type"]]) for e in events]
    own: dict[int, int] = {}
    for e, h in zip(events, hours, strict=True):
        for s in _own_ships(e):
            own[s] = max(own.get(s, 0), h)
    hours = [max(h, own.get(e["ship"], 0)) if e["type"] in ("priority", "emergency") else h
             for e, h in zip(events, hours, strict=True)]
    notices = [Notice(f"{e['type']}-{i}", i, e["type"],
                      task.ships[e["ship"]].name if e["type"] in ("late", "extra") else SPEAKERS[e["type"]],
                      hours[i], _event_hour(task, e), task.notices[i], e)
               for i, e in enumerate(events)]
    bulletins = sorted({h // BULLETIN_HOURS * BULLETIN_HOURS for h in hours} | {0})
    return [Watch(k, b, tuple(n for n in notices if n.hour // BULLETIN_HOURS * BULLETIN_HOURS == b))
            for k, b in enumerate(bulletins)]


def label(hour: int) -> str:
    return when(hour)


def virtual_time(task: Task, hour: int) -> str:
    return (datetime.strptime(task.week_start_utc, TIME_FORMAT) + timedelta(hours=hour)).strftime(TIME_FORMAT)
