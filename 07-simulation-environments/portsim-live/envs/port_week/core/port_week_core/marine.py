"""PortSim marine: the live week with the port's pilots and tugs, shared hour by hour with the port's other 2024 traffic
(``rules["marine"]``) and cut by the outages the tug company and the pilot station announce; an hour short of either is
a rule break for each of our ships moving in it."""

import json
from collections import defaultdict
from pathlib import Path

import berth_core
from berth_core import Plan, Task, TaskPack
from berth_core.check import PlanResult

from . import packs
from .schedule import SPEAKERS, label
from .world import Week

CONTAINER = "Portacontenidors"
NO_TUGS = {"Transbordadors", "Passatge", "Iot"}
WIND_EXEMPT = {"Transbordadors", "Ro-Ro"}
YACHT = "Iot"
CUTS = {"tug_outage": "tugs", "pilot_shortage": "pilots"}
REFERENCES = packs.references("marine")


def pack_dir() -> Path:
    return packs.tasks_root() / packs.MARINE_PACK


def pack() -> TaskPack:
    return TaskPack(pack_dir())


def references() -> list[dict]:
    return [json.loads(line) for line in REFERENCES.read_text().splitlines()]


def task_ids() -> list[str]:
    return [r["task_id"] for r in references() if r["qualifies"]]


def tugs_needed(kind: str, length_m: float, windy: bool) -> int:
    base = 0 if kind in NO_TUGS or length_m < 120 else 1 if length_m < 200 else 2 if length_m < 300 else 3
    return base + int(windy and kind != YACHT and not (kind in WIND_EXEMPT and length_m < 200))


def wind(view: Task) -> list[tuple[int, int]]:
    """The announced windows of wind above 25 kn."""
    return [(w["start"], w["end"]) for w in view.rules["no_moves"] if w["min_length"] >= 300]


def _windy(hour: int, windows: list[tuple[int, int]]) -> bool:
    return any(a <= hour < b for a, b in windows)


def free(view: Task) -> dict[str, list[int]]:
    marine, windows = view.rules["marine"], wind(view)
    left = {pool: [marine[pool]] * marine["hours"] for pool in ("pilots", "tugs")}
    for hour, kind, length_m in marine["movements"]:
        left["pilots"][hour] -= 1
        left["tugs"][hour] -= tugs_needed(kind, length_m, _windy(hour, windows))
    for e in view.disruptions:
        if pool := CUTS.get(e["type"]):
            for hour in range(e["start"], e["end"]):
                left[pool][hour] -= e[pool]
    return {pool: [max(n, 0) for n in hours] for pool, hours in left.items()}


def own_moves(task: Task, res: PlanResult) -> list[tuple[int, int, int]]:
    windows = wind(task)
    return [(r.ship, hour, tugs_needed(CONTAINER, task.ships[r.ship].length_m, _windy(hour, windows)))
            for r in res.ships if r.berth_hour is not None for hour in (r.berth_hour, r.departure)]


def evaluate(task: Task, plan: Plan) -> PlanResult:
    res = berth_core.evaluate(task, plan)
    marine, room = task.rules["marine"], free(task)
    moving: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for ship, hour, tugs in own_moves(task, res):
        moving[hour].append((ship, tugs))
    short = False
    for hour, moves in sorted(moving.items()):
        for pool, use in (("pilots", len(moves)), ("tugs", sum(tugs for _, tugs in moves))):
            room_now = room[pool][hour] if 0 <= hour < marine["hours"] else marine[pool]
            if use > room_now:
                short = True
                for ship, _ in moves:
                    res.ships[ship].problems.append(
                        f"{pool} short at hour {hour}: your ships need {use}, {room_now} free")
    return PlanResult(False, res.ships, None, None, None) if short and res.feasible else res


def _ids(ids: list[int]) -> str:
    return ", ".join(map(str, ids))


def section(view: Task) -> str:
    marine, room = view.rules["marine"], free(view)
    pilots, tugs, hours = marine["pilots"], marine["tugs"], marine["hours"]
    lines = ["## Pilots and tugs",
             f"- Each berthing and each departure takes a pilot and tugs in its hour, from the port's {pilots} pilots "
             f"and {tugs} tugs on duty, shared with the port's other traffic. In any hour your ships may need no more "
             "pilots or tugs than are free for them; each ship moving in an hour short of either breaks this rule.",
             "- Tugs per movement: 0 under 120 m, 1 under 200 m, 2 under 300 m, 3 from 300 m."]
    lines += [f"- From hour {a} to hour {b} (wind above 25 kn), each movement takes one more tug."
              for a, b in wind(view)]
    by_tugs: dict[int, list[int]] = defaultdict(list)
    for s in view.ships:
        by_tugs[tugs_needed(CONTAINER, s.length_m, False)].append(s.id)
    lines.append("- Your ships take " + "; ".join(f"{k} {'tug' if k == 1 else 'tugs'}: ships {_ids(ids)}"
                                                    for k, ids in sorted(by_tugs.items(), reverse=True)) + ".")
    lines += [f"- {SPEAKERS[e['type']]}: {e[pool]} of the {marine[pool]} {pool} out from hour {e['start']} to hour "
              f"{e['end']}." for e in view.disruptions if (pool := CUTS.get(e["type"]))]
    lines.append("- Free for your ships after the other traffic, the cuts and the wind, pilots | tugs, each hour from "
                 "00:00 in blocks of 6 hours:")
    for day in range(hours // 24):
        blocks = {pool: " ".join("".join(map(str, room[pool][h:h + 6])) for h in range(24 * day, 24 * day + 24, 6))
                  for pool in ("pilots", "tugs")}
        lines.append(f"- {label(24 * day).split()[0]}: {blocks['pilots']} | {blocks['tugs']}")
    lines.append(f"- From hour {hours}: {pilots} pilots and {tugs} tugs free.")
    return "\n".join(lines)


def step_marine(view: Task, plan: Plan) -> dict:
    marine = view.rules["marine"]
    room = free(view)
    ours: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for _, hour, tugs in own_moves(view, berth_core.evaluate(view, plan)):
        ours[hour][0] += 1
        ours[hour][1] += tugs
    return {"hours": marine["hours"],
            "pilots": {"pool": marine["pilots"], "free": room["pilots"]},
            "tugs": {"pool": marine["tugs"], "free": room["tugs"]},
            "ours": [[hour, *use] for hour, use in sorted(ours.items())],
            "cuts": [{"from": SPEAKERS[e["type"]], "pool": pool, "count": e[pool], "start": e["start"], "end": e["end"]}
                     for e in view.disruptions if (pool := CUTS.get(e["type"]))]}


class MarineWeek(Week):
    evaluate = staticmethod(evaluate)

    def situation(self) -> dict:
        out = super().situation()
        return out | {"situation": out["situation"] + "\n\n" + section(self.view)}
