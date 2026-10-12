"""PortSim wind: the marine week in a real weather week, read at Y7 in place of the Dique Sur anemometer. The rules
follow the wind that blew (``rules["no_moves"]``, which the grade uses); at every watch Barcelona Port Control sends the
latest forecast published by then, and the week as known holds that forecast's windows and the wind observed so far,
nothing later."""

import json
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

from berth_core import Plan, Task, TaskPack
from berth_core.prompts import DAYS

from . import packs, world
from .marine import MarineWeek
from .schedule import FREEZE_HOURS, SPEAKERS, label, schedule

REFERENCES = packs.references("wind")
THRESHOLDS = ((300, 25), (0, 30))
REASONS = {300: "wind above 25 kn", 0: "wind above 30 kn"}
GAP_HOURS = 2
STEP_HOURS = 3
HORIZON_HOURS = 72
PUBLISHED_HOURS = 9
WARNING_HOURS = range(12, 168, 12)
WARNING_SPAN = 36
HOURLY = ("no_move", "moves", "pilots", "tugs")
SHARED = ("moves", "pilots", "tugs")


class NotBuilt(FileNotFoundError):
    """The wind weeks are built locally from Meteocat readings and never committed."""


def _built(path: Path) -> Path:
    if not path.exists():
        raise NotBuilt(f"wind weeks not built ({path} is missing): run python tools/build_wind.py in "
                       "07-simulation-environments/portsim-live/envs/port_week")
    return path


def pack_dir() -> Path:
    return packs.tasks_root() / packs.WIND_PACK


def pack() -> TaskPack:
    return TaskPack(_built(pack_dir()))


def references() -> list[dict]:
    return [json.loads(line) for line in _built(REFERENCES).read_text().splitlines()]


def task_ids() -> list[str]:
    return [r["task_id"] for r in references() if r["qualifies"]]


def merge(hours: Iterable[int], min_length: int, gap: int = GAP_HOURS) -> list[dict]:
    out: list[dict] = []
    for h in sorted(set(hours)):
        if out and h <= out[-1]["end"] + gap:
            out[-1]["end"] = h + 1
        else:
            out.append({"start": h, "end": h + 1, "min_length": min_length})
    return out


def run_windows(init: int, kn: list[int]) -> list[dict]:
    """The run's hours above each threshold, its 3-hourly knots interpolated linearly in time (in thirds, exactly)."""
    thirds = {init + STEP_HOURS * i + r: (STEP_HOURS - r) * kn[i] + r * kn[i + 1]
              for i in range(len(kn) - 1) for r in range(STEP_HOURS)}
    thirds[init + STEP_HOURS * (len(kn) - 1)] = STEP_HOURS * kn[-1]
    return [w for min_length, kn_limit in THRESHOLDS
            for w in merge((h for h, x in thirds.items() if x > STEP_HOURS * kn_limit), min_length)]


def delivered(weather: dict, hour: int) -> dict:
    return max((r for r in weather["runs"] if r["issued"] <= hour), key=lambda r: r["init"])


def observed_before(weather: dict, hour: int) -> list[dict]:
    return [w for min_length, _ in THRESHOLDS for w in merge(
        (h for o in weather["observed"] if o["min_length"] == min_length
         for h in range(o["start"], min(o["end"], hour))), min_length)]


def _restricted(run: dict, a: int, b: int) -> set[tuple[int, int]]:
    return {(h, w["min_length"]) for w in run["windows"] for h in range(max(w["start"], a), min(w["end"], b))}


def watch_hours(schedule_task: Task, weather: dict) -> list[int]:
    """The marine week's watches, plus each 12-hourly hour whose delivered run restricts an hour of the next 36 past
    the freeze line that the run delivered at the watch before didn't."""
    hours = [w.hour for w in schedule(schedule_task)]
    for h in WARNING_HOURS:
        if h not in hours:
            prior = max(x for x in hours if x < h)
            span = (h + FREEZE_HOURS, h + FREEZE_HOURS + WARNING_SPAN)
            if _restricted(delivered(weather, h), *span) - _restricted(delivered(weather, prior), *span):
                hours = sorted(hours + [h])
    return hours


def _reasons(windows: Iterable[dict], prefix: str = "") -> list[dict]:
    return [{"start": w["start"], "end": w["end"], "min_length": w["min_length"],
             "reason": prefix + REASONS[w["min_length"]]} for w in windows]


def _entry(weather: dict, hour: int) -> dict:
    run = delivered(weather, hour)
    entry = {"type": "forecast", "start": hour + FREEZE_HOURS, "hour": hour, "init": run["init"],
             "issued": run["issued"], "horizon": run["init"] + HORIZON_HOURS, "kn": list(run["kn"]),
             "windows": _reasons(({**w, "start": max(w["start"], hour)} for w in run["windows"] if w["end"] > hour),
                                 "forecast: "),
             "observed": _reasons(observed_before(weather, hour), "observed: ")}
    assert entry["issued"] == entry["init"] + PUBLISHED_HOURS and entry["issued"] <= hour, entry
    assert entry["horizon"] > hour + FREEZE_HOURS, entry
    return entry


def transplant(schedule_task: Task, weather: dict) -> Task:
    """The marine week with its gale dropped, the weather week's wind as its rules and a forecast at every watch."""
    hours = watch_hours(schedule_task, weather)
    kept = [i for i, e in enumerate(schedule_task.disruptions) if e["type"] != "gale"]
    entries = [_entry(weather, h) for h in hours]
    task = replace(schedule_task, task_id=f"{schedule_task.task_id}-{weather['id']}",
                   rules=schedule_task.rules | {"no_moves": _reasons(weather["windows"])},
                   disruptions=[schedule_task.disruptions[i] for i in kept] + entries,
                   notices=[schedule_task.notices[i] for i in kept] + [notice(e) for e in entries])
    assert [w.hour for w in schedule(task)] == hours
    return task


def issued(hour: int) -> str:
    return label(hour) if hour >= 0 else f"{DAYS[hour // 24 % 7]} {hour % 24:02d}:00"


def _spans(windows: list[dict]) -> str:
    return ", ".join(f"{w['start']}–{w['end']}" for w in windows)


def _nodes(entry: dict) -> list[tuple[int, int]]:
    return [(entry["init"] + STEP_HOURS * i, x) for i, x in enumerate(entry["kn"])]


def _by_length(windows: list[dict]) -> list[list[dict]]:
    return [[w for w in windows if w["min_length"] == min_length] for min_length, _ in THRESHOLDS]


def forecast_text(entry: dict) -> str:
    w25, w30 = _by_length(entry["windows"])
    peak = max(x for h, x in _nodes(entry) if entry["hour"] <= h <= entry["horizon"])
    return ((f"above 25 kn hours {_spans(w25)}" if w25 else "no hour above 25 kn") + f", peak {peak} kn"
            + (f"; above 30 kn {_spans(w30)}" if w30 else ""))


def notice(entry: dict) -> str:
    return (f"Wind forecast issued {issued(entry['issued'])} (ECMWF, adjusted to the Dique Sur anemometer), to hour "
            f"{entry['horizon']}: {forecast_text(entry)}.")


def forecast(view: Task) -> dict:
    [entry] = [e for e in view.disruptions if e["type"] == "forecast"]
    return entry


def known(task: Task, revealed: Iterable[str]) -> Task:
    """world.known with only the latest forecast kept, its observed and forecast windows the no-movement windows."""
    view = world.known(task, revealed)
    forecasts = [i for i, e in enumerate(view.disruptions) if e["type"] == "forecast"]
    if not forecasts:
        return view
    latest = max(forecasts, key=lambda i: view.disruptions[i]["hour"])
    keep = [i for i, e in enumerate(view.disruptions) if e["type"] != "forecast" or i == latest]
    entry = view.disruptions[latest]
    return replace(view, rules=view.rules | {"no_moves": entry["observed"] + entry["windows"]},
                   disruptions=[view.disruptions[i] for i in keep], notices=[view.notices[i] for i in keep])


def section(view: Task) -> str:
    e = forecast(view)
    o25, o30 = _by_length(e["observed"])
    start = e["hour"] + FREEZE_HOURS
    return "\n".join([
        "## Wind at the Dique Sur",
        "- Above 25 kn (10-minute mean) ships of 300 m or more may not berth or leave and every movement takes one "
        "more tug; above 30 kn no ship moves. The rules follow the wind that blows; these windows are forecast.",
        f"- {SPEAKERS['forecast']}, issued {issued(e['issued'])} (ECMWF, adjusted to this anemometer), to hour "
        f"{e['horizon']}: {forecast_text(e)}.",
        f"- kn every {STEP_HOURS} h from hour {start}: "
        + " ".join(str(x) for h, x in _nodes(e) if start <= h <= e["horizon"]),
        "- Observed since hour 0: " + (f"above 25 kn hours {_spans(o25)}" if o25 else "no hour above 25 kn")
        + (f"; above 30 kn {_spans(o30)}" if o30 else "") + "."])


def _plain(windows: list[dict]) -> list[dict]:
    return [{"start": w["start"], "end": w["end"], "min_length": w["min_length"]} for w in windows]


def step_wind(view: Task) -> dict:
    e = forecast(view)
    return {"hour": e["hour"], "from": SPEAKERS["forecast"], "issued": e["issued"], "time": issued(e["issued"]),
            "horizon": e["horizon"], "kn": [[h, x] for h, x in _nodes(e) if h >= e["hour"]],
            "windows": _plain(e["windows"]), "observed": _plain(e["observed"])}


def _key(problem: str) -> tuple[str, int | None]:
    rule = world.rule(problem)
    return rule, int(re.search(r"hour (\d+)", problem)[1]) if rule in HOURLY else None


class WindWeek(MarineWeek):
    notice_excuse = False

    @property
    def view(self) -> Task:
        return known(self.task, self.revealed)

    def situation(self) -> dict:
        out = super().situation()
        return out | {"situation": out["situation"] + "\n\n" + section(self.view)}

    def target(self) -> Task:
        """What the grade holds the plan to: the week as known, the truth once done."""
        return self.task if self.done else self.view

    def past(self, revealed: list[str], target: Task) -> Task:
        return known(self.task, revealed)

    def excuses(self, plan: Plan | None = None) -> tuple[set[tuple[int, str]], dict[int, int]]:
        """Each ship is excused the (ship, rule) problems and the cost that the target (the week as known, the truth
        once done) adds over the week as known at its last watch before it froze, the whole plan evaluated on both;
        a ship never frozen compares with the last watch reached. An hourly rule is charged only at an hour it broke
        then, so a warning that never blew doesn't stand for wind no forecast showed. A problem ships share, an
        overlap or a pool short at an hour, is judged for each as the ship decided last saw it: that decision put
        them together, on the latest news."""
        plan = self.plan if plan is None else plan
        target = self.target()
        now = {r.ship: r for r in self.evaluate(target, plan).ships}
        first: dict[int, int] = {}
        for k, f in enumerate(self.frozen):
            for s in f["ships"]:
                first.setdefault(s, k)
        decided = {s: first[s] - 1 if s in first else self.watch for s in plan}
        seen: dict[int, set[tuple[str, int | None]]] = {}
        cost = {}
        for j in sorted(set(decided.values())):
            view = self.past([e["event_id"] for e in self.log if e["watch"] <= j], target)
            before = {r.ship: r for r in self.evaluate(view, plan).ships}
            for s in (s for s in plan if decided[s] == j):
                seen[s] = {_key(p) for p in before[s].problems}
                if now[s].cost > before[s].cost:
                    cost[s] = now[s].cost - before[s].cost
        sharing: dict[tuple[str, int | None], list[int]] = defaultdict(list)
        for s in plan:
            for p in now[s].problems:
                if _key(p)[0] in SHARED:
                    sharing[_key(p)].append(s)

        def was_seen(s: int, key: tuple[str, int | None]) -> bool:
            rule, _ = key
            if rule.startswith("ship "):
                other = int(rule.split()[1])
                return key in seen[s] if decided[s] >= decided[other] else (f"ship {s}", None) in seen[other]
            return key in seen[max(sharing[key], key=decided.__getitem__) if rule in SHARED else s]

        problems = set()
        for s in plan:
            charged = {world.rule(p) for p in now[s].problems if was_seen(s, _key(p))}
            problems |= {(s, world.rule(p)) for p in now[s].problems if world.rule(p) not in charged}
        return problems, cost


def held(week: WindWeek) -> Task:
    """The week as known with every hour any forecast delivered so far showed from now on, each once: the
    hold-every-warning re-planner's view."""
    view, hour = week.view, week.hour
    shown = [week.scheduled[i].event for i in week.revealed if week.scheduled[i].kind == "forecast"]
    windows = [w for min_length, _ in THRESHOLDS for w in merge(
        (h for e in shown for w in e["windows"] if w["min_length"] == min_length
         for h in range(max(w["start"], hour), w["end"])), min_length, gap=0)]
    return replace(view, rules=view.rules | {"no_moves": forecast(view)["observed"] + _reasons(windows, "forecast: ")})
