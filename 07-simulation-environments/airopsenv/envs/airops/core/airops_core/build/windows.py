"""The window of a (day, hub): whole local hours, 3 to 6 of them, starting from 05:00 to 20:00.

A day's disruption (B6's hub movements cancelled for weather or the National Airspace System, plus their weather and
NAS delay minutes) ranks the disrupted days and places a control day's window. A disrupted day's window is where
planning pays most: the one where the empty plan (naive) is furthest above the hindsight optimum."""

import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from ..core.model import Task
from ..core.sim import execute
from ..refs import mip, policies
from . import fetch

CANCEL_WEIGHT = 523
"""A weather or NAS cancellation weighs as the cost model's cancellation per passenger, in minutes of delay."""
MIN_HOURS, MAX_HOURS = 3, 6
FIRST_HOUR, LAST_START, LAST_HOUR = 5, 20, 24
SHARE = 0.8
"""The disruption window is the shortest that holds this share of the most disruption any allowed window holds."""
PROVE_SECONDS = 60.0


def spans() -> list[tuple[int, int]]:
    """Every candidate window as (first hour, end hour), shortest first, then earliest."""
    return [(a, a + n) for n in range(MIN_HOURS, MAX_HOURS + 1)
            for a in range(FIRST_HOUR, min(LAST_START, LAST_HOUR - n) + 1)]


def hourly(day: pd.DataFrame, hub: str) -> dict[int, float]:
    """Disruption by local hour at the hub; `day` is the carrier's flights of the date with sd/sa as local minutes."""
    weather = (day.CancellationCode.isin(["B", "C"]) & (day.Cancelled == 1)) * CANCEL_WEIGHT
    weather = weather + day.WeatherDelay.fillna(0) + day.NASDelay.fillna(0)
    out: dict[int, float] = {}
    for col, place in (("sd", "Origin"), ("sa", "Dest")):
        at = day[day[place] == hub]
        for h, v in weather[at.index].groupby((at[col] // 60).astype(int)).sum().items():
            out[int(h)] = out.get(int(h), 0.0) + float(v)
    return out


def movements(day: pd.DataFrame, hub: str) -> dict[int, int]:
    out: dict[int, int] = {}
    for col, place in (("sd", "Origin"), ("sa", "Dest")):
        for h, n in (day[day[place] == hub][col] // 60).astype(int).value_counts().items():
            out[int(h)] = out.get(int(h), 0) + int(n)
    return out


def choose(disruption: dict[int, float], busy: dict[int, int],
           allowed: Callable[[int, int], bool]) -> tuple[int, int, float] | None:
    """(first hour, end hour, disruption) of the shortest allowed window holding SHARE of the best allowed window's
    disruption; among those the most disrupted, then the busiest, then the earliest. None if no window is allowed."""
    windows = [(b - a, a, sum(disruption.get(h, 0.0) for h in range(a, b)), sum(busy.get(h, 0) for h in range(a, b)))
               for a, b in spans() if allowed(a, b)]
    if not windows:
        return None
    best = max(w[2] for w in windows)
    length, a, score, _ = min((w for w in windows if w[2] >= SHARE * best), key=lambda w: (w[0], -w[2], -w[3], w[1]))
    return a, a + length, score


def leverage(task: dict, cache: Path) -> dict:
    """Naive's cost, the hindsight optimum of a window's task and whether HiGHS proved it within PROVE_SECONDS;
    cached under the digest of the task and the solver settings, so a rebuild reads what the first build solved."""
    path = cache / f"{fetch.digest({'task': task, 'gap': mip.GAP, 'seconds': PROVE_SECONDS})[:32]}.json"
    if path.exists():
        return json.loads(path.read_text())
    t = Task.from_dict(task)
    sol = mip.solve(t, time_limit=PROVE_SECONDS)
    out = {"naive": execute(t, policies.naive(t)).cost.total, "hindsight": sol.objective,
           "proven": sol.proven and sol.wall_s <= PROVE_SECONDS}
    fetch.save(path, json.dumps(out).encode())
    return out


def gap(r: dict) -> float:
    """How far naive is above the hindsight optimum, relative to it."""
    return round((r["naive"] - r["hindsight"]) / max(r["hindsight"], 1), 6)


def most_leverage(results: list[dict]) -> dict | None:
    """Of {"window": [w0, w1], **leverage} rows, the proven one of the largest gap; ties to the earlier start, then
    the shorter. None if none is proven."""
    return min((r for r in results if r["proven"]), key=lambda r: (-gap(r), *r["window"]), default=None)
