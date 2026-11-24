"""Synthetic weather weeks in tasks/wind/weather.jsonl's format: invented wind and invented forecasts, nothing real.
``python tests/synthetic.py`` rewrites the fixture tests/fixtures/wind/weather.jsonl from WEEKS."""

import json
from collections.abc import Callable, Iterable
from datetime import datetime, timedelta
from pathlib import Path

from port_week_core import wind

FIXTURE = Path(__file__).parent / "fixtures" / "wind" / "weather.jsonl"
CALM = 12
RAMP = 4
INITS = range(-12, 247, 6)
Episodes = Callable[[int], list[tuple[int, int, int]]]


def knots(init: int, episodes: list[tuple[int, int, int]]) -> list[int]:
    """Whole knots every 3 hours from ``init``: calm, but within each (start, end, kn) episode its kn, falling by RAMP
    knots an hour outside it."""
    def at(h: int) -> int:
        return max([CALM] + [kn - RAMP * max(start - h, h - end, 0) for start, end, kn in episodes])
    return [at(init + wind.STEP_HOURS * i) for i in range(25)]


def spans(hours: Iterable[int], min_length: int) -> list[dict]:
    out: list[dict] = []
    for h in sorted(set(hours)):
        if out and h == out[-1]["end"]:
            out[-1]["end"] = h + 1
        else:
            out.append({"start": h, "end": h + 1, "min_length": min_length})
    return out


def record(index: int, kind: str, monday: str, above25: Iterable[int], above30: Iterable[int], episodes: Episodes,
           missing: Iterable[int] = (), unvalidated: float = 1.0) -> dict:
    """A weather week whose wind blew above 25 and 30 kn in the given hours, its run at each init forecasting
    ``episodes(init)``; the runs at ``missing`` inits never came."""
    above25, above30 = set(above25) | set(above30), set(above30)
    start = datetime.fromisoformat(monday)
    year, week, _ = start.isocalendar()
    runs = [{"run": (start + timedelta(hours=init)).strftime("%Y-%m-%dT%H:%M:%SZ"), "init": init,
             "issued": init + wind.PUBLISHED_HOURS, "kn": (kn := knots(init, episodes(init))),
             "windows": wind.run_windows(init, kn)} for init in INITS if init not in set(missing)]
    return {"id": f"e{index:02d}", "iso_week": f"{year}-W{week:02d}", "monday": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "kind": kind,
            "windows": [w | {"unvalidated": unvalidated} for w in wind.merge(above25, 300) + wind.merge(above30, 0)],
            "observed": spans(above25, 300) + spans(above30, 0), "runs": runs}


def storm(init: int) -> list[tuple[int, int, int]]:
    """The storm from the run at hour 12 on, its core a little early; a lull and a later peak in the last runs."""
    if init < 12:
        return []
    return [(75, 85, 27), (76, 78, 32)] if init < 60 else [(78, 88, 27), (82, 84, 31)]


def false_alarm(init: int) -> list[tuple[int, int, int]]:
    """A blow for hours 76 to 100 in the runs from hour 18 to hour 36, gone from the next one on; a short one late in
    the week that came."""
    return ([(77, 99, 28)] if 18 <= init <= 36 else []) + ([(198, 200, 27)] if init >= 150 else [])


WEEKS = [
    record(0, "storm", "2000-01-10", [*range(37, 41), *range(74, 88)], range(80, 84), storm),
    record(1, "bust", "2000-01-17", range(198, 201), (), false_alarm, missing=[108]),
]


def lines(weeks: list[dict]) -> str:
    return "".join(json.dumps(w, ensure_ascii=False, separators=(",", ":")) + "\n" for w in weeks)


if __name__ == "__main__":
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(lines(WEEKS))
    print(f"Wrote {len(WEEKS)} weeks to {FIXTURE}")
