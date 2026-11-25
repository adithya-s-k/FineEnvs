"""Costs in passenger-minutes.

A cancelled passenger costs 523: NEXTOR's 456.9-min average delay to disrupted passengers (TDI report, 2010) plus
refund exposure, 0.3 take-up of the $235 DB1B median fare (Q1 2025) at the FAA's $1.07 per passenger-minute (Table 1-1,
$47.10/h in 2015 dollars, CPI to 2025). A ferry costs 92 × (30 + miles / 8): A4A's $98.41 airline cost per block
minute (2025) ÷ $1.07, over 30 minutes plus one minute per 8 great-circle miles (an assumed block time).
"""

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .model import Airport, Task

K_CANCEL = 523
FERRY_PER_MIN = 92
FERRY_BASE_MIN = 30
FERRY_MILES_PER_MIN = 8
EARTH_MILES = 3958.8


@dataclass(frozen=True)
class Cost:
    pax_delay: int
    cancel: int
    ferry: int

    @property
    def total(self) -> int:
        return self.pax_delay + self.cancel + self.ferry

    def to_dict(self) -> dict:
        return {"pax_delay": self.pax_delay, "cancel": self.cancel, "ferry": self.ferry, "total": self.total}


def miles(a: Airport, b: Airport) -> float:
    if a.code == b.code:
        return 0.0
    p1, p2, dl = math.radians(a.lat), math.radians(b.lat), math.radians(b.lon - a.lon)
    c = math.sin(p1) * math.sin(p2) + math.cos(p1) * math.cos(p2) * math.cos(dl)
    return EARTH_MILES * math.acos(max(-1.0, min(1.0, c)))


def ferry_cost(task: Task, a: str, b: str) -> int:
    if a == b:
        return 0
    return round(FERRY_PER_MIN * (FERRY_BASE_MIN + miles(task.airports[a], task.airports[b]) / FERRY_MILES_PER_MIN))


def cancel_cost(pax: int) -> int:
    return K_CANCEL * pax


def breakdown(task: Task, arr: Mapping[int, int | None], ferries: Iterable[tuple[str, str]]) -> Cost:
    """The cost of a day: ``arr`` maps each flight to its arrival minute, None when cancelled; ``ferries`` are the
    end-of-day (from, to) moves."""
    delay = cancel = 0
    for i, a in arr.items():
        f = task.flight(i)
        if a is None:
            cancel += cancel_cost(f.pax)
        else:
            delay += f.pax * max(0, a - f.sched_arr)
    return Cost(delay, cancel, sum(ferry_cost(task, a, b) for a, b in ferries))
