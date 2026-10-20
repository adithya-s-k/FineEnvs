"""Costs in passenger-minutes: the passengers' delay and cancellations, priced by notice, and the airline's own cost of
its aircraft's minutes, at the FAA's rates.

A cancelled passenger costs v1's K_CANCEL (523) plus one passenger-minute for every minute the cancellation was decided
later than NOTICE minutes before the flight's scheduled departure: a cancellation decided two hours or more ahead costs
the flat 523 per passenger, one decided at the scheduled departure 643. Every confirmed cancellation is final and
counts as decided at its confirm; an auto-cancel counts as decided at its deadline, 523 + (deadline - sched_dep + 120)
per passenger (1,543 at a 15-h deadline). Passenger delay is v1's, pax x minutes late at arrival.

The airline's cost comes from the FAA's "Economic Values for FAA Investment and Regulatory Decisions", Section 4
(Aircraft Operating Costs), Table 4-7: the Part 121 passenger carriers' direct operating and fixed costs per block
hour over the year ended June 2023. Narrowbodies of 165k lb MTOW or more (JetBlue's A320, A321 and A321P): fuel and oil
$2,663, maintenance $954, crew $1,378, total fixed $656, total $5,650. Narrowbodies under 165k lb (its A220 and E190):
$2,322, $1,004, $1,336, $587 and $5,248. CPI-U (BLS series CUUR0000SA0) takes them to 2025 dollars: the 2025 annual
average, 321.943, over the mean of July 2022 to June 2023, 299.685, a factor of 1.0743. The FAA's $1.07 per
passenger-minute in 2025 dollars (v1's core/cost.py) takes them to whole passenger-minutes per aircraft-minute
(`RATES`):
- each minute a flight leaves after its scheduled departure costs crew and fixed: 34 (165k lb or more), 32 (under);
- each minute its block runs past its scheduled block, on the taxiway or in the air, costs the total: 95, 88;
- a cancellation, by the desk or at its deadline, saves fuel and oil and maintenance over the scheduled block, 61 and
  56 a minute, credited against its price; crew pay is taken as guaranteed, and fixed costs are not saved.
A flight that left but never arrived (a diversion on the recorded day) cancels for its passengers and saves nothing.
The end-of-day ferries are v1's, (30 + miles / 8) minutes of block, now at the FAA total rate in place of A4A's 92 so
that every aircraft-minute is priced alike. The flat cost, every cancellation at 523 whenever decided, carries the
airline's cost too: only the notice term sets it apart, and it is kept for the leverage and control clauses, which the
notice term would otherwise decide.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from ..core.cost import FERRY_BASE_MIN, FERRY_MILES_PER_MIN, K_CANCEL, miles
from .model import Flight, Task

NOTICE = 120
FAA_YE_JUNE_2023 = {
    "165k lb or more": {"fuel_oil": 2663, "maintenance": 954, "crew": 1378, "fixed": 656, "total": 5650},
    "under 165k lb": {"fuel_oil": 2322, "maintenance": 1004, "crew": 1336, "fixed": 587, "total": 5248}}
CPI_2025, CPI_YE_JUNE_2023 = 321.943, 299.685
DOLLARS_PER_PAX_MINUTE = 1.07
MTOW = {"A320": "165k lb or more", "A321": "165k lb or more", "A321P": "165k lb or more", "A220": "under 165k lb",
        "E190": "under 165k lb"}


@dataclass(frozen=True)
class Rates:
    """Passenger-minutes per aircraft-minute: at the gate past the schedule, of block past the scheduled block, and
    saved by a cancellation over its scheduled block."""

    gate: int
    block: int
    saved: int


def per_minute(dollars_per_block_hour: float) -> int:
    return round(dollars_per_block_hour / 60 * CPI_2025 / CPI_YE_JUNE_2023 / DOLLARS_PER_PAX_MINUTE)


def rates(c: Mapping[str, int]) -> Rates:
    return Rates(per_minute(c["crew"] + c["fixed"]), per_minute(c["total"]),
                 per_minute(c["fuel_oil"] + c["maintenance"]))


RATES = {family: rates(FAA_YE_JUNE_2023[c]) for family, c in MTOW.items()}


@dataclass(frozen=True)
class Cost:
    delay: int
    cancel: int
    notice: int
    ferry: int
    gate: int
    overrun: int
    saved: int

    @property
    def total(self) -> int:
        return self.flat + self.notice

    @property
    def flat(self) -> int:
        return self.delay + self.cancel + self.ferry + self.gate + self.overrun - self.saved

    def to_dict(self) -> dict:
        return {"delay": self.delay, "cancel": self.cancel, "notice": self.notice, "ferry": self.ferry,
                "gate": self.gate, "overrun": self.overrun, "saved": self.saved, "total": self.total, "flat": self.flat}


def late(f: Flight, decided: int) -> int:
    """The minutes a cancellation decided at `decided` came after the notice line."""
    return max(0, decided - (f.sched_dep - NOTICE))


def cancel_price(f: Flight, decided: int) -> int:
    return flat_price(f) + f.pax * late(f, decided)


def flat_price(f: Flight) -> int:
    return f.pax * K_CANCEL - RATES[f.family].saved * f.sched_block


def flown_price(f: Flight, dep: int, arr: int) -> int:
    """A flight that left at `dep` and arrived at `arr`: its passengers' delay and its aircraft's minutes."""
    r = RATES[f.family]
    return (f.pax * max(0, arr - f.sched_arr) + r.gate * max(0, dep - f.sched_dep)
            + r.block * max(0, arr - dep - f.sched_block))


def ferry_cost(task: Task, family: str, a: str, b: str) -> int:
    if a == b:
        return 0
    block = FERRY_BASE_MIN + miles(task.airports[a], task.airports[b]) / FERRY_MILES_PER_MIN
    return round(RATES[family].block * block)


def priced(task: Task, dep: Mapping[int, int | None], arr: Mapping[int, int | None], decided: Mapping[int, int],
           ferries: Iterable[tuple[str, str, str]]) -> Cost:
    """The cost of a day: `dep` and `arr` map each flight to its departure and its arrival, None when it had none,
    `decided` each flight that never arrived to the minute its cancellation was decided, `ferries` are the end-of-day
    (family, from, to) moves."""
    delay = cancel = notice = gate = overrun = saved = 0
    for i, a in arr.items():
        f, d = task.flight(i), dep[i]
        r = RATES[f.family]
        if d is not None:
            gate += r.gate * max(0, d - f.sched_dep)
        if a is None:
            cancel += K_CANCEL * f.pax
            notice += f.pax * late(f, decided[i])
            saved += 0 if d is not None else r.saved * f.sched_block
        else:
            delay += f.pax * max(0, a - f.sched_arr)
            overrun += r.block * max(0, a - d - f.sched_block)
    return Cost(delay, cancel, notice, sum(ferry_cost(task, *x) for x in ferries), gate, overrun, saved)
