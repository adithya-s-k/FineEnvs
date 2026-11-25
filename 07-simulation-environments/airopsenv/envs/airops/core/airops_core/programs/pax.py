"""Passengers (v3): who is booked on a storm day's flights and where they are going, what the day does to them, and
what that costs: cost P, as registered for step 0, with JetBlue's own contract for who can't get there the same day.

Bookings (`book`). A flight's booked passengers are its `pax` (seats x T-100 route-month load factor, model.Flight).
At JetBlue's connecting stations (CONNECT) some of a route's departing passengers arrived on an earlier JetBlue
flight: the flows per origin, connecting station and destination of the date's survey period (`Flows.load`: the DB1B
quarter up to 2025 Q2, then the DB1C month; tasks/pax/b6-transfers.csv), each its share of the route's departing
passengers in the survey times the route's booked passengers that day. A route-day's connecting passengers are split
over origins by largest remainder, and an origin's over the day's scheduled JetBlue connections onto the route from
that origin (scheduled connection from the station's MCT to DWELL minutes) in proportion to DB1C's B6-to-B6
connection-time distribution at the station (`Dwell`, tasks/pax/b6-dwell.csv), never more than either flight has left:
a passenger connects at most once (three-coupon itineraries are 0.04 to 0.2% of B6's), so an itinerary has one leg or
two. Routes in (connecting station, destination) order, origins in name order. Everyone else on a flight is local.
Deterministic. Not modelled: international connections (the survey and the BTS on-time network are domestic only),
interline connections, spill.

The day (`fly`), a pass over a finished day: passengers never change what aircraft and crews do. Crews keep seat
priority, so a flight's seats for passengers are its aircraft's seats less the people of the crew units riding it (the
family's smallest seats for a flight that never flew); the crew sim is unchanged. A group flies its booked legs; it is
disrupted when a leg it holds is cancelled (at the decision, or when it reaches the leg's origin if that is later) or
when it misses a connection (it arrives later than the outbound's departure less the station's MCT; at the outbound's
pushback). JetBlue rebooks on JetBlue only (Contract of Carriage §37(c): "reaccommodation on the next available JetBlue
flight in the same class of service"), first come, first served, the earliest scheduled arrival first (the default
rebooking): disruptions in time order, ties by the scheduled departure of the flight they were on, then group id; at its
minute a disrupted group gives up its seats on the legs it can no longer fly and takes, in order of scheduled arrival at
its final destination (then scheduled departure, fewer legs, ids), the JetBlue itineraries to its final destination the
schedule sells from where it is, direct or through one connecting station (from its MCT to DWELL minutes), every leg not
cancelled by then and leaving at or after the group is ready, as many of the group as each has seats for. Ready: at its
origin one MCT after the decision (an assumption: a passenger at its origin is taken to be at the airport), at a
connection its arrival plus the MCT. A flight leaving 3 h or more before the one it booked from there is a significant
change the passenger may refuse (DOT, 14 CFR 260.2; not in regulation before 2024-10-28, the same line applied on every
date): it takes one only when no other seat is left that day or in the DAYS following days, else such a move would be a
refund. A rebooked flight cancelled later, or a connection missed again, disrupts it again. Who has no seat that day
takes, in the order stranded and in the same order of arrival, the seats of the DAYS following days' published BTS
schedules (`Seat`, seats left after their own T-100 loads; with none given, today's schedule on each following day,
seats not limited), a night away for each night it waits; who has none within those days, nor an earlier flight today,
is refunded at that minute, per JetBlue's Customer Service Plan ("Rebook to an alternate JetBlue flight within five
days" or a full refund).

Cost P (cost.priced, in passenger-minutes at the FAA's $1.07): every v2 term but two. Each passenger's delay is charged
at the final destination, rebooked or a later day's arrival included, in place of v2's delay at every leg; a refunded
passenger's delay is the time it waited, from its booked departure to the refund. A cancelled flight's booked
passengers cost 66 each (K_CANCEL's refund-exposure part, v1's assumed 0.3 take-up of the $235 DB1B median fare) in
place of 523, their real delay replacing its 457 minutes. Every night away costs one GSA lodging night for the city
(crews.lodging; labelled: many of those stranded are at their own origin and may be at home). Every refunded passenger
costs the full fare (`fare`: the period's median one-way B6 market fare, tasks/pax/b6-fares.csv, CPI to 2025). JetBlue
owes no compensation, meals or hotel on a storm day (force majeure), so none is charged.

The MIP's passengers (`Proxy`, mip.Network with `pax`): a flight's copies carry the delay of the passengers it takes to
their destination, connecting passengers included; every booked connection a binary that is 1 when its inbound flew and
its outbound was cancelled or left within the MCT of the inbound's arrival; and every way a group can be disrupted (its
first leg cancelled at its decision minute, or let auto-cancel at its deadline; or, its inbound flown, its connection
broken) rebooks it onto every itinerary the pass would offer it, as the schedule has them (from one MCT after the
decision; from the outbound's scheduled departure for a broken connection), at their scheduled arrival plus the nights
away, within each flight's seats as booked (its recorded tail's seats less its passengers, none on a cancelled one; the
following days' seats left), a flight 3 h or more earlier weighing more than any other seat and a refund more than any
seat (`last_resort`), so that, as in the pass, the earlier flight is taken only when no other seat is left and only who
has no seat at all is refunded. It departs from the day's pass in four things: a disrupted connecting passenger also
keeps the delay of the outbound it was booked on, in place of the rebooked flights' own delays, which it does not see;
it does not count the seats freed by passengers disrupted off a flight; it reads the seats of the tail JetBlue flew, not
the one the plan's dispatch picks; and it assigns seats optimally where the pass serves first come first. `Proxy.cost`
is a day under it (the assignment an LP), what the MIP's objective is for that day; `optimal` is a day's passengers
rebooked optimally over its actual flights and seats.
"""

import csv
import heapq
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path

import highspy
import numpy as np

from ..core.sim import AUTO_CANCELLED, CANCELLED, FLOWN
from . import crews
from .cost import RATES, REFUND, flown_price, late
from .model import Flight, Task

CONNECT = ("JFK", "BOS", "FLL", "MCO", "SJU")
MCT = {"JFK": 45, "BOS": 45}
MCT_ELSEWHERE = 40
DWELL = 360
DAYS = 5
EARLIER = 180
DATA = Path(__file__).resolve().parents[3] / "tasks" / "pax"
FLEW, REBOOKED, STRANDED, REFUNDED = "flew", "rebooked", "stranded", "refunded"
INF = highspy.kHighsInf


def mct(station: str) -> int:
    """DB1C's scheduled minimums: B6-to-B6 connections hardly ever under 45 minutes at JFK and BOS, 40 elsewhere."""
    return MCT.get(station, MCT_ELSEWHERE)


@dataclass(frozen=True)
class Group:
    id: int
    legs: tuple[int, ...]
    pax: int

    @classmethod
    def from_list(cls, v: list) -> "Group":
        return cls(v[0], tuple(v[1]), v[2])

    def to_list(self) -> list:
        return [self.id, list(self.legs), self.pax]


@dataclass(frozen=True)
class Seat:
    """A flight of the `day`-th following day's published schedule, in minutes from the task date's midnight, its
    passengers booked and its seats left."""

    flight: str
    orig: str
    dest: str
    dep: int
    arr: int
    free: int
    day: int = 1
    booked: int = 0

    @classmethod
    def from_list(cls, v: list) -> "Seat":
        return cls(*v)

    def to_list(self) -> list:
        return [self.flight, self.orig, self.dest, self.dep, self.arr, self.free, self.day, self.booked]


@dataclass(frozen=True)
class Passengers:
    """`later` the following days' schedules with their seats left, `fare` a refunded passenger's fare in
    passenger-minutes."""

    groups: tuple[Group, ...]
    later: tuple[Seat, ...] = ()
    source: str = ""
    fare: float = 0.0

    def booked(self) -> Counter:
        out: Counter = Counter()
        for g in self.groups:
            for k in g.legs:
                out[k] += g.pax
        return out

    def to_dict(self) -> dict:
        return {"source": self.source, "fare": self.fare, "groups": [g.to_list() for g in self.groups],
                "later": [s.to_list() for s in self.later]}

    @classmethod
    def from_dict(cls, d: Mapping) -> "Passengers":
        return cls(tuple(Group.from_list(g) for g in d["groups"]), tuple(Seat.from_list(s) for s in d["later"]),
                   d["source"], d["fare"])


@dataclass(frozen=True)
class Flows:
    """Per route (connecting station, destination) of a survey period, the share of its departing passengers that
    arrived on a JetBlue flight from each origin ("*": from any, spread over the inbound flights by passengers)."""

    shares: Mapping[tuple[str, str], Mapping[str, float]]
    label: str = ""

    def scaled(self, k: float) -> "Flows":
        return Flows({r: {o: k * v for o, v in by.items()} for r, by in self.shares.items()}, f"{self.label} x{k:g}")

    @classmethod
    def synthetic(cls, task: Task, share: float) -> "Flows":
        return cls({(f.orig, f.dest): {"*": share} for f in task.flights if f.orig in CONNECT}, f"synthetic {share:g}")

    @classmethod
    def load(cls, date: str, path: Path | str | None = None) -> "Flows":
        with open(str(path or DATA / "b6-transfers.csv"), newline="") as fh:
            rows = list(csv.DictReader(fh))
        p = period(date, {r["period"] for r in rows})
        shares: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
        source = ""
        for r in rows:
            if r["period"] == p and r["orig"] != "*" and r["dest"] != "*":
                shares[(r["hub"], r["dest"])][r["orig"]] = int(r["transfers"]) / int(r["departing"])
                source = r["source"]
        return cls(dict(shares), f"{source} {p}")


def fare(date: str, path: Path | str | None = None) -> float:
    """The full fare a refunded passenger costs: the date's survey period's median one-way market fare on B6
    (tasks/pax/b6-fares.csv), in passenger-minutes."""
    with open(str(path or DATA / "b6-fares.csv"), newline="") as fh:
        rows = {r["period"]: float(r["median_fare"]) for r in csv.DictReader(fh)}
    return crews.value(rows[period(date, rows)], date)


def period(date: str, periods: Iterable[str]) -> str:
    """The survey period of a date: its DB1B quarter up to 2025 Q2, then its DB1C month; where that is missing, the
    period whose middle is nearest."""
    y, m = int(date[:4]), int(date[5:7])
    periods = sorted(periods)
    own = f"{y}Q{(m - 1) // 3 + 1}" if (y, m) <= (2025, 6) else f"{y}-{m:02d}"
    if own in periods:
        return own
    day = Date.fromisoformat(date)
    return min(periods, key=lambda p: abs((_middle(p) - day).days))


def _middle(p: str) -> Date:
    if "Q" in p:
        y, q = p.split("Q")
        return Date(int(y), 3 * int(q) - 1, 15)
    return Date(int(p[:4]), int(p[5:7]), 15)


@dataclass(frozen=True)
class Dwell:
    """DB1C's B6-to-B6 transfers per scheduled connection time at each connecting station, pooled over its months,
    in bins [lo, hi); a station without its own takes every station's."""

    bins: Mapping[str, tuple[tuple[int, int, int], ...]]

    def weight(self, hub: str, minutes: int) -> float:
        for lo, hi, n in self.bins.get(hub) or self.bins["*"]:
            if lo <= minutes < hi:
                return n / (hi - lo)
        return 0.0

    @classmethod
    def uniform(cls) -> "Dwell":
        return cls({"*": ((0, DWELL + 1, 1),)})

    @classmethod
    def load(cls, path: Path | str | None = None) -> "Dwell":
        with open(str(path or DATA / "b6-dwell.csv"), newline="") as fh:
            rows = [r for r in csv.DictReader(fh) if r["period"] == "all"]
        bins: dict[str, list] = defaultdict(list)
        pooled: Counter = Counter()
        for r in rows:
            lo, hi, n = int(r["lo"]), int(r["hi"]), int(r["transfers"])
            bins[r["hub"]].append((lo, hi, n))
            pooled[(lo, hi)] += n
        bins["*"] = [(lo, hi, n) for (lo, hi), n in sorted(pooled.items())]
        return cls({h: tuple(sorted(v)) for h, v in bins.items()})


def book(task: Task, flows: Flows, dwell: Dwell, later: Iterable[Seat] = (), refunded: float | None = None
         ) -> Passengers:
    left = {f.id: f.pax for f in task.flights}
    inbound: dict[str, list[Flight]] = defaultdict(list)
    outbound: dict[tuple[str, str], list[Flight]] = defaultdict(list)
    for f in sorted(task.flights, key=lambda f: (f.sched_dep, f.id)):
        if f.dest in CONNECT:
            inbound[f.dest].append(f)
        if f.orig in CONNECT:
            outbound[(f.orig, f.dest)].append(f)
    pairs: Counter = Counter()
    for (h, d), outs in sorted(outbound.items()):
        by = {o: v for o, v in flows.shares.get((h, d), {}).items() if o != d and v > 0}
        load = sum(f.pax for f in outs)
        for o, n in sorted(split(round(sum(by.values()) * load), by).items()):
            ins = [i for i in inbound[h] if (i.orig == o if o != "*" else i.orig != d)]
            weight = {(i.id, j.id): dwell.weight(h, j.sched_dep - i.sched_arr) * (i.pax if o == "*" else 1)
                      for i in ins for j in outs if mct(h) <= j.sched_dep - i.sched_arr <= DWELL}
            pairs.update(allot(n, weight, left))
    legs = [(p, n) for p, n in pairs.items() if n] + [((i,), n) for i, n in left.items() if n > 0]
    return Passengers(tuple(Group(k, legs, n) for k, (legs, n) in enumerate(sorted(legs))), tuple(later),
                      flows.label, fare(task.date) if refunded is None else refunded)


def split(n: int, weight: Mapping) -> dict:
    """n over the keys by weight, by largest remainder, ties to the smaller key."""
    total = sum(weight.values())
    if n <= 0 or total <= 0:
        return {}
    share = {k: n * w / total for k, w in weight.items()}
    out = {k: int(s) for k, s in share.items()}
    for k in sorted(share, key=lambda k: (int(share[k]) - share[k], k))[:n - sum(out.values())]:
        out[k] += 1
    return {k: v for k, v in out.items() if v}


def allot(n: int, weight: Mapping[tuple[int, int], float], left: dict[int, int]) -> Counter:
    """n over the flight pairs by weight, by largest remainder, none past what either flight has `left` (taken)."""
    out: Counter = Counter()
    while n > 0:
        live = {p: w for p, w in weight.items() if w > 0 and left[p[0]] > 0 and left[p[1]] > 0}
        if not live:
            break
        got = 0
        for p, k in sorted(split(n, live).items()):
            k = min(k, left[p[0]], left[p[1]])
            left[p[0]] -= k
            left[p[1]] -= k
            out[p] += k
            got += k
        if not got:
            break
        n -= got
    return out


@dataclass(frozen=True)
class Part:
    """Some of a group's passengers and how their day ended: `legs` the flights they flew, `arr` their minute at the
    destination (past 1440 a later day; the refund's minute for those refunded), `at` where they spent `nights` away or
    were refunded."""

    group: int
    pax: int
    kind: str
    legs: tuple[int | str, ...]
    arr: int
    delay: int
    at: str | None
    nights: int = 0

    def to_dict(self) -> dict:
        return {"group": self.group, "pax": self.pax, "kind": self.kind, "legs": list(self.legs), "arr": self.arr,
                "delay": self.delay, "at": self.at, "nights": self.nights}


@dataclass(frozen=True)
class Day:
    parts: tuple[Part, ...]
    events: tuple[dict, ...]
    delay: int
    lodging: int
    fare: int

    def outcomes(self) -> dict[str, int]:
        c: Counter = Counter()
        for p in self.parts:
            c[p.kind] += p.pax
        return {k: c[k] for k in (FLEW, REBOOKED, STRANDED, REFUNDED)}

    def misconnected(self) -> int:
        return sum(e["pax"] for e in self.events if e["kind"] == "misconnect")

    def to_dict(self) -> dict:
        return {"delay": self.delay, "lodging": self.lodging, "fare": self.fare, "outcomes": self.outcomes(),
                "misconnected": self.misconnected(), "parts": [p.to_dict() for p in self.parts],
                "events": list(self.events)}


class Timetable:
    """A schedule as sold: itineraries of one flight, or two through a connecting station, connecting from its MCT to
    DWELL minutes (DB1C's B6-to-B6 connections: 2.0 to 2.7% wait 6 to 24 h)."""

    def __init__(self, flights: Iterable):
        self.od: dict[tuple[str, str], list] = defaultdict(list)
        for f in sorted(flights, key=lambda f: (_dep(f), _arr(f), _key(f))):
            self.od[(f.orig, f.dest)].append(f)

    def itineraries(self, s: str, d: str, ok: Callable, then: Callable) -> list[tuple]:
        """The itineraries from s to d whose first flight passes `ok` and second `then`, by scheduled arrival, then
        scheduled departure, fewer flights and the flights."""
        out = [(_arr(f), _dep(f), 1, (_key(f),), (f,)) for f in self.od.get((s, d), ()) if ok(f)]
        for h in CONNECT:
            if h in (s, d) or not self.od.get((s, h)) or not self.od.get((h, d)):
                continue
            seconds = [g for g in self.od[(h, d)] if then(g)]
            for f in self.od[(s, h)]:
                if ok(f):
                    out += [(_arr(g), _dep(f), 2, (_key(f), _key(g)), (f, g)) for g in seconds
                            if mct(h) <= _dep(g) - _arr(f) <= DWELL]
        return [legs for *_, legs in sorted(out, key=lambda x: x[:4])]


def _dep(f) -> int:
    return f.sched_dep if isinstance(f, Flight) else f.dep


def _arr(f) -> int:
    return f.sched_arr if isinstance(f, Flight) else f.arr


def _key(f) -> int | str:
    return f.id if isinstance(f, Flight) else f.flight


class Later:
    """The DAYS following days' published schedules and their seats left, first come first served, the earliest
    scheduled arrival first; without them, today's schedule on each following day, seats not limited."""

    def __init__(self, task: Task, seats: tuple[Seat, ...]):
        self.flights = seats or tuple(Seat(f"{f.id} +{k}", f.orig, f.dest, f.sched_dep + 1440 * k,
                                           f.sched_arr + 1440 * k, -1, k) for k in range(1, DAYS + 1)
                                      for f in task.flights)
        self.table = Timetable(self.flights)
        self.free = {s.flight: s.free for s in self.flights}

    def itineraries(self, s: str, d: str, after: int) -> list[tuple]:
        return self.table.itineraries(s, d, lambda f: f.dep >= after, lambda f: True)

    def take(self, s: str, d: str, n: int, after: int) -> list[tuple[tuple[Seat, ...], int]]:
        """Seats for n from s to d leaving at or after `after`: (flights, how many) in order, as many as there are."""
        out = []
        for legs in self.itineraries(s, d, after):
            limited = [self.free[f.flight] for f in legs if self.free[f.flight] >= 0]
            m = min(n, *limited) if limited else n
            if m <= 0:
                continue
            for f in legs:
                if self.free[f.flight] >= 0:
                    self.free[f.flight] -= m
            out.append((legs, m))
            n -= m
            if not n:
                break
        return out


def booked_from(task: Task, g: Group, s: str, was: Flight) -> int:
    """The scheduled departure the group booked from station s: its leg from there, else the flight it lost."""
    return next((task.flight(k).sched_dep for k in g.legs if task.flight(k).orig == s), was.sched_dep)


def nights(legs: tuple[Seat, ...]) -> int:
    """The nights away before a following day's itinerary: the days from the task's date to the schedule day it is
    on."""
    return legs[0].day


def seats(task: Task, runs: Mapping, riders: Mapping[int, int]) -> dict[int, int]:
    """Each flight's seats for passengers: its aircraft's (the family's smallest if it never flew) less its riders."""
    by_tail = {t.tail: t.seats for t in task.tails}
    least = smallest(task)
    return {f.id: (by_tail[runs[f.id].tail] if runs[f.id].status == FLOWN else least[f.family])
            - riders.get(f.id, 0) for f in task.flights}


def smallest(task: Task) -> dict[str, int]:
    out: dict[str, int] = {}
    for t in task.tails:
        out[t.family] = min(out.get(t.family, t.seats), t.seats)
    return out


def fly(task: Task, passengers: Passengers, runs: Mapping, riders: Mapping[int, int] | None = None) -> Day:
    """The day's passengers on its flights `runs` (sim.FlightRun by id), `riders` the crew people riding each."""
    return _Pass(task, passengers, runs, riders or {}).run()


def ledger(task: Task, passengers: Passengers, runs: Mapping, riders: Mapping[int, int] | None = None
           ) -> tuple[Day, dict[str, Counter]]:
    """`fly`'s day and its ledger by flight: who flew it, who was added to it (every leg of an itinerary they were
    rebooked onto), who dropped it (booked or added, disrupted on an earlier leg of that itinerary), and by group who
    made its booked connection."""
    p = _Pass(task, passengers, runs, riders or {})
    return p.run(), {"flown": p.flown, "added": p.added, "dropped": p.dropped, "made": p.made}


class _Pass:
    def __init__(self, task: Task, passengers: Passengers, runs: Mapping, riders: Mapping[int, int]):
        self.task, self.groups, self.runs, self.refunded = task, passengers.groups, runs, passengers.fare
        self.table = Timetable(task.flights)
        self.later = Later(task, passengers.later)
        self.cap = seats(task, runs, riders)
        self.hold = passengers.booked()
        self.heap: list = []
        self.parts: list[Part] = []
        self.events: list[dict] = []
        self.flown, self.added, self.dropped, self.made = Counter(), Counter(), Counter(), Counter()

    def run(self) -> Day:
        for g in self.groups:
            self.walk(g, g.pax, g.legs, (), None, self.task.start)
        while self.heap:
            self.disrupted(*heapq.heappop(self.heap)[-1])
        parts = tuple(sorted(self.parts, key=lambda p: (p.group, p.kind, p.arr, tuple(map(str, p.legs)))))
        date = self.task.date
        lodging = sum(x.pax * x.nights * crews.lodging(x.at, date) for x in parts if x.kind == STRANDED)
        return Day(parts, tuple(sorted(self.events, key=lambda e: (e["t"], e["group"], e["kind"]))),
                   sum(x.pax * x.delay for x in parts), round(lodging),
                   round(self.refunded * sum(x.pax for x in parts if x.kind == REFUNDED)))

    def walk(self, g: Group, n: int, legs: tuple[int, ...], flew: tuple[int, ...], prev: int | None,
             now: int) -> None:
        """n of g on `legs` from the first's origin, there from `prev` (their arrival; None at their origin)."""
        for x, k in enumerate(legs):
            f, r = self.task.flight(k), self.runs[k]
            if r.status != FLOWN:
                self.push(max(r.decided, now if prev is None else prev), g, n, f, legs[x + 1:], flew, prev)
                return
            if prev is not None and r.dep < prev + mct(f.orig):
                self.push(max(r.dep, now), g, n, f, legs[x + 1:], flew, prev)
                self.events.append({"t": r.dep, "kind": "misconnect", "at": f.orig, "flight": k, "from": flew[-1],
                                    "pax": n, "group": g.id})
                return
            flew, prev = (*flew, k), r.arr
            self.flown[k] += n
        if flew == g.legs and len(flew) == 2:
            self.made[g.id] += n
        self.done(g, n, FLEW if flew == g.legs else REBOOKED, flew, prev, None)

    def push(self, t: int, g: Group, n: int, f: Flight, *rest) -> None:
        heapq.heappush(self.heap, (t, f.sched_dep, g.id, len(self.heap), (t, g, n, f, *rest)))

    def disrupted(self, t: int, g: Group, n: int, was: Flight, rest: tuple[int, ...], flew: tuple[int, ...],
                  prev: int | None) -> None:
        for k in rest:
            self.hold[k] -= n
            self.dropped[k] += n
        task, runs, s = self.task, self.runs, was.orig
        dest = task.flight(g.legs[-1]).dest
        ready = t + mct(s) if prev is None else max(t, prev + mct(s))

        line = booked_from(task, g, s, was) - EARLIER

        def ok(f: Flight) -> bool:
            r = runs[f.id]
            return r.dep >= ready if r.status == FLOWN else r.decided > t and f.sched_dep >= ready

        def then(f: Flight) -> bool:
            r = runs[f.id]
            return r.dep >= ready if r.status == FLOWN else r.decided > t

        n = self.today(t, g, n, was, flew, prev, s, dest, lambda f: ok(f) and f.sched_dep > line, then)
        sched = task.flight(g.legs[-1]).sched_arr
        for legs, m in self.later.take(s, dest, n, ready):
            self.events.append({"t": t, "kind": "hotel", "at": s, "flight": was.id, "pax": m, "group": g.id,
                                "until": legs[0].dep})
            self.parts.append(Part(g.id, m, STRANDED, (*flew, *(f.flight for f in legs)), legs[-1].arr,
                                   max(0, legs[-1].arr - sched), s, nights(legs)))
            n -= m
        n = self.today(t, g, n, was, flew, prev, s, dest, lambda f: ok(f) and f.sched_dep <= line, then)
        if n:
            self.events.append({"t": t, "kind": "home", "at": s, "flight": was.id, "pax": n, "group": g.id})
            self.parts.append(Part(g.id, n, REFUNDED, flew, t, max(0, t - task.flight(g.legs[0]).sched_dep), s))

    def today(self, t: int, g: Group, n: int, was: Flight, flew: tuple[int, ...], prev: int | None, s: str,
              dest: str, ok, then) -> int:
        """n of g onto today's itineraries passing `ok` and `then`, in the default order: how many found no seat."""
        for legs in self.table.itineraries(s, dest, ok, then):
            ids = tuple(f.id for f in legs)
            m = min(n, *(self.cap[k] - self.hold[k] for k in ids))
            if m <= 0:
                continue
            for k in ids:
                self.hold[k] += m
                self.added[k] += m
            self.events.append({"t": t, "kind": "rebook", "at": s, "flight": was.id, "to": ids[0], "pax": m,
                                "group": g.id})
            self.walk(g, m, ids, flew, prev, t)
            n -= m
            if not n:
                break
        return n

    def done(self, g: Group, n: int, kind: str, flew: tuple, arr: int, at: str | None) -> None:
        self.parts.append(Part(g.id, n, kind, flew, arr, max(0, arr - self.task.flight(g.legs[-1]).sched_arr), at))


@dataclass(frozen=True)
class Link:
    """A booked connection: `pax` from flight i onto j at `hub`."""

    i: int
    j: int
    hub: str
    pax: int


@dataclass(frozen=True)
class Mode:
    """One way a group is disrupted: its first leg `cut` (cancelled, or with `lapse` let auto-cancel), or its `link`
    broken with its inbound flown; `options` the itineraries it is rebooked onto (today's flight ids, the following
    days' Seat names) and each one's price per passenger, `refund` the price per passenger of a refund."""

    group: int
    pax: int
    cut: int | None
    lapse: bool
    link: tuple[int, int] | None
    options: tuple[tuple[tuple[int | str, ...], int], ...]
    refund: int


class Proxy:
    """The passengers as the MIP prices them (see the module docstring); `decided` the minute each cancellation is
    decided (the first watch by default)."""

    def __init__(self, task: Task, passengers: Passengers, decided: Mapping[int, int] | None = None):
        self.task, decided = task, decided or {}
        table, later = Timetable(task.flights), Later(task, passengers.later)
        booked, least = passengers.booked(), smallest(task)
        tails = {t.tail: t.seats for t in task.tails}
        self.free: dict[int | str, int] = {f.id: max(0, tails.get(f.actual.tail, least[f.family]) - booked[f.id])
                                           for f in task.flights}
        self.free |= {k: v for k, v in later.free.items() if v >= 0}
        self.lined = {i for t in task.tails if any(leg in task.moved for leg in t.legs) for i in t.legs}
        self.term: Counter = Counter()
        links: Counter = Counter()
        modes = []
        for g in passengers.groups:
            first, last = task.flight(g.legs[0]), task.flight(g.legs[-1])
            self.term[last.id] += g.pax
            t = decided.get(first.id, task.start)
            modes.append(self._mode(table, later, passengers, g, first.orig, t, t + mct(first.orig), first.id, False,
                                    None))
            if first.id in self.lined:
                t = task.auto_at(first)
                modes.append(self._mode(table, later, passengers, g, first.orig, t, t + mct(first.orig), first.id,
                                        True, None))
            if len(g.legs) == 2:
                links[(first.id, last.id)] += g.pax
                modes.append(self._mode(table, later, passengers, g, last.orig, last.sched_dep, last.sched_dep, None,
                                        False, (first.id, last.id)))
        self.links = tuple(Link(i, j, task.flight(j).orig, n) for (i, j), n in sorted(links.items()))
        self.modes = tuple(modes)

    def _mode(self, table: Timetable, later: Later, passengers: Passengers, g: Group, s: str, t: int, ready: int,
              cut: int | None, lapse: bool, link: tuple[int, int] | None) -> Mode:
        first, last = self.task.flight(g.legs[0]), self.task.flight(g.legs[-1])
        line = next(self.task.flight(k).sched_dep for k in g.legs if self.task.flight(k).orig == s) - EARLIER

        def ok(f: Flight) -> bool:
            return f.sched_dep >= ready and f.id not in g.legs

        night = crews.lodging(s, self.task.date)
        options = [(tuple(f.id for f in legs), max(0, legs[-1].sched_arr - last.sched_arr))
                   for legs in table.itineraries(s, last.dest, lambda f: ok(f) and f.sched_dep > line, ok)]
        options += [(tuple(f.flight for f in legs), round(max(0, legs[-1].arr - last.sched_arr) + nights(legs) * night))
                    for legs in later.itineraries(s, last.dest, ready)]
        options += [(tuple(f.id for f in legs), round(last_resort(max(0, legs[-1].sched_arr - last.sched_arr),
                                                                  [c for _, c in options])))
                    for legs in table.itineraries(s, last.dest, lambda f: ok(f) and f.sched_dep <= line, ok)]
        charge = round(max(0, t - first.sched_dep) + passengers.fare)
        return Mode(g.id, g.pax, cut, lapse, link, tuple(options), last_resort(charge, [c for _, c in options]))

    @classmethod
    def floor(cls, task: Task, passengers: Passengers) -> "Proxy":
        """Every passenger term P adds at its floor, for a lower bound: the delay of local passengers only, a
        cancellation's refund exposure and notice with no rebooking, no misconnect, rebooking, lodging or refund
        (each left out is at least 0)."""
        out = cls(task, Passengers(()), {})
        out.term = Counter()
        for g in passengers.groups:
            if len(g.legs) == 1:
                out.term[g.legs[0]] += g.pax
        return out

    @classmethod
    def lower(cls, task: Task, passengers: Passengers, arrive: Mapping[int, int]) -> "Proxy":
        """A model whose optimum is a lower bound on P: the local passengers' delay on their flight, exact; a
        connection that holds, 0; and every way a group is disrupted (the misconnect binaries as in the proxy) sending
        each passenger to the cheapest outcome P could give it: a flight into its destination that day from where it
        is or a connecting station at its earliest possible arrival (`arrive`, whatever the leg before or the minute
        the passenger is ready), within the largest seats of its family less its local passengers (riders, freed
        seats and other passengers' legs relaxed); the first scheduled arrival into its destination over the
        following days and one night at the cheapest lodging of the day's stations (their seats relaxed); or a
        refund at the fare alone. A flight that costs no less than the later days or the refund, both always open,
        is left out. A cancellation costs its refund exposure and the notice at the first watch, less the saving, as
        in P."""
        out = cls(task, Passengers(()), {})
        most: dict[str, int] = {}
        for t in task.tails:
            most[t.family] = max(most.get(t.family, t.seats), t.seats)
        local: Counter = Counter()
        for g in passengers.groups:
            if len(g.legs) == 1:
                local[g.legs[0]] += g.pax
        out.term = local
        out.free = {f.id: max(0, most[f.family] - local[f.id]) for f in task.flights}
        into: dict[str, list[Flight]] = defaultdict(list)
        for f in task.flights:
            into[f.dest].append(f)
        first = {}
        for s in Later(task, passengers.later).flights:
            first[s.dest] = min(first.get(s.dest, s.arr), s.arr)
        night = min(crews.lodging(s, task.date) for s in {f.orig for f in task.flights} | set(CONNECT))
        refund = math.floor(passengers.fare)
        links: Counter = Counter()
        modes = []
        for g in passengers.groups:
            head, last = task.flight(g.legs[0]), task.flight(g.legs[-1])
            later = [((), math.floor(max(0, first[last.dest] - last.sched_arr) + night))] if last.dest in first else []
            worst = min([refund] + [c for _, c in later])
            today = [(k.orig, (k.id,), c) for k in into[last.dest]
                     if (c := max(0, arrive[k.id] - last.sched_arr)) < worst]
            kinds = [(head.id, False, None, head.orig)]
            kinds += [(head.id, True, None, head.orig)] if head.id in out.lined else []
            if len(g.legs) == 2:
                links[(head.id, last.id)] += g.pax
                kinds.append((None, False, (head.id, last.id), last.orig))
            modes += [Mode(g.id, g.pax, cut, lapse, link,
                           tuple([(k, c) for o, k, c in today if o == s or o in CONNECT] + later), refund)
                      for cut, lapse, link, s in kinds]
        out.links = tuple(Link(i, j, task.flight(j).orig, n) for (i, j), n in sorted(links.items()))
        out.modes = tuple(modes)
        return out

    def flown(self, f: Flight, dep: int, arr: int) -> int:
        return flown_price(f, dep, arr, self.term.get(f.id, 0))

    def price(self, f: Flight, decided: int, flat: bool = False) -> int:
        """A cancellation's price before rebooking: its passengers' refund exposure and notice, less the saving."""
        return f.pax * (REFUND + (0 if flat else late(f, decided))) - RATES[f.family].saved * f.sched_block

    def broken(self, link: Link | tuple[int, int], runs: Mapping) -> bool:
        """Whether a booked connection broke with its inbound flown: its outbound cancelled, or gone within the MCT."""
        i, j = (link.i, link.j) if isinstance(link, Link) else link
        a, b = runs[i], runs[j]
        return a.status == FLOWN and (b.status != FLOWN or b.dep < a.arr + mct(self.task.flight(j).orig))

    def hit(self, mode: Mode, runs: Mapping) -> bool:
        if mode.link is not None:
            return self.broken(mode.link, runs)
        status = runs[mode.cut].status
        if mode.lapse:
            return status == AUTO_CANCELLED
        return status == CANCELLED or (status == AUTO_CANCELLED and mode.cut not in self.lined)

    def cost(self, run, flat: bool = False) -> float:
        """A day (sim.Execution) under the proxy, its passengers rebooked as the LP would: the MIP's objective."""
        task, runs = self.task, run.flights
        total = float(run.cost.ferry)
        for f in task.flights:
            r = runs[f.id]
            total += self.flown(f, r.dep, r.arr) if r.status == FLOWN else self.price(f, r.decided, flat)
        return total + self.assign(runs)[0]

    def assign(self, runs: Mapping) -> tuple[float, dict[tuple[int, int], float], dict[int, float]]:
        """The rebooking LP of a day: each disrupted mode's passengers over its options within the seats as booked of
        the flights that flew and the following days' seats, or refunded; its cost, and the passengers per (mode,
        option) and refunded per mode."""
        cols, cost, rows = [], [], []
        room: dict[int | str, dict[int, float]] = defaultdict(dict)
        for m, mode in enumerate(self.modes):
            if not self.hit(mode, runs):
                continue
            row = {}
            for o, (legs, price) in enumerate(mode.options):
                if all(isinstance(k, str) or runs[k].status == FLOWN for k in legs):
                    row[len(cols)] = 1.0
                    for k in legs:
                        if k in self.free:
                            room[k][len(cols)] = 1.0
                    cols.append((m, o))
                    cost.append(price)
            row[len(cols)] = 1.0
            cols.append((m, None))
            cost.append(mode.refund)
            rows.append((row, mode.pax, mode.pax))
        rows += [(r, -INF, self.free[k]) for k, r in sorted(room.items(), key=lambda kv: str(kv[0]))]
        x = _lp(cost, [float(self.modes[m].pax) for m, _ in cols], rows) if cols else np.zeros(0)
        r = {(m, o): v for (m, o), v in zip(cols, x, strict=True) if o is not None and v > 1e-9}
        s = {m: v for (m, o), v in zip(cols, x, strict=True) if o is None and v > 1e-9}
        return float(np.dot(cost, x)) if cols else 0.0, r, s


def _lp(cost: list[float], ub: list[float], rows: list[tuple[dict[int, float], float, float]]) -> np.ndarray:
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", 1)
    n = len(cost)
    h.addVars(n, np.zeros(n), np.array(ub, dtype=float))
    h.changeColsCost(n, np.arange(n, dtype=np.int32), np.array(cost, dtype=float))
    starts, idx, val = [], [], []
    for coefs, _, _ in rows:
        starts.append(len(idx))
        idx += list(coefs)
        val += list(coefs.values())
    h.addRows(len(rows), np.array([r[1] for r in rows], dtype=float), np.array([r[2] for r in rows], dtype=float),
              len(idx), np.array(starts, dtype=np.int32), np.array(idx, dtype=np.int32), np.array(val, dtype=float))
    h.run()
    return np.array(h.getSolution().col_value)


def optimal(task: Task, passengers: Passengers, runs: Mapping, riders: Mapping[int, int] | None = None) -> Day:
    """The day's passengers rebooked optimally instead of first come first served: every group disrupted off its
    booked legs is assigned, over the day's flights as they flew (each leg leaving at or after it is ready, every
    connection at the MCT on the actual minutes) and the following days' schedules, at the least total delay, lodging
    and refunded fares, within each flight's seats less its riders and the booked passengers it carried, or refunded
    (an LP, so parts may be fractional)."""
    cap = seats(task, runs, riders or {})
    table, later = Timetable(task.flights), Later(task, passengers.later)
    carried: Counter = Counter()
    parts, lost = [], []
    for g in passengers.groups:
        prev, flew, hit = None, (), None
        for k in g.legs:
            f, r = task.flight(k), runs[k]
            if r.status != FLOWN or (prev is not None and r.dep < prev + mct(f.orig)):
                t = max(r.decided, task.start if prev is None else prev) if r.status != FLOWN else r.dep
                hit = (f.orig, t, t + mct(f.orig) if prev is None else max(t, prev + mct(f.orig)), flew,
                       f.sched_dep - EARLIER)
                break
            carried[k] += g.pax
            flew, prev = (*flew, k), r.arr
        if hit is None:
            parts.append(Part(g.id, g.pax, FLEW, flew, prev, max(0, prev - task.flight(g.legs[-1]).sched_arr), None))
        else:
            lost.append((g, *hit))
    cols: list[tuple[int, tuple, str, int, float]] = []
    rows, room, extra = [], defaultdict(dict), []

    def add(n: int, legs: tuple, kind: str, d: int, h: float, keys: tuple, rank: float = 0.0) -> None:
        row[len(cols)] = 1.0
        for k in keys:
            room[k][len(cols)] = 1.0
        cols.append((n, legs, kind, d, h))
        extra.append(rank)

    for n, (g, s, t, ready, _, line) in enumerate(lost):
        last = task.flight(g.legs[-1])

        def ok(f: Flight, ready=ready) -> bool:
            return runs[f.id].status == FLOWN and runs[f.id].dep >= ready

        def feasible(x: tuple) -> bool:
            return len(x) == 1 or runs[x[1].id].dep >= runs[x[0].id].arr + mct(x[0].dest)

        night = crews.lodging(s, task.date)
        row = {}
        for x in table.itineraries(s, last.dest, lambda f, ok=ok, line=line: ok(f) and f.sched_dep > line, ok):
            if feasible(x):
                add(n, tuple(f.id for f in x), REBOOKED, max(0, runs[x[-1].id].arr - last.sched_arr), 0.0,
                    tuple(f.id for f in x))
        for x in later.itineraries(s, last.dest, ready):
            add(n, tuple(f.flight for f in x), STRANDED, max(0, x[-1].arr - last.sched_arr), nights(x) * night,
                tuple(f.flight for f in x if later.free[f.flight] >= 0))
        floor = [d + h for m, _, _, d, h in cols if m == n]
        for x in table.itineraries(s, last.dest, lambda f, ok=ok, line=line: ok(f) and f.sched_dep <= line, ok):
            if feasible(x):
                d = max(0, runs[x[-1].id].arr - last.sched_arr)
                add(n, tuple(f.id for f in x), REBOOKED, d, 0.0, tuple(f.id for f in x), last_resort(d, floor) - d)
        d = max(0, t - task.flight(g.legs[0]).sched_dep)
        mine = [c[3] + c[4] + e for c, e in zip(cols, extra, strict=True) if c[0] == n]
        add(n, (), REFUNDED, d, passengers.fare, (), last_resort(d + passengers.fare, mine) - d - passengers.fare)
        rows.append((row, g.pax, g.pax))
    rows += [(r, -INF, float(later.free[k] if isinstance(k, str) else cap[k] - carried[k]))
             for k, r in sorted(room.items(), key=lambda kv: str(kv[0]))]
    x = _lp([d + h + e for (*_, d, h), e in zip(cols, extra, strict=True)],
            [float(lost[n][0].pax) for n, *_ in cols], rows) if cols else []
    delay, lodging, fares = sum(p.pax * p.delay for p in parts), 0.0, 0.0
    for v, (n, legs, kind, d, h) in zip(x, cols, strict=True):
        if v <= 1e-9:
            continue
        g, s, t, _, flew, _ = lost[n]
        sched = task.flight(g.legs[-1]).sched_arr
        parts.append(Part(g.id, v, kind, (*flew, *legs), t if kind == REFUNDED else sched + d, d,
                          None if kind == REBOOKED else s, nights_of(legs, later) if kind == STRANDED else 0))
        delay += v * d
        lodging += v * h if kind == STRANDED else 0.0
        fares += v * h if kind == REFUNDED else 0.0
    return Day(tuple(parts), (), round(delay), round(lodging), round(fares))


def last_resort(charge: float, options: list[float]) -> float:
    """What a refund weighs in an optimal assignment: its charge, or more than any seat the group could take, so that
    the assignment refunds only who JetBlue has no seat for within DAYS days, as the policy does."""
    return max(charge, max(options, default=0) + 1)


def nights_of(names: tuple[str, ...], later: Later) -> int:
    return next(s.day for s in later.flights if s.flight == names[0])
