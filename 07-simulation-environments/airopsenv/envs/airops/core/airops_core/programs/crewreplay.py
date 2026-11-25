"""Crews alone over fixed departures: the fewest flights without a legal crew first, then the least crew cost.

The legs fly at given minutes (JetBlue's recorded day, the record before the start, a plan's departures). Every leg that
needs a crew takes one unit of each kind of its type group or goes without a legal crew, at UNCREWED, counted once per
flight; each unit takes one column: its day as a path of legs it operates or deadheads on, at most two duty periods (the
second after rest, a hotel away from its base), and where it ends, at its marginal crew cost (crews.py): deadheads off
its trip, a night away from base, and the end-of-day position `end` prices. Deadheads carry their people within each
leg's free seats; Table A is a row per pilot unit and period: the period's worst takeoff (the actual blocks so far plus
the leg's scheduled block) over its limit, at most zero.

Pricing is exact for Table B, the short-call cap, the attendants' duty, standby, reports and rests, and relaxes Table A
(each leg counts the lesser of its actual and scheduled block) and the first period's sits, so the Lagrangian bound z_LP
plus the units' least reduced costs is valid at every round: a day whose first period deadheads a full rest after a
release is left out of the columns, as the crew sim starts the second period there (a hotel away from base). It runs per
unit as one pass over the legs in departure order: every arrival comes after its departure, so the best way to be at a
station by a departure is known when that departure comes up. A short-call reserve reports, and a second duty period
reports, on the last whole hour that leaves BRIEF before its first departure (a world rule the sim follows too).
Identical units are priced once.

The column pool starts with each unit's day following its trip as far as it legally can (priced like any other). Column
generation to the LP optimum (or half the time limit), then a dive: fix the columns the LP has above SURE (the largest
if none, or a few of the largest), price again the units not yet fixed, repeat until the LP is integral, which is a
feasible integer plan; then the integer program over every column generated, started from the dive's plan or the
trip-following one, whichever is cheaper (every unit idle if neither is a plan). The LP and the Lagrangian bound give
how far from the fewest uncrewed flights the plan can be.

A unit may have other places to start from (`starts`, each with a preference added to its day's cost, as the morning
replay prefers real layovers). Each unit's day is priced with its end of the day against its own trip's night station;
the plan's crew cost then takes the end of the day per base (crews.Ends), which costs no more.
"""

import bisect
import heapq
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace

import highspy
import numpy as np

from . import crews
from .crews import Unit

INF = highspy.kHighsInf
UNCREWED = 1_000_000
EPS = 1e-6
ENDS = 24
SURE = 0.5
GREEDY = 10
OWN = 1e5
OP, DH = 0, 1


@dataclass(frozen=True)
class Leg:
    """A leg at a fixed departure: it arrives at `arr` at `dest`, has `free` seats, and `need`s a crew of these kinds
    (a leg flown by crews the task does not model only carries deadheads), going without one at `price`."""

    id: int
    flight: str
    orig: str
    dest: str
    dep: int
    arr: int
    sched_block: int
    group: str
    free: int
    need: tuple[str, ...] = crews.KINDS
    price: float = UNCREWED

    @property
    def block(self) -> int:
        return self.arr - self.dep


@dataclass(frozen=True)
class Column:
    """A unit's day: its first duty period's steps (leg, OP or DH), a second period's after rest at `rest` (station,
    report), a short-call reserve's `report`, where it ends, its crew cost and its Table A excess per period, and where
    it starts when not at its own station."""

    unit: str
    first: tuple[tuple[int, int], ...] = ()
    second: tuple[tuple[int, int], ...] = ()
    report: int | None = None
    rest: tuple[str, int] | None = None
    end: str = ""
    cost: float = 0.0
    over: tuple[float, float] = (0.0, 0.0)
    start: str = ""

    @property
    def steps(self) -> tuple[tuple[int, int], ...]:
        return self.first + self.second

    @property
    def ops(self) -> tuple[int, ...]:
        return tuple(i for i, how in self.steps if how == OP)

    @property
    def dhs(self) -> tuple[int, ...]:
        return tuple(i for i, how in self.steps if how == DH)

    @property
    def key(self) -> tuple:
        return self.unit, self.first, self.second, self.report, self.rest, self.end, self.start


EndCost = Callable[[Unit, str], float]


class Problem:
    """Legs at fixed departures, the units, and what a day costs: `end(unit, station)` the cost of ending there,
    `call` the first minute a short-call reserve may be called (its availability period's start when None), `foreign`
    each leg a unit flies off its trip (a preference for the published trips, not a cost of the world), `starts` the
    stations a unit may start from with the preference each adds (its own station alone, at none, when absent), and
    `ends` the end of the day per base the plan's crew cost takes."""

    def __init__(self, legs: Sequence[Leg], units: Sequence[Unit], d: str, local: crews.Local, end: EndCost,
                 call: int | None = None, rests: bool = True, foreign: float = 0.0,
                 starts: Mapping[str, Sequence[tuple[str, float]]] | None = None, ends: crews.Ends | None = None):
        self.legs = sorted(legs, key=lambda g: (g.dep, g.id))
        self.units, self.d, self.local, self.end, self.call, self.rests = list(units), d, local, end, call, rests
        self.foreign, self.starts, self.ends = foreign, dict(starts or {}), ends
        self.unit = {u.id: u for u in units}
        self.leg = {g.id: g for g in self.legs}
        self.pos = {g.id: k for k, g in enumerate(self.legs)}
        self.stations = sorted({g.orig for g in legs} | {g.dest for g in legs} | {u.station for u in units}
                               | {u.home for u in units} | {u.end for u in units}
                               | {st for v in self.starts.values() for st, _ in v})
        self.si = {s: k for k, s in enumerate(self.stations)}
        self.deps = [g.dep for g in self.legs]
        self.dep = np.array(self.deps, dtype=np.int64)
        self.arr = np.array([g.arr for g in self.legs], dtype=np.int64)
        self.sched = np.array([g.sched_block for g in self.legs], dtype=np.int64)
        self.weight = np.minimum(self.arr - self.dep, self.sched).astype(float)
        self.orig = np.array([self.si[g.orig] for g in self.legs], dtype=np.int64)
        self.dest = np.array([self.si[g.dest] for g in self.legs], dtype=np.int64)
        self.need = {k: np.array([k in g.need for g in self.legs], dtype=bool) for k in crews.KINDS}
        self.group = {grp: np.array([g.group == grp for g in self.legs], dtype=bool) for grp in crews.GROUPS}
        self.dh_base = {(k, grp): np.array([crews.deadhead(k, grp, g.block, d) for g in self.legs])
                        for k in crews.KINDS for grp in crews.GROUPS}
        self.hours: dict[tuple[str, int], int] = {}
        self.offsets: dict[tuple[str, int], int] = {}
        self.whole = sorted({(t - crews.BRIEF) // 60 * 60 for t in self.deps})

    def hour(self, station: str, minute: int) -> int:
        k = (station, minute)
        if k not in self.hours:
            self.hours[k] = crews.local_hour(self.local, station, minute)
        return self.hours[k]

    def offset(self, station: str, minute: int) -> int:
        k = (station, minute)
        if k not in self.offsets:
            self.offsets[k] = self.local(station, minute) - minute
        return self.offsets[k]

    def window(self, report: int) -> range:
        """The legs whose last whole hour before them, less BRIEF, is `report`."""
        return range(bisect.bisect_left(self.deps, report + crews.BRIEF),
                     bisect.bisect_left(self.deps, report + crews.BRIEF + 60))

    def first_call(self, u: Unit) -> int:
        return u.rap if self.call is None else max(u.rap, self.call)

    def origins(self, u: Unit) -> list[tuple[str, float]]:
        """Where u may start, and the preference each adds."""
        return list(self.starts.get(u.id, ((u.station, 0.0),)))

    def idles(self, u: Unit) -> list[Column]:
        """u's days doing nothing, from each place it may start, cheapest first."""
        out = [Column(u.id, end=st, cost=self.end(u, st) + pref, start=self.tag(u, st)) for st, pref in self.origins(u)]
        return sorted(out, key=lambda c: (c.cost, c.end))

    def tag(self, u: Unit, station: str) -> str:
        return "" if station == u.station and u.id not in self.starts else station

    def began(self, u: Unit, c: Column) -> str:
        return c.start or u.station

    def preference(self, u: Unit, station: str) -> float:
        return next((pref for st, pref in self.origins(u) if st == station), 0.0)

    def sat_out(self, u: Unit, c: Column) -> bool:
        """Whether c's first period deadheads a full rest after a release, where the crew sim would start the
        second period."""
        release = u.landed + crews.DEBRIEF if u.landed is not None else (
            u.until if u.reserve == "standby" and u.until is not None else
            None if u.report is None else max(u.report, self.call or u.report))
        if c.report is not None:
            release = c.report
        for i, how in c.first:
            g = self.leg[i]
            if how == DH and release is not None and (g.dep - crews.BRIEF) // 60 * 60 >= release + crews.rest(u.kind):
                return True
            release = g.arr + crews.DEBRIEF
        return False

    def crew_cost(self, cols: Mapping[str, Column]) -> float:
        """The plan's crew cost: each day's, its end of the day taken per base when `ends` is given."""
        cost = sum(c.cost for c in cols.values())
        if self.ends is None:
            return cost
        own = sum(self.end(self.unit[uid], c.end) for uid, c in cols.items())
        return cost - own + self.ends.total({uid: c.end for uid, c in cols.items()})

    def priced(self, u: Unit, c: Column) -> Column:
        """c with its crew cost and Table A excess worked out from its steps."""
        planned = set(u.deadheads)
        dh = sum(crews.deadhead(u.kind, u.group, self.leg[i].block, self.d) for i, how in c.steps
                 if how == DH and i not in planned)
        night = crews.hotel(c.rest[0], self.d) * u.size if c.rest and c.rest[0] != u.home else 0.0
        trip = set(u.trip)
        dh += self.foreign * sum(how == OP and i not in trip for i, how in c.steps)
        over = (0.0, 0.0)
        if u.kind == "pilot":
            first = self.excess(c.first, 0 if c.report is not None else u.minutes,
                                self.hour(u.home, c.report) if c.report is not None else u.hour)
            over = (first, self.excess(c.second, 0, self.hour(*c.rest)) if c.rest else 0.0)
        return replace(c, cost=dh + night + self.end(u, c.end) + self.preference(u, self.began(u, c)), over=over)

    def excess(self, steps: Sequence[tuple[int, int]], minutes: int, hour: int | None) -> float:
        """A period's worst takeoff: the flight time so far plus the leg's scheduled block, over Table A (0: none)."""
        worst, flown = None, minutes
        for i, how in steps:
            if how == OP:
                g = self.leg[i]
                worst = max(flown + g.sched_block, -np.inf if worst is None else worst)
                flown += g.block
        return 0.0 if worst is None else float(worst - crews.flight_time(hour))


@dataclass
class Duals:
    pi: dict[str, np.ndarray]
    seat: np.ndarray
    mu: dict[str, float]
    nu: dict[str, tuple[float, float]]


@dataclass(frozen=True)
class Period:
    """A duty period to price: its report, the report's hour, a short-call period's start, and where it may start
    (station, earliest and latest first departure, segment layer, value)."""

    report: int
    hour: int
    rap: int | None
    sources: tuple[tuple[int, int, float, int, float], ...]
    called: bool = False


class Pricer:
    """Each unit's least reduced cost over its legal days, and its best columns."""

    def __init__(self, p: Problem):
        self.p = p
        self.cache: dict[tuple, object] = {}
        self.kinds: dict[str, Unit] = {}

    def kind(self, u: Unit) -> Unit:
        """The unit without its id: identical units share their legality and end costs."""
        if u.id not in self.kinds:
            self.kinds[u.id] = replace(u, id="")
        return self.kinds[u.id]

    def price(self, u: Unit, y: Duals, keep: int = 3) -> tuple[float, list[Column]]:
        p = self.p
        best, found = p.idles(u)[0].cost - y.mu[u.id], []
        for period in self.periods(u):
            rc, cols = self.run(u, period, y, keep)
            best = min(best, rc)
            found += cols
        found.sort(key=lambda rc_c: rc_c[0])
        seen, out = set(), []
        for _, c in found:
            if c.key not in seen and len(out) < keep:
                seen.add(c.key)
                out.append(c)
        return best, out

    def periods(self, u: Unit) -> list[Period]:
        p = self.p
        s0 = p.si[u.station]
        if u.reserve == "short" and u.report is None:
            lo, hi = crews.on_hour(p.first_call(u) + u.lead), u.until + u.lead
            out = []
            for r in p.whole[bisect.bisect_left(p.whole, lo):bisect.bisect_right(p.whole, hi)]:
                if any(p.orig[k] == s0 for k in p.window(r)):
                    out.append(Period(r, p.hour(u.home, r), u.rap, ((s0, r + crews.BRIEF, r + crews.BRIEF + 59, 0,
                                                                      0.0),), True))
            return out
        hi = u.until if u.reserve == "standby" and u.flown == 0 and u.until is not None else np.inf
        layer = min(u.flown, 7) if u.kind == "pilot" else 0
        rap = u.rap if u.reserve == "short" else None
        if u.id not in p.starts:
            return [Period(u.report, u.hour, rap, ((s0, u.ready, hi, layer, 0.0),))]
        return [Period(u.report, p.hour(st, u.report), rap, ((p.si[st], u.ready, hi, layer, pref),))
                for st, pref in p.origins(u)]

    def layers(self, u: Unit) -> int:
        return 8 if u.kind == "pilot" else 2

    def legal(self, u: Unit, report: int, hour: int, rap: int | None, minutes: int, own: bool) -> np.ndarray:
        """ops[k, l]: the unit may operate leg k as its period's l-th segment (attendants: 1, any leg flown)."""
        key = ("legal", self.kind(u), report, hour, rap, minutes, own)
        if key in self.cache:
            return self.cache[key]
        p = self.p
        mask = p.group[u.group] & p.need[u.kind]
        out = np.zeros((len(p.legs), self.layers(u)), dtype=bool)
        if u.kind == "pilot":
            fits = p.sched + minutes <= crews.flight_time(hour)
            for seg in range(1, 8):
                out[:, seg] = mask & fits & (p.dep + p.sched <= crews.pilot_limit(report, hour, seg, rap))
        else:
            trip = set(u.trip) if own else set()
            own_legs = np.array([g.id in trip for g in p.legs], dtype=bool)
            out[:, 1] = mask & (own_legs | (p.dep + p.sched + crews.DEBRIEF - report <= crews.CABIN_DUTY))
        self.cache[key] = out
        return out

    def dh_legal(self, u: Unit, report: int, own: bool = True) -> np.ndarray:
        key = ("dh", self.kind(u), report, own)
        if key not in self.cache:
            p = self.p
            planned = set(u.deadheads) if own else set()
            self.cache[key] = np.ones(len(p.legs), dtype=bool) if u.kind == "pilot" else \
                np.array([g.id in planned for g in p.legs], dtype=bool) | \
                (p.dep + p.sched + crews.DEBRIEF - report <= crews.CABIN_DUTY)
        return self.cache[key]

    def ends(self, u: Unit) -> np.ndarray:
        key = ("ends", self.kind(u))
        if key not in self.cache:
            self.cache[key] = np.array([self.p.end(u, s) for s in self.p.stations])
        return self.cache[key]

    def costs(self, u: Unit, y: Duals, nu: float) -> tuple[np.ndarray, np.ndarray]:
        p = self.p
        op = -y.pi[u.kind] - (nu * p.weight if u.kind == "pilot" else 0.0)
        if p.foreign:
            trip = set(u.trip)
            op = op + np.array([0.0 if g.id in trip else p.foreign for g in p.legs])
        dh = p.dh_base[(u.kind, u.group)].copy()
        for i in u.deadheads:
            if i in p.pos:
                dh[p.pos[i]] = 0.0
        return op, dh - u.size * y.seat

    def run(self, u: Unit, period: Period, y: Duals, keep: int) -> tuple[float, list[tuple[float, Column]]]:
        p = self.p
        L = self.layers(u)
        l0 = period.sources[0][3]
        minutes = 0 if period.called else u.minutes
        nu1, nu2 = y.nu.get(u.id, (0.0, 0.0))
        op_cost, dh_cost = self.costs(u, y, nu1)
        one = walk(p, L, period.sources, self.legal(u, period.report, period.hour, period.rap, minutes, True),
                   op_cost, self.dh_legal(u, period.report), dh_cost)
        const = np.zeros(L)
        if u.kind == "pilot":
            const[l0 + 1:] = -nu1 * (minutes - crews.flight_time(period.hour))
        endc = self.ends(u)
        mu = y.mu[u.id]
        cands = [(v - mu, e, None) for v, e in best_ends(p, one, endc, const)]
        if not period.called:
            cands.append((endc[period.sources[0][0]] + period.sources[0][4] - mu, None, None))
        if p.rests:
            cands += self.second(u, one, const, y, nu2, endc, period)
        if not cands:
            return np.inf, []
        cands.sort(key=lambda c: c[0])
        out = []
        for rc, e, two in cands[:ENDS]:
            if rc > -EPS:
                break
            col = self.trace(u, one, e, two, period.report if period.called else None,
                             p.stations[period.sources[0][0]])
            if col is None:
                continue
            col = p.priced(u, col)
            if p.sat_out(u, col):
                continue
            exact = self.reduced(u, col, y)
            if exact < -EPS:
                out.append((exact, col))
            if len(out) >= keep:
                break
        return cands[0][0], out

    def second(self, u: Unit, one: dict, const: np.ndarray, y: Duals, nu2: float, endc: np.ndarray,
               period: Period) -> list[tuple]:
        """Ends of days with a second duty period: rested from release (a hotel away from base), reporting on the
        last whole hour before its first departure at the station it rested at, flying at least one leg."""
        p = self.p
        rest = crews.rest(u.kind)
        val = one["val"] + const[None, None, :]
        layer = val.argmin(axis=2)
        best = val.min(axis=2)
        released = defaultdict(list)
        for how, k in zip(*np.nonzero(np.isfinite(best)), strict=True):
            released[int(p.dest[k])].append((int(p.arr[k]) + crews.DEBRIEF, float(best[how, k]),
                                             (int(how), int(k), int(layer[how, k]))))
        idle = u.landed + crews.DEBRIEF if u.landed is not None else max(period.report, p.call or period.report)
        if u.reserve == "standby" and u.landed is None and u.until is not None:
            idle = u.until
        if not period.called:
            released[period.sources[0][0]].append((idle, period.sources[0][4], None))
        prefix = {}
        for s, rs in released.items():
            rs.sort(key=lambda r: r[0])
            vals = [r[1] for r in rs]
            arg = np.minimum.accumulate(vals)
            prefix[s] = ([r[0] for r in rs], arg, rs)
        if not prefix:
            return []
        first = min(ts[0] for ts, _, _ in prefix.values()) + rest
        mine = p.group[u.group] & p.need[u.kind]
        out = []
        for r2 in p.whole[bisect.bisect_left(p.whole, first):]:
            if not mine[bisect.bisect_left(p.deps, r2 + crews.BRIEF):].any():
                break
            starts: dict[int, dict[int, tuple]] = defaultdict(dict)
            for k in p.window(r2):
                s = int(p.orig[k])
                if s not in prefix or s in starts.get(p.offset(p.stations[s], r2), {}):
                    continue
                ts, vals, rs = prefix[s]
                j = bisect.bisect_right(ts, r2 - rest) - 1
                if j < 0:
                    continue
                a = min(range(j + 1), key=lambda i: rs[i][1])
                night = crews.hotel(p.stations[s], p.d) * u.size if p.stations[s] != u.home else 0.0
                starts[p.offset(p.stations[s], r2)][s] = (float(vals[j]) + night, rs[a][2])
            for at in starts.values():
                h2 = p.hour(p.stations[next(iter(at))], r2)
                sources = tuple((s, r2 + crews.BRIEF, r2 + crews.BRIEF + 59, 0, v) for s, (v, _) in at.items())
                op_cost, dh_cost = self.costs(u, y, nu2)
                two = walk(p, self.layers(u), sources, self.legal(u, r2, h2, None, 0, False), op_cost,
                           self.dh_legal(u, r2, False), dh_cost)
                const2 = np.zeros(self.layers(u))
                if u.kind == "pilot":
                    const2[1:] = nu2 * crews.flight_time(h2)
                for v, e in best_ends(p, two, endc, const2):
                    out.append((v - y.mu[u.id], e, (two, r2, {s: o for s, (_, o) in at.items()})))
        return out

    def trace(self, u: Unit, one: dict, e: tuple | None, two: tuple | None, report: int | None,
              station: str) -> Column | None:
        p = self.p
        tag = p.tag(u, station)
        if two is None:
            if e is None:
                return Column(u.id, end=station, start=tag)
            steps = back(p, one, *e)
            return None if steps is None else Column(u.id, steps, report=report, end=p.legs[p.pos[steps[-1][0]]].dest,
                                                     start=tag)
        w2, r2, origin = two
        steps2 = back(p, w2, *e)
        if not steps2:
            return None
        s = p.si[p.leg[steps2[0][0]].orig]
        start = origin.get(s)
        steps1 = () if start is None else back(p, one, *start)
        if steps1 is None:
            return None
        return Column(u.id, tuple(steps1), tuple(steps2), report, (p.stations[s], r2), p.leg[steps2[-1][0]].dest,
                      start=tag)

    def own(self, u: Unit) -> Column | None:
        """The unit's day following its trip as far as it legally can: its own legs rewarded, no other leg flown."""
        p = self.p
        trip = set(u.trip)
        pi = {k: np.zeros(len(p.legs)) for k in crews.KINDS}
        pi[u.kind][:] = [OWN if g.id in trip else -OWN * 10 for g in p.legs]
        y = Duals(pi, np.zeros(len(p.legs)), {u.id: 0.0}, {})
        cols = self.price(u, y, 1)[1] if trip & set(p.leg) else []
        return cols[0] if cols and max(cols[0].over) <= 0 and set(cols[0].ops) <= trip else None

    def reduced(self, u: Unit, c: Column, y: Duals) -> float:
        p = self.p
        nu1, nu2 = y.nu.get(u.id, (0.0, 0.0))
        rc = c.cost - y.mu[u.id] - nu1 * c.over[0] - nu2 * c.over[1]
        for i, how in c.steps:
            rc -= y.pi[u.kind][p.pos[i]] if how == OP else u.size * y.seat[p.pos[i]]
        return rc


def walk(p: Problem, L: int, sources: Sequence[tuple], ops: np.ndarray, op_cost: np.ndarray, dh_ok: np.ndarray,
         dh_cost: np.ndarray) -> dict:
    """One pass over the legs in departure order: the best value after operating or deadheading each leg, at each
    layer, and where each came from (2k + how for leg k, -2 - s for source s)."""
    n = len(p.legs)
    val = np.full((2, n, L), np.inf)
    pred = np.full((2, n, L), -1, dtype=np.int64)
    src = np.zeros((n, L), dtype=np.int64)
    best_v: dict[int, np.ndarray] = {}
    best_p: dict[int, np.ndarray] = {}
    heaps: dict[int, list] = defaultdict(list)
    at: dict[int, list] = defaultdict(list)
    for s, (st, lo, hi, layer, v) in enumerate(sources):
        at[st].append((lo, hi, layer, v, -2 - s))
    any_op = ops.any(axis=1)
    below = np.arange(L) - 1
    for k in range(bisect.bisect_left(p.deps, min((s[1] for s in sources), default=0)), n):
        s, t = p.orig[k], p.deps[k]
        v, pr = best_v.get(s), best_p.get(s)
        h = heaps.get(s)
        if h and h[0][0] <= t:
            while h and h[0][0] <= t:
                _, code = heapq.heappop(h)
                j, how = divmod(code, 2)
                w = val[how, j]
                if v is None:
                    v, pr = w.copy(), np.where(np.isfinite(w), code, -1)
                else:
                    better = w < v
                    if better.any():
                        v, pr = np.where(better, w, v), np.where(better, code, pr)
            best_v[s], best_p[s] = v, pr
        for lo, hi, layer, sv, code in at.get(s, ()):
            if lo <= t <= hi and (v is None or sv < v[layer]):
                v = np.full(L, np.inf) if v is None else v.copy()
                pr = np.full(L, -1, dtype=np.int64) if pr is None else pr.copy()
                v[layer], pr[layer] = sv, code
        if v is None or not np.isfinite(v).any():
            continue
        ta = p.arr[k] + crews.CONNECT
        if dh_ok[k]:
            val[DH, k], pred[DH, k] = v + dh_cost[k], pr
            heapq.heappush(heaps[p.dest[k]], (ta, 2 * k + DH))
        if any_op[k]:
            sv, sp, sl = np.empty(L), np.empty(L, dtype=np.int64), below.copy()
            sv[0], sp[0], sv[1:], sp[1:] = np.inf, -1, v[:-1], pr[:-1]
            if v[L - 1] < sv[L - 1]:
                sv[L - 1], sp[L - 1], sl[L - 1] = v[L - 1], pr[L - 1], L - 1
            val[OP, k] = np.where(ops[k], sv + op_cost[k], np.inf)
            pred[OP, k], src[k] = sp, sl
            if np.isfinite(val[OP, k]).any():
                heapq.heappush(heaps[p.dest[k]], (ta, 2 * k + OP))
    return {"val": val, "pred": pred, "src": src}


def best_ends(p: Problem, w: dict, endc: np.ndarray, const: np.ndarray) -> list[tuple[float, tuple]]:
    """The best ways to end a pass: (value, (how, leg position, layer)), best first."""
    total = w["val"] + endc[p.dest][None, :, None] + const[None, None, :]
    flat = total.ravel()
    k = min(ENDS, int(np.isfinite(flat).sum()))
    if not k:
        return []
    idx = np.argpartition(flat, k - 1)[:k]
    idx = idx[np.argsort(flat[idx], kind="stable")]
    return [(float(flat[i]), tuple(int(x) for x in np.unravel_index(i, total.shape))) for i in idx]


def back(p: Problem, w: dict, how: int, k: int, layer: int) -> tuple[tuple[int, int], ...] | None:
    """The steps into (how, leg position k, layer), walked back to a source: (leg id, how)."""
    out = []
    while True:
        out.append((p.legs[k].id, how))
        code = int(w["pred"][how, k, layer])
        if how == OP:
            layer = int(w["src"][k, layer])
        if code < -1:
            break
        if code == -1:
            return None
        k, how = divmod(code, 2)
    return tuple(reversed(out))


class Master:
    """Per needed leg and kind, the units operating it plus its slack equal one, the slack at most the leg's uncrewed
    variable (at the leg's price); per unit one column; per pilot unit and period, Table A's excess at most zero; per
    leg its deadheading people within its free seats."""

    def __init__(self, p: Problem):
        self.p = p
        self.h = h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("threads", 1)
        need = [k for k, g in enumerate(p.legs) if g.need]
        self.cover = {key: n for n, key in enumerate((kind, k) for kind in crews.KINDS for k in need
                                                       if kind in p.legs[k].need)}
        nc = len(self.cover)
        self.urow = {u.id: 2 * nc + j for j, u in enumerate(p.units)}
        r = 2 * nc + len(p.units)
        self.arow = {}
        for u in p.units:
            if u.kind == "pilot":
                self.arow[u.id] = (r, r + 1)
                r += 2
        a = len(self.arow) * 2
        lo = np.array([1.0] * nc + [-INF] * nc + [1.0] * len(p.units) + [-INF] * a)
        hi = np.array([1.0] * nc + [0.0] * nc + [1.0] * len(p.units) + [0.0] * a)
        h.addRows(r, lo, hi, 0, np.zeros(r, dtype=np.int32), np.array([], dtype=np.int32), np.array([]))
        for n in self.cover.values():
            h.addCol(0.0, 0.0, 1.0, 2, np.array([n, nc + n], dtype=np.int32), np.array([1.0, 1.0]))
        self.z = {}
        for k in need:
            rows = [nc + self.cover[(kind, k)] for kind in p.legs[k].need]
            self.z[k] = h.getNumCol()
            h.addCol(float(p.legs[k].price), 0.0, 1.0, len(rows), np.array(rows, dtype=np.int32),
                     -np.ones(len(rows)))
        self.first = h.getNumCol()
        self.seat_row: dict[int, int] = {}
        self.columns: list[Column] = []
        self.seen: set[tuple] = set()
        self.idle: dict[str, int] = {}
        for u in p.units:
            for c in p.idles(u):
                self.add(c)
                self.idle.setdefault(u.id, len(self.columns) - 1)

    def add(self, c: Column) -> bool:
        if c.key in self.seen:
            return False
        self.seen.add(c.key)
        p = self.p
        u = p.unit[c.unit]
        rows = {self.urow[u.id]: 1.0}
        for r, v in zip(self.arow.get(u.id, ()), c.over, strict=False):
            if v:
                rows[r] = float(v)
        for i, how in c.steps:
            k = p.pos[i]
            if how == OP:
                rows[self.cover[(u.kind, k)]] = 1.0
                continue
            if k not in self.seat_row:
                self.seat_row[k] = self.h.getNumRow()
                self.h.addRow(-INF, float(p.legs[k].free), 0, np.array([], dtype=np.int32), np.array([]))
            rows[self.seat_row[k]] = rows.get(self.seat_row[k], 0.0) + u.size
        self.h.addCol(float(c.cost), 0.0, INF, len(rows), np.array(list(rows), dtype=np.int32),
                      np.array(list(rows.values())))
        self.columns.append(c)
        return True

    def solve(self) -> tuple[str, float]:
        self.h.run()
        return self.h.modelStatusToString(self.h.getModelStatus()), self.h.getInfo().objective_function_value

    def duals(self) -> Duals:
        p = self.p
        y = np.asarray(self.h.getSolution().row_dual)
        pi = {kind: np.zeros(len(p.legs)) for kind in crews.KINDS}
        for (kind, k), r in self.cover.items():
            pi[kind][k] = y[r]
        seat = np.zeros(len(p.legs))
        for k, r in self.seat_row.items():
            seat[k] = y[r]
        return Duals(pi, seat, {u: float(y[r]) for u, r in self.urow.items()},
                     {u: (float(y[a]), float(y[b])) for u, (a, b) in self.arow.items()})

    def x(self) -> np.ndarray:
        return np.asarray(self.h.getSolution().col_value)

    def plan(self, chosen: Mapping[str, int]) -> np.ndarray | None:
        """The solution taking column `chosen[unit]` for each unit (idle otherwise), or None if it breaks a row."""
        p = self.p
        x = np.zeros(self.h.getNumCol())
        kinds = {}
        for u in p.units:
            c = chosen.get(u.id, self.idle[u.id])
            x[self.first + c] = 1.0
            for i in self.columns[c].ops:
                kinds.setdefault(p.pos[i], set()).add(u.kind)
        for (kind, k), n in self.cover.items():
            x[n] = 0.0 if kind in kinds.get(k, ()) else 1.0
        for k, c in self.z.items():
            x[c] = float(any(kind not in kinds.get(k, ()) for kind in p.legs[k].need))
        lp = self.h.getLp()
        a = lp.a_matrix_
        start, index, value = np.asarray(a.start_), np.asarray(a.index_), np.asarray(a.value_)
        act = np.zeros(self.h.getNumRow())
        for j in np.nonzero(x)[0]:
            act[index[start[j]:start[j + 1]]] += value[start[j]:start[j + 1]] * x[j]
        ok = np.all(act >= np.asarray(lp.row_lower_) - 1e-6) and np.all(act <= np.asarray(lp.row_upper_) + 1e-6)
        return x if ok else None

    def trivial(self) -> np.ndarray:
        """Every unit idle and every needed leg uncrewed."""
        x = np.zeros(self.h.getNumCol())
        x[:len(self.cover)] = 1.0
        x[list(self.z.values())] = 1.0
        x[[self.first + j for j in self.idle.values()]] = 1.0
        return x


@dataclass(frozen=True)
class Solved:
    """The integer plan (a column per unit), the legs it leaves without a legal crew, its objective (each uncrewed
    leg's price plus each day's cost) with the LP and Lagrangian bounds on it, and its crew cost (Problem.crew_cost)."""

    columns: dict[str, Column]
    uncrewed: tuple[int, ...]
    objective: float
    crew_cost: float
    lp: float
    bound: float
    converged: bool
    rounds: int
    dives: int
    generated: int
    seconds: float
    dive_integral: bool
    polished: bool
    pool: tuple[Column, ...] = field(default=(), repr=False)

    def crews(self, p: Problem) -> dict[int, dict[str, str]]:
        out: dict[int, dict[str, str]] = defaultdict(dict)
        for c in self.columns.values():
            for i in c.ops:
                out[i][p.unit[c.unit].kind] = c.unit
        return dict(out)

    def summary(self) -> dict:
        return {"uncrewed": len(self.uncrewed), "objective": round(self.objective, 1),
                "crew_cost": round(self.crew_cost, 1), "lp": round(self.lp, 1), "bound": round(self.bound, 1),
                "uncrewed_lp": round(self.lp / UNCREWED, 3), "uncrewed_bound": round(self.bound / UNCREWED, 3),
                "converged": self.converged, "rounds": self.rounds, "dives": self.dives,
                "columns": self.generated, "seconds": self.seconds, "dive_integral": self.dive_integral,
                "polished": self.polished}


def solve(p: Problem, time_limit: float = 1800.0, rounds: int = 1000, keep: int = 3, polish: float = 300.0,
          starts: Sequence[Mapping[str, Column]] = (), pool: Sequence[Column] = (), greedy: int = GREEDY,
          log: Callable[[str], None] | None = None) -> Solved:
    """`starts` are plans to start from too, a column per unit that moves (crewday.columns_of); `pool` columns go in
    the master from the first round. Each step of theirs must be a leg of `p`. The dive fixes with the columns at SURE
    or more the largest others up to `greedy` in all, of different units flying different legs."""
    t0 = time.monotonic()
    master = Master(p)
    pricer = Pricer(p)
    classes = defaultdict(list)
    for u in p.units:
        classes[replace(u, id="")].append(u)
    fixed: set[int] = set()
    bound = -np.inf
    count = 0
    seeds = {}
    for members in classes.values():
        c = pricer.own(members[0])
        for u in members if c is not None else ():
            if master.add(replace(c, unit=u.id)):
                seeds[u.id] = len(master.columns) - 1
    for c in pool:
        master.add(p.priced(p.unit[c.unit], c))
    index = {c.key: j for j, c in enumerate(master.columns)}
    given = []
    for plan in starts:
        chosen = {}
        for uid, c in plan.items():
            c = p.priced(p.unit[uid], c)
            if master.add(c):
                index[c.key] = len(master.columns) - 1
            chosen[uid] = index[c.key]
        given.append(chosen)

    def generate(limit: int, deadline: float) -> tuple[float, bool]:
        nonlocal bound, count
        z = np.nan
        for _ in range(limit):
            status, z = master.solve()
            if status != "Optimal":
                raise RuntimeError(f"crew master {status}")
            y = master.duals()
            total, added = 0.0, 0
            done = {master.columns[j].unit for j in fixed}
            for members in classes.values():
                members = [u for u in members if u.id not in done]
                if not members:
                    continue
                rc, cols = pricer.price(members[0], y, keep)
                for u in members:
                    total += min(0.0, rc + y.mu[members[0].id] - y.mu[u.id])
                    for c in cols:
                        c = replace(c, unit=u.id)
                        if pricer.reduced(u, c, y) < -EPS:
                            added += master.add(c)
            count += 1
            if not fixed:
                bound = max(bound, z + total)
            if log and not fixed and count % 5 == 0:
                log(f"round {count} z {z:.0f} bound {bound:.0f} columns {len(master.columns)} "
                    f"{time.monotonic() - t0:.0f}s")
            if not added:
                return z, True
            if time.monotonic() > deadline:
                break
        return z, False

    lp, converged = generate(rounds, t0 + time_limit / 2)
    if converged:
        bound = max(bound, lp)
    root_rounds, dives = count, 0
    while time.monotonic() - t0 < time_limit:
        x = master.x()[master.first:]
        frac = [(x[j], j) for j in range(len(master.columns))
                if EPS < x[j] < 1 - EPS and j not in fixed and max(master.columns[j].over) <= 0]
        if not frac:
            break
        top = max(frac)[1]
        for batch in (apart(master, frac, greedy), [top]):
            for j in batch:
                master.h.changeColBounds(master.first + j, 1.0, INF)
            if master.solve()[0] == "Optimal":
                fixed.update(batch)
                break
            for j in batch:
                master.h.changeColBounds(master.first + j, 0.0, INF)
        else:
            master.h.changeColBounds(master.first + top, 0.0, 0.0)
            master.solve()
        dives += 1
        generate(5, t0 + time_limit)
        if log and dives % 10 == 0:
            log(f"dive {dives} fixed {len(fixed)} columns {len(master.columns)} {time.monotonic() - t0:.0f}s")
    x = master.x()
    tail = np.concatenate([x[list(master.z.values())], x[master.first:]])
    integral = bool(np.all(np.abs(tail - np.round(tail)) <= 1e-6))
    cost = np.asarray(master.h.getLp().col_cost_)
    begin = [np.round(x)] if integral else [master.trivial()]
    for chosen in (seeds, *given):
        if chosen and (plan := master.plan(chosen)) is not None:
            begin.append(plan)
    start = min(begin, key=lambda v: float(cost @ v))
    for j in fixed:
        master.h.changeColBounds(master.first + j, 0.0, INF)
    chosen, polished = integer(master, start, polish)
    cols = {master.columns[j].unit: master.columns[j] for j in chosen}
    crewed = defaultdict(set)
    for c in cols.values():
        for i in c.ops:
            crewed[i].add(p.unit[c.unit].kind)
    uncrewed = tuple(sorted(g.id for g in p.legs if not set(g.need) <= crewed[g.id]))
    cost = sum(c.cost for c in cols.values())
    return Solved(cols, uncrewed, sum(p.leg[i].price for i in uncrewed) + cost, p.crew_cost(cols), lp, bound, converged,
                  root_rounds, dives, len(master.columns), round(time.monotonic() - t0, 1), integral, polished,
                  tuple(master.columns))


def apart(master: Master, frac: Sequence[tuple[float, int]], n: int) -> list[int]:
    """The fractional columns at SURE or more, else the largest; then more of the largest up to n in all, no two of
    one unit or flying one leg."""
    out, units, legs = [], set(), set()
    for v, j in sorted(frac, reverse=True):
        c = master.columns[j]
        if out and ((v < SURE and len(out) >= n) or c.unit in units or legs & set(c.ops)):
            continue
        out.append(j)
        units.add(c.unit)
        legs |= set(c.ops)
    return out


def integer(master: Master, start: np.ndarray, time_limit: float) -> tuple[list[int], bool]:
    """The integer program over every column from `start` (a feasible plan): the chosen columns, and whether it
    improved on the start."""
    h = master.h
    n = h.getNumCol()
    cost = np.asarray(h.getLp().col_cost_)
    ints = np.array(list(master.z.values()) + list(range(master.first, n)), dtype=np.int32)
    h.changeColsIntegrality(len(ints), ints, np.array([highspy.HighsVarType.kInteger] * len(ints)))
    h.setOptionValue("time_limit", float(max(time_limit, 1.0)))
    h.setOptionValue("mip_rel_gap", 0.0)
    h.setSolution(n, np.arange(n, dtype=np.int32), start)
    h.run()
    x = start
    if h.getInfo().primal_solution_status == 2:
        y = np.asarray(h.getSolution().col_value)
        if float(cost @ y) < float(cost @ start) - 1e-6:
            x = y
    return [j for j in range(len(master.columns)) if x[master.first + j] > 0.5], x is not start
