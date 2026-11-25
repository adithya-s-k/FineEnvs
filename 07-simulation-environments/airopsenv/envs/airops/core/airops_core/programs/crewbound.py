"""Lower bounds on the best day over aircraft and crews. Every one holds for every day the crew sim can run, whatever
the plan: each relaxes the crew rules (crewsim.py) to a superset of the days a unit can have, and crew costs are never
below zero. The relaxed rules: Table B read for the fewest
segments the unit could have flown in the period, Table A on each leg alone against the period's flight time so far, a
short-call reserve reporting at any whole hour it could (its limits at the latest), the second duty period after rest
reporting at any minute and with no limit, a deadhead in the first period at any time without a night (the sim starts
the second period, a hotel, after a full rest), seats free.

- `reach`: per flight, the departure minutes at which some unit of each kind could operate it, every unit on its own
  (an earliest-arrival search per unit over every departure minute of every flight). A day flies each flight at one
  of these.
- A*: the aircraft optimum (its MIP's dual bound). The restricted A*: the aircraft optimum with every flight kept to
  its minutes in `reach`, every flight with none cancelled.
- L(0): an aircraft bound plus each unit's least crew cost on its own (`least`, `Board` at no duals), its end of the
  day at its least share of its base's (crews.Ends.least: its cheapest move to any of the base's trips tonight).
- The covering bound: for any day, its aircraft cost is at least the aircraft LP's optimum z plus the reduced cost d+ of
  each flight's choice (its departure minute, its cancellation; at least zero, and a choice the LP has at its upper
  bound with d < 0 counted on the other choices), the other columns' terms being at least zero. So the day costs at
  least z + the least, over flight choices and crew days covering every flown flight with a unit of each kind, of the
  choices' d+ and the crew cost: a covering problem where the two kinds may choose different minutes and each pays half
  a minute's price, and the end of the day per base a transport from where the days end to the base's trips tonight (as
  the sim prices it), solved by column generation over `Board`'s relaxed days, priced at duals smoothed toward those of
  the best bound so far (at the LP's own when that finds no column), its Lagrangian bound valid at any duals (each
  base's trips tonight taken at their cheapest station at the stations' duals). With a plan of cost UB, a minute or a
  cancellation priced over UB - z can't be in a day cheaper than UB and is left out (reduced-cost fixing); the bound is
  then min(UB, z + the covering bound).

`Board`'s relaxed days run over buckets of each flight's minutes, at most a bucket's width apart, the width under the
flight's block plus CONNECT: a unit takes a bucket ready at its origin by its last minute, lands as if it left at its
first, is judged legal at its first, pays the bucket's least price, so every real day is one of its days at no more
cost.
"""

import bisect
import heapq
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import highspy
import numpy as np

from . import crews, crewsim, mip
from .crews import Unit
from .model import Task

INF = np.inf
WIDTH = 30
BATCH = 128
EPS = 1e-6
TOL = 1.0
SMOOTH = 0.5


def release(u: Unit, start: int) -> int | None:
    """When the unit's duty period ends if it flies no more (crewsim.State.release); None for an uncalled reserve."""
    if u.landed is not None:
        return u.landed + crews.DEBRIEF
    if u.reserve == "standby" and u.until is not None:
        return u.until
    return None if u.report is None else max(u.report, start)


def reports(u: Unit, start: int) -> list[int]:
    """The whole hours an uncalled short-call reserve could report at."""
    lo = crews.on_hour(max(u.rap, start) + u.lead)
    return list(range(lo, (u.until + u.lead) // 60 * 60 + 1, 60))


def limits(u: Unit, local: crews.Local, start: int) -> tuple[int, np.ndarray, float, float]:
    """The unit's earliest departure, its latest block-in expected at takeoff per segment count (pilots; index n for
    the n-th segment of the period, 1-8), its cabin release limit and its pilots' flight time left in the period
    (Table A), at the most a report it could have allows."""
    if u.reserve == "short" and u.report is None:
        rs = reports(u, start)
        if not rs:
            return 10 ** 9, np.full(9, -INF), -INF, -INF
        hours = [crews.local_hour(local, u.home, r) for r in rs]
        pilot = np.array([-INF] + [max(crews.pilot_limit(r, h, n, u.rap) for r, h in zip(rs, hours, strict=True))
                                   for n in range(1, 9)])
        return rs[0] + crews.BRIEF, pilot, rs[-1] + crews.CABIN_DUTY - crews.DEBRIEF, max(map(crews.flight_time, hours))
    rap = u.rap if u.reserve == "short" else None
    pilot = np.array([-INF] + [crews.pilot_limit(u.report, u.hour, n, rap) for n in range(1, 9)])
    return u.ready, pilot, u.report + crews.CABIN_DUTY - crews.DEBRIEF, crews.flight_time(u.hour) - u.minutes


@dataclass(frozen=True)
class Flights:
    """The task's flights as arrays: origin and destination stations, scheduled and world blocks, type group, and the
    departure minutes the aircraft MIP has for each (`minutes`)."""

    ids: np.ndarray
    orig: np.ndarray
    dest: np.ndarray
    sched: np.ndarray
    block: np.ndarray
    group: np.ndarray
    minutes: dict[int, np.ndarray]
    stations: tuple[str, ...]

    @classmethod
    def of(cls, task: Task, block: Mapping[int, int], copies: Mapping[int, Sequence]) -> "Flights":
        stations = tuple(sorted({f.orig for f in task.flights} | {f.dest for f in task.flights}))
        si = {s: k for k, s in enumerate(stations)}
        fs = task.flights
        return cls(np.array([f.id for f in fs]), np.array([si[f.orig] for f in fs]), np.array([si[f.dest] for f in fs]),
                   np.array([f.sched_block for f in fs]), np.array([block[f.id] for f in fs]),
                   np.array([crews.GROUPS.index(crews.TYPE[f.family]) for f in fs]),
                   {f.id: np.array(sorted(d if isinstance(d, int | np.integer) else d[0] for d in copies[f.id]),
                                   dtype=np.int64) for f in fs}, stations)

    @property
    def index(self) -> dict[str, int]:
        return {s: k for k, s in enumerate(self.stations)}


def reach(task: Task, fl: Flights, units: Sequence[Unit], local: crews.Local) -> dict[int, np.ndarray]:
    """Per flight, the minutes of `fl.minutes` at which some pilot unit and some cabin unit could each operate it."""
    out = {}
    windows = {k: defaultdict(list) for k in crews.KINDS}
    for u in units:
        for i, lo, hi in unit_windows(task, fl, u, local):
            windows[u.kind][i].append((lo, hi))
    for i in fl.ids:
        ds = fl.minutes[int(i)]
        ok = np.ones(len(ds), dtype=bool)
        for kind in crews.KINDS:
            mask = np.zeros(len(ds), dtype=bool)
            for lo, hi in windows[kind].get(int(i), ()):
                mask |= (ds >= lo) & (ds <= hi)
            ok &= mask
        out[int(i)] = ds[ok]
    return out


def unit_windows(task: Task, fl: Flights, u: Unit, local: crews.Local) -> list[tuple[int, float, float]]:
    """(flight, first, last minute) at which unit u alone could operate a flight of its group, relaxed as `reach`
    says."""
    si = fl.index
    t0, pilot, cabin, room = limits(u, local, task.start)
    if t0 >= 10 ** 9:
        return []
    own, planned = set(u.trip), set(u.deadheads)
    g = crews.GROUPS.index(u.group)
    s0 = si.get(u.station)
    first = min(u.flown + 1, 8)
    until = u.until if u.reserve == "standby" and u.landed is None and u.until is not None else INF

    def carries(k: int) -> float:
        i = int(fl.ids[k])
        if u.kind == "pilot" or i in planned or i in own:
            return INF
        return cabin - fl.sched[k]

    period1 = dijkstra(fl, {s0: t0} if s0 is not None else {}, carries, until if s0 is not None else INF, s0)
    rested: dict[int, float] = {}
    rest = crews.rest(u.kind)
    for s, t in period1.items():
        rested[s] = min(rested.get(s, INF), t - crews.CONNECT + crews.DEBRIEF + rest + crews.BRIEF)
    idle = release(u, task.start)
    if idle is not None and s0 is not None:
        rested[s0] = min(rested.get(s0, INF), idle + rest + crews.BRIEF)
    period2 = dijkstra(fl, rested, lambda k: INF, INF, None)
    for s, t in rested.items():
        period2[s] = min(period2.get(s, INF), t)
    out = []
    for k in np.nonzero(fl.group == g)[0]:
        i, o = int(fl.ids[k]), int(fl.orig[k])
        if u.kind == "pilot":
            last = pilot[first] - fl.sched[k] if fl.sched[k] <= room else -INF
        else:
            last = INF if i in own else cabin - fl.sched[k]
        if o == s0 and t0 <= min(until, last):
            out.append((i, t0, min(until, last)))
        if o in period1 and period1[o] <= last:
            out.append((i, period1[o], last))
        if o in period2:
            out.append((i, period2[o], INF))
    return out


def dijkstra(fl: Flights, sources: Mapping[int, float], carries: Callable[[int], float], until: float,
             start: int | None) -> dict[int, float]:
    """The earliest minute a unit is ready at each station it can reach by flying or riding flights at any of their
    minutes from `sources` (station: ready); a flight leaving `start` as the first move must leave by `until`, and
    flight k must leave by carries(k). A source counts as reached only through a flight."""
    by_orig = defaultdict(list)
    for k in range(len(fl.ids)):
        by_orig[int(fl.orig[k])].append(k)
    best: dict[int, float] = {}
    heap = []
    for s, t in sources.items():
        for k in by_orig.get(s, ()):
            hi = min(carries(k), until if s == start else INF)
            ds = fl.minutes[int(fl.ids[k])]
            j = bisect.bisect_left(ds, t)
            if j < len(ds) and ds[j] <= hi:
                heapq.heappush(heap, (int(ds[j]) + int(fl.block[k]) + crews.CONNECT, int(fl.dest[k])))
    while heap:
        t, s = heapq.heappop(heap)
        if s in best:
            continue
        best[s] = t
        for k in by_orig.get(s, ()):
            ds = fl.minutes[int(fl.ids[k])]
            j = bisect.bisect_left(ds, t)
            if j < len(ds) and ds[j] <= carries(k):
                a = int(ds[j]) + int(fl.block[k]) + crews.CONNECT
                if int(fl.dest[k]) not in best:
                    heapq.heappush(heap, (a, int(fl.dest[k])))
    return best


@dataclass(frozen=True)
class Legs:
    """Buckets of flights' minutes as legs, by last minute: a unit ready at the origin by `hi` takes it, judged at
    `lo`, landing at lo plus the flight's block; `price` its least minute's price; `flight` the flight's position in
    Flights."""

    flight: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    price: np.ndarray

    @classmethod
    def of(cls, fl: Flights, minutes: Mapping[int, np.ndarray], prices: Mapping[int, np.ndarray] | None = None,
           width: int = WIDTH) -> "Legs":
        rows = []
        for k, i in enumerate(fl.ids):
            ds = minutes.get(int(i))
            if ds is None or not len(ds):
                continue
            p = prices[int(i)] if prices is not None else np.zeros(len(ds))
            span = max(1, min(width, int(fl.block[k]) + crews.CONNECT))
            cell = (ds - ds[0]) // span
            for c in np.unique(cell):
                m = cell == c
                rows.append((int(ds[m][-1]), k, int(ds[m][0]), float(p[m].min())))
        rows.sort()
        a = np.array(rows, dtype=float).reshape(-1, 4)
        return cls(a[:, 1].astype(np.int64), a[:, 2].astype(np.int64), a[:, 0].astype(np.int64), a[:, 3])

    def __len__(self) -> int:
        return len(self.flight)


@dataclass(frozen=True)
class Day:
    """A relaxed day of a unit: its steps (leg, OP or DH) in the first period and after rest, where it rested, where
    it ended, its crew cost and the flights it operates."""

    unit: str
    first: tuple[tuple[int, int], ...]
    second: tuple[tuple[int, int], ...]
    rest: int | None
    end: int
    cost: float
    ops: tuple[int, ...]


OP, DH = 0, 1


class Board:
    """The relaxed days of a set of units over `Legs`, priced per kind by `gain` (each leg's dual less its price) in
    one pass over the legs per batch of units; with `trace`, each unit's best day is returned."""

    def __init__(self, task: Task, fl: Flights, legs: Legs, units: Sequence[Unit], local: crews.Local,
                 end: Callable[[Unit, str], float]):
        self.task, self.fl, self.legs, self.units = task, fl, legs, list(units)
        self.local, self.end = local, end
        d = task.date
        self.S = len(fl.stations)
        si = fl.index
        self.start = []
        for u in self.units:
            t0, pilot, cabin, room = limits(u, local, task.start)
            idle = release(u, task.start)
            until = u.until if u.reserve == "standby" and u.landed is None and u.until is not None else INF
            self.start.append((si.get(u.station, -1), t0, pilot, cabin, idle, until, room))
        self.endc = np.array([[end(u, s) for s in fl.stations] for u in self.units])
        self.idle = np.array([end(u, u.station) for u in self.units])
        self.hotel = np.array([[crews.hotel(s, d) * u.size if s != u.home else 0.0 for s in fl.stations]
                               for u in self.units])
        self.rate = np.array([crews.rate(u.kind, u.group) for u in self.units])
        self.seats = np.array([crews.seat(d) * u.size for u in self.units])
        self.group = np.array([crews.GROUPS.index(u.group) for u in self.units])
        self.lflight = legs.flight
        self.lorig = fl.orig[legs.flight]
        self.ldest = fl.dest[legs.flight]
        self.larr = legs.lo + fl.block[legs.flight]
        self.lsched = fl.sched[legs.flight]
        self.lblock = fl.block[legs.flight]
        self.lgroup = fl.group[legs.flight]
        self.fpos = {int(i): k for k, i in enumerate(fl.ids)}
        self.pos = {u.id: b for b, u in enumerate(self.units)}
        self.buckets: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
        for k in range(len(legs)):
            self.buckets[int(fl.ids[legs.flight[k]])].append((int(legs.lo[k]), int(legs.hi[k]), k))
        self.own = defaultdict(list)
        self.planned = defaultdict(list)
        for b, u in enumerate(self.units):
            for i in u.trip:
                if i in self.fpos:
                    self.own[self.fpos[i]].append(b)
            for i in u.deadheads:
                if i in self.fpos:
                    self.planned[self.fpos[i]].append(b)

    def layers(self, kind: str) -> int:
        return 9 if kind == "pilot" else 2

    def price(self, gain: Mapping[str, np.ndarray], trace: bool = False, members: Sequence[int] | None = None,
              ends: tuple[np.ndarray, np.ndarray] | None = None) -> tuple[np.ndarray, list[Day | None]]:
        """Each unit's least relaxed day value (crew cost less the gains of the legs it operates), and with `trace`
        its day; `members` limits it to those units (the others' values are left at inf); `ends` values each unit's
        end at each station and at its own (instead of its end of the day)."""
        idx = list(range(len(self.units))) if members is None else list(members)
        value = np.full(len(self.units), INF)
        days: list[Day | None] = [None] * len(self.units)
        for kind in crews.KINDS:
            mine = [b for b in idx if self.units[b].kind == kind]
            for n in range(0, len(mine), BATCH):
                batch = mine[n:n + BATCH]
                v, ds = self.walk(kind, np.array(batch), gain[kind], trace, ends)
                value[batch] = v
                for b, day in zip(batch, ds, strict=True):
                    days[b] = day
        return value, days

    def walk(self, kind: str, batch: np.ndarray, gain: np.ndarray, trace: bool,
             ends: tuple[np.ndarray, np.ndarray] | None = None) -> tuple[np.ndarray, list]:
        L = self.layers(kind)
        R = L - 1
        B = len(batch)
        n = len(self.legs)
        units = [self.units[b] for b in batch]
        rest = crews.rest(kind)
        pilot = kind == "pilot"
        lim = np.full((B, L), INF)
        cab = np.full(B, INF)
        room = np.full(B, INF)
        until = np.empty(B)
        t0 = np.empty(B)
        l0 = np.zeros(B, dtype=np.int64)
        s0 = np.empty(B, dtype=np.int64)
        for j, b in enumerate(batch):
            s, t, plim, clim, _, unt, rm = self.start[b]
            s0[j], t0[j], until[j] = s, t, unt
            if pilot:
                lim[j, :8] = plim[1:9]
                room[j] = rm
                l0[j] = min(units[j].flown, 7)
            else:
                cab[j] = clim
        endc = (self.endc if ends is None else ends[0])[batch]
        hotel = self.hotel[batch]
        rate, seats, group = self.rate[batch], self.seats[batch], self.group[batch]
        local = {int(b): j for j, b in enumerate(batch)}
        starts = defaultdict(list)
        for j in range(B):
            if s0[j] >= 0 and t0[j] < 10 ** 9:
                starts[int(s0[j])].append(j)
        heaps: dict[int, list] = defaultdict(list)
        seq = 0
        for j, b in enumerate(batch):
            idle = self.start[b][4]
            if idle is not None and s0[j] >= 0:
                v = np.full((B, L), INF)
                v[j, R] = hotel[j, s0[j]]
                heaps[int(s0[j])].append((idle + rest + crews.BRIEF, seq, -3, v))
                seq += 1
        for h in heaps.values():
            heapq.heapify(h)
        pred = np.full((2, n, B, L), -1, dtype=np.int32) if trace else None
        src = np.zeros((n, B, L), dtype=np.int8) if trace and pilot else None
        restfrom = np.zeros((n, B), dtype=np.int16) if trace else None
        best_v: dict[int, np.ndarray] = {}
        best_p: dict[int, np.ndarray] = {}
        best_end = (self.idle if ends is None else ends[1])[batch].copy()
        best_at = [None] * B
        rows = np.arange(B)
        for k in range(n):
            s = int(self.lorig[k])
            hi, lo = int(self.legs.hi[k]), int(self.legs.lo[k])
            h = heaps.get(s)
            v, pr = best_v.get(s), best_p.get(s)
            if h and h[0][0] <= hi:
                while h and h[0][0] <= hi:
                    _, _, code, w = heapq.heappop(h)
                    if v is None:
                        v = w.copy()
                        pr = np.where(np.isfinite(w), code, -1).astype(np.int32) if trace else None
                    else:
                        better = w < v
                        if better.any():
                            v = np.where(better, w, v)
                            if trace:
                                pr = np.where(better, code, pr).astype(np.int32)
                best_v[s], best_p[s] = v, pr
            here = starts.get(s)
            if here:
                ok = [j for j in here if t0[j] <= hi and lo <= until[j]]
                if ok:
                    v = np.full((B, L), INF) if v is None else v.copy()
                    pr = (np.full((B, L), -1, dtype=np.int32) if pr is None else pr.copy()) if trace else None
                    for j in ok:
                        if 0.0 < v[j, l0[j]]:
                            v[j, l0[j]] = 0.0
                            if trace:
                                pr[j, l0[j]] = -2
            if v is None:
                continue
            fk = int(self.lflight[k])
            d = int(self.ldest[k])
            sched = self.lsched[k]
            dh = rate * self.lblock[k] + seats
            if self.planned.get(fk):
                for b in self.planned[fk]:
                    if b in local:
                        dh[local[b]] = 0.0
            val_dh = v + dh[:, None]
            if not pilot:
                okdh = np.full(B, lo + sched <= cab)
                for b in self.planned.get(fk, ()):
                    if b in local:
                        okdh[local[b]] = True
                val_dh[:, 0] = np.where(okdh, val_dh[:, 0], INF)
            fin_dh = np.isfinite(val_dh).any()
            mine = group == self.lgroup[k]
            val_op = None
            if mine.any():
                cost = -gain[k]
                if pilot:
                    okl = (lo + sched <= lim) & (mine & (sched <= room))[:, None]
                    okl[:, R] = mine
                    base = np.where(okl, v + cost, INF)
                    val_op = np.full((B, L), INF)
                    val_op[:, 1:8] = base[:, 0:7]
                    cap = base[:, 7] < base[:, 6]
                    val_op[:, 7] = np.where(cap, base[:, 7], base[:, 6])
                    val_op[:, R] = base[:, R]
                    if trace:
                        src[k] = np.arange(L)[None, :] - 1
                        src[k][:, 7] = np.where(cap, 7, 6)
                        src[k][:, R] = R
                else:
                    ok0 = (lo + sched <= cab) & mine
                    for b in self.own.get(fk, ()):
                        if b in local:
                            ok0[local[b]] = mine[local[b]]
                    val_op = np.where(np.stack([ok0, mine], axis=1), v + cost, INF)
                if not np.isfinite(val_op).any():
                    val_op = None
            arr = int(self.larr[k])
            for how, val in ((DH, val_dh if fin_dh else None), (OP, val_op)):
                if val is None:
                    continue
                if trace:
                    pred[how, k] = pr if how == DH else self._shift(pr, src[k] if pilot else None, L)
                heapq.heappush(heaps[d], (arr + crews.CONNECT, seq, 2 * k + how, val))
                seq += 1
                cand = val.min(axis=1) + endc[:, d]
                better = cand < best_end - 1e-9
                if better.any():
                    lay = val.argmin(axis=1)
                    best_end = np.where(better, cand, best_end)
                    for j in np.nonzero(better)[0]:
                        best_at[j] = (how, k, int(lay[j]))
            first = [val[:, :R] for val in (val_dh if fin_dh else None, val_op) if val is not None]
            if first:
                stack = np.concatenate(first, axis=1)
                arg = stack.argmin(axis=1)
                v1 = stack[rows, arg]
                if np.isfinite(v1).any():
                    w = np.full((B, L), INF)
                    w[:, R] = v1 + hotel[:, d]
                    if trace:
                        hows = [h for h, val in ((DH, val_dh if fin_dh else None), (OP, val_op)) if val is not None]
                        restfrom[k] = np.array([hows[a // R] * L + a % R for a in arg], dtype=np.int16)
                    heapq.heappush(heaps[d], (arr + crews.DEBRIEF + rest + crews.BRIEF, seq, -4 - k, w))
                    seq += 1
        days = [None] * B
        if trace:
            for j in range(B):
                days[j] = self._trace(j, units[j], best_at[j], pred, restfrom, src, L, s0[j])
        return best_end, days

    @staticmethod
    def _shift(pr: np.ndarray, src: np.ndarray | None, L: int) -> np.ndarray:
        if src is None:
            return pr
        out = np.full_like(pr, -1)
        for layer in range(L):
            col = src[:, layer]
            ok = col >= 0
            out[ok, layer] = pr[np.nonzero(ok)[0], col[ok]]
        return out

    def _trace(self, j: int, u: Unit, at: tuple | None, pred: np.ndarray, restfrom: np.ndarray,
               src: np.ndarray | None, L: int, s0: int) -> Day:
        if at is None:
            return self.day(u, (), (), None)
        steps: list[tuple[int, int]] = []
        restk = None
        how, k, layer = at
        while True:
            steps.append((k, how))
            code = int(pred[how, k, j, layer])
            if how == OP and src is not None:
                layer = int(src[k, j, layer])
            if code >= 0:
                k, how = divmod(code, 2)
                continue
            if code <= -4:
                kk = -4 - code
                restk = (len(steps), int(self.ldest[kk]))
                hl = int(restfrom[kk, j])
                how, layer, k = hl // L, hl % L, kk
                continue
            if code == -3:
                restk = (len(steps), s0)
            break
        steps.reverse()
        if restk is None:
            first, second, rested = tuple(steps), (), None
        else:
            cut = len(steps) - restk[0]
            first, second, rested = tuple(steps[:cut]), tuple(steps[cut:]), restk[1]
        return self.day(u, first, second, rested)

    def _b(self, u: Unit) -> int:
        return self.pos[u.id]

    def day(self, u: Unit, first: tuple, second: tuple, rested: int | None) -> Day:
        """A relaxed day priced: its deadheads off the unit's planned ones, a night away from base if it rested, its
        end; legs' prices are not in it."""
        b = self._b(u)
        planned = set(u.deadheads)
        cost = 0.0
        for k, how in (*first, *second):
            if how == DH and int(self.fl.ids[self.lflight[k]]) not in planned:
                cost += self.rate[b] * self.lblock[k] + self.seats[b]
        if rested is not None:
            cost += self.hotel[b, rested]
        steps = (*first, *second)
        end = int(self.ldest[steps[-1][0]]) if steps else self.start[b][0]
        cost += self.endc[b, end] if steps else self.idle[b]
        return Day(u.id, first, second, rested, end, cost,
                   tuple(int(self.fl.ids[self.lflight[k]]) for k, how in steps if how == OP))

    def days_of(self, day: crewsim.CrewDay) -> list[tuple[int, Day]]:
        """A crew sim day's units' days as relaxed days, each step in the bucket holding its minute; a unit with a step
        no bucket holds is left out."""
        out = []
        for uid, rec in day.units.items():
            periods: list[list[tuple[int, int]]] = [[], []]
            rested = None
            for d in rec["duties"]:
                second = d["begins"] == "rest"
                if second:
                    rested = self.fl.index.get(d["at"])
                steps = sorted([(t, i, OP) for i, t, _ in d["legs"]] + [(t, i, DH) for i, t, _ in d["deadheads"]])
                periods[second] += [(self.bucket(i, t), how) for t, i, how in steps]
            if all(k is not None for k, _ in periods[0] + periods[1]) and (rested is not None or not periods[1]):
                out.append((self.pos[uid], self.day(self.units[self.pos[uid]], tuple(periods[0]),
                                                    tuple(periods[1]), rested)))
        return out

    def bucket(self, i: int, t: int) -> int | None:
        return next((k for lo, hi, k in self.buckets.get(i, ()) if lo <= t <= hi), None)


@dataclass
class Covered:
    """The covering bound's pieces: its Lagrangian bound over the crews (the best round's), the master's last LP, the
    rounds and columns, whether it converged, the buckets kept by reduced-cost fixing and the flights that must fly."""

    bound: float
    lp: float
    rounds: int
    columns: int
    converged: bool
    legs: int
    must: int
    seconds: float


def cover(task: Task, fl: Flights, units: Sequence[Unit], local: crews.Local, ends: crews.Ends,
          choices: Mapping[int, tuple[np.ndarray, np.ndarray, float]], time_limit: float = 600.0,
          width: int = WIDTH, start: crewsim.CrewDay | None = None,
          log: Callable[[str], None] | None = None) -> Covered:
    """The covering bound's crew part: `choices[i]` is flight i's kept minutes, their prices and its cancellation's
    price (inf: it must fly), over buckets at most `width` minutes wide; the end of the day per base as `ends` prices
    it; the master starts with the days of `start`'s units (a crew sim day)."""
    t0 = time.monotonic()
    legs = Legs.of(fl, {i: c[0] for i, c in choices.items()}, {i: c[1] for i, c in choices.items()}, width)
    board = Board(task, fl, legs, units, local, lambda u, s: 0.0)
    m = Cover(fl, units, choices, board, ends)
    for b, day in board.days_of(start) if start is not None else ():
        m.add(b, day)
    best = -INF
    rounds = 0
    converged = False
    lp = 0.0
    center: tuple | None = None
    alpha = SMOOTH
    while time.monotonic() - t0 < time_limit:
        lp, pi, mu, sigma = m.solve()
        rounds += 1
        center = (pi, sigma) if center is None else center
        point = ({k: alpha * center[0][k] + (1 - alpha) * pi[k] for k in crews.KINDS},
                 alpha * center[1] + (1 - alpha) * sigma)
        value, days = board.price({k: point[0][k][legs.flight] - legs.price / 2 for k in crews.KINDS}, trace=True,
                                  ends=m.ends_at(point[1]))
        if (got := m.lagrangian(point[0], point[1], value)) > best:
            best, center = got, point
        added = sum(m.add(b, day) for b, day in enumerate(days) if m.reduced(b, day, pi, mu, sigma) < -EPS)
        if log:
            log(f"cover round {rounds} lp {lp:.0f} bound {best:.0f} added {added} {time.monotonic() - t0:.0f}s")
        if not added and not alpha:
            converged = True
            best = max(best, lp)
            break
        alpha = SMOOTH if added else 0.0
    return Covered(best, lp, rounds, len(m.days), converged, len(legs),
                   sum(1 for c in choices.values() if not np.isfinite(c[2])), round(time.monotonic() - t0, 1))


class Cover:
    """Per flight and kind, the units operating it plus the flight's cancellation (or, where it must fly, an
    artificial at BIG) equal one; per unit one day; and the end of the day per base, a transport problem: per kind,
    type group and base, the days ending at each station supply it, the base's units' trips tonight (crews.Ends) demand
    it, a move costing what Ends prices. A day costs its crew cost before the end of the day plus half its legs'
    prices."""

    def __init__(self, fl: Flights, units: Sequence[Unit], choices: Mapping[int, tuple], board: Board,
                 ends: crews.Ends):
        self.fl, self.units, self.board = fl, list(units), board
        self.h = h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("threads", 1)
        nf = len(fl.ids)
        self.row = {(kind, int(i)): r for r, (kind, i) in enumerate((k, i) for k in crews.KINDS for i in fl.ids)}
        self.urow = {b: 2 * nf + b for b in range(len(self.units))}
        self.places = list(fl.stations) + sorted({u.station for u in self.units} - set(fl.stations))
        self.at = {s: k for k, s in enumerate(self.places)}
        keys = sorted({ends.of[u.id] for u in self.units})
        self.cls = np.array([keys.index(ends.of[u.id]) for u in self.units])
        first = {c: next(u for u in self.units if ends.of[u.id] == key) for c, key in enumerate(keys)}
        self.wanted = [sorted(ends.wanted[key]) for key in keys]
        self.need = [np.array([ends.wanted[key][t] for t in ts], dtype=float) for key, ts in zip(keys, self.wanted,
                                                                                                strict=True)]
        self.move = [np.array([[ends.move(first[c], p, t) for t in self.wanted[c]] for p in self.places])
                     for c in range(len(keys))]
        n = len(self.places)
        self.srow = 2 * nf + len(self.units)
        self.trow = [self.srow + len(keys) * n + sum(len(w) for w in self.wanted[:c]) for c in range(len(keys))]
        nr = self.trow[-1] + len(self.wanted[-1]) if keys else self.srow
        lo = np.concatenate([np.ones(self.srow), np.zeros(len(keys) * n), *self.need])
        h.addRows(nr, lo, lo, 0, np.zeros(nr, dtype=np.int32), np.array([], dtype=np.int32), np.array([]))
        self.big = big = 1 + 2 * max([c[2] for c in choices.values() if np.isfinite(c[2])] + [1e6])
        self.cancel = np.array([choices.get(int(i), (np.array([]), np.array([]), 0.0))[2] for i in fl.ids])
        for i in fl.ids:
            i = int(i)
            c = choices.get(i, (np.array([]), np.array([]), 0.0))[2]
            rows = [self.row[(k, i)] for k in crews.KINDS]
            if np.isfinite(c):
                h.addCol(float(c), 0.0, 1.0, 2, np.array(rows, dtype=np.int32), np.ones(2))
            else:
                for r in rows:
                    h.addCol(float(big), 0.0, 1.0, 1, np.array([r], dtype=np.int32), np.ones(1))
        for c in range(len(keys)):
            k = len(self.wanted[c])
            src = np.repeat(self.srow + c * n + np.arange(n), k)
            dst = np.tile(self.trow[c] + np.arange(k), n)
            index = np.stack([src, dst], axis=1).ravel().astype(np.int32)
            h.addCols(n * k, self.move[c].ravel(), np.zeros(n * k), np.full(n * k, highspy.kHighsInf), 2 * n * k,
                      np.arange(0, 2 * n * k, 2, dtype=np.int32), index, np.tile([-1.0, 1.0], n * k))
        self.days: list = []
        self.seen: set = set()
        for b, u in enumerate(self.units):
            self.add(b, board.day(u, (), (), None))

    def place(self, b: int, day: Day) -> int:
        """Where the day ends, among `places`."""
        steps = (*day.first, *day.second)
        return int(self.fl.dest[self.board.lflight[steps[-1][0]]]) if steps else self.at[self.units[b].station]

    def add(self, b: int, day: Day | None) -> int:
        if day is None:
            return 0
        key = (b, day.first, day.second, day.rest)
        if key in self.seen:
            return 0
        self.seen.add(key)
        end = self.srow + int(self.cls[b]) * len(self.places) + self.place(b, day)
        coef: dict[int, float] = {self.urow[b]: 1.0, end: 1.0}
        for r in self.ops(b, day):
            coef[r] = coef.get(r, 0.0) + 1.0
        self.h.addCol(float(self.cost(day)), 0.0, INF, len(coef), np.array(list(coef), dtype=np.int32),
                      np.array(list(coef.values())))
        self.days.append(day)
        return 1

    def ops(self, b: int, day: Day) -> list[int]:
        kind = self.units[b].kind
        return [self.row[(kind, int(self.fl.ids[self.board.lflight[k]]))] for k, how in (*day.first, *day.second)
                if how == OP]

    def cost(self, day: Day) -> float:
        """The day's crew cost before the end of the day plus half the prices of the legs it operates."""
        return day.cost + sum(self.board.legs.price[k] / 2 for k, how in (*day.first, *day.second) if how == OP)

    def reduced(self, b: int, day: Day | None, pi: Mapping[str, np.ndarray], mu: np.ndarray,
                sigma: np.ndarray) -> float:
        if day is None:
            return INF
        nf = len(self.fl.ids)
        kind = self.units[b].kind
        return self.cost(day) - mu[b] - sum(pi[kind][r % nf] for r in self.ops(b, day)) - \
            sigma[self.cls[b], self.place(b, day)]

    def ends_at(self, sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Each unit's value of ending at each flight station and at its own station, at the transport duals."""
        own = np.array([self.at[u.station] for u in self.units])
        return -sigma[self.cls][:, :len(self.fl.stations)], -sigma[self.cls, own]

    def lagrangian(self, pi: Mapping[str, np.ndarray], sigma: np.ndarray, value: np.ndarray) -> float:
        """The covering problem's Lagrangian bound at duals `pi` on the flight rows and `sigma` on the stations each
        base's days end at, `value` each unit's least day less those duals: the duals, each cancellation (or
        artificial) where it pays, each base's trips tonight at their cheapest station, and the units' days."""
        both = pi["pilot"] + pi["cabin"]
        cancel = np.where(np.isfinite(self.cancel), np.minimum(0.0, np.where(np.isfinite(self.cancel), self.cancel,
                                                                              0.0) - both), 0.0)
        art = np.where(np.isfinite(self.cancel), 0.0,
                       np.minimum(0.0, self.big - pi["pilot"]) + np.minimum(0.0, self.big - pi["cabin"]))
        tonight = sum(float(need @ (move + sigma[c][:, None]).min(axis=0))
                      for c, (need, move) in enumerate(zip(self.need, self.move, strict=True)))
        return float(both.sum() + cancel.sum() + art.sum() + tonight + value.sum())

    def solve(self) -> tuple[float, dict[str, np.ndarray], np.ndarray, np.ndarray]:
        self.h.run()
        y = np.asarray(self.h.getSolution().row_dual)
        nf = len(self.fl.ids)
        pi = {kind: y[n * nf:(n + 1) * nf] for n, kind in enumerate(crews.KINDS)}
        n = len(self.places)
        sigma = y[self.srow:self.srow + len(self.need) * n].reshape(len(self.need), n)
        return self.h.getInfo().objective_function_value, pi, y[2 * nf:self.srow], sigma


def prices(air, lp: mip.Relaxed, upper: np.ndarray, cap: float) -> dict[int, tuple[np.ndarray, np.ndarray, float]]:
    """Per flight, from the aircraft LP within `upper`: its minutes priced at most `cap`, their prices, and its
    cancellation's price (inf where over `cap`). A choice's price is its reduced cost if positive, plus that of every
    other choice the LP has at its upper bound with a negative one."""
    net = air.net
    d, x = lp.reduced, lp.x
    out = {}
    for f in air.task.flights:
        cols = [(dd, net.xs[(f.id, dd)]) for dd, _ in net.copies[f.id]]
        cancel = [net.cancel[f.id]] + ([net.lapsed[f.id]] if f.id in net.lapsed else [])
        held = {c: -d[c] for c in [c for _, c in cols] + cancel
                if upper[c] > 0 and d[c] < 0 and x[c] >= upper[c] - 1e-9}
        extra = sum(held.values())
        ds, ps = [], []
        for dd, c in cols:
            if upper[c] > 0 and (p := max(0.0, d[c]) + extra - held.get(c, 0.0)) <= cap:
                ds.append(dd)
                ps.append(p)
        pc = min((max(0.0, d[c]) + extra - held.get(c, 0.0) for c in cancel if upper[c] > 0), default=INF)
        out[f.id] = (np.array(ds, dtype=np.int64), np.array(ps), pc if pc <= cap else INF)
    return out


def least(task: Task, fl: Flights, units: Sequence[Unit], local: crews.Local, end: Callable[[Unit, str], float],
          minutes: Mapping[int, np.ndarray]) -> np.ndarray:
    """Each unit's least crew cost over its relaxed days through `minutes`."""
    legs = Legs.of(fl, minutes)
    board = Board(task, fl, legs, units, local, end)
    value, _ = board.price({k: np.zeros(len(legs)) for k in crews.KINDS})
    return np.maximum(value, 0.0)
