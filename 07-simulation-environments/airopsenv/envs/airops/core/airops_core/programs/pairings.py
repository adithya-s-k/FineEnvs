"""Trips over the published schedule: per crew kind and type group, the duties that fly every leg of the date as
scheduled (flown and cancelled alike, never the flown day), by set partitioning with column generation.

The builder covers the date and the two days before it at once (where the cache lacks the earliest day, the day after
it stands in, a day earlier; labelled). A duty reports REPORT before its first departure, at a base or at an outstation
where a duty of the earlier days ended at least the rest before (one row per outstation and report time) or where a
planned rotation of its group starts the first day (one crew per aircraft): crews lay over wherever a duty ends
(labelled; the overnight-aircraft rule alone leaves west-coast turns and dawn red-eyes without a legal trip). Its legs
connect at a station in CONNECT at least, as crews connect in the crew sim and the crew replay, staying with the
aircraft or not, and no sit is longer than MAX_SIT. It may begin by deadheading from a base, and end by deadheading on
to a base or to an outstation where it lays over, and it ends anywhere (a night away from where it reported is a
hotel). It is legal as scheduled: a pilot duty within Table B for its report's local hour at the report station and its
segments and within Table A, without the extension (kept for the unforeseen); a cabin duty within 14 h from report to
release, its deadheads included. The objective counts crews first, then hotels, deadhead minutes, aircraft changes
(crews prefer to stay with their aircraft, along the planned rotations) and duty minutes. The column generation runs
until no duty prices out; a leg it leaves uncovered then has no legal duty under these rules. The date's duties are
those with a leg departing on the date, so a duty spanning midnight belongs to the date whose legs it flies; the earlier
days' duties only feed its outstation starts.

The planned rotations come from the schedule alone, never from the tails that flew: per family and station, each
scheduled arrival is turned into a departure of the family at least its turn later (the turn rule) and within a day,
the fewest aircraft first (a departure no arrival turns into starts a rotation where the schedule has it), then the
least sum of squared ground times (each aircraft turned as soon as the rule allows, evenly), an assignment HiGHS solves.
A leg's type group and family are its tail's (the record's only equipment field).
"""

import bisect
import time
from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace

import highspy
import numpy as np
import pandas as pd

from ..build import pack
from . import candidates, crews, network

CREW_COST = 10_000
HOTEL_COST = 500
DEADHEAD_COST = 2
CHANGE_COST = 50
DUTY_COST = 1
UNCOVERED = 1_000_000
ROUNDS = 5000
EARLY_KEY = 10_000_000
PER_START = 3
MAX_SIT = 240
TURN_WITHIN = 1440


@dataclass(frozen=True)
class Leg:
    """A scheduled leg: `key` its row, `hour` the local hour at its origin of a report REPORT before it, `after`
    its planned rotation's next leg."""

    key: int
    flight: str
    group: str
    orig: str
    dest: str
    sd: int
    sa: int
    block: int
    family: str
    hour: int
    after: int | None = None


@dataclass(frozen=True)
class Duty:
    kind: str
    group: str
    legs: tuple[int, ...]
    deadheads: tuple[int, ...]
    start: str
    end: str
    report: int
    release: int
    hour: int
    cost: float

    @property
    def key(self) -> tuple:
        return self.legs, self.deadheads, self.start, self.end


@dataclass
class Built:
    kind: str
    group: str
    duties: list[Duty]
    uncovered: list[int]
    lp: float
    ip: float
    columns: int
    rounds: int
    converged: bool
    seconds: float

    @property
    def gap_pct(self) -> float:
        return 100 * (self.ip - self.lp) / self.ip if self.ip else 0.0


def schedule(ctx: candidates.Context, inputs, derived) -> pd.DataFrame:
    """The carrier's legs of the date and the two days before it as scheduled, on the date's clock, keyed uniquely
    (the date's and the day before's by their row in the task's day, the earliest by an offset; the day before standing
    in for it a day earlier where the cache lacks it)."""
    day = ctx.day
    raw = candidates.frame(inputs, derived, pack.shift(day.date, -1))
    early = raw[(raw.FlightDate == pack.shift(day.date, -2)) & (raw.Reporting_Airline == pack.CARRIER)]
    early = pack.local(early, day.base) if len(early) else day.prev.assign(sd=day.prev.sd - 1440,
                                                                           sa=day.prev.sa - 1440)
    early.index = early.index + EARLY_KEY
    return pd.concat([early, day.prev, day.cur])


def legs_of(day: pack.Day, x: pd.DataFrame, families: Mapping[str, str]) -> list[Leg]:
    """Every typed leg of `x` as scheduled, with its planned rotation's next leg."""
    x = x[x.Tail_Number.isin(day.fleet) & x.Tail_Number.isin(families)]
    out = [Leg(int(r.Index), f"{pack.CARRIER} {int(r.Flight_Number_Reporting_Airline)}",
               crews.TYPE[families[r.Tail_Number]], r.Origin, r.Dest, int(r.sd), int(r.sa), int(r.CRSElapsedTime),
               families[r.Tail_Number], crews.local_hour(lambda s, m: network.local(day, s, m), r.Origin,
                                                         int(r.sd) - crews.REPORT))
           for r in x.itertuples()]
    nxt = rotations(out, {f: day.turn.get(f, day.turn_all) for f in {g.family for g in out}})
    return sorted((replace(g, after=nxt.get(g.key)) for g in out), key=lambda leg: (leg.sd, leg.key))


def rotations(legs: Sequence[Leg], turn: Mapping[str, int]) -> dict[int, int]:
    """The planned rotations (see the module doc): each leg's next leg, where one turns into it."""
    at: dict[tuple[str, str], tuple[list[Leg], list[Leg]]] = defaultdict(lambda: ([], []))
    for leg in legs:
        at[(leg.family, leg.dest)][0].append(leg)
        at[(leg.family, leg.orig)][1].append(leg)
    out = {}
    for (fam, _), (ins, outs) in sorted(at.items()):
        outs = sorted(outs, key=lambda g: (g.sd, g.key))
        times = [g.sd for g in outs]
        pairs = []
        for a in ins:
            lo = bisect.bisect_left(times, a.sa + turn[fam])
            hi = bisect.bisect_right(times, a.sa + TURN_WITHIN)
            pairs += [(a, b, b.sd - a.sa) for b in outs[lo:hi] if b.key != a.key]
        out |= turned(pairs)
    return out


def turned(pairs: Sequence[tuple[Leg, Leg, int]]) -> dict[int, int]:
    """The assignment over (arrival, departure, ground minutes): the most turns, then the least sum of squared ground
    times."""
    if not pairs:
        return {}
    rows: dict[tuple[int, int], int] = {}
    for a, b, _ in pairs:
        rows.setdefault((0, a.key), len(rows))
        rows.setdefault((1, b.key), len(rows))
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", 1)
    n = len(rows)
    h.addRows(n, np.zeros(n), np.ones(n), 0, np.zeros(n, dtype=np.int32), np.array([], dtype=np.int32), np.array([]))
    turn_once = n + 1
    for a, b, g in pairs:
        h.addCol((g / TURN_WITHIN) ** 2 - turn_once, 0.0, 1.0, 2,
                 np.array([rows[(0, a.key)], rows[(1, b.key)]], dtype=np.int32), np.ones(2))
    h.run()
    x = h.getSolution().col_value
    return {a.key: b.key for (a, b, _), v in zip(pairs, x, strict=True) if v > 0.5}


def build_all(legs: Sequence[Leg], d: str, time_limit: float = 600.0) -> dict[tuple[str, str], Built]:
    """Every crew kind and type group's trips of date d."""
    return {(kind, g): build(legs, kind, g, crews.bases(kind, d), time_limit=time_limit)
            for kind in crews.KINDS for g in crews.GROUPS if any(leg.group == g for leg in legs)}


def build(legs: Sequence[Leg], kind: str, group: str, bases: Sequence[str], time_limit: float = 600.0,
          first_end: int = -1440) -> Built:
    """Duties over `legs` (the date and the days before it, the first ending at `first_end`); the date's are the
    trips."""
    t0 = time.monotonic()
    mine = [leg for leg in legs if leg.group == group]
    pricer = Pricer(legs, mine, kind, group, bases)
    master = Master(mine, overnights(legs, group, bases, first_end), bases, crews.rest(kind))
    master.add(pricer.seed())
    rounds, converged = 0, False
    while rounds < ROUNDS and time.monotonic() - t0 < time_limit:
        rounds += 1
        legs_dual, lay, _ = master.solve_lp()
        if not master.add(pricer.price(legs_dual, lay, master.seen)):
            converged = True
            break
    lp = master.solve_lp()[2]
    ip, chosen = master.solve_ip(max(60.0, time_limit - (time.monotonic() - t0)))
    duties = [master.columns[j] for j in chosen]
    covered = {k for d in duties for k in d.legs}
    sd = {leg.key: leg.sd for leg in legs}
    today = [d for d in duties if max(sd[k] for k in d.legs) >= 0]
    return Built(kind, group, today, sorted(leg.key for leg in mine if leg.key not in covered and leg.sd >= 0),
                 lp, ip, len(master.columns), rounds, converged, round(time.monotonic() - t0, 1))


def overnights(legs: Sequence[Leg], group: str, bases: Sequence[str], before: int) -> dict[str, int]:
    """Per outstation, the planned rotations of the group that start there before `before`: a crew each."""
    turned_into = {leg.after for leg in legs if leg.after is not None}
    out: dict[str, int] = defaultdict(int)
    for leg in legs:
        if leg.group == group and leg.key not in turned_into and leg.sd < before and leg.orig not in bases:
            out[leg.orig] += 1
    return dict(out)


class Master:
    """min sum c_j x_j + UNCOVERED u: each leg flown by one duty or uncovered; a duty that reports at an outstation
    takes a crew there, from the aircraft overnighting there before the first day or from a duty that ended there at
    least `rest` earlier (one row per outstation and report time)."""

    def __init__(self, legs: Sequence[Leg], starts: Mapping[str, int], bases: Sequence[str], rest: int):
        self.bases, self.rest = set(bases), rest
        self.row = {leg.key: n for n, leg in enumerate(legs)}
        times: dict[str, set[int]] = defaultdict(set)
        for leg in legs:
            if leg.orig not in self.bases:
                times[leg.orig].add(leg.sd - crews.REPORT)
        self.lay = {s: sorted(ts) for s, ts in times.items()}
        n = len(legs)
        self.lay_row = {}
        lo = [1.0] * n
        for s, ts in sorted(self.lay.items()):
            for t in ts:
                self.lay_row[(s, t)] = n + len(self.lay_row)
                lo.append(-float(starts.get(s, 0)))
        hi = [1.0] * n + [highspy.kHighsInf] * len(self.lay_row)
        self.h = h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("threads", 1)
        h.addRows(len(lo), np.array(lo), np.array(hi), 0, np.zeros(len(lo), dtype=np.int32),
                  np.array([], dtype=np.int32), np.array([]))
        for k in range(n):
            h.addCol(UNCOVERED, 0.0, 1.0, 1, np.array([k], dtype=np.int32), np.array([1.0]))
        self.slack = n
        self.columns: list[Duty] = []
        self.seen: set[tuple] = set()

    def coefs(self, d: Duty) -> dict[int, float]:
        out = {self.row[k]: 1.0 for k in d.legs}
        if d.start not in self.bases:
            for t in self.lay.get(d.start, ()):
                if t >= d.report:
                    out[self.lay_row[(d.start, t)]] = -1.0
        if d.end not in self.bases:
            for t in self.lay.get(d.end, ()):
                if t >= d.release + self.rest:
                    out[self.lay_row[(d.end, t)]] = out.get(self.lay_row[(d.end, t)], 0.0) + 1.0
        return out

    def add(self, duties: Sequence[Duty]) -> int:
        n = len(self.columns)
        for d in duties:
            if d.key in self.seen:
                continue
            self.seen.add(d.key)
            c = self.coefs(d)
            self.h.addCol(d.cost, 0.0, highspy.kHighsInf, len(c), np.array(list(c), dtype=np.int32),
                          np.array(list(c.values())))
            self.columns.append(d)
        return len(self.columns) - n

    def solve_lp(self) -> tuple[dict[int, float], dict[str, tuple[list[int], np.ndarray]], float]:
        self.h.run()
        dual = np.asarray(self.h.getSolution().row_dual)
        legs = {k: dual[r] for k, r in self.row.items()}
        lay = {}
        for s, ts in self.lay.items():
            y = np.array([dual[self.lay_row[(s, t)]] for t in ts])
            lay[s] = (ts, np.cumsum(y[::-1])[::-1])
        return legs, lay, self.h.getInfo().objective_function_value

    def solve_ip(self, time_limit: float) -> tuple[float, list[int]]:
        h = self.h
        n = h.getNumCol()
        h.changeColsIntegrality(n, np.arange(n, dtype=np.int32), np.array([highspy.HighsVarType.kInteger] * n))
        h.setOptionValue("time_limit", float(time_limit))
        h.setOptionValue("mip_rel_gap", 1e-6)
        h.run()
        x = h.getSolution().col_value
        return h.getInfo().objective_function_value, [j for j in range(len(self.columns)) if x[self.slack + j] > 0.5]


def after(lay: Mapping[str, tuple[list[int], np.ndarray]], s: str, t: int) -> float:
    """The sum of the layover duals at s from t on."""
    if s not in lay:
        return 0.0
    ts, tail = lay[s]
    k = bisect.bisect_left(ts, t)
    return float(tail[k]) if k < len(ts) else 0.0


class Pricer:
    """Duties of least reduced cost, from each first operated leg (and a positioning deadhead before it): a depth-first
    search over connecting legs with dominance on segments and flight time."""

    def __init__(self, legs: Sequence[Leg], mine: Sequence[Leg], kind: str, group: str, bases: Sequence[str]):
        self.kind, self.group, self.bases = kind, group, set(bases)
        self.mine = mine
        self.by_orig: dict[str, list[Leg]] = defaultdict(list)
        for leg in mine:
            self.by_orig[leg.orig].append(leg)
        self.times = {s: [leg.sd for leg in ls] for s, ls in self.by_orig.items()}
        self.into: dict[str, list[Leg]] = defaultdict(list)
        self.home: dict[str, list[Leg]] = defaultdict(list)
        self.away: dict[str, list[Leg]] = defaultdict(list)
        for leg in sorted(legs, key=lambda g: g.sd):
            if leg.orig in self.bases:
                self.into[leg.dest].append(leg)
            (self.home if leg.dest in self.bases else self.away)[leg.orig].append(leg)
        self.homes = {s: [leg.sd for leg in ls] for s, ls in self.home.items()}
        self.aways = {s: [leg.sd for leg in ls] for s, ls in self.away.items()}
        self.rate = crews.rate(kind, group)
        self.rest = crews.rest(kind)

    def seed(self) -> list[Duty]:
        return [d for leg in self.mine for d in self.price_from(leg, None, {}, {}, keep=1)]

    def price(self, legs: Mapping[int, float], lay: Mapping, skip: Collection[tuple] = ()) -> list[Duty]:
        """The duties of negative reduced cost not in `skip`, the best PER_START from each start."""
        out = []
        for leg in self.mine:
            out += self.price_from(leg, None, legs, lay, skip=skip)
            for dh in self.into.get(leg.orig, ()):
                if dh.sa + crews.CONNECT <= leg.sd <= dh.sa + MAX_SIT and dh.key != leg.key:
                    out += self.price_from(leg, dh, legs, lay, skip=skip)
        return out

    def price_from(self, first: Leg, dh: Leg | None, pi: Mapping[int, float], lay: Mapping,
                   keep: int = PER_START, skip: Collection[tuple] = ()) -> list[Duty]:
        where = dh.orig if dh else first.orig
        lead = dh or first
        report, hour = lead.sd - crews.REPORT, lead.hour
        if not self.legal(report, hour, 1, first.sa):
            return []
        flying = crews.flight_time(hour)
        base = CREW_COST + ((DEADHEAD_COST + self.rate) * dh.block if dh else 0)
        shift = -after(lay, where, report) if where not in self.bases else 0.0
        found = []
        stack = [(first, 1, first.block, base, pi.get(first.key, 0.0), (first.key,))]
        kept: dict[tuple[int, int], list[tuple[int, float, float]]] = defaultdict(list)
        while stack:
            leg, n, ft, cost, gain, path = stack.pop()
            labels = kept[(leg.key, n)]
            if any(f <= ft and c - g <= cost - gain + 1e-9 for f, c, g in labels):
                continue
            labels.append((ft, cost, gain))
            for c, d in self.ends_at(leg, report, hour, cost, path, dh, where, lay):
                credit = after(lay, d.end, d.release + self.rest) if d.end not in self.bases else 0.0
                found.append((c - gain - shift - credit, d))
            if n >= 7:
                continue
            for nxt in self.next_legs(leg):
                if ft + nxt.block > flying or not self.legal(report, hour, n + 1, nxt.sa):
                    continue
                step = DUTY_COST * (nxt.sa - leg.sa) + CHANGE_COST * (leg.after != nxt.key)
                stack.append((nxt, n + 1, ft + nxt.block, cost + step, gain + pi.get(nxt.key, 0.0),
                              path + (nxt.key,)))
        found.sort(key=lambda p: p[0])
        out = [d for rc, d in found if (rc < -1e-6 or not pi) and d.key not in skip]
        return out[:keep]

    def legal(self, report: int, hour: int, n: int, sa: int) -> bool:
        if self.kind == "pilot":
            return sa - report <= crews.fdp(hour, n)
        return sa + crews.DEBRIEF - report <= crews.CABIN_DUTY

    def next_legs(self, leg: Leg) -> list[Leg]:
        ls, ts = self.by_orig.get(leg.dest, []), self.times.get(leg.dest, [])
        lo = bisect.bisect_left(ts, leg.sa)
        hi = bisect.bisect_right(ts, leg.sa + MAX_SIT)
        return [g for g in ls[lo:hi] if g.sd >= leg.sa + crews.CONNECT]

    def ends_at(self, leg: Leg, report: int, hour: int, cost: float, path: tuple[int, ...], dh: Leg | None,
                where: str, lay: Mapping) -> list[tuple[float, Duty]]:
        """The duty ending after `leg`: there, or deadheading on to a base, or to an outstation where a crew laid
        over is worth something at the duals `lay`."""
        first = (dh.key,) if dh else ()
        release = leg.sa + crews.DEBRIEF
        c = cost + DUTY_COST * (release - report) + HOTEL_COST * (leg.dest != where or leg.dest not in self.bases)
        out = [(c, Duty(self.kind, self.group, path, first, where, leg.dest, report, release, hour,
                        c))]
        backs = []
        if leg.dest not in self.bases:
            backs += self.first_out(self.home, self.homes, leg, lambda g: True)[:1]
        seen = set()
        backs += [g for g in self.first_out(self.away, self.aways, leg, lambda g: g.dest != leg.dest)
                  if g.dest not in seen and not seen.add(g.dest)
                  and after(lay, g.dest, g.sa + crews.DEBRIEF + self.rest) > 0]
        for back in backs:
            release = back.sa + crews.DEBRIEF
            if self.kind == "pilot" or release - report <= crews.CABIN_DUTY:
                c = (cost + DUTY_COST * (release - report) + (DEADHEAD_COST + self.rate) * back.block
                     + HOTEL_COST * (back.dest != where or back.dest not in self.bases))
                out.append((c, Duty(self.kind, self.group, path, first + (back.key,), where, back.dest, report,
                                    release, hour, c)))
        return out

    def first_out(self, by: Mapping[str, list[Leg]], times: Mapping[str, list[int]], leg: Leg, ok) -> list[Leg]:
        """The legs leaving leg's destination from CONNECT after it lands to MAX_SIT after, that pass `ok`."""
        ts = times.get(leg.dest, [])
        lo = bisect.bisect_left(ts, leg.sa + crews.CONNECT)
        hi = bisect.bisect_right(ts, leg.sa + MAX_SIT)
        return [g for g in by[leg.dest][lo:hi] if ok(g)] if leg.dest in by else []
