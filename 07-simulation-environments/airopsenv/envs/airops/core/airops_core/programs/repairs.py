"""The data repairs that make the truth JetBlue's world rather than JetBlue's day.

- Families split A320, A321 (190 seats and more), A321P (Mint and LR), A220 and E190; a flight's pax is capped at the
  smallest seat count of its family's tails in the task (`pax_raw` kept; spill is not modelled).
- Chains and positions come from each tail's scheduled legs, cancelled ones included. A break in a task tail's chain is
  closed by a fixed, uncosted repositioning of the aircraft that flies the leg before it (model.Move): it reappears at
  the next origin, ready at its scheduled departure less the turn (before the first leg the tail itself does, which
  sets `pos` and `ready`). A same-station gap of BUSY_GAP or more that starts in BUSY_HOURS is a busy block on the leg
  before it, standing for an international leg BTS lacks: the aircraft that flies that leg is busy until the next
  leg's scheduled departure less the turn (before the first leg it delays `ready`).
- Turns are the 5th percentile of scheduled ground times per family on the month's calm days (no GDP or ground stop
  listed for JFK or BOS), floored at MIN_TURN. A turn JetBlue flew faster, between a tail's consecutive flown legs, may
  be as fast in the model, for that connection only (model.Connection, Task.quick): the truth allows what JetBlue did.
  Tomorrow's need is the tail's last scheduled destination today, cancelled legs included: the schedule known at the
  start, never how JetBlue's day ended or its next day began.
- lb = DepDelay - LateAircraftDelay - NASDelay x [controlled or GS-held], never below the recorded early departure
  (floored at EARLY) nor below 0 for a flight that left on time or late: removing NAS delay a flight met in the air
  must not move its departure before the one it made. A cancelled flight takes the LIFT_PCT percentile of that lb over
  operated flights of any carrier to its destination within LIFT_WINDOW of its scheduled departure, else from its
  origin, else over the day.
- The truth block is the realized one, clipped at scheduled + BLOCK_SLACK; a diverted flight (even one that reached
  its destination in the end, its realized block holding the diversion) or a cancelled one takes the median realized
  block of the day's other flights on its route within an hour of it, else the scheduled block. A flight that pushed
  back while a ground stop at its hub covered its origin waited out the stop on the taxiway, so its realized block
  already holds the wait the model gives it at the gate (BTS dates the pushback; the stop held the takeoff): the block
  loses its taxi-out beyond the origin's median taxi-out, at most the time from its pushback to the stop's end. The
  model keeps the stop at the gate rather than holding wheels-off: that leaves its departure rules as they are, and the
  flight still lands when it did, which is what the costs read.
- The deadline is sched_dep + dmax, or the start + 60 for a flight already that late (pack.py's rule), capped at the
  hub cutoff for a departure from either hub.
- The crew model, an option the build turns on with crew `bases` (JetBlue's BASES), stands in for crew duty limits
  from the schedule and the regulation alone, since no crew data is public. Pairings are a proxy: each tail's scheduled
  legs of the day, cancelled ones and those before the window included, in order, chained from its first leg, from
  every departure at a base and from the first leg after an arrival at one (a break in its schedule), each chain up to
  and including the next arrival at a base. A chain reports REPORT minutes before its first scheduled departure; its
  maximum flight duty period is 14 CFR 117 Table B (Flight Duty Period: Unaugmented Operations; TABLE_B, as eCFR
  prints it, ECFR) for the report's local hour at its first origin, taken as the crew's acclimated time, and its
  number of segments (TABLE_B holds the hours for 1 to 7 or more segments, keyed by the first hour of each band of
  start times), plus the EXTENSION a certificate holder may add for unforeseen operational circumstances before
  takeoff (117.19(a)(1)). A departure from a base has no crew deadline: reserves crew it. A departure from an
  outstation must block in by the report + the limit + EXTENSION, so its crew deadline is that less its scheduled
  block, static as the schedule is; after it the crew may fly again only after REST, the 10 consecutive hours of rest
  117.25(e) requires (`rest`).
"""

import math
from collections import defaultdict
from collections.abc import Callable, Collection, Hashable, Iterable, Mapping, Sequence
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from ..build import fleet, pack
from .advisories import HUBS
from .model import Busy, Connection, Crew, Flight, Move, Outcome, Tail

FAMILIES = ("A320", "A321", "A321P", "A220", "E190")
MIN_TURN = 25
TURN_PCT = 5
BUSY_GAP = 240
BUSY_HOURS = (360, 1260)
EARLY = -15
BLOCK_SLACK = 120
LIFT_PCT = 75
LIFT_WINDOW = 90
LIFT_MIN = 3
DMAX = 900
CUTOFF = 1440 + 180
BASES = ("JFK", "BOS", "FLL", "MCO")
REPORT = 45
EXTENSION = 120
REST = 600
ECFR = "https://www.ecfr.gov/current/title-14/chapter-I/subchapter-G/part-117"
TABLE_B = {0: (9, 9, 9, 9, 9, 9, 9), 4: (10, 10, 10, 10, 9, 9, 9), 5: (12, 12, 12, 12, 11.5, 11, 10.5),
           6: (13, 13, 12, 12, 11.5, 11, 10.5), 7: (14, 14, 13, 13, 12.5, 12, 11.5),
           12: (13, 13, 13, 13, 12.5, 12, 11.5), 13: (12, 12, 12, 12, 11.5, 11, 10.5), 17: (12, 12, 11, 11, 10, 9, 9),
           22: (11, 11, 10, 10, 9, 9, 9), 23: (10, 10, 10, 9, 9, 9, 9)}


def family(a: fleet.Aircraft) -> str:
    if a.family != "A32X":
        return a.family
    if a.seats >= 190:
        return "A321"
    return "A321P" if "321" in a.type else "A320"


def turns(legs: pd.DataFrame, aircraft: Mapping[str, fleet.Aircraft]) -> dict[str, int]:
    """Per family, the TURN_PCT percentile of scheduled ground times between a tail's consecutive legs of a day that
    connect (BTS rows with Tail_Number, FlightDate, Origin, Dest and UTC sched_dep/sched_arr), floored at MIN_TURN; a
    family without such a pair takes the all-family value."""
    x = legs[legs.Tail_Number.isin(aircraft)].sort_values(["Tail_Number", "FlightDate", "sched_dep"])
    nxt = x.groupby(["Tail_Number", "FlightDate"])
    gap = nxt.sched_dep.shift(-1) - x.sched_arr
    ok = (x.Dest == nxt.Origin.shift(-1)) & (gap >= 0)
    fams = x.Tail_Number.map(lambda t: family(aircraft[t]))
    out = {fam: p5(g) for fam, g in gap[ok].groupby(fams[ok])}
    every = p5(gap[ok])
    return {fam: out.get(fam, every) for fam in FAMILIES}


def p5(values: Iterable[float]) -> int:
    v = sorted(values)
    return max(MIN_TURN, int(v[int(TURN_PCT / 100 * len(v))]))


def flights(day: pack.Day, sc: pack.Scope, x: pd.DataFrame, families: Mapping[str, str],
            tests: Callable[[str, str, int, int, int], tuple[str | None, str | None, bool]], dmax: int = DMAX,
            cutoff: int = CUTOFF, bases: Collection[str] = ()) -> list[Flight]:
    """The scope's flights, ids in (scheduled departure, flight number, origin) order. `x` holds every carrier's rows
    around the date with local sd/sa/ad/aa and the repaired `lb`; `tests(orig, dest, sched_arr, departs, sched_dep)`
    gives (controlled, gs_held, exempt); with crew `bases` each flight has its crew."""
    crew = {}
    if bases:
        legs = day.cur[day.cur.Tail_Number.isin(sc.tails)]
        crew = crews(legs, day.base, bases, {c: a["tz"] for c, a in pack.airports(set(legs.Origin)).items()})
    dec = sc.dec.assign(num=sc.dec.Flight_Number_Reporting_Airline.astype(int)).sort_values(["sd", "num", "Origin"])
    seats = defaultdict(list)
    for t in sc.tails:
        seats[families[t]].append(day.fleet[t].seats)
    pool = x[x.Cancelled == 0]
    chains = defaultdict(list)
    rows = []
    for i, r in enumerate(dec.itertuples()):
        chains[r.Tail_Number].append((i, r.Origin, r.Dest))
        rows.append((i, r))
    paired = {k: v for legs in chains.values() for k, v in pairs(legs).items()}
    out = []
    for i, r in rows:
        departed = r.Cancelled == 0
        departs = int(r.ad) if departed else int(r.sd)
        controlled, held, exempt = tests(r.Origin, r.Dest, int(r.sa), departs, int(r.sd))
        removed = controlled is not None or held is not None
        if departed:
            lb, kept = lb_of(r, removed), lb_of(r, False)
        else:
            lb = kept = imputed_lb(pool, r.Origin, r.Dest, r.sd)
        fam = families[r.Tail_Number]
        raw = fleet.pax(day.fleet[r.Tail_Number].seats, day.lf, r.Origin, r.Dest)
        reached = departed and (r.Diverted == 0 or r.DivReachedDest == 1)
        actual = Outcome(cancelled=not departed, diverted=bool(r.Diverted == 1),
                         code=r.CancellationCode if not departed and isinstance(r.CancellationCode, str) else "",
                         dep=int(r.ad) if departed else None, arr=int(r.aa) if reached else None,
                         end=r.end if departed else None, tail=r.Tail_Number)
        out.append(Flight(id=i, flight=f"{pack.CARRIER} {r.num}", orig=r.Origin, dest=r.Dest, sched_dep=int(r.sd),
                          sched_arr=int(r.sa), sched_block=int(r.CRSElapsedTime), block=truth_block(r, x),
                          pax=min(raw, min(seats[fam])), pax_raw=raw, family=fam, lb=lb, nas_removed=kept - lb,
                          controlled=controlled, gs_held=held, exempt=exempt, pair=paired.get(i),
                          deadline=deadline(int(r.sd), r.Origin, sc.start, dmax, cutoff), actual=actual,
                          crew=crew.get(r.Index)))
    return out


def tails(day: pack.Day, sc: pack.Scope, flights: list[Flight], families: Mapping[str, str],
          turn: Mapping[str, int]) -> tuple[list[Tail], dict[str, int]]:
    """The scope's tails and counts of what closure did: breaks before the first leg, between legs, busy blocks."""
    by_tail = defaultdict(list)
    for f in flights:
        by_tail[f.actual.tail].append(f)
    counts = {"closed_at_start": 0, "repositions": 0, "busy": 0}
    out = []
    for t in sc.tails:
        a = day.fleet[t]
        fam = families[t]
        tt = turn[fam]
        legs = sorted(by_tail[t], key=lambda f: (f.sched_dep, f.id))
        frozen = sc.frozen[sc.frozen.Tail_Number == t].sort_values("ad")
        prev = day.prev[(day.prev.Tail_Number == t) & (day.prev.Cancelled == 0)].sort_values("ad")
        last = frozen.iloc[-1] if len(frozen) else prev.iloc[-1] if len(prev) else None
        if last is not None:
            arrived = int(last.aa) if pd.notna(last.aa) else int(last.ad + last.CRSElapsedTime + 60)
            ready = arrived + tt if len(frozen) else max(0, arrived + tt)
        today = day.cur[day.cur.Tail_Number == t]
        if not legs and last is None:
            pos, ready = today.sort_values("sd").Origin.iloc[0], 0
        elif not legs:
            pos = last.end
        elif last is not None and last.end == legs[0].orig:
            pos = legs[0].orig
            if legs[0].sched_dep - int(last.sa) >= BUSY_GAP and BUSY_HOURS[0] <= int(last.sa) < BUSY_HOURS[1]:
                ready = max(ready, legs[0].sched_dep - tt)
                counts["busy"] += 1
        else:
            pos, ready = legs[0].orig, legs[0].sched_dep - tt
            counts["closed_at_start"] += 1
        busy, moves = [], []
        for f, g in zip(legs, legs[1:], strict=False):
            if f.dest != g.orig:
                moves.append(Move(f.id, f.dest, g.orig, f.sched_arr, g.sched_dep - tt))
            elif g.sched_dep - f.sched_arr >= BUSY_GAP and BUSY_HOURS[0] <= f.sched_arr < BUSY_HOURS[1]:
                busy.append(Busy(f.id, g.sched_dep - tt))
        counts["repositions"] += len(moves)
        counts["busy"] += len(busy)
        flown_today = today[today.Cancelled == 0].sort_values("ad")
        out.append(Tail(tail=t, family=fam, type=a.type, seats=a.seats, pos=pos, ready=int(ready),
                        legs=tuple(f.id for f in legs), busy=tuple(busy), repositions=tuple(moves),
                        end_req=today.sort_values("sd").Dest.iloc[-1] if len(today) else pos,
                        actual_end=flown_today.end.iloc[-1] if len(flown_today) else pos,
                        connections=connections(legs)))
    return out, counts


def connections(legs: Iterable[Flight]) -> tuple[Connection, ...]:
    """The turns JetBlue flew between one tail's legs: each flown leg that reached its destination to the tail's next
    flown leg, when that leaves from there, with the minutes from gate arrival to gate departure."""
    flown = sorted((f for f in legs if f.actual.dep is not None), key=lambda f: (f.actual.dep, f.id))
    return tuple(Connection(f.id, g.id, g.actual.dep - f.actual.arr) for f, g in zip(flown, flown[1:], strict=False)
                 if f.actual.arr is not None and f.actual.end == f.dest == g.orig and g.actual.dep >= f.actual.arr)


def new_lb(x: pd.DataFrame, removed: pd.Series) -> pd.Series:
    """The exogenous delay of each departed row, NAS delay taken out where `removed` (NaN when not departed)."""
    raw = x.DepDelay - x.LateAircraftDelay.fillna(0) - x.NASDelay.fillna(0) * removed
    floor = x.DepDelay.clip(upper=0).clip(lower=EARLY)
    return raw.where(raw > floor, floor).where(x.Cancelled == 0)


def lb_of(r, removed: bool) -> int:
    """new_lb for one departed row."""
    late = 0 if pd.isna(r.LateAircraftDelay) else r.LateAircraftDelay
    nas = 0 if pd.isna(r.NASDelay) else r.NASDelay
    return int(math.ceil(max(r.DepDelay - late - nas * removed, max(min(r.DepDelay, 0), EARLY))))


def imputed_lb(pool: pd.DataFrame, orig: str, dest: str, sd: float) -> int:
    near = (pool.sd - sd).abs() <= LIFT_WINDOW
    for sel in (near & (pool.Dest == dest), near & (pool.Origin == orig), pool.sd.notna()):
        v = pool.lb[sel].dropna()
        if len(v) >= LIFT_MIN:
            return int(math.ceil(v.quantile(LIFT_PCT / 100)))
    return 0


def truth_block(r, x: pd.DataFrame) -> int:
    """The truth block of a row of `x` or of a scope (both with `held_taxi`)."""
    sched = int(r.CRSElapsedTime)
    if r.Cancelled == 0 and r.Diverted == 0:
        return min(int(r.aa - r.ad) - r.held_taxi, sched + BLOCK_SLACK)
    same = x[(x.Origin == r.Origin) & (x.Dest == r.Dest) & ((x.sd - r.sd).abs() <= 60) & (x.Cancelled == 0)
             & (x.Diverted == 0)]
    return min(int(round((same.aa - same.ad - same.held_taxi).median())), sched + BLOCK_SLACK) if len(same) else sched


def held(pushback: int, released: int, taxi_out: float, median: float) -> int:
    """The part of the taxi-out of a flight that pushed back at `pushback` that a ground stop holding its takeoff
    until `released` already put in its realized block: beyond the origin's `median`, and no longer than the stop."""
    return int(max(0, min(released - pushback, taxi_out - median)))


def deadline(sched_dep: int, orig: str, start: int, dmax: int = DMAX, cutoff: int = CUTOFF) -> int:
    d = sched_dep + max(dmax, start - sched_dep + 60)
    return min(d, cutoff) if orig in HUBS else d


def pairs(legs: list, hubs: Collection[str] | None = HUBS) -> dict[int, int]:
    """Out-and-backs from a hub (from anywhere with `hubs` None) on one tail's scheduled legs (id, orig, dest),
    consecutive and paired left to right."""
    out = {}
    k = 0
    while k + 1 < len(legs):
        (a, ao, ad), (b, bo, bd) = legs[k], legs[k + 1]
        if (hubs is None or ao in hubs) and ao == bd and ad == bo:
            out[a], out[b] = b, a
            k += 2
        else:
            k += 1
    return out


def crews(legs: pd.DataFrame, base: int, bases: Collection[str], zones: Mapping[str, str]) -> dict[Hashable, Crew]:
    """The crew of each of `legs` by row label: a day's scheduled legs with Tail_Number, Origin, Dest, local sd and
    CRSElapsedTime (cancelled ones included), `base` the UTC epoch minute of their midnight, `zones` each origin's
    time zone."""
    out = {}
    for tail, x in legs.sort_values(["Tail_Number", "sd"]).groupby("Tail_Number"):
        for k, chain in enumerate(pairings(list(x.itertuples()), bases)):
            first = chain[0]
            report = int(first.sd) - REPORT
            limit = fdp(datetime.fromtimestamp(60 * (base + report), ZoneInfo(zones[first.Origin])).hour, len(chain))
            for r in chain:
                d = None if r.Origin in bases else report + limit + EXTENSION - int(r.CRSElapsedTime)
                out[r.Index] = Crew(f"{tail}/{k}", report, first.Origin, len(chain), limit, d,
                                    None if d is None else d + REST)
    return out


def pairings(legs: Sequence, bases: Collection[str]) -> list[list]:
    """A tail's legs (rows with Origin and Dest) in scheduled order as chains: one starts at the first leg, at every
    departure from a base and after every arrival at one."""
    out: list[list] = []
    for r in legs:
        if not out or r.Origin in bases or out[-1][-1].Dest in bases:
            out.append([r])
        else:
            out[-1].append(r)
    return out


def fdp(hour: int, segments: int) -> int:
    """Table B's maximum flight duty period in minutes for a report in local `hour` and `segments` flight segments."""
    return round(60 * TABLE_B[max(h for h in TABLE_B if h <= hour)][min(segments, 7) - 1])
