"""Whole-network tasks: a storm date with every JetBlue flight BTS has that day and every aircraft that flew one, as the
airline's control center runs it.

A date qualifies when the per-hub candidate rule (candidates.candidates) holds at JFK or BOS, those hubs being its storm
hubs, and the programs control or hold at least MIN_CONTROLLED of the task's JetBlue flights; a control date is one the
control rule (candidates.controls) picks, with the windows it picks there. The storm hub windows are
candidates.build_candidate's rule windows, or the control's; where no storm hub has a window that passes the hub
window's data clauses, they are taken from the first hour a hub program is in force (`fallback`). Every task starts at
START, 04:00 on the hub's clock (Eastern at JFK and BOS), and runs the whole day: its flights are every JetBlue leg of
the date from the start, flown and cancelled, and any aircraft of a leg's family may fly it: aircraft are pooled by
family at each station. Every typed tail with a leg that day is an aircraft of the task, unless it flew none that day
nor the day before: that one is out of service, and its cancelled legs stay for any aircraft. An aircraft starts where
the record has it: after the last leg it flew before its first leg of the task (one of the date's that left before the
start, or one of the day before's, delayed past the start or not), ready after its turn. A tail's legs are the legs it
flew from the start, in the order it flew them; a station gap between two of them is a repositioning the record lacks
(model.Move, bound to the aircraft of the leg before it), and one before the first moves the start to that leg's origin,
ready at its scheduled departure less the turn or after its arrival there, whichever is later. With no record before the
start the tail starts at its first flown leg's origin, ready at its scheduled departure less the turn. A long daytime
same-station gap before or between flown legs is a busy block, as repairs.tails reads one on the scheduled chain: none
where the tail's recorded ground time there is shorter, or where a cancelled leg BTS leaves on the tail was scheduled to
leave inside it. A tail whose last flown leg of the date lands at one of JetBlue's international gateways (GATEWAYS)
before DAYTIME station-local, no cancelled leg left on the tail after it, is away on flying abroad that BTS lacks: busy
past the task's last deadline (a busy block on that leg, or from the start if it landed before). Over the 19 months,
before 18:00 at least 64% of every half hour's last landings at the gateways are more than other stations' ordinary end
of day explains (84% of them in all), from 18:00 to 19:30 about half, later a minority. A turn JetBlue flew between
flown legs may be as fast (repairs.connections). Swaps and the stale tails BTS leaves on cancelled legs make no Move.
Tomorrow's needs are pooled: per family and station, where flying the whole schedule and its Moves from the start
positions leaves the aircraft; a count the schedule would leave below zero is zero, and the stations with the most
aircraft of the family give up the difference, one at a time.

The calm day starts from the positions the schedule assumes (`scheduled`), the storm day from the record's; the record's
morning against the schedule's is a finding (`shortfall`).
"""

import functools
import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from ..build import bts, pack
from . import candidates, rbs, repairs
from .advisories import HUBS
from .candidates import CARRIER, EVAL_PER_HUB, EVAL_WINTER, EVAL_WINTER_SHORT, MIN_CONTROLLED, MONTHS, Built, season
from .model import Busy, Flight, Move, Tail, Task

Windows = dict[str, tuple[int, int]]
START = 4 * 60
GATEWAYS = ("JFK", "BOS", "FLL", "MCO", "SJU")
DAYTIME = 18 * 60


def storms(inputs: Path, months: Sequence[str] = MONTHS) -> tuple[dict[str, tuple[str, ...]], dict]:
    """Every qualifying date with its storm hubs, and candidates.candidates' counts."""
    keys, counts = candidates.candidates(inputs, months)
    out: dict[str, list[str]] = defaultdict(list)
    for d, hub in keys:
        out[d].append(hub)
    return {d: tuple(h for h in HUBS if h in hubs) for d, hubs in sorted(out.items())}, counts


def control_dates(inputs: Path, derived: Path, months: Sequence[str] = MONTHS) -> dict[str, Windows]:
    """The dates the control rule picks, each with its hubs' control windows."""
    out: dict[str, Windows] = defaultdict(dict)
    for d, hub, w in candidates.controls(inputs, derived, months):
        out[d][hub] = w
    return dict(sorted(out.items()))


def build(d: str, hubs: Sequence[str], inputs: Path, derived: Path, e_fit: Mapping[str, float],
          e_median: Mapping[str, float], scale: Mapping[str, float], cutoff: int = repairs.CUTOFF,
          windows: Windows | None = None, split: str = "") -> Built:
    """The whole-network task of date d with storm `hubs`, from their rule windows or the given `windows` (a
    control's), or None and the reason."""
    row = {"date": d, "hubs": list(hubs), "windows": {}, "slides": {}, "first_hours": {}, "fallback": False,
           "task_id": None, "reason": ""}
    if windows:
        row["windows"] = {h: list(w) for h, w in windows.items()}
    else:
        for hub in hubs:
            b = candidates.build_candidate(d, hub, inputs, derived, e_fit, e_median, scale, cutoff)
            row["first_hours"][hub] = b.row["p0"]
            if b.task:
                row["windows"][hub], row["slides"][hub] = list(b.task.window), b.task.slide
    if not row["windows"]:
        first = {h: p0 for h, p0 in row["first_hours"].items() if p0 is not None}
        if not first:
            return Built(None, {**row, "reason": "no hub program in force 05:00-20:00"})
        row["fallback"] = True
        row["windows"] = {h: [60 * p0, 60 * (p0 + candidates.LENGTHS[-1])] for h, p0 in first.items()}
    hub = min(row["windows"], key=lambda h: (row["windows"][h][0], HUBS.index(h)))
    task = assemble(candidates.context(inputs, derived, d, hub), d, hub, tuple(hubs), row, e_fit, e_median, scale,
                    cutoff, split)
    held = task.clauses["controlled_or_held"]
    if not windows and held < MIN_CONTROLLED:
        return Built(None, {**row, "controlled_or_held": held,
                            "reason": f"fewer than {MIN_CONTROLLED} JetBlue flights controlled or held"})
    return Built(task, {**row, "task_id": task.task_id})


def assemble(ctx: candidates.Context, d: str, hub: str, hubs: tuple[str, ...], row: Mapping, e_fit: Mapping[str, float],
             e_median: Mapping[str, float], scale: Mapping[str, float], cutoff: int, split: str) -> Task:
    day = ctx.day
    sc = scope(day, (min(w[0] for w in row["windows"].values()), max(w[1] for w in row["windows"].values())), START)
    flights = repairs.flights(day, sc, ctx.x, ctx.families, candidates.tests(ctx), cutoff=cutoff)
    tails, counts = network_tails(day, sc, flights, ctx.families, day.turn)
    need, short = needs(tails, flights)
    programs = candidates.hub_programs(ctx, flights, sc.start, e_fit, e_median, scale)
    codes = ({f.orig for f in flights} | {f.dest for f in flights} | set(HUBS)
             | {s for t in tails for s in (t.pos, t.end_req, t.actual_end)}
             | {m.dest for t in tails for m in t.repositions} | {a.orig for p in programs.values() for a in p.demand})
    held = sum(1 for f in flights if f.controlled or f.gs_held)
    clauses = {"tails_known": sc.untailed == 0 and sc.typed, "untailed": sc.untailed, "tails": len(sc.tails),
               "weather_share": round(sc.weather_share, 3), "controlled_or_held": held,
               "windows": dict(row["windows"]), "slides": dict(row["slides"]), "fallback": row["fallback"],
               "network": counts, "short": short}
    task = Task(task_id=f"{CARRIER.lower()}-{d.replace('-', '')}-{sc.start // 60:02d}00", split=split,
                carrier=CARRIER, hub=hub, date=d, tz=day.tz, base=day.base, window=sc.window, start=sc.start,
                slide=row["slides"].get(hub, 0), dmax=repairs.DMAX, cutoff=cutoff,
                turn={f: day.turn[f] for f in sorted({t.family for t in tails} | {f.family for f in flights})},
                airports=candidates.airports_of(codes),
                artcc={c: ctx.artcc[c] for c in sorted(codes) if c in ctx.artcc}, tails=tuple(tails),
                flights=tuple(flights), programs=programs, clauses=clauses, sources=candidates.pins(ctx, sc),
                hubs=hubs, needs=need)
    return rbs.rescale(task, scale, e_fit, e_median)


def scope(day: pack.Day, window: tuple[int, int], start: int) -> pack.Scope:
    """Every typed tail's legs of the day: those decided from `start` and those that left before it."""
    x = day.cur
    typed = x[x.Tail_Number.isin(day.fleet)]
    departed = typed.Cancelled == 0
    dec = typed[(departed & (typed.ad >= start)) | (~departed & (typed.sd >= start))]
    codes = dec[dec.Cancelled == 1].CancellationCode.value_counts()
    return pack.Scope(window, start, sorted(set(typed.Tail_Number)), int(x.Tail_Number.isna().sum()),
                      len(typed) == int(x.Tail_Number.notna().sum()), dec, typed[departed & (typed.ad < start)],
                      {k: int(v) for k, v in sorted(codes.items())})


def network_tails(day: pack.Day, sc: pack.Scope, flights: Sequence[Flight], families: Mapping[str, str],
                  turn: Mapping[str, int]) -> tuple[list[Tail], dict[str, int]]:
    """Every tail of the scope in service with its start, the legs it flew from the start in the order it flew them,
    their Moves and busy blocks, and counts of where the starts came from, of the Moves and blocks, of the tails away
    from a gateway and of those out of service."""
    flew, dropped = defaultdict(list), defaultdict(list)
    for f in flights:
        (dropped if f.actual.dep is None else flew)[f.actual.tail].append(f)
    end = max((f.deadline for f in flights), default=0) + 1
    counts: Counter = Counter()
    out = []
    for t in sc.tails:
        a, fam = day.fleet[t], families[t]
        tt = turn[fam]
        path = sorted(flew[t], key=lambda f: (f.actual.dep, f.id))
        start = start_of(day, sc, t, path, dropped[t], tt)
        if start is None:
            counts["out of service"] += 1
            continue
        pos, ready, source = start
        busy, moves = [], []
        for f, g in zip(path, path[1:], strict=False):
            if f.dest != g.orig:
                moves.append(Move(f.id, f.dest, g.orig, f.sched_arr, g.sched_dep - tt))
                counts["moves after a diversion" if f.actual.end == g.orig else "moves"] += 1
            elif busy_gap(f.sched_arr, g.sched_dep, g.actual.dep - arrival(f), dropped[t]):
                busy.append(Busy(f.id, g.sched_dep - tt))
        counts[f"start: {source}"] += 1
        counts["busy"] += len(busy)
        if away(day, sc, t, path, dropped[t]):
            counts["gateway busy"] += 1
            if path:
                busy.append(Busy(path[-1].id, end))
            else:
                ready = end
        done = day.cur[(day.cur.Tail_Number == t) & (day.cur.Cancelled == 0)].sort_values("ad")
        out.append(Tail(tail=t, family=fam, type=a.type, seats=a.seats, pos=pos, ready=int(ready),
                        legs=tuple(f.id for f in path), busy=tuple(busy), repositions=tuple(moves),
                        end_req=path[-1].dest if path else pos, actual_end=done.end.iloc[-1] if len(done) else pos,
                        connections=repairs.connections(path)))
    return out, dict(sorted(counts.items()))


def start_of(day: pack.Day, sc: pack.Scope, tail: str, path: Sequence[Flight], dropped: Sequence[Flight],
             turn: int) -> tuple[str, int, str] | None:
    """Where and when the tail is ready at the start, and where that comes from; None for a tail out of service."""
    first = path[0].actual.dep if path else math.inf
    today = sc.frozen[sc.frozen.Tail_Number == tail].sort_values("ad")
    before = day.prev[(day.prev.Tail_Number == tail) & (day.prev.Cancelled == 0) & (day.prev.ad < first)]
    before = before.sort_values("ad")
    last, source = max(((x.iloc[-1], s) for s, x in (("day before", before), ("today", today)) if len(x)),
                       key=lambda p: p[0].ad, default=(None, ""))
    if last is None:
        return (path[0].orig, path[0].sched_dep - turn, "first flown leg") if path else None
    arrived = landed(last)
    ready = max(0, arrived + turn)
    if not path:
        return last.end, ready, f"{source}, no flown leg"
    if last.end != path[0].orig:
        return path[0].orig, max(ready, path[0].sched_dep - turn), f"{source}, moved to the first flown leg"
    if busy_gap(int(last.sa), path[0].sched_dep, path[0].actual.dep - arrived, dropped):
        return last.end, max(ready, path[0].sched_dep - turn), f"{source}, busy"
    return last.end, ready, source


def landed(leg: pd.Series) -> int:
    """When a flown BTS leg reached the gate, or for one that diverted and never got there, an hour after its
    scheduled block."""
    return int(leg.aa) if pd.notna(leg.aa) else int(leg.ad + leg.CRSElapsedTime + 60)


def busy_gap(arrives: int, leaves: int, ground: int, dropped: Iterable[Flight]) -> bool:
    """A long daytime gap between a tail's scheduled arrival and its next flown leg, as long on the ground in the
    record, that none of its cancelled legs was scheduled to fill: flying the record lacks."""
    long = min(leaves - arrives, ground) >= repairs.BUSY_GAP
    return long and repairs.BUSY_HOURS[0] <= arrives < repairs.BUSY_HOURS[1] and \
        not any(arrives <= f.sched_dep < leaves for f in dropped)


def away(day: pack.Day, sc: pack.Scope, tail: str, path: Sequence[Flight], dropped: Sequence[Flight]) -> bool:
    """Whether the tail's last flown leg of the date (the last of `path`, else of those it flew before the start)
    landed at a gateway before DAYTIME on the gateway's clock with no cancelled leg BTS leaves on the tail scheduled
    after it."""
    if path:
        station, at, sd = path[-1].actual.end, arrival(path[-1]), path[-1].sched_dep
    else:
        today = sc.frozen[sc.frozen.Tail_Number == tail]
        if not len(today):
            return False
        last = today.sort_values("ad").iloc[-1]
        station, at, sd = last.end, landed(last), last.sd
    x = day.cur
    later = any(f.sched_dep > sd for f in dropped) or ((x.Tail_Number == tail) & (x.Cancelled == 1) & (x.sd > sd)).any()
    return station in GATEWAYS and not later and local(day, station, at) < DAYTIME


def local(day: pack.Day, station: str, minute: int) -> int:
    """The task's minute on the station's clock."""
    at = datetime.fromtimestamp(60 * (day.base + minute), UTC)
    return minute + int((at.astimezone(zone(station)).utcoffset() - at.astimezone(ZoneInfo(day.tz)).utcoffset())
                        .total_seconds()) // 60


@functools.cache
def zone(station: str) -> ZoneInfo:
    return ZoneInfo(bts.timezones()[station])


def arrival(f: Flight) -> int:
    """When a flown leg reached the gate, or for one that diverted and never got there, `landed`'s estimate."""
    return f.actual.arr if f.actual.arr is not None else f.actual.dep + f.sched_block + 60


def needs(tails: Sequence[Tail], flights: Sequence[Flight]) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    """Tomorrow's aircraft per family and station, and the stations the schedule would leave short, by how many."""
    raw: Counter = Counter((t.family, t.pos) for t in tails)
    for f in flights:
        raw[(f.family, f.dest)] += 1
        raw[(f.family, f.orig)] -= 1
    for t in tails:
        for m in t.repositions:
            raw[(t.family, m.dest)] += 1
            raw[(t.family, m.orig)] -= 1
    out = {}
    for fam in sorted({t.family for t in tails}):
        have = {s: n for (g, s), n in raw.items() if g == fam and n > 0}
        for _ in range(sum(have.values()) - sum(t.family == fam for t in tails)):
            have[max(sorted(have), key=lambda s: have[s])] -= 1
        out[fam] = {s: n for s, n in sorted(have.items()) if n}
    return out, {f"{fam} {s}": -n for (fam, s), n in sorted(raw.items()) if n < 0}


def scheduled(task: Task) -> Task:
    """The task from the positions its schedule assumes, every aircraft ready at the start and tomorrow's needs
    following them (`needs`). Per family, each station gets the aircraft `assumed` puts there. While the family has
    more aircraft than that, the station the schedule leaves with the most of them at the end of the day takes one;
    while it has fewer, that station among those given one gives one up, its departures waiting for an arrival (ties
    to the first code). A tail stays where it is while its station has room, the others fill the stations left in code
    order, in tail order. A tail away past every deadline (from a gateway) stays as it is, and so does a family without
    departures."""
    want = assumed(task)
    last = max((f.deadline for f in task.flights), default=0)
    net: Counter = Counter()
    for f in task.flights:
        m = task.moved.get(f.id)
        net[(f.family, m.dest if m else f.dest)] += 1
        net[(f.family, f.orig)] -= 1
    at = {t.tail: t.pos for t in task.tails}
    for fam in sorted({t.family for t in task.tails}):
        mine = sorted((t for t in task.tails if t.family == fam and t.ready <= last), key=lambda t: t.tail)
        room = Counter({s: n for (g, s), n in want.items() if g == fam})
        if not room:
            continue
        stations = sorted({s for g, s in net if g == fam})
        while sum(room.values()) < len(mine):
            room[max(stations, key=lambda s: room[s] + net[(fam, s)])] += 1
        while sum(room.values()) > len(mine):
            room[max((s for s in stations if room[s] > 0), key=lambda s: room[s] + net[(fam, s)])] -= 1
        moved = []
        for t in mine:
            if room[t.pos] > 0:
                room[t.pos] -= 1
            else:
                moved.append(t.tail)
        at |= dict(zip(moved, sorted(room.elements()), strict=True))
    tails = [t if t.ready > last else replace(t, pos=at[t.tail], ready=task.start) for t in task.tails]
    return replace(task, tails=tuple(tails), needs=needs(tails, task.flights)[0])


def assumed(task: Task) -> Counter:
    """Per (family, station), the aircraft the schedule needs there at the start so that every flight leaves on
    schedule (no earlier than the start), each aircraft back after its turn or busy block, at its Move's destination
    from the Move's ready: the most the station's departures run ahead of its aircraft back by then."""
    return Counter({k: -n for k, n in lowest(task, ()).items() if n < 0})


def lowest(task: Task, tails: Iterable[Tail]) -> dict[tuple[str, str], int]:
    """Per (family, station), the fewest aircraft on the ground when every flight leaves on schedule from `tails`'
    positions, below zero by the departures that find none."""
    events = defaultdict(list)
    for t in tails:
        events[(t.family, t.pos)].append((t.ready, 1))
    for f in task.flights:
        dep = task.earliest(f, 0)
        back, m = task.back(f, dep + f.sched_block), task.moved.get(f.id)
        events[(f.family, f.orig)].append((dep, -1))
        events[(f.family, m.dest if m else f.dest)].append((max(back, m.ready) if m else back, 1))
    out = {}
    for k, xs in events.items():
        n = low = 0
        for _, d in sorted(xs, key=lambda e: (e[0], -e[1])):
            n += d
            low = min(low, n)
        out[k] = low
    return out


def shortfall(task: Task) -> dict:
    """The record's morning against the schedule's (`scheduled`): per family and station the aircraft it has fewer
    than the schedule's positions, and the departures that, flying the schedule from the record's positions, find no
    aircraft of their family on time (`departures`), beside those the schedule's own positions leave so (`fleet`)."""
    plan = scheduled(task)
    want = Counter((t.family, t.pos) for t in plan.tails)
    have = Counter((t.family, t.pos) for t in task.tails)
    short = {f"{fam} {s}": n - have[(fam, s)] for (fam, s), n in sorted(want.items()) if n > have[(fam, s)]}
    return {"aircraft": sum(short.values()), "departures": -sum(min(0, n) for n in lowest(task, task.tails).values()),
            "fleet": -sum(min(0, n) for n in lowest(plan, plan.tails).values()), "short": short}


def build_all(jobs: Sequence[tuple[str, tuple[str, ...], Windows | None, str]], inputs: Path, derived: Path,
              e_fit: Mapping[str, float], e_median: Mapping[str, float], scale: Mapping[str, float],
              cutoff: int = repairs.CUTOFF, workers: int = 1) -> list[Built]:
    """`build` for every (date, storm hubs, windows, split), in a pool of `workers` processes, in date order."""
    args = [(d, hubs, inputs, derived, dict(e_fit), dict(e_median), dict(scale), cutoff, windows, split)
            for d, hubs, windows, split in sorted(jobs, key=lambda j: j[0])]
    with ProcessPoolExecutor(workers) if workers > 1 else nullcontext() as pool:
        if not pool:
            return [build_one(a) for a in args]
        return list(pool.map(build_one, args, chunksize=max(1, math.ceil(len(args) / (4 * workers)))))


def build_one(args: tuple) -> Built:
    d, hubs, inputs, derived, e_fit, e_median, scale, cutoff, windows, split = args
    return build(d, hubs, inputs, derived, e_fit, e_median, scale, cutoff, windows, split)


def split(hubs: Mapping[str, Sequence[str]], event: Mapping[str, int], seed: int) -> dict[str, str]:
    """eval for the draw, train for every date of an event without an eval date."""
    chosen = draw(hubs, event, seed)
    taken = {event[d] for d in chosen}
    return {d: "eval" if d in chosen else "train" for d in sorted(hubs) if d in chosen or event[d] not in taken}


def draw(hubs: Mapping[str, Sequence[str]], event: Mapping[str, int], seed: int) -> dict[str, str]:
    """The eval dates, each to the storm hub it was drawn for: per hub in turn, 12 of the dates it storms on that no
    hub drew before, 8 summer and 4 winter (9 and 3 if winter has fewer than 4 events), each from a distinct event,
    events and dates drawn with `seed`."""
    rng = random.Random(seed)
    out: dict[str, str] = {}
    for hub in HUBS:
        by = {s: defaultdict(list) for s in ("summer", "winter")}
        for d in sorted(hubs):
            if hub in hubs[d] and d not in out and season(d) in by:
                by[season(d)][event[d]].append(d)
        winter = min(EVAL_WINTER, len(by["winter"]))
        if winter < EVAL_WINTER:
            winter = min(EVAL_WINTER_SHORT, len(by["winter"]))
        for s, n in (("summer", min(EVAL_PER_HUB - winter, len(by["summer"]))), ("winter", winter)):
            order = sorted(by[s])
            rng.shuffle(order)
            out |= {rng.choice(sorted(by[s][e])): hub for e in order[:n]}
    return dict(sorted(out.items()))


def settle(drawn: Mapping[str, str], qualifying: Iterable[str], event: Mapping[str, int]) -> dict[str, str]:
    """candidates.settle over dates: the drawn eval dates that still qualify, and train every qualifying date of an
    event without a drawn eval date."""
    taken = {event[d] for d, v in drawn.items() if v == "eval"}
    return {d: "eval" if drawn.get(d) == "eval" else "train" for d in sorted(qualifying)
            if drawn.get(d) == "eval" or event[d] not in taken}


def label(task: Task) -> str:
    """The task's storm hubs and season, as the day counts read them."""
    return f"{'+'.join(task.storms)} {season(task.date)}"
