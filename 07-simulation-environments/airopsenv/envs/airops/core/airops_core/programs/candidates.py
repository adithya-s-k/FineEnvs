"""The candidates, their windows and splits, over the 19 cached months (MONTHS).

A candidate is a local date and hub with a program that binds JetBlue (a GDP or an all-carrier ground stop at the hub)
in force on the truth between 05:00 and 24:00 local, both neighbouring days loaded (one whose neighbour isn't is
rejected, with the day missing) and every advisory it reads archived: a date whose read range meets an outage of the
FAA's advisory archive (GAPS) is no candidate, no control and no calm day for the turns. Its window is a rule, not an
argmax: P0 is the first hour from 05:00 to 20:00 with a hub program in force (a GDP from its signature or its period's
start, whichever is later, to its run's end; a stop piece over its window), the window [P0, P0 + L) with L the
longest of 6/5/4/3 h whose scope holds 30 to 80 tails, and the first watch an hour before P0. While the data clauses
fail (tails known and typed, weather and NAS at least 80% of coded cancellations, 30 to 80 tails, at least 5 task
flights controlled or GS-held on the truth) the window slides by an hour, up to 3; the first that passes wins. The
clauses that need a solver (airline replay, recorded day, fidelity, leverage, proof) are the runner's.

Storm events chain program dates within 2 days, both hubs pooled. The seeded eval draw takes 12 day-hubs per hub,
8 summer (Jun-Aug) and 4 winter (Dec-Feb), 9 and 3 if winter is short, at most one per hub per event; train is every
qualifying candidate of an event with no eval day-hub. When clauses rerun after the draw, the drawn eval stays and only
loses the days that fail (`settle`). The controls are chosen by rule: per season, the date with no program listed at
JFK, BOS, EWR or LGA (a GDP or ground stop on the date's or the next UTC day's ATCSCC list) and JetBlue's highest
network-wide on-time rate (`on_time`), the earlier date on a tie, windowed at each hub on its busiest 4 h of JetBlue
movements.

The fleet of a month is typed by the two published Form 41 B-43 inventories nearest it (`b43_years`, over the years
B43_PUBLISHED pins, whatever the cache holds): 2026's is not published, so the 2026 months take 2025's and 2024's.

Every function a process pool maps takes picklable arguments (dates, paths, small dicts) and reads the pinned inputs
through per-process caches, so the ~300 candidates build in parallel.
"""

import csv
import functools
import io
import math
import os
import random
import statistics
import zipfile
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from datetime import date as Date
from pathlib import Path

import airportsdata
import numpy as np
import pandas as pd

from ..build import advisories as v1
from ..build import bts, fetch, fleet, pack, windows
from ..core.model import Airport
from . import advisories, rbs, repairs, unseen
from .advisories import HUBS, Run, Stop
from .model import Arrival, Flight, Programs, Task, coverage
from .rbs import Covers

CARRIER = pack.CARRIER
MONTHS = tuple(sorted((*pack.MONTHS, "2025-08", "2025-12", "2026-01", "2026-02", "2026-06", "2026-07")))
GAPS = (("2025-02-16T05:15", "2025-02-16T13:00"), ("2025-08-23T09:50", "2025-08-26T03:41"))
"""UTC spans the FAA's advisory archive has nothing for: 2025-02-16's list skips advisories 23 to 46, from 22 (05:15Z)
to 47 (13:00Z); its lists of 2025-08-24 and 08-25 are empty, 08-23's stops at advisory 34 (09:50Z) and 08-26's starts
at advisory 22 (03:41Z); refetching gives the same pages."""
B43_PUBLISHED = (2021, 2022, 2023, 2024, 2025)
SUMMER, WINTER = ("06", "07", "08"), ("12", "01", "02")
FIRST, LAST_START, DAY_END = 5, 20, 24
LENGTHS = (6, 5, 4, 3)
SLIDES = 3
MIN_CONTROLLED = 5
EVENT_GAP = 2
EVAL_PER_HUB, EVAL_WINTER, EVAL_WINTER_SHORT = 12, 4, 3
CONTROL_HOURS = 4
LISTED = ("JFK", "BOS", "EWR", "LGA")


@dataclass(frozen=True)
class Built:
    task: Task | None
    row: dict


def candidates(inputs: Path, months: Sequence[str] = MONTHS) -> tuple[list[tuple[str, str]], dict]:
    """Every (local date, hub) candidate in order, and the counts behind it: UTC day-hubs with a hub GDP or ground
    stop of any kind, local day-hubs with a hub program of any kind in force 05:00-24:00, those of them whose only
    programs are carrier-, GA- or approach-limited stops, the local dates an archive outage drops, and under
    "rejected" each day-hub with a binding program that can't be a candidate, with the reason."""
    loaded = {d for m in months for d in pack.month_days(m)}
    out, utc, local, limited, rejected = [], set(), set(), set(), {}
    gapped = sorted(d for d in loaded if gap(d))
    for d in sorted(loaded):
        for hub in HUBS:
            if any(v1.kind(v1.title_of(r["header"])) in (advisories.GDP, advisories.GS)
                   for r in v1.load_day(inputs, d, hub)):
                utc.add((d, hub))
            if d in gapped:
                continue
            missing = [x for x in (pack.shift(d, -1), pack.shift(d, 1)) if x not in loaded]
            if missing:
                if any_in_force(*programs_of(inputs, d, hub), 300, 1440):
                    rejected[f"{d}/{hub}"] = f"no BTS day {', '.join(missing)}"
                continue
            if any_in_force(*programs_of(inputs, d, hub, binding=False), 300, 1440):
                local.add((d, hub))
                if any_in_force(*programs_of(inputs, d, hub), 300, 1440):
                    out.append((d, hub))
                else:
                    limited.add((d, hub))
    return out, {"utc_day_hubs": len(utc), "local_day_hubs": len(local), "limited_only": len(limited),
                 "candidates": len(out), "archive_gap_dates": gapped, "rejected": rejected}


def gap(d: str) -> bool:
    """Whether local date d reads advisories (programs_of: from LOOKBACK before its midnight, for two days) signed in
    an archive outage."""
    lo = midnight(d) - advisories.LOOKBACK
    return any(lo < utc_minute(b) and utc_minute(a) < lo + advisories.LOOKBACK + 2 * 1440 for a, b in GAPS)


def utc_minute(text: str) -> int:
    return bts.epoch_minutes(datetime.fromisoformat(text).replace(tzinfo=UTC))


def program_title(title: str) -> bool:
    t = title.upper()
    return "GROUND DELAY" in t or "GROUND STOP" in t or " GS " in f" {t} " or " GDP " in f" {t} "


def midnight(d: str) -> int:
    return bts.epoch_minutes(pd.Timestamp(d).tz_localize(bts.airports()["JFK"]["tz"]).tz_convert("UTC").to_pydatetime())


@functools.lru_cache(maxsize=64)
def programs_of(inputs: Path, d: str, hub: str, binding: bool = True) -> tuple[tuple[Run, ...], tuple[Stop, ...]]:
    """The hub's programs as they ran around local date d (minutes from its midnight); `binding` False keeps the
    carrier-limited stops too."""
    advs = advisories.load(inputs, hub, midnight(d), 2 * 1440)
    if not binding:
        advs = tuple(replace(a, only=None) for a in advs)
    return advisories.as_ran(advs)


def in_force(runs: Iterable[Run], stops: Iterable[Stop]) -> list[tuple[int, int]]:
    """The intervals a program is in force: a GDP from the later of its signature and its period's start to its run's
    end, a stop piece over its window."""
    out = [(max(r.since, r.adv.start), min(r.until, r.adv.end + 1)) for r in runs]
    return sorted([(a, b) for a, b in out if b > a] + [(p.start, p.end) for p in stops])


def any_in_force(runs: Iterable[Run], stops: Iterable[Stop], lo: int, hi: int) -> bool:
    return any(a < hi and b > lo for a, b in in_force(runs, stops))


def first_hour(runs: Iterable[Run], stops: Iterable[Stop]) -> int | None:
    spans = in_force(runs, stops)
    return next((h for h in range(FIRST, LAST_START + 1) if any(a < 60 * h + 60 and b > 60 * h for a, b in spans)),
                None)


@functools.lru_cache(maxsize=3)
def month_frame(inputs: Path, derived: Path, m: str) -> pd.DataFrame:
    """v1's rows of the month with each one's TaxiOut beside it."""
    x = bts.read_ontime(bts.ontime_path(inputs, m), derived)
    return x.assign(TaxiOut=month_taxi(inputs, derived, m))


def month_taxi(inputs: Path, derived: Path, m: str) -> np.ndarray:
    """The month's TaxiOut in the zip's row order, which is read_ontime's, cached beside it under the zip's sha256."""
    path = bts.ontime_path(inputs, m)
    cached = derived / f"taxi_{fetch.sha256(path)[:16]}.parquet"
    if not cached.exists():
        with zipfile.ZipFile(path) as z, z.open(next(n for n in z.namelist() if n.endswith(".csv"))) as f:
            taxi = pd.read_csv(f, usecols=["TaxiOut"])
        derived.mkdir(parents=True, exist_ok=True)
        part = cached.with_name(f"{cached.name}.{os.getpid()}.part")
        taxi.to_parquet(part, index=False)
        part.replace(cached)
    return pd.read_parquet(cached).TaxiOut.to_numpy()


def months_of(d: str) -> list[str]:
    """The cached months that FlightDate d - 1 to d + 1 fall in."""
    return sorted({pack.shift(d, k)[:7] for k in (-1, 0, 1)} & set(MONTHS))


@functools.lru_cache(maxsize=6)
def frame(inputs: Path, derived: Path, d: str) -> pd.DataFrame:
    """Every carrier's flights of FlightDate d - 1 to d + 1 that the cache holds."""
    days = [pack.shift(d, k) for k in (-1, 0, 1)]
    x = pd.concat([p[p.FlightDate.isin(days)] for p in (month_frame(inputs, derived, m) for m in months_of(d))],
                  ignore_index=True)
    return x[x.sched_dep.notna()]


@functools.lru_cache(maxsize=6)
def taxi_median(inputs: Path, derived: Path, d: str) -> dict[str, float]:
    """Each origin's median taxi-out, every carrier, over the months d's frame falls in."""
    x = pd.concat([month_frame(inputs, derived, m) for m in months_of(d)], ignore_index=True)
    return x.groupby("Origin").TaxiOut.median().to_dict()


@functools.lru_cache(maxsize=len(MONTHS))
def month_fleet(inputs: Path, derived: Path, m: str) -> dict[str, fleet.Aircraft]:
    x = month_frame(inputs, derived, m)
    return typed(set(x[x.Reporting_Airline == CARRIER].Tail_Number.dropna()), b43_years(m), inputs)


def typed(tails: set[str], years: Sequence[int], inputs: Path) -> dict[str, fleet.Aircraft]:
    """fleet.fleet with the B-43 inventories given: every tail of `tails` they or the FAA registry can type."""
    out: dict[str, fleet.Aircraft] = {}
    seats_by_family: dict[str, list[int]] = {}
    for year in years:
        for r in csv.DictReader(io.StringIO(fleet.b43_path(inputs, year).read_text(encoding="utf-8-sig"))):
            seats = r["NUMBER_OF_SEATS"].strip()
            if r["CARRIER"] != CARRIER or seats in ("", "0", "0.00"):
                continue
            a = fleet.Aircraft(r["TAIL_NUMBER"].strip(), fleet.family(r["MODEL"]), r["MODEL"].strip(),
                               int(float(seats)), f"B-43 {year}")
            seats_by_family.setdefault(a.family, []).append(a.seats)
            if a.tail in tails and a.tail not in out:
                out[a.tail] = a
    missing = tails - set(out)
    if missing:
        median = {f: int(statistics.median(v)) for f, v in seats_by_family.items()}
        for tail, model in fleet.registry_models(fleet.registry_path(inputs), missing).items():
            if fleet.family(model) in median:
                out[tail] = fleet.Aircraft(tail, fleet.family(model), model, median[fleet.family(model)],
                                           "FAA registry")
    return out


def b43_years(m: str) -> list[int]:
    """fleet.b43_years over the published inventories: the two nearest the month, nearest first."""
    year, month = (int(v) for v in m.split("-"))
    at = year + (month - 0.5) / 12
    return sorted(B43_PUBLISHED, key=lambda y: (abs(y + 1 - at), y))[:2]


@functools.lru_cache(maxsize=len(MONTHS))
def month_loads(inputs: Path, m: str) -> dict:
    return fleet.load_factors(CARRIER, m, inputs)


@functools.lru_cache(maxsize=len(MONTHS))
def month_turns(inputs: Path, derived: Path, m: str) -> dict[str, int]:
    """Schedule turns per family on the month's calm days: those whose ATCSCC list is cached, whole, and shows no GDP
    or ground stop for JFK or BOS."""
    calm = [d for d in pack.month_days(m) if (v1.day_dir(inputs, d) / "list.html").exists() and not gap(d)
            and not any(e in HUBS and program_title(t) for e, t in advisories.listed(inputs, d))]
    x = month_frame(inputs, derived, m)
    return repairs.turns(x[(x.Reporting_Airline == CARRIER) & x.FlightDate.isin(calm)], month_fleet(inputs, derived, m))


@functools.lru_cache(maxsize=1)
def sources(inputs: Path) -> pack.Sources:
    return pack.Sources(inputs)


@dataclass
class Context:
    inputs: Path
    day: pack.Day
    x: pd.DataFrame
    families: dict[str, str]
    advs: dict[str, tuple[advisories.Advisory, ...]]
    truth: dict[str, tuple[tuple[Run, ...], tuple[Stop, ...]]]
    artcc: dict[str, str]
    covers: Covers


def context(inputs: Path, derived: Path, d: str, hub: str) -> Context:
    """The date's JetBlue day (v1's Day, schedule turns by split family), every carrier's flights around it with local
    times, the repaired lb and the taxi-out a hub ground stop held (`held_taxi`, repairs.held), and both hubs'
    programs as they ran."""
    base = midnight(d)
    raw = frame(inputs, derived, d)
    advs = {h: advisories.load(inputs, h, base, 2 * 1440) for h in HUBS}
    truth = {h: advisories.as_ran(advs[h]) for h in HUBS}
    x = pack.local(raw, base)
    codes = sorted(set(x.Origin) | set(x.Dest))
    artcc = {c: a for c in codes if (a := advisories.artcc(inputs, c))}
    covers = coverage(airports_of(codes), artcc)
    median = taxi_median(inputs, derived, d)
    removed = pd.Series(False, index=x.index)
    held = pd.Series(0, index=x.index)
    for r in x[x.Dest.isin(HUBS)].itertuples():
        runs, stops = truth[r.Dest]
        departs = int(r.ad) if r.Cancelled == 0 else int(r.sd)
        removed[r.Index] = (rbs.governing(runs, covers, r.Origin, int(r.sa), departs) is not None
                            or rbs.gs_held(stops, covers, r.Origin, int(r.sd)) is not None)
        if r.Cancelled == 0:
            held[r.Index] = repairs.held(departs, advisories.held_until(stops, covers, r.Origin, departs), r.TaxiOut,
                                         median[r.Origin])
    raw = raw.assign(held_taxi=held)
    x = x.assign(held_taxi=held, lb=repairs.new_lb(x, removed))
    ours = raw[raw.Reporting_Airline == CARRIER]
    near = raw[raw.Origin.isin(pack.CONSTRAINED) | raw.Dest.isin(pack.CONSTRAINED)]
    m = d[:7]
    aircraft = month_fleet(inputs, derived, m)
    day = pack.context(d, hub, dict(tuple(ours.groupby("FlightDate"))), dict(tuple(near.groupby("FlightDate"))),
                       aircraft, month_loads(inputs, m))
    families = {t: repairs.family(a) for t, a in aircraft.items()}
    day = replace(day, turn=month_turns(inputs, derived, m))
    return Context(inputs, day, x, families, advs, truth, artcc, covers)


def airports_of(codes: Iterable[str]) -> dict[str, Airport]:
    return {c: Airport.from_dict(c, a) for c, a in pack.airports(set(codes)).items()}


def build_candidate(d: str, hub: str, inputs: Path, derived: Path, e_fit: Mapping[str, float],
                    e_median: Mapping[str, float], scale: Mapping[str, float], cutoff: int = repairs.CUTOFF,
                    window: tuple[int, int] | None = None, split: str = "", bases: Sequence[str] = ()) -> Built:
    """The task of (d, hub) at its rule window (or at `window`, as for a control), or None and the reasons; `e_fit`
    and `e_median` are unseen.table's at `scale`; crew `bases` turn the crew model on (repairs.crews)."""
    ctx = context(inputs, derived, d, hub)
    row = {"date": d, "hub": hub, "p0": None, "window": None, "slide": None, "task_id": None, "reason": "",
           "tried": []}
    if window:
        tries = [(window, 0)]
    else:
        p0 = first_hour(*ctx.truth[hub])
        row["p0"] = p0
        if p0 is None:
            return Built(None, {**row, "reason": "no hub program in force 05:00-20:00"})
        tries = [(span(ctx.day, h), h - p0) for h in range(p0, min(p0 + SLIDES, LAST_START) + 1)]
    for w, slide in tries:
        if w is None:
            row["tried"].append({"slide": slide, "reason": "no length gives 30-80 tails"})
            continue
        sc = pack.scope(ctx.day, *w)
        if not sc.ok:
            row["tried"].append({"slide": slide, "window": list(w), "clauses": sc.clauses})
            continue
        task = assemble(ctx, sc, d, hub, e_fit, e_median, scale, cutoff, slide, split, bases)
        if window or task.clauses["program_in_window"]:
            return Built(task, {**row, "window": list(w), "slide": slide, "task_id": task.task_id,
                                "clauses": task.clauses})
        row["tried"].append({"slide": slide, "window": list(w), "clauses": task.clauses})
    return Built(None, {**row, "reason": "no window passes the data clauses"})


def span(day: pack.Day, h: int) -> tuple[int, int] | None:
    """[h, h + L) for the longest L of LENGTHS ending by midnight whose scope holds 30 to 80 tails."""
    for n in LENGTHS:
        if h + n <= DAY_END and pack.MIN_TAILS <= len(pack.scope(day, 60 * h, 60 * (h + n)).tails) <= pack.MAX_TAILS:
            return 60 * h, 60 * (h + n)
    return None


def assemble(ctx: Context, sc: pack.Scope, d: str, hub: str, e_fit: Mapping[str, float],
             e_median: Mapping[str, float], scale: Mapping[str, float], cutoff: int, slide: int, split: str,
             bases: Sequence[str] = ()) -> Task:
    day = ctx.day
    flights = repairs.flights(day, sc, ctx.x, ctx.families, tests(ctx), cutoff=cutoff, bases=bases)
    tails, closure = repairs.tails(day, sc, flights, ctx.families, day.turn)
    programs = hub_programs(ctx, flights, sc.start, e_fit, e_median, scale)
    codes = ({f.orig for f in flights} | {f.dest for f in flights} | set(HUBS)
             | {s for t in tails for s in (t.pos, t.end_req, t.actual_end)}
             | {a.orig for p in programs.values() for a in p.demand})
    held = sum(1 for f in flights if f.controlled or f.gs_held)
    clauses = {**sc.clauses, "program_in_window": held >= MIN_CONTROLLED, "controlled_or_held": held,
               "weather_share": round(sc.weather_share, 3), "tails": len(sc.tails), "closure": closure}
    task = Task(task_id=f"{CARRIER.lower()}-{hub.lower()}-{d.replace('-', '')}-{sc.window[0] // 60:02d}00",
                split=split, carrier=CARRIER, hub=hub, date=d, tz=day.tz, base=day.base, window=tuple(sc.window),
                start=sc.start, slide=slide, dmax=repairs.DMAX, cutoff=cutoff,
                turn={f: day.turn[f] for f in sorted({t.family for t in tails})}, airports=airports_of(codes),
                artcc={c: ctx.artcc[c] for c in sorted(codes) if c in ctx.artcc}, tails=tuple(tails),
                flights=tuple(flights), programs=programs, clauses=clauses, sources=pins(ctx, sc), bases=tuple(bases))
    return rbs.rescale(task, scale, e_fit, e_median)


def tests(ctx: Context) -> Callable[[str, str, int, int, int], tuple[str | None, str | None, bool]]:
    """repairs.flights' tests on the programs as they ran: (controlled, gs_held, exempt) of a leg."""
    covers = ctx.covers

    def test(orig: str, dest: str, sa: int, departs: int, sd: int) -> tuple[str | None, str | None, bool]:
        if dest not in HUBS:
            return None, None, False
        runs, stops = ctx.truth[dest]
        run = rbs.governing(runs, covers, orig, sa, departs)
        exempt = not any(covers(r.adv.scope, dest, orig) for r in runs) and not any(
            covers(p.scope, dest, orig) for p in stops)
        return run.adv.id if run else None, rbs.gs_held(stops, covers, orig, sd), exempt

    return test


def hub_programs(ctx: Context, flights: Sequence[Flight], start: int, e_fit: Mapping[str, float],
                 e_median: Mapping[str, float], scale: Mapping[str, float]) -> dict[str, Programs]:
    """Both hubs' programs over the flights' horizon: the advisories issued by then, the stops starting by then and
    every scheduled arrival over the task and the fittable periods, the segments left to rbs.rescale."""
    horizon = max((f.deadline + f.block for f in flights), default=start)
    ids = {(f.number, f.orig, f.sched_dep): f.id for f in flights}
    out = {}
    for h in HUBS:
        runs, stops = ctx.truth[h]
        kept = tuple(a for a in ctx.advs[h] if a.issued <= horizon)
        periods = [(a.start, a.end) for a in kept if unseen.fittable(a)]
        lo = min([0, start] + [p[0] for p in periods])
        hi = max([1440, horizon] + [p[1] for p in periods])
        out[h] = Programs(h, rbs.AAR[h], scale[h], e_median[h], {a.id: e_fit[a.id] for a in kept if a.id in e_fit},
                          kept, (), tuple(p for p in stops if p.start <= horizon),
                          arrivals(ctx.x, h, lo, hi, ids, runs, ctx.covers))
    return out


def pins(ctx: Context, sc: pack.Scope) -> dict:
    """The sha256 of each pinned input the task reads (an advisory day by its parsed records')."""
    src = sources(ctx.inputs)
    day = ctx.day
    months = sorted({pack.shift(day.date, k)[:7] for k in (-1, 0, 1)} & set(MONTHS))
    years = sorted({int(day.fleet[t].source.split()[-1]) for t in sc.tails if day.fleet[t].source.startswith("B-43")})
    out = {"bts_ontime": {m: src.file(bts.ontime_path(ctx.inputs, m)) for m in months},
           "t100": src.file(fleet.t100_path(ctx.inputs, day.date[:7])),
           "b43": {str(y): src.file(fleet.b43_path(ctx.inputs, y)) for y in years},
           "nasr": src.file(advisories.nasr_path(ctx.inputs)),
           "capacity": {h: src.file(rbs.profile_path(ctx.inputs, h)) for h in HUBS},
           "advisories": {f"{u}/{h}": src.advisories(u, h)
                          for u in pack.utc_days(day.base - advisories.LOOKBACK, day.base + 2 * 1440)
                          if (v1.day_dir(ctx.inputs, u) / "list.html").exists() for h in HUBS},
           "airportsdata": airportsdata.__version__}
    if any(day.fleet[t].source == "FAA registry" for t in sc.tails):
        out["faa_registry"] = src.file(fleet.registry_path(ctx.inputs))
    return out


def arrivals(x: pd.DataFrame, hub: str, lo: int, hi: int, ids: Mapping[tuple, int], runs: Sequence[Run],
             covers: Covers) -> tuple[Arrival, ...]:
    """Every BTS-scheduled arrival at `hub` with its scheduled arrival in [lo, hi], in RBS order."""
    at = x[(x.Dest == hub) & (x.sa >= lo) & (x.sa <= hi)]
    out = []
    for r in at.itertuples():
        num = int(r.Flight_Number_Reporting_Airline)
        departed = r.Cancelled == 0
        departs = int(r.ad) if departed else int(r.sd)
        run = rbs.governing(runs, covers, r.Origin, int(r.sa), departs)
        out.append(Arrival(r.Reporting_Airline, num, r.Origin, int(r.sd), int(r.sa), int(r.ad) if departed else None,
                           not departed, 0 if pd.isna(r.NASDelay) else int(r.NASDelay),
                           ids.get((num, r.Origin, int(r.sd))) if r.Reporting_Airline == CARRIER else None,
                           run.adv.id if run else None, bool(r.Diverted == 1)))
    return tuple(sorted(out, key=lambda a: a.key))


def events(dates: Iterable[str]) -> dict[str, int]:
    """Storm events: program dates chained within EVENT_GAP days, both hubs pooled."""
    out, last, k = {}, None, -1
    for d in sorted(set(dates)):
        x = Date.fromisoformat(d)
        if last is None or (x - last).days > EVENT_GAP:
            k += 1
        out[d], last = k, x
    return out


def season(d: str) -> str:
    return "summer" if d[5:7] in SUMMER else "winter" if d[5:7] in WINTER else "other"


def draw(qualifying: Sequence[tuple[str, str]], event: Mapping[str, int], seed: int) -> list[tuple[str, str]]:
    """The eval day-hubs: per hub 12, 8 summer + 4 winter (9 + 3 if winter has fewer than 4 events), each from a
    distinct event, events and day-hubs drawn with `seed`."""
    rng = random.Random(seed)
    out = []
    for hub in HUBS:
        by = {s: defaultdict(list) for s in ("summer", "winter")}
        for d, h in sorted(qualifying):
            if h == hub and season(d) in by:
                by[season(d)][event[d]].append((d, h))
        winter = min(EVAL_WINTER, len(by["winter"]))
        if winter < EVAL_WINTER:
            winter = min(EVAL_WINTER_SHORT, len(by["winter"]))
        for s, n in (("summer", min(EVAL_PER_HUB - winter, len(by["summer"]))), ("winter", winter)):
            order = sorted(by[s])
            rng.shuffle(order)
            out += [rng.choice(sorted(by[s][e])) for e in order[:n]]
    return sorted(out)


def split(qualifying: Sequence[tuple[str, str]], event: Mapping[str, int], seed: int) -> dict[tuple[str, str], str]:
    """eval for the draw, train for every qualifying day-hub of an event without an eval day-hub."""
    chosen = set(draw(qualifying, event, seed))
    taken = {event[d] for d, _ in chosen}
    return {k: "eval" if k in chosen else "train" for k in sorted(qualifying)
            if k in chosen or event[k[0]] not in taken}


def settle(drawn: Mapping[tuple[str, str], str], qualifying: Iterable[tuple[str, str]],
           event: Mapping[str, int]) -> dict[tuple[str, str], str]:
    """The split once later clauses have run: the drawn eval day-hubs that still qualify (one that no longer does is
    dropped, not replaced, so no day the draw's train fitted anything on can become eval), and train every qualifying
    day-hub of an event without a drawn eval day-hub."""
    taken = {event[d] for (d, _), v in drawn.items() if v == "eval"}
    return {k: "eval" if drawn.get(k) == "eval" else "train" for k in sorted(qualifying)
            if drawn.get(k) == "eval" or event[k[0]] not in taken}


def controls(inputs: Path, derived: Path, months: Sequence[str] = MONTHS) -> list[tuple[str, str, tuple[int, int]]]:
    """The rule-chosen controls as (date, hub, window): per season, the quiet date (no program listed, both neighbours
    loaded, no archive gap) with JetBlue's highest on-time rate, the earlier on a tie, at each hub."""
    loaded = {d for m in months for d in pack.month_days(m)}
    quiet = [d for d in sorted(loaded) if pack.shift(d, -1) in loaded and pack.shift(d, 1) in loaded and not gap(d)
             and not any(e in LISTED and program_title(t) for day in (d, pack.shift(d, 1))
                         for e, t in advisories.listed(inputs, day))]
    out = []
    for d in best_on_time(quiet, on_time(inputs, derived, months)):
        for hub in HUBS:
            busy = windows.movements(context(inputs, derived, d, hub).day.cur, hub)
            h = max(range(FIRST, DAY_END - CONTROL_HOURS + 1),
                    key=lambda a: (sum(busy.get(k, 0) for k in range(a, a + CONTROL_HOURS)), -a))
            out.append((d, hub, (60 * h, 60 * (h + CONTROL_HOURS))))
    return out


def best_on_time(dates: Iterable[str], rate: Mapping[str, float]) -> list[str]:
    """Per season, summer then winter, the date with the highest on-time rate, the earlier on a tie."""
    out = []
    for s in ("summer", "winter"):
        days = [d for d in dates if season(d) == s and d in rate]
        if days:
            out.append(min(days, key=lambda d: (-rate[d], d)))
    return out


def on_time(inputs: Path, derived: Path, months: Sequence[str] = MONTHS) -> dict[str, float]:
    """Per date, JetBlue's on-time rate over all its flights that day by DOT's A14 rule: arrived less than 15 minutes
    after its schedule; a cancelled or diverted flight is not on time."""
    out = {}
    for m in months:
        x = bts.read_ontime(bts.ontime_path(inputs, m), derived)
        x = x[x.Reporting_Airline == CARRIER]
        ok = (x.Cancelled == 0) & (x.Diverted == 0) & (x.ArrDelay < 15)
        out |= {d: round(float(v), 6) for d, v in ok.groupby(x.FlightDate).mean().items()}
    return dict(sorted(out.items()))


def e_table(inputs: Path, derived: Path, months: Sequence[str] = MONTHS, workers: int = 1) -> list[unseen.Fit]:
    """The in-sample fit of every fittable advisory of the months, a job per local date and hub."""
    jobs = [(d, hub, inputs, derived) for m in months for d in pack.month_days(m) for hub in HUBS]
    with ProcessPoolExecutor(workers) if workers > 1 else nullcontext() as pool:
        found = pool.map(fits_of, jobs, chunksize=max(1, len(jobs) // (4 * workers))) if pool else map(fits_of, jobs)
        return [f for fs in found for f in fs]


def fits_of(job: tuple) -> list[unseen.Fit]:
    """The in-sample e of every fittable GDP or proposed advisory signed on local date d at hub."""
    d, hub, inputs, derived = job
    base = midnight(d)
    signed = advisories.load(inputs, hub, base, 1439)
    advs = [a for a in signed if a.issued >= 0 and unseen.fittable(a)]
    if not advs:
        return []
    x = pack.local(frame(inputs, derived, d), base)
    codes = sorted(set(x.Origin) | {hub})
    covers = coverage(airports_of(codes), {c: a for c in codes if (a := advisories.artcc(inputs, c))})
    lo, hi = min(a.start for a in advs), max(a.end for a in advs)
    demand = arrivals(x, hub, lo, hi, {}, (), covers)
    stops = advisories.as_ran(signed)[1]
    return [unseen.in_sample(a, demand, covers, stops) for a in advs]


def build_all(keys: Sequence[tuple[str, str]], inputs: Path, derived: Path, e_fit: Mapping[str, float],
              e_median: Mapping[str, float], scale: Mapping[str, float], cutoff: int = repairs.CUTOFF,
              workers: int = 1, bases: Sequence[str] = ()) -> list[Built]:
    """build_candidate for every key, in a pool of `workers` processes, dates kept together for the frame caches."""
    jobs = [(d, hub, inputs, derived, dict(e_fit), dict(e_median), dict(scale), cutoff, tuple(bases))
            for d, hub in sorted(keys)]
    with ProcessPoolExecutor(workers) if workers > 1 else nullcontext() as pool:
        if not pool:
            return [build_one(j) for j in jobs]
        return list(pool.map(build_one, jobs, chunksize=max(1, math.ceil(len(jobs) / (2 * workers)))))


def build_one(job: tuple) -> Built:
    d, hub, inputs, derived, e_fit, e_median, scale, cutoff, bases = job
    return build_candidate(d, hub, inputs, derived, e_fit, e_median, scale, cutoff, bases=bases)
