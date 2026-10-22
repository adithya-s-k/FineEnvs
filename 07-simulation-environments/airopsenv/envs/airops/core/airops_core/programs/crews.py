"""Crews: pilots and flight attendants, their bases, reserves, duty rules and marginal costs, every number from
crew_params.json (each value there carries its public sources; nothing is fitted to JetBlue's or the FAA's
decisions).

A pilot unit is a captain and a first officer kept together, rated on one type group (the A320 family, the A220, the
E190). A cabin unit is a flight-attendant team kept together for the day on one type group, sized for the group's
largest complement (14 CFR 121.391 and JetBlue's reported staffing): every flight needs one unit of each kind (labelled
"pilot pairs" and "flight-attendant teams"). Bases are dated per group.

Pilots fly under 14 CFR 117, unaugmented, starting the day rested (cumulative and contract limits are not modelled):
- a leg may leave when the block-in expected at takeoff (its departure plus its scheduled block) is within the report
  plus Table B, for the report's local hour at the report station (one theater: the crew stays acclimated) and the
  period's segments counting this one, plus the 117.19 extension of 2 h (captain's agreement assumed, any delay
  unforeseen; available in every duty period, the once-per-rest limit on more than 30 min not tracked);
- the flight time so far (actual blocks) plus this leg's scheduled block is within Table A for the report's hour;
- a short-call reserve's flight duty period also ends by its availability period's start plus Table B + 4 h or 16 h,
  whichever is less, plus the extension (117.21(c)); an airport-standby reserve is on duty from its shift's start;
- a deadhead is duty, not a segment; a new duty period needs 10 h of rest from release. Flight attendants fly under
  121.467(b): a leg of the team's own trip is legal as scheduled, whatever its delay ((b)(14): every delay counted
  beyond the airline's control); any other leg or deadhead must be expected, at its departure, to release the team
  within 14 h of its report; 10 h of rest.

A trip reports REPORT before its first scheduled departure and its crew may leave once reported (the record's early
departures included); a called reserve, an airport-standby reserve and a crew starting a second period may leave from
BRIEF after reporting. A crew is released DEBRIEF after block-in and connects in CONNECT. Short-call reserves report on
a whole hour (labelled), LEAD after the call. Marginal costs, in passenger-minutes as cost.py converts dollars: deadhead
minutes at the crew's rate (the FAA's Table 4-7 crew rate for a pilot pair, BLS pay with its benefits load per
attendant-minute), each deadhead seat as a lost average fare, a night away from base at GSA's (CONUS) or DoD's (Puerto
Rico, the Virgin Islands) lodging and M&IE for the station and date (GSA's standard rate where neither lists the
station), and the end of the day per base (Ends).
"""

import functools
import json
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date as Date
from datetime import datetime
from pathlib import Path

import highspy
import numpy as np

from ..core.cost import FERRY_BASE_MIN, FERRY_MILES_PER_MIN, miles
from ..core.model import Airport
from .cost import MTOW

SOURCES = Path(__file__).with_name("crew_params.json")
TYPE = {"A320": "A32X", "A321": "A32X", "A321P": "A32X", "A220": "A220", "E190": "E190"}
GROUPS = ("A32X", "A220", "E190")
KINDS = ("pilot", "cabin")
BANDS = ("low", "middle", "high")
NAMES = {"pilot": "pilots", "cabin": "flight_attendants"}


@functools.cache
def params() -> dict:
    return json.loads(SOURCES.read_text())


def _hours(text: str) -> int:
    return int(text[:2])


def _minutes(v: float) -> int:
    return round(60 * v)


P = params()
_117, _121 = P["far117_pilots"], P["far121_flight_attendants"]
TABLE_A = {_hours(r["report_acclimated"]): _minutes(r["max_flight_time_h"])
           for r in _117["table_a_max_flight_time_unaugmented"]["value"]}
TABLE_B = {_hours(r["start_acclimated"]): tuple(_minutes(v) for v in r["max_fdp_h_by_segments"].values())
           for r in _117["table_b_max_fdp_unaugmented_lineholders"]["value"]}
EXTENSION = _minutes(_117["fdp_extension_before_takeoff"]["value"]["max_h"])
REST = _minutes(_117["rest_min_h"]["value"])
CABIN_REST = _minutes(_121["rest_min_h"]["value"])
CABIN_DUTY = _minutes(_121["scheduled_duty_max_h"]["value"])
_cap = re.search(r"\+ (\d+) h, (\d+) h", _117["reserve_short_call"]["value"]["rap_plus_fdp_cap_unaugmented"])
SHORT_PLUS, SHORT_MAX = 60 * int(_cap.group(1)), 60 * int(_cap.group(2))
_conv = P["duty_conventions"]
REPORT = _conv["report_before_first_departure_min"]["value"]["proposed_both"]
BRIEF = _conv["earliest_departure_after_report_min"]["value"]
DEBRIEF = _conv["release_after_block_in_min"]["value"]["middle"]
CONNECT = _conv["crew_connection_min"]["value"]
_res = P["reserves"]
RAP = {k: _minutes(_res["reserve_availability_period_h"]["value"][NAMES[k]]) for k in KINDS}
STANDBY = _minutes(_res["airport_standby_duration_h"]["value"]["middle"])
STANDBY_SHARE = _res["type_split"]["airport_standby_share_of_reserve_days"]["value"]["middle"]
_days = _res["same_day_conversion"]["value"]
ON_CALL = _days["reserve_days_per_month"] / int(re.search(r"\d+", _days["lineholder_duty_days_per_month"]).group())
SHARES = {k: _res[NAMES[k]]["band"]["value"] for k in KINDS}
_lead = _res["short_call_report_time"]
LEAD = {"pilot": _minutes(_lead["pilots_h"]["value"]["middle"]),
        "cabin": {b: _minutes(h) for b, h in _lead["flight_attendants_h"]["value"].items()}}
WINDOWS = tuple(tuple(int(t[:2]) * 60 + int(t[3:]) for t in w) for w in _res["short_call_windows"]["value"].values())
SHIFTS = tuple(int(t[:2]) * 60 + int(t[3:]) for t in _res["airport_standby_starts"]["value"])
TEAM = {g: max(n for f, n in P["fleet"]["proposed_fa_by_family"]["value"].items() if TYPE[f] == g) for g in GROUPS}
PILOTS = max(P["fleet"]["pilots_by_family"]["value"].values())
_costs = P["costs"]
_rate = _costs["pilot_rate_pax_min_per_crew_minute"]["value"]
PILOT_RATE = {g: next(v for k, v in _rate.items() if k.startswith(MTOW[f]))
              for f, g in TYPE.items()}
CABIN_RATE = _costs["fa_rate_proposal"]["value"]["pax_min_per_fa_minute"]
CPI_2025 = _costs["cpi_u"]["value"]["annual_2025"]
CPI = _costs["cpi_u"]["value"]["monthly"]
USD_PER_PAX_MINUTE = _costs["dollars_per_passenger_minute_2025"]["value"]
FARE = _costs["average_fare"]["value"]
GSA, DOD, STANDARD = (_costs[k]["value"] for k in ("gsa_conus", "dod_non_conus", "standard_conus"))


def _bases() -> dict[str, tuple[tuple[str, str], ...]]:
    b = P["bases"]
    out = {}
    for kind in KINDS:
        own = b[NAMES[kind]]
        core = re.findall(r"\b[A-Z]{3}\b", own["core"]["value"].split(":")[0])
        sju = own["SJU"]["value"]
        since = re.search(r"\d{4}-\d\d-\d\d", sju["proposed_pilots_open" if kind == "pilot" else
                                                 "flight_attendants_open"]).group()
        out[kind] = tuple((c, "") for c in core) + (("EWR", ""), ("SJU", since))
    return out


BASES = _bases()


def bases(kind: str, d: str) -> tuple[str, ...]:
    return tuple(b for b, since in BASES[kind] if not since or Date.fromisoformat(d) >= Date.fromisoformat(since))


def band(table: Mapping[int, object], hour: int):
    return table[max(h for h in table if h <= hour % 24)]


def fdp(hour: int, segments: int) -> int:
    """Table B's limit in minutes for a report in local `hour` and `segments` flight segments."""
    return band(TABLE_B, hour)[min(max(segments, 1), 7) - 1]


def flight_time(hour: int) -> int:
    """Table A's limit in minutes for a report in local `hour`."""
    return band(TABLE_A, hour)


def pilot_limit(report: int, hour: int, segments: int, rap: int | None = None, extend: bool = True) -> int:
    """The latest block-in expected at takeoff for a period's `segments`-th segment, with the extension if `extend`."""
    end = report + fdp(hour, segments)
    if rap is not None:
        end = min(end, rap + min(fdp(hour, segments) + SHORT_PLUS, SHORT_MAX))
    return end + EXTENSION * extend


def pilot_fits(report: int, hour: int, flown: int, minutes: int, dep: int, sched_block: int,
               rap: int | None = None, extend: bool = True) -> bool:
    return dep + sched_block <= pilot_limit(report, hour, flown + 1, rap, extend) and minutes + sched_block <= \
        flight_time(hour)


def cabin_fits(report: int, dep: int, sched_block: int, own: bool = False) -> bool:
    return own or dep + sched_block + DEBRIEF - report <= CABIN_DUTY


def rest(kind: str) -> int:
    return REST if kind == "pilot" else CABIN_REST


def size(kind: str, group: str) -> int:
    return PILOTS if kind == "pilot" else TEAM[group]


def lead(kind: str, base: str) -> int:
    v = LEAD[kind]
    return v if isinstance(v, int) else v[base]


def on_hour(t: int) -> int:
    return -(-t // 60) * 60


def rate(kind: str, group: str) -> float:
    """Passenger-minutes per minute of a deadheading unit."""
    return PILOT_RATE[group] if kind == "pilot" else CABIN_RATE * TEAM[group]


def _cpi(d: str) -> float:
    """CPI-U for d's month, else the latest published before it (BLS skipped October 2025)."""
    return CPI[max(k for k, v in CPI.items() if k <= d[:7] and v is not None)]


@functools.cache
def seat(d: str) -> float:
    """A deadhead seat as a lost average fare, in passenger-minutes."""
    usd = FARE.get(d[:4]) or FARE[max(y for y in FARE if y <= d[:4])]
    return usd * CPI_2025 / _cpi(d) / USD_PER_PAX_MINUTE


def value(usd: float, d: str) -> float:
    """Dollars of d's month in passenger-minutes (CPI-U to 2025, the FAA's $1.07)."""
    return usd * CPI_2025 / _cpi(d) / USD_PER_PAX_MINUTE


@functools.cache
def hotel(station: str, d: str) -> float:
    """A night's lodging and M&IE for one person at `station` on date d, in passenger-minutes."""
    return sum(_per_diem(station, d)) * CPI_2025 / _cpi(d) / USD_PER_PAX_MINUTE


@functools.cache
def lodging(station: str, d: str) -> float:
    """A night's lodging alone for one person at `station` on date d, in passenger-minutes."""
    return _per_diem(station, d)[0] * CPI_2025 / _cpi(d) / USD_PER_PAX_MINUTE


def _per_diem(station: str, d: str) -> tuple[float, float]:
    """The GSA (CONUS), DoD (PR, USVI) or GSA standard lodging and M&IE in dollars at `station` on date d."""
    md = d[5:]
    fy = str(int(d[:4]) + (d[5:7] >= "10"))
    if station in GSA:
        rows = GSA[station]["fy"].get(fy) or GSA[station]["fy"][max(GSA[station]["fy"])]
        s = next(s for s in rows["seasons"] if _season(s["from"], s["to"], md))
    elif station in DOD:
        versions = [v for v in DOD[station]["versions"] if _bulletin(v["bulletin_effective"]) <= d]
        rows = (versions[-1] if versions else DOD[station]["versions"][0])["rows"]
        s = next(s for s in rows if _season(s["season_from"], s["season_to"], md))
    else:
        s = STANDARD.get(fy) or STANDARD[max(STANDARD)]
    return s["lodging"], s["mie"]


def _season(lo: str, hi: str, md: str) -> bool:
    return lo <= md <= hi if lo <= hi else md >= lo or md <= hi


def _bulletin(text: str) -> str:
    return datetime.strptime(text, "%B %d, %Y").strftime("%Y-%m-%d")


def deadhead(kind: str, group: str, block: int, d: str) -> float:
    """A unit's deadhead of `block` minutes: its pay and its seats."""
    return rate(kind, group) * block + seat(d) * size(kind, group)


@dataclass(frozen=True)
class Unit:
    """A crew unit at the start: at `station`, free from `ready`; `report` is when its flight duty period began (None
    for a short-call reserve not yet called), `hour` that report's local hour at the report station, `flown` and
    `minutes` the segments and flight time (actual blocks) of the period so far, `landed` its last block-in in it,
    `at` where it reported. `trip` and `deadheads` are its
    planned flights from the start, `end` where its trip has it tonight, `home` its base. A reserve is "short" (its
    availability period starts at `rap`, it may be called until `until` and reports `lead` after the call) or
    "standby" (on duty at the airport from `report`, it may leave until `until`)."""

    id: str
    kind: str
    group: str
    size: int
    home: str
    end: str
    station: str
    ready: int
    report: int | None
    hour: int | None
    flown: int = 0
    minutes: int = 0
    landed: int | None = None
    at: str = ""
    trip: tuple[int, ...] = ()
    deadheads: tuple[int, ...] = ()
    reserve: str = ""
    rap: int | None = None
    until: int | None = None
    lead: int = 0

    def to_dict(self) -> dict:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in self.__dict__.items()}


Local = Callable[[str, int], int]


def local_hour(local: Local, station: str, minute: int) -> int:
    return (local(station, minute) // 60) % 24


def task_minute(local: Local, station: str, at: int) -> int:
    """The task's minute at which `station`'s clock reads `at` (minutes from the date's local midnight)."""
    return at - (local(station, at) - at)


def reserves(working: Mapping[tuple[str, str, str], int], d: str, local: Local, share: str = "middle") -> list[Unit]:
    """The date's reserves per kind, type group and base: the band's share of crews on reserve, as reserves on call
    per working crew (share / (1 - share) x reserve days / lineholder days) times the units based there, rounded;
    STANDBY_SHARE of them, rounded, on airport standby in SHIFTS taken in turn, the others on short call with their
    availability periods split between the AM and PM WINDOWS, spread evenly over each."""
    out = []
    for (kind, group, base), n in sorted(working.items()):
        if base not in bases(kind, d):
            continue
        s = SHARES[kind][share]
        k = round(n * s / (1 - s) * ON_CALL)
        standby = round(STANDBY_SHARE * k)
        short = k - standby
        people = size(kind, group)
        tag = f"R{kind[0].upper()}-{group}-{base}"
        for j in range(standby):
            start = task_minute(local, base, SHIFTS[j % len(SHIFTS)])
            out.append(Unit(f"{tag}-A{j}", kind, group, people, base, base, base, start + BRIEF, start,
                            local_hour(local, base, start), at=base, reserve="standby", until=start + STANDBY))
        for j in range(short):
            lo, hi = WINDOWS[j % 2]
            count = (short + 1 - j % 2) // 2
            rap = task_minute(local, base, lo + round((j // 2 + 0.5) * (hi - lo) / count))
            ld = lead(kind, base)
            out.append(Unit(f"{tag}-S{j}", kind, group, people, base, base, base, on_hour(rap + ld) + BRIEF, None,
                            None, at=base, reserve="short", rap=rap, until=rap + RAP[kind], lead=ld))
    return out


def ferry_block(a: Airport, b: Airport) -> float:
    return FERRY_BASE_MIN + miles(a, b) / FERRY_MILES_PER_MIN


def reposition(u: Unit, station: str, airports: Mapping[str, Airport], d: str, to: str | None = None) -> float:
    """A unit at `station` that should be at `to` (where its trip has it tonight by default): it deadheads there
    tomorrow (the ferry formula's block at its rate, and its seats), with a night's hotel away from its base."""
    to = u.end if to is None else to
    if station == to:
        return 0.0
    out = deadhead(u.kind, u.group, round(ferry_block(airports[station], airports[to])), d)
    return out + (hotel(station, d) * u.size if station != u.home else 0.0)


class Ends:
    """The end of the day per kind, type group and base: tomorrow the base's units take its trips again, so the units
    where the day leaves them go to where those trips have them tonight (each unit's `end`, the base's units together)
    at the least cost of `reposition`s (a transport problem); `least` is a unit's cheapest such move alone, a lower
    bound on its share."""

    def __init__(self, units: Sequence[Unit], airports: Mapping[str, Airport], d: str):
        self.airports, self.d = airports, d
        self.of = {u.id: (u.kind, u.group, u.home) for u in units}
        self.unit = {u.id: u for u in units}
        self.wanted: dict[tuple, Counter] = defaultdict(Counter)
        self.members: dict[tuple, list[str]] = defaultdict(list)
        for u in units:
            self.wanted[self.of[u.id]][u.end] += 1
            self.members[self.of[u.id]].append(u.id)
        self.memo: dict[tuple, float] = {}
        self.spreads: dict[tuple, float] = {}

    def move(self, u: Unit, station: str, to: str) -> float:
        key = (self.of[u.id], station, to)
        if key not in self.memo:
            self.memo[key] = reposition(u, station, self.airports, self.d, to)
        return self.memo[key]

    def least(self, u: Unit, station: str) -> float:
        return min(self.move(u, station, t) for t in self.wanted[self.of[u.id]])

    def cost(self, at: Mapping[str, str]) -> dict[str, float]:
        """Each unit's share of the end of the day with the units at `at` (unit id: station)."""
        out = {}
        by: dict[tuple, list[str]] = defaultdict(list)
        for uid in sorted(at):
            by[self.of[uid]].append(uid)
        for key, uids in by.items():
            have = Counter(at[uid] for uid in uids)
            need = Counter(self.wanted[key])
            stay = {s: min(have[s], need[s]) for s in have}
            have, need = have - Counter(stay), need - Counter(stay)
            u0 = self.unit[uids[0]]
            sent = transport(have, need, lambda s, t, u0=u0: self.move(u0, s, t))
            for uid in uids:
                s = at[uid]
                if stay.get(s, 0):
                    stay[s] -= 1
                    out[uid] = 0.0
                    continue
                t = next(t for (a, t), n in sent.items() if a == s and n > 0)
                sent[(s, t)] -= 1
                out[uid] = self.move(u0, s, t)
        return out

    def total(self, at: Mapping[str, str]) -> float:
        return sum(self.cost(at).values())

    def spread(self, key: tuple, stations: tuple[str, ...]) -> float:
        """The end of the day of one kind, type group and base whose units are at `stations` (sorted), all of them."""
        if (key, stations) not in self.spreads:
            have, need = Counter(stations), Counter(self.wanted[key])
            stay = have & need
            u0 = self.unit[self.members[key][0]]
            sent = transport(have - stay, need - stay, lambda s, t: self.move(u0, s, t))
            self.spreads[(key, stations)] = sum(n * self.move(u0, s, t) for (s, t), n in sent.items())
        return self.spreads[(key, stations)]


def transport(have: Mapping[str, int], need: Mapping[str, int], cost: Callable[[str, str], float],
              most: bool = False) -> dict[tuple[str, str], int]:
    """The least-cost flows from `have` to `need` (equal totals), by HiGHS; with `most`, totals may differ and the
    flows are the most there can be, then the least cost."""
    pairs = [(s, t) for s in sorted(have) for t in sorted(need)]
    if not pairs:
        return {}
    rows = {("s", s): n for n, s in enumerate(sorted(have))}
    rows |= {("t", t): len(rows) + n for n, t in enumerate(sorted(need))}
    level = np.array([have[s] for s in sorted(have)] + [need[t] for t in sorted(need)], dtype=float)
    costs = [cost(s, t) for s, t in pairs]
    big = 1 + 2 * max(map(abs, costs)) if most else 0.0
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", 1)
    h.addRows(len(level), np.zeros(len(level)) if most else level, level, 0, np.zeros(len(level), dtype=np.int32),
              np.array([], dtype=np.int32), np.array([]))
    for (s, t), c in zip(pairs, costs, strict=True):
        h.addCol(c - big, 0.0, highspy.kHighsInf, 2, np.array([rows[("s", s)], rows[("t", t)]], dtype=np.int32),
                 np.ones(2))
    h.run()
    x = h.getSolution().col_value
    return {p: int(round(v)) for p, v in zip(pairs, x, strict=True) if v > 0.5}
