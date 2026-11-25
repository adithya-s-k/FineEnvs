"""The FAA's programs at JetBlue's two hubs, parsed from the ATCSCC advisories as issued, and two views of them: as
issued by a watch (the forecast) and as they ran (the truth).

Each advisory whose control element is JFK or BOS becomes an Advisory in minutes from the task's local midnight: kind,
signature, period, program rate per hour (a single value applies to every hour; the rates count the hours of ARRIVALS
ESTIMATED FOR, the revision's own period), stated average and maximum delay, probability of extension, FLT INCL and the
departure scope: an ARTCC list, a distance in NM, listed airports and exemptions. Origins map to their ARTCC through the
FAA's NASR APT_BASE (one pinned 28-day cycle). A stop limited to one carrier (DAL, AAL, JetBlue's own requests), to
general aviation or to CAT I/II approaches never binds JetBlue and stays news; JetBlue's jets fly CAT III, so an
approach-category stop is read as not holding them. A GDP is limited only by its FLT INCL ("CARRIER DAL ONLY"): its
comments may report another carrier's stop. A bare "1ST TIER" or "2ND TIER" takes the ARTCCs the hub's
tier-labelled advisories list.

As they ran: each GDP advisory governs the departures from its signature until the next GDP advisory for its airport, a
GDP cancellation, or its period's end. A ground stop chains from its first issue through every later stop at the hub
signed before the chain's end (plus GS_GRACE, for extensions signed a few minutes late), each from its signature with
its own scope, to the last one's end; a cancellation cuts it at its event time, and a partial release drops only its
facilities. A GDP cancellation whose comment cancels the stops too ends the chain. Proposed GDPs never run. As
issued by t: the same over the advisories signed by t, where a proposed GDP signed after the last GDP is in force, so
the latest program runs to its as-issued end and no stop is extended.
"""

import csv
import functools
import io
import math
import re
import zipfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from ..build import advisories as v1
from ..build import bts, fetch, pack

HUBS = ("JFK", "BOS")
GDP, PROPOSED, GDP_CNX, GS, GS_CNX = "gdp", "gdp_proposed", "gdp_cnx", "gs", "gs_cnx"
NONCONTIGUOUS = frozenset({"ZAN", "ZHN", "ZSU", "ZUA"})
"""Alaska, Honolulu, San Juan and Guam: outside FLT INCL "CONTIGUOUS US"."""
TIERS = {1: frozenset({"ZBW", "ZDC", "ZNY", "ZOB"}),
         2: frozenset({"ZAU", "ZBW", "ZDC", "ZID", "ZJX", "ZMP", "ZNY", "ZOB", "ZTL"})}
"""The tiers JFK's and BOS's tier-labelled stops list (the same for both hubs in the 13 months)."""
GS_GRACE = 15
LOOKBACK = 720
"""Advisories signed this many minutes before local midnight still count: a program issued the evening before."""
NASR = "https://nfdc.faa.gov/webContent/28DaySub/extra/01_Oct_2026_APT_CSV.zip"
NM_PER_MILE = 0.868976
EARTH_MILES = 3958.8
LABEL = re.compile(r"^([A-Z][A-Z0-9 ,/()&'.-]*?):\s*(.*)$")
ARTCC = re.compile(r"\bZ[A-Z]{2}\b")
PLACE = re.compile(r"\b(C[A-Z]{3}|[A-Y][A-Z][A-Z0-9])\b")
WORDS = frozenset({"ALL", "AND", "CZY", "THE", "FOR", "NOT", "ARE", "DEP", "ARR", "GDP"})
CARRIERS = (("JBU", r"\bJBU\b|JETBLUE"), ("DAL", r"\bDAL\b|DELTA"), ("AAL", r"\bAAL\b|AMERICAN"), ("UAL", r"\bUAL\b"),
            ("GA", r"\bGA ONLY\b"))
STOPS_TOO = re.compile(r"\b(GS|GSP|STOPS?|GROUND ?STOPS?)\b")
STOPS_STAY = re.compile(r"REMAIN|STILL|REVISED|POSSIBLE|GROUND STOP FOR")


@dataclass(frozen=True)
class Scope:
    """Which departures a program covers: every origin, or those in `artccs`, within `nm` of the airport, or listed
    in `airports`; never one in `exempt` (airports or ARTCCs), nor outside the contiguous US when `contiguous`."""

    every: bool
    artccs: frozenset[str]
    nm: int | None
    airports: frozenset[str]
    exempt: frozenset[str]
    contiguous: bool

    @classmethod
    def from_dict(cls, d: Mapping) -> "Scope":
        return cls(d["every"], frozenset(d["artccs"]), d["nm"], frozenset(d["airports"]), frozenset(d["exempt"]),
                   d["contiguous"])

    def to_dict(self) -> dict:
        return {"every": self.every, "artccs": sorted(self.artccs), "nm": self.nm, "airports": sorted(self.airports),
                "exempt": sorted(self.exempt), "contiguous": self.contiguous}

    def covers(self, origin: str, artcc: str | None, nm: float | None) -> bool:
        if (self.contiguous and artcc in NONCONTIGUOUS) or origin in self.exempt or artcc in self.exempt:
            return False
        return (self.every or artcc in self.artccs or origin in self.airports
                or (self.nm is not None and nm is not None and nm <= self.nm))

    def without(self, released: frozenset[str]) -> "Scope":
        return replace(self, artccs=self.artccs - released, airports=self.airports - released,
                       exempt=self.exempt | released)

    @property
    def empty(self) -> bool:
        return not (self.every or self.artccs or self.airports or self.nm is not None)


@dataclass(frozen=True)
class Advisory:
    id: str
    hub: str
    kind: str
    issued: int
    start: int | None
    end: int | None
    rates: tuple[int, ...]
    avg_delay: int | None
    max_delay: int | None
    extension: str | None
    flt_incl: str
    scope: Scope | None
    only: str | None
    released: frozenset[str] | None
    stops_too: bool

    @classmethod
    def from_dict(cls, d: Mapping) -> "Advisory":
        return cls(d["id"], d["hub"], d["kind"], d["issued"], d["start"], d["end"], tuple(d["rates"]), d["avg_delay"],
                   d["max_delay"], d["extension"], d["flt_incl"], Scope.from_dict(d["scope"]) if d["scope"] else None,
                   d["only"], None if d["released"] is None else frozenset(d["released"]), d["stops_too"])

    def to_dict(self) -> dict:
        return {"id": self.id, "hub": self.hub, "kind": self.kind, "issued": self.issued, "start": self.start,
                "end": self.end, "rates": list(self.rates), "avg_delay": self.avg_delay,
                "max_delay": self.max_delay, "extension": self.extension, "flt_incl": self.flt_incl,
                "scope": self.scope.to_dict() if self.scope else None, "only": self.only,
                "released": None if self.released is None else sorted(self.released), "stops_too": self.stops_too}

    @property
    def binds(self) -> bool:
        return self.only is None

    def rate_at(self, minute: int) -> int:
        """The program rate of the hour holding `minute` (hours count from the one holding the period's start)."""
        return self.rates[minute // 60 - self.start // 60]


@dataclass(frozen=True)
class Run:
    """A GDP advisory in force: departures in [since, until) to its airport are governed by it."""

    adv: Advisory
    since: int
    until: int


@dataclass(frozen=True)
class Stop:
    """A piece of a ground-stop chain: departures to `hub` from origins `scope` covers wait during [start, end)."""

    chain: str
    hub: str
    start: int
    end: int
    scope: Scope

    @classmethod
    def from_dict(cls, d: Mapping) -> "Stop":
        return cls(d["chain"], d["hub"], d["start"], d["end"], Scope.from_dict(d["scope"]))

    def to_dict(self) -> dict:
        return {"chain": self.chain, "hub": self.hub, "start": self.start, "end": self.end,
                "scope": self.scope.to_dict()}


def load(inputs: Path, hub: str, base: int, hi: int) -> tuple[Advisory, ...]:
    """The hub's advisories signed from LOOKBACK minutes before local midnight (UTC epoch minute `base`) to `hi`, in
    signature order, every kind kept (the views pick theirs)."""
    out = []
    for day in pack.utc_days(base - LOOKBACK, base + hi):
        if (v1.day_dir(inputs, day) / "list.html").exists():
            out += [parse(r, hub, base) for r in v1.load_day(inputs, day, hub)]
    return tuple(sorted((a for a in out if -LOOKBACK <= a.issued <= hi), key=lambda a: (a.issued, a.id)))


def parse(record: Mapping, hub: str, base: int) -> Advisory:
    """One advisory of `hub` as v1.load_day reads it, with times in minutes from the local midnight at `base`."""
    signed = v1.signature(record["signature"])
    title = v1.title_of(record["header"])
    kind = v1.kind(title)
    f = fields(record["message"])
    said = ("FLT INCL",) if kind in (GDP, PROPOSED) else ("FLT INCL", "REMARKS", "COMMENTS", "REASON")
    head = " ".join([title, *(f.get(k, "") for k in said)])
    if title.endswith("EXPIRATION"):
        kind = "other"
    labels = (("ARRIVALS ESTIMATED FOR", "CUMULATIVE PROGRAM PERIOD", "ANTICIPATED CUMULATIVE PROGRAM PERIOD")
              if kind in (GDP, PROPOSED) else ("GROUND STOP PERIOD", "EVENT TIME") if kind == GS
              else ("GDP CNX PERIOD", "GS CNX PERIOD", "EVENT TIME"))
    start, end = span(next((f[k] for k in labels if f.get(k)), ""), signed, base)
    if kind in (GDP_CNX, GS_CNX) and start is None:
        start = minutes(signed, base)
    gs_delays = [int(v) for v in re.findall(r"\d+", f.get("NEW TOTAL, MAXIMUM, AVERAGE DELAYS", ""))]
    avg = number(f.get("AVERAGE DELAY") or f.get("ANTICIPATED AVERAGE DELAY"))
    mx = number(f.get("MAXIMUM DELAY") or f.get("ANTICIPATED MAXIMUM DELAY"))
    if len(gs_delays) == 3:
        mx, avg = gs_delays[1], gs_delays[2]
    released = None
    if kind == GS_CNX and f.get("RELEASED FACILITIES"):
        text = f["RELEASED FACILITIES"].upper()
        named = set(ARTCC.findall(text)) | ({"CZY"} if "CZY" in text else set())
        released = None if text.startswith("ALL") or not named else frozenset(named)
    comments = f.get("COMMENTS", "")
    return Advisory(
        id=f"adv-{record['day'].replace('-', '')}-{record['advn']:03d}", hub=hub, kind=kind,
        issued=minutes(signed, base), start=start, end=end,
        rates=rates(f.get("PROGRAM RATE") or f.get("ANTICIPATED PROGRAM RATE") or "", start, end),
        avg_delay=avg, max_delay=mx, extension=extension(f.get("PROBABILITY OF EXTENSION")),
        flt_incl=f.get("FLT INCL", ""), scope=scope(kind, f), only=only(title.upper(), head.upper()), released=released,
        stops_too=kind == GDP_CNX and bool(STOPS_TOO.search(comments)) and not STOPS_STAY.search(comments))


def fields(message: str) -> dict[str, str]:
    """LABEL: value lines, a following line without a colon continuing the value; a blank line ends it."""
    out: dict[str, str] = {}
    last = None
    for line in message.splitlines():
        m = LABEL.match(line.strip())
        if m and not line.startswith(" "):
            last = m.group(1)
            out[last] = m.group(2).strip()
        elif last and line.strip() and ":" not in line:
            out[last] = f"{out[last]} {line.strip()}".strip()
        elif not line.strip():
            last = None
    return out


def span(text: str, signed: datetime, base: int) -> tuple[int | None, int | None]:
    whole = v1.period(text, signed)
    if whole:
        return minutes(whole[0], base), minutes(whole[1], base)
    m = re.search(r"(\d{2})/(\d{2})(\d{2})", text)
    if not m:
        return None, None
    return minutes(v1.near(signed, int(m.group(1)), int(m.group(2)), int(m.group(3))), base), None


def rates(text: str, start: int | None, end: int | None) -> tuple[int, ...]:
    """One rate per hour of [start, end]: a single value for every hour, a short list extended by its last value."""
    values = [int(v) for v in re.findall(r"\d+", text.split()[0] if text.split() else "")]
    if not values or start is None or end is None:
        return tuple(values)
    hours = end // 60 - start // 60 + 1
    return tuple((values + [values[-1]] * hours)[:hours])


def scope(kind: str, f: Mapping[str, str]) -> Scope | None:
    if kind in (GDP, PROPOSED):
        text = f"{f.get('DEP SCOPE', '')} {f.get('ADDITIONAL DEP FACILITIES INCLUDED', '')}"
        canada, exempt = f.get("CANADIAN DEP ARPTS INCLUDED", ""), f.get("EXEMPT DEP FACILITIES", "")
    elif kind == GS:
        text = f.get("DEP FACILITIES INCLUDED") or f.get("FACILITIES INCLUDED") or "ALL"
        canada = exempt = ""
    else:
        return None
    upper = text.upper()
    label = re.match(r"\s*\(([^)]*)\)", upper)
    body = upper[label.end():] if label else upper
    artccs = set(ARTCC.findall(body))
    tier = re.search(r"(\d)\s*(?:ST|ND|RD|TH)\s*TIER", upper)
    if tier and not artccs:
        artccs = set(TIERS.get(int(tier.group(1)), ()))
    distance = re.match(r"\s*(\d+)\b", body)
    places = {p for p in PLACE.findall(f"{body} {canada.upper()}") if p not in WORDS and not ARTCC.fullmatch(p)}
    return Scope(every=body.strip().startswith("ALL"), artccs=frozenset(artccs),
                 nm=int(distance.group(1)) if distance else None, airports=frozenset(places),
                 exempt=frozenset(ARTCC.findall(exempt.upper())) | frozenset(PLACE.findall(exempt.upper())),
                 contiguous="CONTIGUOUS" in f.get("FLT INCL", "").upper())


def only(title: str, text: str) -> str | None:
    """Who a carrier-, GA- or approach-limited program applies to, else None."""
    flt = re.search(r"CARRIER ([A-Z]{3}) ONLY", text)
    if flt:
        return flt.group(1)
    if re.search(r"\bCAT\s*(I|1|II|2)\b", title):
        return "CAT"
    limited = re.search(r"\bONLY\b|MAINLINE|\bSUBS\b|AIRLINE REQUEST|CARRIER REQUEST|STAKEHOLDER REQUEST", text)
    for code, pattern in CARRIERS:
        if re.search(pattern, text) and (limited or code in ("JBU", "GA")):
            return code
    return None


def extension(text: str | None) -> str | None:
    if not text:
        return None
    word = text.split()[0].upper()
    return {"MED": "MEDIUM", "HI": "HIGH", "LO": "LOW"}.get(word, word)


def number(text: str | None) -> int | None:
    m = re.match(r"\s*(\d+)", text or "")
    return int(m.group(1)) if m else None


def minutes(when: datetime, base: int) -> int:
    return bts.epoch_minutes(when) - base


def as_ran(advs: Iterable[Advisory]) -> tuple[tuple[Run, ...], tuple[Stop, ...]]:
    """The truth: every GDP advisory's run (proposals never run) and the ground-stop pieces."""
    advs = list(advs)
    return gdp_runs([a for a in advs if a.kind != PROPOSED]), stop_pieces(advs)


def issued_by(advs: Iterable[Advisory], t: int) -> tuple[tuple[Run, ...], tuple[Stop, ...]]:
    """The programs as known at watch t: the advisories signed by t, a proposed GDP signed after the last GDP in
    force."""
    seen = [a for a in advs if a.issued <= t]
    last = max((a.issued for a in seen if a.kind == GDP), default=None)
    proposals = [a for a in seen if a.kind == PROPOSED and (last is None or a.issued > last)]
    return gdp_runs([a for a in seen if a.kind in (GDP, GDP_CNX)] + proposals[-1:]), stop_pieces(seen)


def gdp_runs(advs: list[Advisory]) -> tuple[Run, ...]:
    programs = sorted((a for a in advs if a.kind in (GDP, PROPOSED) and a.binds and a.rates and a.start is not None
                       and a.end is not None), key=lambda a: (a.issued, a.id))
    cancels = sorted(a.start for a in advs if a.kind == GDP_CNX)
    out = []
    for i, a in enumerate(programs):
        stops = [a.end + 1] + [b.issued for b in programs[i + 1:i + 2]] + [c for c in cancels if c >= a.issued]
        until = min(stops)
        if until > a.issued:
            out.append(Run(a, a.issued, until))
    return tuple(out)


def stop_pieces(advs: list[Advisory]) -> tuple[Stop, ...]:
    """Ground-stop chains as pieces of constant scope, from the binding stops and the cancellations among `advs`."""
    events = sorted((a for a in advs if a.binds and (a.kind in (GS, GS_CNX) or (a.kind == GDP_CNX and a.stops_too))),
                    key=lambda a: (a.issued, a.id))
    out: list[Stop] = []
    chain: list[Stop] = []
    end = None
    for a in events:
        if a.kind == GS and a.start is not None and a.end is not None:
            if chain and a.issued <= end + GS_GRACE:
                at = max(a.issued, chain[-1].start)
                chain[-1] = replace(chain[-1], end=at)
                chain.append(Stop(chain[0].chain, a.hub, at, a.end, a.scope))
            else:
                out += close(chain, end)
                chain = [Stop(a.id, a.hub, a.start, a.end, a.scope)]
            end = a.end
        elif chain and a.kind != GS and a.issued <= end + GS_GRACE:
            at = max(min(a.start, end), chain[-1].start)
            if a.released is None:
                out += close(chain, at)
                chain, end = [], None
            else:
                chain[-1] = replace(chain[-1], end=at)
                chain.append(Stop(chain[0].chain, a.hub, at, end, chain[-1].scope.without(a.released)))
    out += close(chain, end)
    return tuple(p for p in out if p.end > p.start and not p.scope.empty)


def close(chain: list[Stop], end: int | None) -> list[Stop]:
    return [replace(p, end=min(p.end, end)) for p in chain]


def held_until(stops: Iterable[Stop], covers: Callable[[Scope, str, str], bool], origin: str, d: int) -> int:
    """The first minute from `d` at which no stop piece covering `origin` holds a departure for its hub; `covers` as
    Task.covers."""
    for p in sorted(stops, key=lambda p: p.start):
        if p.start <= d < p.end and covers(p.scope, p.hub, origin):
            d = p.end
    return d


def nasr_path(inputs: Path) -> Path:
    return inputs / "faa" / "nasr" / NASR.rsplit("/", 1)[1]


def fetch_nasr(inputs: Path) -> Path:
    path = nasr_path(inputs)
    if not path.exists():
        fetch.save(path, fetch.get(NASR, timeout=600))
        fetch.record(inputs, str(path.relative_to(inputs)), NASR, bts.today())
    return path


@functools.cache
def stations(inputs: Path) -> dict[str, str]:
    """The responsible ARTCC of every NASR airport, by its FAA id and by its ICAO id."""
    with zipfile.ZipFile(nasr_path(inputs)) as z:
        rows = list(csv.DictReader(io.TextIOWrapper(z.open("APT_BASE.csv"), "latin-1")))
    out: dict[str, str] = {}
    for r in rows:
        if r["SITE_TYPE_CODE"] == "A" and r["RESP_ARTCC_ID"]:
            out.setdefault(r["ARPT_ID"], r["RESP_ARTCC_ID"])
            if r["ICAO_ID"]:
                out.setdefault(r["ICAO_ID"], r["RESP_ARTCC_ID"])
    return out


def artcc(inputs: Path, code: str) -> str | None:
    """A BTS (IATA) airport's ARTCC: by FAA id, else by the ICAO id airportsdata gives the IATA code."""
    found = stations(inputs)
    return found.get(code) or found.get(bts.airports().get(code, {}).get("icao", ""))


def nm(a: tuple[float, float], b: tuple[float, float]) -> float:
    p1, p2, dl = math.radians(a[0]), math.radians(b[0]), math.radians(b[1] - a[1])
    c = math.sin(p1) * math.sin(p2) + math.cos(p1) * math.cos(p2) * math.cos(dl)
    return EARTH_MILES * math.acos(max(-1.0, min(1.0, c))) * NM_PER_MILE


def listed(inputs: Path, day: str) -> list[tuple[str, str]]:
    """(element, title) of every advisory the day's ATCSCC list shows, details cached or not."""
    page = (v1.day_dir(inputs, day) / "list.html").read_text(encoding="iso-8859-1")
    out = []
    for m in re.finditer(r'href="/adv/adv_otherdis\?adv_date=\d{8}&advn=\d+">.*?</tr>', page, re.S):
        cells = [re.sub(r"\s+", " ", v1.strip(c)).strip() for c in re.findall(r"<td[^>]*>(.*?)</td>", m.group(0), re.S)]
        if len(cells) >= 3:
            out.append((cells[0].split("/")[0], cells[2]))
    return out
