"""Aircraft and passengers: each tail's type and seats from BTS Form 41 Schedule B-43 (the carrier's own cabin
count), the FAA registry for a tail B-43 misses (type only, seats the family's median), and the load factor from T-100
Domestic Segment (route-month, else the carrier's month)."""

import csv
import io
import statistics
import zipfile
from dataclasses import dataclass
from pathlib import Path

from . import bts, fetch

B43_TABLE, T100_TABLE = "GEH", "FIM"
REGISTRY = "https://registry.faa.gov/database/ReleasableAircraft.zip"
T100_FIELDS = ["DEPARTURES_SCHEDULED", "DEPARTURES_PERFORMED", "SEATS", "PASSENGERS", "UNIQUE_CARRIER", "CARRIER",
               "ORIGIN", "DEST", "AIRCRAFT_TYPE", "YEAR", "MONTH", "CLASS"]
MIN_ROUTE_DEPARTURES = 10
"""A route-month with fewer departures performed takes the carrier's month load factor instead."""


@dataclass(frozen=True)
class Aircraft:
    tail: str
    family: str
    type: str
    seats: int
    source: str


def b43_path(inputs: Path, year: int) -> Path:
    return inputs / "bts" / f"b43_{year}.csv"


def t100_path(inputs: Path, month: str) -> Path:
    return inputs / "bts" / f"t100d_{month.replace('-', '_')}.csv"


def registry_path(inputs: Path) -> Path:
    return inputs / "faa" / "ReleasableAircraft.zip"


def fetch_b43(inputs: Path, year: int) -> Path:
    path = b43_path(inputs, year)
    if not path.exists():
        fetch.save(path, bts.transtats(B43_TABLE, year, "All"))
        fetch.record(inputs, str(path.relative_to(inputs)), bts.TRANSTATS.format(table=B43_TABLE), bts.today())
    return path


def fetch_t100(inputs: Path, month: str) -> Path:
    path = t100_path(inputs, month)
    if not path.exists():
        year, m = month.split("-")
        fetch.save(path, bts.transtats(T100_TABLE, int(year), str(int(m)), T100_FIELDS))
        fetch.record(inputs, str(path.relative_to(inputs)), bts.TRANSTATS.format(table=T100_TABLE), bts.today())
    return path


def fetch_registry(inputs: Path) -> Path:
    path = registry_path(inputs)
    if not path.exists():
        fetch.save(path, fetch.get(REGISTRY, timeout=900))
        fetch.record(inputs, str(path.relative_to(inputs)), REGISTRY, bts.today())
    return path


def b43_years(month: str) -> list[int]:
    """The two B-43 inventories (each as of 31 December) nearest the month, nearest first."""
    year, m = (int(v) for v in month.split("-"))
    at = year + (m - 0.5) / 12
    return sorted(range(year - 1, year + 2), key=lambda y: (abs(y + 1 - at), y))[:2]


def family(model: str) -> str:
    m = model.upper().replace(" ", "")
    if "A32" in m:
        return "A32X"
    if "190" in m:
        return "E190"
    if "A220" in m or "BD-500" in m:
        return "A220"
    return m


def fleet(carrier: str, month: str, tails: set[str], inputs: Path) -> dict[str, Aircraft]:
    """Every tail of `tails` that B-43 or the FAA registry can type; the rest are left out."""
    out: dict[str, Aircraft] = {}
    seats_by_family: dict[str, list[int]] = {}
    for year in b43_years(month):
        for r in csv.DictReader(io.StringIO(b43_path(inputs, year).read_text(encoding="utf-8-sig"))):
            seats = r["NUMBER_OF_SEATS"].strip()
            if r["CARRIER"] != carrier or seats in ("", "0", "0.00"):
                continue
            a = Aircraft(r["TAIL_NUMBER"].strip(), family(r["MODEL"]), r["MODEL"].strip(), int(float(seats)),
                         f"B-43 {year}")
            seats_by_family.setdefault(a.family, []).append(a.seats)
            if a.tail in tails and a.tail not in out:
                out[a.tail] = a
    missing = tails - set(out)
    if missing:
        median = {f: int(statistics.median(v)) for f, v in seats_by_family.items()}
        for tail, model in registry_models(registry_path(inputs), missing).items():
            if family(model) in median:
                out[tail] = Aircraft(tail, family(model), model, median[family(model)], "FAA registry")
    return out


def registry_models(path: Path, tails: set[str]) -> dict[str, str]:
    """The model of each tail in the registry's current (MASTER) or deregistered (DEREG) aircraft, current first."""
    found: dict[str, str] = {}
    with zipfile.ZipFile(path) as z:
        ref = {r[0].strip(): r[2].strip() for r in csv.reader(io.TextIOWrapper(z.open("ACFTREF.txt"), "utf-8-sig"))}
        for name in ("MASTER.txt", "DEREG.txt"):
            reader = csv.reader(io.TextIOWrapper(z.open(name), "utf-8-sig", errors="replace"))
            for r in reader:
                tail = "N" + r[0].strip()
                if tail in tails and tail not in found:
                    found[tail] = ref.get(r[2].strip(), "")
    return {t: m for t, m in found.items() if m}


def load_factors(carrier: str, month: str, inputs: Path) -> dict[tuple[str, str] | None, float]:
    """Passengers / seats by (origin, dest) for scheduled passenger service (CLASS F), and the month's under None."""
    year, m = (int(v) for v in month.split("-"))
    route: dict[tuple[str, str], list[float]] = {}
    for r in csv.DictReader(io.StringIO(t100_path(inputs, month).read_text(encoding="utf-8-sig"))):
        if r["UNIQUE_CARRIER"] != carrier or r["CLASS"] != "F" or (int(r["YEAR"]), int(r["MONTH"])) != (year, m):
            continue
        acc = route.setdefault((r["ORIGIN"], r["DEST"]), [0.0, 0.0, 0.0])
        acc[0] += float(r["SEATS"])
        acc[1] += float(r["PASSENGERS"])
        acc[2] += float(r["DEPARTURES_PERFORMED"])
    seats, pax = sum(v[0] for v in route.values()), sum(v[1] for v in route.values())
    out: dict[tuple[str, str] | None, float] = {None: pax / seats}
    out.update({k: v[1] / v[0] for k, v in route.items() if v[0] > 0 and v[2] >= MIN_ROUTE_DEPARTURES})
    return out


def route_loads(carrier: str, month: str, inputs: Path) -> dict[tuple[str, str], float]:
    """Passengers per departure performed by (origin, dest) for scheduled passenger service (CLASS F) in the month."""
    year, m = (int(v) for v in month.split("-"))
    acc: dict[tuple[str, str], list[float]] = {}
    for r in csv.DictReader(io.StringIO(t100_path(inputs, month).read_text(encoding="utf-8-sig"))):
        if r["UNIQUE_CARRIER"] != carrier or r["CLASS"] != "F" or (int(r["YEAR"]), int(r["MONTH"])) != (year, m):
            continue
        a = acc.setdefault((r["ORIGIN"], r["DEST"]), [0.0, 0.0])
        a[0] += float(r["PASSENGERS"])
        a[1] += float(r["DEPARTURES_PERFORMED"])
    return {k: v[0] / v[1] for k, v in acc.items() if v[1] > 0}


def pax(seats: int, lf: dict, orig: str, dest: str) -> int:
    return int(seats * lf.get((orig, dest), lf[None]) + 0.5)
