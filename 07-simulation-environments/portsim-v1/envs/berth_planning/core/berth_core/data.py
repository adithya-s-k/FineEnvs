"""Port of Barcelona container calls, 2024 (open data, CC BY-SA 4.0; see data/barcelona/SOURCE.md)."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve()
DEFAULT_CSV = Path(os.environ.get(
    "BERTH_CALLS_CSV", _HERE.parents[4] / "data" / "barcelona" / "container_calls_2024.csv"))


@dataclass(frozen=True)
class Quay:
    code: str
    terminal: str
    first_section: int
    last_section: int
    section_m: float = 42.0   # median ship length / sections used, over all 2024 calls


QUAYS = {
    "36A": Quay("36A", "Terminal Catalunya (BEST)", 2, 30),
    "24B": Quay("24B", "APM Terminals Barcelona", 2, 22),
}


@dataclass(frozen=True)
class Call:
    call: str
    quay: str
    ship: str
    imo: str
    length_m: float
    beam_m: float | None
    draught_m: float | None
    first: int           # first section occupied (port numbering)
    last: int            # last section occupied
    eta: datetime        # UTC
    etd: datetime        # UTC
    from_port: str
    to_port: str


def _f(x: str) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_calls(path: Path | str = DEFAULT_CSV) -> list[Call]:
    out = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            a, b = (int(x) for x in r["sections"].split("-"))
            ts = lambda s: datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            out.append(Call(r["call"], r["quay"], r["ship"], r["imo"], float(r["length_m"]), _f(r["beam_m"]),
                            _f(r["draught_m"]), a, b, ts(r["eta_utc"]), ts(r["etd_utc"]), r["from_port"], r["to_port"]))
    return out
