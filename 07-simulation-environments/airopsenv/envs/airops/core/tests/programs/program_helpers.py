"""What the programs tests share: the fixture paths, and advisory records and Advisory values by hand."""

from pathlib import Path

from airops_core.programs.advisories import Advisory, Scope

ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT / "tools"
CACHE = ROOT / "cache" / "inputs"
DERIVED = ROOT / "cache" / "derived"
SLICE = ROOT / "core" / "tests" / "pack" / "fixtures" / "inputs"
APT_BASE = Path(__file__).resolve().parent / "fixtures" / "apt_base.csv"
TAXI_OUT = Path(__file__).resolve().parent / "fixtures" / "taxi_out.csv"
DATE, HUB, WINDOW = "2023-06-26", "JFK", (1020, 1320)
EVERY = Scope(every=True, artccs=frozenset(), nm=None, airports=frozenset(), exempt=frozenset(), contiguous=False)


def record(advn: int, title: str, message: str, signature: str, day: str = DATE, element: str = "JFK/ZNY") -> dict:
    """A record as build.advisories.load_day reads it from the archive."""
    return {"day": day, "advn": advn, "url": "", "header": f"ATCSCC ADVZY {advn:03d} {element} 06/26/2023 {title}",
            "message": message.strip("\n"), "signature": signature, "effective": ""}


def adv(id: str, kind: str, issued: int, start: int | None, end: int | None, rates: tuple[int, ...] = (),
        scope: Scope | None = EVERY, only: str | None = None, released: frozenset | None = None,
        stops_too: bool = False, avg: int | None = None, hub: str = "JFK") -> Advisory:
    return Advisory(id, hub, kind, issued, start, end, rates, avg, None, None, "", scope if kind in ("gdp", "gs",
                    "gdp_proposed") else None, only, released, stops_too)
