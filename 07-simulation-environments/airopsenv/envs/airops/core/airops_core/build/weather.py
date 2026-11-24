"""Weather as issued, from the Iowa Environmental Mesonet: the routine METARs (ASOS archive), the TAFs and the NWS
Area Forecast Discussion's .AVIATION section (AFOS raw text), a month per file, turned into news at issue time."""

import csv
import io
import re
from datetime import UTC, datetime
from pathlib import Path

from . import bts, fetch
from .advisories import near

IEM = "https://mesonet.agron.iastate.edu"
ASOS = (IEM + "/cgi-bin/request/asos.py?station={station}&data=all&sts={sts}T00:00Z&ets={ets}T00:00Z&tz=Etc/UTC"
        "&format=onlycomma&latlon=no&missing=empty&trace=T&report_type=3")
AFOS = IEM + "/cgi-bin/afos/retrieve.py?pil={pil}&sdate={sts}&edate={ets}&fmt=text&limit=9999"
PAUSE = 1.0
WMO = re.compile(r"^[A-Z]{4}\d{2} [A-Z]{4} (\d{2})(\d{2})(\d{2})(?: [A-Z]{3})?$")
TAF_VALID = re.compile(r"\b(\d{2})(\d{2})/(\d{2})(\d{2})\b")
LEGEND = re.compile(r"Forecaster Confidence Levels:\s*Low - less than 30 percent\.\s*Medium - 30 to 60 percent\.\s*"
                    r"High - greater than 60 percent\.\s*")


def metar_path(inputs: Path, hub: str, month: str) -> Path:
    return inputs / "iem" / "metar" / f"{hub}_{month.replace('-', '_')}.csv"


def afos_path(inputs: Path, pil: str, month: str) -> Path:
    return inputs / "iem" / "afos" / f"{pil}_{month.replace('-', '_')}.txt"


def span(month: str) -> tuple[str, str]:
    """The month's first UTC day and the next month's: the archives' half-open request range."""
    year, m = (int(v) for v in month.split("-"))
    return f"{year}-{m:02d}-01", f"{year + m // 12}-{m % 12 + 1:02d}-01"


def metar_url(hub: str, month: str) -> str:
    sts, ets = span(month)
    return ASOS.format(station=hub, sts=sts, ets=ets)


def afos_url(pil: str, month: str) -> str:
    sts, ets = span(month)
    return AFOS.format(pil=pil, sts=sts, ets=ets)


def fetch_metars(inputs: Path, hub: str, month: str) -> Path:
    path = metar_path(inputs, hub, month)
    if not path.exists():
        fetch.save(path, fetch.get(metar_url(hub, month), pause=PAUSE))
        fetch.record(inputs, str(path.relative_to(inputs)), metar_url(hub, month), bts.today())
    return path


def fetch_afos(inputs: Path, pil: str, month: str) -> Path:
    path = afos_path(inputs, pil, month)
    if not path.exists():
        fetch.save(path, fetch.get(afos_url(pil, month), pause=PAUSE))
        fetch.record(inputs, str(path.relative_to(inputs)), afos_url(pil, month), bts.today())
    return path


def products(text: str, month: str) -> list[tuple[datetime, str]]:
    """The AFOS file's products as (issue time, text from the line after the PIL), oldest first."""
    year, m = (int(v) for v in month.split("-"))
    out = []
    for chunk in text.split("\x01"):
        lines = [line.rstrip() for line in chunk.replace("\x03", "").replace("\r", "").splitlines()]
        for i, line in enumerate(lines):
            w = WMO.match(line)
            if w:
                day, hour, minute = (int(v) for v in w.groups())
                body = "\n".join(lines[i + 2:]).strip("\n")
                out.append((datetime(year, m, day, hour, minute, tzinfo=UTC), body))
                break
    return sorted(out, key=lambda p: (p[0], p[1]))


def taf_news(text: str, month: str, hub: str, base: int, lo: int, hi: int) -> list[dict]:
    """The TAF in force at `lo` and every TAF issued in (lo, hi]."""
    items = []
    for issued, body in products(text, month):
        at = bts.epoch_minutes(issued) - base
        valid = TAF_VALID.search(body)
        data = {"station": f"K{hub}", "amendment": "AMD" in body.split("\n", 1)[0]}
        if valid:
            d1, h1, d2, h2 = (int(v) for v in valid.groups())
            data["valid"] = [bts.epoch_minutes(near(issued, d1, h1, 0)) - base,
                             bts.epoch_minutes(near(issued, d2, h2, 0)) - base]
        items.append({"id": f"taf-{hub.lower()}-{issued:%Y%m%d%H%M}", "at": at, "kind": "taf",
                      "speaker": f"NWS {hub} TAF", "text": body, "data": data})
    return in_force(items, lo, hi)


def outlook_news(text: str, month: str, office: str, speaker: str, base: int, lo: int, hi: int) -> list[dict]:
    """The forecast discussion's .AVIATION section in force at `lo`, and each later one whose text changed."""
    items, last = [], None
    for issued, body in products(text, month):
        section = aviation(body)
        if not section or section == last:
            continue
        last = section
        items.append({"id": f"afd-{office.lower()}-{issued:%Y%m%d%H%M}", "at": bts.epoch_minutes(issued) - base,
                      "kind": "outlook", "speaker": speaker, "text": section,
                      "data": {"office": office, "issued": issue_line(body)}})
    return in_force(items, lo, hi)


def metar_news(text: str, hub: str, base: int, lo: int, hi: int) -> list[dict]:
    out = []
    for r in csv.DictReader(io.StringIO(text)):
        observed = datetime.strptime(r["valid"], "%Y-%m-%d %H:%M").replace(tzinfo=UTC)
        at = bts.epoch_minutes(observed) - base
        if lo <= at <= hi and r["metar"].strip():
            out.append({"id": f"metar-{hub.lower()}-{observed:%Y%m%d%H%M}", "at": at, "kind": "metar",
                        "speaker": f"{hub} METAR", "text": r["metar"].strip(), "data": observation(r)})
    return out


def observation(r: dict) -> dict:
    vis = number(r["vsby"])
    layers = [(r[f"skyc{i}"], number(r[f"skyl{i}"])) for i in range(1, 5)]
    ceiling = min((h for c, h in layers if c in ("BKN", "OVC", "VV") and h is not None), default=None)
    return {"vis_sm": vis, "ceiling_ft": None if ceiling is None else int(ceiling), "wind_dir": number(r["drct"]),
            "wind_kt": number(r["sknt"]), "gust_kt": number(r["gust"]), "wx": r["wxcodes"].strip(),
            "category": category(vis, ceiling)}


def category(vis: float | None, ceiling: float | None) -> str:
    """FAA flight category from visibility (statute miles) and ceiling (feet)."""
    v = 99.0 if vis is None else vis
    c = 99999.0 if ceiling is None else ceiling
    if c < 500 or v < 1:
        return "LIFR"
    if c < 1000 or v < 3:
        return "IFR"
    if c <= 3000 or v <= 5:
        return "MVFR"
    return "VFR"


def aviation(body: str) -> str:
    """The .AVIATION section up to its "&&", without Boston's boilerplate legend of confidence levels."""
    m = re.search(r"^\.AVIATION\b.*?(?=^&&|\Z)", body, re.S | re.M)
    if not m:
        return ""
    text = LEGEND.sub("", "\n".join(line.rstrip() for line in m.group(0).strip().splitlines()))
    return re.sub(r"\n{3,}", "\n\n", text)


def issue_line(body: str) -> str:
    m = re.search(r"^\d{3,4} [AP]M [A-Z]{3,4} .+ \d{4}$", body, re.M)
    return m.group(0) if m else ""


def in_force(items: list[dict], lo: int, hi: int) -> list[dict]:
    before = [n for n in items if n["at"] <= lo]
    return before[-1:] + [n for n in items if lo < n["at"] <= hi]


def number(s: str) -> float | None:
    try:
        v = float(s)
    except (TypeError, ValueError):
        return None
    return int(v) if v.is_integer() else v
