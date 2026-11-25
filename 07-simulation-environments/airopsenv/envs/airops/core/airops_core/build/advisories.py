"""FAA ATCSCC advisories as issued: the archive's list for a UTC day, then each advisory whose control element is a
hub, kept as raw HTML under inputs/atcscc/<day>/ and turned into news at its signature time."""

import html
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

from . import bts, fetch

BASE = "https://www.fly.faa.gov"
LIST = (BASE + "/adv/adv_list?whichAdvisories=ATCSCC&advisoryCategory=All&date={day}"
        "&gDelay=true&gStop=true&airflow=true&ctop=true&route=true&other=true")
DETAIL = BASE + "/adv/adv_otherdis?adv_date={mmddyyyy}&advn={advn}"
SPEAKER = "FAA Command Center"
PAUSE = 0.4
FIELDS = {
    "program_period": r"(?<!ANTICIPATED )CUMULATIVE PROGRAM PERIOD:\s*(.+)",
    "proposed_period": r"ANTICIPATED CUMULATIVE PROGRAM PERIOD:\s*(.+)",
    "arrivals_for": r"ARRIVALS ESTIMATED FOR:\s*(.+)",
    "gs_period": r"GROUND STOP PERIOD:\s*(.+)",
    "cnx_period": r"(?:GDP|GS) CNX PERIOD:\s*(.+)",
    "event_time": r"EVENT TIME:\s*(.+)",
    "program_rate": r"(?:ANTICIPATED )?PROGRAM RATE:\s*(\S+)",
    "flt_incl": r"FLT INCL:\s*(.+)",
    "max_delay": r"(?:ANTICIPATED )?MAXIMUM DELAY:\s*(\d+)",
    "avg_delay": r"(?:ANTICIPATED )?AVERAGE DELAY:\s*(\d+)",
    "gs_delays": r"NEW TOTAL, MAXIMUM, AVERAGE DELAYS:\s*(.+)",
    "probability_of_extension": r"PROBABILITY OF EXTENSION:\s*(\S+)",
    "condition": r"IMPACTING CONDITION:\s*(.+)",
}
PERIODS = ("program_period", "proposed_period", "arrivals_for", "gs_period", "cnx_period", "event_time")
PERIOD = re.compile(r"(\d{2})/(\d{2})(\d{2})Z?\s*-\s*(?:(\d{2})/)?(\d{2})(\d{2})Z?")


def day_dir(inputs: Path, day: str) -> Path:
    return inputs / "atcscc" / day


def fetch_day(inputs: Path, day: str, elements: set[str]) -> list[dict]:
    """The day's list (re-read from the cache when present) and every listed advisory for `elements` not yet cached."""
    folder = day_dir(inputs, day)
    listing = folder / "list.html"
    if not listing.exists():
        fetch.save(listing, fetch.get(LIST.format(day=day), timeout=120, pause=PAUSE))
        fetch.record(inputs, str(listing.relative_to(inputs)), LIST.format(day=day), bts.today())
    rows = [r for r in parse_list(listing.read_text(encoding="iso-8859-1")) if r["element"] in elements]
    for r in rows:
        raw = folder / f"{r['advn']:04d}.html"
        if not raw.exists():
            fetch.save(raw, fetch.get(r["url"], timeout=120, pause=PAUSE))
    return rows


def parse_list(page: str) -> list[dict]:
    rows = []
    for m in re.finditer(r'href="/adv/adv_otherdis\?adv_date=(\d{8})&advn=(\d+)">.*?</tr>', page, re.S):
        cells = [re.sub(r"\s+", " ", strip(c)).strip() for c in re.findall(r"<td[^>]*>(.*?)</td>", m.group(0), re.S)]
        rows.append({"advn": int(m.group(2)), "element": cells[0].split("/")[0] if cells else "",
                     "url": DETAIL.format(mmddyyyy=m.group(1), advn=int(m.group(2)))})
    return sorted(rows, key=lambda r: r["advn"])


def parse_detail(page: str) -> dict:
    head = re.search(r"<TH class=header[^>]*>(.*?)</TH>", page, re.S | re.I)
    pre = re.search(r"<PRE>(.*?)</PRE>", page, re.S | re.I)
    out = {"header": re.sub(r"\s+", " ", strip(head.group(1))).strip() if head else "",
           "message": "\n".join(line.rstrip() for line in strip(pre.group(1)).strip("\n").splitlines()) if pre else ""}
    for key, label in (("effective", "EFFECTIVE TIME"), ("signature", "SIGNATURE")):
        m = re.search(label + r":.*?<TD class=val>(.*?)</TD>", page, re.S | re.I)
        out[key] = re.sub(r"\s+", " ", strip(m.group(1))).strip() if m else ""
    return out


def load_day(inputs: Path, day: str, element: str) -> list[dict]:
    """The hub's advisories of a UTC day as parsed records, in advisory-number order."""
    folder = day_dir(inputs, day)
    rows = [r for r in parse_list((folder / "list.html").read_text(encoding="iso-8859-1")) if r["element"] == element]
    return [{"day": day, "advn": r["advn"], "url": r["url"],
             **parse_detail((folder / f"{r['advn']:04d}.html").read_text(encoding="iso-8859-1"))} for r in rows]


def day_digest(records: list[dict]) -> str:
    return fetch.digest([{k: r[k] for k in ("advn", "header", "message", "signature", "effective")} for r in records])


def news(records: list[dict], hub: str, base: int, lo: int, hi: int) -> list[dict]:
    """News for the advisories signed in [lo, hi] (minutes from the hub's local midnight at UTC epoch minute `base`)."""
    out = []
    for r in records:
        signed = signature(r["signature"])
        at = bts.epoch_minutes(signed) - base
        if not lo <= at <= hi:
            continue
        title = title_of(r["header"])
        data = {"advzy": r["advn"], "element": hub, "kind": kind(title), "title": title}
        for key, pat in FIELDS.items():
            m = re.search(pat, r["message"])
            if not m:
                continue
            value = m.group(1).strip()
            if key in PERIODS:
                span = period(value, signed)
                if span:
                    data[key] = [bts.epoch_minutes(t) - base for t in span]
            elif key == "program_rate":
                data[key] = [int(v) for v in value.split("/") if v.isdigit()]
            elif key in ("max_delay", "avg_delay"):
                data[key] = int(value)
            else:
                data[key] = value
        out.append({"id": f"adv-{r['day'].replace('-', '')}-{r['advn']:03d}", "at": at, "kind": "advisory",
                    "speaker": SPEAKER, "text": f"{r['header']}\n{r['message']}".rstrip(), "data": data})
    return out


def kind(title: str) -> str:
    t = f" {title.upper()} "
    gdp = "GROUND DELAY" in t or " GDP " in t
    cnx = " CNX " in t or "CANCEL" in t
    if gdp and "PROPOSED" in t:
        return "gdp_proposed_cnx" if cnx else "gdp_proposed"
    if gdp:
        return "gdp_cnx" if cnx else "gdp"
    if "GROUND STOP" in t or " GS " in t:
        return "gs_cnx" if cnx else "gs"
    if "AFP" in t or "AIRSPACE FLOW" in t:
        return "afp"
    if "CTOP" in t:
        return "ctop"
    if "ROUTE" in t:
        return "route"
    return "other"


def title_of(header: str) -> str:
    m = re.match(r"ATCSCC ADVZY \d+ \S+ \S+ (.*)", header)
    return m.group(1).strip() if m else header


def signature(text: str) -> datetime:
    return datetime.strptime(text, "%y/%m/%d %H:%M").replace(tzinfo=UTC)


def period(text: str, signed: datetime) -> tuple[datetime, datetime] | None:
    """A DD/HHMMZ - [DD/]HHMMZ span, its days resolved to the month nearest the signature; an end without a day is
    the first such time after the start."""
    m = PERIOD.search(text)
    if not m:
        return None
    d1, h1, m1, h2, m2 = (int(m.group(i)) for i in (1, 2, 3, 5, 6))
    start = near(signed, d1, h1, m1)
    if m.group(4):
        return start, near(signed, int(m.group(4)), h2, m2)
    end = start.replace(hour=h2 % 24, minute=m2) + timedelta(days=h2 // 24)
    return start, end if end >= start else end + timedelta(days=1)


def near(signed: datetime, day: int, hour: int, minute: int) -> datetime:
    if hour == 24:
        hour, extra = 0, timedelta(days=1)
    else:
        extra = timedelta()
    options = []
    for shift in (-1, 0, 1):
        y, m = signed.year + (signed.month + shift - 1) // 12, (signed.month + shift - 1) % 12 + 1
        try:
            options.append(datetime(y, m, day, hour, minute, tzinfo=UTC) + extra)
        except ValueError:
            continue
    return min(options, key=lambda t: abs(t - signed))


def strip(s: str) -> str:
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    return html.unescape(s).replace("\xa0", " ")

