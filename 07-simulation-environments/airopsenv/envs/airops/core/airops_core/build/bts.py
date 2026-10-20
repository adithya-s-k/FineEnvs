"""BTS Reporting Carrier On-Time Performance: the monthly zips, read into one row per scheduled flight with its times as
UTC epoch minutes; and the Transtats form downloads (Form 41 B-43, T-100 Domestic Segment).

Scheduled departure is FlightDate + CRSDepTime localised at the origin (airportsdata's IANA zone); every other time is
that plus minutes (CRSElapsedTime, DepDelay, ArrDelay or DivArrDelay), which keeps flights across midnight on the
right date."""

import functools
import http.cookiejar
import io
import os
import re
import urllib.parse
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import airportsdata
import numpy as np
import pandas as pd

from . import fetch

ONTIME = ("https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_"
          "{year}_{month}.zip")
TRANSTATS = "https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ={table}&QO_fu146_anzr=Nv4%20Pn44vr45"
STRINGS = ["FlightDate", "Reporting_Airline", "Tail_Number", "Flight_Number_Reporting_Airline", "Origin", "Dest",
           "CancellationCode", "Div1Airport", "Div2Airport", "CRSDepTime", "CRSArrTime", "DepTime"]
NUMBERS = ["DepDelay", "ArrDelay", "Cancelled", "Diverted", "CRSElapsedTime", "Distance", "CarrierDelay",
           "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay", "DivReachedDest", "DivArrDelay"]
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def ontime_path(inputs: Path, month: str) -> Path:
    return inputs / "bts" / f"ontime_{month.replace('-', '_')}.zip"


def ontime_url(month: str) -> str:
    year, m = month.split("-")
    return ONTIME.format(year=int(year), month=int(m))


def fetch_ontime(inputs: Path, month: str) -> Path:
    path = ontime_path(inputs, month)
    if not path.exists():
        fetch.save(path, fetch.get(ontime_url(month), timeout=900))
        fetch.record(inputs, str(path.relative_to(inputs)), ontime_url(month), today())
    return path


def transtats(table: str, year: int, period: str, fields: list[str] | None = None) -> bytes:
    """One Transtats DL_SelectFields download: the form's CSV, unzipped."""
    url = TRANSTATS.format(table=table)
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    page = fetch.get(url, opener=opener).decode("utf-8", "replace")
    form = dict(re.findall(r'<input type="hidden" name="([^"]+)" id="[^"]*" value="([^"]*)"', page))
    boxes = re.findall(r'type="checkbox" name="([A-Z0-9_]+)"', page)
    body = {k: v for k, v in form.items() if k.startswith("__")}
    body.update({"__EVENTTARGET": "", "__EVENTARGUMENT": "", "cboGeography": "All", "cboYear": str(year),
                 "cboPeriod": period, "btnDownload": "Download"})
    body.update({f: "on" for f in fields or boxes})
    action = re.search(r'<form method="post" action="([^"]+)"', page).group(1).replace("&amp;", "&")
    blob = fetch.get(urllib.parse.urljoin(url, action), data=urllib.parse.urlencode(body).encode(), opener=opener,
                     headers={"Content-Type": "application/x-www-form-urlencoded", "Referer": url}, timeout=900)
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        return z.read(next(n for n in z.namelist() if n.lower().endswith(".csv")))


def read_ontime(path: Path, derived: Path) -> pd.DataFrame:
    """The month's rows with UTC epoch minutes, cached as parquet beside the inputs under the zip's sha256. Each
    process writes its own temporary file and renames it into place, so workers starting on a cold cache don't move
    one another's file away."""
    cached = derived / f"ontime_{fetch.sha256(path)[:16]}.parquet"
    if cached.exists():
        return pd.read_parquet(cached)
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.endswith(".csv"))
        with z.open(name) as f:
            df = pd.read_csv(f, usecols=STRINGS + NUMBERS, dtype={c: str for c in STRINGS}, low_memory=False)
    df = with_utc(df[STRINGS + NUMBERS])
    derived.mkdir(parents=True, exist_ok=True)
    part = cached.with_name(f"{cached.name}.{os.getpid()}.part")
    df.to_parquet(part, index=False)
    part.replace(cached)
    return df


def with_utc(df: pd.DataFrame) -> pd.DataFrame:
    """Adds sched_dep, sched_arr, act_dep, act_arr as UTC epoch minutes (float, NaN when the flight has none)."""
    tz_of = timezones()
    local = pd.to_datetime(df.FlightDate) + pd.to_timedelta(hhmm(df.CRSDepTime), unit="m")
    sched_dep = pd.Series(np.nan, index=df.index)
    for tz, idx in df.groupby(df.Origin.map(tz_of)).groups.items():
        utc = local.loc[idx].dt.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert("UTC")
        sched_dep.loc[idx] = (utc - EPOCH).dt.total_seconds() / 60
    df = df.assign(sched_dep=sched_dep, sched_arr=sched_dep + df.CRSElapsedTime)
    reached = (df.Diverted == 1) & (df.DivReachedDest == 1)
    return df.assign(act_dep=df.sched_dep + df.DepDelay,
                     act_arr=df.sched_arr + df.ArrDelay.where(~reached, df.DivArrDelay))


def load(paths: list[Path], derived: Path) -> pd.DataFrame:
    return pd.concat([read_ontime(p, derived) for p in paths], ignore_index=True)


def hhmm(s: pd.Series) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce")
    return (v // 100) * 60 + v % 100


def timezones() -> dict[str, str]:
    return {code: a["tz"] for code, a in airports().items()}


@functools.cache
def airports() -> dict[str, dict]:
    return airportsdata.load("IATA")


def epoch_minutes(when: datetime) -> int:
    return int((when - datetime(1970, 1, 1, tzinfo=UTC)).total_seconds() // 60)


def today() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")
