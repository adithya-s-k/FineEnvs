"""Cut the committed fixture slice (core/tests/pack/fixtures/inputs) from a full cache/inputs: B6's legs on 2023-06-25
to 27 of eight tails that touched JFK on the 26th between 17:00 and 22:00, Delta's and American's JFK movements that
evening, two JFK advisories, two TAFs, two forecast discussions and the evening's METARs. N729JB is left out of B-43
so the FAA registry types it.

    python core/tests/pack/fixtures/make_fixture.py [cache/inputs]     # from envs/airops
"""

import csv
import io
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd

from airops_core.build import advisories, bts, fleet, weather

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent / "inputs"
TAILS = ["N249JB", "N337JB", "N229JB", "N306JB", "N763JB", "N729JB", "N976JT", "N972JT"]
REGISTRY_ONLY = "N729JB"
DAYS = ["2023-06-25", "2023-06-26", "2023-06-27"]
ADVISORIES = {"2023-06-26": [44, 46, 50]}
LIST_ROW = ('<tr>\n<td>\n<a\nhref="/adv/adv_otherdis?adv_date={mmddyyyy}&advn={advn}">{advn}</a>\n</td>\n'
            "<td>{element}</td>\n<td>{date}</td>\n<td>{title}</td>\n<td>{signed}</td>\n</tr>\n")
DETAIL = ('<html><body><table>\n<TR><TH class=header align="center" colspan=2>{header}</TH></TR>\n'
          "<TR><TD class=nam>MESSAGE:</TD><TD class=val><PRE>{message}</PRE></TD></TR>\n"
          "<TR><TD class=nam>EFFECTIVE TIME:</TD><TD class=val>{effective}</TD></TR>\n"
          "<TR><TD class=nam>SIGNATURE:</TD><TD class=val>{signature}</TD></TR>\n</table></body></html>\n")


def main(inputs: Path) -> None:
    shutil.rmtree(OUT, ignore_errors=True)
    ontime(inputs)
    references(inputs)
    atcscc(inputs)
    iem(inputs)
    log = {p: {"url": "https://example.invalid/fixture", "retrieved": "2026-10-13"}
           for p in ("bts/ontime_2023_06.zip", "bts/t100d_2023_06.csv", "faa/ReleasableAircraft.zip")}
    (OUT / "fetched.json").write_text(json.dumps(log, indent=1, sort_keys=True) + "\n")


def ontime(inputs: Path) -> None:
    df = bts.read_ontime(bts.ontime_path(inputs, "2023-06"), ROOT / "cache" / "derived")
    df = df[df.FlightDate.isin(DAYS)]
    ours = df[(df.Reporting_Airline == "B6") & df.Tail_Number.isin(TAILS)]
    midnight = pd.Timestamp("2023-06-26").tz_localize("America/New_York").tz_convert("UTC")
    base = bts.epoch_minutes(midnight.to_pydatetime())
    local = (df.sched_dep - base) // 60
    others = df[df.Reporting_Airline.isin(["DL", "AA"]) & (df.FlightDate == "2023-06-26")
                & ((df.Origin == "JFK") | (df.Dest == "JFK")) & local.between(16, 22)]
    rows = pd.concat([ours, others])[bts.STRINGS + bts.NUMBERS].sort_values(["FlightDate", "Reporting_Airline"])
    path = OUT / "bts" / "ontime_2023_06.zip"
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        info = zipfile.ZipInfo("On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2023_6.csv",
                               (2023, 9, 27, 5, 43, 0))
        z.writestr(info, rows.to_csv(index=False))


def references(inputs: Path) -> None:
    for year in fleet.b43_years("2023-06"):
        text = fleet.b43_path(inputs, year).read_text(encoding="utf-8-sig")
        reader = csv.DictReader(io.StringIO(text))
        keep = [r for r in reader if r["CARRIER"] == "B6" and r["TAIL_NUMBER"].strip() in TAILS
                and r["TAIL_NUMBER"].strip() != REGISTRY_ONLY]
        write_csv(fleet.b43_path(OUT, year), reader.fieldnames, keep)
    df = bts.read_ontime(bts.ontime_path(inputs, "2023-06"), ROOT / "cache" / "derived")
    legs = df[(df.Reporting_Airline == "B6") & df.Tail_Number.isin(TAILS) & df.FlightDate.isin(DAYS)]
    routes = set(zip(legs.Origin, legs.Dest, strict=True))
    reader = csv.DictReader(io.StringIO(fleet.t100_path(inputs, "2023-06").read_text(encoding="utf-8-sig")))
    keep = [r for r in reader if r["UNIQUE_CARRIER"] == "B6" and (r["ORIGIN"], r["DEST"]) in routes]
    write_csv(fleet.t100_path(OUT, "2023-06"), reader.fieldnames, keep)
    with zipfile.ZipFile(fleet.registry_path(inputs)) as z:
        master = [line for line in z.read("MASTER.txt").decode("utf-8-sig").splitlines()
                  if line.split(",")[0].strip() in ("N-NUMBER", REGISTRY_ONLY[1:])]
        code = master[1].split(",")[2].strip()
        ref = [line for line in z.read("ACFTREF.txt").decode("utf-8-sig").splitlines()
               if line.split(",")[0].strip() in ("CODE", code)]
        dereg = z.read("DEREG.txt").decode("utf-8-sig").splitlines()[:1]
    path = fleet.registry_path(OUT)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, lines in (("ACFTREF.txt", ref), ("MASTER.txt", master), ("DEREG.txt", dereg)):
            z.writestr(zipfile.ZipInfo(name, (2026, 10, 12, 22, 51, 0)), "\n".join(lines) + "\n")


def atcscc(inputs: Path) -> None:
    for day in ["2023-06-25", "2023-06-26", "2023-06-27", "2023-06-28"]:
        folder = advisories.day_dir(OUT, day)
        folder.mkdir(parents=True, exist_ok=True)
        rows = []
        for advn in ADVISORIES.get(day, []):
            src = advisories.day_dir(inputs, day) / f"{advn:04d}.html"
            rec = advisories.parse_detail(src.read_text(encoding="iso-8859-1")) if src.exists() else {
                "header": f"ATCSCC ADVZY {advn:03d} LGA/ZNY 06/26/2023 CDM GROUND DELAY PROGRAM",
                "message": "CTL ELEMENT: LGA", "effective": "261244-270459", "signature": "23/06/26 12:44"}
            element = rec["header"].split()[3]
            rows.append(LIST_ROW.format(mmddyyyy="06262023", advn=advn, element=element, date="06/26/23",
                                        title=advisories.title_of(rec["header"]), signed=rec["signature"]))
            (folder / f"{advn:04d}.html").write_text(DETAIL.format(**rec), encoding="iso-8859-1")
        (folder / "list.html").write_text("<html><body><table>\n" + "".join(rows) + "</table></body></html>\n",
                                          encoding="iso-8859-1")


def iem(inputs: Path) -> None:
    text = weather.metar_path(inputs, "JFK", "2023-06").read_text()
    lines = text.splitlines()
    keep = [lines[0]] + [line for line in lines[1:] if re.match(r"JFK,2023-06-2[67] ", line)
                         and "2023-06-26 12" <= line.split(",")[1] <= "2023-06-27 06"]
    path = weather.metar_path(OUT, "JFK", "2023-06")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(keep) + "\n")
    for pil, stamps in (("TAFJFK", ("261120", "261937")), ("AFDOKX", ("261318", "261821", "261959"))):
        raw = weather.afos_path(inputs, pil, "2023-06").read_text(errors="replace")
        chunks = [c for c in raw.split("\x01") if any(issued(c) == s for s in stamps)]
        if pil.startswith("AFD"):
            chunks = [trim_afd(c) for c in chunks]
        path = weather.afos_path(OUT, pil, "2023-06")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join("\x01" + c for c in chunks))


def issued(chunk: str) -> str:
    m = next((weather.WMO.match(line) for line in chunk.splitlines() if weather.WMO.match(line)), None)
    return "".join(m.groups()) if m else ""


def trim_afd(chunk: str) -> str:
    head = chunk.split(".SYNOPSIS")[0]
    section = re.search(r"^\.AVIATION\b.*?^&&\n", chunk, re.S | re.M)
    return head + (section.group(0) if section else "") + "\x03\n"


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    path.write_text(buf.getvalue())


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "cache" / "inputs")
