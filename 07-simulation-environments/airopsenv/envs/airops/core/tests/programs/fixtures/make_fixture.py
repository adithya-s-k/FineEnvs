"""Cut the committed NASR slice (core/tests/programs/fixtures/apt_base.csv) from the pinned APT_BASE: the airport id,
ICAO id and responsible ARTCC of every airport in v1's fixture slice of BTS (core/tests/pack/fixtures/inputs), so a
test can build a v2 task offline from that slice; and the TaxiOut of each of the slice's rows, in its order, from the
pinned month (core/tests/programs/fixtures/taxi_out.csv), which v1's slice left out.

    python core/tests/programs/fixtures/make_fixture.py [cache/inputs]     # from envs/airops
"""

import csv
import io
import sys
import zipfile
from pathlib import Path

import pandas as pd

from airops_core.build import bts
from airops_core.programs import advisories

ROOT = Path(__file__).resolve().parents[4]
SLICE = ROOT / "core" / "tests" / "pack" / "fixtures" / "inputs"
OUT = Path(__file__).resolve().parent / "apt_base.csv"
TAXI = Path(__file__).resolve().parent / "taxi_out.csv"
COLUMNS = ["SITE_TYPE_CODE", "ARPT_ID", "ICAO_ID", "RESP_ARTCC_ID"]
KEY = ["FlightDate", "Reporting_Airline", "Flight_Number_Reporting_Airline", "Origin", "Dest", "CRSDepTime"]


def main(inputs: Path) -> None:
    with zipfile.ZipFile(bts.ontime_path(SLICE, "2023-06")) as z:
        df = pd.read_csv(z.open(z.namelist()[0]), dtype=str)
    codes = set(df.Origin) | set(df.Dest)
    icao = {bts.airports()[c]["icao"] for c in codes}
    with zipfile.ZipFile(advisories.nasr_path(inputs)) as z:
        rows = [r for r in csv.DictReader(io.TextIOWrapper(z.open("APT_BASE.csv"), "latin-1"))
                if r["SITE_TYPE_CODE"] == "A" and (r["ARPT_ID"] in codes or r["ICAO_ID"] in icao)]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLUMNS, lineterminator="\n", extrasaction="ignore")
    w.writeheader()
    w.writerows(sorted(rows, key=lambda r: r["ARPT_ID"]))
    OUT.write_text(buf.getvalue())
    with zipfile.ZipFile(bts.ontime_path(inputs, "2023-06")) as z:
        month = pd.read_csv(z.open(next(n for n in z.namelist() if n.endswith(".csv"))), usecols=[*KEY, "TaxiOut"],
                            dtype=str)
    df[KEY].merge(month, on=KEY, how="left", validate="1:1")[["TaxiOut"]].to_csv(TAXI, index=False)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "cache" / "inputs")
