"""JetBlue's B6-to-B6 inbound transfers at its connecting stations, per period, from the DOT's Origin and Destination
Survey: DB1B Market (10% ticket sample, quarterly, to 2025 Q2) and DB1C Market (OD40, 40% sample, monthly, from July
2025). Both are domestic itineraries only (US points, Puerto Rico and the USVI included).

    python tools/survey_shares.py 2025Q2=market_b6_2025_2.csv 2025-07=market_202507.parquet ... [--out PATH]

A DB1B Market CSV (pre-filtered to markets with a B6 segment or not) or a DB1C Market parquet per period. Each market's
AirportGroup and OpCarrierGroup split into segments; a B6-operated segment departing a hub counts as an inbound
transfer from `orig` when the market's segment before it is B6-operated too and flies orig to that hub (a surface leg,
'--', is no connection). Passengers weight every row (a count in DB1B, 1 per passenger in DB1C). Counts are the
sample's, unscaled.

Rows (`tasks/pax/b6-transfers.csv`): per period and hub, one row per destination with orig `*` (B6 departing passengers
hub to dest and the inbound transfers among them), one per origin and destination with transfers (departing repeated,
the route's), and the hub's total with orig and dest `*`. Rows (`tasks/pax/b6-dwell.csv`): DB1C's B6-to-B6 inbound
transfers at each hub by scheduled dwell (DwellTimeGroup, the minutes from the inbound coupon's arrival to the
departure) in BIN-minute bins [lo, hi) up to MAX_DWELL, per month and pooled over the months as period `all`. Rows
(`tasks/pax/b6-fares.csv`): per period, the passenger-weighted median one-way market fare, in dollars of the period's
year, over the directional markets with a B6-operated segment, and the sample passengers it is taken over. DB1B: MktFare
(ItinYield x MktMilesFlown, the itinerary fare prorated to the market by miles), bulk fares (BulkFare = 1) left out;
BTS bases DB1B fares on "the total ticket value which consists of the price charged by the airlines plus any additional
taxes and fees levied by an outside entity at the time of purchase" (https://www.transtats.bts.gov/AverageFare/).
DB1C: MktAmount, the market's amount paid per passenger "inclusive of taxes and fees" (DB1C Market record description);
it has no bulk-fare flag. Both therefore include taxes and fees. In both, fares under MIN_FARE (award and zero-fare
tickets) are left out.

Sources: BTS, Airline Origin and Destination Survey, https://www.bts.gov/topics/airlines-and-airports/origin-and-destination-survey-data;
DB1B https://transtats.bts.gov/PREZIP/Origin_and_Destination_Survey_DB1BMarket_{YYYY}_{Q}.zip; DB1C
https://ostrapeispubdownloadprod.blob.core.windows.net/ostrapeis-pub-download-prod/ond40/products/db1c_public/DB1C.MARKET.{YYYYMM}.{DDMONYYYY}.zip
(the release suffix from https://www.bts.gov/topics/airlines-and-airports/origin-and-destination-survey-data-market).
"""

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

HUBS = ("JFK", "BOS", "FLL", "MCO", "SJU")
COLUMNS = ["AirportGroup", "OpCarrierGroup", "Passengers"]
OUT = Path(__file__).resolve().parents[1] / "tasks" / "pax" / "b6-transfers.csv"
DWELL_OUT = OUT.with_name("b6-dwell.csv")
FARES_OUT = OUT.with_name("b6-fares.csv")
BIN, MAX_DWELL = 15, 360
MIN_FARE = 10.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("inputs", nargs="+", help="PERIOD=PATH, a DB1B Market CSV or a DB1C Market parquet")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--dwell-out", type=Path, default=DWELL_OUT)
    ap.add_argument("--fares-out", type=Path, default=FARES_OUT)
    args = ap.parse_args()
    rows, dwell, fares = [], [], []
    for item in args.inputs:
        period, path = item.split("=", 1)
        source = "DB1C" if path.endswith(".parquet") else "DB1B"
        by = markets(Path(path))
        rows += period_rows(period, source, by)
        if source == "DB1C":
            dwell += dwell_rows(period, by)
        fares.append((period, source, *median_fare(Path(path))))
    out = pd.DataFrame(rows, columns=["period", "source", "hub", "orig", "dest", "departing", "transfers"])
    out = out.sort_values(["period", "hub", "dest", "orig"], kind="stable")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    hubs = out[(out.orig == "*") & (out.dest == "*")]
    for r in hubs.itertuples():
        print(f"{r.period} {r.hub}: {r.transfers}/{r.departing} = {r.transfers / r.departing:.2%}")
    print(f"{len(out)} rows -> {args.out}")
    if dwell:
        d = pd.DataFrame(dwell, columns=["source", "period", "hub", "lo", "hi", "transfers"])
        pooled = d.groupby(["hub", "lo", "hi"], as_index=False).transfers.sum().assign(source="DB1C", period="all")
        d = pd.concat([d, pooled[d.columns]], ignore_index=True).sort_values(["period", "hub", "lo"], kind="stable")
        d.to_csv(args.dwell_out, index=False)
        print(f"{len(d)} rows -> {args.dwell_out}")
    f = pd.DataFrame(fares, columns=["period", "source", "median_fare", "passengers"])
    f = f.sort_values("period", kind="stable")
    f.to_csv(args.fares_out, index=False)
    for r in f.itertuples():
        print(f"{r.period} {r.source}: median fare ${r.median_fare:.2f} over {r.passengers} passengers")


def markets(path: Path) -> pd.Series:
    """Passengers per (AirportGroup, OpCarrierGroup, DwellTimeGroup) over the markets with a B6-operated segment;
    DB1B has no dwell, its DwellTimeGroup is empty."""
    if path.suffix == ".parquet":
        parts = []
        f = pq.ParquetFile(path)
        for g in range(f.num_row_groups):
            t = f.read_row_group(g, columns=[*COLUMNS, "DwellTimeGroup"])
            t = t.filter(pc.match_substring(t["OpCarrierGroup"], "B6"))
            parts.append(t.to_pandas())
        df = pd.concat(parts, ignore_index=True)
    else:
        df = pd.read_csv(path, usecols=COLUMNS, dtype={"AirportGroup": str, "OpCarrierGroup": str})
        df = df[df.OpCarrierGroup.str.contains("B6", na=False)].assign(DwellTimeGroup="")
    keys = ["AirportGroup", "OpCarrierGroup", "DwellTimeGroup"]
    return df.fillna({"DwellTimeGroup": ""}).groupby(keys).Passengers.sum()


def median_fare(path: Path) -> tuple[float, int]:
    """The passenger-weighted median market fare over the markets with a B6-operated segment, and the passengers."""
    if path.suffix == ".parquet":
        parts = []
        f = pq.ParquetFile(path)
        for g in range(f.num_row_groups):
            t = f.read_row_group(g, columns=["OpCarrierGroup", "Passengers", "MktAmount"])
            t = t.filter(pc.match_substring(t["OpCarrierGroup"], "B6"))
            parts.append(t.to_pandas())
        df = pd.concat(parts, ignore_index=True).rename(columns={"MktAmount": "fare"})
    else:
        df = pd.read_csv(path, usecols=["OpCarrierGroup", "Passengers", "MktFare", "BulkFare"],
                         dtype={"OpCarrierGroup": str})
        df = df[df.OpCarrierGroup.str.contains("B6", na=False) & (df.BulkFare != 1)].rename(columns={"MktFare": "fare"})
    df = df[df.fare >= MIN_FARE].sort_values("fare", kind="stable")
    cum = df.Passengers.cumsum()
    return round(float(df.fare[cum >= cum.iloc[-1] / 2].iloc[0]), 2), int(round(cum.iloc[-1]))


def period_rows(period: str, source: str, by: pd.Series) -> list[tuple]:
    dep: dict[tuple[str, str], float] = defaultdict(float)
    tr: dict[tuple[str, str, str], float] = defaultdict(float)
    for (group, carriers, _), pax in by.items():
        ap, op = group.split(":"), carriers.split(":")
        if len(ap) != len(op) + 1:
            continue
        for i, c in enumerate(op):
            if c != "B6" or ap[i] not in HUBS:
                continue
            dep[(ap[i], ap[i + 1])] += pax
            if i and op[i - 1] == "B6":
                tr[(ap[i], ap[i - 1], ap[i + 1])] += pax
    rows = []
    for hub in HUBS:
        dests = sorted(d for h, d in dep if h == hub)
        if not dests:
            continue
        total = sum(dep[(hub, d)] for d in dests)
        moved = sum(v for (h, _, _), v in tr.items() if h == hub)
        rows.append((period, source, hub, "*", "*", round(total), round(moved)))
        for d in dests:
            rows.append((period, source, hub, "*", d, round(dep[(hub, d)]),
                         round(sum(v for (h, _, dd), v in tr.items() if h == hub and dd == d))))
            rows += [(period, source, hub, o, d, round(dep[(hub, d)]), round(v))
                     for (h, o, dd), v in sorted(tr.items()) if h == hub and dd == d and round(v) > 0]
    return rows


def dwell_rows(period: str, by: pd.Series) -> list[tuple]:
    """B6-to-B6 inbound transfers at each hub by the inbound coupon's scheduled dwell, in [0, MAX_DWELL)."""
    n: dict[tuple[str, int], float] = defaultdict(float)
    for (group, carriers, dwells), pax in by.items():
        ap, op, dw = group.split(":"), carriers.split(":"), dwells.split(":")
        if len(ap) != len(op) + 1 or len(dw) != len(op):
            continue
        for i in range(1, len(op)):
            if op[i] == "B6" and op[i - 1] == "B6" and ap[i] in HUBS and dw[i - 1].isdigit() \
                    and int(dw[i - 1]) < MAX_DWELL:
                n[(ap[i], int(dw[i - 1]) // BIN)] += pax
    return [("DB1C", period, hub, b * BIN, (b + 1) * BIN, round(n[(hub, b)]))
            for hub in HUBS for b in range(MAX_DWELL // BIN)]


if __name__ == "__main__":
    main()
