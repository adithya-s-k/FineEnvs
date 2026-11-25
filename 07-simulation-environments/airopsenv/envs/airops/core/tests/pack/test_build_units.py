import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from airops_core.build import advisories, bts, fleet, pack, weather, windows

INPUTS = Path(__file__).resolve().parent / "fixtures" / "inputs"
JFK_MIDNIGHT = bts.epoch_minutes(datetime(2023, 6, 26, 4, tzinfo=UTC))


def flights(rows: list[dict]) -> pd.DataFrame:
    defaults = {"DepDelay": 0.0, "ArrDelay": 0.0, "Diverted": 0.0, "DivReachedDest": np.nan, "DivArrDelay": np.nan}
    return pd.DataFrame([{**defaults, **r} for r in rows])


def test_utc_conversion_localises_at_the_origin_and_adds_minutes_across_midnight():
    df = bts.with_utc(flights([
        {"FlightDate": "2023-06-26", "Origin": "JFK", "CRSDepTime": "1930", "CRSElapsedTime": 390.0,
         "DepDelay": 30.0, "ArrDelay": 20.0},
        {"FlightDate": "2023-06-25", "Origin": "LAX", "CRSDepTime": "2359", "CRSElapsedTime": 330.0},
        {"FlightDate": "2022-12-16", "Origin": "BOS", "CRSDepTime": "0700", "CRSElapsedTime": 200.0},
        {"FlightDate": "2023-06-26", "Origin": "BOS", "CRSDepTime": "1000", "CRSElapsedTime": 70.0,
         "DepDelay": 5.0, "ArrDelay": np.nan, "Diverted": 1.0, "DivReachedDest": 1.0, "DivArrDelay": 95.0},
    ]))
    local = (df[["sched_dep", "sched_arr", "act_dep", "act_arr"]] - JFK_MIDNIGHT).to_numpy()
    assert local[0].tolist() == [1170, 1560, 1200, 1580]
    assert local[1, :2].tolist() == [179, 509]
    assert df.sched_dep[2] == bts.epoch_minutes(datetime(2022, 12, 16, 12, tzinfo=UTC))
    assert local[3].tolist() == [600, 670, 605, 765]


def test_hourly_disruption_weighs_weather_and_nas_cancellations_and_delay_minutes():
    day = pd.DataFrame([
        {"Origin": "JFK", "Dest": "BOS", "sd": 1030, "sa": 1100, "Cancelled": 1, "CancellationCode": "C",
         "WeatherDelay": np.nan, "NASDelay": np.nan},
        {"Origin": "BOS", "Dest": "JFK", "sd": 960, "sa": 1050, "Cancelled": 0, "CancellationCode": np.nan,
         "WeatherDelay": 10.0, "NASDelay": 30.0},
        {"Origin": "JFK", "Dest": "MCO", "sd": 1090, "sa": 1250, "Cancelled": 1, "CancellationCode": "A",
         "WeatherDelay": np.nan, "NASDelay": np.nan},
        {"Origin": "BOS", "Dest": "MCO", "sd": 1030, "sa": 1200, "Cancelled": 1, "CancellationCode": "B",
         "WeatherDelay": np.nan, "NASDelay": np.nan},
    ])
    assert windows.hourly(day, "JFK") == {17: windows.CANCEL_WEIGHT + 40.0, 18: 0.0}
    assert windows.movements(day, "JFK") == {17: 2, 18: 1}


def test_the_window_is_the_shortest_holding_most_of_the_disruption():
    disruption = {10: 50.0, 17: 5000.0, 18: 4000.0, 19: 3000.0, 20: 100.0}
    assert windows.choose(disruption, {}, lambda a, b: True) == (17, 20, 12000.0)
    assert windows.choose(disruption, {}, lambda a, b: b - a >= 4) == (17, 21, 12100.0)
    assert windows.choose(disruption, {}, lambda a, b: False) is None
    assert windows.choose({}, {14: 9, 6: 1}, lambda a, b: True)[:2] == (12, 15)


def test_the_spans_start_on_the_hour_from_5_to_20_and_end_by_midnight():
    spans = windows.spans()
    assert len(spans) == 61 and spans[0] == (5, 8) and (20, 23) in spans and (21, 24) not in spans
    assert all(3 <= b - a <= 6 and 5 <= a <= 20 and b <= 24 for a, b in spans)


def test_the_candidate_window_is_the_proven_one_of_most_leverage():
    def row(w0, w1, naive, proven=True):
        return {"window": [w0 * 60, w1 * 60], "naive": naive, "hindsight": 100, "proven": proven}

    rows = [row(9, 12, 115), row(14, 20, 130), row(13, 18, 140, proven=False), row(14, 17, 130), row(12, 18, 130)]
    assert windows.most_leverage(rows)["window"] == [720, 1080]
    assert windows.most_leverage(rows[1:2] + rows[3:4])["window"] == [840, 1020]
    assert windows.gap(rows[0]) == 0.15
    assert windows.most_leverage(rows[2:3]) is None


def test_caps_are_the_rbs_share_of_realized_movements_never_below_the_scope():
    rows = []
    for k in range(10):
        b6 = k < 4
        rows.append({"Origin": "JFK", "Dest": "BOS", "Reporting_Airline": "B6" if b6 else "DL", "sd": 1030 + k,
                     "sa": 1100, "ad": 1035 + k if k < 6 else 1090 + k, "aa": np.nan})
    around = pd.DataFrame(rows)
    dec = around.loc[[0, 1]]
    sc = pack.Scope((1080, 1260), 1020, ["N1"], 0, True, dec, around.iloc[:0], {})
    day = pack.Day("2023-06-26", "JFK", "America/New_York", 0, *[None] * 3, around, {}, {}, {}, 45)
    caps, nominal = pack.make_caps(day, sc)
    assert caps["JFK:dep"]["17"] == max(2, math.floor(0.4 * 6 + 0.5) - 2)
    assert caps["JFK:dep"]["18"] == 0
    assert nominal["JFK:dep"]["17"] == 2 and nominal["JFK:dep"]["18"] == 0
    assert set(caps) == {"BOS:arr", "BOS:dep", "JFK:arr", "JFK:dep"}
    assert list(caps["JFK:dep"]) == [str(h) for h in range(17, 24)]


def test_a_cancelled_flight_takes_the_75th_percentile_of_nearby_departures():
    pool = pd.DataFrame({"Origin": ["JFK"] * 5 + ["BOS"], "sd": [600, 620, 650, 700, 900, 600],
                         "lb_exo": [10.0, 20.0, 30.0, 40.0, 500.0, 99.0]})
    assert pack.lift(pool, "JFK", 640) == 33
    assert pack.lift(pool, "BOS", 1200) == 99
    assert pack.lift(pool, "MCO", 600) == 0


def test_the_fixture_registry_types_a_tail_b43_misses():
    a = fleet.fleet("B6", "2023-06", {"N729JB", "N249JB"}, INPUTS)
    assert a["N249JB"].source.startswith("B-43")
    assert (a["N729JB"].source, a["N729JB"].family, a["N729JB"].type) == ("FAA registry", "A32X", "A320-232")
    assert fleet.b43_years("2022-12") == [2022, 2021] and fleet.b43_years("2024-01") == [2023, 2024]


def test_load_factors_by_route_with_the_month_as_fallback():
    lf = fleet.load_factors("B6", "2023-06", INPUTS)
    assert 0.5 < lf[None] < 1
    route = next(k for k in lf if k)
    assert fleet.pax(162, lf, *route) == int(162 * lf[route] + 0.5)
    assert fleet.pax(100, lf, "XXX", "YYY") == int(100 * lf[None] + 0.5)


def test_advisories_keep_the_hub_and_turn_periods_into_local_minutes():
    records = advisories.load_day(INPUTS, "2023-06-26", "JFK")
    assert [r["advn"] for r in records] == [46, 50]
    news = advisories.news(records, "JFK", JFK_MIDNIGHT, 0, 2000)
    gdp = news[1]
    assert (gdp["id"], gdp["at"], gdp["speaker"]) == ("adv-20230626-050", 551, "FAA Command Center")
    assert gdp["data"]["kind"] == "gdp" and gdp["data"]["program_period"] == [720, 1439]
    assert gdp["data"]["program_rate"][-1] == 50 and gdp["text"].startswith("ATCSCC ADVZY 050 JFK/ZNY")
    assert advisories.news(records, "JFK", JFK_MIDNIGHT, 540, 2000) == news[1:]


@pytest.mark.parametrize(("text", "end"), [("26/1650 - 1800", (26, 18, 0)), ("26/2333Z - 27/0100Z", (27, 1, 0)),
                                          ("26/2300 - 0100", (27, 1, 0))])
def test_advisory_periods(text, end):
    _, stop = advisories.period(text, advisories.signature("23/06/26 16:50"))
    assert (stop.day, stop.hour, stop.minute) == end


def test_weather_news_in_force_at_start_then_as_issued():
    taf = (INPUTS / "iem/afos/TAFJFK_2023_06.txt").read_text()
    both = weather.taf_news(taf, "2023-06", "JFK", JFK_MIDNIGHT, 500, 2000)
    assert [n["at"] for n in both] == [440, 937] and both[1]["data"]["amendment"]
    assert both[1]["data"]["valid"] == [960, 2640] and both[0]["speaker"] == "NWS JFK TAF"
    assert [n["at"] for n in weather.taf_news(taf, "2023-06", "JFK", JFK_MIDNIGHT, 950, 2000)] == [937]
    afd = (INPUTS / "iem/afos/AFDOKX_2023_06.txt").read_text()
    out = weather.outlook_news(afd, "2023-06", "OKX", "NWS New York forecast discussion", JFK_MIDNIGHT, 600, 2000)
    assert [n["at"] for n in out] == [558, 861, 959]
    assert all(n["text"].startswith(".AVIATION") and "&&" not in n["text"] for n in out)
    metars = weather.metar_news((INPUTS / "iem/metar/JFK_2023_06.csv").read_text(), "JFK", JFK_MIDNIGHT, 800, 900)
    assert [(n["at"], n["speaker"], n["data"]["category"]) for n in metars] == [(831, "JFK METAR", "VFR"),
                                                                                (891, "JFK METAR", "VFR")]


def test_flight_category():
    assert weather.category(10, None) == "VFR"
    assert weather.category(4, 5000) == "MVFR"
    assert weather.category(10, 800) == "IFR"
    assert weather.category(0.5, 5000) == "LIFR"
