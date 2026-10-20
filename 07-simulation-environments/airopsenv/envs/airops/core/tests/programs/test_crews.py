"""The crew rules and costs by hand: 14 CFR 117 Tables A and B, the extension, the short-call cap, standby, rest,
121.467 with (b)(14), the reserves per base, and the marginal costs from crew_params.json's sources."""

from collections import Counter

import pytest

from airops_core.core.model import Airport
from airops_core.programs import crews

TABLE_B = {
    "0000-0359": (9, 9, 9, 9, 9, 9, 9), "0400-0459": (10, 10, 10, 10, 9, 9, 9),
    "0500-0559": (12, 12, 12, 12, 11.5, 11, 10.5), "0600-0659": (13, 13, 12, 12, 11.5, 11, 10.5),
    "0700-1159": (14, 14, 13, 13, 12.5, 12, 11.5), "1200-1259": (13, 13, 13, 13, 12.5, 12, 11.5),
    "1300-1659": (12, 12, 12, 12, 11.5, 11, 10.5), "1700-2159": (12, 12, 11, 11, 10, 9, 9),
    "2200-2259": (11, 11, 10, 10, 9, 9, 9), "2300-2359": (10, 10, 10, 9, 9, 9, 9)}


@pytest.mark.parametrize("row", sorted(TABLE_B))
def test_table_b_every_row_and_segment_band(row):
    first, last = int(row[:2]), int(row[5:7])
    for hour in range(first, last + 1):
        for segments, hours in enumerate(TABLE_B[row], start=1):
            assert crews.fdp(hour, segments) == round(60 * hours)
        assert crews.fdp(hour, 9) == crews.fdp(hour, 7)


@pytest.mark.parametrize("hour,limit", [(0, 480), (4, 480), (5, 540), (12, 540), (19, 540), (20, 480), (23, 480)])
def test_table_a(hour, limit):
    assert crews.flight_time(hour) == limit


def test_extension_two_hours_past_table_b():
    report, hour = 300, 5
    assert crews.pilot_limit(report, hour, 1) == 300 + 720 + 120
    assert crews.pilot_fits(report, hour, 0, 0, 1140 - 60, 60)
    assert not crews.pilot_fits(report, hour, 0, 0, 1141 - 60, 60)


def test_segments_shrink_table_b():
    assert crews.pilot_fits(600, 10, 4, 0, 600 + 750 + 120 - 60, 60)
    assert not crews.pilot_fits(600, 10, 4, 0, 600 + 750 + 121 - 60, 60)


def test_table_a_judged_on_the_flight_time_so_far_and_the_scheduled_block():
    assert crews.pilot_fits(600, 10, 2, 480, 900, 60)
    assert not crews.pilot_fits(600, 10, 2, 481, 900, 60)
    assert not crews.pilot_fits(60, 1, 0, 421, 120, 60)


def test_short_call_cap_from_the_availability_period():
    rap, report, hour = 120, 600, 10
    assert crews.pilot_limit(report, hour, 1) == 600 + 840 + 120
    assert crews.pilot_limit(report, hour, 1, rap) == 120 + 960 + 120
    early = crews.pilot_limit(300, 5, 7, 240)
    assert early == min(300 + 630, 240 + 630 + 240) + 120


def test_standby_is_duty_from_its_shift():
    shift = crews.task_minute(lambda s, m: m, "JFK", 300)
    u = crews.Unit("R", "pilot", "A32X", 2, "JFK", "JFK", "JFK", shift + crews.BRIEF, shift, 5, at="JFK",
                   reserve="standby", until=shift + crews.STANDBY)
    assert u.report == 300 and u.until == 660
    assert crews.pilot_limit(u.report, u.hour, 1) == 300 + 720 + 120


def test_flight_attendants_fourteen_hours_and_b14():
    assert crews.cabin_fits(0, 840 - 15 - 60, 60)
    assert not crews.cabin_fits(0, 840 - 15 - 59, 60)
    assert crews.cabin_fits(0, 2000, 60, own=True)


def test_rest_ten_hours():
    assert crews.rest("pilot") == 600 and crews.rest("cabin") == 600


def test_bases_dated():
    assert crews.bases("pilot", "2025-02-04") == ("BOS", "JFK", "FLL", "MCO", "LAX", "EWR")
    assert crews.bases("pilot", "2025-03-01")[-1] == "SJU"
    assert crews.bases("cabin", "2024-12-11") == ("BOS", "JFK", "FLL", "MCO", "LAX", "EWR")
    assert crews.bases("cabin", "2024-12-12")[-1] == "SJU"


def test_unit_sizes():
    assert crews.size("pilot", "E190") == 2
    assert (crews.size("cabin", "A32X"), crews.size("cabin", "A220"), crews.size("cabin", "E190")) == (4, 3, 2)


def test_reserves_per_base_at_each_band():
    local = lambda s, m: m  # noqa: E731
    working = Counter({("pilot", "A32X", "JFK"): 100, ("cabin", "A32X", "JFK"): 100, ("pilot", "A32X", "SJU"): 10})
    mid = crews.reserves(working, "2024-08-21", local)
    by = Counter((u.kind, u.home, u.reserve) for u in mid)
    assert by[("pilot", "JFK", "standby")] + by[("pilot", "JFK", "short")] == round(100 * 0.16 / 0.84 * 18 / 15)
    assert by[("pilot", "JFK", "standby")] == 2
    assert by[("cabin", "JFK", "standby")] + by[("cabin", "JFK", "short")] == round(100 * 0.19 / 0.81 * 18 / 15)
    assert not any(u.home == "SJU" for u in mid)
    low = crews.reserves(working, "2024-08-21", local, "low")
    high = crews.reserves(working, "2024-08-21", local, "high")
    assert sum(u.kind == "pilot" for u in low) == round(100 * 0.13 / 0.87 * 1.2)
    assert sum(u.kind == "pilot" for u in high) == round(100 * 0.20 / 0.80 * 1.2)
    short = [u for u in mid if u.kind == "cabin" and u.reserve == "short"]
    assert {u.lead for u in short} == {165}
    am = [u.rap for u in short if u.rap < 720]
    pm = [u.rap for u in short if u.rap >= 720]
    assert min(am) >= 120 and max(am) <= 420 and min(pm) >= 780 and max(pm) <= 1080
    assert abs(len(am) - len(pm)) <= 1
    assert all(u.until == u.rap + 720 and u.report is None for u in short)
    standby = sorted(u.report for u in mid if u.kind == "pilot" and u.reserve == "standby")
    assert standby == [300, 660]


def test_costs_from_the_sources():
    d = "2024-08-21"
    cpi = 321.943 / 314.796 / 1.07
    assert crews.seat(d) == pytest.approx(212.78 * cpi)
    assert crews.hotel("JFK", d) == pytest.approx((225 + 79) * cpi)
    assert crews.hotel("SJU", d) == pytest.approx((217 + 148) * cpi)
    assert crews.hotel("CUN", "2024-01-16") == pytest.approx((107 + 59) * 321.943 / 308.417 / 1.07)
    assert crews.rate("pilot", "A32X") == pytest.approx(1378 / 60 * 321.943 / 299.685 / 1.07, rel=1e-3)
    assert crews.rate("pilot", "E190") == pytest.approx(1336 / 60 * 321.943 / 299.685 / 1.07, rel=1e-3)
    assert crews.rate("cabin", "A220") == pytest.approx(3 * 1.88)
    assert crews.deadhead("cabin", "A220", 100, d) == pytest.approx(3 * 1.88 * 100 + 3 * crews.seat(d))


def test_reposition_tomorrow_with_a_hotel_away_from_base():
    a = {"JFK": Airport("JFK", "JFK", 40.64, -73.78, "America/New_York"),
         "BOS": Airport("BOS", "BOS", 42.36, -71.01, "America/New_York"),
         "DCA": Airport("DCA", "DCA", 38.85, -77.04, "America/New_York")}
    u = crews.Unit("P", "pilot", "A32X", 2, "JFK", "BOS", "JFK", 0, 0, 8)
    d = "2024-08-21"
    assert crews.reposition(u, "BOS", a, d) == 0
    block = round(crews.ferry_block(a["DCA"], a["BOS"]))
    assert crews.reposition(u, "DCA", a, d) == pytest.approx(crews.deadhead("pilot", "A32X", block, d)
                                                             + 2 * crews.hotel("DCA", d))
    assert crews.reposition(u, "JFK", a, d) == pytest.approx(
        crews.deadhead("pilot", "A32X", round(crews.ferry_block(a["JFK"], a["BOS"])), d))


def test_a_month_bls_skipped_takes_the_latest_cpi_before_it():
    assert crews.CPI["2025-10"] is None and crews._cpi("2025-10-14") == crews.CPI["2025-09"]
    assert crews.hotel("JFK", "2025-10-14") > 0 and crews.seat("2025-10-14") > 0


def test_the_earliest_departure_note_says_what_the_crews_do():
    note = crews.params()["duty_conventions"]["earliest_departure_after_report_min"]["note"]
    assert crews.BRIEF == 45 and "as soon as it reports" in note and "second duty period" in note
