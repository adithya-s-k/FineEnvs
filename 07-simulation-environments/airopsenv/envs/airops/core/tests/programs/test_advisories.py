
import pytest
from program_helpers import adv, record

from airops_core.programs import advisories, candidates
from airops_core.programs.advisories import TIERS, Scope

BASE = candidates.midnight("2024-08-19")
GDP = """
CTL ELEMENT: JFK
ELEMENT TYPE: APT
ADL TIME: 1531Z
DELAY ASSIGNMENT MODE: UDP
ARRIVALS ESTIMATED FOR: 19/1700Z - 20/0459Z
CUMULATIVE PROGRAM PERIOD: 19/1700Z - 20/0459Z
PROGRAM RATE: 40/38/38/38/38/36/36/36/36/36/36/36
FLT INCL: ALL CONTIGUOUS US DEP
DEP SCOPE: (ALL+CZY_AP) ZLA ZAU ZLC ZTL ZDC ZNY ZHU ZJX ZFW ZOB ZDV
 ZOA ZSE ZBW ZMA ZKC ZME ZID ZAB ZMP
CANADIAN DEP ARPTS INCLUDED: CYHZ CYOW CYUL CYYZ CYTZ CYQB
DELAY ASSIGNMENT TABLE APPLIES TO: ZNY
MAXIMUM DELAY: 287
AVERAGE DELAY: 196
IMPACTING CONDITION: WEATHER / THUNDERSTORMS
COMMENTS: ARR 04R/04L, DEP 04L, LOW POP UP, BY STATUS
"""
SINGLE = """
CTL ELEMENT: BOS
ARRIVALS ESTIMATED FOR: 19/2130Z - 20/0429Z
CUMULATIVE PROGRAM PERIOD: 19/2130Z - 20/0429Z
PROGRAM RATE: 34
FLT INCL: CONTIGUOUS US JET DEP ONLY
DEP SCOPE: 1250
ADDITIONAL DEP FACILITIES INCLUDED: ZBW
EXEMPT DEP FACILITIES: DFW DAL
CANADIAN DEP ARPTS INCLUDED: CYHZ CYOW
MAXIMUM DELAY: 138
AVERAGE DELAY: 46
"""
STOP = """
CTL ELEMENT: JFK
ELEMENT TYPE: APT
ADL TIME: 0144Z
GROUND STOP PERIOD: 19/0134Z - 19/0300Z
CUMULATIVE PROGRAM PERIOD: 18/1700Z - 19/0259Z
DEP FACILITIES INCLUDED: (Manual) ZAU ZTL ZDC ZNY ZJX ZOB ZBW ZID
 CYHZ CYOW CYUL CYYZ CYTZ CYQB
PREVIOUS TOTAL, MAXIMUM, AVERAGE DELAYS: 5534 / 391 / 221
NEW TOTAL, MAXIMUM, AVERAGE DELAYS: 7021 / 451 / 281
PROBABILITY OF EXTENSION: MED
IMPACTING CONDITION: WEATHER / THUNDERSTORMS
COMMENTS: 2ND TIER + CZY MINUS ZMP
"""


def parse(title: str, message: str, signature: str, advn: int = 85, element: str = "JFK/ZNY"):
    return advisories.parse(record(advn, title, message, signature, "2024-08-19", element), element[:3], BASE)


def test_a_gdp_reads_its_period_rates_delays_and_scope():
    a = parse("CDM GROUND DELAY PROGRAM", GDP, "24/08/19 15:33")
    assert (a.id, a.kind, a.issued, a.start, a.end) == ("adv-20240819-085", "gdp", 693, 780, 1499)
    assert a.rates == (40, 38, 38, 38, 38, 36, 36, 36, 36, 36, 36, 36) and a.rate_at(1499) == 36
    assert (a.avg_delay, a.max_delay, a.binds) == (196, 287, True)
    assert len(a.scope.artccs) == 20 and a.scope.contiguous and {"CYYZ", "CYQB"} <= a.scope.airports
    assert a.scope.covers("ORD", "ZAU", 640) and not a.scope.covers("SJU", "ZSU", 1390)


def test_a_single_rate_applies_to_every_hour_and_a_distance_scope_reads_as_nm():
    a = parse("CDM GROUND DELAY PROGRAM", SINGLE, "24/08/19 19:03", element="BOS/ZBW")
    assert a.rates == (34,) * 8 and a.rate_at(a.start) == 34
    s = a.scope
    assert (s.nm, s.artccs, s.exempt) == (1250, frozenset({"ZBW"}), frozenset({"DFW", "DAL"}))
    assert s.covers("ORD", "ZAU", 750) and not s.covers("DEN", "ZDV", 1500) and s.covers("PWM", "ZBW", 2000)
    assert not s.covers("DFW", "ZFW", 1200)


def test_a_ground_stop_reads_its_window_scope_and_stated_delays():
    a = parse("CDM GROUND STOP", STOP, "24/08/19 01:49", advn=21)
    assert (a.kind, a.issued, a.start, a.end) == ("gs", -131, -146, -60)
    assert (a.avg_delay, a.max_delay, a.extension) == (281, 451, "MEDIUM")
    assert a.scope.artccs == {"ZAU", "ZTL", "ZDC", "ZNY", "ZJX", "ZOB", "ZBW", "ZID"} and not a.scope.every


@pytest.mark.parametrize(("facilities", "artccs", "every"), [
    ("(2ND TIER) ZNY ZBW ZOB ZDC ZMP ZAU ZID ZTL\nZJX", TIERS[2], False),
    ("1ST TIER + CZY", TIERS[1], False),
    ("ZDC/ZTL/ZJX/ZMA", {"ZDC", "ZTL", "ZJX", "ZMA"}, False),
    ("ALL + CZY", set(), True),
])
def test_stop_scopes_read_lists_tiers_and_all(facilities, artccs, every):
    a = parse("JFK GROUND STOP", f"EVENT TIME: 19/1815 - 19/1915\nFACILITIES INCLUDED: {facilities}\nREASON: X",
              "24/08/19 18:23")
    assert (a.scope.artccs, a.scope.every) == (artccs, every)


def test_a_stop_without_facilities_covers_every_origin():
    a = parse("BOS GROUND STOP", "EVENT TIME: 19/2230 - 19/2330\nIMPACTING CONDITION: LACK OF ROUTES",
              "24/08/19 22:39", element="BOS/ZBW")
    assert a.scope.every and a.scope.covers("SEA", "ZSE", 2100)


@pytest.mark.parametrize(("title", "body", "who"), [
    ("CDM GROUND STOP", "FLT INCL: CARRIER DAL ONLY\nCOMMENTS: DAL AND SUBS ONLY. CARRIER REQUEST", "DAL"),
    ("CDM GROUND STOP", "FLT INCL: CARRIER JBU ONLY\nCOMMENTS: JETBLUE ONLY GROUND STOP. CARRIER REQUEST", "JBU"),
    ("JFK GROUND STOP DAL MAINLINE AND SUBS", "REMARKS: DAL MAINLINE AND SUBS", "DAL"),
    ("GA ONLY GROUND STOP", "REASON: GA PARKING FULL", "GA"),
    ("JFK GROUND STOP - CAT I ONLY", "REASON: CEILINGS AT 200", "CAT"),
    ("CAT I GS CNX", "", "CAT"),
    ("CDM GROUND STOP", "COMMENTS: ZOB ONLY. NO ROUTES AVAILABLE.", None),
])
def test_carrier_ga_and_approach_limited_stops_never_bind(title, body, who):
    a = parse(title, f"GROUND STOP PERIOD: 19/1700Z - 19/1800Z\n{body}", "24/08/19 17:01")
    assert a.only == who and a.binds == (who is None)


def test_a_gdp_is_limited_only_by_its_flt_incl_whatever_its_comments_report():
    head = "ARRIVALS ESTIMATED FOR: 16/1820Z - 16/2059Z\nPROGRAM RATE: 28/28/28\n"
    revised = parse("CDM GROUND DELAY PROGRAM", head + "FLT INCL: CONTIGUOUS US JET DEP ONLY\nDEP SCOPE: ZBW ZNY\n"
                    "COMMENTS: REVISED RATES. DAL MAINLINE + SUBS GROUND STOP CANCELLED.", "23/07/16 18:30",
                    element="BOS/ZBW")
    delta = parse("CDM GROUND DELAY PROGRAM", head + "FLT INCL: CONTIGUOUS US DEP CARRIER DAL ONLY\nDEP SCOPE: 1200\n"
                  "COMMENTS: DELTA AND DELTA SUBS ONLY AIRLINE REQUEST", "24/07/19 15:39")
    assert (revised.kind, revised.only, delta.only) == ("gdp", None, "DAL")


def test_a_gdp_cancellation_ends_the_stops_only_when_its_comment_says_so():
    ends = parse("CDM GROUND DELAY PROGRAM CNX", "GDP CNX PERIOD: 19/0356Z - 19/1801Z\nCOMMENTS: GDP AND GS CANCELED",
                 "24/08/19 03:59", advn=37)
    stays = parse("CDM GROUND DELAY PROGRAM CNX",
                  "GDP CNX PERIOD: 19/0356Z - 19/1801Z\nCOMMENTS: GDP CANCELED. ZNY/ZOB REMAIN IN GROUND STOP",
                  "24/08/19 03:59", advn=37)
    assert (ends.kind, ends.start, ends.stops_too, stays.stops_too) == ("gdp_cnx", -4, True, False)


def test_a_stop_chains_its_extensions_and_ends_at_the_cancellation():
    tier1 = Scope(False, TIERS[1], None, frozenset(), frozenset(), False)
    tier2 = Scope(False, TIERS[2], None, frozenset(), frozenset(), False)
    chain = [adv("a1", "gs", 600, 590, 660, scope=tier1), adv("a2", "gs", 650, 590, 720, scope=tier2),
             adv("a3", "gs_cnx", 700, 700, None, released=frozenset({"ZOB"})),
             adv("a4", "gs_cnx", 705, 705, None, only="DAL"),
             adv("a5", "gs_cnx", 712, 710, None),
             adv("b1", "gs", 724, 724, 780, scope=tier1), adv("b2", "gs", 790, 724, 840, scope=tier1),
             adv("c1", "gs", 900, 900, 960), adv("c2", "gdp_cnx", 930, 930, None, stops_too=True)]
    pieces = [(p.chain, p.start, p.end, p.scope.artccs - p.scope.exempt) for p in advisories.as_ran(chain)[1]]
    assert pieces == [("a1", 590, 650, TIERS[1]), ("a1", 650, 700, TIERS[2]), ("a1", 700, 710, TIERS[2] - {"ZOB"}),
                      ("b1", 724, 790, TIERS[1]), ("b1", 790, 840, TIERS[1]), ("c1", 900, 930, frozenset())]
    covers = lambda scope, hub, origin: scope.covers(origin, {"PIT": "ZOB", "ATL": "ZTL"}.get(origin), 300)  # noqa: E731
    stops = advisories.as_ran(chain)[1]
    assert advisories.held_until(stops, covers, "ATL", 655) == 710
    assert advisories.held_until(stops, covers, "PIT", 655) == 700
    assert advisories.held_until(stops, covers, "PIT", 785) == 840
    assert advisories.held_until(stops, covers, "PIT", 850) == 850


def test_the_forecast_runs_the_programs_as_issued_and_the_truth_as_they_ran():
    advs = [adv("p", "gdp_proposed", 500, 600, 899, (30,) * 5), adv("g", "gdp", 520, 600, 899, (30,) * 5),
            adv("r", "gdp", 700, 700, 799, (36,) * 2), adv("x", "gdp_cnx", 750, 748, None),
            adv("s1", "gs", 600, 600, 660), adv("s2", "gs", 650, 600, 720)]
    runs = lambda view: [(r.adv.id, r.since, r.until) for r in view[0]]  # noqa: E731
    assert runs(advisories.issued_by(advs, 510)) == [("p", 500, 900)]
    assert runs(advisories.issued_by(advs, 600)) == [("g", 520, 900)]
    assert runs(advisories.issued_by(advs, 720)) == [("g", 520, 700), ("r", 700, 800)]
    assert runs(advisories.as_ran(advs)) == [("g", 520, 700), ("r", 700, 748)]
    assert [(p.start, p.end) for p in advisories.issued_by(advs, 620)[1]] == [(600, 660)]
    assert [(p.start, p.end) for p in advisories.issued_by(advs, 680)[1]] == [(600, 650), (650, 720)]
