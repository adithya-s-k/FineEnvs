import importlib.util
import json
from concurrent.futures import Future
from types import SimpleNamespace

from day_helpers import day, flight, tail
from program_helpers import TOOLS

from airops_core.core.sim import AUTO_CANCELLED, CANCELLED, FLOWN
from airops_core.programs import crews, sim
from airops_core.programs.crews import Unit

SPEC = importlib.util.spec_from_file_location("crew_check", TOOLS / "crew_check.py")
crew_check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(crew_check)


def test_the_record_guard_reads_one_percent_of_the_flights_flown_and_at_most_two():
    assert [crew_check.replays(a, n) for a, n in ((0, 0), (1, 100), (1, 99), (2, 200), (2, 150), (3, 1000))] == [
        True, True, False, True, False, False]


def test_the_calm_guard_allows_two_percent_and_from_zero_only_zero():
    assert crew_check.rise(102.0, 100.0) == 2.0 and crew_check.calm_pass(102.0, 100.0)
    assert not crew_check.calm_pass(102.1, 100.0)
    assert crew_check.calm_pass(0.0, 0.0) and crew_check.rise(5.0, 0.0) is None and not crew_check.calm_pass(5.0, 0.0)


def crew(uid: str, kind: str, station: str, report: int, trip: tuple) -> Unit:
    return Unit(uid, kind, "A32X", crews.size(kind, "A32X"), "JFK", station, station, report + crews.BRIEF, report,
                report // 60, at=station, trip=trip)


def test_the_baseline_is_the_standard_crew_desk_on_the_empty_plan_beside_naive_without_crews():
    flights = [flight(0, "JFK", "BOS", 600, 60, deadline=700), flight(1, "JFK", "DCA", 800, 60, deadline=1000)]
    task = day(flights, [tail("N1", "JFK"), tail("N2", "JFK")])
    units = [crew("P1", "pilot", "BUF", 540, (0,)), crew("C1", "cabin", "BUF", 540, (0,)),
             crew("P2", "pilot", "JFK", 500, (1,)), crew("C2", "cabin", "JFK", 500, (1,))]
    w = SimpleNamespace(local=lambda s, m: m, airports=task.airports)
    played, plan = crew_check.plays(task, units, sim.truth(task), w)
    assert played["crew_desk"]["statuses"] == {CANCELLED: 1, FLOWN: 1} and played["crew_desk"]["uncrewed"] == 0
    assert plan[0] == {"cancel": True, "decided": 0} and plan[1]["crew"] == {"pilot": "P2", "cabin": "C2"}
    assert played["naive"]["statuses"] == {FLOWN: 2} and played["naive"]["flat"] < played["crew_desk"]["flat"]
    assert played["airline"]["statuses"] == {AUTO_CANCELLED: 1, FLOWN: 1}


def notice(best=100.0, bound=99.0, exact=True, crewless=0, below=None) -> dict:
    return {"best": best, "bound": bound, "exact": exact, "flown_without_crew": crewless, "below_bound": below or {},
            "gap_pct": round(100 * (best - bound) / best, 4)}


def test_a_best_plan_counts_when_it_replays_exactly_crewed_and_nothing_played_is_under_its_bound():
    assert crew_check.best_ok(notice())
    assert not crew_check.best_ok(notice(exact=False))
    assert not crew_check.best_ok(notice(crewless=1))
    assert not crew_check.best_ok(notice(below={"naive": 98.0}))
    assert not crew_check.best_ok(notice(bound=101.0))


def joint(best=100.0, flat=95.0, recorded=120.0, desk=96.0) -> dict:
    return {"notice": notice(best), "flat": notice(flat, flat), "recorded": {"total": recorded},
            "played": {"crew_desk": {"flat": desk, "notice": desk, "uncrewed": 0,
                                     "statuses": {"auto_cancelled": 2, "cancelled": 1}},
                       "naive": {"flat": 90.0, "notice": 90.0}}}


def test_a_control_needs_the_recorded_day_over_the_best_plan_and_the_crew_desk_within_three_percent_of_the_flat_best():
    assert crew_check.CONTROL_MARGIN == 0.03
    assert crew_check.control_checks(joint())["ok"]
    assert not crew_check.control_checks(joint(recorded=99.0))["ok"]
    assert crew_check.control_checks(joint(desk=97.8))["ok"]
    c = crew_check.control_checks(joint(desk=97.9))
    assert not c["ok"] and c["crew_desk_flat_over_best"] == round(97.9 / 95 - 1, 4) and c["naive_flat"] == 90.0


def row(date, split="train", rec=True, calm=True, best=100.0, bound=99.0, uncovered=0) -> dict:
    hours = crew_check.duty_hours([9.0, 15.0])
    band = {"record": {"pass": rec, "uncrewed": 0, "cabin_duty_h": hours},
            "calm": {"pass": calm, "rise_pct": 0.0, "cabin_duty_h": hours},
            "joint": {**joint(best), "notice": {**notice(best, bound), "crew_cancelled": 0, "optimum": {
                "objective": 99.0}, "crew_delayed": {"flights": 0, "minutes": 0}, "crew_cost": {
                "deadhead": 0.0, "hotel": 0.0, "end": 1.0}, "pieces": {"aircraft": bound, "cover": None},
                "cabin_duty_h": hours}}}
    trips = {"pilot/A32X": {"uncovered": uncovered, "converged": True, "rounds": 8, "gap_pct": 0.1}}
    return {"date": date, "task_id": f"b6-{date}", "split": split, "seconds": 1.0, "peak_rss_gb": 1.0,
            "bands": {"middle": band}, "trips": trips}


def test_the_guards_count_storm_dates_at_ninety_percent_and_gaps_within_each_threshold():
    rows = [row(f"d{k}") for k in range(9)] + [row("d9", rec=False, calm=False, bound=90.0),
                                              row("c1", "control"), row("c2", "control")]
    g = crew_check.guards(rows, ("middle",))["guards"]
    assert g["record"]["value"] == 0.9 and g["record"]["pass"] and g["record"]["n"] == 10
    assert g["calm"]["value"] == 0.9 and g["calm"]["pass"]
    assert g["best"]["kept"] == 12 and g["best"]["gap_within"] == {"0.01%": 0, "1.0%": 11, "5.0%": 11}
    assert g["controls"]["pass"] and g["go"]
    rows[2] = row("d2", uncovered=3)
    r = crew_check.guards(rows, ("middle",))["reported"]
    assert r["trips"]["uncovered_legs"] == 3 and r["trips"]["dates_with_uncovered"] == 1
    assert r["trips"]["converged"] == r["trips"]["builds"] == 12
    assert r["crew_desk"]["auto_cancelled"]["median"] == 2 and r["crew_desk"]["cancelled"]["median"] == 1
    assert r["cabin_duty_h"]["best"]["over_14h"] == 12 and r["cabin_duty_h"]["record"]["max"]["max"] == 15.0
    assert "stage" not in str(crew_check.guards(rows, ("middle",))).lower()
    rows[1] = {"date": "d1", "task_id": "b6-d1", "split": "train", "error": "boom"}
    g = crew_check.guards(rows, ("middle",))["guards"]
    assert g["best"]["left_out"] == 1 and g["record"]["n"] == 9 and not g["record"]["pass"]


def test_cabin_duties_count_report_to_release_and_those_past_fourteen_hours():
    day = SimpleNamespace(units={"C1": {"kind": "cabin", "duties": [{"report": 300, "release": 1260},
                                                                    {"report": 1900, "release": None}]},
                                 "P1": {"kind": "pilot", "duties": [{"report": 0, "release": 1200}]}})
    assert crew_check.day_duties(day) == [16.0]
    h = crew_check.duty_hours([16.0, 14.0, 9.5])
    assert h["over_14h"] == 1 and h["max"] == 16.0 and h["n"] == 3


class Inline:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def submit(self, fn, *args):
        done = Future()
        done.set_result(fn(*args))
        return done


def test_the_desk_replay_rewrites_only_the_desk_and_what_reads_it_and_says_over_which_run(tmp_path, monkeypatch):
    run, out = tmp_path / "run", tmp_path / "replay"
    (run / "dates").mkdir(parents=True)
    (out / "dates").mkdir(parents=True)
    rows = [row("2024-07-03", "control"), row("2025-01-15", "control"), row("d1")]
    for r in rows:
        r["budget"] = {"morning": 1, "polish": 1}
        r["bands"]["middle"]["record"] |= {"before_start": {"objective": 5.0, "uncrewed": 0}, "flown": 10,
                                            "solve": {"uncrewed_bound": 0.0}}
        (run / "dates" / f"{r['date']}.json").write_text(json.dumps(r))
    before = {p.name: p.read_text() for p in (run / "dates").iterdir()}
    desk = {"notice": 97.9, "flat": 97.9, "uncrewed": 0, "statuses": {"cancelled": 2}}
    monkeypatch.setattr(crew_check, "ProcessPoolExecutor", Inline)
    monkeypatch.setattr(crew_check, "Task", SimpleNamespace(from_dict=lambda d: SimpleNamespace(**d)))
    monkeypatch.setattr(crew_check.crewday, "world", lambda task, inputs, derived: SimpleNamespace(local=None,
                                                                                                   airports=None))
    monkeypatch.setattr(crew_check.crewday, "storm_units", lambda w, band, limit, polish: (
        [], {"objective": 5.0 if w is not None else 0.0, "uncrewed": 0}))
    monkeypatch.setattr(crew_check.sim, "truth", lambda task: None)
    monkeypatch.setattr(crew_check.crewsim, "execute", lambda *args, **kwargs: None)
    monkeypatch.setattr(crew_check, "outcome", lambda d: desk)
    crew_check.desk_replay(run, out, {r["date"]: json.dumps({"date": r["date"]}) for r in rows}, 2)
    g = json.loads((out / "guards.json").read_text())
    assert g["desk_replay"]["over"] == str(run) and g["desk_replay"]["units_as_run"] == g["desk_replay"]["dates"] == 3
    c = g["guards"]["controls"]["controls"]["b6-2024-07-03"]
    assert c["crew_desk_flat"] == 97.9 and not c["crew_desk_within_margin"] and not g["guards"]["controls"]["pass"]
    again = json.loads((out / "dates" / "d1.json").read_text())["bands"]["middle"]["joint"]
    assert again["played"]["crew_desk"] == desk and again["notice"]["below_bound"] == {"crew_desk": 97.9}
    assert again["played"]["naive"] == rows[2]["bands"]["middle"]["joint"]["played"]["naive"]
    assert (out / "report.md").read_text().startswith(f"The standard crew desk replayed over {run} at commit ")
    assert {p.name: p.read_text() for p in (run / "dates").iterdir()} == before


def test_the_band_ends_run_over_a_middle_band_run_add_low_and_high_and_keep_the_rest(tmp_path, monkeypatch):
    run, out = tmp_path / "run", tmp_path / "ends"
    (run / "dates").mkdir(parents=True)
    (run / "plans").mkdir()
    rows = [row("2024-07-03", "control"), row("d1"), {"date": "d2", "task_id": "b6-d2", "split": "train",
                                                       "error": "boom"}]
    for r in rows:
        if "error" not in r:
            r["budget"] = {"end_morning": 1, "end_record": 1, "end_calm": 1, "polish": 1, "width": 30}
            r["bands"]["middle"]["calm"]["base"] = 42.0
            r["bands"]["middle"]["record"] |= {"flown": 10, "solve": {"uncrewed_bound": 0.0}}
        (run / "dates" / f"{r['date']}.json").write_text(json.dumps(r))
        (run / "plans" / f"{r['date']}.json").write_text(json.dumps({"notice": {r["date"]: 1}}))
    (run / "guards.json").write_text(json.dumps({"desk_replay": {"over": "full2", "commit": "abc"}}))
    before = {p.name: p.read_text() for p in (run / "dates").iterdir()}
    seen = []

    def end_band(w, band, optimum, base, budget, clock):
        seen.append((band, base, optimum.objective, budget["end_calm"]))
        return {"record": {"pass": band == "low", "uncrewed": 1, "flown": 10, "causes": []},
                "calm": {"pass": True, "rise_pct": 0.5}}

    monkeypatch.setattr(crew_check, "ProcessPoolExecutor", Inline)
    monkeypatch.setattr(crew_check, "Task", SimpleNamespace(from_dict=lambda d: SimpleNamespace(**d)))
    monkeypatch.setattr(crew_check.crewday, "world", lambda task, inputs, derived: None)
    monkeypatch.setattr(crew_check.crewplan, "calm_optimum", lambda w: SimpleNamespace(objective=40.0))
    monkeypatch.setattr(crew_check, "end_band", end_band)
    lines = {r["date"]: json.dumps({"date": r["date"]}) for r in rows}
    crew_check.ends_over(run, out, lines, 2, False)
    assert sorted(seen) == [("high", 42.0, 40.0, 1), ("high", 42.0, 40.0, 1), ("low", 42.0, 40.0, 1),
                            ("low", 42.0, 40.0, 1)]
    g = json.loads((out / "guards.json").read_text())
    assert g["ends_over"]["over"] == str(run) and g["ends_over"]["dates"] == 3 and g["ends_over"]["failed"] == []
    assert g["desk_replay"] == {"over": "full2", "commit": "abc"}
    assert g["reported"]["bands"]["low"]["record"] == 1.0 and g["reported"]["bands"]["high"]["record"] == 0.0
    d1 = json.loads((out / "dates" / "d1.json").read_text())
    assert set(d1["bands"]) == {"middle", "low", "high"} and d1["bands"]["middle"] == rows[1]["bands"]["middle"]
    assert json.loads((out / "dates" / "d2.json").read_text()) == rows[2]
    assert (out / "plans" / "d1.json").read_text() == (run / "plans" / "d1.json").read_text()
    assert (out / "report.md").read_text().startswith(f"The reserve-band ends (low, high) run over {run} at commit ")
    assert {p.name: p.read_text() for p in (run / "dates").iterdir()} == before
    seen.clear()
    crew_check.ends_over(run, out, lines, 2, True)
    assert seen == []
