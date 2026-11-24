import importlib.util

from program_helpers import TOOLS

SPEC = importlib.util.spec_from_file_location("world_check", TOOLS / "world_check.py")
world_check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(world_check)


def test_the_airline_replay_reads_one_percent_of_the_flights_flown_and_at_most_two_literally():
    assert [world_check.replays(a, n) for a, n in ((0, 0), (1, 100), (1, 99), (2, 200), (2, 150), (3, 1000))] == [
        True, True, False, True, False, False]


def test_fidelity_pools_modelled_over_observed_delay_over_the_days_within_half_to_twice():
    days = [(120.0, 100.0, 5), (80.0, 100.0, 4), (300.0, 100.0, 3), (40.0, 100.0, 2), (10.0, 0.0, 1)]
    assert world_check.pooled(days) == {"ratio": 1.0, "days": 2, "dropped": 2, "no_delay": 1}
    assert world_check.pooled([(50.0, 100.0, 1), (200.0, 100.0, 1)]) == {"ratio": 1.25, "days": 2, "dropped": 0,
                                                                          "no_delay": 0}
    assert world_check.pooled([(0.0, 0.0, 0)])["ratio"] is None


def test_fidelity_pools_each_storm_hub_of_a_date_and_passes_only_with_every_pool_in_band_out_of_sample():
    fid = {("a", "JFK"): (110.0, 100.0, 3), ("a", "BOS"): (90.0, 100.0, 3), ("c", "JFK"): (100.0, 100.0, 3),
           ("d", "BOS"): (95.0, 100.0, 3), ("e", "BOS"): (500.0, 100.0, 3)}
    split = {"a": "eval", "c": "train", "d": "train", "e": "train"}
    gate = world_check.fidelity_gate(fid, split, True)
    assert gate["pass"] and gate["train_BOS"] == {"ratio": 0.95, "days": 1, "dropped": 1, "no_delay": 0}
    assert gate["eval_JFK"]["ratio"] == 1.1 and gate["eval_BOS"]["ratio"] == 0.9
    assert not world_check.fidelity_gate(fid, split, False)["pass"]
    assert not world_check.fidelity_gate(fid, {**split, "c": "eval"}, True)["pass"]
    high = world_check.fidelity_gate({**fid, ("a", "BOS"): (130.0, 100.0, 3)}, split, True)
    assert high["eval_BOS"]["ratio"] == 1.3 and not high["pass"]


def test_a_task_is_left_out_for_every_reference_it_ran_that_failed_or_did_not_prove():
    ok = {k: {"proven": True} for k in world_check.REFERENCES}
    assert world_check.unproven(ok) == []
    r = {**ok, "follower": {"error": "RuntimeError: watch 600 solve Time limit reached"}, "flat": {"proven": False}}
    assert world_check.unproven(r) == ["follower", "flat"]
    assert world_check.unproven({k: v for k, v in ok.items() if k != "calm"}) == ["calm"]
    ran = {"refs": list(world_check.DEFAULT_REFS), **{k: ok[k] for k in world_check.DEFAULT_REFS}}
    assert world_check.unproven(ran) == [] and world_check.unproven({**ran, "refs": ["rule"]}) == ["rule"]


def test_the_references_run_are_the_default_set_unless_named_and_only_takes_dates_or_task_ids():
    assert world_check.chosen(",".join(world_check.DEFAULT_REFS)) == world_check.DEFAULT_REFS == (
        "naive", "airline", "recorded", "anchor", "flat", "calm")
    assert world_check.chosen("calm,rule,airline") == ("airline", "rule", "calm")
    only = world_check.wanted("2024-08-19,b6-20250209-0600")
    assert [only(d) for d in ("2024-08-19", "2025-02-09", "2025-02-10")] == [True, True, False]
    assert world_check.wanted("")("2023-07-16")


def test_the_references_gate_counts_the_references_it_compared_with_each_anchor():
    run = {"anchor": {"exact": True, "gap_pct": 0.0, "seats_ok": True}, "naive": {"total": 9, "seats_ok": True},
           "airline": {"total": 8, "seats_ok": True}, "calm": {"cancelled": 0}}
    gate = world_check.references_gate({"a": run, "b": {**run, "below_anchor": {"airline": 7}}}, {})
    assert gate["compared_with_anchor"] == {"naive": 2, "airline": 2} and gate["below_anchor"] == {"b": {"airline": 7}}
    assert not gate["pass"] and world_check.references_gate({"a": run}, {})["pass"]


def test_a_control_needs_the_recorded_day_over_the_anchor_and_naive_within_three_percent_of_the_flat_anchor():
    run = {"naive": {"flat": 103.0}, "recorded": {"total": 120.0, "delay": 1},
           "anchor": {"objective": 100.0, "delay": 1}, "flat": {"objective": 100.0}}
    assert world_check.CONTROL_MARGIN == 0.03 and world_check.control_checks(run, True)["ok"]
    late = world_check.control_checks({**run, "naive": {"flat": 103.1}}, True)
    assert not late["ok"] and not late["naive_within_margin"] and late["naive_flat_over_anchor"] == 0.031
    assert not world_check.control_checks({**run, "recorded": {"total": 99.0, "delay": 1}}, True)["ok"]
