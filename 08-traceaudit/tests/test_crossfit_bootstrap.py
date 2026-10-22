import itertools

import pytest
from test_evidence_audit import TAU, heldout_cohorts, make_read_only
from test_training_shortcuts import runs
from trace_audit.core.shortcuts import (
    _interval,
    compare_write_baseline,
    crossfit_direction,
    training_shortcut_audit,
)

"""Achievable uncertainty checks and honest task-dependent shortcut evaluation."""


@pytest.mark.parametrize("families", [2, 8, 20])
def test_perfect_auditor_half_written_failures_needs_realistic_task_count(rows, families):
    train = runs(rows[0], source=TAU, successes=10, failures=10)
    for row in [r for r in train if r["outcome"]["failed"]][:5]:
        make_read_only(row)
    baseline = training_shortcut_audit(train)["write_baseline"]
    data = []
    for i in range(families):
        members = runs(rows[0], source=TAU, task=str(i), successes=10, failures=10)
        for row in [r for r in members if r["outcome"]["failed"]][:5]:
            make_read_only(row)
        data.extend(members)
    # Small calibration differences make per-task gains nonconstant; every label
    # is still classified correctly. No hundreds-of-tasks fixture is needed.
    p = []
    for row in data:
        task = int(row["group"]["task_family_id"].rsplit(":", 1)[1])
        error = (task % 5) * 0.01
        p.append(1 - error if row["outcome"]["failed"] else error)
    result = compare_write_baseline(data, p, baseline)
    assert 0.15 < result["brier_gain_task_cluster_interval"]["estimate"] < 0.20
    if families == 2:
        assert not result["passes"]
        assert "insufficient_brier_task_clusters" in result["reasons"]
        assert result["brier_gain_task_cluster_interval"]["lower"] is None
    else:
        assert result["passes"]
        assert result["brier_gain_task_cluster_interval"]["lower"] > 0.1
        assert result["written_runs"]["confidence_interval"]["lower"] > 0.5


def test_cluster_bootstrap_uses_observed_variance_and_is_reproducible():
    gains = [0.10, 0.12, 0.13, 0.14, 0.15, 0.17, 0.20, 0.22]
    result = _interval(gains, lower=-1, upper=1)
    assert 0 < result["lower"] < result["estimate"] < result["upper"] < 0.23
    assert result == _interval(list(reversed(gains)), lower=-1, upper=1)
    noisy = _interval([-0.2] * 10 + [0.2] * 10, lower=-1, upper=1)
    assert noisy["lower"] < 0 < noisy["upper"]
    heterogeneous = _interval([-0.4] * 4 + [0.5] * 4, lower=-1, upper=1)
    assert heterogeneous["lower"] < 0 < heterogeneous["upper"]
    assert not result["degenerate_distribution"]


def test_percentile_bootstrap_matches_simple_exact_distribution():
    # For two observations {-1,1}, resampled means are {-1,0,1} with
    # probabilities {1/4,1/2,1/4}; the 95% percentile endpoints are -1 and 1.
    result = _interval([-1, 1], lower=-1, upper=1, min_task_clusters=2)
    assert (result["estimate"], result["lower"], result["upper"]) == (0, -1, 1)
    assert _interval([0.17] * 2, lower=-1, upper=1)["status"] == "insufficient_task_clusters"


@pytest.mark.parametrize("families", [2, 8])
def test_opposite_task_rules_are_detected_out_of_fold(rows, families):
    records = []
    for i in range(families):
        records += runs(
            rows[0],
            source=TAU,
            task=str(i),
            successes=4,
            failures=4,
            write_success=i % 2 == 0,
            write_failure=i % 2 != 0,
        )
    probe = training_shortcut_audit(records)["pooled_probes"]["any_write"]
    assert probe["auroc"] == 0.5  # The original signed diagnostic still cancels.
    cv = probe["cross_fitted_direction"]
    assert cv["auroc"] == 1
    if families == 8:
        assert cv["confidence_interval"]["lower"] > 0.5
        assert cv["support_status"] == "sufficient"
    else:
        assert cv["confidence_interval"]["lower"] is None
        assert cv["support_status"] == "insufficient_evidence"
    assert cv["unique_outcome_pairs"] == 4 * families
    for detail in probe["per_stratum"]:
        folds = detail["cross_fitted_direction"]["folds"]
        assert len(folds) == 2
        assert all(f["membership_disjoint"] for f in folds)
        assert sum(f["evaluation_records"] for f in folds) == 8
    # It is a diagnostic about a learned feature rule, not automatic rejection of
    # legitimate task evidence or a new reward term.
    assert cv["decision_role"] == "diagnostic_only"


def test_crossfit_does_not_reorient_using_evaluation_labels(rows):
    records = runs(rows[0], successes=2, failures=2)
    results = []
    for pattern in itertools.product((0.0, 1.0), repeat=4):
        values = {r["record_id"]: v for r, v in zip(records, pattern, strict=False)}
        result = crossfit_direction(records, values)
        results.append(result["auroc"])
        assert result == crossfit_direction(list(reversed(records)), values)
    # Contradictory fold rules must score zero, not be flipped into a perfect
    # apparent shortcut after inspecting their held-out outcomes.
    assert 0 in results and 1 in results
    assert sum(results) / len(results) == 0.5  # All binary-noise assignments average to chance.
    assert (
        crossfit_direction(records[:1] + records[2:3], {r["record_id"]: 0.0 for r in records})[
            "auroc"
        ]
        is None
    )


def test_crossfit_replays_do_not_inflate_support_and_single_pairs_are_skipped(rows):
    records = runs(rows[0], source=TAU, successes=2, failures=2, write_failure=False)
    original = training_shortcut_audit(records)["pooled_probes"]["any_write"][
        "cross_fitted_direction"
    ]
    replayed = training_shortcut_audit(records * 20)["pooled_probes"]["any_write"][
        "cross_fitted_direction"
    ]
    assert replayed == original
    assert original["confidence_interval"]["task_clusters"] == 1
    assert original["support_status"] == "insufficient_evidence"
    single = runs(rows[0], source=TAU, write_failure=False)
    skipped = training_shortcut_audit(single)["pooled_probes"]["any_write"][
        "cross_fitted_direction"
    ]
    assert skipped["auroc"] is None and skipped["skipped_strata"] == 1


def test_reversed_heldout_auditor_is_never_direction_fitted(rows):
    baseline = training_shortcut_audit(runs(rows[0], source=TAU, successes=10, failures=10))[
        "write_baseline"
    ]
    data = [r for r in heldout_cohorts(rows[0]) if r["source"]["dataset"] == TAU]
    result = compare_write_baseline(
        data, [1 - float(r["outcome"]["failed"]) for r in data], baseline
    )
    assert not result["passes"]
    assert result["written_runs"]["auroc"] == 0
    assert "cross_fitted_direction" not in result["written_runs"]
