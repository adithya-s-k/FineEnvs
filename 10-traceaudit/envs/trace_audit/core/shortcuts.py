"""Stratified training diagnostics and held-out comparison with task evidence."""

import hashlib
import math
import random
from collections import defaultdict

from .baselines import closing_claim, tool_call_count, write_tool_call_count
from .metrics import binary_metrics, shortcut_metrics
from .provenance import generator_key, sampling_key
from .render import render_record


def model_label_probe(rows: list[dict], *, sampled: bool = False) -> dict:
    """Descriptive upper bound: use each model pair's observed label frequency.

    This is an in-sample label lookup, not a fitted classifier's held-out score.
    """
    groups = defaultdict(list)
    unknown = 0
    for row in rows:
        key = generator_key(row)
        if key is None:
            unknown += 1
        else:
            groups[(row["source"]["dataset"], row["source"].get("domain") or "", *key)].append(row)
    known, predictions, rates = [], [], []
    for (source, domain, agent, user), members in sorted(groups.items()):
        rate = sum(r["outcome"]["failed"] for r in members) / len(members)
        known.extend(r["outcome"]["failed"] for r in members)
        predictions.extend([rate] * len(members))
        rates.append(
            {
                "source": source,
                "domain": domain,
                "agent_model": agent,
                "user_model": user,
                "n": len(members),
                "failure_rate": rate,
            }
        )
    return {
        "role": "balance_invariant_check" if sampled else "original_distribution_diagnostic",
        "unknown_records": unknown,
        "conditional_failure_rates": rates,
        "metrics": shortcut_metrics(known, predictions),
    }


def _interval(
    values, *, confidence=0.95, lower=0.0, upper=1.0, min_task_clusters=8, resamples=2000, seed=42
):
    """Approximate percentile bootstrap of equally weighted task-family means.

    Resample independent tasks, not trials/model variants. Small-cluster coverage
    is not guaranteed; the minimum cluster count is an explicit pending policy
    default, not a claim that a given count ensures nominal coverage.
    """
    if (
        not 0 < confidence < 1
        or type(min_task_clusters) is not int
        or min_task_clusters < 2
        or type(resamples) is not int
        or resamples < 100
    ):
        raise ValueError(
            "Valid confidence, at least two task clusters and >=100 resamples are required"
        )
    values = sorted(values)  # Fixed-seed output must not depend on input order.
    if any(not math.isfinite(v) or not lower <= v <= upper for v in values):
        raise ValueError("Task statistics must be finite and within their declared bounds")
    estimate = sum(values) / len(values) if values else None
    result = {
        "estimate": estimate,
        "lower": None,
        "upper": None,
        "task_clusters": len(values),
        "confidence": confidence,
        "minimum_task_clusters": min_task_clusters,
        "resamples": resamples,
        "seed": seed,
        "status": "insufficient_task_clusters",
    }
    if len(values) < min_task_clusters:
        return result
    rng = random.Random(seed)
    draws = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(resamples))

    def quantile(q):
        index = (len(draws) - 1) * q
        left = int(index)
        right = min(left + 1, len(draws) - 1)
        return draws[left] + (draws[right] - draws[left]) * (index - left)

    tail = (1 - confidence) / 2
    result.update(
        lower=max(lower, quantile(tail)),
        upper=min(upper, quantile(1 - tail)),
        status="estimated",
        degenerate_distribution=draws[0] == draws[-1],
    )
    return result


def _unique(rows):
    # Explicit sample reuse must not create statistical support.
    return list({row["record_id"]: row for row in rows}.values())


def crossfit_direction(members, values, *, seed=42):
    """Learn task/model feature direction on one half; score only the other half.

    Each unique record is scored exactly once across two label-stratified folds.
    No direction is selected or reversed using its evaluation-fold outcomes.
    """
    by_label = {label: [] for label in (False, True)}
    for row in _unique(members):
        by_label[row["outcome"]["failed"]].append(row)
    if min(map(len, by_label.values())) < 2:
        return {
            "auroc": None,
            "unique_outcome_pairs": 0,
            "folds": [],
            "status": "needs_two_pairs_per_stratum",
        }
    folds = [[], []]
    for label in (False, True):
        ordered = sorted(
            by_label[label],
            key=lambda r: hashlib.sha256(f"{seed}:{r['record_id']}".encode()).hexdigest(),
        )
        for index, row in enumerate(ordered):
            folds[index % 2].append(row)
    measured = []
    for index in (0, 1):
        measured.append(_crossfit_fold(folds, index, values))
    # Weight folds by their scored class pairs when the stratum has odd counts.
    pairs = sum(f["evaluation_pairs"] for f in measured)
    return {
        "auroc": sum(f["evaluation_auroc"] * f["evaluation_pairs"] for f in measured) / pairs,
        "unique_outcome_pairs": pairs,
        "folds": measured,
        "status": "cross_fitted",
    }


def stratified_probe(
    rows, values, *, min_pairs=10, confidence=0.95, by_task_only=False, min_task_clusters=8
):
    strata = _probe_strata(rows, values, by_task_only)
    details, clusters, pairs = [], defaultdict(list), 0
    crossfit_clusters, crossfit_pairs = defaultdict(list), 0
    skipped_crossfit = 0
    crossfit_records = 0
    for key, members in sorted(strata.items(), key=lambda pair: str(pair[0])):
        labels = [r["outcome"]["failed"] for r in members]
        metric = shortcut_metrics(labels, [values[r["record_id"]] for r in members])
        n = min(sum(labels), len(labels) - sum(labels))
        fitted = crossfit_direction(members, values) if not by_task_only else None
        detail = {
            "stratum": key,
            "unique_records": len(members),
            "outcome_pairs": n,
            "metrics": metric,
        }
        if fitted is not None:
            detail["cross_fitted_direction"] = fitted
            if fitted["auroc"] is not None:
                crossfit_clusters[members[0]["group"]["task_family_id"]].append(fitted["auroc"])
                crossfit_pairs += fitted["unique_outcome_pairs"]
                crossfit_records += len(members)
            else:
                skipped_crossfit += 1
        details.append(detail)
        if metric["auroc"] is not None:
            clusters[members[0]["group"]["task_family_id"]].append(metric["auroc"])
            pairs += n
    # Pool signed AUROCs first. Maximizing direction separately for each singleton
    # pair would turn random noise into a perfect apparent predictor.
    interval = _interval(
        [sum(v) / len(v) for v in clusters.values()],
        confidence=confidence,
        min_task_clusters=min_task_clusters,
    )
    auc = interval["estimate"]
    result = {
        "auroc": auc,
        "predictive_strength_auroc": max(auc, 1 - auc) if auc is not None else None,
        "confidence_interval": interval,
        "unique_outcome_pairs": pairs,
        "support_status": "sufficient" if pairs >= min_pairs else "insufficient_pairs",
        "decision_role": "diagnostic_only",
        "per_stratum": details,
    }
    _crossfit_summary(
        by_task_only,
        crossfit_clusters,
        crossfit_pairs,
        skipped_crossfit,
        crossfit_records,
        strata,
        confidence,
        min_task_clusters,
        min_pairs,
        result,
    )
    return result


def _training_features(rows, render):
    writes = {r["record_id"]: write_tool_call_count(r) for r in rows}
    lengths = {r["record_id"]: len(render(r)) for r in rows}
    features = {
        "any_write": {key: float(n > 0) if n is not None else None for key, n in writes.items()},
        "write_count": {key: n / (1 + n) if n is not None else None for key, n in writes.items()},
        "trace_length": {key: n / (1 + n) for key, n in lengths.items()},
        "tool_calls": {r["record_id"]: (n := tool_call_count(r)) / (1 + n) for r in rows},
        "closing_claim": {r["record_id"]: float(closing_claim(r)) for r in rows},
    }
    return writes, features


def training_shortcut_audit(
    batch: list[dict],
    *,
    min_pairs: int = 10,
    confidence: float = 0.95,
    min_task_clusters: int = 8,
    render=None,
) -> dict:
    """Measure the exact sample without rejecting legitimate task evidence.

    Write presence is a train-fitted baseline. Held-out auditor predictions and
    written-run discrimination are required for the owner-selected comparison design.
    """
    if type(min_pairs) is not int or min_pairs < 1 or not 0 < confidence < 1:
        raise ValueError("Positive min_pairs and confidence in (0,1) are required")
    rows = _unique(batch)
    writes, features = _training_features(rows, render or render_record)
    probes = {
        name: stratified_probe(
            rows,
            values,
            min_pairs=min_pairs,
            confidence=confidence,
            min_task_clusters=min_task_clusters,
        )
        for name, values in features.items()
    }
    known, baseline = _fit_write_baseline(rows, writes)
    pending = []
    if not known:
        pending.append("no_write_baseline_coverage")
    if probes["any_write"]["unique_outcome_pairs"] < min_pairs:
        pending.append("insufficient_pooled_training_pairs")
    if any(sampling_key(r, allow_fixtures=True) is None for r in rows):
        pending.append("missing_model_provenance")
    if any(r["source"].get("data_kind") == "smoke_fixture" for r in rows):
        pending.append("fixture_data")
    return {
        "status": "inconclusive" if pending else "pending_auditor_comparison",
        "comparison_ready": not pending,
        "comparison_required": True,
        "release_allowed": False,
        "policy_choice": "baseline_and_written_run_checks",
        "policy_choice_status": "owner_selected",
        "sampled_records": len(batch),
        "unique_records": len(rows),
        "minimum_pooled_pairs": min_pairs,
        "confidence": confidence,
        "minimum_task_clusters": min_task_clusters,
        "pending_reasons": pending,
        "pooled_probes": probes,
        "write_baseline": baseline,
        "model_label_probe": model_label_probe(batch, sampled=True),
    }


def compare_write_baseline(
    rows, probabilities, baseline, *, min_pairs=10, confidence=0.95, min_task_clusters=8
):
    """Held-out Brier gain and discrimination within runs that attempted writes.

    Baseline probabilities are fixed from train; neither labels nor score direction
    in this evaluation cohort fit or tune the baseline.
    """
    values = {r["record_id"]: p for r, p in zip(rows, probabilities, strict=False)}
    unique = _unique(rows)
    classified = [(r, write_tool_call_count(r)) for r in unique]
    known = [(r, n) for r, n in classified if n is not None]
    buckets = (baseline or {}).get("buckets", {})
    ready = all(
        type((buckets.get(key) or {}).get("p_failed")) in (int, float)
        for key in ("no_write", "write")
    )
    y = [r["outcome"]["failed"] for r, n in known]
    p = [values[r["record_id"]] for r, n in known]
    bp = [buckets["write" if n else "no_write"]["p_failed"] for r, n in known] if ready else []
    gains = defaultdict(list)
    if ready:
        for (row, _n), score, base in zip(known, p, bp, strict=False):
            label = int(row["outcome"]["failed"])
            gains[row["group"]["task_family_id"]].append((base - label) ** 2 - (score - label) ** 2)
    gain = _interval(
        [sum(v) / len(v) for v in gains.values()],
        confidence=confidence,
        lower=-1.0,
        upper=1.0,
        min_task_clusters=min_task_clusters,
    )
    written = [r for r, n in known if n > 0]
    written_probe = stratified_probe(
        written,
        values,
        min_pairs=min_pairs,
        confidence=confidence,
        by_task_only=True,
        min_task_clusters=min_task_clusters,
    )
    pairs = min(sum(y), len(y) - sum(y))
    reasons = _comparison_reasons(ready, known, unique, pairs, min_pairs, gain, written_probe)
    return {
        "passes": not reasons,
        "reasons": reasons,
        "classified_records": len(known),
        "unclassified_records": len(unique) - len(known),
        "heldout_unique_pairs": pairs,
        "auditor": binary_metrics(y, p),
        "write_presence_baseline": binary_metrics(y, bp) if ready else None,
        "brier_gain_task_cluster_interval": gain,
        "written_runs": written_probe,
        "policy_choice": "baseline_and_written_run_checks",
        "policy_choice_status": "owner_selected",
    }


def _crossfit_fold(folds, index, values):
    train, evaluation = folds[1 - index], folds[index]
    train_ids = {r["record_id"] for r in train}
    evaluation_ids = {r["record_id"] for r in evaluation}
    if not train_ids.isdisjoint(evaluation_ids):
        raise RuntimeError("Cross-fit training and evaluation records overlap")
    train_auc = binary_metrics(
        [r["outcome"]["failed"] for r in train], [values[r["record_id"]] for r in train]
    )["auroc"]
    direction = "increasing" if train_auc > 0.5 else "decreasing" if train_auc < 0.5 else "none"
    p = [
        values[r["record_id"]]
        if direction == "increasing"
        else 1 - values[r["record_id"]]
        if direction == "decreasing"
        else 0.5
        for r in evaluation
    ]
    labels = [r["outcome"]["failed"] for r in evaluation]
    return {
        "train_records": len(train),
        "evaluation_records": len(evaluation),
        "membership_disjoint": True,
        "train_auroc": train_auc,
        "learned_direction": direction,
        "evaluation_auroc": binary_metrics(labels, p)["auroc"],
        "evaluation_pairs": min(sum(labels), len(labels) - sum(labels)),
    }


def _probe_strata(rows, values, by_task_only):
    strata = defaultdict(list)
    for row in _unique(rows):
        key = (
            row["group"]["task_family_id"]
            if by_task_only
            else sampling_key(row, allow_fixtures=True)
        )
        if key is not None and values.get(row["record_id"]) is not None:
            strata[key].append(row)
    return strata


def _crossfit_summary(
    by_task_only,
    crossfit_clusters,
    crossfit_pairs,
    skipped_crossfit,
    crossfit_records,
    strata,
    confidence,
    min_task_clusters,
    min_pairs,
    result,
):
    if not by_task_only:
        crossfit_interval = _interval(
            [sum(v) / len(v) for v in crossfit_clusters.values()],
            confidence=confidence,
            min_task_clusters=min_task_clusters,
        )
        result["cross_fitted_direction"] = {
            "auroc": crossfit_interval["estimate"],
            "confidence_interval": crossfit_interval,
            "unique_outcome_pairs": crossfit_pairs,
            "skipped_strata": skipped_crossfit,
            "evaluated_unique_records": crossfit_records,
            "total_strata": len(strata),
            "support_status": "sufficient"
            if crossfit_pairs >= min_pairs and crossfit_interval["status"] == "estimated"
            else "insufficient_evidence",
            "decision_role": "diagnostic_only",
        }


def _fit_write_baseline(rows, writes):
    known = [r for r in rows if writes[r["record_id"]] is not None]
    prior = sum(r["outcome"]["failed"] for r in known) / len(known) if known else None
    baseline = {
        "fit_split": "sampled_train_unique",
        "known_unique_records": len(known),
        "training_prior": prior,
        "buckets": {},
    }
    for bucket, condition in (("no_write", False), ("write", True)):
        members = [r for r in known if (writes[r["record_id"]] > 0) is condition]
        baseline["buckets"][bucket] = {
            "n": len(members),
            "p_failed": (sum(r["outcome"]["failed"] for r in members) + 1) / (len(members) + 2)
            if members
            else prior,
        }
    return known, baseline


def _comparison_reasons(ready, known, unique, pairs, min_pairs, gain, written_probe):
    reasons = []
    if not ready:
        reasons.append("missing_train_fitted_baseline")
    if len(known) != len(unique):
        reasons.append("unclassified_write_tools")
    if pairs < min_pairs:
        reasons.append("insufficient_heldout_pairs")
    if gain["status"] != "estimated":
        reasons.append("insufficient_brier_task_clusters")
    if gain["lower"] is None or gain["lower"] <= 0:
        reasons.append("brier_gain_not_established")
    if written_probe["unique_outcome_pairs"] < min_pairs:
        reasons.append("insufficient_written_outcome_pairs")
    written_lower = written_probe["confidence_interval"]["lower"]
    if written_probe["confidence_interval"]["status"] != "estimated":
        reasons.append("insufficient_written_task_clusters")
    if written_lower is None or written_lower <= 0.5:
        reasons.append("written_discrimination_not_established")
    return reasons
