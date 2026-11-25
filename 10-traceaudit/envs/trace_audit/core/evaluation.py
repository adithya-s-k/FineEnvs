"""Offline baseline evaluation and calibration plots."""

from __future__ import annotations

from collections import Counter

from . import TraceAuditEnv, parse_action
from .baselines import (
    TfidfCentroid,
    closing_claim,
    fit_baselines,
    tool_call_count,
    write_tool_call_count,
)
from .gate import evaluate_gate
from .metrics import binary_metrics, choose_threshold, shortcut_metrics
from .records import DataError, validate_collection
from .render import SYSTEM_PROMPT
from .sampling import balanced_batch
from .shortcuts import model_label_probe, training_shortcut_audit


def calibration_svg(report, path):
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="510" viewBox="0 0 720 510">',
        '<rect width="720" height="510" fill="white"/>'
        '<text x="50" y="30" font-family="sans-serif" font-size="18">'
        "Test reliability (synthetic if fixture mode)</text>",
        '<path d="M60 430 L460 30 M60 30 V430 H460" fill="none" stroke="#999"/>',
        '<text x="140" y="475" font-family="sans-serif">'
        "Mean predicted failure probability</text>"
        '<text x="6" y="240" font-family="sans-serif" transform="rotate(-90 16 240)">'
        "Observed failure rate</text>",
    ]
    for i, (name, result) in enumerate(report["baselines"].items()):
        color = colors[i % len(colors)]
        points = " ".join(
            f"{60 + 400 * b['mean_p']:.2f},{430 - 400 * b['failure_rate']:.2f}"
            for b in result["test"]["calibration"]
        )
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
        for b in result["test"]["calibration"]:
            parts.append(
                f'<circle cx="{60 + 400 * b["mean_p"]:.2f}" '
                f'cy="{430 - 400 * b["failure_rate"]:.2f}" r="4" fill="{color}"/>'
            )
        parts.append(
            f'<text x="480" y="{60 + i * 26}" fill="{color}" font-size="12" '
            f'font-family="sans-serif">{name}</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def evaluate(
    records,
    *,
    allow_fixtures=False,
    predictions=None,
    max_tokens=32000,
    token_counter=None,
    training_size=None,
    source_cap=0.5,
    sampling_seed=42,
    max_record_reuse=1,
    validated=False,
):
    if not validated:
        validate_collection(records)
    splits, exclusions, environments, prompts = {}, [], {}, {}
    for split in ("train", "dev", "test"):
        env = TraceAuditEnv(
            records,
            split=split,
            allow_fixtures=allow_fixtures,
            max_tokens=max_tokens,
            token_counter=token_counter,
            validated=True,
        )
        environments[split] = env
        splits[split] = list(env.records)
        prompts.update({row["record_id"]: env.prompt(row["record_id"]) for row in env.records})
        exclusions.extend(
            {"split": split, "source": source, "reason": reason, "n": n}
            for (source, reason), n in env.exclusions.items()
        )
    if len({r["outcome"]["failed"] for r in splits["train"]}) < 2:
        raise ValueError("Train needs both outcomes for classifier probes")

    def render(row):
        return prompts[row["record_id"]]

    methods = fit_baselines(splits["train"], render=render)
    _add_judge(splits, predictions, methods)
    report = {
        "fixture_only": all(
            r["source"].get("data_kind") == "smoke_fixture"
            for rows in splits.values()
            for r in rows
        ),
        "baselines": {},
        "exclusions": exclusions,
        "prompted_judge": "saved predictions loaded"
        if predictions is not None
        else "pending: model and budget unspecified",
        "counting": getattr(
            token_counter, "encoding_name", "UTF-8 byte upper bound, no truncation"
        ),
        "max_tokens": max_tokens,
    }
    scores = _baseline_reports(splits, methods, report)
    _source_probe(splits, report, render)
    training_audit = _training_audit(
        splits,
        training_size,
        source_cap,
        sampling_seed,
        max_record_reuse,
        allow_fixtures,
        audit_records=records,
        render=render,
    )
    report["shortcut_audit"]["paired_training"] = training_audit
    _write_probes(splits, report)
    _localization(splits, predictions, report)
    _feature_probes(splits, report, render)
    report["rl_gate"] = evaluate_gate(
        splits["test"],
        {name: p["test"] for name, p in scores.items()},
        {
            name: result["threshold_selection"]["threshold"]
            for name, result in report["baselines"].items()
        },
        training_audit=training_audit,
        validated=True,
    )
    _episode(environments, splits, methods, report)
    return report


def _add_judge(splits, predictions, methods):
    if predictions is not None:
        # Saved judge output is joined privately; every evaluated record must have one response.
        expected = {r["record_id"] for rows in splits.values() for r in rows}
        if set(predictions) != expected:
            raise ValueError(
                "Saved judge predictions must cover exactly all admitted train/dev/test records"
            )
        for rows in splits.values():
            for row in rows:
                action = parse_action(
                    predictions[row["record_id"]],
                    {s["step_id"] for s in row["trajectory"]["steps"]},
                )
                if action.p_failed is None:
                    raise ValueError(
                        "Malformed saved judge probability; "
                        "fix predictions before comparing classifiers"
                    )
        methods["prompted_judge"] = lambda row: (
            parse_action(
                predictions[row["record_id"]], {s["step_id"] for s in row["trajectory"]["steps"]}
            ).p_failed
        )


def _baseline_reports(splits, methods, report):
    scores = {}
    for name, predict in methods.items():
        p = {split: [predict(row) for row in rows] for split, rows in splits.items()}
        labels = {
            split: [row["outcome"]["failed"] for row in rows] for split, rows in splits.items()
        }
        selection = choose_threshold(labels["dev"], p["dev"])
        threshold = selection["threshold"]
        result = {"threshold_selection": selection}
        _split_metrics(splits, labels, p, threshold, result)
        report["baselines"][name] = result
        scores[name] = p
    return scores


def _source_probe(splits, report, render):
    # Fit source probes on rendered text; source labels are used by the offline audit.
    source_model = TfidfCentroid(
        [render(r) for r in splits["train"]],
        [r["source"]["dataset"] for r in splits["train"]],
    )
    source_probe = {}

    def predict(row):
        scores = source_model.scores(render(row))
        return max(scores, key=scores.get)

    for split in ("dev", "test"):
        seen = [r for r in splits[split] if r["source"]["dataset"] in source_model.labels]
        accuracy = (
            sum(predict(r) == r["source"]["dataset"] for r in seen) / len(seen) if seen else None
        )
        source_probe[split] = {
            "n_seen_sources": len(seen),
            "accuracy": accuracy,
            "majority_chance": max(Counter(r["source"]["dataset"] for r in seen).values())
            / len(seen)
            if seen
            else None,
            "unseen_source_records": len(splits[split]) - len(seen),
        }
    report["shortcut_audit"] = {"source_probe": source_probe}
    report["shortcut_audit"]["model_label_probe"] = {
        split: model_label_probe(rows) for split, rows in splits.items()
    }


def _training_audit(
    splits,
    training_size,
    source_cap,
    sampling_seed,
    max_record_reuse,
    allow_fixtures,
    *,
    audit_records=None,
    render=None,
):
    try:
        sampled, sampling = balanced_batch(
            splits["train"],
            training_size,
            source_cap=source_cap,
            seed=sampling_seed,
            max_record_reuse=max_record_reuse,
            allow_fixtures=allow_fixtures,
            validated=True,
            audit_records=audit_records,
        )
        training_audit = training_shortcut_audit(sampled, render=render)
        training_audit["sampling"] = sampling
    except DataError as exc:
        training_audit = {
            "status": "inconclusive",
            "release_allowed": False,
            "reason": "training_sample_unavailable",
            "detail": str(exc),
        }
    return training_audit


def _write_probes(splits, report):
    # Report write shortcuts within domain/source on every split, rather than
    # hiding airline training-label artifacts inside a mixed held-out aggregate.
    report["shortcut_audit"]["write_calls"] = {}
    for split, rows in splits.items():
        cohorts = {}
        for source, domain in sorted(
            {(r["source"]["dataset"], r["source"].get("domain") or "") for r in rows}
        ):
            cohorts[f"{source}:{domain}"] = _write_cohort(rows, source, domain)
        report["shortcut_audit"]["write_calls"][split] = cohorts


def _localization(splits, predictions, report):
    human_test = [
        r for r in splits["test"] if r["outcome"]["label_source"] in {"expert", "trace_evident"}
    ]
    gold = [(r, f) for r in human_test for f in r["failures"] if f["critical"]]
    categories = Counter(f["category"] for r in splits["train"] for f in r["failures"])
    majority = categories.most_common(1)[0][0] if categories else None
    report["localization_diagnostics"] = {
        "human_test_with_critical_step": len(gold),
        "reward_trained": False,
        "interpretation": (
            "Optional v1 fields are unscored; these are diagnostics only, "
            "not validated v2 capabilities."
        ),
        "fixed_last_step_accuracy": sum(
            f["step_id"] == len(r["trajectory"]["steps"]) for r, f in gold
        )
        / len(gold)
        if gold
        else None,
        "train_majority_category": majority,
        "majority_category_accuracy": sum(f["category"] == majority for _, f in gold) / len(gold)
        if gold and majority
        else None,
    }
    _judge_localization(gold, predictions, report)


def _feature_probes(splits, report, render):
    report["shortcut_audit"]["original_feature_probes"] = {split: {} for split in splits}
    for name, feature in (
        ("trace_length", lambda r: len(render(r))),
        ("tool_calls", tool_call_count),
        ("closing_claim", lambda r: int(closing_claim(r))),
    ):
        for split, rows in splits.items():
            values = [feature(r) for r in rows]
            span = max(values) - min(values)
            prob = [(v - min(values)) / span if span else 0.5 for v in values]
            report["shortcut_audit"]["original_feature_probes"][split][name] = shortcut_metrics(
                [r["outcome"]["failed"] for r in rows], prob
            )
        report["shortcut_audit"][name] = report["shortcut_audit"]["original_feature_probes"][
            "test"
        ][name]


def _episode(environments, splits, methods, report):
    # A repeated submission earns zero and invalidates the pending reward for trainer consumption.
    env = environments["test"]
    selected = splits["test"][0]
    prompt = env.reset(record_id=selected["record_id"])
    response = {"p_failed": methods["training_prior"](selected)}
    result = env.step(response)
    second_reward = env.step({"p_failed": 0.5}).reward
    if not result.done or second_reward != 0:
        raise RuntimeError("Episode must terminate; repeated submissions must earn zero")
    report["episode"] = {
        "system_prompt": SYSTEM_PROMPT,
        "prompt": prompt,
        "response": response,
        "reward": result.reward,
    }
    report["episode_smoke"] = {
        "terminal": result.done,
        "reward": result.reward,
        "second_submission_reward": 0,
    }


def _split_metrics(splits, labels, p, threshold, result):
    for split in ("dev", "test"):
        rows = splits[split]
        result[split] = binary_metrics(labels[split], p[split], threshold)
        _group_metrics(rows, labels, p, split, threshold, result)
        indexes = [i for i, row in enumerate(rows) if labels[split][i] and closing_claim(row)]
        result[split]["false_success"] = binary_metrics(
            [labels[split][i] for i in indexes], [p[split][i] for i in indexes], threshold
        )


def _group_metrics(rows, labels, p, split, threshold, result):
    for field, key_fn in (
        ("per_source", lambda r: r["source"]["dataset"]),
        ("per_label_source", lambda r: r["outcome"]["label_source"]),
    ):
        result[split][field] = {}
        for key in sorted({key_fn(row) for row in rows}):
            indexes = [i for i, row in enumerate(rows) if key_fn(row) == key]
            result[split][field][key] = binary_metrics(
                [labels[split][i] for i in indexes],
                [p[split][i] for i in indexes],
                threshold,
            )


def _write_cohort(rows, source, domain):
    selected = [
        r
        for r in rows
        if r["source"]["dataset"] == source and (r["source"].get("domain") or "") == domain
    ]
    known = [(r, write_tool_call_count(r)) for r in selected]
    known = [(r, n) for r, n in known if n is not None]
    labels = [r["outcome"]["failed"] for r, _ in known]
    return {
        "classified_records": len(known),
        "unclassified_records": len(selected) - len(known),
        "any_write": shortcut_metrics(labels, [float(n > 0) for _, n in known]),
        "write_count": shortcut_metrics(labels, [n / (1 + n) for _, n in known]),
    }


def _judge_localization(gold, predictions, report):
    if predictions is not None:
        assessed = [
            (
                r,
                f,
                parse_action(
                    predictions[r["record_id"]], {s["step_id"] for s in r["trajectory"]["steps"]}
                ),
            )
            for r, f in gold
        ]
        report["localization_diagnostics"]["judge"] = {
            "critical_step_accuracy": sum(a.critical_step == f["step_id"] for _, f, a in assessed)
            / len(assessed)
            if assessed
            else None,
            "category_accuracy": sum(a.category == f["category"] for _, f, a in assessed)
            / len(assessed)
            if assessed
            else None,
            "critical_step_coverage": sum(a.critical_step is not None for _, _, a in assessed)
            / len(assessed)
            if assessed
            else None,
            "category_coverage": sum(a.category is not None for _, _, a in assessed) / len(assessed)
            if assessed
            else None,
        }
