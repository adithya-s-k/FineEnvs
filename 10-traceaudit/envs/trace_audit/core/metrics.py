"""Binary metrics, dev-only threshold selection and reliability bins."""

from __future__ import annotations

import math


def _check_inputs(y, p):
    if (
        len(y) != len(p)
        or any(type(v) is not bool for v in y)
        or any(type(v) not in (float, int) or not math.isfinite(v) or not 0 <= v <= 1 for v in p)
    ):
        raise ValueError("Metrics require aligned boolean labels and finite probabilities")


def binary_metrics(y: list[bool], p: list[float], threshold: float = 0.5) -> dict:
    _check_inputs(y, p)
    if not y:
        return {
            "n": 0,
            "brier": None,
            "auroc": None,
            "precision": None,
            "recall": None,
            "specificity": None,
            "balanced_accuracy": None,
            "f1": None,
            "calibration": [],
        }
    tp = sum(label and prob >= threshold for label, prob in zip(y, p, strict=False))
    fp = sum(not label and prob >= threshold for label, prob in zip(y, p, strict=False))
    fn = sum(label and prob < threshold for label, prob in zip(y, p, strict=False))
    tn = sum(not label and prob < threshold for label, prob in zip(y, p, strict=False))
    precision, recall = tp / (tp + fp) if tp + fp else 0.0, tp / (tp + fn) if tp + fn else 0.0
    auc = _rank_auc(y, p)
    positive, negative = sum(y), len(y) - sum(y)
    specificity = tn / negative if negative else None
    bins = _calibration_bins(y, p)
    return {
        "n": len(y),
        "brier": sum((prob - label) ** 2 for label, prob in zip(y, p, strict=False)) / len(y),
        "auroc": auc,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2 if positive and negative else None,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "threshold": threshold,
        "calibration": bins,
    }


def shortcut_metrics(y: list[bool], p: list[float]) -> dict:
    """Report direction-free discrimination without changing predictions or tuning on test."""
    result = binary_metrics(y, p)
    auc = result["auroc"]
    result["predictive_strength_auroc"] = max(auc, 1 - auc) if auc is not None else None
    result["signal_direction"] = (
        None
        if auc is None
        else "decreasing"
        if auc < 0.5
        else "increasing"
        if auc > 0.5
        else "none"
    )
    return result


def choose_threshold(dev_y: list[bool], dev_p: list[float]) -> dict:
    if not dev_y or len(set(dev_y)) < 2:
        return {
            "threshold": 0.5,
            "minimums_feasible": False,
            "reason": "Dev must contain both outcomes",
        }
    candidates = sorted(set([0.0, 0.5, 1.0, *dev_p]))
    _check_inputs(dev_y, dev_p)
    ordered = sorted(zip(dev_p, dev_y, strict=False), reverse=True)
    total_positive = sum(dev_y)
    total_negative = len(dev_y) - total_positive
    measured, tp, fp, index = [], 0, 0, 0
    for value in reversed(candidates):
        while index < len(ordered) and ordered[index][0] >= value:
            tp += ordered[index][1]
            fp += not ordered[index][1]
            index += 1
        precision, recall = tp / (tp + fp) if tp + fp else 0.0, tp / total_positive
        measured.append(
            {
                "threshold": value,
                "precision": precision,
                "recall": recall,
                "specificity": (total_negative - fp) / total_negative,
                "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            }
        )
    feasible = [
        m
        for m in measured
        if m["precision"] >= 0.8 and m["recall"] >= 0.8 and m["specificity"] >= 0.8
    ]
    best = max(
        feasible or measured, key=lambda m: (m["f1"], -abs(m["threshold"] - 0.5), m["threshold"])
    )
    return {
        "threshold": best["threshold"],
        "minimums_feasible": bool(feasible),
        "selection": (
            "maximum dev F1 subject to 80% precision/recall/specificity when feasible; "
            "otherwise reported fallback"
        ),
    }


def _rank_auc(y, p):
    # Mann-Whitney statistic with average ranks for ties.
    ordered = sorted(zip(p, y, strict=False))
    rank_sum, i = 0.0, 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        rank_sum += ((i + 1 + j) / 2) * sum(label for _, label in ordered[i:j])
        i = j
    positive, negative = sum(y), len(y) - sum(y)
    auc = (
        (rank_sum - positive * (positive + 1) / 2) / (positive * negative)
        if positive and negative
        else None
    )
    return auc


def _calibration_bins(y, p):
    bins = []
    for index in range(10):
        selected = [
            (label, prob)
            for label, prob in zip(y, p, strict=False)
            if min(9, int(prob * 10)) == index
        ]
        if selected:
            bins.append(
                {
                    "lower": index / 10,
                    "upper": (index + 1) / 10,
                    "n": len(selected),
                    "mean_p": sum(prob for _, prob in selected) / len(selected),
                    "failure_rate": sum(label for label, _ in selected) / len(selected),
                }
            )
    return bins
