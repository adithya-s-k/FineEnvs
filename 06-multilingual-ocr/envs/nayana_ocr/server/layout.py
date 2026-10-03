"""Class-aware maximum-cardinality matching; mean F1 over IoU .50:.05:.95."""

import json
import math

from ..data.schema import LAYOUT_LABELS

POLICY = "layout-mean-f1-iou-50-95-v1"


def parse_regions(text, width=None, height=None):
    if len(text) > 65536:
        raise ValueError("Layout answer exceeds 65536 characters")

    def unique_object(pairs):
        if len(dict(pairs)) != len(pairs):
            raise ValueError("Duplicate JSON keys")
        return dict(pairs)

    value = json.loads(text, object_pairs_hook=unique_object)
    if not isinstance(value, list) or len(value) > 512:
        raise ValueError("Expected at most 512 regions")
    for region in value:
        if not isinstance(region, dict) or set(region) != {"label", "bbox"}:
            raise ValueError("Each region needs exactly label and bbox")
        if region["label"] not in LAYOUT_LABELS:
            raise ValueError("Unknown layout label")
        box = region["bbox"]
        if (
            not isinstance(box, list)
            or len(box) != 4
            or not all(type(v) in (int, float) and math.isfinite(v) for v in box)
        ):
            raise ValueError("Expected four finite numeric coordinates")
        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 and 0 <= y0 < y1):
            raise ValueError("Invalid box bounds")
        if width is not None and (x1 > width or y1 > height):
            raise ValueError("Box outside image")
    return value


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union


def matched_count(overlaps, threshold):
    """Augmenting paths find the maximum number of one-to-one qualifying pairs."""
    matches = {}

    def visit(predicted, seen):
        for target, overlap in enumerate(overlaps[predicted]):
            if overlap < threshold or target in seen:
                continue
            seen.add(target)
            if target not in matches or visit(matches[target], seen):
                matches[target] = predicted
                return True
        return False

    return sum(visit(p, set()) for p in range(len(overlaps)))


def score_layout(prediction, reference, width=None, height=None):
    targets = parse_regions(reference, width, height)
    if not targets:
        raise ValueError("Layout reference is empty")
    try:
        predicted = parse_regions(prediction, width, height)
    except (ValueError, TypeError, RecursionError):
        return 0.0, {"valid_format": False, "mean_f1": 0.0, "exact_match": False}
    overlaps = [
        [
            iou(p["bbox"], t["bbox"]) if p["label"] == t["label"] else 0.0
            for t in targets
        ]
        for p in predicted
    ]
    matches = [matched_count(overlaps, (50 + 5 * i) / 100) for i in range(10)]
    denominator = len(predicted) + len(targets)
    reward = sum(2 * count / denominator for count in matches) / 10
    return reward, {
        "valid_format": True,
        "exact_match": sorted((r["label"], *r["bbox"]) for r in predicted)
        == sorted((r["label"], *r["bbox"]) for r in targets),
        "mean_f1": reward,
        "precision_at_50": matches[0] / len(predicted) if predicted else 0.0,
        "recall_at_50": matches[0] / len(targets),
        "f1_at_50": 2 * matches[0] / denominator,
        "f1_at_75": 2 * matches[5] / denominator,
        "predicted_regions": len(predicted),
        "reference_regions": len(targets),
    }
