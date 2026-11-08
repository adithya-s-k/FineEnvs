"""How hard a design is to recreate, from its ground truth alone.

Each design gets a handful of structural features: how much there is to place,
how much copy to type, and how many of the things that trip agents up (rotation,
transparency, mixed styling, tiny or overlapping elements). The score is a
weighted mean of each feature's percentile over the whole corpus, so it is in
[0, 1] and a weight says how much that feature matters regardless of its units.
Tiers are corpus terciles of the score.

The weights are a prior. `calibrate()` checks them against real rollout rewards:
a difficulty ranking is only useful if reward actually falls as difficulty rises.
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right

from .doc import DesignDoc

DIFFICULTY_VERSION = 1
TIERS = ("easy", "medium", "hard")

# Feature -> weight. Sums to 1.
WEIGHTS = {
    "n_elements": 0.25,   # things to place and stack
    "chars": 0.12,        # copy to type exactly
    "n_fonts": 0.08,      # families to identify
    "styled": 0.07,       # text boxes with mixed colour/weight runs
    "rotated": 0.08,      # rotated elements: angle has to be read off the image
    "semi": 0.05,         # partly transparent elements
    "small": 0.08,        # elements under 5% of the canvas side: easy to miss
    "tiny_text": 0.07,    # text under 2% of the canvas side: hard to read
    "overlap": 0.10,      # share of element pairs that overlap: stacking order matters
    "reuse": 0.03,        # one asset placed several times
    "cover": 0.07,        # total element area over canvas area: busy layouts
}
assert abs(sum(WEIGHTS.values()) - 1) < 1e-9


def features(doc: DesignDoc) -> dict:
    """Structural features of one ground-truth document."""
    els = doc.elements
    side = max(doc.width, doc.height)
    area = doc.width * doc.height
    text = [e for e in els if e.kind == "text"]
    images = [e for e in els if e.kind == "image"]

    def clipped(e):
        x0, y0 = max(0.0, e.x), max(0.0, e.y)
        x1, y1 = min(doc.width, e.x + e.width), min(doc.height, e.y + e.height)
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)

    # Overlap among elements that are not full-canvas backdrops: those overlap everything.
    parts = [e for e in els if clipped(e) < 0.9 * area]
    pairs = overlapping = 0
    for i, a in enumerate(parts):
        for b in parts[i + 1:]:
            pairs += 1
            if (a.x < b.x + b.width and b.x < a.x + a.width
                    and a.y < b.y + b.height and b.y < a.y + a.height):
                overlapping += 1
    return {
        "n_elements": len(els),
        "n_text": len(text),
        "n_images": len(images),
        "chars": sum(len(e.text) for e in text),
        "lines": sum(e.text.count("\n") + 1 for e in text),
        "n_fonts": len({e.font_family for e in text}),
        "styled": sum(1 for e in text if e.runs),
        "rotated": sum(1 for e in els if abs(e.angle) > 0.5),
        "semi": sum(1 for e in els if e.opacity < 0.99),
        "small": sum(1 for e in els if max(e.width, e.height) < 0.05 * side),
        "tiny_text": sum(1 for e in text if e.font_size < 0.02 * side),
        "overlap": round(overlapping / pairs, 4) if pairs else 0.0,
        "reuse": len(images) - len({e.asset_id for e in images}),
        "cover": round(sum(clipped(e) for e in els) / area, 4),
        "aspect": round(doc.width / doc.height, 4),
    }


def shape(aspect: float) -> str:
    """Canvas shape bucket: the layout problem differs more by shape than by format name."""
    if aspect < 0.8:
        return "tall"
    if aspect <= 1.25:
        return "square"
    if aspect <= 2.2:
        return "wide"
    return "banner"


class Scorer:
    """Percentile-rank scoring against a fixed reference population."""

    def __init__(self, population: list[dict], weights: dict | None = None):
        self.weights = dict(weights or WEIGHTS)
        self.sorted = {k: sorted(f[k] for f in population) for k in self.weights}
        scores = sorted(self.score(f) for f in population)
        n = len(scores)
        self.cuts = [scores[n // 3], scores[(2 * n) // 3]] if n else [1 / 3, 2 / 3]

    def _pct(self, key: str, value: float) -> float:
        xs = self.sorted[key]
        if not xs:
            return 0.0
        # Mid-rank for ties, so a zero shared by 80% of designs sits at 0.4, not 0 or 0.8.
        lo, hi = bisect_left(xs, value), bisect_right(xs, value)
        return (lo + hi) / 2 / len(xs)

    def score(self, f: dict) -> float:
        return round(sum(w * self._pct(k, f[k]) for k, w in self.weights.items()), 4)

    def tier(self, score: float) -> str:
        return TIERS[0] if score < self.cuts[0] else TIERS[1] if score < self.cuts[1] else TIERS[2]

    def describe(self) -> dict:
        return {"version": DIFFICULTY_VERSION, "weights": self.weights,
                "cuts": [round(c, 4) for c in self.cuts]}


def spearman(xs: list[float], ys: list[float]) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    sx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    sy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return cov / (sx * sy) if sx and sy else 0.0


def calibrate(rows: list[dict], rewards: dict[str, list[float]], weights: dict | None = None,
              prior_weight: float = 0.5) -> dict:
    """Check the ranking against real rewards and propose weights.

    `rows` are index rows (features + score + tier) for the evaluated tasks;
    `rewards` maps task_id -> rewards from rollouts (any models/surfaces). A
    feature earns weight in proportion to how strongly it predicts a *lower*
    reward; the proposal is blended with the prior so one pilot cannot swing it.
    """
    weights = dict(weights or WEIGHTS)
    rows = [r for r in rows if rewards.get(r["task_id"])]
    mean = {r["task_id"]: sum(rewards[r["task_id"]]) / len(rewards[r["task_id"]]) for r in rows}
    ys = [mean[r["task_id"]] for r in rows]
    per_feature = {k: round(spearman([r[k] for r in rows], ys), 3) for k in weights}
    signal = {k: max(0.0, -rho) for k, rho in per_feature.items()}
    total = sum(signal.values())
    proposed = {k: round(prior_weight * weights[k] + (1 - prior_weight) * (signal[k] / total if total else weights[k]), 4)
                for k in weights}
    by_tier = {t: [mean[r["task_id"]] for r in rows if r["tier"] == t] for t in TIERS}
    return {
        "n_tasks": len(rows),
        "spearman_score_vs_reward": round(spearman([r["score"] for r in rows], ys), 3),
        "mean_reward_by_tier": {t: round(sum(v) / len(v), 4) for t, v in by_tier.items() if v},
        "monotone": all(sum(by_tier[a]) / max(1, len(by_tier[a])) >= sum(by_tier[b]) / max(1, len(by_tier[b]))
                        for a, b in zip(TIERS, TIERS[1:], strict=False)),
        "spearman_feature_vs_reward": per_feature,
        "proposed_weights": proposed,
    }
