# Copyright 2026 Xiaomi / the verl authors. Apache-2.0 (see ../LICENSE-verl.txt).
# `parse_verdict` copied unchanged from XiaomiMiMo/verl recipes/design/webdev/eval_mode.py.
import json
import re

from .eval_rubric import ALL_KEYS, VISUAL_KEYS

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_verdict(raw_text: str) -> dict:
    """Judge JSON -> per-dimension scores, ``visual``, ``score``, ``reason``.

    Returns ``{"_failed": why}`` when it cannot be parsed. A missing dimension counts as a
    parse failure rather than defaulting to 0: defaulting would turn "the judge forgot to
    mention a dimension" into "the page scores 0 on it", inventing a low score out of
    nothing.
    """
    m = _JSON_RE.search(raw_text or "")
    if not m:
        return {"_failed": f"no json in judge output: {(raw_text or '')[:160]}"}
    try:
        data = json.loads(m.group(0))
    except Exception as e:  # noqa: BLE001
        return {"_failed": f"json error: {type(e).__name__}: {e}"}
    dims: dict[str, float] = {}
    for k in ALL_KEYS:
        v = data.get(k)
        if v is None:
            return {"_failed": f"judge omitted dimension {k!r}"}
        try:
            dims[k] = max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return {"_failed": f"dimension {k!r} is not a number: {v!r}"}
    visual = sum(dims[k] for k in VISUAL_KEYS) / len(VISUAL_KEYS)
    score = (visual + dims["query_fulfillment"] + dims["premium_assets"]) / 3.0
    return {
        "dims": dims,
        "visual": round(visual, 4),
        "score": round(score, 4),
        "reason": str(data.get("reason", ""))[:500],
    }
