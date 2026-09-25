"""Deterministic short-answer grader v2 — NO LLM judge.

Tiers (all offline/deterministic):
  1. Exact (case-insensitive, whitespace-collapsed)
  2. Numeric (abs/rel tolerance from ATOL/RTOL; gated to clean-number golds)
     + percent<->fraction bridge (e.g. gold 96.00 == pred 0.9621)
  3. List (comma-separated): split, strip, order-insensitive; numeric-tolerant per element
     (fixes 'a, b' vs 'a,b' spacing and reordering)
  4. Math-Verify (symbolic/numeric equivalence)
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass

_NUMERIC_RE = re.compile(r"-?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?")


@dataclass
class GradeResult:
    reward: float
    method: str


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _to_float(s: str):
    if not s:
        return None
    m = _NUMERIC_RE.search(str(s).replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def _num_close(g, c, abs_tol, rel_tol) -> bool:
    return abs(g - c) <= abs_tol or abs(g - c) / max(abs(g), 1e-9) <= rel_tol


def _is_clean_number(s: str) -> bool:
    t = (s or "").strip().strip("%$").strip().replace(",", "")
    return bool(_NUMERIC_RE.fullmatch(t))


def _elem_match(a, b, abs_tol, rel_tol) -> bool:
    if _normalize(a) == _normalize(b):
        return True
    fa, fb = _to_float(a), _to_float(b)
    if fa is not None and fb is not None:
        return _num_close(fa, fb, abs_tol, rel_tol)
    return False


def _list_match(gold: str, cand: str, abs_tol, rel_tol) -> bool:
    gl = [x.strip() for x in gold.split(",") if x.strip()]
    cl = [x.strip() for x in cand.split(",") if x.strip()]
    if len(gl) < 2 or len(gl) != len(cl):
        return False
    for gs, cs in ((gl, cl), (sorted(gl, key=str.lower), sorted(cl, key=str.lower))):
        if all(_elem_match(a, b, abs_tol, rel_tol) for a, b in zip(gs, cs)):
            return True
    return False


def _math_verify_match(gold: str, candidate: str) -> bool:
    try:
        from math_verify import parse, verify

        return bool(verify(parse(gold), parse(candidate), timeout_seconds=5))
    except Exception:
        return False


def grade(
    gold: str,
    candidate: str,
    *,
    question: str = "",
    reward_mode: str = "",
    judge: bool = True,
    judge_model=None,
    rel_tol: float = 1e-3,
    abs_tol: float = 1e-3,
) -> GradeResult:
    if not gold or candidate is None:
        return GradeResult(0.0, "miss")

    # Tier 1: exact
    if _normalize(gold) == _normalize(candidate):
        return GradeResult(1.0, "exact")

    # Tier 2: numeric (clean-number gold) + percent/fraction bridge
    if reward_mode in ("numeric", "flexible") or _is_clean_number(gold):
        g, c = _to_float(gold), _to_float(candidate)
        if g is not None and c is not None:
            if _num_close(g, c, abs_tol, rel_tol):
                return GradeResult(1.0, "numeric")
            # percent<->fraction: one side is a fraction (<1), the other a percent (>=1)
            if (0 < abs(c) < 1 <= abs(g)) or (0 < abs(g) < 1 <= abs(c)):
                if _num_close(g, c * 100, abs_tol, rel_tol) or _num_close(
                    g, c / 100, abs_tol, rel_tol
                ):
                    return GradeResult(1.0, "numeric_scaled")

    # Tier 3: list (comma-separated), order-insensitive, per-element tolerant
    if reward_mode in ("list", "list_csv") or ("," in gold and "," in candidate):
        if _list_match(gold, candidate, abs_tol, rel_tol):
            return GradeResult(1.0, "list")

    # Tier 4: math-verify
    if _math_verify_match(gold, candidate):
        return GradeResult(1.0, "math_verify")

    return GradeResult(0.0, "miss")


def _tool_calls():
    try:
        with open("/workdir/.n_tool_calls") as fh:
            return int(fh.read().strip())
    except (OSError, ValueError):
        pass
    raw = os.environ.get("N_TOOL_CALLS")
    if raw is not None:
        try:
            return int(str(raw).strip())
        except ValueError:
            return None
    return None


def _tool_efficiency(n):
    if n is None:
        return None
    budget = float(os.environ.get("TOOL_BUDGET", "15") or "15")
    if budget <= 0:
        return None
    return max(0.0, min(1.0, 1.0 - n / budget))


def _tols():
    def _f(name):
        try:
            return float(os.environ.get(name, "") or 1e-3)
        except ValueError:
            return 1e-3

    return _f("ATOL"), _f("RTOL")


def main_cli() -> int:
    gold = (os.environ.get("EXPECTED_ANSWER") or "").strip()
    question = (os.environ.get("QUESTION") or "").strip()
    candidate = sys.stdin.read().strip()
    at, rt = _tols()
    r = grade(
        gold,
        candidate,
        question=question,
        reward_mode=os.environ.get("REWARD_MODE", "") or "",
        abs_tol=at,
        rel_tol=rt,
    )
    print(f"{r.reward:.1f}")
    print(
        f"[grader] gold={gold!r} pred={candidate[:80]!r} reward={r.reward} method={r.method}",
        file=sys.stderr,
    )
    return 0


def main_json() -> int:
    import json

    gold = (os.environ.get("EXPECTED_ANSWER") or "").strip()
    question = (os.environ.get("QUESTION") or "").strip()
    candidate = sys.stdin.read().strip()
    if not candidate:
        print(
            json.dumps({"correctness": 0.0, "submission": 0.0, "tool_efficiency": 0.0})
        )
        return 0
    at, rt = _tols()
    r = grade(
        gold,
        candidate,
        question=question,
        reward_mode=os.environ.get("REWARD_MODE", "") or "",
        abs_tol=at,
        rel_tol=rt,
    )
    n = _tool_calls()
    print(
        json.dumps(
            {
                "correctness": float(r.reward),
                "submission": 1.0,
                "tool_efficiency": _tool_efficiency(n),
            }
        )
    )
    print(
        f"[grader] gold={gold!r} pred={candidate[:80]!r} correctness={r.reward} method={r.method}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    if "--json" in sys.argv[1:]:
        raise SystemExit(main_json())
    raise SystemExit(main_cli())
