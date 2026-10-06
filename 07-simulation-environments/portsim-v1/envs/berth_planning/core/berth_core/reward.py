"""The terminal reward, from a submitted plan and the task's two stored reference costs.

    unreadable or infeasible   0.2 x (ships with no violation / all ships)        in [0, 0.2)
    feasible, worse than naive 0.2 + 0.4 x naive / cost                           in (0.2, 0.6)
    feasible, naive or better  0.6 + 0.4 x (naive - cost) / (naive - optimum)     in [0.6, 1.0]

`naive` is the cost of the hurried re-plan in solve.naive_replan (keep slots that still work, push the rest to the
next free hour); `optimum` is the CP-SAT optimum stored with the task. Bands meet, so every failed group in GRPO still
orders its members, and anchoring the top band on the naive plan cancels delay that late arrivals make unavoidable.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .check import evaluate, unavoidable_cost
from .model import Plan, Task

INFEASIBLE_CAP = 0.2
NAIVE_LEVEL = 0.6


@dataclass
class Grade:
    reward: float
    feasible: bool
    clean_fraction: float      # ships with no violation / all ships
    quality: float             # 0 at the naive plan or worse, 1 at the optimum (0 when infeasible)
    cost: int | None
    delay_cost: int | None
    moves: int | None
    naive_cost: int
    optimal_cost: int
    violations: list[dict]
    parse_problems: list[str]

    def as_dict(self) -> dict:
        return asdict(self)


def score(cost: int | None, feasible: bool, clean_fraction: float, naive: int, optimum: int) -> tuple[float, float]:
    """(reward, quality) from the plan's cost and the two anchors."""
    if not feasible or cost is None:
        return INFEASIBLE_CAP * clean_fraction, 0.0
    if cost > naive:
        return INFEASIBLE_CAP + (NAIVE_LEVEL - INFEASIBLE_CAP) * naive / cost, 0.0
    gap = naive - optimum
    quality = 1.0 if gap <= 0 else min(1.0, (naive - cost) / gap)
    return NAIVE_LEVEL + (1.0 - NAIVE_LEVEL) * quality, quality


GAP_SCALE = 0.25


def score_v2(cost: int | None, feasible: bool, clean_fraction: float, naive: int, optimum: int) -> tuple[float, float]:
    """Reward v2: steep in the gap to the optimum, so near-optimal plans are told apart.

        infeasible   0.2 x clean fraction
        feasible     0.2 + 0.8 x exp(-g / 0.25),  g = (cost - optimum) / (naive - optimum)

    g = 0 is the optimum (1.0); g = 0.05 gives 0.86; g = 0.25 gives 0.49; the naive plan (g = 1) gives 0.21.
    A plan below the stored best-known cost (possible when it is not proven optimal) scores 1.0."""
    if not feasible or cost is None:
        return INFEASIBLE_CAP * clean_fraction, 0.0
    g = max(0.0, (cost - optimum) / max(1, naive - optimum))
    quality = math.exp(-g / GAP_SCALE)
    return INFEASIBLE_CAP + (1.0 - INFEASIBLE_CAP) * quality, quality


GAP_TAU = 0.5


def score_v3(cost: int | None, feasible: bool, clean_fraction: float, optimum: int, gap_k: int,
             unavoidable: int = 0) -> tuple[float, float]:
    """Reward v3 (the dock-v1 packs): the gap to the proven optimum, measured against the part of the cost a planner
    controls (the optimum minus a provable floor; check.unavoidable_cost).

        infeasible   0.2 x clean fraction                                                in [0, 0.2)
        feasible     0.2 + 0.8 x exp(-g / 0.5),  g = (cost - optimum) / (optimum - floor + gap_k)   in (0.2, 1]

    Any feasible plan beats any infeasible one; at or below the stored optimum scores 1.0. With the optimum 120 above
    its floor (gap_k 100), a plan 22 above the optimum scores 0.86, 110 above 0.49, 220 above 0.31."""
    if not feasible or cost is None:
        return INFEASIBLE_CAP * clean_fraction, 0.0
    g = max(0, cost - optimum) / (max(0, optimum - unavoidable) + gap_k)
    quality = math.exp(-g / GAP_TAU)
    return INFEASIBLE_CAP + (1.0 - INFEASIBLE_CAP) * quality, quality


def grade(task: Task, plan: Plan, parse_problems: list[str] | None = None) -> Grade:
    ref = task.reference
    naive, optimum = int(ref["naive_cost"]), int(ref["optimal_cost"])
    res = evaluate(task, plan)
    problems = list(parse_problems or [])
    feasible = res.feasible and not problems
    # entry problems (duplicates, unknown ships, bad values) count as unclean units, so a malformed plan always
    # stays strictly below the 0.2 floor of a feasible one
    clean = sum(1 for r in res.ships if not r.problems) / max(1, len(res.ships) + len(problems))
    version = task.rules.get("reward")
    if version == 3:
        reward, quality = score_v3(res.cost, feasible, clean, optimum, int(task.rules.get("gap_k", 100)),
                                   unavoidable_cost(task))
    else:
        reward, quality = (score_v2 if version == 2 else score)(res.cost, feasible, clean, naive, optimum)
    return Grade(round(reward, 6), feasible, round(clean, 4), round(quality, 6), res.cost if feasible else None,
                 res.delay_cost if feasible else None, res.moves if feasible else None, naive, optimum,
                 res.violations, problems)
