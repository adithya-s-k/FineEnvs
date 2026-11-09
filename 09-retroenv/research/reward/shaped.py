"""v8-FINAL as a training reward, computed in the server for each graded submission.

The evaluation stays pass or fail. For training, a pass/fail reward gives no gradient on a task
the model never solves: on a 2,000-task easy/medium mix the base model solved 0 of its first 313
episodes. v8-FINAL (variants.catalogue(), measured in RESEARCH.md against the hack battery and
889 stored submissions) keeps the bands apart:

- a solve earns 0.70 to 1.0 (efficiency, diversity, agreement with the reference route);
- a failure earns 0 to 0.20, ordered by how much of its tree reaches stock through supported
  steps, so a well-formed cut that reaches nothing earns almost nothing (stopping after the first
  disconnection of a known route: 0.055, against 0.603 under the shipped grade).

The server's own verdict decides the band: a submission it passes earns at least 0.70 even where
v8's stricter support rule would not call it solved, and a submission it fails earns at most
0.20. Enable it on a server with RETROENV_SHAPED_REWARD=research.reward.shaped:v8 (the project
root on PYTHONPATH); train/jobs/run.py does for `--reward shaped`.
"""

from __future__ import annotations

from typing import Any

from .evidence import EvidenceBuilder
from .variants import catalogue, score

V8 = next(config for config in catalogue() if config.name == "v8-FINAL")


class _Stocks:
    """The slice of a TaskStore that EvidenceBuilder reads: each task's stock, by id."""

    def __init__(self) -> None:
        self.by_id: dict[str, frozenset[str]] = {}

    def stock(self, stock_id: str) -> frozenset[str]:
        return self.by_id[stock_id]

    def tasks(self, split: str) -> tuple:
        return ()


_STOCKS = _Stocks()
_BUILDERS: dict[int, EvidenceBuilder] = {}


def v8(task: Any, submission: Any, library: Any, stock: frozenset[str], passed: bool) -> dict[str, Any]:
    """{"shaped_reward", "shaped_solved", "shaped_components"} for one graded submission."""
    _STOCKS.by_id[task.stock_id] = stock
    builder = _BUILDERS.setdefault(id(library), EvidenceBuilder(library, _STOCKS))
    result = score(builder.build(task, submission), V8)
    reward = result["reward"]
    if passed:
        reward = max(reward, V8.pass_floor)
    else:
        reward = min(reward, V8.fail_cap)
    return {
        "shaped_reward": round(reward, 6),
        "shaped_solved": bool(result["solved"]),
        "shaped_components": {name: round(float(value), 4) for name, value in result["components"].items()},
    }
