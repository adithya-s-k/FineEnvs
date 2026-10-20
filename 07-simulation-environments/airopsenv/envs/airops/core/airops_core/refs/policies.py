"""The two fixed policies: naive (the empty plan: everything flies as early as it can) and what the airline did."""

from ..core.model import Task


def naive(task: Task) -> dict[int, dict]:
    return {}


def airline(task: Task) -> dict[int, dict]:
    """JetBlue's own cancellations (diversions count as cancelled) and actual departure times as holds."""
    return {f.id: {"cancel": f.actual.cancelled, "not_before": None if f.actual.cancelled else f.actual.dep}
            for f in task.flights}
