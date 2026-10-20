"""The grade: never below 0.2 (the env never strands a flight), 1 at the anchor."""

import math

KAPPA = 0.02
TEMPERATURE = 0.5


def reward(cost: float, anchor: float, floor: float) -> float:
    """0.2 + 0.8·exp(−g/0.5), g = max(0, cost − anchor) / (anchor − floor + 0.02·anchor): the excess over the anchor
    in units of the anchor's own avoidable cost."""
    g = max(0.0, cost - anchor) / max(1.0, anchor - floor + KAPPA * anchor)
    return 0.2 + 0.8 * math.exp(-g / TEMPERATURE)
