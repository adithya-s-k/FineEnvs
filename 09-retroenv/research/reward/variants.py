"""Candidate reward shapes, scored from cached evidence.

`analyse` recomputes every verdict (support, validity, progress) under one
support rule; the scorers are then pure arithmetic over that analysis, so a
variant sweep costs microseconds per submission.

Two families:

* `weighted`  - the shipped additive form, with the knobs that remove its free
  components, gate step credit on progress toward stock, and cap failures.
* `anchored`  - outcome first: a solve starts at `pass_floor` and earns the rest
  from efficiency, diversity and reference agreement; a failure is confined to
  `[0, fail_cap]` and ordered by how much of the route is genuinely grounded.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from .evidence import RouteEvidence, SubmissionEvidence

SHIPPED_WEIGHTS = {
    "parse": 0.02,
    "structure": 0.08,
    "steps": 0.35,
    "stock": 0.15,
    "constraints": 0.10,
    "efficiency": 0.10,
    "diversity": 0.05,
    "reference_similarity": 0.10,
    "reference_match": 0.05,
}
FALSE_STOCK_CAP = 0.40
OFF_TARGET_CAP = 0.10


@dataclass(frozen=True)
class RewardConfig:
    name: str
    family: str = "weighted"
    min_template_count: int = 5
    support_rule: str = "any"
    # weighted family
    weights: dict[str, float] = field(default_factory=lambda: dict(SHIPPED_WEIGHTS))
    gate_steps_on_progress: bool = False
    failure_cap: float | None = None
    # train/sync_grpo.py's --partial-credit: a failed submission's grade is scaled,
    # leaving the environment's own grade untouched.
    failure_scale: float = 1.0
    # anchored family
    pass_floor: float = 0.60
    fail_cap: float = 0.30
    progress_mode: str = "mass"
    # How hard padding a solve with duplicate or junk routes is punished: the solve
    # reward is scaled by (precision_floor + (1 - precision_floor) * valid / submitted).
    precision_floor: float = 0.0
    # Credit a failed submission still earns for being well-formed and compliant.
    structure_share: float = 0.25
    # Credit for closing in on the hardest unsolved molecule; keeps linear routes from
    # being flat-zero until their last step. Must stay small enough not to pay on its own.
    simplification_share: float = 0.0
    frontier_mode: str = "mass"

    def renamed(self, suffix: str, **changes: Any) -> "RewardConfig":
        return replace(self, name=f"{self.name}{suffix}", **changes)


@dataclass
class RouteAnalysis:
    root_ok: bool
    duplicate: bool
    depth: int
    structure: float
    supported_fraction: float
    leaf_count: int
    leaves_in_stock: int
    leaves_claimed_correctly: int
    false_claims: int
    violations: int
    progress_leaf: float
    progress_mass: float
    progress_worst: float
    simplification: float
    proximity: float
    signature: frozenset
    first_cut: tuple[str, ...]
    errors: int
    template_only: bool

    @property
    def valid(self) -> bool:
        return (
            self.root_ok
            and not self.duplicate
            and not self.errors
            and self.supported_fraction == 1.0
            and self.leaf_count > 0
            and self.leaves_in_stock == self.leaf_count
            and not self.false_claims
            and not self.violations
        )

    @property
    def stock_score(self) -> float:
        if not self.leaf_count:
            return 0.0
        return (self.leaves_in_stock + self.leaves_claimed_correctly) / (2 * self.leaf_count)

    def progress(self, mode: str) -> float:
        return {"leaf": self.progress_leaf, "mass": self.progress_mass, "worst": self.progress_worst}[mode]


@dataclass
class Analysis:
    routes: list[RouteAnalysis]
    solved: bool
    distinct_cuts: int
    similarity: float
    exact: bool
    min_known_depth: int
    min_routes: int
    max_routes: int
    parse_ok: bool
    any_false_claim: bool
    any_root_ok: bool
    over_route_budget: bool


def analyse(evidence: SubmissionEvidence, config: RewardConfig) -> Analysis:
    references = [frozenset((product, tuple(reactants)) for product, reactants in sig) for sig in evidence.reference_signatures]
    routes: list[RouteAnalysis] = []
    seen: set[frozenset] = set()
    for route in evidence.routes:
        analysis = _route(route, evidence, config)
        if analysis.signature and analysis.signature in seen:
            analysis.duplicate = True
        seen.add(analysis.signature)
        routes.append(analysis)
    valid = [route for route in routes if route.valid]
    cuts = {route.first_cut for route in valid}
    similarity = max(
        (_jaccard(route.signature, reference) for route in valid for reference in references), default=0.0
    )
    return Analysis(
        routes=routes,
        solved=len(cuts) >= evidence.min_routes and len(routes) <= evidence.max_routes,
        distinct_cuts=len(cuts),
        similarity=similarity,
        exact=any(route.signature == reference for route in valid for reference in references),
        min_known_depth=evidence.min_known_depth,
        min_routes=evidence.min_routes,
        max_routes=evidence.max_routes,
        parse_ok=evidence.parse_ok,
        any_false_claim=any(route.false_claims for route in routes),
        any_root_ok=any(route.root_ok for route in routes),
        over_route_budget=len(routes) > evidence.max_routes,
    )


def _route(route: RouteEvidence, evidence: SubmissionEvidence, config: RewardConfig) -> RouteAnalysis:
    supported = [step.supported(config.min_template_count, config.support_rule) for step in route.steps]
    forbidden = set(evidence.forbidden_classes)
    violations = sum(1 for name in route.classes if name in forbidden)
    violations += len(route.excluded_used)
    violations += route.depth > evidence.max_depth
    stock = {leaf["smiles"]: leaf["in_stock"] for leaf in route.leaves}
    heavy = {leaf["smiles"]: leaf["heavy_atoms"] for leaf in route.leaves}
    heavy.update({step.product: step.heavy_atoms for step in route.steps})
    by_product = {
        step.product: (step.reactants, ok) for step, ok in zip(route.steps, supported) if step.product
    }
    structure_bits = [
        route.root_ok,
        route.errors == 0,
        bool(route.steps),
        route.molecule_validity == 1.0,
        route.warnings == 0,
    ]
    return RouteAnalysis(
        root_ok=route.root_ok,
        duplicate=False,
        depth=route.depth,
        structure=sum(structure_bits) / len(structure_bits),
        supported_fraction=sum(supported) / len(supported) if supported else 0.0,
        leaf_count=len(route.leaves),
        leaves_in_stock=sum(leaf["in_stock"] for leaf in route.leaves),
        leaves_claimed_correctly=sum(leaf["in_stock"] and leaf["claimed"] for leaf in route.leaves),
        false_claims=sum(leaf["claimed"] and not leaf["in_stock"] for leaf in route.leaves),
        violations=violations,
        progress_leaf=_progress(route.root, by_product, stock, heavy, "leaf"),
        progress_mass=_progress(route.root, by_product, stock, heavy, "mass"),
        progress_worst=_progress(route.root, by_product, stock, heavy, "worst"),
        simplification=_frontier(route.root, by_product, stock, heavy, route.nearest_stock, "mass"),
        proximity=_frontier(route.root, by_product, stock, heavy, route.nearest_stock, "proximity"),
        signature=frozenset((step.product, step.reactants) for step in route.steps if step.product),
        first_cut=tuple(route.first_cut),
        errors=route.errors,
        template_only=bool(route.steps) and any(not step.in_corpus for step in route.steps),
    )


def _progress(
    root: str,
    by_product: dict[str, tuple[tuple[str, ...], bool]],
    stock: dict[str, bool],
    heavy: dict[str, int],
    mode: str,
) -> float:
    """How much of the tree below `root` is grounded in purchasable material.

    A leaf scores 1 when it is really in stock. An expanded molecule scores 0
    unless its step is supported, and otherwise combines its precursors: the
    mean, the heavy-atom weighted mean, or the worst branch. An unreachable
    frontier therefore earns nothing, which is what stops "one accepted cut,
    then stop" from paying.
    """

    def value(smiles: str, ancestry: frozenset[str]) -> float:
        entry = by_product.get(smiles)
        if entry is None or smiles in ancestry:
            return float(stock.get(smiles, False))
        reactants, supported = entry
        if not supported or not reactants:
            return 0.0
        inner = ancestry | {smiles}
        values = [value(reactant, inner) for reactant in reactants]
        if mode == "worst":
            return min(values)
        if mode == "mass":
            weights = [max(heavy.get(reactant, 1), 1) for reactant in reactants]
            return sum(w * v for w, v in zip(weights, values)) / sum(weights)
        return sum(values) / len(values)

    return value(root, frozenset()) if root else 0.0


def _frontier(
    root: str,
    by_product: dict[str, tuple[tuple[str, ...], bool]],
    stock: dict[str, bool],
    heavy: dict[str, int],
    nearest: dict[str, float],
    mode: str,
) -> float:
    """How much work is left on the hardest molecule the route has not resolved.

    A purely linear route reaches no stock until its last step, so progress
    alone gives it nothing to climb.     `mass` scores the frontier by how far it
    has shrunk below the target, which says nothing about a redox chain whose
    molecules keep their size; `proximity` scores it by Tanimoto to the closest
    buyable molecule, which moves whenever a step gets genuinely closer to the
    stock. Only molecules reached through supported steps count, so an
    unsupported cut cannot claim to have made progress.

    Both are measured as a gain over the target itself, so a submission that has
    resolved nothing scores zero however buyable-looking the target happens to be.
    """
    frontier: list[str] = []

    def walk(smiles: str, ancestry: frozenset[str]) -> None:
        entry = by_product.get(smiles)
        if entry is None or smiles in ancestry:
            if not stock.get(smiles, False):
                frontier.append(smiles)
            return
        reactants, supported = entry
        if not supported or not reactants:
            frontier.append(smiles)
            return
        for reactant in reactants:
            walk(reactant, ancestry | {smiles})

    if not root:
        return 0.0
    walk(root, frozenset())
    if not frontier:
        return 1.0
    if mode == "proximity":
        start = nearest.get(root, 0.0)
        worst = min(nearest.get(smiles, 0.0) for smiles in frontier)
        return max(0.0, min(1.0, (worst - start) / (1.0 - start))) if start < 1.0 else 0.0
    target_size = max(heavy.get(root, 0), 1)
    hardest = max(heavy.get(smiles, target_size) for smiles in frontier)
    return max(0.0, min(1.0, 1.0 - hardest / target_size))


def score(evidence: SubmissionEvidence, config: RewardConfig) -> dict[str, Any]:
    analysis = analyse(evidence, config)
    if not evidence.parse_ok:
        return {"reward": 0.0, "solved": False, "components": {}, "analysis": analysis}
    scorer = _score_anchored if config.family == "anchored" else _score_weighted
    reward, components = scorer(analysis, config)
    if analysis.routes and not analysis.any_root_ok:
        reward = min(reward, OFF_TARGET_CAP)
    if analysis.any_false_claim:
        reward = min(reward, FALSE_STOCK_CAP)
    return {
        "reward": round(min(1.0, max(0.0, reward)), 6),
        "solved": analysis.solved,
        "components": components,
        "analysis": analysis,
    }


def _score_weighted(analysis: Analysis, config: RewardConfig) -> tuple[float, dict[str, float]]:
    slots = max(len(analysis.routes), analysis.min_routes)
    totals = dict.fromkeys(("structure", "steps", "stock", "constraints", "efficiency"), 0.0)
    for route in analysis.routes:
        if route.duplicate:
            continue
        totals["structure"] += route.structure
        if not route.root_ok:
            continue
        gate = route.progress(config.progress_mode) if config.gate_steps_on_progress else 1.0
        totals["steps"] += route.supported_fraction * gate
        totals["stock"] += route.stock_score
        totals["constraints"] += float(not route.violations)
        if route.valid:
            totals["efficiency"] += min(1.0, analysis.min_known_depth / max(route.depth, 1))
    components = {
        "parse": 1.0 if analysis.routes else 0.0,
        **{name: value / slots for name, value in totals.items()},
        "diversity": min(1.0, analysis.distinct_cuts / analysis.min_routes),
        "reference_similarity": analysis.similarity,
        "reference_match": float(analysis.exact),
    }
    total_weight = sum(config.weights.values()) or 1.0
    reward = sum(config.weights.get(name, 0.0) * value for name, value in components.items()) / total_weight
    if not analysis.solved:
        reward *= config.failure_scale
        if config.failure_cap is not None:
            reward = min(reward, config.failure_cap)
    return reward, components


def _score_anchored(analysis: Analysis, config: RewardConfig) -> tuple[float, dict[str, float]]:
    """Solve first, then route quality; a failure can never reach a solve."""
    if analysis.solved:
        valid = [route for route in analysis.routes if route.valid]
        efficiency = sum(min(1.0, analysis.min_known_depth / max(route.depth, 1)) for route in valid) / len(valid)
        diversity = min(1.0, analysis.distinct_cuts / analysis.min_routes)
        bonus = (
            0.45 * efficiency
            + 0.20 * diversity
            + 0.20 * analysis.similarity
            + 0.15 * float(analysis.exact)
        )
        # Duplicates and junk alongside a solve dilute it, so padding the set cannot pay.
        precision = len(valid) / max(len(analysis.routes), 1)
        scale = config.precision_floor + (1.0 - config.precision_floor) * precision
        components = {
            "solved": 1.0,
            "efficiency": efficiency,
            "diversity": diversity,
            "precision": precision,
            "reference_similarity": analysis.similarity,
            "reference_match": float(analysis.exact),
        }
        return (config.pass_floor + (1.0 - config.pass_floor) * bonus) * scale, components

    routes = [route for route in analysis.routes if not route.duplicate]
    slots = max(len(routes), analysis.min_routes) or 1
    progress = sum(route.progress(config.progress_mode) for route in routes) / slots
    frontier_values = [route.proximity if config.frontier_mode == "proximity" else route.simplification for route in routes]
    simplification = sum(frontier_values) / slots
    structure = sum(route.structure for route in routes) / slots
    compliance = sum(float(not route.violations) for route in routes) / slots
    # Progress dominates: well-formed JSON with an accepted cut but no stock is near zero.
    structure_share = config.structure_share
    simplification_share = config.simplification_share
    quality = (
        (1.0 - structure_share - simplification_share) * progress
        + simplification_share * simplification
        + structure_share * structure * compliance
    )
    components = {
        "solved": 0.0,
        "progress": progress,
        "simplification": simplification,
        "structure": structure,
        "constraints": compliance,
    }
    return config.fail_cap * quality, components


def _jaccard(left: frozenset, right: frozenset) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


ANCHORED = RewardConfig("anchored", family="anchored", progress_mode="mass", precision_floor=0.5, structure_share=0.25)


def catalogue() -> list[RewardConfig]:
    """One knob at a time off the shipped reward, then the anchored form and its sweeps."""
    free = {"parse", "structure", "constraints"}
    no_free = {name: (0.0 if name in free else weight) for name, weight in SHIPPED_WEIGHTS.items()}
    configs = [
        RewardConfig("v0-shipped"),
        RewardConfig("v0b-partial-credit-0.5", failure_scale=0.5),  # what sync_grpo.py trains on today
        RewardConfig("v1-no-free-components", weights=no_free),
        RewardConfig("v2-progress-gated-steps", gate_steps_on_progress=True),
        RewardConfig("v3-failure-cap-0.30", failure_cap=0.30),
        RewardConfig("v4-gated+cap", gate_steps_on_progress=True, failure_cap=0.30),
        ANCHORED.renamed("-a0-base"),
    ]
    configs += [ANCHORED.renamed(f"-a1-progress-{mode}", progress_mode=mode) for mode in ("leaf", "worst")]
    configs += [ANCHORED.renamed(f"-a2-precision-floor-{f}", precision_floor=f) for f in (0.0, 0.25, 1.0)]
    configs += [ANCHORED.renamed(f"-a3-structure-share-{s}", structure_share=s) for s in (0.0, 0.5)]
    configs += [ANCHORED.renamed(f"-a4-bands-{p}/{c}", pass_floor=p, fail_cap=c) for p, c in ((0.5, 0.4), (0.7, 0.2))]
    configs += [
        ANCHORED.renamed(f"-a5-support-{rule}@{count}", support_rule=rule, min_template_count=count)
        for rule, count in (("any", 25), ("major", 5), ("sole", 10), ("sole", 25))
    ]
    configs += [
        ANCHORED.renamed(f"-a6-frontier-{mode}-{s}", simplification_share=s, structure_share=0.15, frontier_mode=mode)
        for mode in ("mass", "proximity")
        for s in (0.10, 0.20, 0.35)
    ]
    final = replace(
        ANCHORED,
        name="v8-FINAL",
        support_rule="sole",
        pass_floor=0.70,
        fail_cap=0.20,
        structure_share=0.15,
        simplification_share=0.20,
        frontier_mode="proximity",
    )
    return configs + [final]
