"""Chemistry-first scoring of a submitted route set.

A route is valid when it starts at the target, is a well-formed tree within the
depth budget, every step is supported by the frozen reaction library (corpus
reaction or frequent retro-template; never the task's hidden route), every
leaf is in the task's effective stock, no leaf falsely claims stock, and the
task constraints hold. Known routes are evidence for a bonus, not the
definition of correctness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Iterable

from .chemistry import ChemistryError, canonicalize_smiles
from .classes import classify_step
from .graph import ParsedTree, parse_submission, routes_to_submission
from .models import ReferenceRoute, RetroTask, ScoreResult
from .reactions import ReactionLibrary, StepKey, free_key

WEIGHTS = {
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
# A false in-stock claim or a submission that never addresses the target cannot score highly.
FALSE_STOCK_CAP = 0.40
OFF_TARGET_CAP = 0.10


@dataclass
class RouteEvaluation:
    index: int
    root_ok: bool
    signature: frozenset[StepKey] = frozenset()
    first_cut: tuple[str, ...] = ()
    depth: int = 0
    duplicate: bool = False
    step_results: list[dict[str, Any]] = field(default_factory=list)
    leaves_in_stock: int = 0
    leaves_claimed_correctly: int = 0
    leaf_count: int = 0
    false_claims: list[str] = field(default_factory=list)
    missing_stock: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    molecule_validity: float = 0.0

    @property
    def supported_fraction(self) -> float:
        if not self.step_results:
            return 0.0
        return sum(result["supported"] for result in self.step_results) / len(self.step_results)

    @property
    def valid(self) -> bool:
        return (
            self.root_ok
            and not self.duplicate
            and not self.errors
            and bool(self.step_results)
            and self.supported_fraction == 1.0
            and self.leaf_count > 0
            and self.leaves_in_stock == self.leaf_count
            and not self.false_claims
            and not self.violations
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "valid": self.valid,
            "root_matches_target": self.root_ok,
            "duplicate_of_earlier_route": self.duplicate,
            "depth": self.depth,
            "steps": len(self.step_results),
            "supported_steps": sum(result["supported"] for result in self.step_results),
            "leaves": self.leaf_count,
            "leaves_in_stock": self.leaves_in_stock,
            "false_stock_claims": self.false_claims,
            "missing_stock": self.missing_stock,
            "constraint_violations": self.violations,
            "first_cut": list(self.first_cut),
            "errors": self.errors,
            "warnings": self.warnings[:10],
        }


class RouteVerifier:
    """Score submissions with a reaction library; hidden routes only inform the bonus."""

    def __init__(self, library: ReactionLibrary):
        self.library = library

    def score_submission(self, task: RetroTask, submission: Any, stock: Iterable[str]) -> ScoreResult:
        parsed = parse_submission(submission)
        if not parsed.parse_valid:
            return _rejected(parsed.errors)
        stock_set = _canonical_stock(stock)
        excluded = _canonical_stock(task.constraints.excluded_stock)
        effective_stock = stock_set - excluded
        target = canonicalize_smiles(task.target_smiles)
        forbidden = frozenset(task.constraints.forbidden_classes)

        routes: list[RouteEvaluation] = []
        seen: set[frozenset[StepKey]] = set()
        for index, tree in enumerate(parsed.trees):
            route = self._evaluate_tree(index, tree, target, task, effective_stock, excluded, forbidden)
            if route.signature and route.signature in seen:
                route.duplicate = True
            seen.add(route.signature)
            routes.append(route)

        references = compliant_references(task, effective_stock, forbidden)
        min_depth = min((route_depth(task.target_smiles, ref.steps) for ref in references), default=None)
        if min_depth is None:
            min_depth = int(task.difficulty.get("min_depth") or task.max_depth)
        reference_signatures = [(ref, signature(ref.steps)) for ref in references]

        slots = max(len(routes), task.min_routes)
        structure = steps = stock_score = constraints = efficiency = 0.0
        for route in routes:
            if route.duplicate:
                continue
            route_structure = [route.root_ok, not route.errors, bool(route.step_results), route.molecule_validity == 1.0, not route.warnings]
            structure += sum(route_structure) / len(route_structure)
            if not route.root_ok:
                continue
            steps += route.supported_fraction
            if route.leaf_count:
                stock_score += (route.leaves_in_stock + route.leaves_claimed_correctly) / (2 * route.leaf_count)
            constraints += float(not route.violations)
            if route.valid:
                efficiency += min(1.0, min_depth / max(route.depth, 1))

        valid_routes = [route for route in routes if route.valid]
        distinct_cuts = {route.first_cut for route in valid_routes}
        similarity = max(
            (
                _jaccard(route.signature, ref_signature)
                for route in valid_routes
                for _, ref_signature in reference_signatures
            ),
            default=0.0,
        )
        exact = any(route.signature == ref_signature for route in valid_routes for _, ref_signature in reference_signatures)
        components = {
            "parse": 1.0 if routes else 0.0,
            "structure": structure / slots,
            "steps": steps / slots,
            "stock": stock_score / slots,
            "constraints": constraints / slots,
            "efficiency": efficiency / slots,
            "diversity": min(1.0, len(distinct_cuts) / task.min_routes),
            "reference_similarity": similarity,
            "reference_match": float(exact),
        }
        reward = sum(WEIGHTS[name] * value for name, value in components.items())
        failures = list(parsed.errors)
        if routes and not any(route.root_ok for route in routes):
            reward = min(reward, OFF_TARGET_CAP)
            failures.append("no submitted route starts at the target")
        if any(route.false_claims for route in routes):
            reward = min(reward, FALSE_STOCK_CAP)
            failures.append("a leaf claims in_stock=true but is not in the task's stock")
        if len(routes) > task.max_routes:
            failures.append(f"submitted {len(routes)} routes; at most {task.max_routes} allowed")
        if len(distinct_cuts) < task.min_routes:
            failures.append(
                f"{len(distinct_cuts)} valid route(s) with distinct first disconnections; need {task.min_routes}"
            )
        for route in routes:
            failures.extend(f"route {route.index + 1}: {problem}" for problem in _route_problems(route))
        solved = len(distinct_cuts) >= task.min_routes and len(routes) <= task.max_routes
        supports = [result["support"] for route in valid_routes for result in route.step_results]
        tier = "rejected" if not solved else ("template_supported" if "template" in supports else "corpus_supported")
        all_steps = [result for route in routes for result in route.step_results]
        claimed = [route for route in routes if route.root_ok]
        metrics = {
            "route_count": len(routes),
            "valid_routes": len(valid_routes),
            "distinct_valid_first_cuts": len(distinct_cuts),
            "step_validity": round(sum(r["supported"] for r in all_steps) / len(all_steps), 6) if all_steps else 0.0,
            "stock_precision": _stock_precision(claimed),
            "constraint_compliance": round(constraints / slots, 6),
            "min_known_depth": min_depth,
            "best_valid_depth": min((route.depth for route in valid_routes), default=None),
            "reference_similarity": round(similarity, 6),
            "reference_match": exact,
            "routes": [route.to_dict() for route in routes],
        }
        return ScoreResult(
            reward=round(min(1.0, max(0.0, reward)), 6),
            valid=solved,
            verification_tier=tier,
            hard_failures=tuple(dict.fromkeys(failures)),
            components={name: round(value, 6) for name, value in components.items()},
            metrics=metrics,
            step_results=tuple(all_steps),
        )

    def _evaluate_tree(
        self,
        index: int,
        tree: ParsedTree,
        target: str,
        task: RetroTask,
        stock: frozenset[str],
        excluded: frozenset[str],
        forbidden: frozenset[str],
    ) -> RouteEvaluation:
        route = RouteEvaluation(
            index=index,
            root_ok=tree.root_smiles == target and bool(tree.steps),
            depth=tree.depth,
            errors=list(tree.errors),
            warnings=list(tree.warnings),
            molecule_validity=tree.molecule_validity,
        )
        keys: set[StepKey] = set()
        for step in tree.steps:
            support = self.library.support(step.product, step.reactants)
            reaction_class = classify_step(step.product, step.reactants)
            route.step_results.append(
                {
                    "product": step.product,
                    "reactants": list(step.reactants),
                    "supported": support.supported,
                    "support": support.kind,
                    "template_count": support.template_count,
                    "stereo_match": support.stereo_match,
                    "reaction_class": reaction_class.name,
                    **({"error": support.error} if support.error else {}),
                }
            )
            if support.product:
                keys.add((support.product, support.reactants))
            if reaction_class.name in forbidden:
                route.violations.append(f"uses forbidden class {reaction_class.name!r}")
        route.signature = frozenset(keys)
        root_step = next((step for step in tree.steps if step.product == tree.root_smiles), None)
        if root_step is not None:
            try:
                route.first_cut = free_key(root_step.product, root_step.reactants)[1]
            except ChemistryError:
                route.first_cut = ()
        if tree.depth > task.max_depth:
            route.violations.append(f"depth {tree.depth} exceeds max_depth {task.max_depth}")
        for smiles, claim in tree.terminal_claims:
            route.leaf_count += 1
            available = smiles in stock
            route.leaves_in_stock += available
            route.leaves_claimed_correctly += available and claim
            if claim and not available:
                route.false_claims.append(smiles)
            if not available:
                route.missing_stock.append(smiles)
            if smiles in excluded:
                route.violations.append(f"uses excluded building block {smiles}")
        return route


def signature(steps: Iterable[Any]) -> frozenset[StepKey]:
    keys = set()
    for step in steps:
        try:
            keys.add(free_key(step.product, step.reactants))
        except ChemistryError:
            continue
    return frozenset(keys)


def route_depth(target: str, steps: Iterable[Any]) -> int:
    """Longest linear sequence of a flat route rooted at ``target``."""
    by_product = {canonicalize_smiles(step.product): step for step in steps}

    def depth(smiles: str, ancestry: frozenset[str]) -> int:
        step = by_product.get(smiles)
        if step is None or smiles in ancestry:
            return 0
        inner = ancestry | {smiles}
        return 1 + max((depth(canonicalize_smiles(r), inner) for r in step.reactants), default=0)

    return depth(canonicalize_smiles(target), frozenset())


def route_leaves(target: str, steps: Iterable[Any]) -> set[str]:
    by_product = {canonicalize_smiles(step.product): step for step in steps}
    leaves: set[str] = set()

    def walk(smiles: str, ancestry: frozenset[str]) -> None:
        step = by_product.get(smiles)
        if step is None or smiles in ancestry:
            leaves.add(smiles)
            return
        for reactant in step.reactants:
            walk(canonicalize_smiles(reactant), ancestry | {smiles})

    walk(canonicalize_smiles(target), frozenset())
    return leaves


def compliant_references(task: RetroTask, stock: frozenset[str], forbidden: frozenset[str]) -> list[ReferenceRoute]:
    """Known routes that satisfy this task's depth, stock and class constraints."""
    compliant = []
    for route in task.reference_routes:
        if route_depth(task.target_smiles, route.steps) > task.max_depth:
            continue
        if not route_leaves(task.target_smiles, route.steps) <= stock:
            continue
        if forbidden and any(classify_step(step.product, step.reactants).name in forbidden for step in route.steps):
            continue
        compliant.append(route)
    return compliant


def known_routes_submission(task: RetroTask, stock: frozenset[str]) -> dict[str, Any]:
    """The shortest compliant known routes with distinct first cuts, as many as the task needs."""
    available = _canonical_stock(stock) - _canonical_stock(task.constraints.excluded_stock)
    references = sorted(
        compliant_references(task, available, frozenset(task.constraints.forbidden_classes)),
        key=lambda route: route_depth(task.target_smiles, route.steps),
    )
    target = canonicalize_smiles(task.target_smiles)
    chosen, cuts = [], set()
    for route in references:
        root = next((step for step in route.steps if canonicalize_smiles(step.product) == target), None)
        cut = free_key(root.product, root.reactants)[1] if root else ()
        if cut and cut not in cuts:
            cuts.add(cut)
            chosen.append(route)
    return routes_to_submission(task.target_smiles, chosen[: task.min_routes], available)


def _route_problems(route: RouteEvaluation) -> list[str]:
    problems = []
    if route.duplicate:
        problems.append("duplicates an earlier route")
    if not route.root_ok:
        problems.append("does not start with a reaction producing the target")
    problems.extend(route.errors[:5])
    unsupported = [r for r in route.step_results if not r["supported"]]
    if unsupported:
        problems.append(f"{len(unsupported)} step(s) not supported by the reaction library")
    if route.missing_stock:
        problems.append(f"leaves not in stock: {route.missing_stock[:5]}")
    problems.extend(route.violations)
    return problems


def _stock_precision(routes: list[RouteEvaluation]) -> float | None:
    claimed = sum(route.leaves_claimed_correctly + len(route.false_claims) for route in routes)
    if not claimed:
        return None
    return round(sum(route.leaves_claimed_correctly for route in routes) / claimed, 6)


def _jaccard(left: frozenset, right: frozenset) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _rejected(errors: list[str]) -> ScoreResult:
    return ScoreResult(
        reward=0.0,
        valid=False,
        verification_tier="rejected",
        hard_failures=tuple(errors) or ("submission could not be parsed",),
        components={name: 0.0 for name in WEIGHTS},
        metrics={"route_count": 0, "valid_routes": 0},
        step_results=(),
    )


def _canonical_stock(stock: Iterable[str]) -> frozenset[str]:
    if isinstance(stock, frozenset):
        return _canonicalize_frozen(stock)
    return frozenset(canonicalize_smiles(item) for item in stock)


@lru_cache(maxsize=8)
def _canonicalize_frozen(stock: frozenset[str]) -> frozenset[str]:
    return frozenset(canonicalize_smiles(item) for item in stock)
