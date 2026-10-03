"""Conservative step and route verification with explicit evidence tiers."""

from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache
from typing import Any, Iterable

from .chemistry import (
    ChemistryError,
    canonical_step_key,
    canonicalize_components,
    canonicalize_smiles,
    formula_counter,
    template_produces,
)
from .graph import parse_submission
from .models import ReactionStep, RetroTask, ScoreResult, StepValidation


WEIGHTS = {
    "structural_validity": 0.30,
    "route_integrity": 0.25,
    "building_block_completion": 0.20,
    "reference_similarity": 0.15,
    "efficiency": 0.10,
}

# Unlike the legacy all-or-nothing route score, the training-facing graph
# reward stays differentiated for malformed, partial, and nearly-correct
# submissions. This is important for GRPO groups from weak base models.
GRAPH_WEIGHTS = {
    "parse_validity": 0.05,
    "molecule_validity": 0.10,
    "graph_validity": 0.10,
    "step_correctness": 0.20,
    "stock_correctness": 0.10,
    "reference_similarity": 0.10,
    "exact_route_match": 0.10,
    "verified_route_diversity": 0.10,
    # Meeting the requested route-set cardinality is a discrete contract. It
    # receives enough weight that emitting one perfect route cannot nearly tie
    # a compliant set, while all other fields remain densely shaped.
    "route_set_compliance": 0.15,
}


class RouteVerifier:
    """Verify only what RDKit and the task's hidden evidence can establish."""

    def _canonical_stock(self, stock: Iterable[str]) -> frozenset[str]:
        if isinstance(stock, frozenset):
            return _canonicalize_frozen_stock(stock)
        return frozenset(canonicalize_smiles(item) for item in stock)

    def validate_step(
        self,
        task: RetroTask,
        product: str,
        reactants: Iterable[str],
        reaction_class: str | None = None,
    ) -> StepValidation:
        errors: list[str] = []
        checks = {
            "valid_structures": False,
            "no_product_leakage": False,
            "atom_inventory_conserved": False,
            "reaction_class_compatible": False,
            "dataset_or_template_support": False,
        }
        try:
            canonical_product = canonicalize_smiles(product)
            canonical_reactants = canonicalize_components(reactants)
            if not canonical_reactants:
                raise ChemistryError("a step needs at least one reactant")
            checks["valid_structures"] = True
        except Exception as exc:
            return StepValidation(
                valid=False,
                product=None,
                reactants=(),
                support="none",
                matched_reaction_id=None,
                atom_conservation="not_checked",
                checks=checks,
                errors=(str(exc),),
            )

        checks["no_product_leakage"] = canonical_product not in canonical_reactants
        if not checks["no_product_leakage"]:
            errors.append("step product is present unchanged among its reactants")

        product_inventory = formula_counter((canonical_product,))
        reactant_inventory = formula_counter(canonical_reactants)
        missing_atoms = product_inventory - reactant_inventory
        checks["atom_inventory_conserved"] = not missing_atoms
        if missing_atoms:
            rendered = ", ".join(
                f"{element}:{count}" for element, count in sorted(missing_atoms.items())
            )
            errors.append(
                "product atom inventory is not contained in the proposed reactants: "
                + rendered
            )

        key = canonical_step_key(canonical_product, canonical_reactants)
        references = [
            step
            for route in task.reference_routes
            for step in route.steps
        ]
        exact = None
        for reference in references:
            if canonical_step_key(reference.product, reference.reactants) == key:
                exact = reference
                break

        support = "none"
        matched_id = None
        atom_conservation = (
            "element_inventory_passed" if not missing_atoms else "element_inventory_failed"
        )
        class_compatible = True
        if exact is not None:
            support = "dataset_exact"
            matched_id = exact.reaction_id
            class_compatible = _class_compatible(reaction_class, exact.reaction_class)
            if exact.mapping_status in {"complete", "partial"}:
                atom_conservation = f"reference_mapping_{exact.mapping_status}"
            else:
                atom_conservation = "dataset_record_unmapped"
        else:
            for reference in references:
                if not reference.reaction_smarts:
                    continue
                if not _class_compatible(reaction_class, reference.reaction_class):
                    continue
                if template_produces(
                    reference.reaction_smarts,
                    canonical_reactants,
                    canonical_product,
                ):
                    support = "trusted_template"
                    matched_id = reference.reaction_id
                    atom_conservation = "template_execution"
                    class_compatible = True
                    break

        checks["reaction_class_compatible"] = class_compatible
        checks["dataset_or_template_support"] = support != "none"
        if not class_compatible:
            errors.append("provided reaction_class conflicts with hidden step evidence")
        if support == "none":
            errors.append("no exact dataset reaction or trusted template supports this step")

        valid = all(checks.values())
        return StepValidation(
            valid=valid,
            product=canonical_product,
            reactants=canonical_reactants,
            support=support,
            matched_reaction_id=matched_id,
            atom_conservation=atom_conservation,
            checks=checks,
            errors=tuple(errors),
        )

    def score_submission(
        self,
        task: RetroTask,
        submission: Any,
        stock: Iterable[str],
    ) -> ScoreResult:
        """Score one renderable route-set submission with dense components.

        ``valid`` still means that the route-count contract was met and at
        least one whole route passed the conservative chemistry verifier. The
        scalar reward is intentionally denser: valid JSON with valid molecules
        scores above a non-submission, while unsupported chemistry cannot earn
        the high-value correctness components.
        """

        parsed = parse_submission(submission)
        stock_set = self._canonical_stock(stock)
        target = canonicalize_smiles(task.target_smiles)
        route_count = len(parsed.trees)
        count_in_bounds = task.min_routes <= route_count <= task.max_routes
        route_results: list[ScoreResult] = []
        graph_scores: list[float] = []
        stock_claim_scores: list[float] = []
        route_details: list[dict[str, Any]] = []

        for index, tree in enumerate(parsed.trees):
            has_root = tree.root_smiles is not None
            has_reaction = bool(tree.steps)
            checks = {
                "root_matches_target": has_root and tree.root_smiles == target,
                "has_reaction": has_reaction,
                "alternating_connected_tree": has_root and has_reaction and not tree.errors,
                "within_step_budget": has_reaction and len(tree.steps) <= task.max_steps,
            }
            graph_score = sum(checks.values()) / len(checks)
            graph_scores.append(graph_score)
            claim_accuracy = (
                sum((smiles in stock_set) == claim for smiles, claim in tree.terminal_claims)
                / len(tree.terminal_claims)
                if tree.terminal_claims
                else 0.0
            )
            stock_claim_scores.append(claim_accuracy)
            legacy = {
                "route": [step.to_dict(include_evidence=False) for step in tree.steps]
            }
            result = self.score_route(task, legacy, stock_set)
            route_results.append(result)
            step_confidences = [
                1.0
                if step.valid
                else sum(step.checks.values()) / max(1, len(step.checks))
                for step in result.step_results
            ]
            route_score = min(step_confidences) if step_confidences else 0.0
            route_details.append(
                {
                    "index": index,
                    "valid": result.valid,
                    "verification_tier": result.verification_tier,
                    "route_score": round(route_score, 6),
                    "graph_score": round(graph_score, 6),
                    "molecule_validity": round(tree.molecule_validity, 6),
                    "stock_claim_accuracy": round(claim_accuracy, 6),
                    "step_validity": result.metrics.get("step_validity", 0.0),
                    "reference_similarity": result.metrics.get("reference_similarity", 0.0),
                    "exact_reference_match": result.metrics.get("exact_reference_match", False),
                    "first_cut": list(tree.first_cut),
                    "errors": tree.errors,
                    "hard_failures": list(result.hard_failures),
                }
            )

        best_step = max(
            (result.metrics.get("step_validity", 0.0) for result in route_results),
            default=0.0,
        )
        best_stock = max(
            (
                0.5 * result.metrics.get("building_block_completion", 0.0)
                + 0.5 * stock_claim_scores[index]
                for index, result in enumerate(route_results)
            ),
            default=0.0,
        )
        best_similarity = max(
            (result.metrics.get("reference_similarity", 0.0) for result in route_results),
            default=0.0,
        )
        exact_match = any(
            result.metrics.get("exact_reference_match", False) for result in route_results
        )
        valid_first_cuts = {
            parsed.trees[index].first_cut
            for index, result in enumerate(route_results)
            if result.valid and parsed.trees[index].first_cut
        }
        diversity_denominator = max(1, min(task.min_routes, len(task.reference_routes)))
        diversity = min(1.0, len(valid_first_cuts) / diversity_denominator)
        structurally_countable = all(
            tree.root_smiles is not None and tree.molecule_validity > 0
            for tree in parsed.trees
        ) if parsed.trees else False
        count_score = (
            1.0
            if count_in_bounds and structurally_countable
            else (
                route_count / task.min_routes
                if structurally_countable and route_count < task.min_routes
                else (
                    task.max_routes / max(route_count, 1)
                    if structurally_countable
                    else 0.0
                )
            )
        )
        graph_component = (
            0.75 * (sum(graph_scores) / len(graph_scores)) + 0.25 * count_score
            if graph_scores
            else 0.0
        )
        molecule_component = (
            sum(tree.molecule_validity for tree in parsed.trees) / len(parsed.trees)
            if parsed.trees
            else 0.0
        )
        components = {
            "parse_validity": float(parsed.parse_valid),
            "molecule_validity": molecule_component,
            "graph_validity": graph_component,
            "step_correctness": best_step,
            "stock_correctness": best_stock,
            "reference_similarity": best_similarity,
            "exact_route_match": float(exact_match),
            "verified_route_diversity": diversity,
            "route_set_compliance": float(count_in_bounds and structurally_countable),
        }
        reward = sum(GRAPH_WEIGHTS[name] * value for name, value in components.items())
        valid_route_count = sum(result.valid for result in route_results)
        enough_valid_routes = valid_route_count >= task.min_routes
        enough_distinct_routes = len(valid_first_cuts) >= task.min_routes
        valid = (
            parsed.parse_valid
            and count_in_bounds
            and enough_valid_routes
            and enough_distinct_routes
        )
        tiers = [result.verification_tier for result in route_results if result.valid]
        tier = (
            "dataset_supported"
            if "dataset_supported" in tiers
            else ("template_supported" if tiers else "rejected")
        )
        failures = list(parsed.errors)
        for index, tree in enumerate(parsed.trees):
            failures.extend(f"route {index + 1}: {error}" for error in tree.errors)
        if not count_in_bounds:
            failures.append(
                f"submission has {route_count} routes; expected "
                f"{task.min_routes}..{task.max_routes}"
            )
        if not enough_valid_routes:
            failures.append(
                f"only {valid_route_count} submitted routes passed the chemistry verifier; "
                f"need at least {task.min_routes}"
            )
        if not enough_distinct_routes:
            failures.append(
                f"only {len(valid_first_cuts)} distinct verified first cuts; "
                f"need at least {task.min_routes}"
            )
        metrics = {
            "route_count": route_count,
            "route_count_in_bounds": count_in_bounds,
            "valid_routes": valid_route_count,
            "parse_valid": parsed.parse_valid,
            "molecule_validity": round(molecule_component, 6),
            "graph_validity": round(graph_component, 6),
            "step_validity": round(best_step, 6),
            "building_block_completion": round(best_stock, 6),
            "reference_similarity": round(best_similarity, 6),
            "exact_reference_match": exact_match,
            "verified_route_diversity": round(diversity, 6),
            "route_results": route_details,
        }
        return ScoreResult(
            reward=round(reward, 6),
            valid=valid,
            verification_tier=tier,
            hard_failures=tuple(_unique(failures)),
            components={name: round(value, 6) for name, value in components.items()},
            metrics=metrics,
            step_results=tuple(
                step for result in route_results for step in result.step_results
            ),
        )

    def score_route(
        self,
        task: RetroTask,
        route: dict[str, Any] | list[dict[str, Any]],
        stock: Iterable[str],
    ) -> ScoreResult:
        hard_failures: list[str] = []
        raw_steps = route.get("route", route.get("steps", [])) if isinstance(route, dict) else route
        if not isinstance(raw_steps, list):
            return self._failed("route must be a list of step objects")
        if not raw_steps:
            return self._failed("route is empty")
        if len(raw_steps) > task.max_steps:
            hard_failures.append(
                f"route has {len(raw_steps)} steps but max_steps is {task.max_steps}"
            )

        steps: list[ReactionStep] = []
        for index, raw in enumerate(raw_steps):
            try:
                parsed = ReactionStep.from_dict(raw)
                steps.append(
                    ReactionStep(
                        product=canonicalize_smiles(parsed.product),
                        reactants=canonicalize_components(parsed.reactants),
                        reaction_class=parsed.reaction_class,
                    )
                )
            except Exception as exc:
                hard_failures.append(f"step {index + 1}: invalid structure: {exc}")
        if len(steps) != len(raw_steps):
            return self._failed(*hard_failures, step_count=len(raw_steps))

        target = canonicalize_smiles(task.target_smiles)
        stock_set = self._canonical_stock(stock)
        products = [step.product for step in steps]
        product_counts = Counter(products)
        duplicates = sorted(product for product, count in product_counts.items() if count > 1)
        if duplicates:
            hard_failures.append(f"multiple steps produce the same intermediate: {duplicates}")
        by_product = {step.product: step for step in steps}
        if target not in by_product:
            hard_failures.append("route does not expand the task target")
        if any(step.product in step.reactants for step in steps):
            hard_failures.append("a step leaks its unchanged product into its reactants")
        if any(target in step.reactants for step in steps):
            hard_failures.append("target product appears among route reactants")

        reachable: set[str] = set()
        terminals: set[str] = set()
        visiting: set[str] = set()
        cycle = False

        def walk(product: str) -> None:
            nonlocal cycle
            if product in visiting:
                cycle = True
                return
            if product in reachable:
                return
            step = by_product.get(product)
            if step is None:
                terminals.add(product)
                return
            visiting.add(product)
            reachable.add(product)
            for reactant in step.reactants:
                if reactant in by_product:
                    walk(reactant)
                else:
                    terminals.add(reactant)
            visiting.remove(product)

        if target in by_product:
            walk(target)
        if cycle:
            hard_failures.append("route graph contains a cycle")
        unreachable = sorted(set(by_product) - reachable)
        if unreachable:
            hard_failures.append(f"route contains unreachable steps: {unreachable}")

        missing_stock = sorted(terminals - stock_set)
        if missing_stock:
            hard_failures.append(f"terminal molecules are unavailable: {missing_stock}")

        step_results = tuple(
            self.validate_step(
                task, step.product, step.reactants, step.reaction_class
            )
            for step in steps
        )
        for index, result in enumerate(step_results, 1):
            if not result.valid:
                hard_failures.append(
                    f"step {index} unsupported: {'; '.join(result.errors)}"
                )

        hard_failures = _unique(hard_failures)
        reference_similarity, exact_match, best_reference_steps = self._reference_match(
            task, steps
        )
        step_validity = (
            sum(result.valid for result in step_results) / len(step_results)
            if step_results
            else 0.0
        )
        stock_completion = (
            (len(terminals) - len(missing_stock)) / len(terminals)
            if terminals
            else 0.0
        )
        efficiency = min(1.0, best_reference_steps / len(steps)) if steps else 0.0
        metrics = {
            "step_count": len(steps),
            "step_validity": round(step_validity, 6),
            "route_continuity": not cycle and not unreachable and target in by_product,
            "terminal_count": len(terminals),
            "building_block_completion": round(stock_completion, 6),
            "missing_building_blocks": missing_stock,
            "reference_similarity": round(reference_similarity, 6),
            "exact_reference_match": exact_match,
            "best_reference_steps": best_reference_steps,
            "mapping_backed_steps": sum(
                result.atom_conservation.startswith("reference_mapping")
                for result in step_results
            ),
            "template_backed_steps": sum(
                result.support == "trusted_template" for result in step_results
            ),
        }
        if hard_failures:
            return ScoreResult(
                reward=0.0,
                valid=False,
                verification_tier="rejected",
                hard_failures=tuple(hard_failures),
                components={key: 0.0 for key in WEIGHTS},
                metrics=metrics,
                step_results=step_results,
            )

        components = {
            "structural_validity": 1.0,
            "route_integrity": 1.0,
            "building_block_completion": 1.0,
            "reference_similarity": reference_similarity,
            "efficiency": efficiency,
        }
        reward = sum(WEIGHTS[key] * value for key, value in components.items())
        tier = (
            "template_supported"
            if any(result.support == "trusted_template" for result in step_results)
            else "dataset_supported"
        )
        return ScoreResult(
            reward=round(reward, 6),
            valid=True,
            verification_tier=tier,
            hard_failures=(),
            components={key: round(value, 6) for key, value in components.items()},
            metrics=metrics,
            step_results=step_results,
        )

    def _reference_match(
        self, task: RetroTask, steps: list[ReactionStep]
    ) -> tuple[float, bool, int]:
        proposed = {
            canonical_step_key(step.product, step.reactants) for step in steps
        }
        best = 0.0
        exact = False
        best_steps = min(len(route.steps) for route in task.reference_routes)
        for route in task.reference_routes:
            reference = {
                canonical_step_key(step.product, step.reactants) for step in route.steps
            }
            union = proposed | reference
            similarity = len(proposed & reference) / len(union) if union else 0.0
            if similarity > best:
                best = similarity
                best_steps = len(route.steps)
            exact = exact or proposed == reference
        return best, exact, best_steps

    @staticmethod
    def _failed(*reasons: str, step_count: int = 0) -> ScoreResult:
        return ScoreResult(
            reward=0.0,
            valid=False,
            verification_tier="rejected",
            hard_failures=tuple(_unique(list(reasons))),
            components={key: 0.0 for key in WEIGHTS},
            metrics={"step_count": step_count, "step_validity": 0.0},
            step_results=(),
        )


def _class_compatible(proposed: str | None, reference: str | None) -> bool:
    if not proposed or not reference:
        return True
    normalize = lambda value: re.sub(r"[^a-z0-9]+", "", value.lower())
    return normalize(proposed) == normalize(reference)


@lru_cache(maxsize=8)
def _canonicalize_frozen_stock(stock: frozenset[str]) -> frozenset[str]:
    """Canonicalize immutable stock once across verifiers and submissions."""
    return frozenset(canonicalize_smiles(item) for item in stock)


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))
