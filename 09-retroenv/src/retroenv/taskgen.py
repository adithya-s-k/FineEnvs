"""Build leakage-audited single-step and route-planning tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from .chemistry import (
    canonicalize_components,
    canonicalize_smiles,
    inspect_molecule,
    scaffold_smiles,
    stable_hash,
)
from .models import ReactionStep, ReferenceRoute, RetroTask
from .store import load_stock
from .verifier import RouteVerifier


SPLITS = ("train", "dev", "eval", "stress")
DEFAULT_RATIOS = (2 / 3, 1 / 15, 2 / 15, 2 / 15)


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            yield value


def tasks_from_corpus(
    path: Path,
    stock: frozenset[str],
    stock_id: str,
    *,
    include_generated: bool = False,
) -> tuple[list[RetroTask], Counter[str]]:
    references_by_target: dict[str, list[ReferenceRoute]] = defaultdict(list)
    counters: Counter[str] = Counter()
    verifier = RouteVerifier()
    for record in read_jsonl(path):
        counters["corpus_records"] += 1
        evidence = set(record.get("evidence_kinds") or [record.get("evidence_kind")])
        if not include_generated and "observed" not in evidence:
            counters["excluded_non_observed"] += 1
            continue
        product = canonicalize_smiles(record["primary_product"])
        reactants = canonicalize_components(record["reactants"])
        if not reactants or product in reactants:
            counters["excluded_degenerate"] += 1
            continue
        if not set(reactants) <= stock:
            counters["excluded_stock_incomplete"] += 1
            continue
        raw_conditions = record.get("conditions") or ()
        if isinstance(raw_conditions, dict):
            raw_conditions = (raw_conditions,)
        step = ReactionStep(
            product=product,
            reactants=reactants,
            reaction_class=record.get("reaction_class"),
            reaction_id=record["reaction_id"],
            reaction_smarts=record.get("reaction_smarts"),
            mapping_status=(record.get("atom_mapping") or {}).get("status", "unmapped"),
            conditions=tuple(
                item for item in raw_conditions if isinstance(item, dict)
            ),
        )
        route = ReferenceRoute(
            route_id=f"route_{record['reaction_id']}",
            steps=(step,),
            source=tuple(record.get("source") or ()),
        )
        if not _reference_replays(product, route, stock, verifier, mode="single_step"):
            counters["excluded_verifier_invalid"] += 1
            continue
        references_by_target[product].append(route)

    tasks: list[RetroTask] = []
    for target in sorted(references_by_target):
        routes = _dedupe_reference_routes(references_by_target[target])
        details = inspect_molecule(target)
        tasks.append(
            RetroTask(
                task_id=stable_hash(f"single_step:{target}", prefix="retro_", length=20),
                mode="single_step",
                target_smiles=target,
                max_steps=1,
                stock_id=stock_id,
                split="unassigned",
                reference_routes=tuple(routes),
                difficulty={
                    "depth": 1,
                    "heavy_atoms": details["heavy_atoms"],
                    "stereochemistry": details["chiral_centres"] > 0,
                    "scaffold": details["murcko_scaffold"],
                    "reference_alternatives": len(routes),
                },
            )
        )
    counters["tasks_built"] = len(tasks)
    return tasks, counters


def tasks_from_routes(
    path: Path,
    stock: frozenset[str],
    stock_id: str,
    *,
    max_steps: int = 3,
) -> tuple[list[RetroTask], Counter[str]]:
    references_by_target: dict[str, list[ReferenceRoute]] = defaultdict(list)
    counters: Counter[str] = Counter()
    verifier = RouteVerifier()
    for row in read_jsonl(path):
        counters["route_records"] += 1
        raw_steps = row.get("steps") or []
        if not 1 <= len(raw_steps) <= max_steps:
            counters["excluded_depth"] += 1
            continue
        steps: list[ReactionStep] = []
        for raw in raw_steps:
            parsed = ReactionStep.from_dict(raw)
            steps.append(
                ReactionStep(
                    product=canonicalize_smiles(parsed.product),
                    reactants=canonicalize_components(parsed.reactants),
                    reaction_class=parsed.reaction_class,
                    reaction_id=parsed.reaction_id,
                    reaction_smarts=parsed.reaction_smarts,
                    mapping_status=parsed.mapping_status,
                    conditions=parsed.conditions,
                    literature=parsed.literature,
                )
            )
        target = canonicalize_smiles(row.get("target_smiles") or steps[0].product)
        products = {step.product for step in steps}
        terminals = {reactant for step in steps for reactant in step.reactants if reactant not in products}
        if not terminals <= stock:
            counters["excluded_stock_incomplete"] += 1
            continue
        source = tuple(row.get("source") or ())
        if not source or any(not item.get("license") for item in source):
            counters["excluded_missing_provenance"] += 1
            continue
        route_id = str(row.get("route_id") or stable_hash(json.dumps(raw_steps, sort_keys=True), prefix="route_"))
        route = ReferenceRoute(route_id=route_id, steps=tuple(steps), source=source)
        if not _reference_replays(target, route, stock, verifier, mode="route_planning"):
            counters["excluded_verifier_invalid"] += 1
            continue
        references_by_target[target].append(route)

    tasks: list[RetroTask] = []
    for target in sorted(references_by_target):
        routes = _dedupe_reference_routes(references_by_target[target])
        details = inspect_molecule(target)
        depths = [len(route.steps) for route in routes]
        tasks.append(
            RetroTask(
                task_id=stable_hash(f"route_planning:{target}", prefix="retro_", length=20),
                mode="route_planning",
                target_smiles=target,
                max_steps=min(max_steps, max(depths)),
                stock_id=stock_id,
                split="unassigned",
                reference_routes=tuple(routes),
                min_routes=2 if len(routes) >= 2 else 1,
                max_routes=min(5, max(2, len(routes))),
                difficulty={
                    "depth": min(depths),
                    "max_reference_depth": max(depths),
                    "heavy_atoms": details["heavy_atoms"],
                    "stereochemistry": details["chiral_centres"] > 0,
                    "scaffold": details["murcko_scaffold"],
                    "reference_alternatives": len(routes),
                },
            )
        )
    counters["route_tasks_built"] = len(tasks)
    return tasks, counters


def _reference_replays(
    target: str,
    route: ReferenceRoute,
    stock: frozenset[str],
    verifier: RouteVerifier,
    *,
    mode: str,
) -> bool:
    """Require generation-time references to pass the exact serving verifier."""
    task = RetroTask(
        task_id="generation_replay",
        mode=mode,
        target_smiles=target,
        max_steps=len(route.steps),
        stock_id="generation_stock",
        split="unassigned",
        reference_routes=(route,),
    )
    submitted = {
        "route": [step.to_dict(include_evidence=False) for step in route.steps]
    }
    return verifier.score_route(task, submitted, stock).valid


def assign_strict_splits(
    tasks: list[RetroTask],
    *,
    ratios: tuple[float, float, float, float] = DEFAULT_RATIOS,
    near_duplicate_threshold: float = 0.90,
    near_duplicate_max_tasks: int = 25_000,
) -> tuple[list[RetroTask], dict[str, Any]]:
    """Keep route products/scaffolds, provenance groups, and near-duplicates together."""
    if not tasks:
        raise ValueError("cannot split an empty task set")
    if len(ratios) != len(SPLITS) or any(value < 0 for value in ratios):
        raise ValueError("split ratios must be four non-negative numbers")
    total_ratio = sum(ratios)
    if total_ratio <= 0:
        raise ValueError("split ratios sum to zero")
    ratios = tuple(value / total_ratio for value in ratios)

    groups = UnionFind(len(tasks))
    keyed: dict[tuple[str, str], int] = {}
    for index, task in enumerate(tasks):
        keys: list[tuple[str, str]] = [
            ("target", canonicalize_smiles(task.target_smiles)),
            ("scaffold", scaffold_smiles(task.target_smiles)),
        ]
        for route in task.reference_routes:
            keys.append(("route", route.route_id))
            for step in route.steps:
                product = canonicalize_smiles(step.product)
                keys.append(("route_product", product))
                product_scaffold = scaffold_smiles(product)
                if product_scaffold:
                    keys.append(("route_product_scaffold", product_scaffold))
                if step.reaction_id:
                    keys.append(("reaction", step.reaction_id))
            for source in route.source:
                if source.get("group_id"):
                    keys.append((f"source:{source.get('name', '')}", str(source["group_id"])))
        for key in keys:
            if key in keyed:
                groups.union(index, keyed[key])
            else:
                keyed[key] = index

    near_duplicate_edges = 0
    if near_duplicate_threshold > 0:
        if len(tasks) > near_duplicate_max_tasks:
            raise ValueError(
                f"strict near-duplicate clustering is O(n^2) and is capped at "
                f"{near_duplicate_max_tasks} tasks; shard/cluster upstream or set "
                "--near-duplicate-threshold 0 only with an external cluster audit"
            )
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        route_products = [
            sorted(
                {
                    canonicalize_smiles(step.product)
                    for route in task.reference_routes
                    for step in route.steps
                }
                | {canonicalize_smiles(task.target_smiles)}
            )
            for task in tasks
        ]
        fps = [
            [generator.GetFingerprint(Chem.MolFromSmiles(smiles)) for smiles in products]
            for products in route_products
        ]
        for right in range(1, len(fps)):
            for left in range(right):
                similar = any(
                    any(
                        value >= near_duplicate_threshold
                        for value in DataStructs.BulkTanimotoSimilarity(right_fp, fps[left])
                    )
                    for right_fp in fps[right]
                )
                if similar:
                    groups.union(left, right)
                    near_duplicate_edges += 1

    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(tasks)):
        components[groups.find(index)].append(index)

    # Largest constrained components first; stable hashes break every tie.
    ordered = sorted(
        components.values(),
        key=lambda members: (
            -len(members),
            stable_hash("|".join(sorted(tasks[i].task_id for i in members)), length=32),
        ),
    )
    desired = {name: ratios[i] * len(tasks) for i, name in enumerate(SPLITS)}
    assigned_counts = {name: 0 for name in SPLITS}
    assignments: dict[int, str] = {}
    for members in ordered:
        # Choose the split with the largest normalized deficit. This preserves
        # groups while staying close to requested proportions.
        def priority(name: str) -> tuple[float, float, str]:
            target = desired[name]
            deficit = (target - assigned_counts[name]) / max(target, 1.0)
            # When every split is initially equally empty, place a large
            # constrained component in the split with the largest absolute
            # capacity (normally train), rather than choosing by hash alone.
            return (
                deficit,
                target - assigned_counts[name],
                stable_hash(name + tasks[members[0]].task_id, length=12),
            )

        split = max(SPLITS, key=priority)
        for index in members:
            assignments[index] = split
        assigned_counts[split] += len(members)

    result = [
        RetroTask(
            task_id=task.task_id,
            mode=task.mode,
            target_smiles=task.target_smiles,
            max_steps=task.max_steps,
            stock_id=task.stock_id,
            split=assignments[index],
            reference_routes=task.reference_routes,
            min_routes=task.min_routes,
            max_routes=task.max_routes,
            difficulty=task.difficulty,
            schema_version=task.schema_version,
        )
        for index, task in enumerate(tasks)
    ]
    audit = audit_splits(result)
    if not audit["passed"]:
        raise RuntimeError("split leakage audit failed: " + json.dumps(audit, sort_keys=True))
    return result, {
        "strategy": "strict_route_product_scaffold_source_near_duplicate_components",
        "ratios": dict(zip(SPLITS, ratios)),
        "counts": assigned_counts,
        "components": len(components),
        "component_sizes": sorted((len(members) for members in components.values()), reverse=True),
        "largest_component": max(map(len, components.values())),
        "near_duplicate_threshold": near_duplicate_threshold,
        "near_duplicate_edges": near_duplicate_edges,
        "audit": audit,
    }


def audit_splits(tasks: Iterable[RetroTask]) -> dict[str, Any]:
    by_split: dict[str, dict[str, set[str]]] = {
        split: defaultdict(set) for split in SPLITS
    }
    for task in tasks:
        bucket = by_split[task.split]
        bucket["targets"].add(canonicalize_smiles(task.target_smiles))
        bucket["scaffolds"].add(scaffold_smiles(task.target_smiles))
        for route in task.reference_routes:
            bucket["route_ids"].add(route.route_id)
            for step in route.steps:
                product = canonicalize_smiles(step.product)
                bucket["route_products"].add(product)
                product_scaffold = scaffold_smiles(product)
                if product_scaffold:
                    bucket["route_product_scaffolds"].add(product_scaffold)
                if step.reaction_id:
                    bucket["reaction_ids"].add(step.reaction_id)
            for source in route.source:
                if source.get("group_id"):
                    bucket["source_groups"].add(
                        f"{source.get('name', '')}:{source['group_id']}"
                    )
    overlaps: list[dict[str, Any]] = []
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            for key in (
                "targets",
                "scaffolds",
                "route_products",
                "route_product_scaffolds",
                "route_ids",
                "reaction_ids",
                "source_groups",
            ):
                shared = by_split[left][key] & by_split[right][key]
                if shared:
                    overlaps.append(
                        {
                            "left": left,
                            "right": right,
                            "kind": key,
                            "count": len(shared),
                            "examples": sorted(shared)[:5],
                        }
                    )
    return {"passed": not overlaps, "overlaps": overlaps}


def write_tasks(tasks: list[RetroTask], output_dir: Path, manifest: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        path = output_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for task in sorted((task for task in tasks if task.split == split), key=lambda item: item.task_id):
                handle.write(json.dumps(task.to_dict(), sort_keys=True, separators=(",", ":")) + "\n")
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _dedupe_reference_routes(routes: Iterable[ReferenceRoute]) -> list[ReferenceRoute]:
    unique: dict[str, ReferenceRoute] = {}
    for route in routes:
        key = json.dumps(
            [step.to_dict(include_evidence=False) for step in route.steps], sort_keys=True
        )
        unique.setdefault(key, route)
    return [unique[key] for key in sorted(unique)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--routes", type=Path)
    parser.add_argument("--stock-file", type=Path, required=True)
    parser.add_argument("--stock-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-generated", action="store_true")
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.90)
    parser.add_argument("--near-duplicate-max-tasks", type=int, default=25_000)
    parser.add_argument("--ratios", nargs=4, type=float, metavar=("TRAIN", "DEV", "EVAL", "STRESS"), default=DEFAULT_RATIOS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.corpus and not args.routes:
        raise SystemExit("provide --corpus, --routes, or both")
    stock = load_stock(args.stock_file)
    tasks: list[RetroTask] = []
    counters: Counter[str] = Counter()
    inputs: list[dict[str, str]] = []
    if args.corpus:
        built, counts = tasks_from_corpus(
            args.corpus,
            stock,
            args.stock_id,
            include_generated=args.include_generated,
        )
        tasks.extend(built)
        counters.update(counts)
        inputs.append({"path": str(args.corpus), "sha256": _sha256(args.corpus)})
    if args.routes:
        built, counts = tasks_from_routes(
            args.routes, stock, args.stock_id, max_steps=args.max_steps
        )
        tasks.extend(built)
        counters.update(counts)
        inputs.append({"path": str(args.routes), "sha256": _sha256(args.routes)})

    # Mode is part of the task identity. Same target in the two modes is allowed
    # only if it remains in the same component/split, which target grouping ensures.
    split_tasks, split_manifest = assign_strict_splits(
        tasks,
        ratios=tuple(args.ratios),
        near_duplicate_threshold=args.near_duplicate_threshold,
        near_duplicate_max_tasks=args.near_duplicate_max_tasks,
    )
    manifest = {
        "schema_version": "retro-task-build-v1",
        "inputs": inputs,
        "stock": {"id": args.stock_id, "path": str(args.stock_file), "sha256": _sha256(args.stock_file), "molecules": len(stock)},
        "counts": dict(sorted(counters.items())),
        "split": split_manifest,
    }
    write_tasks(split_tasks, args.output_dir, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
