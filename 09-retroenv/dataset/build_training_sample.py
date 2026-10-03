#!/usr/bin/env python3
"""Mine a tiny, training-ready alternate-route sample from PaRoutes v2.

The raw archive is scanned without treating stock as prompt context. Candidate
targets must have at least two routes with distinct first disconnections, no
more than ``max_steps`` reaction nodes, and exact closure in the selected n1
stock. Every retained route is replayed through the serving verifier before a
strict route-product/scaffold/source/reaction component split is written.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from retroenv.chemistry import (
    audit_atom_mapping,
    canonicalize_components,
    canonicalize_smiles,
    inspect_molecule,
    scaffold_smiles,
    stable_hash,
)
from retroenv.models import ReactionStep, ReferenceRoute, RetroTask
from retroenv.store import load_stock
from retroenv.taskgen import assign_strict_splits, write_tasks
from retroenv.verifier import RouteVerifier


SOURCE_NAME = "paroutes-v2-benchmark"
SOURCE_LICENSE = "CC-BY-4.0"
SOURCE_URL = "https://zenodo.org/records/7341155"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _raw_descriptor(
    root: Any, raw_stock: set[str], max_steps: int
) -> tuple[str, str, int] | None:
    """Cheap shape/stock pass before applying RDKit to selected records."""
    if not isinstance(root, dict) or root.get("type") != "mol" or not root.get("smiles"):
        return None
    first_cut: tuple[str, ...] = ()
    reaction_count = 0
    valid = True

    def visit(molecule: dict[str, Any], *, is_root: bool = False) -> None:
        nonlocal first_cut, reaction_count, valid
        children = molecule.get("children") or []
        reactions = [child for child in children if child.get("type") == "reaction"]
        if not reactions:
            if str(molecule.get("smiles", "")) not in raw_stock:
                valid = False
            return
        if len(reactions) != 1:
            valid = False
            return
        precursors = [
            child
            for child in reactions[0].get("children") or []
            if child.get("type") == "mol" and child.get("smiles")
        ]
        if not precursors:
            valid = False
            return
        reaction_count += 1
        if reaction_count > max_steps:
            valid = False
            return
        if is_root:
            first_cut = tuple(sorted(str(child["smiles"]) for child in precursors))
        for precursor in precursors:
            visit(precursor)

    visit(root, is_root=True)
    if not valid or not first_cut or not 1 <= reaction_count <= max_steps:
        return None
    return str(root["smiles"]), json.dumps(first_cut, separators=(",", ":")), reaction_count


def _extract_reference(root: dict[str, Any], route_index: int) -> ReferenceRoute:
    steps: list[ReactionStep] = []
    patent_ids: set[str] = set()

    def visit(molecule: dict[str, Any], path: str) -> None:
        reactions = [
            child for child in molecule.get("children") or []
            if child.get("type") == "reaction"
        ]
        if not reactions:
            return
        reaction = reactions[0]
        precursors = [
            child for child in reaction.get("children") or []
            if child.get("type") == "mol" and child.get("smiles")
        ]
        metadata = reaction.get("metadata") or {}
        raw_id = str(metadata.get("ID") or "")
        patent_id = raw_id.split(";", 1)[0] if raw_id else ""
        if raw_id:
            # PaRoutes releases contain both ``PATENT;;step`` and extended
            # semicolon-delimited IDs. The patent is always the first token.
            patent_ids.add(patent_id)
        mapped = str(metadata.get("smiles") or "")
        try:
            mapping_status = audit_atom_mapping(mapped)["status"]
        except Exception:
            mapping_status = "unmapped"
        rsmi = str(metadata.get("rsmi") or "")
        fields = rsmi.split(">")
        condition = {"reported_reagents": fields[1]} if len(fields) == 3 and fields[1] else {}
        steps.append(
            ReactionStep(
                product=canonicalize_smiles(str(molecule["smiles"])),
                reactants=canonicalize_components(
                    str(precursor["smiles"]) for precursor in precursors
                ),
                reaction_id=str(
                    metadata.get("reaction_hash")
                    or metadata.get("ID")
                    or f"paroutes-all:{route_index}:{path}"
                ),
                mapping_status=mapping_status,
                conditions=(condition,) if condition else (),
                literature=(
                    (
                        {
                            "title": f"Patent {patent_id}",
                            "identifier": patent_id,
                            "url": f"https://patents.google.com/patent/{patent_id}/en",
                            "kind": "patent",
                        },
                    )
                    if patent_id
                    else ()
                ),
            )
        )
        for child_index, precursor in enumerate(precursors):
            visit(precursor, f"{path}.{child_index}")

    visit(root, "0")
    groups = sorted(patent_ids) or [f"route-{route_index}"]
    source = tuple(
        {
            "name": SOURCE_NAME,
            "license": SOURCE_LICENSE,
            "url": SOURCE_URL,
            "record_id": f"all_loaded_routes:{route_index}",
            "group_id": group,
        }
        for group in groups
    )
    return ReferenceRoute(
        route_id=f"paroutes-all:{route_index}", steps=tuple(steps), source=source
    )


def _candidate_rows(
    roots: list[dict[str, Any]], raw_stock: set[str], max_steps: int
) -> tuple[sqlite3.Connection, Counter[str]]:
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE routes (target TEXT, first_cut TEXT, route_index INTEGER, steps INTEGER, "
        "PRIMARY KEY (target, first_cut))"
    )
    counters: Counter[str] = Counter()
    for index, root in enumerate(roots):
        counters["raw_routes"] += 1
        descriptor = _raw_descriptor(root, raw_stock, max_steps)
        if descriptor is None:
            counters["raw_shape_stock_or_depth_rejected"] += 1
            continue
        target, first_cut, reaction_count = descriptor
        cursor = connection.execute(
            "INSERT OR IGNORE INTO routes VALUES (?, ?, ?, ?)",
            (target, first_cut, index, reaction_count),
        )
        if cursor.rowcount:
            counters["distinct_target_first_cuts"] += 1
        else:
            counters["duplicate_first_cuts"] += 1
    connection.commit()
    return connection, counters


def build_sample(
    archive: Path,
    stock_path: Path,
    output_dir: Path,
    *,
    sample_size: int = 16,
    max_steps: int = 3,
    ratios: tuple[float, float, float, float] = (0.5, 1 / 6, 1 / 6, 1 / 6),
) -> dict[str, Any]:
    stock = load_stock(stock_path)
    raw_stock = {
        line.split(maxsplit=1)[0]
        for line in stock_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    with gzip.open(archive, "rt", encoding="utf-8") as handle:
        roots = json.load(handle)
    if not isinstance(roots, list):
        raise ValueError("PaRoutes all_loaded_routes archive must contain a list")

    connection, counters = _candidate_rows(roots, raw_stock, max_steps)
    candidates = [
        (str(target), int(count))
        for target, count in connection.execute(
            "SELECT target, COUNT(*) FROM routes GROUP BY target HAVING COUNT(*) >= 2"
        )
    ]
    counters["targets_with_alternate_first_cuts"] = len(candidates)
    candidates.sort(key=lambda item: stable_hash(item[0], length=32))

    # Prefer one task per scaffold, then fill from remaining targets. This is a
    # diversity preference; the later split still treats scaffold as a hard unit.
    ordered: list[tuple[str, int]] = []
    deferred: list[tuple[str, int]] = []
    seen_scaffolds: set[str] = set()
    for candidate in candidates:
        try:
            scaffold = scaffold_smiles(candidate[0])
        except Exception:
            counters["candidate_invalid_target"] += 1
            continue
        if scaffold in seen_scaffolds:
            deferred.append(candidate)
        else:
            seen_scaffolds.add(scaffold)
            ordered.append(candidate)
    ordered.extend(deferred)

    verifier = RouteVerifier()
    tasks: list[RetroTask] = []
    normalized_routes: list[dict[str, Any]] = []
    for raw_target, _ in ordered:
        if len(tasks) >= sample_size:
            break
        rows = list(
            connection.execute(
                "SELECT route_index FROM routes WHERE target = ? ORDER BY first_cut LIMIT 5",
                (raw_target,),
            )
        )
        references: list[ReferenceRoute] = []
        target = canonicalize_smiles(raw_target)
        for (route_index,) in rows:
            try:
                reference = _extract_reference(roots[int(route_index)], int(route_index))
            except Exception:
                counters["selected_extraction_rejected"] += 1
                continue
            replay_task = RetroTask(
                task_id="generation-replay",
                mode="route_planning",
                target_smiles=target,
                max_steps=max_steps,
                stock_id="paroutes-v2-n1",
                split="unassigned",
                reference_routes=(reference,),
            )
            flat_route = {
                "route": [step.to_dict(include_evidence=False) for step in reference.steps]
            }
            if verifier.score_route(replay_task, flat_route, stock).valid:
                references.append(reference)
            else:
                counters["selected_verifier_rejected"] += 1
        first_cuts = {
            tuple(sorted(route.steps[0].reactants)) for route in references if route.steps
        }
        if len(references) < 2 or len(first_cuts) < 2:
            counters["selected_insufficient_verified_alternatives"] += 1
            continue
        details = inspect_molecule(target)
        task = RetroTask(
            task_id=stable_hash(f"sample-route:{target}", prefix="retro_", length=20),
            mode="route_planning",
            target_smiles=target,
            max_steps=max(len(route.steps) for route in references),
            stock_id="paroutes-v2-n1",
            split="unassigned",
            reference_routes=tuple(references),
            min_routes=2,
            max_routes=min(5, len(references)),
            difficulty={
                "depth": min(len(route.steps) for route in references),
                "max_reference_depth": max(len(route.steps) for route in references),
                "heavy_atoms": details["heavy_atoms"],
                "stereochemistry": details["chiral_centres"] > 0,
                "scaffold": details["murcko_scaffold"],
                "reference_alternatives": len(references),
                "distinct_first_cuts": len(first_cuts),
            },
        )
        tasks.append(task)
        normalized_routes.extend(
            {
                "task_id": task.task_id,
                "target_smiles": target,
                **route.to_dict(),
            }
            for route in references
        )

    connection.close()
    if len(tasks) < sample_size:
        raise RuntimeError(
            f"only {len(tasks)} verified alternate-route tasks were found; "
            f"requested {sample_size}"
        )
    split_tasks, split_manifest = assign_strict_splits(
        tasks,
        ratios=ratios,
        near_duplicate_threshold=0.90,
    )
    private_dir = output_dir / "tasks-private"
    public_dir = output_dir / "tasks-public"
    stocks_dir = output_dir / "stocks"
    manifest = {
        "schema_version": "retro-training-sample-v1",
        "source": {
            "name": SOURCE_NAME,
            "license": SOURCE_LICENSE,
            "url": SOURCE_URL,
            "archive": str(archive),
            "archive_sha256": _sha256(archive),
        },
        "stock": {
            "id": "paroutes-v2-n1",
            "source_path": str(stock_path),
            "sha256": _sha256(stock_path),
            "molecules": len(stock),
        },
        "selection": {
            "sample_size": sample_size,
            "max_steps": max_steps,
            "requested_split_ratios": {
                split: ratio
                for split, ratio in zip(("train", "dev", "eval", "stress"), ratios)
            },
            "requires_distinct_first_cuts": True,
            "requires_exact_stock_closure": True,
            "requires_serving_verifier_replay": True,
            "counters": dict(sorted(counters.items())),
        },
        "split": split_manifest,
    }
    write_tasks(split_tasks, private_dir, manifest)
    public_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "eval", "stress"):
        with (public_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for task in sorted(
                (item for item in split_tasks if item.split == split),
                key=lambda item: item.task_id,
            ):
                handle.write(
                    json.dumps(
                        task.to_dict(include_references=False),
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
    stocks_dir.mkdir(parents=True, exist_ok=True)
    (stocks_dir / "paroutes-v2-n1.smi").write_text(
        "".join(f"{smiles}\n" for smiles in sorted(stock)), encoding="utf-8"
    )
    with (output_dir / "normalized-routes.jsonl").open("w", encoding="utf-8") as handle:
        for row in sorted(normalized_routes, key=lambda value: value["route_id"]):
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("data/raw/paroutes-v2-benchmark/all_loaded_routes.json.gz"),
    )
    parser.add_argument(
        "--stock-file",
        type=Path,
        default=Path("data/raw/paroutes-v2-benchmark/stock_n1.txt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("sample"))
    parser.add_argument("--sample-size", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument(
        "--ratios",
        nargs=4,
        type=float,
        metavar=("TRAIN", "DEV", "EVAL", "STRESS"),
        default=(0.5, 1 / 6, 1 / 6, 1 / 6),
    )
    args = parser.parse_args(argv)
    if args.sample_size < 4:
        parser.error("sample-size must be at least four so every split can be exercised")
    print(
        json.dumps(
            build_sample(
                args.archive,
                args.stock_file,
                args.output_dir,
                sample_size=args.sample_size,
                max_steps=args.max_steps,
                ratios=tuple(args.ratios),
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
