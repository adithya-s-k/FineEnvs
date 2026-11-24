"""Build the RetroEnv release (tasks, stock, reaction library) from the PaRoutes v2 archive.

    uv run python -m dataset.build_release            # writes data/release/RetroEnv-RL
    uv run python -m dataset.audit_release            # independent leakage and solvability audit

Expensive stages are cached under data/build/ (delete it to rebuild from scratch).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from collections import Counter
from pathlib import Path

from retroenv.classes import CLASSIFIER_VERSION, CONSTRAINABLE_CLASSES
from retroenv.reactions import DEFAULT_MIN_COUNT
from retroenv.verifier import WEIGHTS

from dataset.pipeline import library, paroutes, release, solver, splits, targets, variants

RAW = Path("data/raw/paroutes-v2-benchmark")
ARCHIVE = "all_loaded_routes.json.gz"


def cached(path: Path, build):
    if path.exists():
        with path.open("rb") as handle:
            return pickle.load(handle)
    value = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def log(message: str, started: float) -> None:
    print(f"[{time.time() - started:7.1f}s] {message}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, default=RAW)
    parser.add_argument("--build-dir", type=Path, default=Path("data/build"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/release/RetroEnv-RL"))
    args = parser.parse_args(argv)
    started = time.time()
    build = args.build_dir

    reactions, routes, _ = cached(
        build / "flat.pkl", lambda: paroutes.flatten(paroutes.load_archive(args.raw_dir / ARCHIVE))
    )
    log(f"flattened {len(routes)} routes, {len(reactions)} unique reactions", started)
    annotated = cached(build / "annotated.pkl", lambda: library.annotate(reactions))
    log("annotated reactions (class, template, map flags)", started)

    stock = frozenset(leaf for route in routes for leaf in route.leaves)
    graph_reactions = [
        solver.GraphReaction(r.product, r.reactants, annotated[key].reaction_class, r.routes, tuple(sorted(r.patents)))
        for key, r in reactions.items()
        if annotated[key].clean
    ]
    clean = {(r.product, r.reactants) for r in graph_reactions}
    graph = solver.RouteGraph(graph_reactions, stock)
    log(
        f"route graph: {len(graph_reactions)} clean reactions, {len(stock)} stock molecules, {len(graph.depth)} solved molecules",
        started,
    )

    rows, template_counts, reagent_counts = library.library_rows(reactions, annotated)
    reagents = library.reagent_whitelist(reagent_counts)
    meta = {
        key: {
            "reaction_class": annotated[key].reaction_class,
            "reaction_id": f"{annotated[key].free_product}>>{'.'.join(annotated[key].free_reactants)}",
            "patents": sorted(reactions[key].patents),
        }
        for key in reactions
    }
    step_templates = {key: template_counts.get(annotated[key].template, 0) for key in reactions}
    log(
        f"library: {len(rows)} stereo-free reactions, {len(template_counts)} templates, {len(reagents)} reagents",
        started,
    )

    by_target = paroutes.group_routes(routes)
    molecules = set(by_target) | {r.product for r in graph_reactions} | {row["product_smiles"] for row in rows}
    features = cached(build / "molecule_features.pkl", lambda: targets.molecule_features(molecules))

    def candidates_with_neighbours():
        found, funnel = targets.build_candidates(by_target, graph, clean, step_templates, features)
        targets.add_neighbour_features(found)
        return found, funnel

    candidates, funnel = cached(build / "candidates.pkl", candidates_with_neighbours)
    log(f"candidates with novelty and near-duplicates: {len(candidates)} ({dict(funnel)})", started)

    generic = splits.generic_scaffolds(candidates)
    rules = splits.ScaffoldRules(generic, {smiles: info["scaffold"] for smiles, info in features.items()})
    reaction_ids = {key: m["reaction_id"] for key, m in meta.items()}
    keys = {c.task_id: splits.candidate_keys(c, rules, reaction_ids) for c in candidates}
    log("leakage keys", started)
    assignment = splits.assign(candidates, keys)
    log(f"splits: {json.dumps(assignment.report['splits'])}", started)

    step_class = {key: m["reaction_class"] for key, m in meta.items()}

    def witness_keys(spec) -> tuple[set, set]:
        steps = [step for steps in spec.witnesses for step in steps]
        molecules = {product for product, _ in steps}
        found = {("reaction", reaction_ids[step]) for step in steps}
        found |= {("molecule", m) for m in molecules} | {("scaffold", rules.key(m)) for m in molecules}
        return found, molecules

    held = [c for c in candidates if assignment.split_of.get(c.task_id) not in (None, "train")]
    held_specs = variants.generate(held, graph, step_class)
    extra_keys, extra_molecules = set(), set()
    for spec in held_specs:
        found, molecules = witness_keys(spec)
        extra_keys |= found
        extra_molecules |= molecules
    splits.extend_held_out(assignment, candidates, keys, extra_keys, extra_molecules)
    train = [c for c in candidates if assignment.split_of.get(c.task_id) == "train"]
    train_specs = [
        s for s in variants.generate(train, graph, step_class) if not witness_keys(s)[0] & assignment.held_out_keys
    ]
    log(f"held-out variant witnesses barred {assignment.report['barred_by_held_out_variants']} train targets", started)
    assigned = held + train
    specs, variant_report = variants.cap_variants(held_specs + train_specs, assignment.split_of)
    specs = [variants.standard(c) for c in assigned] + specs
    log(f"tasks: {len(specs)} including {len(specs) - len(assigned)} variants", started)

    held_patents = {value for kind, value in assignment.held_out_keys if kind == "patent"}
    near_products = splits.near_duplicate_hits([row["product_smiles"] for row in rows], assignment.held_out_molecules)
    visible_templates: Counter = Counter()
    for row in rows:
        product = row["product_smiles"]
        hidden = (
            set(row["patents"]) & held_patents
            or ("molecule", product) in assignment.held_out_keys
            or ("scaffold", rules.key(product)) in assignment.held_out_keys
            or ("reaction", f"{row['product']}>>{'.'.join(row['reactants'])}") in assignment.held_out_keys
            or product in near_products
        )
        row["visible"] = not hidden
        if row["visible"] and row["template"]:
            visible_templates[row["template"]] += 1
        row.pop("clean")
    log(f"library visibility: {sum(r['visible'] for r in rows)} of {len(rows)} reactions train-visible", started)

    rows_by_split = release.by_split(
        [release.task_row(spec, assignment.split_of[spec.candidate.task_id], meta) for spec in specs]
    )
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    release.write_tasks(out, rows_by_split)
    release.write_stock(out, stock)
    release.write_library(out, rows, template_counts, visible_templates, reagents)
    manifest = {
        "schema_version": "retroenv-release-1",
        "task_schema": "retro-task-v2",
        "source": {
            "name": "PaRoutes v2",
            "url": "https://zenodo.org/records/7341155",
            "license": "CC-BY-4.0",
            "archive": ARCHIVE,
            "archive_sha256": sha256(args.raw_dir / ARCHIVE),
            "archive_routes": len(routes),
        },
        "stock": {"id": release.STOCK_ID, "molecules": len(stock), "definition": "every leaf of every archive route"},
        "library": {
            "reactions": len(rows),
            "train_visible_reactions": sum(r["visible"] for r in rows),
            "templates": len(template_counts),
            "template_extractor": "rdchiral 1.1 extract_from_reaction (radius 1, special groups)",
            "min_template_count": DEFAULT_MIN_COUNT,
            "reagent_whitelist": len(reagents),
            "classifier": CLASSIFIER_VERSION,
            "constrainable_classes": list(CONSTRAINABLE_CLASSES),
        },
        "leakage": {
            "keys": [
                "patent",
                "molecule (target and route intermediates)",
                "reaction",
                "scaffold group",
                "near-duplicate",
            ],
            "generic_scaffolds": sorted(generic),
            "near_duplicate_threshold": targets.NEAR_DUPLICATE,
        },
        "design": {
            "target_filters": {
                "heavy_atoms": [targets.MIN_HEAVY, targets.MAX_HEAVY],
                "min_depth": [targets.MIN_DEPTH, solver.MAX_DEPTH],
                "no_metal": True,
            },
            "held_out_size": splits.SPLIT_SIZE,
            "depth_quotas": splits.DEPTH_QUOTAS,
            "test_hard_rule": splits.HARD_RULE,
            "cost_cap": assignment.report["cost_cap"],
            "depth_slack": variants.DEPTH_SLACK,
            "variant_share_cap": variants.VARIANT_SHARE,
            "split_use": release.SPLIT_USE,
        },
        "verifier_weights": WEIGHTS,
        "funnel": dict(funnel),
        "splits": assignment.report,
        "variants": variant_report,
        "summary": release.summarize(rows_by_split),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    release.write_card(out, manifest)
    release.write_checksums(out)
    log(f"wrote {out}", started)
    print(json.dumps(manifest["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
