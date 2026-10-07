"""Stage 7: write the Hugging Face-ready release directory."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from retroenv.models import TASK_SCHEMA

from .solver import Step, steps_depth
from .targets import tier
from .variants import TaskSpec

STOCK_ID = "paroutes-archive-leaves"
SPLITS = ("train", "dev", "test_id", "test_hard")
PATENTS_PER_ROUTE = 20


def _route(route_id: str, kind: str, steps: list[Step], patents: list[str], meta: dict) -> dict:
    return {
        "route_id": route_id,
        "kind": kind,
        "source": [{"dataset": "paroutes-v2", "patent": patent} for patent in patents[:PATENTS_PER_ROUTE]],
        "steps": [
            {
                "product": product,
                "reactants": list(reactants),
                "reaction_class": meta[(product, reactants)]["reaction_class"],
                "reaction_id": meta[(product, reactants)]["reaction_id"],
            }
            for product, reactants in steps
        ],
    }


def task_row(spec: TaskSpec, split: str, meta: dict) -> dict:
    candidate = spec.candidate
    task_id = candidate.task_id if spec.variant == "standard" else f"{candidate.task_id}-{spec.variant}"
    routes = []
    seen: set[frozenset[Step]] = set()
    for i, route in enumerate(candidate.patent_routes):
        seen.add(frozenset(route.steps))
        routes.append(_route(f"{task_id}:patent:{i}", "patent", route.steps, route.patents, meta))
    for i, steps in enumerate(spec.witnesses):
        if frozenset(steps) in seen:
            continue
        seen.add(frozenset(steps))
        patents = sorted({p for step in steps for p in meta[step]["patents"]})
        routes.append(_route(f"{task_id}:witness:{i}", "witness", steps, patents, meta))
    features = {key: value for key, value in candidate.features.items() if key != "scaffold"}
    return {
        "schema_version": TASK_SCHEMA,
        "task_id": task_id,
        "parent_id": candidate.task_id,
        "variant": spec.variant,
        "split": split,
        "target_smiles": candidate.target,
        "stock_id": STOCK_ID,
        "max_depth": spec.max_depth,
        "min_routes": spec.min_routes,
        "max_routes": 5,
        "constraints": {
            "forbidden_classes": list(spec.forbidden_classes),
            "excluded_stock": list(spec.excluded_stock),
        },
        "difficulty": {
            "min_depth": candidate.min_depth,
            "constrained_min_depth": spec.constrained_min_depth,
            "witness_depths": [steps_depth(candidate.target, steps) for steps in spec.witnesses],
            "tier": tier(candidate.features, candidate.min_depth),
            "scaffold": candidate.features["scaffold"],
            **features,
        },
        "reference_routes": routes,
    }


def public_row(row: dict) -> dict:
    return {key: value for key, value in row.items() if key not in {"reference_routes", "difficulty"}}


def write_jsonl(path: Path, rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            count += 1
    return count


def write_tasks(root: Path, rows_by_split: dict[str, list[dict]]) -> None:
    for split in SPLITS:
        rows = sorted(rows_by_split.get(split, []), key=lambda r: (r["parent_id"], r["variant"]))
        write_jsonl(root / "tasks-private" / f"{split}.jsonl", rows)
        write_jsonl(root / "tasks-public" / f"{split}.jsonl", (public_row(r) for r in rows))


def write_stock(root: Path, stock: frozenset[str]) -> None:
    path = root / "stocks" / f"{STOCK_ID}.smi"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(sorted(stock)) + "\n")


def write_library(root: Path, rows: list[dict], templates: Counter, visible_templates: Counter, reagents: list[str]) -> None:
    library = root / "library"
    library.mkdir(parents=True, exist_ok=True)
    with gzip.open(library / "reactions.jsonl.gz", "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    (library / "templates.json").write_text(json.dumps(dict(templates.most_common()), indent=0))
    (library / "templates-visible.json").write_text(json.dumps(dict(visible_templates.most_common()), indent=0))
    (library / "reagents.json").write_text(json.dumps(reagents, indent=0))


def summarize(rows_by_split: dict[str, list[dict]]) -> dict:
    summary = {}
    for split in SPLITS:
        rows = rows_by_split.get(split, [])
        standard = [r for r in rows if r["variant"] == "standard"]
        summary[split] = {
            "tasks": len(rows),
            "parents": len(standard),
            "variants": dict(sorted(Counter(r["variant"] for r in rows).items())),
            "min_depth": dict(sorted(Counter(r["difficulty"]["min_depth"] for r in standard).items())),
            "tier": dict(sorted(Counter(r["difficulty"]["tier"] for r in standard).items())),
            "convergent": sum(r["difficulty"]["convergent"] for r in standard),
            "stereo_targets": sum(r["difficulty"]["stereocentres"] > 0 for r in standard),
            "forbidden_classes": dict(Counter(c for r in rows for c in r["constraints"]["forbidden_classes"]).most_common()),
        }
    return summary


CARD = """---
license: cc-by-4.0
pretty_name: RetroEnv-RL
task_categories: [reinforcement-learning, text-generation]
tags: [chemistry, retrosynthesis, tool-use, rl-environment, openenv]
size_categories: [10K<n<100K]
configs:
- config_name: tasks
  default: true
  data_files:
{public}
- config_name: tasks_with_known_routes
  data_files:
{private}
---

# RetroEnv-RL

Multistep retrosynthesis tasks for agentic RL. Each task gives a target molecule, a depth
budget (longest linear sequence) and optionally a constraint; the agent plans routes with
tools and submits synthesis trees whose every leaf must be in the frozen stock. A
deterministic verifier judges each step against a frozen reaction library (known
reactions and frequent rdchiral retro-templates), never against a hidden answer; known
routes only add a similarity bonus.

| Split | Targets | Tasks | Use |
|---|---:|---:|---|
{split_rows}

Variants: `max_depth` (cap at the shortest known depth while the patent route is longer),
`restricted_stock` (building blocks of both known routes removed), `forbidden_class` (a
reaction class used by both known routes forbidden) and `diversity` (two or three routes
with different first disconnections). Every task ships a witness route proving it is
solvable under its constraint.

Held-out targets share no patent, route molecule, reaction, scaffold group or Tanimoto ≥
0.90 near-duplicate with train or with each other; library reactions touching any
held-out key are hidden from the agent's tools (`library/reactions.jsonl.gz`, `visible`).

| Path | Contents |
|---|---|
| `tasks-public/` | what a policy may see |
| `tasks-private/` | the same tasks with hidden known routes and difficulty labels |
| `stocks/{stock}.smi` | {molecules:,} building blocks: every leaf of every PaRoutes route |
| `library/` | {reactions:,} corpus reactions, {templates:,} retro-templates, reagent whitelist |
| `manifest.json`, `audit.json`, `checksums.json` | build provenance, leakage and solvability audit, SHA-256 |

Source: PaRoutes v2 (Genheden & Bjerrum, Zenodo 7341155, CC-BY-4.0), patent reactions
from USPTO. Known routes are public, so report results knowing a model may have seen them.
"""
SPLIT_USE = {
    "train": "RL rollouts and SFT bootstrap",
    "dev": "checkpoint selection, reward calibration",
    "test_id": "unseen targets, training distribution",
    "test_hard": "longer, novel, convergent or rare chemistry",
}


def write_card(root: Path, manifest: dict) -> None:
    summary = manifest["summary"]
    def files(folder: str) -> str:
        return "\n".join(f"  - split: {s}\n    path: {folder}/{s}.jsonl" for s in SPLITS)

    rows = "\n".join(
        f"| {s} | {summary[s]['parents']:,} | {summary[s]['tasks']:,} | {SPLIT_USE[s]} |" for s in SPLITS if s in summary
    )
    (root / "README.md").write_text(
        CARD.replace("{public}", files("tasks-public"))
        .replace("{private}", files("tasks-private"))
        .replace("{split_rows}", rows)
        .replace("{stock}", STOCK_ID)
        .replace("{molecules:,}", f"{manifest['stock']['molecules']:,}")
        .replace("{reactions:,}", f"{manifest['library']['reactions']:,}")
        .replace("{templates:,}", f"{manifest['library']['templates']:,}")
    )


def write_checksums(root: Path) -> None:
    sums = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in {"checksums.json", "README.md"} and ".cache" not in path.parts:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(block)
            sums[str(path.relative_to(root))] = digest.hexdigest()
    (root / "checksums.json").write_text(json.dumps({"sha256": sums}, indent=1, sort_keys=True) + "\n")


def by_split(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["split"]].append(row)
    return grouped
