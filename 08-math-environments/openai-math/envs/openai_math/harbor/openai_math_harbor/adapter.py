"""Generate Harbor tasks from openai/math's Comparator challenges.

One script covers every challenge: the statement, theorem names, axioms and definition holes all
come from the challenge's own files, so no task is written by hand. What varies per task is data
(see challenges.py for the hole policy that rewrites 8 configs and excludes 1).

    python -m openai_math_harbor.adapter \\
        --openai-math /path/to/openai-math      # checkout with lean/ComparatorChallenges, lean/docs, overview.tex
        --graph graph.json                       # import graph from import_graph.py (oracle closures)
        --out datasets/openai-math
        [--only TruffetCounterexample,PlaneColoring]
        [--embed-reference]                      # bundle the reference proof into solution/ (needs lean/OAI)

Output layout (the FineEnvs Harbor dataset convention):

    <out>/registry.json            Harbor registry: openai-math@<version> -> tasks/*
    <out>/README.md                dataset card
    <out>/NOTICE, LICENSE-openai-math   attribution for the redistributed statements
    <out>/manifest.json            per-task metadata and the excluded challenges
    <out>/tasks/<slug>/            one Harbor task per challenge

Every task gets the same environment/ directory, so Daytona builds one snapshot for all of them.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from string import Template

from .challenges import Challenge, load_challenges
from .oracle import bundle, closure, external_packages

PKG = Path(__file__).parent
IMAGE_DIR = PKG.parent / "image"
TEMPLATES = PKG / "templates"
SOURCE_COMMIT = "adc7f1241b42e322a6451854ab7e4b4c146bf78a"
DATASET_NAME = "openai-math"
DATASET_VERSION = "1.0"
IMAGE_FILES = ["Dockerfile", "lean_setup.sh", "fetch_mathlib.sh", "lakefile.lean", "lake-manifest.json", "pack_submission.py"]


def slug(name: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "-", name).lower()


def toml_str(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def toml_list(values: list[str]) -> str:
    return "[" + ", ".join(toml_str(v) for v in values) + "]"


def grade_timeout(reference_bytes: int) -> int:
    """Seconds the verifier gets to build, export and kernel-replay a submission.

    Most proofs grade in 5 to 30 minutes, but a short file can still be slow (a few heavy
    `nlinarith` calls took over 45 minutes in one 246-line module), so the floor is generous.
    The limit only costs time when a proof actually needs it.
    """
    return int(min(6 * 3600, 3 * 3600 + 900 * reference_bytes / 1e6))


def targets_sentence(c: Challenge) -> str:
    names = [f"`{n}`" for n in c.theorem_names]
    if not names:
        return "The definition below"
    return ("The theorem " if len(names) == 1 else "The theorems ") + ", ".join(names)


def holes_sentence(c: Challenge) -> str:
    if not c.definition_holes:
        return ""
    names = ", ".join(f"`{n}`" for n in c.definition_holes)
    return (f", and the body of {names}, which you must supply: its type states the claim, so any "
            "value of that type proves it")


def instruction(c: Challenge) -> str:
    fam = c.family
    return Template((TEMPLATES / "instruction.md").read_text()).substitute(
        title=f"openai/math challenge `{c.name}` (family {fam.number})",
        family_number=fam.number, family_title=fam.title, subject=fam.subject,
        family_summary=fam.summary, challenge=c.name, lean=c.lean.rstrip(),
        targets=targets_sentence(c), holes=holes_sentence(c),
    )


def task_toml(c: Challenge, ref: dict, agent_timeout: int, oracle: dict | None = None) -> str:
    fam = c.family
    verifier_timeout = ref["grade_timeout_sec"] + 900
    oracle_lines = ""
    if oracle:
        oracle_lines = f'oracle_result = {toml_str(oracle["cause"])}\n'
        if oracle.get("grade_seconds") is not None:
            oracle_lines += f'oracle_grade_seconds = {oracle["grade_seconds"]}\n'
    return f"""schema_version = "1.4"

[task]
name = "openai-math/{slug(c.name)}"
version = "1.0.0"
description = {toml_str(f"Prove {c.name} (openai/math family {fam.number}: {fam.title}) in Lean 4 with Mathlib; graded by Comparator in a separate sandbox.")}
keywords = ["lean4", "mathlib", "theorem-proving", "openai-math", {toml_str(fam.subject.lower().replace(" ", "-"))}]

[metadata]
source = "https://github.com/openai/math"
source_commit = "{SOURCE_COMMIT}"
source_license = "Apache-2.0"
source_notice = "tests/Challenge.lean is lean/ComparatorChallenges/{c.name}.lean from openai/math, (c) OpenAI, unchanged; see NOTICE at the dataset root"
challenge = {toml_str(c.name)}
family = {toml_str(fam.number)}
family_title = {toml_str(fam.title)}
subject = {toml_str(fam.subject)}
theorems = {toml_list(c.theorem_names)}
definition_holes = {toml_list(c.definition_holes)}
fixed_definitions = {toml_list(c.dropped_holes)}
reference_module = {toml_str(c.config["solution_module"])}
reference_modules = {ref["n_modules"]}
reference_lean_bytes = {ref["bytes"]}
reference_requires_packages = {toml_list(ref["packages"])}
{oracle_lines}difficulty = "research"

[agent]
timeout_sec = {agent_timeout}
network_mode = "no-network"

[environment]
build_timeout_sec = 5400
cpus = 4
memory_mb = 8192
storage_mb = 10240
workdir = "/workspace"
# Lean sees all host CPUs rather than the sandbox's 4; this bounds `lake build` to 3 parallel jobs.
env = {{ LEAN_NUM_THREADS = "3" }}

[verifier]
timeout_sec = {verifier_timeout}
environment_mode = "separate"
network_mode = "no-network"

[[verifier.collect]]
command = "python3 /opt/openai-math/pack_submission.py"
timeout_sec = 300
"""


def write_task(c: Challenge, out: Path, graph: dict | None, lean_dir: Path | None,
               embed_reference: bool, agent_timeout: int, oracle: dict | None = None) -> dict:
    task_dir = out / "tasks" / slug(c.name)
    if task_dir.exists():
        shutil.rmtree(task_dir)
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "tests").mkdir()
    (task_dir / "solution").mkdir()

    for name in IMAGE_FILES:
        shutil.copy(IMAGE_DIR / name, task_dir / "environment" / name)

    solution_module = c.config["solution_module"]
    modules = closure(graph, solution_module) if graph else []
    ref = {
        "n_modules": len(modules),
        "bytes": sum(graph[m]["bytes"] for m in modules) if graph else 0,
        "packages": external_packages(graph, modules) if graph else [],
    }
    ref["grade_timeout_sec"] = grade_timeout(ref["bytes"])

    (task_dir / "instruction.md").write_text(instruction(c))
    (task_dir / "task.toml").write_text(task_toml(c, ref, agent_timeout, oracle))

    (task_dir / "tests" / "Challenge.lean").write_text(c.lean)
    (task_dir / "tests" / "task.json").write_text(json.dumps({
        "challenge": c.name,
        "comparator": c.comparator,
        "grade_timeout_sec": ref["grade_timeout_sec"],
    }, indent=2))
    for name in ("test.sh", "grade.py"):
        shutil.copy(TEMPLATES / "tests" / name, task_dir / "tests" / name)
    (task_dir / "tests" / "test.sh").chmod(0o755)

    (task_dir / "solution" / "closure.txt").write_text("\n".join(modules) + "\n")
    shutil.copy(PKG / "reference.py", task_dir / "solution" / "reference.py")
    solve = Template((TEMPLATES / "solution" / "solve.sh").read_text()).substitute(
        challenge=c.name, solution_module=solution_module, n_modules=len(modules),
        source_commit=SOURCE_COMMIT,
    )
    (task_dir / "solution" / "solve.sh").write_text(solve)
    (task_dir / "solution" / "solve.sh").chmod(0o755)
    if embed_reference:
        if not (lean_dir and (lean_dir / "OAI").is_dir()):
            raise SystemExit("--embed-reference needs the openai/math checkout's lean/OAI directory")
        (task_dir / "solution" / "reference.tar.gz").write_bytes(bundle(lean_dir, modules, solution_module))
    return {
        "task": slug(c.name), "challenge": c.name, "family": c.family.number,
        "family_title": c.family.title, "subject": c.family.subject,
        "theorems": c.theorem_names, "definition_holes": c.definition_holes,
        "fixed_definitions": c.dropped_holes, **ref,
        "oracle_result": (oracle or {}).get("cause"),
        "oracle_grade_seconds": (oracle or {}).get("grade_seconds"),
    }


ORACLE_MEANING = {
    "accepted": "OpenAI's proof is accepted: the environment is verified end to end",
    "missing-package": "OpenAI's proof uses a Lean library beyond Mathlib that the sandbox does not include",
    "definition-mismatch": "OpenAI's proof writes a definition with the same text but it compiles differently, so the exact-match check fails",
    "memory": "OpenAI's proof needs more than the sandbox's 8 GB of memory to build",
    "disk": "OpenAI's proof is too large for the sandbox's 10 GB disk",
    "timeout": "grading OpenAI's proof exceeded the time limit",
    "rejected": "Comparator rejected OpenAI's proof",
}


def validation_block(written: list[dict]) -> str:
    """Markdown summary of the oracle run for the dataset card."""
    results = [w for w in written if w.get("oracle_result")]
    if not results:
        return "The oracle has not been run on this build of the dataset."
    counts: dict[str, list[str]] = {}
    # No verdict: not run yet, or the run was interrupted (`error: ...`) rather than graded.
    pending = [w["task"] for w in written if w.get("oracle_result") not in ORACLE_MEANING]
    for w in results:
        if w["oracle_result"] in ORACLE_MEANING:
            counts.setdefault(w["oracle_result"], []).append(w["task"])
    order = [k for k in ORACLE_MEANING if k in counts]
    rows = ["| Oracle result | Tasks | Meaning |", "|---|---|---|"]
    for key in order:
        rows.append(f"| {key} | {len(counts[key])} | {ORACLE_MEANING[key]} |")
    if pending:
        rows.append(f"| not completed | {len(pending)} | grading did not complete in our run |")
    lines = ["\n".join(rows), ""]
    secs = sorted(w["oracle_grade_seconds"] for w in results
                  if w["oracle_result"] == "accepted" and w.get("oracle_grade_seconds"))
    if secs:
        lines.append(f"Grading an accepted proof took {secs[len(secs) // 2] // 60} minutes at the median. "
                     "Each task's result is in its `task.toml` (`oracle_result`) and in `manifest.parquet`.")
    unexplained = sorted(t for k in ("rejected", "timeout") for t in counts.get(k, []))
    if unexplained:
        lines.append("Other: " + ", ".join(f"`{t}`" for t in unexplained) + ".")
    return "\n".join(lines)


def write_parquet_manifest(out: Path, written: list[dict]) -> None:
    """manifest.parquet: one row per task, for filtering without walking tasks/."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError:
        print("pyarrow not installed; skipping manifest.parquet")
        return
    columns = ["task", "challenge", "family", "family_title", "subject", "theorems", "definition_holes",
               "fixed_definitions", "n_modules", "bytes", "packages", "grade_timeout_sec",
               "oracle_result", "oracle_grade_seconds"]
    rename = {"n_modules": "reference_modules", "bytes": "reference_lean_bytes",
              "packages": "reference_requires_packages"}
    table = pa.table({rename.get(c, c): [w.get(c) for w in written] for c in columns})
    pq.write_table(table, out / "manifest.parquet")


def not_validated_block(not_validated: list[dict], excluded: list[dict]) -> str:
    """Markdown list of the statements that are not in the dataset, with the reason for each."""
    rows = [*({"task": n["task"], "reason": ORACLE_MEANING.get(n["reason"], "grading did not complete in our run")} for n in not_validated),
            *({"task": slug(e["challenge"]), "reason": "left out by design: " + e["reason"]} for e in excluded)]
    if not rows:
        return "None: every statement is included."
    lines = ["<details><summary>" + f"{len(rows)} statements not included (click to expand)" + "</summary>", "",
             "| Statement | Why it could not be validated |", "|---|---|"]
    lines += [f"| `{r['task']}` | {r['reason']} |" for r in sorted(rows, key=lambda r: r["task"])]
    lines += ["", "</details>"]
    return "\n".join(lines)


def write_dataset_files(out: Path, included: list[Challenge], written: list[dict],
                        not_validated: list[dict] | None = None, excluded: list[dict] | None = None,
                        n_statements: int | None = None) -> None:
    """registry.json, the dataset card, NOTICE and the openai/math license copy."""
    registry = [{
        "name": DATASET_NAME,
        "version": DATASET_VERSION,
        "description": f"{len(written)} Lean 4 theorem-proving tasks from openai/math, graded by Comparator in a separate sandbox.",
        "tasks": [{"name": w["task"], "path": f"tasks/{w['task']}"} for w in written],
    }]
    (out / "registry.json").write_text(json.dumps(registry, indent=2) + "\n")
    families = {c.family.number for c in included}
    subjects = sorted({c.family.subject for c in included})
    counts = {s: sum(1 for c in included if c.family.subject == s) for s in subjects}
    card = Template((TEMPLATES / "dataset_README.md").read_text()).substitute(
        validation=validation_block(written + [{"task": n["task"], "oracle_result": n["reason"]}
                                               for n in (not_validated or []) if n["reason"] != "pending"]
                                    + [{"task": n["task"]} for n in (not_validated or []) if n["reason"] == "pending"]),
        not_validated=not_validated_block(not_validated or [], excluded or []),
        n_statements=n_statements or len(written),
        n_not_included=(n_statements or len(written)) - len(written),
        n_tasks=len(written), n_families=len(families), n_subjects=len(subjects),
        source_commit=SOURCE_COMMIT[:7],
        subjects_line=", ".join(f"{s} ({counts[s]})" for s in sorted(subjects, key=lambda s: -counts[s])),
    )
    (out / "README.md").write_text(card)
    write_parquet_manifest(out, written)
    (out / "NOTICE").write_text(Template((TEMPLATES / "NOTICE").read_text()).substitute(source_commit=SOURCE_COMMIT))
    shutil.copy(TEMPLATES / "LICENSE-openai-math", out / "LICENSE-openai-math")
    shutil.copy(TEMPLATES / "LICENSE", out / "LICENSE")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--openai-math", type=Path, required=True)
    ap.add_argument("--graph", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", help="comma-separated challenge names")
    ap.add_argument("--embed-reference", action="store_true")
    ap.add_argument("--agent-timeout", type=int, default=4 * 3600)
    ap.add_argument("--oracle-results", type=Path,
                    help="oracle verdicts from `python -m openai_math_harbor.report`, stamped into each task")
    ap.add_argument("--only-validated", action="store_true",
                    help="keep only tasks whose official proof the oracle verified; list the rest in the card")
    args = ap.parse_args()

    graph = json.loads(args.graph.read_text())["modules"] if args.graph else None
    oracle = json.loads(args.oracle_results.read_text()) if args.oracle_results else {}
    challenges = load_challenges(args.openai_math)
    if args.only:
        wanted = set(args.only.split(","))
        challenges = [c for c in challenges if c.name in wanted]
        missing = wanted - {c.name for c in challenges}
        if missing:
            raise SystemExit(f"unknown challenges: {sorted(missing)}")

    args.out.mkdir(parents=True, exist_ok=True)
    written, excluded, not_validated = [], [], []
    for c in challenges:
        if c.excluded:
            excluded.append({"challenge": c.name, "reason": c.excluded})
            continue
        entry = write_task(c, args.out, graph, args.openai_math / "lean", args.embed_reference,
                           args.agent_timeout, oracle.get(slug(c.name)))
        if args.only_validated and entry["oracle_result"] != "accepted":
            shutil.rmtree(args.out / "tasks" / entry["task"])
            not_validated.append({"task": entry["task"], "challenge": c.name, "family": c.family.number,
                                  "reason": entry["oracle_result"] if entry["oracle_result"] in ORACLE_MEANING else "not completed"})
            continue
        written.append(entry)
    (args.out / "manifest.json").write_text(json.dumps({
        "source": "https://github.com/openai/math", "source_commit": SOURCE_COMMIT,
        "tasks": written, "excluded": excluded, "not_validated": not_validated,
    }, indent=2))
    kept = {w["task"] for w in written}
    write_dataset_files(args.out, [c for c in challenges if slug(c.name) in kept], written,
                        not_validated=not_validated, excluded=excluded, n_statements=len(challenges))
    print(f"wrote {len(written)} tasks to {args.out / 'tasks'}; excluded {len(excluded)}")
    for e in excluded:
        print(f"  excluded {e['challenge']}: {e['reason']}")


if __name__ == "__main__":
    main()
