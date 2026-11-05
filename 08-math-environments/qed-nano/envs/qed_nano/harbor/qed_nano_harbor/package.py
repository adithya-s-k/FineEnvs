"""Assemble the folder published as the Hugging Face dataset FineEnvs/qed-nano-envs.

    python -m qed_nano_harbor.package out/release out/hub --oracle oracle.jsonl
    # after uploading the tasks, pin the registry to that commit so `harbor run --repo ... -d <name>@1.0` resolves:
    python -m qed_nano_harbor.package out/release out/hub --oracle oracle.jsonl \\
        --git-url https://huggingface.co/datasets/FineEnvs/qed-nano-envs --git-commit <sha>

Copies the adapter's output (train/, test/, registry.json, manifest.*), adds generator/ (this package and its
tests), harness/ (the qed-proof agent), the dataset card filled with the oracle sweep's numbers, and the licences.
"""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from pathlib import Path
from string import Template

HARBOR_DIR = Path(__file__).resolve().parents[1]        # envs/qed_nano/harbor
PROJECT = HARBOR_DIR.parents[2]                          # 08-math-environments/qed-nano
REPO_ROOT = PROJECT.parents[1]
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store", "*.egg-info")
QED_NANO_LICENSE = "https://raw.githubusercontent.com/CMU-AIRe/QED-Nano/main/LICENSE"


def oracle_stats(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    graded = [r for r in rows if r.get("score") is not None]
    scores = [r["score"] for r in graded]
    judge = f"{graded[0]['judge'].split(':')[0]} ({graded[0]['effort']} reasoning)" if graded else "—"
    return {"oracle_judge": judge, "oracle_graded": f"{len(graded):,}",
            "oracle_mean": f"{sum(scores) / len(scores):.2f}" if scores else "—",
            "oracle_full": f"{sum(s >= 7 for s in scores):,}", "oracle_validated": f"{sum(s >= 6 for s in scores):,}"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("release", type=Path, help="adapter output (--sets train,test --oracle-results ...)")
    ap.add_argument("out", type=Path)
    ap.add_argument("--oracle", type=Path, required=True, help="the oracle sweep's JSONL")
    ap.add_argument("--git-url")
    ap.add_argument("--git-commit")
    args = ap.parse_args()

    if args.out.exists():
        shutil.rmtree(args.out)
    shutil.copytree(args.release, args.out, ignore=IGNORE)
    gen = args.out / "generator"
    shutil.copytree(HARBOR_DIR / "qed_nano_harbor", gen / "qed_nano_harbor", ignore=IGNORE)
    shutil.copytree(HARBOR_DIR / "tests", gen / "tests", ignore=IGNORE)
    shutil.copytree(PROJECT / "training" / "harness", args.out / "harness", ignore=IGNORE)
    (gen / "qed_nano_harbor" / "templates" / "dataset_README.md").unlink()  # it is the card itself

    manifest = json.loads((args.release / "manifest.json").read_text())
    counts = {}
    for split in ("train", "test"):
        rows = [m for m in manifest if m["set"] == split]
        counts[f"n_{split}"] = f"{len(rows):,}"
        counts[f"n_{split}_oracle"] = f"{sum(1 for m in rows if m['oracle']):,}"
        counts[f"n_{split}_validated"] = f"{sum(1 for m in rows if (m.get('oracle_score') or 0) >= 6):,}"
    card = Template((HARBOR_DIR / "qed_nano_harbor" / "templates" / "dataset_README.md").read_text())
    (args.out / "README.md").write_text(card.substitute(**counts, **oracle_stats(args.oracle)))

    if args.git_url and args.git_commit:
        registry = json.loads((args.out / "registry.json").read_text())
        for entry in registry:
            for task in entry["tasks"]:
                task.update(git_url=args.git_url, git_commit_id=args.git_commit)
        (args.out / "registry.json").write_text(json.dumps(registry, indent=2))

    shutil.copy(REPO_ROOT / "LICENSE", args.out / "LICENSE")
    with urllib.request.urlopen(QED_NANO_LICENSE, timeout=60) as resp:
        (args.out / "LICENSE-QED-Nano").write_bytes(resp.read())
    (args.out / "NOTICE").write_text(
        "QED-Nano Envs\n\n"
        "Problems and marking schemes: lm-provers/FineProofs-RL (Apache-2.0 per its dataset card), curated by the\n"
        "LM-Provers team (CMU, Hugging Face, ETH Zurich, Numina) from AI-MO/olympiads and Art of Problem Solving forum\n"
        "posts (AI-MO/aops). Problems from AoPS remain their authors'.\n\n"
        "Reference proofs (solution/proof.md): DeepSeek-Math-V2 proofs from lm-provers/FineProofs-SFT (Apache-2.0),\n"
        "with the model's appended self-evaluation removed.\n\n"
        "Judge prompt (tests/judge_prompt.md): CMU-AIRe/QED-Nano, training/prompts/evaluator_prompts/v1.md\n"
        "(Apache-2.0, see LICENSE-QED-Nano), unchanged apart from placeholder names.\n\n"
        "Generator, verifier (tests/grade.py) and harness: FineEnvs, Apache-2.0 (LICENSE).\n")
    files = sum(1 for p in args.out.rglob("*") if p.is_file())
    print(f"staged {args.out}: {files} files")


if __name__ == "__main__":
    main()
