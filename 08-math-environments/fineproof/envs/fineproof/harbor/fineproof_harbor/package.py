"""Assemble the folders published as the Hugging Face datasets FineEnvs/fineproof-envs-train and -test.

    python -m fineproof_harbor.package out/release out/hub
    # after uploading, pin each registry to its upload commit so `harbor run --repo ...` resolves:
    python -m fineproof_harbor.package out/release out/hub --pin train=<sha> --pin test=<sha>

One repo per split, as the source has (FineProofs-RL, FineProofs-RL-test), each with its tasks under `tasks/` (the
layout Harbor Visualiser and other Hub tools read), its own registry.json and manifest.parquet, generator/ (this
package and its tests), harness/ (the fineproof agent), the dataset card and the licences.
"""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from pathlib import Path
from string import Template

from .adapter import DATASET, VALIDATED_SCORE, VERSION, write_parquet
from .data import REVISIONS

HARBOR_DIR = Path(__file__).resolve().parents[1]        # envs/fineproof/harbor
PROJECT = HARBOR_DIR.parents[2]                          # 08-math-environments/fineproof
REPO_ROOT = PROJECT.parents[1]
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store", "*.egg-info")
QED_NANO_LICENSE = "https://raw.githubusercontent.com/CMU-AIRe/QED-Nano/main/LICENSE"
SPLITS = {
    "train": {"source": "FineProofs-RL", "desc": "the proof problems used to train",
              "overlap": "FineProofs-RL also holds the 128 test problems; they are left out here."},
    "test": {"source": "FineProofs-RL-test", "desc": "the held-out proof problems used to evaluate",
             "overlap": "These problems also appear in FineProofs-RL; the train set leaves them out."},
}
NOTICE = """FineProofEnvs

Problems and marking schemes: lm-provers/FineProofs-RL and FineProofs-RL-test (Apache-2.0 per their dataset cards),
curated by the LM-Provers team (CMU, Hugging Face, ETH Zurich, Numina) from AI-MO/olympiads and Art of Problem
Solving forum posts (AI-MO/aops). Problems from AoPS remain their authors'.

Reference proofs (solution/proof.md): DeepSeek-Math-V2 proofs from lm-provers/FineProofs-SFT (Apache-2.0), with the
model's appended self-evaluation removed.

Judge prompt (tests/judge_prompt.md): CMU-AIRe/QED-Nano, training/prompts/evaluator_prompts/v1.md (Apache-2.0, see
LICENSE-QED-Nano), unchanged apart from placeholder names.

Generator, verifier (tests/grade.py) and harness: FineEnvs, Apache-2.0 (LICENSE).
"""


def registry(split: str, rows: list[dict], git_url: str | None, commit: str | None) -> list[dict]:
    def entry(suffix: str, subset: list[dict], what: str) -> dict:
        tasks = [{"name": r["task"], "path": f"tasks/{r['task']}"} for r in subset]
        if git_url and commit:
            for t in tasks:
                t.update(git_url=git_url, git_commit_id=commit)
        return {"name": f"{DATASET}-{split}{suffix}", "version": VERSION,
                "description": f"{len(subset)} olympiad proof tasks ({split}{what}) from QED-Nano's release, graded "
                               "0-7 by an LLM judge against each problem's marking scheme; reward = score / 7.",
                "metrics": [{"type": "mean"}], "tasks": tasks}

    validated = [r for r in rows if (r.get("oracle_score") or 0) >= VALIDATED_SCORE]
    return [entry("", rows, ""), entry("-validated", validated, f", reference proof scored >= {VALIDATED_SCORE}/7")]


def example_task(rows: list[dict], tasks: Path) -> str:
    """A short problem whose reference proof scored 7, for the card's example links."""
    full = [r["task"] for r in rows if r.get("oracle_score") == 7] or [r["task"] for r in rows]
    return min(sorted(full), key=lambda t: (tasks / t / "instruction.md").stat().st_size)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("release", type=Path, help="adapter output (--sets train,test --oracle-results ...)")
    ap.add_argument("out", type=Path, help="one folder per split is staged here")
    ap.add_argument("--org", default="FineEnvs")
    ap.add_argument("--pin", action="append", default=[], metavar="SPLIT=SHA",
                    help="pin that split's registry to this commit of its Hub repo")
    args = ap.parse_args()
    pins = dict(p.split("=", 1) for p in args.pin)

    manifest = json.loads((args.release / "manifest.json").read_text())
    with urllib.request.urlopen(QED_NANO_LICENSE, timeout=60) as resp:
        qed_license = resp.read()
    card = Template((HARBOR_DIR / "fineproof_harbor" / "templates" / "dataset_README.md").read_text())

    for split, info in SPLITS.items():
        name = f"{DATASET}-{split}"
        repo, out = f"{args.org}/{name}", args.out / name
        if out.exists():
            shutil.rmtree(out)
        rows = sorted((m for m in manifest if m["set"] == split), key=lambda m: m["task"])
        shutil.copytree(args.release / split / "tasks", out / "tasks", ignore=IGNORE)
        git_url = f"https://huggingface.co/datasets/{repo}" if split in pins else None
        (out / "registry.json").write_text(json.dumps(registry(split, rows, git_url, pins.get(split)), indent=2))
        write_parquet(out / "manifest.parquet", rows)

        gen = out / "generator"
        shutil.copytree(HARBOR_DIR / "fineproof_harbor", gen / "fineproof_harbor", ignore=IGNORE)
        shutil.copytree(HARBOR_DIR / "tests", gen / "tests", ignore=IGNORE)
        (gen / "fineproof_harbor" / "templates" / "dataset_README.md").unlink()  # it is the card itself
        shutil.copytree(PROJECT / "training" / "harness", out / "harness", ignore=IGNORE)

        oracle = [r["oracle_score"] for r in rows if r.get("oracle_score") is not None]
        source = f"lm-provers/{info['source']}"
        (out / "README.md").write_text(card.substitute(
            org=args.org, repo=repo, split=split, version=VERSION,
            size_category="1K<n<10K" if len(rows) >= 1000 else "n<1K",
            source_repo=source, source_rev=REVISIONS[info["source"]][:12], source_desc=info["desc"],
            overlap_note=info["overlap"], example=example_task(rows, out / "tasks"),
            n_tasks=f"{len(rows):,}", n_oracle=f"{sum(1 for r in rows if r['oracle']):,}",
            n_validated=f"{sum(s >= VALIDATED_SCORE for s in oracle):,}",
            oracle_mean=f"{sum(oracle) / len(oracle):.1f}" if oracle else "-"))

        shutil.copy(REPO_ROOT / "LICENSE", out / "LICENSE")
        (out / "LICENSE-QED-Nano").write_bytes(qed_license)
        (out / "NOTICE").write_text(NOTICE)
        files = sum(1 for p in out.rglob("*") if p.is_file())
        print(f"staged {out}: {len(rows)} tasks, {files} files" + (f", registry pinned to {pins[split]}" if split in pins else ""))


if __name__ == "__main__":
    main()
