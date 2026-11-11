"""Load QED-Nano's released problems into one record format.

Every record has: `id` (stable slug), `set`, `problem`, `marking_scheme`, `prompt_style` ("rubric": QED-Nano's
training judge with a 0-7 marking scheme; "imo": its IMO-ProofBench judge with a reference solution and
7/6/1/0 guidelines), and optional `reference_solution`, `reference_proof` (+ `reference_grade`) and metadata.

Sources (all from https://huggingface.co/lm-provers):
  train     FineProofs-RL          5,227 problems, Gemini 3 Pro rubrics, 128 Qwen3-4B-Thinking attempts each
  test      FineProofs-RL-test       128 problems, same fields
  imo       IMOProofBench             60 problems, reference solutions and grading guidelines
  proofbench ProofBench (24_25, other) 145 problems, reference solutions and 0-7 grading schemes
Reference proofs for the oracle come from FineProofs-SFT (DeepSeek-Math-V2 proofs graded by Gemini 3 Pro).
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path

import pyarrow.parquet as pq

REPO = "lm-provers"
FILES = {
    "train": ("FineProofs-RL", "data/train-00000-of-00001.parquet"),
    "test": ("FineProofs-RL-test", "data/test-00000-of-00001.parquet"),
    "imo": ("IMOProofBench", "data/train-00000-of-00001.parquet"),
    "proofbench_24_25": ("ProofBench", "data/24_25-00000-of-00001.parquet"),
    "proofbench_other": ("ProofBench", "data/other-00000-of-00001.parquet"),
}
SFT = ("FineProofs-SFT", ["data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet"])
# The revisions the published tasks were built from, so a rerun reproduces them. None = the latest.
REVISIONS = {
    "FineProofs-RL": "61c9ade2e04ab4ee7e257434e0826cb9b737f63e",
    "FineProofs-RL-test": "f2f55ba4a125aa0cb2fde0e3e75df9ce8971774a",
    "FineProofs-SFT": "73661e62811cf2940a0d3f82788a4f4332204c2f",
}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().lower()


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def problem_id(source: str, problem: str) -> str:
    return f"{source}-{hashlib.sha1(normalize(problem).encode()).hexdigest()[:10]}"


def _read(path: Path, columns: list[str] | None = None) -> list[dict]:
    return pq.read_table(path, columns=columns).to_pylist()


def fetch(cache: Path) -> dict[str, Path]:
    """Download the small parquet files (about 12 MB) and the three SFT columns we use (about 7 MB)."""
    from huggingface_hub import HfFileSystem, hf_hub_download
    import pyarrow as pa

    cache.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, (repo, file) in FILES.items():
        paths[key] = Path(hf_hub_download(f"{REPO}/{repo}", file, repo_type="dataset", revision=REVISIONS.get(repo),
                                            cache_dir=cache / "hub"))
    sft = cache / "FineProofs-SFT.proofs.parquet"
    if not sft.exists():
        fs, parts = HfFileSystem(), []
        for file in SFT[1]:
            with fs.open(f"datasets/{REPO}/{SFT[0]}@{REVISIONS[SFT[0]]}/{file}", "rb", block_size=2**20) as fh:
                parts.append(pq.ParquetFile(fh).read(columns=["problem", "proof", "gemini-3-pro-grade", "category",
                                                              "competition"]))
        pq.write_table(pa.concat_tables(parts), sft, compression="zstd")
    paths["sft"] = sft
    return paths


# DeepSeek-Math-V2 answers with a solution and then grades it ("## Self Evaluation ... the final overall score should
# be 1"). 780 of FineProofs-SFT's 4,281 proofs carry that section; an oracle submits the solution only.
SELF_EVALUATION = re.compile(r"(?:^\s*-{3,}\s*\n\s*)?^\s*#{1,6}\s*self[- ]evaluation\b", re.I | re.M)


def strip_self_evaluation(proof: str) -> str:
    match = SELF_EVALUATION.search(proof)
    return (proof[: match.start()] if match else proof).rstrip()


def reference_proofs(sft_path: Path) -> dict[str, dict]:
    """Normalized problem text -> the best-graded DeepSeek-Math-V2 proof and its metadata."""
    best: dict[str, dict] = {}
    for r in _read(sft_path):
        key = normalize(r["problem"])
        if key not in best or r["gemini-3-pro-grade"] > best[key]["grade"]:
            best[key] = {"proof": strip_self_evaluation(r["proof"]), "grade": r["gemini-3-pro-grade"], "category": r["category"],
                         "competition": r["competition"]}
    return best


def fineproofs(path: Path, split: str, refs: dict[str, dict]) -> list[dict]:
    out = []
    for r in _read(path):
        ref = refs.get(normalize(r["problem"]), {})
        out.append({
            "id": problem_id(r["source"], r["problem"]),
            "set": split,
            "problem": strip_solution_leak(r["problem"].strip()),
            "marking_scheme": r["rubrics"].strip(),
            "prompt_style": "rubric",
            "source": r["source"],
            "category": ref.get("category"),
            "competition": ref.get("competition"),
            "reward_mean": r["reward_mean"],
            "reward_std": r["reward_std"],
            "num_rewards": r["num_rewards"],
            "reference_proof": ref.get("proof"),
            "reference_grade": ref.get("grade"),
        })
    return out


def imo_proofbench(path: Path) -> list[dict]:
    return [{
        "id": f"imo-{slugify(r['question_id'])}",
        "set": "imo",
        "problem": r["problem"].strip(),
        "marking_scheme": r["grading_guidelines"].strip(),
        "reference_solution": r["solution"].strip(),
        "prompt_style": "imo",
        "source": r["source"],
        "category": r["category"],
        "level": r["level"],
    } for r in _read(path)]


def proofbench(path: Path, split: str) -> list[dict]:
    return [{
        "id": f"pb-{slugify(r['problem_id'])}",
        "set": "proofbench",
        "split": split,
        "problem": r["problem"].strip(),
        "marking_scheme": r["grading_scheme"].strip(),
        "reference_solution": r["solution"].strip(),
        "prompt_style": "rubric",
        "source": r["problem_id"],
    } for r in _read(path)]


def load_all(paths: dict[str, Path]) -> dict[str, list[dict]]:
    """Every set, with train decontaminated (see `decontaminate`); `sets["removed"]` lists what train lost and why."""
    refs = reference_proofs(paths["sft"])
    sets = {
        "train": fineproofs(paths["train"], "train", refs),
        "test": fineproofs(paths["test"], "test", refs),
        "imo": imo_proofbench(paths["imo"]),
        "proofbench": proofbench(paths["proofbench_24_25"], "24_25") + proofbench(paths["proofbench_other"], "other"),
    }
    for name, records in sets.items():
        ids = [r["id"] for r in records]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise SystemExit(f"{name}: duplicate task ids {sorted(dupes)[:5]}")
    sets["train"], sets["removed"] = decontaminate(sets["train"], sets["test"], sets["imo"] + sets["proofbench"])
    return sets


# Where a FineProofs problem statement runs on into its official solution (4 rows, all in train): the text from the
# marker on is cut. The task id still hashes the original statement, so ids don't move.
SOLUTION_MARKER = re.compile(
    r"(?im)\bcommon remarks?\b|^\s*#+\s*(?:[a-z]\)\s*)?(?:solution|answer|proof)\b|^\s*(?:solution|answer)\s*\d*\s*[.:]")

# Paraphrases a word-level filter can't see, found by review: train id -> the benchmark problem it restates.
KNOWN_OVERLAPS = {"aops-c8f456a645": "pb-tst-2023-4"}

NEAR_EVAL = 0.7    # word 5-gram containment: a train problem this close to an eval problem is dropped
NEAR_TRAIN = 0.9   # ... and this close to another train problem, the larger id is dropped


def strip_solution_leak(problem: str) -> str:
    match = SOLUTION_MARKER.search(problem)
    return problem[: match.start()].rstrip() if match and match.start() > 0 else problem


def _shingles(text: str, k: int = 5) -> set[str]:
    words = re.findall(r"[a-z0-9]+", normalize(re.sub(r"\\[a-zA-Z]+", " ", text)))
    return {" ".join(words[i:i + k]) for i in range(max(1, len(words) - k + 1))}


def _near(rows: list[dict], others: list[dict], threshold: float, skip=lambda a, b: False):
    """(row, other, containment) for each pair whose shared 5-grams cover `threshold` of the shorter one."""
    index: dict[str, set[int]] = {}
    other_sh = [_shingles(o["problem"]) for o in others]
    for j, sh in enumerate(other_sh):
        for s in sh:
            index.setdefault(s, set()).add(j)
    for row in rows:
        sh = _shingles(row["problem"])
        hits: dict[int, int] = {}
        for s in sh:
            for j in index.get(s, ()):
                hits[j] = hits.get(j, 0) + 1
        for j, n in hits.items():
            score = n / max(1, min(len(sh), len(other_sh[j])))
            if score >= threshold and not skip(row, others[j]):
                yield row, others[j], round(score, 2)


def decontaminate(train: list[dict], test: list[dict], benchmarks: list[dict]) -> tuple[list[dict], list[dict]]:
    """Drop from train: test problems (by id), near-copies of test or benchmark problems, known paraphrases of them,
    and duplicates of other train problems (same text under another source, or near-identical). Returns the kept
    rows and one record per dropped row."""
    removed: dict[str, dict] = {}
    test_ids = {t["id"] for t in test}

    def drop(row: dict, reason: str, other: str, score: float | None = None) -> None:
        removed.setdefault(row["id"], {"id": row["id"], "reason": reason, "matches": other, "containment": score})

    for row in train:
        if row["id"] in test_ids:
            drop(row, "test problem", row["id"])
        elif row["id"] in KNOWN_OVERLAPS:
            drop(row, "paraphrase of a benchmark problem", KNOWN_OVERLAPS[row["id"]])
    pool = [r for r in train if r["id"] not in removed]
    for row, other, score in _near(pool, test, NEAR_EVAL):
        drop(row, "near-copy of a test problem", other["id"], score)
    for row, other, score in _near(pool, benchmarks, NEAR_EVAL):
        drop(row, "near-copy of a benchmark problem", other["id"], score)
    first: dict[str, str] = {}
    for row in sorted(pool, key=lambda r: r["id"]):
        key = normalize(row["problem"])
        if key in first:
            drop(row, "duplicate of a train problem", first[key], 1.0)
        else:
            first[key] = row["id"]
    pool = [r for r in pool if r["id"] not in removed]
    for row, other, score in _near(pool, pool, NEAR_TRAIN, skip=lambda a, b: a["id"] <= b["id"]):
        drop(row, "near-duplicate of a train problem", other["id"], score)
    kept = [r for r in train if r["id"] not in removed]
    return kept, sorted(removed.values(), key=lambda r: (r["reason"], r["id"]))
