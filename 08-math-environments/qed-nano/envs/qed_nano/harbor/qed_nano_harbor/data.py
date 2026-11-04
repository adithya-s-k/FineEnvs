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
        paths[key] = Path(hf_hub_download(f"{REPO}/{repo}", file, repo_type="dataset", cache_dir=cache / "hub"))
    sft = cache / "FineProofs-SFT.proofs.parquet"
    if not sft.exists():
        fs, parts = HfFileSystem(), []
        for file in SFT[1]:
            with fs.open(f"datasets/{REPO}/{SFT[0]}/{file}", "rb", block_size=2**20) as fh:
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
            "problem": r["problem"].strip(),
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
    return sets
