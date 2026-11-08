"""Check the verifier's judge against QED-Nano's two grader benchmarks, the way the QED-Nano post measured graders.

    python -m fineproof_harbor.judge_check --bench matharena --per-problem 8 --out judge-matharena.json
    python -m fineproof_harbor.judge_check --bench olympiads --problems 15 --out judge-olympiads.json

matharena  lm-provers/matharena-gradingbench: 438 MathArena proofs (10 problems) with human points
olympiads  lm-provers/olympiads-proof-graderbench: 480 in-distribution proofs (60 problems) with Gemini 3 Pro points

Each proof is graded by the task verifier's own code (templates/tests/grade.py: same prompt, safeguards and score
parsing). The headline number is QED-Nano's: the mean absolute difference between the judge's and the reference's
*advantages* (a grade minus its problem's mean grade), since a constant shift between graders does not change GRPO.
QED-Nano reports about 1.2 to 1.4 points on the 0-7 scale for gpt-oss-20b. Needs JUDGE_API_KEY (and JUDGE_PROVIDER/JUDGE_MODEL for a judge other than the default).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import statistics
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

HERE = Path(__file__).parent
BENCHES = {
    "matharena": ("lm-provers/matharena-gradingbench", "problem_idx", "problem", "answer", "details", "points"),
    "olympiads": ("lm-provers/olympiads-proof-graderbench", "problem_id", "text", "model_solution", "schema_0", "points"),
}


def load_grade():
    spec = importlib.util.spec_from_file_location("grade", HERE / "templates" / "tests" / "grade.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scheme(items: list[dict], bench: str) -> str:
    """QED-Nano's dict-to-string format. MathArena's `desc` is the human grader's verdict, so it is never shown."""
    parts = []
    for item in items:
        points = item["max_points"] if bench == "matharena" else item["points"]
        text = item["grading_scheme_desc"] if bench == "matharena" else item["desc"]
        parts.append(f"# {item.get('title') or 'Checkpoint'} ({points} points)\nDescription: {text}")
    return "\n\n".join(parts)


def sample(rows: list[dict], key: str, per_problem: int, problems: int, seed: int) -> list[dict]:
    by_problem = defaultdict(list)
    for row in rows:
        by_problem[row[key]].append(row)
    rng = random.Random(seed)
    ids = sorted(by_problem, key=str)
    if problems:
        ids = rng.sample(ids, min(problems, len(ids)))
    out = []
    for pid in ids:
        group = by_problem[pid]
        out += rng.sample(group, min(per_problem, len(group))) if per_problem else group
    return out


def advantages(rows: list[dict], key: str, field: str) -> dict[int, float]:
    by_problem = defaultdict(list)
    for i, row in enumerate(rows):
        by_problem[row[key]].append(i)
    adv = {}
    for idx in by_problem.values():
        mean = statistics.fmean(rows[i][field] for i in idx)
        for i in idx:
            adv[i] = rows[i][field] - mean
    return adv


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", choices=list(BENCHES), required=True)
    ap.add_argument("--per-problem", type=int, default=8, help="proofs per problem (0 = all)")
    ap.add_argument("--problems", type=int, default=0, help="number of problems (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    repo, key, problem, proof, items, points = BENCHES[args.bench]
    path = hf_hub_download(repo, "data/train-00000-of-00001.parquet", repo_type="dataset")
    rows = sample(pq.read_table(path).to_pylist(), key, args.per_problem, args.problems, args.seed)
    grade = load_grade()
    template = (HERE / "prompts" / "rubric.md").read_text()
    done = {}
    if args.out.exists():
        done = {r["i"]: r for r in json.loads(args.out.read_text())["rows"] if r.get("judge") is not None}

    def one(i: int) -> dict:
        if i in done:
            return done[i]
        row = rows[i]
        task = {"id": str(row[key]), "problem": row[problem], "marking_scheme": scheme(row[items], args.bench)}
        out = {"i": i, "problem": str(row[key]), "reference": float(row[points])}
        try:
            result = grade.grade(row[proof] or "", task, template, samples=1)
            out["judge"] = result["score"]
        except Exception as exc:  # noqa: BLE001 - record and keep going
            out["judge"], out["error"] = None, f"{type(exc).__name__}: {exc}"[:300]
        return out

    with ThreadPoolExecutor(args.concurrency) as pool:
        results = list(pool.map(one, range(len(rows))))
    graded = [r for r in results if r["judge"] is not None]
    ref_adv, judge_adv = advantages(graded, "problem", "reference"), advantages(graded, "problem", "judge")
    pairs = [(r["judge"], r["reference"]) for r in graded]
    summary = {
        "bench": args.bench,
        "proofs": len(results),
        "graded": len(graded),
        "problems": len({r["problem"] for r in graded}),
        "advantage_mae": statistics.fmean(abs(judge_adv[i] - ref_adv[i]) for i in range(len(graded))),
        "score_mae": statistics.fmean(abs(j - r) for j, r in pairs),
        "mean_judge": statistics.fmean(j for j, _ in pairs),
        "mean_reference": statistics.fmean(r for _, r in pairs),
        "pearson": statistics.correlation([j for j, _ in pairs], [r for _, r in pairs]) if len(pairs) > 2 else None,
        "exact": sum(j == r for j, r in pairs) / len(pairs),
    }
    args.out.write_text(json.dumps({"summary": summary, "rows": results}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
