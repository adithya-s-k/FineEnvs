"""Re-grade an eval's proofs with another judge, to see how far the training judge's scores can be trusted.

The training reward and the checkpoint evals both use gpt-oss-20b. A rising eval curve could then mean better proofs
or proofs that please that judge. This takes every proof an eval wrote (`<eval>/trials/*/`), grades it again through
the task's own grade.py (same prompt, sanitising and length limit) with the judge set in the environment, and
compares the two. QED-Nano reports its benchmarks with gpt-oss-120b.

    JUDGE_API_KEY=... JUDGE_MODEL=openai/gpt-oss-120b:groq,openai/gpt-oss-120b:deepinfra JUDGE_REASONING_EFFORT=high \\
        python regrade.py runs/fp-bench-4b-step0 --data prepared

Rows go to <eval>/regrade-<judge>.jsonl (a rerun resumes) and the comparison to <eval>/regrade-<judge>.json.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GRADE = ROOT.parent / "envs" / "fineproof" / "harbor" / "fineproof_harbor" / "templates" / "tests" / "grade.py"


def load_grade():
    spec = importlib.util.spec_from_file_location("grade", GRADE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def proof_of(trial: Path) -> str | None:
    """What the verifier read (/workspace/proof.md, kept as an artifact), else the harness's own copy."""
    for path in (trial / "artifacts" / "workspace" / "proof.md", trial / "agent" / "proof.md"):
        if path.is_file():
            return path.read_text(errors="replace")
    return None


def compare(rows: list[dict], max_score: int) -> dict:
    """Attempt-level and problem-level agreement between the original grades and the new ones (0-7 points)."""
    both = [r for r in rows if r.get("new_score") is not None and r.get("old_score") is not None]
    if not both:
        return {"compared": 0}
    mean = statistics.fmean
    old, new = [r["old_score"] for r in both], [r["new_score"] for r in both]
    by_problem: dict[str, list[dict]] = {}
    for r in both:
        by_problem.setdefault(r["task"], []).append(r)
    # QED-Nano's grader metric: compare advantages (a grade minus its problem's mean), which is what GRPO uses.
    adv_diff = []
    for group in by_problem.values():
        if len(group) > 1:
            mo, mn = mean([g["old_score"] for g in group]), mean([g["new_score"] for g in group])
            adv_diff += [abs((g["old_score"] - mo) - (g["new_score"] - mn)) for g in group]
    problem_old = [mean([g["old_score"] for g in v]) for v in by_problem.values()]
    problem_new = [mean([g["new_score"] for g in v]) for v in by_problem.values()]
    return {
        "compared": len(both),
        "problems": len(by_problem),
        "old_mean_reward": round(mean(old) / max_score, 4),
        "new_mean_reward": round(mean(new) / max_score, 4),
        "old_full_marks": round(mean([float(x >= max_score - 1e-9) for x in old]), 4),
        "new_full_marks": round(mean([float(x >= max_score - 1e-9) for x in new]), 4),
        "old_any_credit": round(mean([float(x > 0) for x in old]), 4),
        "new_any_credit": round(mean([float(x > 0) for x in new]), 4),
        "mean_abs_diff_points": round(mean([abs(a - b) for a, b in zip(old, new)]), 3),
        "new_lower_by_3plus": round(mean([float(a - b >= 3) for a, b in zip(old, new)]), 4),
        "new_higher_by_3plus": round(mean([float(b - a >= 3) for a, b in zip(old, new)]), 4),
        "attempt_correlation": round(statistics.correlation(old, new), 3) if len(set(old)) > 1 and len(set(new)) > 1 else None,
        "problem_correlation": round(statistics.correlation(problem_old, problem_new), 3)
        if len(problem_old) > 2 and len(set(problem_old)) > 1 and len(set(problem_new)) > 1 else None,
        "advantage_mad_points": round(mean(adv_diff), 3) if adv_diff else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("eval_dir", type=Path, help="an eval's output folder, with trials/")
    ap.add_argument("--data", default="prepared", help="prepare.py's output, for each task's tests/")
    ap.add_argument("--split", default="test")
    ap.add_argument("--samples", type=int, default=1, help="new grades per proof")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--step", type=int, help="With --trackio-run: the checkpoint step to chart the comparison at")
    ap.add_argument("--trackio-run", help="Trackio run (e.g. <run>-evals) to log `regrade_<judge>/*` into")
    ap.add_argument("--space-id")
    ap.add_argument("--trackio-bucket")
    args = ap.parse_args()

    grade = load_grade()
    judge = os.environ.get("JUDGE_MODEL") or grade.DEFAULT_MODEL
    effort = os.environ.get("JUDGE_REASONING_EFFORT", "medium")
    slug = re.sub(r"[^a-z0-9]+", "-", judge.split(",")[0].split(":")[0].split("/")[-1].lower()) + f"-{effort}"
    out = args.eval_dir / f"regrade-{slug}.jsonl"
    tasks_dir = Path(args.data) / "datasets" / args.split / "tasks"
    done = {}
    if out.exists():
        for line in out.read_text().splitlines():
            row = json.loads(line)
            if row.get("new_score") is not None:
                done[row["trial"]] = row
    trials = [t for t in sorted((args.eval_dir / "trials").iterdir()) if t.is_dir() and t.name not in done]
    print(f"{len(done)} already re-graded, {len(trials)} to grade with {judge} ({effort})", flush=True)

    def one(trial: Path) -> dict:
        result = read_json(trial / "result.json") or {}
        old = read_json(trial / "verifier" / "grading.json") or {}
        name = str(result.get("task_name") or old.get("task") or "").split("/")[-1]
        row = {"trial": trial.name, "task": name, "judge": judge, "effort": effort,
               "old_reward": old.get("reward"), "old_scores": old.get("scores"),
               "old_score": old.get("score") if isinstance(old.get("score"), (int, float))
               else (old["reward"] * grade.MAX_SCORE if isinstance(old.get("reward"), (int, float)) else None)}
        start = time.time()
        proof = proof_of(trial)
        tests = tasks_dir / name / "tests"
        try:
            if proof is None or not proof.strip():
                row.update(new_score=0.0, reason="no proof")
            elif len(proof) > grade.MAX_PROOF_CHARS:
                row.update(new_score=0.0, reason="proof too long")
            else:
                task = json.loads((tests / "task.json").read_text())
                graded = grade.grade(proof, task, (tests / "judge_prompt.md").read_text(), max(1, args.samples))
                row.update(new_score=graded["score"], new_scores=graded["scores"],
                           new_models=[g.get("model") for g in graded["grades"]],
                           usage=graded["usage"], assessment=graded["grades"][0]["assessment"][-1500:])
        except Exception as exc:  # noqa: BLE001 - recorded; a rerun retries it
            row.update(new_score=None, error=f"{type(exc).__name__}: {exc}"[:300])
        row["new_reward"] = round(row["new_score"] / grade.MAX_SCORE, 4) if row.get("new_score") is not None else None
        row["seconds"] = round(time.time() - start, 1)
        return row

    rows = list(done.values())
    with out.open("a") as sink, ThreadPoolExecutor(args.concurrency) as pool:
        for i, future in enumerate(as_completed([pool.submit(one, t) for t in trials]), 1):
            row = future.result()
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            sink.flush()
            rows.append(row)
            if i % 32 == 0:
                print(f"  {i}/{len(trials)}", flush=True)
    summary = {"eval": args.eval_dir.name, "judge": judge, "effort": effort,
               "errors": sum(r.get("new_score") is None for r in rows), **compare(rows, grade.MAX_SCORE)}
    usage = [u for r in rows for u in r.get("usage") or [] if isinstance(u, dict)]
    summary["cost_usd"] = round(sum(u.get("cost") or 0 for u in usage), 3)
    (args.eval_dir / f"regrade-{slug}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if args.trackio_run and args.step is not None:
        import trackio

        run = trackio.init(project="fineproof-rl", name=args.trackio_run, space_id=args.space_id,
                           bucket_id=args.trackio_bucket, private=False if args.space_id else None, resume="allow")
        run.log({f"regrade_{slug}/{k}": v for k, v in summary.items()
                 if isinstance(v, (int, float)) and not isinstance(v, bool)}, step=args.step)
        run.finish()


if __name__ == "__main__":
    main()
