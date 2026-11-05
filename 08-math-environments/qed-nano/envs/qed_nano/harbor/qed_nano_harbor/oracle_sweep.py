"""Grade every task's reference proof once, so each task records how its own oracle scores.

    JUDGE_API_KEY=... JUDGE_MODEL=openai/gpt-oss-120b:deepinfra JUDGE_REASONING_EFFORT=high \\
        python -m qed_nano_harbor.oracle_sweep out/qed-nano-envs/{train,test}/tasks --out oracle.jsonl --limit 200

Each task's solution/proof.md is graded by that task's own tests/grade.py logic, prompt and rubric, with the judge
set in the environment (a stronger one than the training reward's is the point: this checks the task, not a
policy). The full Harbor path (sandboxes, artifact, separate verifier) is checked separately on samples; running
every oracle through sandboxes would cost more than the grading. Results append to a JSONL file, so a sweep
resumes where it stopped. A low score flags a task: a wrong reference proof (one DeepSeek-Math-V2 proof misses a
solution its Gemini grade let through) or a rubric that does not fit the problem.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .judge_check import load_grade


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tasks", nargs="+", type=Path, help="folders of generated tasks")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="stop after this many graded tasks (0 = all)")
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--price-in", type=float, default=0.0, help="USD per million input tokens, for the cost line")
    ap.add_argument("--price-out", type=float, default=0.0, help="USD per million output tokens")
    args = ap.parse_args()

    grade = load_grade()
    done = {}
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            row = json.loads(line)
            if row.get("score") is not None:
                done[row["task"]] = row
    todo = [d for folder in args.tasks for d in sorted(folder.iterdir())
            if (d / "solution" / "proof.md").exists() and d.name not in done]
    if args.limit:
        todo = todo[: max(0, args.limit - len(done))]
    judge = {"model": os.environ.get("JUDGE_MODEL") or grade.DEFAULT_MODEL,
             "effort": os.environ.get("JUDGE_REASONING_EFFORT", "medium")}
    print(f"{len(done)} already graded, {len(todo)} to grade with {judge}", flush=True)

    def one(d: Path) -> dict:
        task = json.loads((d / "tests" / "task.json").read_text())
        row = {"task": d.name, "set": task["set"], "judge": judge["model"], "effort": judge["effort"]}
        start = time.time()
        try:
            result = grade.grade((d / "solution" / "proof.md").read_text(), task,
                                 (d / "tests" / "judge_prompt.md").read_text(), samples=1)
            usage = result["usage"][0]
            row.update(score=result["score"], prompt_tokens=usage.get("prompt_tokens"),
                       completion_tokens=usage.get("completion_tokens"),
                       assessment=result["grades"][0]["assessment"][-1500:])
        except Exception as exc:  # noqa: BLE001 - record and keep going; a rerun retries it
            row.update(score=None, error=f"{type(exc).__name__}: {exc}"[:300])
        row["seconds"] = round(time.time() - start, 1)
        return row

    rows = list(done.values())
    with args.out.open("a") as sink, ThreadPoolExecutor(args.concurrency) as pool:
        for i, future in enumerate(as_completed([pool.submit(one, d) for d in todo]), 1):
            row = future.result()
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            sink.flush()
            rows.append(row)
            if i % 50 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
    graded = [r for r in rows if r.get("score") is not None]
    tokens_in = sum(r.get("prompt_tokens") or 0 for r in graded)
    tokens_out = sum(r.get("completion_tokens") or 0 for r in graded)
    scores = [r["score"] for r in graded]
    summary = {
        "graded": len(graded), "errors": sum(r.get("score") is None for r in rows),
        "mean_score": round(sum(scores) / len(scores), 3) if scores else None,
        "full_marks": sum(s >= 7 for s in scores), "at_least_6": sum(s >= 6 for s in scores),
        "below_4": sum(s < 4 for s in scores),
        "tokens_in": tokens_in, "tokens_out": tokens_out,
        "usd": round(tokens_in / 1e6 * args.price_in + tokens_out / 1e6 * args.price_out, 2),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
