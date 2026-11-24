"""Summarize a Harbor oracle job into one verdict per task.

The oracle submits openai/math's own proof of each challenge, so its grade says whether the task is
verified end to end: the statement builds, the official proof compiles in the image, and Comparator
accepts it. Every non-accepted result gets a cause, so a reader can tell an environment limit (disk,
memory, a library the image does not ship) from a real disagreement between proof and statement.

    python -m openai_math_harbor.report <harbor job dir> [<re-run job dir> ...] oracle_results.json
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

# Checked in order against grading.json's reason and the Comparator log.
CAUSES = [
    ("disk", ("no space left on device", "ran out of disk space")),
    # The reference proof states a definition with the same text as the challenge, but compiled in
    # its own library's context it elaborates to a different term, so exact matching fails.
    ("definition-mismatch", ("Const does not match between challenge and target",)),
    ("memory", ("most likely out of memory", "Child exited with 137", "Lean exited with code 137")),
    ("missing-package", ("unknown module prefix", "bad import", "unknown package")),
    ("timeout", ("timed out",)),
]


def classify(trial: Path) -> dict:
    result = json.loads((trial / "result.json").read_text())
    grading = trial / "verifier" / "grading.json"
    log = trial / "verifier" / "comparator.log"
    g = json.loads(grading.read_text()) if grading.exists() else {}
    text = (g.get("reason") or "") + "\n" + (log.read_text(errors="replace") if log.exists() else "")
    reward = ((result.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    if reward == 1:
        cause = "accepted"
    elif reward is None:
        exc = result.get("exception_info") or {}
        if "no space left on device" in (exc.get("exception_traceback") or "") + (exc.get("exception_message") or ""):
            # The verifier's disk filled so completely that Harbor could not even fetch the result.
            cause = "disk"
        else:
            cause = "error: " + str(exc.get("exception_type") or g.get("reason", "unknown"))
    else:
        cause = next((name for name, keys in CAUSES if any(k in text for k in keys)), "rejected")
    return {
        "reward": reward,
        "cause": cause,
        # Graders since the overlay copy-up fix record `permission_fixes`; earlier verdicts other
        # than "accepted" may be artifacts of that bug and are re-graded.
        "current_grader": "permission_fixes" in g,
        "grade_seconds": round(g["finished_at"] - g["started_at"]) if g.get("finished_at") else None,
        "reason": (g.get("reason") or "")[:500],
    }


def summarize(job_dir: Path) -> dict[str, dict]:
    """Task directory name -> verdict. A task attempted twice (a re-run) keeps its latest trial."""
    out, finished = {}, {}
    for trial in sorted(p for p in job_dir.iterdir() if p.is_dir() and (p / "result.json").exists()):
        result = json.loads((trial / "result.json").read_text())
        if not result.get("finished_at"):
            continue
        # Trial directory names are truncated; the task path is not.
        name = Path(result["task_id"]["path"]).name
        if name not in finished or result["finished_at"] > finished[name]:
            out[name], finished[name] = classify(trial), result["finished_at"]
    return out


def merge(*job_dirs: Path) -> dict[str, dict]:
    """Verdicts from several jobs; later jobs (re-runs) override earlier ones."""
    merged = {}
    for job_dir in job_dirs:
        merged.update(summarize(job_dir))
    return merged


if __name__ == "__main__":
    *jobs, out_path = sys.argv[1:]
    results = merge(*(Path(j) for j in jobs))
    Path(out_path).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(len(results), "tasks:", dict(collections.Counter(r["cause"] for r in results.values())))
