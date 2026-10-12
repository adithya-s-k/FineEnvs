#!/usr/bin/env python3
"""Re-grade no-repair runs as if the harness had repaired their submissions; no model calls.

    uv run --extra eval python eval/regrade.py runs/core30-nothink --server http://127.0.0.1:8010

A run made with --no-repair grades a submission sent as a broken JSON string as invalid, as
training does. This repairs each such final submission the way the harness would have
(normalize_arguments) and sends it to emit_routes on a fresh session of the same server.
emit_routes is the episode's single graded call and its grade depends only on the task and the
submission, so the reward is the one the episode would have had. Writes regraded.json in each
run and REPAIR.md beside the board in <board>/results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from retroenv_openenv.agent import normalize_arguments
from retroenv_openenv.client import RetroEnvClient


def regrade(run: Path, url: str) -> dict | None:
    identity = json.loads((run / "identity.json").read_text())
    if identity.get("repair", True):
        return None
    episodes = [json.loads(path.read_text()) for path in sorted((run / "episodes").glob("*.json"))]
    rows = []
    for episode in (e for e in episodes if e.get("graded")):
        row = {
            "task_id": episode["task_id"],
            "attempt": episode.get("attempt", 0),
            "reward": episode["reward"],
            "valid": episode["valid"],
        }
        broken = any(failure.startswith("invalid JSON") for failure in episode["hard_failures"])
        fixed, coerced = normalize_arguments("emit_routes", {"submission": episode["submission"]})
        if broken and coerced:
            with RetroEnvClient(url) as env:
                env.reset(identity["split"], index=episode["index"])
                outcome = env.call("emit_routes", fixed)
            score = (outcome.result or {}).get("score") or {}
            row.update(reward=float(outcome.reward or 0.0), valid=bool(score.get("valid")), repaired=True)
        rows.append(row)
    n = len(rows) or 1
    result = {
        "label": identity["label"],
        "tasks": len(rows),
        "broken": sum(any(f.startswith("invalid JSON") for f in e["hard_failures"]) for e in episodes),
        "repaired": sum(bool(row.get("repaired")) for row in rows),
        "as_run": {
            "pass_at_1": sum(e["valid"] for e in episodes if e.get("graded")) / n,
            "mean_reward": sum(e["reward"] for e in episodes if e.get("graded")) / n,
        },
        "repaired_view": {
            "pass_at_1": sum(row["valid"] for row in rows) / n,
            "mean_reward": sum(row["reward"] for row in rows) / n,
        },
        "episodes": rows,
    }
    (run / "regraded.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("board", type=Path, help="an evaluate.py output directory")
    parser.add_argument("--server", required=True, help="URL of a RetroEnv server serving the run's split")
    args = parser.parse_args()

    results = [r for run in sorted(args.board.glob("*/summary.json")) if (r := regrade(run.parent, args.server))]
    results.sort(key=lambda r: -r["as_run"]["mean_reward"])
    lines = [
        "| Model | Tasks | Broken JSON | Pass@1 as run | Pass@1 repaired | Reward as run | Reward repaired |",
        "|---|---|---|---|---|---|---|",
        *(
            f"| {r['label']} | {r['tasks']} | {r['broken']} | {r['as_run']['pass_at_1']:.3f} | "
            f"{r['repaired_view']['pass_at_1']:.3f} | {r['as_run']['mean_reward']:.3f} | "
            f"{r['repaired_view']['mean_reward']:.3f} |"
            for r in results
        ),
    ]
    note = (
        "\n**Broken JSON** counts final submissions sent as a JSON string that does not parse; as run, "
        "they score 0, as in training. **Repaired** re-grades them after the harness's repair "
        "(eval/regrade.py), through the same server: the score the model would get from a lenient harness.\n"
    )
    (args.board / "results").mkdir(exist_ok=True)
    (args.board / "results" / "REPAIR.md").write_text("# Submission repair\n\n" + "\n".join(lines) + "\n" + note)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
