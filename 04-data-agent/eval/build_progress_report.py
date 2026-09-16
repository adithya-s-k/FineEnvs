"""Snapshot the real resume chain and first-graded checkpoint evals without changing a run."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
import subprocess

WORKSPACE = Path(__file__).resolve().parents[3]
LOGS = WORKSPACE / "experiments/async_grpo_harbor_data_agent/logs"
HARNESSES = ["opencode", "claude-code", "codex", "mini-swe-agent"]
NAMES = ["OpenCode", "Claude Code", "Codex", "Mini-SWE-Agent"]
LEVELS = ["easy", "medium", "hard"]


def read(path):
    return json.loads(path.read_text())


def records(path):
    # A writer may currently be appending the last line. Never read half a record.
    return [json.loads(s) for s in path.read_text().splitlines(keepends=True) if s.endswith("\n")]


def count(rows):
    correct = int(sum(r["reward"] for r in rows))
    return {"graded": len(rows), "correct": correct,
            "pass_at_1": correct / len(rows) if rows else None}


def cell(value):
    if not value["graded"]:
        return "pending"
    return f"{value['pass_at_1']:.1%} ({value['correct']}/{value['graded']})"


def evaluation_paths(main):
    paths, seen, upper = {}, set(), float('inf')
    root = main
    while root not in seen:
        seen.add(root)
        for p in (root / 'checkpoint-evals').glob('step-*'):
            step = int(p.name.split('-')[1])
            if step > upper:
                continue
            if step in paths and paths[step] != p:
                raise ValueError('Conflicting checkpoint evaluation ancestry')
            paths[step] = p
        resume = read(root / 'run_config.json').get('resume_state') or {}
        if not resume.get('checkpoint'):
            break
        upper, root = resume['step'], Path(resume['checkpoint']).parents[2]
    return paths


def matched_counts(selected):
    common = set.intersection(*(set(rows) for rows in selected.values()))
    return {name: count([rows[k] for k in sorted(common)]) for name, rows in selected.items()}


def collect(main):
    base = LOGS / "multi4-baseline-20260914"
    manifest = read(base / "manifest.json")
    identities = [(t["name"], t["question_hash"], t["difficulty"]) for t in manifest["tasks"]]
    expected = {(h, i) for h in HARNESSES for i in range(250)}
    source = base / "job-78215/canonical_results.json"
    selected = {"base": {(r["harness"], r["index"]): r for r in read(source)["selected"]}}
    sources = {"base": str(source)}
    final = {"base": True}
    ungraded = {}
    evaluations = {}
    paths = evaluation_paths(main)
    for step, directory in sorted(paths.items()):
        if not (directory / "manifest.json").exists():
            continue
        assert identities == [(t["name"], t["question_hash"], t["difficulty"])
                              for t in read(directory / "manifest.json")["tasks"]]
        rows, failed = {}, set()
        for path in sorted(directory.glob("job-*/traces/*.jsonl")):
            for row in records(path):
                assert row.get("rep", 0) == 0
                key = (row["harness"], row["index"])
                assert key in expected
                if row.get("reward") in (0, 1) and row.get("n_turns", 0) > 0:
                    if key in rows:
                        assert rows[key]["reward"] == row["reward"], "Graded result was replaced"
                    rows.setdefault(key, row)
                else:
                    failed.add(json.dumps(row, sort_keys=True))
        if not rows:
            continue
        name = str(step)
        selected[name] = rows
        score = read(directory / "scores.json") if (directory / "scores.json").exists() else {}
        final[name] = score.get("comparison_ready", False)
        if final[name]:
            assert len(rows) == 1000 and count(list(rows.values()))["correct"] == round(score["average_pass_at_1"] * 1000)
        ungraded[name] = len(failed)
        sources[name] = str(directory)
    for name, rows in selected.items():
        evaluation = {**count(list(rows.values())), "final_audited": final[name], "harnesses": {}, "difficulty": {}}
        for harness in HARNESSES:
            values = [r for (h, _), r in rows.items() if h == harness]
            evaluation["harnesses"][harness] = {**count(values), "difficulty": {
                level: count([r for r in values if manifest["tasks"][r["index"]]["difficulty"] == level])
                for level in LEVELS}}
        for level in LEVELS:
            evaluation["difficulty"][level] = count([r for r in rows.values()
                if manifest["tasks"][r["index"]]["difficulty"] == level])
        evaluations[name] = evaluation
    latest = next(reversed(selected))
    matched = matched_counts(selected)

    # Retain only optimizer steps inherited through the actual checkpoint ancestry.
    root, upper, metrics, provenance = main, float("inf"), {}, []
    current_job = str(read(main / "submission.json")["training"])
    while True:
        config = read(root / "run_config.json")
        job = str(read(root / "submission.json")["training"])
        path = root / f"job-{job}/audit/metrics.jsonl"
        resume = config.get("resume_state") or {}
        lower = resume.get("step", 0)
        rows = [r for r in records(path) if "grad_norm" in r and lower < r["step"] <= upper]
        for row in rows:
            assert row["step"] not in metrics
            metrics[row["step"]] = {**row, "source_job": job}
        provenance.append({"job": job, "resume_step": lower, "file": str(path),
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        if not resume.get("checkpoint"):
            break
        upper, root = lower, Path(resume["checkpoint"]).parents[2]
    rows = [metrics[k] for k in sorted(metrics)]
    assert [r["step"] for r in rows] == list(range(1, rows[-1]["step"] + 1))
    current = [r for r in rows if r["source_job"] == current_job]
    windows = []
    for lower, upper in [(54, 100), (101, 150), (151, 200), (201, 250), (251, 280), (281, 310), (311, rows[-1]["step"])]:
        subset = [r for r in rows if lower <= r["step"] <= upper]
        if subset:
            windows.append({"from": lower, "to": subset[-1]["step"], "updates": len(subset),
                "mean_reward": mean(r["reward"] for r in subset),
                "nonzero_gradients": sum(r["grad_norm"] > 0 for r in subset),
                "zero_group_variance_updates": sum(r.get("reward_std", -1) == 0 for r in subset),
                "mean_step_seconds": mean(r["perf/step_s"] for r in subset if "perf/step_s" in r)
                    if any("perf/step_s" in r for r in subset) else None})
    audit = main / f"job-{current_job}/audit"
    checkpoints = []
    for p in sorted((audit.parent / "run").glob("checkpoint-*")):
        marker = p / "checkpoint.saved.json"
        if marker.exists():
            m = read(marker)
            checkpoints.append({"step": m["step"], "saved": True, "ready_marker": (p / "checkpoint.ready.json").exists()})
    monitor = read(main / "monitor/status.json")
    training = {"job": current_job, "latest_step": rows[-1]["step"], "current_job_updates": len(current),
                "nonzero_gradient_updates": sum(r["grad_norm"] > 0 for r in current),
                "nonfinite_updates": sum(any(not math.isfinite(r[k]) for k in ["loss", "grad_norm", "ratio", "kl", "entropy"]
                    if isinstance(r.get(k), (int, float))) for r in current),
                "max_observed_staleness": max(r.get("sample/staleness_max", 0) for r in current),
                "stale_rollouts_dropped": sum(r.get("admission/stale_rollouts_dropped_total", 0) for r in current),
                "stale_rows_dropped": sum(r.get("sample/dropped_stale_total", 0) for r in current),
                "oversized_rows_dropped": sum(r.get("batch/dropped_oversize_total", 0) for r in current),
                "last20_nonzero_gradients": sum(r["grad_norm"] > 0 for r in current[-20:]),
                "last20_zero_group_variance": sum(r.get("reward_std", -1) == 0 for r in current[-20:]),
                "last_nonzero_gradient_step": max(r["step"] for r in current if r["grad_norm"] > 0),
                "windows": windows, "coverage": read(audit / "coverage.json"),
                "tito": read(audit / "tito_summary.json"),
                "audit_timestamp": datetime.fromtimestamp((audit / "tito_summary.json").stat().st_mtime, timezone.utc).isoformat(),
                "checkpoints": sorted(checkpoints, key=lambda x: x["step"]),
                "trackio": read(main / "trackio/status.json"),
                "monitor_timestamp": monitor["checked_at"], "monitor_alerts": monitor["alerts"],
                "supervisor": read(main / "supervisor/status.json"), "source_chain": provenance}
    return {"snapshot_utc": datetime.now(timezone.utc).isoformat(), "evaluations": evaluations,
            "training": training, "matched_cells": matched, "latest_checkpoint": latest,
            "ungraded_attempts": ungraded, "sources": sources}, rows


def render(report):
    ev, t = report["evaluations"], report["training"]
    labels = ["Base" if k == "base" else f"Checkpoint {k}" + (" (partial)" if not v["final_audited"] else "") for k,v in ev.items()]
    header = " | ".join(labels)
    sep = " | ".join(["---:"] * len(labels))
    lines = [f"# Multi-harness progress — {report['snapshot_utc']}", "",
        f"Trainer **{t['job']}** is at **step {t['latest_step']}** on hopper-prod. "
        "Qwen3.5-2B; E2B; OpenCode, Claude Code, Codex and Mini-SWE-Agent. "
        "LR 3e-6, eight generations per task, max staleness four. Saves every 50 steps plus hourly recovery saves; independent eval every 100 steps.", "",
        "The fixed test set contains 250 tasks: 33 easy, 118 medium and 99 hard. Each complete checkpoint evaluation has 1,000 pass@1 cells. Partial scores remain provisional until coverage, token and harness-version audits pass.", "",
        f"| Harness | {header} |", f"| --- | {sep} |"]
    for h, name in zip(HARNESSES, NAMES):
        lines.append("| " + name + " | " + " | ".join(cell(e["harnesses"][h]) for e in ev.values()) + " |")
    lines += ["| Overall | " + " | ".join(cell(e) for e in ev.values()) + " |", "",
              f"| Difficulty, all harnesses | {header} |", f"| --- | {sep} |"]
    for level in LEVELS:
        lines.append("| " + level + " | " + " | ".join(cell(e["difficulty"][level]) for e in ev.values()) + " |")
    lines += ["", f"| Harness | Difficulty | {header} |", f"| --- | --- | {sep} |"]
    for h, name in zip(HARNESSES, NAMES):
        for level in LEVELS:
            lines.append("| " + name + " | " + level + " | " + " | ".join(cell(e["harnesses"][h]["difficulty"][level]) for e in ev.values()) + " |")
    lines += ["", "On the cells completed by every displayed checkpoint: " + "; ".join(
        f"{k}: {cell(v)}" for k, v in report["matched_cells"].items()) + ".", "",
        "Infrastructure attempts without a graded result are retained separately and may be retried. A scored zero is never replaced by a retry. Ungraded attempts by checkpoint: " + json.dumps(report["ungraded_attempts"]) + ".", "",
        "## Training signal and reliability", "",
        "| Steps | Mean logged reward | Nonzero-gradient updates | Mean step time |",
        "| --- | ---: | ---: | ---: |"]
    for w in t["windows"]:
        seconds = f"{w['mean_step_seconds']:.1f}s" if w['mean_step_seconds'] is not None else "not logged"
        lines.append(f"| {w['from']}–{w['to']} | {w['mean_reward']:.3f} | {w['nonzero_gradients']}/{w['updates']} | {seconds} |")
    passed = sum(a["tito_pass"] for a in t["tito"].values())
    completed = sum(a["completed_results"] for a in t["tito"].values())
    retained = sum(a["retained_tokens"] for a in t["tito"].values())
    eligible = sum(a["eligible_tokens"] for a in t["tito"].values())
    lines += ["", "Reward is the unweighted mean of logged optimizer-update reward, not fixed-task pass@1. The changing task/harness mix affects it. Zero within-group reward variance produces no relative-advantage learning signal; high mean reward alone does not guarantee useful updates.", "",
        f"The current continuation has {t['nonzero_gradient_updates']}/{t['current_job_updates']} nonzero-gradient updates and {t['nonfinite_updates']} nonfinite updates. "
        f"The latest 20 have {t['last20_nonzero_gradients']} nonzero gradients and {t['last20_zero_group_variance']} zero-variance updates. "
        f"Observed staleness is at most {t['max_observed_staleness']}; {int(t['stale_rollouts_dropped'])} whole rollouts ({int(t['stale_rows_dropped'])} rows) were rejected by the staleness policy; {int(t['oversized_rows_dropped'])} oversized rows were dropped.", "",
        f"Latest saved checkpoints: {', '.join(str(c['step']) for c in t['checkpoints'])}. "
        f"The audited captures pass TiTO on {passed}/{completed} completed rollouts, retaining {retained:,}/{eligible:,} eligible tokens. "
        f"Audit timestamp: {t['audit_timestamp']}. This is capture/sequence-assembly evidence, not a claim that every captured rollout reached the optimizer.", "",
        f"Continuation coverage: {t['coverage']['unique_tasks_covered']}/1,000 unique training tasks. The continuation started with a saved schedule cursor, so this count excludes the parent run’s earlier coverage; optimizer steps are not unique tasks.", "",
        f"Monitor alerts: {json.dumps(t['monitor_alerts'])} at {t['monitor_timestamp']}. "
        f"The latest nonzero-gradient update is step {t['last_nonzero_gradient_step']}; monitor alerts may precede newer updates. "
        f"Offline logging healthy: {t['trackio'].get('local_ok')}; latest online Trackio sync successful: {t['trackio'].get('sync', {}).get('ok')} at {t['trackio'].get('checked_at')}.", "",
        "Source paths, exact counts, resume-chain provenance and timestamps are preserved in the companion JSON snapshot."]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=LOGS / "multi4-long-prod-20260915")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    report, rows = collect(args.run)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "snapshot.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.out / "training_metrics.json").write_text(json.dumps(rows) + "\n")
    markdown = render(report)
    (args.out / "REPORT.md").write_text(markdown)
    (Path(__file__).resolve().parents[1] / "train/PROGRESS.md").write_text(markdown)
    print(json.dumps({"snapshot": report["snapshot_utc"], "step": report["training"]["latest_step"],
                      "evaluations": {k: {q:v[q] for q in ["graded","correct","pass_at_1","final_audited"]} for k,v in report["evaluations"].items()}}))


if __name__ == "__main__":
    main()
