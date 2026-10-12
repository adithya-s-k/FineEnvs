#!/usr/bin/env python3
"""Reward ablation: score cached evidence under every candidate and report.

    uv run python -m research.reward.ablate build  --runs runs/rewardlab runs/openrouter-30
    uv run python -m research.reward.ablate report

`build` loads the release once, distils every stored episode submission and a
battery of synthetic probes into research/reward/evidence/*.jsonl. `report`
needs none of that: it scores the cached evidence under each variant and writes
research/reward/report.json plus a console summary.

What the report answers, per variant:

* hack margin   - can any failure out-score any solve?
* staircase     - does pushing a route one level closer to stock always pay?
* grounding     - does mean reward rank models the way pass rate does?
* signal        - is there still within-task variance for a policy to learn from?
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .evidence import EvidenceBuilder, SubmissionEvidence, read_evidence, write_evidence
from .probes import build as build_probes
from .variants import RewardConfig, catalogue, score

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_DIR = Path(__file__).resolve().parent / "evidence"
SHIPPED = RewardConfig("shipped-truth")
STRICT = RewardConfig("strict-truth", min_template_count=25, support_rule="sole")
WORKERS = min(48, os.cpu_count() or 8)

# Workers fork after the release and its task/stock caches are warm, so the
# template index is shared copy-on-write instead of loaded 48 times.
_BUILDER: EvidenceBuilder | None = None


def _distil(job: tuple[str, str, str, str, Any]) -> dict[str, Any]:
    task_id, split, label, run, submission = job
    assert _BUILDER is not None
    task = _BUILDER.task(task_id, split)
    return _BUILDER.build(task, submission, label=label, run=run).to_dict()


def episodes(run_dirs: Iterable[Path]) -> list[dict[str, Any]]:
    rows = []
    for run_dir in run_dirs:
        for identity_path in sorted(run_dir.rglob("identity.json")):
            identity = json.loads(identity_path.read_text())
            for path in sorted((identity_path.parent / "episodes").glob("*.json")):
                row = json.loads(path.read_text())
                if not row.get("graded") or row.get("submission") is None:
                    continue
                rows.append(
                    {
                        "label": identity["label"],
                        "run": str(identity_path.parent.relative_to(run_dir)),
                        "split": row["split"],
                        "task_id": row["task_id"],
                        "attempt": row["attempt"],
                        "submission": row["submission"],
                        "shipped_reward": row["reward"],
                        "shipped_valid": row["valid"],
                    }
                )
    return rows


def build(run_dirs: list[Path], release: Path, probe_tasks: int) -> None:
    global _BUILDER
    from retroenv.benchmark import Benchmark

    print(f"loading {release} ...", file=sys.stderr)
    benchmark = Benchmark.load(release)
    _BUILDER = EvidenceBuilder(benchmark.library, benchmark.store)

    rows = episodes(run_dirs)
    splits = {row["task_id"]: row["split"] for row in rows}
    print(f"{len(rows)} graded submissions over {len(splits)} tasks", file=sys.stderr)

    jobs = [
        (row["task_id"], row["split"], f"{row['label']}|{row['attempt']}", row["run"], row["submission"])
        for row in rows
    ]
    probe_jobs = []
    for task_id in sorted(splits)[:probe_tasks]:
        task = _BUILDER.task(task_id, splits[task_id])
        stock = benchmark.store.stock(task.stock_id)
        probe_jobs += [(task_id, splits[task_id], name, "probe", sub) for name, sub in build_probes(task, stock).items()]
    for task_id, split in splits.items():
        _BUILDER.context(_BUILDER.task(task_id, split))
    print(f"{len(probe_jobs)} probe submissions; forking {WORKERS} workers", file=sys.stderr)

    for name, batch in (("episodes", jobs), ("probes", probe_jobs)):
        with Pool(WORKERS) as pool:
            records = [SubmissionEvidence.from_dict(row) for row in pool.imap(_distil, batch, chunksize=4)]
        write_evidence(EVIDENCE_DIR / f"{name}.jsonl", records)
        print(f"  {name}: {len(records)} records", file=sys.stderr)


def verify(run_dirs: list[Path], tolerance: float = 2e-3) -> int:
    """Replay every stored grade from the cached evidence. Any drift means the harness lies."""
    stored = {
        (row["label"], row["run"], row["task_id"], row["attempt"]): row
        for row in episodes(run_dirs)
    }
    mismatched = []
    checked = 0
    for record in read_evidence(EVIDENCE_DIR / "episodes.jsonl"):
        label, _, attempt = record.label.rpartition("|")
        row = stored.get((label, record.run, record.task_id, int(attempt)))
        if row is None:
            continue
        checked += 1
        result = score(record, SHIPPED)
        if result["solved"] != row["shipped_valid"] or abs(result["reward"] - row["shipped_reward"]) > tolerance:
            mismatched.append((record.run, record.task_id, row["shipped_reward"], result["reward"]))
    print(f"{checked} episodes replayed, {len(mismatched)} disagree with the server's grade", file=sys.stderr)
    for run, task, want, got in mismatched[:10]:
        print(f"  {run} {task}: stored {want:.4f} vs replay {got:.4f}", file=sys.stderr)
    return 0 if not mismatched else 1


def frame(records: list[SubmissionEvidence], configs: list[RewardConfig]) -> pd.DataFrame:
    rows = []
    for record in records:
        row = {
            "task_id": record.task_id,
            "label": record.label.split("|")[0],
            "run": record.run,
            "variant_task": record.variant,
            "pass_shipped": score(record, SHIPPED)["solved"],
            "pass_strict": score(record, STRICT)["solved"],
        }
        for config in configs:
            result = score(record, config)
            row[f"r:{config.name}"] = result["reward"]
            row[f"s:{config.name}"] = result["solved"]
        rows.append(row)
    return pd.DataFrame(rows)


def hack_margin(table: pd.DataFrame, column: str, solved: pd.Series) -> dict[str, float]:
    passes, failures = table[column][solved], table[column][~solved]
    if passes.empty or failures.empty:
        return {}
    return {
        "min_pass": round(float(passes.min()), 4),
        "max_fail": round(float(failures.max()), 4),
        "margin": round(float(passes.min() - failures.max()), 4),
        "mean_pass": round(float(passes.mean()), 4),
        "mean_fail": round(float(failures.mean()), 4),
        "overlap_rate": round(float((failures > passes.min()).mean()), 4),
    }


def padding(table: pd.DataFrame, column: str) -> dict[str, float]:
    """What a solve loses by being padded with duplicates or junk. Must be positive."""
    clean = table[table["label"] == "solve-one-route"].set_index("task_id")[column]
    out = {}
    for name in ("pad-duplicate-x3", "pad-junk"):
        padded = table[table["label"] == name].set_index("task_id")[column]
        shared = clean.index.intersection(padded.index)
        if len(shared):
            out[name] = round(float((clean[shared] - padded[shared]).mean()), 4)
    return out


def staircase(table: pd.DataFrame, column: str) -> dict[str, float]:
    stairs = table[table["label"].str.contains("stair-")].copy()
    if stairs.empty:
        return {}
    stairs["level"] = stairs["label"].str.extract(r"stair-(\d+)of").astype(int)
    monotone, lifts = [], []
    for _, group in stairs.groupby("task_id"):
        ordered = group.sort_values("level")[column].tolist()
        if len(ordered) < 2:
            continue
        monotone.append(all(b > a for a, b in zip(ordered, ordered[1:])))
        lifts.append(ordered[-1] - ordered[0])
    if not monotone:
        return {}
    return {
        "monotone_rate": round(sum(monotone) / len(monotone), 4),
        "mean_lift_first_to_full": round(sum(lifts) / len(lifts), 4),
    }


MIN_EPISODES_PER_MODEL = 20  # a model's pass rate is too noisy below this to rank on


def grounding(table: pd.DataFrame, column: str, truth: str) -> dict[str, float]:
    by_model = table.groupby("label").agg(reward=(column, "mean"), rate=(truth, "mean"), n=(column, "size"))
    by_model = by_model[by_model["n"] >= MIN_EPISODES_PER_MODEL]
    if len(by_model) < 3:
        return {}
    return {
        "spearman_vs_pass": round(float(by_model["reward"].corr(by_model["rate"], method="spearman")), 4),
        "pearson_vs_pass": round(float(by_model["reward"].corr(by_model["rate"], method="pearson")), 4),
        "mean_abs_gap": round(float((by_model["reward"] - by_model["rate"]).abs().mean()), 4),
        "max_abs_gap": round(float((by_model["reward"] - by_model["rate"]).abs().max()), 4),
    }


def signal(table: pd.DataFrame, column: str, truth: str) -> dict[str, float]:
    """Within-task spread is what a group-relative objective turns into advantage."""
    groups = table.groupby("task_id")[column]
    spread = groups.std().fillna(0.0)
    failures = table[column][~table[truth]]
    return {
        "mean_within_task_std": round(float(spread.mean()), 4),
        "tasks_with_signal": round(float((spread > 0.01).mean()), 4),
        "failure_std": round(float(failures.std()), 4) if len(failures) > 1 else 0.0,
        "failure_distinct_rate": round(float(failures.round(3).nunique() / max(len(failures), 1)), 4),
        "zero_reward_rate": round(float((table[column] < 1e-9).mean()), 4),
    }


def report(configs: list[RewardConfig]) -> dict[str, Any]:
    episode_records = read_evidence(EVIDENCE_DIR / "episodes.jsonl")
    probe_records = read_evidence(EVIDENCE_DIR / "probes.jsonl")
    episode_table = frame(episode_records, configs)
    probe_table = frame(probe_records, configs)
    # True hacks cost nothing to produce and must stay below every real solve;
    # padded solves are scored separately because they do solve the task.
    graded_probes = probe_table[~probe_table["label"].str.startswith("pad-")].copy()
    probe_is_solve = graded_probes["label"].str.startswith("solve-")

    out: dict[str, Any] = {
        "episodes": len(episode_table),
        "probes": len(probe_table),
        "models": sorted(episode_table["label"].unique()),
        "pass_rate_shipped": round(float(episode_table["pass_shipped"].mean()), 4),
        "pass_rate_strict": round(float(episode_table["pass_strict"].mean()), 4),
        "variants": {},
    }
    for config in configs:
        column = f"r:{config.name}"
        own = episode_table[f"s:{config.name}"]
        out["variants"][config.name] = {
            "episodes": {
                # Internal consistency: can a submission this reward calls failed
                # out-score one it calls solved?
                "hack_margin": hack_margin(episode_table, column, own),
                "hack_margin_vs_strict_pass": hack_margin(episode_table, column, episode_table["pass_strict"]),
                "own_pass_rate": round(float(own.mean()), 4),
                "grounding_shipped": grounding(episode_table, column, "pass_shipped"),
                "grounding_strict": grounding(episode_table, column, "pass_strict"),
                "grounding_own": grounding(episode_table.assign(own=own), column, "own"),
                "signal": signal(episode_table, column, "pass_shipped"),
            },
            "probes": {
                "hack_margin": hack_margin(graded_probes, column, probe_is_solve),
                "staircase": staircase(probe_table, column),
                "padding_penalty": padding(probe_table, column),
                "by_probe": {
                    name: round(float(group[column].mean()), 4)
                    for name, group in probe_table.groupby("label")
                    if not name.startswith("stair-")
                },
            },
        }
    out["per_model"] = (
        episode_table.groupby("label")
        .agg(
            n=("task_id", "size"),
            pass_shipped=("pass_shipped", "mean"),
            pass_strict=("pass_strict", "mean"),
            **{config.name: (f"r:{config.name}", "mean") for config in configs},
        )
        .round(4)
        .to_dict("index")
    )
    return out


def console(out: dict[str, Any]) -> None:
    rows = []
    for name, data in out["variants"].items():
        episodes_data, probes_data = data["episodes"], data["probes"]
        rows.append(
            {
                "variant": name,
                "margin": episodes_data["hack_margin"].get("margin"),
                "max_fail": episodes_data["hack_margin"].get("max_fail"),
                "min_pass": episodes_data["hack_margin"].get("min_pass"),
                "probe_margin": probes_data["hack_margin"].get("margin"),
                "stub": probes_data["by_probe"].get("hack-stock-stub"),
                "pad_dup": probes_data["padding_penalty"].get("pad-duplicate-x3"),
                "pad_junk": probes_data["padding_penalty"].get("pad-junk"),
                "stair_lift": probes_data["staircase"].get("mean_lift_first_to_full"),
                "stair_mono": probes_data["staircase"].get("monotone_rate"),
                "spearman": episodes_data["grounding_own"].get("spearman_vs_pass"),
                "gap": episodes_data["grounding_own"].get("mean_abs_gap"),
                "signal": episodes_data["signal"]["tasks_with_signal"],
                "fail_std": episodes_data["signal"]["failure_std"],
            }
        )
    pd.set_option("display.width", 200, "display.max_columns", 50)
    print(pd.DataFrame(rows).to_string(index=False))
    print()
    print(pd.DataFrame(out["per_model"]).T.to_string())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    builder = sub.add_parser("build")
    builder.add_argument("--runs", type=Path, nargs="+", default=[ROOT / "runs" / "rewardlab"])
    builder.add_argument("--release", type=Path, default=ROOT / "data" / "release" / "RetroEnv-RL")
    builder.add_argument("--probe-tasks", type=int, default=40)
    reporter = sub.add_parser("report")
    reporter.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "report.json")
    checker = sub.add_parser("verify")
    checker.add_argument("--runs", type=Path, nargs="+", default=[ROOT / "runs" / "rewardlab"])
    args = parser.parse_args(argv)

    if args.command == "build":
        build(args.runs, args.release, args.probe_tasks)
        return 0
    if args.command == "verify":
        return verify(args.runs)
    out = report(catalogue())
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    console(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
