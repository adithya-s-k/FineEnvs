"""Label train tasks by how often the base model solves them. Read the numbered sections in order.

In the first 100-step AsyncGRPO run, 119 of 165 task groups never solved once, so most updates
came from partial credit rather than from solving versus failing. Before the next run, this
sweep samples every task in a fixed sample several times with RL's sampling settings and keeps
three things:

- pass@k by difficulty tier: whether the base model ever solves medium and hard tasks, which
  decides whether RL needs a supervised warm start first;
- solves per task: RL learns from tasks solved in some attempts but not all, where a group's
  rewards disagree;
- the solved episodes, as chat transcripts for that warm start.

    python train/difficulty_sweep.py select
    python train/jobs/hf_job.py eval --set train --task-ids train/sweeps/train-sweep250-0.txt \\
        --attempts 8 --temperature 0.8 --concurrency 64 --name sweep250-0 --bucket <owner>/<bucket> --submit
    python train/difficulty_sweep.py report runs/sweep250 --server http://127.0.0.1:8010

The evaluation runs with the board's harness and protocol (eval/run_eval.py: thinking off,
16 turns, no submission repair), so a label measures the task as the board would see it.
"""

# %% 1. Choose the sample: the evaluation sets' own tiered draw, over the train split.
import argparse
import itertools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SWEEPS = ROOT / "train" / "sweeps"
NAME, SEED = "train-sweep250", "retroenv-train-sweep-v1"
# Equal tiers, like core30. Train tasks are mostly easy and shallow (the previous run's depth
# limit kept only easy and medium tasks whose shortest route is 2 steps), while core30's medium
# and hard tasks need 2 to 8; the draw visits shortest-route depths round-robin, as the
# evaluation sets do, so deep tasks get measured too.
QUOTAS = {"easy": 84, "medium": 83, "hard": 83}
SHARDS = 2  # one H200 job each; one vLLM per job keeps every turn of an episode in its prefix cache


def select(args):
    from retroenv.evalsets import make_evalset, select_tiered
    from retroenv.store import TaskStore

    store = TaskStore(args.release / "tasks-private", args.release / "stocks")
    quotas = dict(QUOTAS) if not args.quotas else {
        tier: int(count) for tier, count in (part.split("=") for part in args.quotas.split(","))
    }
    quotas = {tier: count for tier, count in quotas.items() if count}
    candidates = [task for task in store.tasks("train") if task.variant == "standard"]
    if args.from_set:
        # Draw inside a curated set (data/eval-<set>.json), e.g. train12k, LiteFold/RetroEnv-Curated's
        # training tasks, which share no leakage key with val150, core150 or heldout300.
        curated = {t["task_id"] for t in json.loads((ROOT / "data" / f"eval-{args.from_set}.json").read_text())["tasks"]}
        candidates = [task for task in candidates if task.task_id in curated]
    chosen = select_tiered(candidates, quotas, args.seed)
    design = {
        "source_splits": ["train"],
        **({"drawn_from": args.from_set} if args.from_set else {}),
        "variant": "standard",
        "quotas": quotas,
        "tier": "the release's difficulty.tier",
        "strata": "source split x shortest-route depth, visited round-robin from shallow to deep",
        "within_stratum": "sha256(seed:task_id); a first-reaction class the tier has not used is preferred",
        "purpose": "difficulty labels and warm-start transcripts for RL, not an evaluation set",
    }
    record = make_evalset(args.name, chosen, seed=args.seed, design=design)
    SWEEPS.mkdir(parents=True, exist_ok=True)
    (SWEEPS / f"{args.name}.json").write_text(json.dumps(record, indent=2) + "\n")
    # Alternate tiers so each shard holds the same mix and its first episodes already show it.
    by_tier = [[t["task_id"] for t in record["tasks"] if t["tier"] == tier] for tier in quotas]
    order = [task_id for row in itertools.zip_longest(*by_tier) for task_id in row if task_id]
    # The whole sample in draw order, one id a line: a --task-pool for training. A smaller quota
    # with the same seed draws a prefix of this order, so samples of one seed nest.
    (SWEEPS / f"{args.name}.txt").write_text("\n".join(t["task_id"] for t in record["tasks"]) + "\n")
    for shard in range(args.shards):
        (SWEEPS / f"{args.name}-{shard}.txt").write_text("\n".join(order[shard::args.shards]) + "\n")
    tasks = record["tasks"]
    print(json.dumps({"tiers": Counter(t["tier"] for t in tasks), "shards": args.shards}, indent=1))
    print("shortest-route depth by tier:")
    for tier in quotas:
        print(f"  {tier}: {dict(sorted(Counter(t['min_depth'] for t in tasks if t['tier'] == tier).items()))}")


# %% 2. Run it on HF Jobs: one evaluation job per shard (the commands in the docstring above).
#
# --attempts 8 at --temperature 0.8 matches a GRPO group: 8 rollouts per task at the
# trainer's temperature. 64 episodes in flight keep one H200 busy; a rerun with the same
# --name resumes where a job stopped.


# %% 3. Report: pass@k by tier, a label per task, and the solved transcripts.
#
# A submission sent as a broken JSON string scores 0 as run. With --server, each run is also
# re-graded as if the harness had repaired it (eval/regrade.py), and the per-task labels use
# that view, since the next RL run grades with repair on. Transcripts for the warm start come
# only from episodes solved as run, so they never teach a malformed submission.
def outcome(episode):
    if episode["valid"]:
        return "solved"
    first = (episode.get("hard_failures") or [""])[0]
    if episode.get("auto_emitted") or "no route trees" in first:
        return "no submission"
    if first.startswith("invalid JSON"):
        return "invalid JSON"
    if "starts at the target" in first:
        return "off target"
    if "in_stock" in first:
        return "false stock claim"
    if first.startswith("0 valid route"):
        return "no valid route"
    return "other"


OUTCOMES = ("solved", "no valid route", "false stock claim", "off target", "invalid JSON", "no submission", "other")


def report(args):
    sys.path.insert(0, str(ROOT / "eval"))
    from regrade import regrade
    from run_eval import pass_at_k

    sample = {t["task_id"]: t for t in json.loads((SWEEPS / f"{args.name}.json").read_text())["tasks"]}
    runs = sorted(path.parent for path in args.runs.glob("**/identity.json"))
    if not runs:
        raise SystemExit(f"no evaluation runs under {args.runs}")
    identity = json.loads((runs[0] / "identity.json").read_text())
    episodes, repaired = [], {}
    for run in runs:
        episodes += [json.loads(p.read_text()) for p in sorted((run / "episodes").glob("*.json"))]
        result = regrade(run, args.server) if args.server else None
        for row in (result or {}).get("episodes", []):
            repaired[(row["task_id"], row["attempt"])] = row["valid"]
    episodes = [e for e in episodes if e.get("graded")]
    for e in episodes:
        e["valid_repaired"] = repaired.get((e["task_id"], e["attempt"]), e["valid"])
        e["outcome"] = outcome(e)

    by_task = defaultdict(list)
    for e in episodes:
        by_task[e["task_id"]].append(e)
    # A task still running has fewer attempts, and pass@k needs all k: report finished tasks only.
    pending = sum(len(runs_of_task) < identity["attempts"] for runs_of_task in by_task.values())
    by_task = {t: r for t, r in by_task.items() if len(r) >= identity["attempts"]}
    episodes = [e for e in episodes if e["task_id"] in by_task]
    labels = []
    for task_id, runs_of_task in sorted(by_task.items()):
        n, task = len(runs_of_task), sample.get(task_id, {})
        solved = sum(e["valid"] for e in runs_of_task)
        solved_repaired = sum(e["valid_repaired"] for e in runs_of_task)
        labels.append(
            {
                "task_id": task_id,
                "tier": task.get("tier"),
                "min_depth": task.get("min_depth"),
                "max_depth": task.get("max_depth"),
                "attempts": n,
                "solved": solved,
                "solved_repaired": solved_repaired,
                "label": "never" if solved_repaired == 0 else "always" if solved_repaired == n else "learnable",
                "mean_reward": round(sum(e["reward"] for e in runs_of_task) / n, 4),
                "outcomes": dict(Counter(e["outcome"] for e in runs_of_task)),
            }
        )

    output = args.output or args.runs / "report"
    output.mkdir(parents=True, exist_ok=True)
    (output / "labels.jsonl").write_text("".join(json.dumps(row) + "\n" for row in labels))
    learnable = [row["task_id"] for row in labels if row["label"] == "learnable"]
    (output / "learnable.txt").write_text("\n".join(learnable) + "\n")

    # Warm-start transcripts: the board's own messages, system prompt first.
    from retroenv_openenv.agent import SYSTEM_PROMPT

    system = SYSTEM_PROMPT.format(max_turns=identity["max_turns"])
    kept = 0
    with (output / "sft.jsonl").open("w") as sft:
        for task_id, runs_of_task in sorted(by_task.items()):
            solved = sorted((e for e in runs_of_task if e["valid"]), key=lambda e: -e["reward"])
            for e in solved[: args.sft_per_task]:
                messages = [{"role": "system", "content": system}, {"role": "user", "content": e["prompt"]}]
                record = {
                    "task_id": task_id,
                    "attempt": e["attempt"],
                    "tier": sample.get(task_id, {}).get("tier"),
                    "reward": e["reward"],
                    "messages": messages + e["transcript"],
                }
                sft.write(json.dumps(record) + "\n")
                kept += 1
    if args.server:
        from retroenv_openenv.client import RetroEnvClient

        with RetroEnvClient(args.server) as env:
            env.reset("train", index=episodes[0]["index"])
            (output / "tools.json").write_text(json.dumps(env.openai_tools(), indent=1) + "\n")

    def table(key, values):
        lines = [
            f"| {key} | Tasks | Pass@1 | Pass@{identity['attempts']} | Pass@{identity['attempts']} repaired "
            "| Never solved | Learnable | Always solved |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for value in [*values, "all"]:
            rows = [r for r in labels if value == "all" or r[key] == value]
            if not rows:
                continue

            def mean_pass(k, field):
                return sum(pass_at_k(r["attempts"], r[field], k) for r in rows) / len(rows)

            counts = Counter(r["label"] for r in rows)
            lines.append(
                f"| {value} | {len(rows)} | {mean_pass(1, 'solved'):.3f} | "
                f"{mean_pass(identity['attempts'], 'solved'):.3f} | "
                f"{mean_pass(identity['attempts'], 'solved_repaired'):.3f} | "
                f"{counts['never']} | {counts['learnable']} | {counts['always']} |"
            )
        return "\n".join(lines)

    tiers = [tier for tier in QUOTAS if any(t["tier"] == tier for t in sample.values())]
    depths = sorted({r["min_depth"] for r in labels if r["min_depth"] is not None})
    mix = ["| Tier | Episodes | " + " | ".join(OUTCOMES) + " |", "|---|---|" + "---|" * len(OUTCOMES)]
    for tier in [*tiers, "all"]:
        rows = [e for e in episodes if tier == "all" or sample.get(e["task_id"], {}).get("tier") == tier]
        if not rows:
            continue
        counts = Counter(e["outcome"] for e in rows)
        mix.append(f"| {tier} | {len(rows)} | " + " | ".join(f"{counts[o] / len(rows):.0%}" for o in OUTCOMES) + " |")
    sampling = identity["sampling"]
    view = (
        "with broken submissions re-graded as repaired"
        if args.server
        else "as run (no --server to re-grade broken submissions)"
    )
    text = "\n\n".join(
        [
            f"# {args.name}: {identity['label']}",
            f"{len(labels)} finished tasks{f' ({pending} still running, left out)' if pending else ''}, "
            f"{len(episodes)} graded episodes; {identity['attempts']} attempts per task at "
            f"temperature {sampling.get('temperature')}, thinking {sampling.get('thinking')}, "
            f"{identity['max_turns']} turns, repair {'on' if identity['repair'] else 'off'}. "
            f"Labels count solves {view}. Learnable: solved in some attempts but not all.",
            "## By tier\n\n" + table("tier", tiers),
            "## By shortest-route depth\n\n" + table("min_depth", depths),
            "## What each episode ended in\n\n" + "\n".join(mix),
            f"{len(learnable)} learnable tasks in learnable.txt; {kept} solved transcripts in sft.jsonl "
            f"(at most {args.sft_per_task} per task).",
        ]
    )
    (output / "REPORT.md").write_text(text + "\n")
    print(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    choose = commands.add_parser("select", help="Write the sample and its shard files to train/sweeps/")
    choose.add_argument("--release", type=Path, default=ROOT / "data" / "release" / "RetroEnv-RL")
    choose.add_argument("--name", default=NAME, help="sample name: train/sweeps/<name>.json and <name>-<shard>.txt")
    choose.add_argument("--seed", default=SEED, help="draw seed (a new seed draws a new sample)")
    choose.add_argument(
        "--quotas", help=f"tasks per tier, e.g. easy=500,medium=500,hard=0 (default {QUOTAS})"
    )
    choose.add_argument("--shards", type=int, default=SHARDS, help="evaluation jobs to split the sample into")
    choose.add_argument("--from-set", help="draw only tasks in data/eval-<set>.json (e.g. train12k)")
    summarize = commands.add_parser("report", help="Label tasks from the finished evaluation runs")
    summarize.add_argument("runs", type=Path, help="A directory holding the shards' evaluation outputs")
    summarize.add_argument("--server", help="A RetroEnv server, to re-grade broken submissions as repaired")
    summarize.add_argument("--name", default=NAME, help="the sample the runs evaluated (train/sweeps/<name>.json)")
    summarize.add_argument("--output", type=Path, help="Default: <runs>/report")
    summarize.add_argument("--sft-per-task", type=int, default=2, help="Solved transcripts kept per task")
    args = parser.parse_args()
    select(args) if args.command == "select" else report(args)


if __name__ == "__main__":
    main()
