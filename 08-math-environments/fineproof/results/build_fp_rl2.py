"""Rebuild results/fp-rl2-2b.json from the run's public bucket: every held-out eval, the gpt-oss-120b regrades and each
training update.

    python results/build_fp_rl2.py            # needs the `hf` CLI; reads hf://buckets/AdithyaSK/qed-nano-rl-runs

Per eval point it reports the eval's own mean (a proof cut off at the output cap keeps the judge's partial credit,
as `evaluate.py` scores it) and `mean_reward_cutoff_zero`, the same attempts scored as training scores them (a
cut-off proof gets 0). Changes are paired by problem against step 0: the mean and standard error of the per-problem
difference in mean reward over its 4 attempts.
"""

import ast
import json
import math
import statistics as st
import subprocess
import sys
import tempfile
from pathlib import Path

BUCKET = "hf://buckets/AdithyaSK/qed-nano-rl-runs"
RUN = "fp-rl2-2b"
STEPS = [0, 20, 40, 60, 80, 100, 150, 200, 250, 300, 350, 400, 450, 500]
# Step 100's own folder also holds the trials of an eval launched twice by mistake when the run resumed; "-original"
# is the first eval's untouched copy.
EVAL_DIRS = {0: "fp-bench-2b-32k", 100: "fp-rl2-2b-eval-100-original"}
CACHE = Path(tempfile.gettempdir()) / "fineproof-results-cache"


def fetch(folder: str, name: str, refresh: bool = False) -> Path | None:
    path = CACHE / folder / name
    if refresh or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["hf", "buckets", "cp", f"{BUCKET}/{folder}/{name}", str(path)], capture_output=True)
    return path if path.exists() and path.stat().st_size else None


def per_problem(folder: str, zero_cutoff: bool = False) -> dict[str, float]:
    by: dict[str, list[float]] = {}
    for line in fetch(folder, "rollouts.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("reward") is None:
            continue
        reward = 0.0 if zero_cutoff and r.get("truncated") else float(r["reward"])
        by.setdefault(r["task"], []).append(reward)
    return {task: st.fmean(v) for task, v in by.items()}


def regrade(folder: str) -> dict[str, float] | None:
    path = fetch(folder, "regrade-gpt-oss-120b-high.jsonl")
    if not path:
        return None
    by: dict[str, list[float]] = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r.get("new_score") is not None:
            by.setdefault(r["task"], []).append(r["new_score"] / 7)
    return {task: st.fmean(v) for task, v in by.items()}


def paired(base: dict[str, float], new: dict[str, float]) -> dict:
    tasks = sorted(set(base) & set(new))
    d = [new[t] - base[t] for t in tasks]
    return {"mean": round(st.fmean(d), 6), "se": round(st.stdev(d) / math.sqrt(len(d)), 6),
            "problems_up": sum(x > 0 for x in d), "problems_down": sum(x < 0 for x in d), "problems": len(tasks)}


def number(value, digits: int):
    value = float(value)
    return None if math.isnan(value) else round(value, digits)


def train_curve() -> tuple[dict, float]:
    rows = []
    for line in fetch(RUN, "train.log", refresh=True).read_text(errors="ignore").splitlines():
        if "{'loss'" in line:
            try:
                rows.append(ast.literal_eval(line[line.index("{"):line.rindex("}") + 1]))
            except (ValueError, SyntaxError):
                pass
    col = lambda key, digits: [number(r.get(key, "nan"), digits) for r in rows]  # noqa: E731
    cost = sum(float(r.get("judge/cost_usd_sum", 0) or 0) for r in rows)
    return {"update": list(range(1, len(rows) + 1)), "reward": col("reward", 4),
            "truncated_frac": col("rollouts/truncated_frac", 4), "mean_completion_tokens": col("completions/mean_length", 0),
            "kl": col("kl", 6), "seconds": col("perf/step_s", 0)}, round(cost, 2)


def main() -> None:
    folders = {s: EVAL_DIRS.get(s, f"{RUN}-eval-{s}") for s in STEPS}
    base, base_zero, base_rg = per_problem(folders[0]), per_problem(folders[0], True), regrade(folders[0])
    points, regrades = [], []
    for step, folder in folders.items():
        s = json.loads(fetch(folder, "summary.json").read_text())
        point = {"step": step, "eval_run": folder, "mean_reward": round(s["mean_reward"], 6),
                 "mean_reward_se": round(s["mean_reward_se"], 6),
                 "mean_reward_cutoff_zero": round(st.fmean(per_problem(folder, True).values()), 6),
                 "full_marks": round(s["full_marks"], 6), "any_credit": round(s["any_credit"], 6),
                 "pass_at_4_full_marks": round(s["pass_at_k_full_marks"], 6), "best_of_4": round(s["best_of_k_mean"], 6),
                 "graded": s["graded"], "expected": s["expected"]}
        if step:
            point["vs_step0"] = paired(base, per_problem(folder))
            point["vs_step0_cutoff_zero"] = paired(base_zero, per_problem(folder, True))
        points.append(point)
        rg = regrade(folder)
        if rg is not None:
            entry = {"step": step, "mean_reward": round(st.fmean(rg.values()), 6)}
            if step and base_rg:
                entry["vs_step0"] = paired(base_rg, rg)
            regrades.append(entry)
    curve, judge_cost = train_curve()
    out = {
        "run": RUN, "model": "Qwen/Qwen3.5-2B", "mode": "non-thinking", "harness": "FineProof, one call (--rounds 0)",
        "recipe": {"updates": len(curve["update"]), "rollouts_per_update": 128, "group_size": 16,
                   "learning_rate": 1e-6, "warmup_updates": 10, "max_completion_tokens": 32768,
                   "context_tokens": 131072, "temperature": 0.8, "advantage": "reward minus group mean",
                   "loss_normaliser": "32768 tokens per rollout", "truncated_reward": 0.0,
                   "reward": "gpt-oss-20b rubric score / 7, mean of 2 samples",
                   "train_problems": 1024, "train_problem_list": "training/benchmarks/fp-rl2-2b-train-tasks.txt"},
        "eval": {"set": "64 validated FineProofs-RL-test problems x 4 attempts, temperature 0.8",
                 "judge": "gpt-oss-20b, 2 samples per proof",
                 "cutoff_scoring": "mean_reward keeps the judge's partial credit for a proof cut off at 32,768 tokens; "
                                   "mean_reward_cutoff_zero scores it 0, as training does",
                 "points": points},
        "regrade_gpt_oss_120b_high": regrades,
        "train_curve": curve,
        "judge_cost_usd_training": judge_cost,
    }
    path = Path(__file__).with_name(f"{RUN}.json")
    path.write_text(json.dumps(out, indent=1, allow_nan=False) + "\n")
    print(f"wrote {path}: {len(points)} evals, {len(regrades)} regrades, {len(curve['update'])} updates", file=sys.stderr)


if __name__ == "__main__":
    main()
