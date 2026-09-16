"""One entry point for the pinned data-agent training and evaluation recipes.

Start with `python reproduce.py prepare --recipe harbor-opencode --env-file .env`.
See reproduce.md for credentials, hardware and checkpoint evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

PROJECT = Path(__file__).resolve().parent
RECIPES = {"harbor-multi": "blackbox", "harbor-opencode": "blackbox",
           "native-opencode": "opencode", "seta": "whitebox"}


def run(command, dry_run=False):
    command = list(map(str, command))
    print(shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["prepare", "upload", "spaces", "smoke", "eval", "train", "status"])
    p.add_argument("--recipe", choices=RECIPES, default="harbor-opencode")
    p.add_argument("--platform", choices=["hub", "local"], default="hub")
    p.add_argument("--out", type=Path, help="Isolated run directory (default: temp/reproduction/<recipe>)")
    p.add_argument("--env-file", type=Path, default=PROJECT / ".env")
    p.add_argument("--namespace", default="HuggingEnvs", help="HF namespace where you can create resources")
    p.add_argument("--run-id", help="Unique artifact identity; set when preparing a new experiment")
    p.add_argument("--flavor", help="HF hardware: h200x2/a100x4 for training, a100-large for eval")
    p.add_argument("--partition", default="hopper-prod", help="Local Slurm GPU partition")
    p.add_argument("--cpu-partition", default="hopper-cpu")
    p.add_argument("--timeout", help="HF duration, e.g. 2h or 24h")
    p.add_argument("--concurrency", type=int, help="Eval concurrency (Hub default 35; local default 50)")
    p.add_argument("--space-bundle-sha", help="Pin an existing Space for a training smoke without redeploying it")
    p.add_argument("--baseline-job")
    p.add_argument("--comparison-baseline-job")
    p.add_argument("--smoke-job")
    p.add_argument("--smoke-run", type=Path, help="Local smoke output directory")
    p.add_argument("--baseline-score", type=Path)
    p.add_argument("--comparison-baseline-score", type=Path)
    p.add_argument("--checkpoint-eval-proof", type=Path)
    p.add_argument("--qualify-checkpoint-eval", action="store_true")
    p.add_argument("--train-venv", type=Path, help="Optional existing local training venv; otherwise create locked venvs")
    p.add_argument("--env-venv", type=Path)
    p.add_argument("--submit", action="store_true", help="Submit a prepared local Slurm script")
    p.add_argument("--dry-run", action="store_true", help="Print commands without allocating resources")
    a = p.parse_args()
    out = (a.out or PROJECT / "temp/reproduction" / a.recipe).resolve()
    arm = RECIPES[a.recipe]
    if a.action == "prepare":
        if a.concurrency is not None and a.concurrency < 1:
            p.error("Concurrency must be positive")
        if (out / "config.json").exists():
            p.error("This run directory already exists. Use a new --out to preserve its configuration and artifacts.")
        config = json.loads((PROJECT / "hf/configs/deployment.json").read_text())
        config["namespace"] = a.namespace
        config["run_id"] = a.run_id or f"reproduction-{a.recipe}"
        config["recipe"] = a.recipe
        if a.recipe == "harbor-multi":
            config["arms"][arm]["training_harnesses"] = ["opencode", "claude-code", "codex", "mini-swe-agent"]
        if a.namespace != "HuggingEnvs":
            # All resources become owned copies. The immutable task seed still
            # requires access to the original dataset repository.
            def relocate(value):
                if isinstance(value, str) and value.startswith("HuggingEnvs/"):
                    return a.namespace + value[len("HuggingEnvs"):]
                if isinstance(value, dict): return {k: relocate(v) for k, v in value.items()}
                if isinstance(value, list): return [relocate(v) for v in value]
                return value
            config["resources"] = relocate(config["resources"])
        if a.concurrency:
            config["evaluation"]["concurrency_per_job"] = a.concurrency
            config["evaluation"]["concurrency_per_arm"] = {k: a.concurrency for k in RECIPES.values()}
            config["evaluation"]["opencode_backend_concurrency"] = {k: a.concurrency for k in ("daytona", "hf")}
        if not a.dry_run:
            out.mkdir(parents=True, exist_ok=True)
            (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        run([sys.executable, PROJECT / "hf/build.py", "--out", out / "bundle",
             "--config", out / "config.json", "--env-file", a.env_file], a.dry_run)
        return
    if not (out / "config.json").is_file() and not a.dry_run:
        p.error("Run prepare first, using the same --recipe and --out")
    if a.platform == "local" and a.action == "status":
        for path in (out / "baseline/launch.json", out / "smoke/launch.json", out / "long/plan.json"):
            if not path.exists():
                continue
            record = json.loads(path.read_text())
            jobs = [str(record[k]) for k in ("slurm_job", "training_job", "controller_job") if record.get(k)]
            print(json.dumps({"record": str(path), "jobs": jobs}))
            if jobs:
                if not all(job.isdigit() for job in jobs):
                    p.error("Stored Slurm job IDs must be numeric")
                run(["sacct", "-X", "-j", ",".join(jobs), "--format=JobID,State,Elapsed,ExitCode"], a.dry_run)
        return
    if a.platform == "local" and a.action in {"eval", "smoke", "train"}:
        if a.action == "train":
            required = (a.smoke_run, a.baseline_score)
            if not all(required): p.error("Local training needs --smoke-run and --baseline-score")
            cmd = [sys.executable, PROJECT / "hf/local_long.py", "--arm", arm,
                   "--smoke-run", a.smoke_run, "--baseline-score", a.baseline_score,
                   "--env-file", a.env_file.resolve(), "--out", out / "long",
                   "--coordination-dir", out.parent / "eval-admission", "--partition", a.partition,
                   "--cpu-partition", a.cpu_partition]
            if a.qualify_checkpoint_eval: cmd += ["--qualify-checkpoint-eval"]
            if a.checkpoint_eval_proof: cmd += ["--checkpoint-eval-proof", a.checkpoint_eval_proof]
            if a.comparison_baseline_score: cmd += ["--comparison-baseline-score", a.comparison_baseline_score]
        else:
            cmd = [sys.executable, PROJECT / "hf/cluster.py", "--bundle", out / "bundle",
                   "--out", out / ("baseline" if a.action == "eval" else "smoke"),
                   "--env-file", a.env_file.resolve(), "--arm", arm,
                   "--phase", "baseline" if a.action == "eval" else "smoke", "--partition", a.partition]
            for key in ("train_venv", "env_venv"):
                if getattr(a, key): cmd += ["--" + key.replace("_", "-"), getattr(a, key).resolve()]
        if a.submit: cmd += ["--submit"]
    else:
        action = "job" if a.action in {"smoke", "eval", "train"} else a.action
        cmd = [sys.executable, PROJECT / "hf/deploy.py", action, "--config", out / "config.json",
               "--out", out, "--env-file", a.env_file.resolve(), "--arm", arm]
        if action == "spaces": cmd += ["--only", arm]
        if action == "job":
            cmd += ["--role", "eval" if a.action == "eval" else "train", "--phase",
                    {"smoke": "smoke", "eval": "baseline", "train": "long"}[a.action]]
            cmd += ["--timeout", a.timeout or ("24h" if a.action == "train" else "4h" if a.action == "eval" else "2h")]
            for key in ("flavor", "space_bundle_sha", "baseline_job", "comparison_baseline_job", "smoke_job"):
                if getattr(a, key): cmd += ["--" + key.replace("_", "-"), getattr(a, key)]
    run(cmd, a.dry_run)


if __name__ == "__main__":
    main()
