"""Submit the same job-local recipe to HF Jobs or Slurm; dry-run by default."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from recipe import MODES, write_json


def slurm_time(value):
    match = re.fullmatch(r"([1-9][0-9]*)([hms])", value)
    if not match:
        raise ValueError("Timeout must be a positive duration such as 12h or 40m")
    seconds = int(match[1]) * {"h": 3600, "m": 60, "s": 1}[match[2]]
    return f"{seconds // 3600:02}:{seconds // 60 % 60:02}:{seconds % 60:02}"


def stage(destination):
    destination = Path(destination)
    allowed = {"run.py", "recipe.py", "prepare.py", "requirements.txt", "requirements.lock", "README.md", "REPRODUCE.md"}
    for folder in ("configs", "data", "train", "eval", "runtime"):
        allowed.update(str(p.relative_to(ROOT)) for p in (ROOT / folder).rglob("*")
                       if p.is_file() and p.suffix in {".py", ".sh", ".json"} and "__pycache__" not in p.parts)
    for name in sorted(allowed):
        if (ROOT / name).is_file():
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("backend", choices=["hf", "slurm"])
    p.add_argument("action", choices=["train", "eval", "smoke", "pilot", "watch"])
    p.add_argument("--model", choices=["lfm", "qwen"], default="lfm")
    p.add_argument("--config", type=Path)
    p.add_argument("--mode", choices=MODES, default="multi-harness")
    p.add_argument("--run-name", required=True)
    p.add_argument("--namespace", default="FineEnvs")
    p.add_argument("--bucket", help="Existing HF bucket, e.g. FineEnvs/smoldataenv-runs")
    p.add_argument("--image", default="huggingface/trl@sha256:8433cde7eaf3b289f3daffdaefcab6bb71499ca1ff59dc5d05abba7a71603a58")
    p.add_argument("--flavor", default="h200x2")
    p.add_argument("--timeout", default="24h")
    p.add_argument("--partition")
    p.add_argument("--data", type=Path, default=ROOT / "prepared")
    p.add_argument("--output-root", type=Path, default=ROOT / "runs")
    p.add_argument("--checkpoint")
    p.add_argument("--resume")
    p.add_argument("--concurrency", type=int, default=35)
    p.add_argument("--limit", type=int)
    p.add_argument("--smoke-eval", action="store_true")
    p.add_argument("--preflight-smoke", action="store_true")
    p.add_argument("--max-active-evals", type=int, default=1)
    p.add_argument("--submit", action="store_true")
    args = p.parse_args()
    if args.smoke_eval and args.action != "smoke":
        p.error("--smoke-eval requires the smoke action")
    if args.preflight_smoke and args.action != "pilot":
        p.error("--preflight-smoke requires the pilot action")
    if Path(args.run_name).name != args.run_name or args.run_name in {".", ".."}:
        p.error("run-name must be a single directory name")
    output = str(Path("/outputs" if args.backend == "hf" else args.output_root.resolve()) / args.run_name)
    flags = [args.action, "--model", args.model, "--mode", args.mode, "--run-name", args.run_name,
             "--output", output, "--concurrency", str(args.concurrency)]
    if args.smoke_eval:
        flags.append("--smoke-eval")
    if args.preflight_smoke:
        flags.append("--preflight-smoke")
    if args.config:
        path = args.config
        if args.backend == "hf" and not str(path).startswith("/outputs/"):
            try:
                relative = path.resolve().relative_to(ROOT)
            except ValueError:
                p.error("Put a custom HF config inside this recipe's configs directory")
            if relative.parts[0] != "configs":
                p.error("Custom configs must be in configs/")
            path = Path("/workspace/recipe") / relative
        flags += ["--config", str(path)]
    for name in ("checkpoint", "resume", "limit"):
        value = getattr(args, name)
        if value is not None:
            flags += ["--" + name, str(value)]
    if args.backend == "hf":
        if not args.bucket:
            p.error("HF Jobs require --bucket for durable checkpoints and logs")
        if args.action == "watch":
            flags = ["watch", "--root", "/outputs", "--backend", "hf", "--bucket", args.bucket,
                     "--namespace", args.namespace, "--flavor", args.flavor,
                     "--concurrency", str(args.concurrency), "--max-active", str(args.max_active_evals), "--submit"]
        specification = {"image": args.image, "command": ["bash", "/recipe-source/runtime/job.sh", *flags],
            "flavor": "cpu-basic" if args.action == "watch" else args.flavor, "timeout": args.timeout, "namespace": args.namespace,
            "name": args.run_name, "labels": {"recipe": "smoldataenv-multi-harness", "role": args.action},
            "env": {"PYTHONUNBUFFERED": "1", **{k: os.environ[k] for k in
                    ("DAYTONA_API_URL", "DAYTONA_TARGET") if os.environ.get(k)}}}
        print(json.dumps({**specification, "output_bucket": args.bucket, "secret_names": ["HF_TOKEN", "DAYTONA_API_KEY"]}, indent=2))
        if args.submit:
            from huggingface_hub import get_token, run_job, sync_job_volume, Volume
            token = get_token()
            if not token or not os.environ.get("DAYTONA_API_KEY"):
                p.error("HF login and DAYTONA_API_KEY are required")
            with tempfile.TemporaryDirectory(prefix="smoldataenv-recipe-") as temp:
                stage(temp)
                source = sync_job_volume(temp, "/recipe-source", namespace=args.namespace)
            volume = Volume(type="bucket", source=args.bucket, mount_path="/outputs", read_only=False)
            job = run_job(**specification, volumes=[source, volume],
                          secrets={"HF_TOKEN": token, "DAYTONA_API_KEY": os.environ["DAYTONA_API_KEY"]})
            record = {"id": job.id, "url": job.url, "run_name": args.run_name, "namespace": args.namespace}
            write_json(ROOT / "runs/submissions" / f"{args.run_name}.json", record)
            print(json.dumps(record))
    else:
        if not args.partition:
            p.error("Choose an available Slurm partition explicitly")
        args.output_root.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(ROOT / "run.py"), *flags, "--data", str(args.data.resolve())]
        if args.action == "watch":
            command = [sys.executable, str(ROOT / "eval/watch.py"), "--root", str(args.output_root.resolve()),
                       "--backend", "slurm", "--partition", args.partition, "--data", str(args.data.resolve()),
                       "--concurrency", str(args.concurrency), "--max-active", str(args.max_active_evals), "--submit"]
        submit = ["sbatch", "--parsable", "--partition", args.partition, "--nodes=1", "--gpus=2",
                  "--cpus-per-task=16", "--mem=192G", "--time=" + slurm_time(args.timeout), "--job-name", args.run_name,
                  "--output", str(args.output_root / (args.run_name + "-%j.log")), "--wrap", shlex.join(command)]
        if args.action == "watch":
            submit.remove("--gpus=2")
            submit[submit.index("--cpus-per-task=16")] = "--cpus-per-task=1"
            submit[submit.index("--mem=192G")] = "--mem=4G"
        print(shlex.join(submit), flush=True)
        if args.submit:
            subprocess.run(submit, check=True)


if __name__ == "__main__":
    main()
