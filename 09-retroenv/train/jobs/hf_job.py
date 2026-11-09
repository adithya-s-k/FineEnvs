"""Submit RetroEnv RL training, evaluation or a smoke test to HF Jobs.

    python train/jobs/hf_job.py smoke --mode sync --name sync-smoke --bucket <owner>/<bucket>
    python train/jobs/hf_job.py train --mode async --name async-100 --bucket <owner>/<bucket> --steps 100
    python train/jobs/hf_job.py eval --name async-100-eval --bucket <owner>/<bucket> \\
        --checkpoint /outputs/async-100/checkpoint-100
    python train/jobs/hf_job.py watch --run async-100 --job <training job id> --bucket <owner>/<bucket>

`watch` starts a CPU job that evaluates each checkpoint of a training run as it lands (see
eval_watch.py). Training is h200x2 by default; h200x4 and h200x8 give half the GPUs to vLLM
replicas and the other half to an FSDP2 trainer. Evaluation runs on one H200.

Without --submit it prints the job plan. The job mounts three volumes: this project's source
(an allowlist, so no credentials or local data leave the machine), the output bucket at
/outputs (checkpoints, logs and boards land there as they are written), and the public RetroEnv
bucket at /data, from which the server's serving snapshot is prepared and checked.
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_BUCKET = "FineEnvs/retroenv-bucket"
# The image 05-multi-harness-rl runs: CUDA, Python and uv. Everything else is installed by install.sh.
IMAGE = "huggingface/trl@sha256:8433cde7eaf3b289f3daffdaefcab6bb71499ca1ff59dc5d05abba7a71603a58"
SOURCE_SUFFIXES = {".py", ".sh", ".json", ".toml", ".txt", ".jinja", ".js", ".md", ".lock", ".cfg", ".yaml"}
FLAVORS = {
    "check": ("cpu-upgrade",),
    "watch": ("cpu-basic",),
    "eval": ("h200", "h200x2"),
    "smoke": ("h200x2", "h200x4", "h200x8"),
    "train": ("h200x2", "h200x4", "h200x8"),
}
SKIP_PARTS = {".venv", ".venv-env", ".venv-train", "prepared", "runs", "__pycache__", "static", "tests"}


def source_files():
    """The project files a job needs: packages, scripts, the lockfile and the frozen evaluation sets."""
    files = [ROOT / name for name in ("pyproject.toml", "uv.lock", "README.md")]
    files += sorted((ROOT / "data").glob("eval-*.json"))
    for folder in ("envs", "eval", "train"):
        files += [
            path
            for path in (ROOT / folder).rglob("*")
            if path.is_file()
            and path.suffix in SOURCE_SUFFIXES
            and not any(part in SKIP_PARTS or part.endswith(".egg-info") for part in path.relative_to(ROOT).parts)
        ]
    # The server serves its 3D viewer from static/; the playground is off in jobs, but the package imports it.
    files += sorted((ROOT / "envs/retro_route/openenv/retroenv_openenv/static").glob("*"))
    return files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=sorted(FLAVORS))
    parser.add_argument("--mode", choices=["sync", "async"], default="sync")
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--name", help="Job name and output directory under /outputs (watch: <run>-evals)")
    parser.add_argument("--bucket", required=True, help="An existing owner/bucket for checkpoints and logs")
    parser.add_argument("--flavor", help="Default: the first listed for the action in FLAVORS")
    parser.add_argument(
        "--timeout", help="Default: 24h for train (100 sync steps on the 27B take about 13 h), 4h otherwise"
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--checkpoint", help="For eval: /outputs/<run>/checkpoint-N; omit for the base model")
    parser.add_argument("--set", default="core30")
    parser.add_argument("--tasks", type=int)
    parser.add_argument("--task-ids", help="For eval: a task-id file in this project (train/sweeps/<name>-0.txt) or in --bucket (/outputs/<path>)")
    parser.add_argument("--attempts", type=int, default=1, help="For eval: episodes per task")
    parser.add_argument("--temperature", type=float, help="For eval: sampling temperature (default: the board's 0)")
    parser.add_argument("--harness", choices=("board", "training"), help="For eval: the agent loop (default: board)")
    parser.add_argument("--thinking", action="store_true", help="For eval and async train: thinking on")
    parser.add_argument(
        "--reasoning-effort", choices=["low", "medium", "high", "xhigh"], help="For train: with --thinking"
    )
    parser.add_argument("--context", type=int, help="For train: tokens per episode (default 65536)")
    parser.add_argument("--turn-tokens", type=int, help="For train: tokens per model turn (default 16384)")
    parser.add_argument("--mask-truncated", action="store_true", help="For train: cut-off episodes out of the loss")
    parser.add_argument("--source-from", help="With --submit: mount this job's uploaded source instead of uploading it again")
    parser.add_argument("--task-pool", help="For train: the training pool, e.g. a sweep's learnable.txt (shuffled)")
    parser.add_argument("--rows-per-episode", type=float, help="For train: training rows per episode (sizes overfit updates)")
    parser.add_argument("--init-adapter", help="For train: start from this LoRA checkpoint (a /outputs/... path)")
    parser.add_argument("--env-url", help="A running RetroEnv server, e.g. https://fineenvs-retroenv.hf.space")
    parser.add_argument(
        "--train-tasks", help="For train: a project JSON {task_id: index}, e.g. train/sweeps/overfit10-index.json"
    )
    parser.add_argument("--generations", type=int, help="For train: rollouts per task in a GRPO group")
    parser.add_argument("--reward", choices=["graded", "solved"], help="For train: see train/sync_grpo.py")
    parser.add_argument("--lr", type=float, help="For train: learning rate")
    parser.add_argument("--updates-per-round", type=int, help="For train: optimizer steps per round (overfit runs)")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--label")
    parser.add_argument("--vllm-gpus", type=int, help="For train/smoke: GPUs serving vLLM (default: half)")
    parser.add_argument("--inflight", type=int, help="For async train: episodes in flight (default: 32 per replica)")
    parser.add_argument("--run", help="For watch: the training run whose checkpoints to evaluate")
    parser.add_argument("--job", help="For watch: the training job id; the watch ends after it does")
    parser.add_argument("--space-id")
    parser.add_argument("--submit", action="store_true", help="Without this flag, print the job plan")
    args = parser.parse_args(argv)
    if args.action == "watch":
        if not args.run or not args.job:
            parser.error("watch needs --run and --job")
        args.name = args.name or f"{args.run}-evals"
    if not args.name:
        parser.error("--name is required")
    if Path(args.name).name != args.name or args.name in {".", ".."}:
        parser.error("Use a single directory name for --name")
    namespace, separator, bucket = args.bucket.partition("/")
    if not separator or not namespace or not bucket:
        parser.error("Use owner/bucket for --bucket")
    flavor = args.flavor or FLAVORS[args.action][0]
    if flavor not in FLAVORS[args.action]:
        parser.error(f"{args.action} runs on {', '.join(FLAVORS[args.action])}")

    command = ["bash", "/source/train/jobs/entrypoint.sh", args.action]
    if args.action == "watch":
        # A CPU job with no GPU stack: refresh huggingface_hub, then follow the run.
        watch = (
            "pip install -q 'huggingface_hub>=1.32,<2' && python /source/train/jobs/eval_watch.py "
            f"--run {args.run} --job {args.job} --bucket {args.bucket} --model {args.model} --set {args.set}"
        )
        command = ["bash", "-c", watch]
    elif args.action != "check":
        command += ["--model", args.model, "--output", f"/outputs/{args.name}"]
    if args.action in ("smoke", "train"):
        command += ["--mode", args.mode]
    if args.action == "train":
        if args.train_tasks and not (ROOT / args.train_tasks).is_file():
            parser.error(f"{args.train_tasks} is not a file in this project")
        command += ["--train-tasks", args.train_tasks] if args.train_tasks else []
        command += ["--generations", str(args.generations)] if args.generations else []
        command += ["--reward", args.reward] if args.reward else []
        command += ["--lr", str(args.lr)] if args.lr is not None else []
        command += ["--updates-per-round", str(args.updates_per_round)] if args.updates_per_round else []
        command += ["--thinking"] if args.thinking else []
        command += ["--reasoning-effort", args.reasoning_effort] if args.reasoning_effort else []
        command += ["--context", str(args.context)] if args.context else []
        command += ["--turn-tokens", str(args.turn_tokens)] if args.turn_tokens else []
        command += ["--mask-truncated"] if args.mask_truncated else []
        command += ["--rows-per-episode", str(args.rows_per_episode)] if args.rows_per_episode else []
        command += ["--task-pool", args.task_pool] if args.task_pool else []
        command += ["--init-adapter", args.init_adapter] if args.init_adapter else []
        command += [
            "--steps",
            str(args.steps),
            "--save-steps",
            str(args.save_steps),
            "--max-depth",
            str(args.max_depth),
        ]
    if args.action == "eval":
        command += ["--set", args.set, "--concurrency", str(args.concurrency)]
        command += ["--checkpoint", args.checkpoint] if args.checkpoint else []
        command += ["--label", args.label] if args.label else []
        if args.task_ids:
            # A list in the outputs bucket (mounted at /outputs) runs on an earlier source upload.
            if not args.task_ids.startswith("/outputs/") and not (ROOT / args.task_ids).is_file():
                parser.error(f"{args.task_ids} is not a file in this project or in the bucket (/outputs/...)")
            command += ["--task-ids", args.task_ids]  # the job runs from its copy of the project
        command += ["--attempts", str(args.attempts)] if args.attempts > 1 else []
        command += ["--temperature", str(args.temperature)] if args.temperature is not None else []
        command += ["--harness", args.harness] if args.harness else []
        command += ["--thinking"] if args.thinking else []
    if args.env_url and args.action in ("train", "eval"):
        command += ["--env-url", args.env_url]
    if args.action in ("smoke", "eval") and args.tasks:
        command += ["--tasks", str(args.tasks)]
    if args.action in ("smoke", "train"):
        command += ["--vllm-gpus", str(args.vllm_gpus)] if args.vllm_gpus else []
        command += ["--inflight", str(args.inflight)] if args.inflight else []
    if args.space_id:
        command += ["--space-id", args.space_id]
    specification = {
        "image": IMAGE,
        "command": command,
        "namespace": namespace,
        "flavor": flavor,
        "timeout": args.timeout or {"train": "24h", "watch": "30h"}.get(args.action, "4h"),
        "name": args.name,
        "env": {"PYTHONUNBUFFERED": "1", "RETROENV_MODEL": args.model},
    }
    files = source_files()
    print(
        json.dumps({**specification, "bucket": args.bucket, "data": DATA_BUCKET, "source_files": len(files)}, indent=2)
    )
    if not args.submit:
        return specification

    from huggingface_hub import Volume, get_token, run_job, sync_job_volume

    token = get_token()
    if not token:
        parser.error("Log in to the Hub first (hf auth login)")
    if args.source_from:
        # Mount the source another job uploaded (same code, launched moments ago): no new upload,
        # which a full private storage quota refuses.
        from huggingface_hub import inspect_job

        mounted = [v for v in inspect_job(job_id=args.source_from, namespace=namespace).volumes or [] if v.mount_path == "/source"]
        if not mounted:
            parser.error(f"job {args.source_from} has no /source volume")
        source_volume = mounted[0]
    else:
        with tempfile.TemporaryDirectory() as directory:
            for source in files:
                target = Path(directory) / source.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
            source_volume = sync_job_volume(directory, "/source", namespace=namespace)
    job = run_job(
        **specification,
        volumes=[
            source_volume,
            Volume(type="bucket", source=args.bucket, mount_path="/outputs", read_only=False),
            Volume(type="bucket", source=DATA_BUCKET, mount_path="/data", read_only=True),
        ],
        secrets={"HF_TOKEN": token},
    )
    print(job.url)
    return job


if __name__ == "__main__":
    main()
