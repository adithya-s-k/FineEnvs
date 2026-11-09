"""Submit a run to HF Jobs, with the FineProofEnvs source and a bucket for outputs. Adapted from 05's jobs/hf_job.py.

A run writes to the Job's local disk and jobs/entrypoint.sh copies it to <bucket>/<name>/ every 5 minutes and at the
end. The bucket is also mounted read-only at /outputs, so an eval can load a checkpoint an earlier run saved there.

    python jobs/hf_job.py smoke --name qed-smoke --bucket <owner>/<bucket>              # prints the plan
    python jobs/hf_job.py smoke --name qed-smoke --bucket <owner>/<bucket> --submit
    python jobs/hf_job.py eval  --name qed-eval-base --bucket ... --submit
    python jobs/hf_job.py train --name qed-train --bucket ... --steps 100 --submit
    python jobs/hf_job.py train --name qed-train --bucket ... --eval-checkpoints --submit   # + an eval Job per checkpoint
    python jobs/hf_job.py eval  --name qed-eval-step100 --bucket ... --checkpoint /outputs/qed-train/checkpoint-100 --step 100 --submit

Secrets sent: HF_TOKEN (model downloads), DAYTONA_API_KEY (sandboxes) and JUDGE_API_KEY (the verifier's judge).
JUDGE_PROVIDER / JUDGE_MODEL / JUDGE_BASE_URL are passed through when set.
"""

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]  # 08-math-environments/fineproof
SKIP = {".venv", ".deps", ".cache", ".local", "prepared", "runs", "__pycache__"}
SUFFIXES = {".py", ".sh", ".json", ".toml", ".txt", ".yaml", ".md", ".jinja"}


def source_files():
    for folder in ("envs", "training"):
        for path in (PROJECT / folder).rglob("*"):
            parts = path.relative_to(PROJECT).parts
            if not path.is_file() or any(p in SKIP or p.endswith(".egg-info") for p in parts):
                continue
            if path.suffix in SUFFIXES or path.name == "Dockerfile":
                yield path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["train", "eval", "smoke"])
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--name", required=True)
    parser.add_argument("--bucket", required=True, help="An existing owner/bucket for checkpoints and logs")
    parser.add_argument("--flavor", default="h200x2")
    parser.add_argument("--timeout", default="12h")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--checkpoint", help="For eval: /outputs/<run>/checkpoint-N")
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--tasks", type=int)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--prepare-args", default="", help='Passed to prepare.py, e.g. "--train-tasks 64"')
    parser.add_argument("--space-id")
    parser.add_argument("--trackio-bucket", help="Bucket the Trackio Space keeps its data in")
    parser.add_argument("--trackio-run", help="Eval: the Trackio run to log into")
    parser.add_argument("--eval-checkpoints", action="store_true",
                        help="Train: submit an eval Job for each checkpoint as it is saved (jobs/checkpoint_evals.py)")
    parser.add_argument("--eval-concurrency", type=int, default=64, help="With --eval-checkpoints: rollouts in flight")
    parser.add_argument("--rounds", type=int, default=1, help="Harness check-and-rewrite rounds (0: one call)")
    parser.add_argument("--max-tokens", type=int, default=16384, help="Output cap per model call")
    parser.add_argument("--grad-accum", type=int, default=40, help="Train: rollouts per update")
    parser.add_argument("--inflight", type=int, default=32, help="Train: rollouts in flight")
    parser.add_argument("--learning-rate", type=float, default=2e-6, help="Train: AsyncGRPO learning rate")
    parser.add_argument("--max-restarts", type=int, default=3, help="Train: resumes from the latest checkpoint")
    parser.add_argument("--resume-from", help="Train: start from another run's checkpoint, /outputs/<run>/checkpoint-N")
    parser.add_argument("--evals-run", help="Train: the Trackio run checkpoint evals log to (default <name>-evals)")
    parser.add_argument("--task-seed", type=int, help="Eval: a seeded random --tasks subset of test")
    parser.add_argument("--task-list", help="Eval: a file under training/ listing test tasks, e.g. benchmarks/test-validated-64.txt")
    parser.add_argument("--sandbox", default="daytona", choices=["daytona", "hf-sandbox"],
                        help="Where Harbor runs each task's sandboxes (hf-sandbox: HF Jobs, needs a docker_image task)")
    parser.add_argument("--submit", action="store_true", help="Without this flag, print the job plan")
    args = parser.parse_args()
    args.tasks = args.tasks or (4 if args.action == "smoke" else 128)
    args.concurrency = args.concurrency or (4 if args.action == "smoke" else 32)
    if Path(args.name).name != args.name or args.name in {".", ".."}:
        parser.error("Use a single directory name for --name")
    if args.flavor not in ("h200x2", "h200x4"):
        parser.error("Use h200x2 or h200x4: vLLM on all GPUs but one, the trainer on the last (AsyncGRPO needs "
                     "FlashAttention 3, so Hopper)")
    namespace, separator, bucket = args.bucket.partition("/")
    if not separator or not namespace or not bucket:
        parser.error("Use owner/bucket for --bucket")
    command = ["bash", "/fineproof-source/training/jobs/entrypoint.sh", args.action, "--model", args.model,
               "--output", "/workspace/outputs/" + args.name, "--steps", str(args.steps),
               "--save-steps", str(args.save_steps), "--step", str(args.step), "--tasks", str(args.tasks), "--samples", str(args.samples),
               "--concurrency", str(args.concurrency), "--sandbox", args.sandbox, "--rounds", str(args.rounds),
               "--max-tokens", str(args.max_tokens)]
    if args.action != "smoke":
        command += ["--grad-accum", str(args.grad_accum), "--inflight", str(args.inflight)]
    if args.action == "train":
        command += ["--max-restarts", str(args.max_restarts)]
        command += ["--resume-from", args.resume_from] if args.resume_from else []
        command += ["--evals-run", args.evals_run] if args.evals_run else []
    if args.action in ("train", "smoke"):
        command += ["--learning-rate", str(args.learning_rate)]
    if args.task_seed is not None:
        command += ["--task-seed", str(args.task_seed)]
    if args.task_list:
        command += ["--task-list", args.task_list]
    if args.checkpoint:
        command += ["--checkpoint", args.checkpoint]
    if args.space_id:
        command += ["--space-id", args.space_id]
    if args.trackio_bucket:
        command += ["--trackio-bucket", args.trackio_bucket]
    if args.trackio_run:
        command += ["--trackio-run", args.trackio_run]
    if args.eval_checkpoints:
        if args.action != "train":
            parser.error("--eval-checkpoints is for train")
        command += ["--eval-checkpoints", "--eval-concurrency", str(args.eval_concurrency)]
    prepare_args = args.prepare_args + (" --sandbox hf-sandbox" if args.sandbox == "hf-sandbox" else "")
    env = {"PYTHONUNBUFFERED": "1", "FINEPROOF_PREPARE_ARGS": prepare_args.strip(), "FINEPROOF_OUTPUT_BUCKET": args.bucket}
    for name in ("DAYTONA_API_URL", "JUDGE_PROVIDER", "JUDGE_MODEL", "JUDGE_BASE_URL", "JUDGE_SAMPLES"):
        if os.environ.get(name):
            env[name] = os.environ[name]
    specification = {
        "image": "huggingface/trl@sha256:8433cde7eaf3b289f3daffdaefcab6bb71499ca1ff59dc5d05abba7a71603a58",
        "command": command,
        "namespace": namespace,
        "flavor": args.flavor,
        "timeout": args.timeout,
        "name": args.name,
        "env": env,
    }
    files = sorted(source_files())
    print(json.dumps({**specification, "bucket": args.bucket, "secrets": ["HF_TOKEN", "DAYTONA_API_KEY", "JUDGE_API_KEY"],
                      "source_files": len(files)}, indent=2))
    if not args.submit:
        return
    from huggingface_hub import Volume, get_token, run_job, sync_job_volume

    token = os.environ.get("HF_TOKEN") or get_token()
    if not token or not os.environ.get("JUDGE_API_KEY"):
        parser.error("Set HF_TOKEN (or log in to HF) and JUDGE_API_KEY")
    if args.sandbox == "daytona" and not os.environ.get("DAYTONA_API_KEY"):
        parser.error("Set DAYTONA_API_KEY, or pass --sandbox hf-sandbox")
    with tempfile.TemporaryDirectory() as directory:
        for source in files:
            target = Path(directory) / source.relative_to(PROJECT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        source_volume = sync_job_volume(directory, "/fineproof-source", namespace=namespace)
    job = run_job(
        **specification,
        volumes=[source_volume, Volume(type="bucket", source=args.bucket, mount_path="/outputs", read_only=True)],
        secrets={"HF_TOKEN": token, "JUDGE_API_KEY": os.environ["JUDGE_API_KEY"],
                 **({"DAYTONA_API_KEY": os.environ["DAYTONA_API_KEY"]} if args.sandbox == "daytona" else {})},
    )
    print(job.url)


if __name__ == "__main__":
    main()
