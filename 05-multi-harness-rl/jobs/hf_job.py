"""Submit a tutorial to HF Jobs, with source and persistent output volumes."""

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["train", "eval", "smoke"])
    parser.add_argument(
        "--mode",
        choices=["whitebox", "opencode", "multi_harness"],
        default="multi_harness",
    )
    parser.add_argument("--model", default="LiquidAI/LFM2.5-2.6B")
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--bucket",
        required=True,
        help="An existing owner/bucket for checkpoints and logs",
    )
    parser.add_argument("--flavor", default="h200x2")
    parser.add_argument("--timeout", default="12h")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--checkpoint", help="For eval: /outputs/<run>/checkpoint-100")
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--tasks", type=int)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--space-id")
    parser.add_argument(
        "--submit", action="store_true", help="Without this flag, print the job plan"
    )
    args = parser.parse_args()
    args.tasks = args.tasks or (2 if args.action == "smoke" else 250)
    args.concurrency = args.concurrency or (4 if args.action == "smoke" else 35)
    if Path(args.name).name != args.name or args.name in {".", ".."}:
        parser.error("Use a single directory name for --name")
    if args.flavor != "h200x2":
        parser.error(
            "Use h200x2 for this two-GPU HF recipe; local H100 pairs are also supported"
        )
    namespace, separator, bucket = args.bucket.partition("/")
    if not separator or not namespace or not bucket:
        parser.error("Use owner/bucket for --bucket")
    command = [
        "bash",
        "/tutorial-source/jobs/entrypoint.sh",
        args.action,
        "--mode",
        args.mode,
        "--model",
        args.model,
        "--output",
        "/outputs/" + args.name,
        "--steps",
        str(args.steps),
        "--save-steps",
        str(args.save_steps),
        "--step",
        str(args.step),
        "--tasks",
        str(args.tasks),
        "--concurrency",
        str(args.concurrency),
    ]
    if args.checkpoint:
        command += ["--checkpoint", args.checkpoint]
    if args.space_id:
        command += ["--space-id", args.space_id]
    specification = {
        "image": "huggingface/trl@sha256:8433cde7eaf3b289f3daffdaefcab6bb71499ca1ff59dc5d05abba7a71603a58",
        "command": command,
        "namespace": namespace,
        "flavor": args.flavor,
        "timeout": args.timeout,
        "name": args.name,
        "env": {
            "PYTHONUNBUFFERED": "1",
            "TUTORIAL_MODE": args.mode,
        },
    }
    print(
        json.dumps(
            {
                **specification,
                "bucket": args.bucket,
                "secrets": ["HF_TOKEN", "DAYTONA_API_KEY"],
            },
            indent=2,
        )
    )
    if not args.submit:
        return
    from huggingface_hub import Volume, get_token, run_job, sync_job_volume

    token = get_token()
    if not token or not os.environ.get("DAYTONA_API_KEY"):
        parser.error("Log in to HF and set DAYTONA_API_KEY")
    with tempfile.TemporaryDirectory() as directory:
        # An allowlist keeps credentials, prepared tasks and old experiments out of the source upload.
        paths = list(ROOT.glob("*.py")) + [ROOT / "requirements.txt"]
        for folder in ("jobs", "envs", "train", "eval", "data"):
            paths += [
                p
                for p in (ROOT / folder).rglob("*")
                if p.suffix in {".py", ".sh", ".json", ".toml", ".txt", ".yaml"}
                and not any(
                    part in {".deps", ".venv", "prepared", ".archive", "__pycache__"}
                    or part.endswith(".egg-info")
                    for part in p.relative_to(ROOT).parts
                )
            ]
        for source in paths:
            target = Path(directory) / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        source_volume = sync_job_volume(
            directory, "/tutorial-source", namespace=namespace
        )
    job = run_job(
        **specification,
        volumes=[
            source_volume,
            Volume(
                type="bucket",
                source=args.bucket,
                mount_path="/outputs",
                read_only=False,
            ),
        ],
        secrets={"HF_TOKEN": token, "DAYTONA_API_KEY": os.environ["DAYTONA_API_KEY"]},
    )
    print(job.url)


if __name__ == "__main__":
    main()
