"""Launch DesignGym TRL SFT on Hugging Face Jobs, on the released SFT data (FineEnvs/designworld-sft).

    .venv/bin/python launch.py sft --flavor h200 --name smoke -- --max-steps 20 --save-steps 10 --grad-accum 4
    .venv/bin/python launch.py sft --flavor h200x4 --timeout 8h --name big -- --max-steps 700 --save-steps 60 --hours 7.3
    .venv/bin/python launch.py sft --model Qwen/Qwen3.8-27B --flavor h200x4 --timeout 8h --name big-27b -- ...
    .venv/bin/python launch.py merge --adapter <namespace>/designgym-sft-qwen3.6-35b-a3b-big --checkpoint checkpoint-120
    .venv/bin/python launch.py logs <job_id> [--follow]

The job runs this folder's code (bundled with designgym/cua_format.py into the runs repo), pinned to the versions
checked locally. Training data is read from the dataset repo over HTTP; `--bucket` additionally mounts an HF bucket
mirror of it read-only at /data (see train.py `--data-source`). Checkpoints go to the public model repo
<namespace>/designgym-sft-<model>-<name> as checkpoint-<step>/. Jobs run under `--namespace` (default
$HF_JOBS_NAMESPACE, else the token's user); bundles go to `--runs-repo` (default $DESIGNWORLD_SFT_RUNS_REPO, else
<namespace>/designworld-sft-runs). HF_TOKEN comes from the environment or the nearest .env up the tree and is passed
as a Job secret; it is never printed.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import tarfile
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import HfApi, Volume

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1]
CODE = ("train.py", "sft_format.py", "collator.py", "check_format.py", "sft_job.sh")


def _token() -> str:
    """HF_TOKEN from the environment, else the nearest .env up the tree (read by name, never printed)."""
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    for d in [HERE, *HERE.parents]:
        env = d / ".env"
        if not env.is_file():
            continue
        for line in env.read_text().splitlines():
            k, _, v = line.strip().removeprefix("export ").partition("=")
            if k == "HF_TOKEN":
                return v.strip().strip("'\"")
    raise SystemExit("HF_TOKEN not found in the environment or a .env up the tree")


def _defaults(a) -> None:
    if not a.namespace:
        a.namespace = os.environ.get("HF_JOBS_NAMESPACE") or HfApi(token=_token()).whoami()["name"]
    if getattr(a, "runs_repo", "") is None:
        a.runs_repo = os.environ.get("DESIGNWORLD_SFT_RUNS_REPO") or f"{a.namespace}/designworld-sft-runs"


def _bundle() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name in CODE:
            tar.add(HERE / name, arcname=f"sft/{name}")
        tar.add(EXP / "envs" / "designgym" / "cua_format.py", arcname="sft/cua_format.py")
    return buf.getvalue()


def sft(a) -> None:
    api = HfApi(token=_token())
    run = f"{datetime.now(UTC):%m%d-%H%M}-{a.name}"
    model_tag = a.model.split("/")[-1].lower()
    hub_id = a.hub_model_id or f"{a.namespace}/designgym-sft-{model_tag}-{a.name}"
    api.create_repo(a.runs_repo, repo_type="dataset", private=False, exist_ok=True)
    api.create_repo(hub_id, private=False, exist_ok=True)
    bundle = f"sft/{run}/code.tar.gz"
    api.upload_file(path_or_fileobj=_bundle(), path_in_repo=bundle, repo_id=a.runs_repo, repo_type="dataset",
                    commit_message=f"sft code {run}")
    train_args = " ".join([*a.train_args, "--hub-model-id", hub_id])
    job = api.run_job(
        image=a.image, command=["bash", "-c", (HERE / "sft_job.sh").read_text()],
        env={"BUNDLE_REPO": a.runs_repo, "BUNDLE_PATH": bundle, "RUN_NAME": run, "MODEL": a.model,
             "TRAIN_ARGS": train_args},
        secrets={"HF_TOKEN": _token()}, flavor=a.flavor, timeout=a.timeout, namespace=a.namespace,
        **({"volumes": [Volume(type="bucket", source=a.bucket, mount_path="/data", read_only=True)]} if a.bucket else {}),
        labels={"project": "designgym", "kind": "sft", "run": run},
    )
    print(json.dumps({"job": job.id, "url": job.url, "flavor": a.flavor, "run": run, "checkpoints": hub_id,
                      "train_args": train_args}, indent=1))


def merge(a) -> None:
    """A checkpoint's LoRA merged into the base weights (vLLM cannot serve a LoRA on this MoE), for evaluation."""
    sub = a.checkpoint or ""
    repo = a.repo or f"{a.adapter}-{sub.replace('checkpoint-', 'ck') or 'final'}-merged"
    job = HfApi(token=_token()).run_uv_job(
        str(HERE / "merge_lora.py"), script_args=["--base", a.base, "--adapter", a.adapter, "--subfolder", sub,
                                                  "--repo", repo],
        flavor=a.flavor, timeout=a.timeout, namespace=a.namespace, secrets={"HF_TOKEN": _token()},
        labels={"project": "designgym", "kind": "merge"},
    )
    print(json.dumps({"job": job.id, "url": job.url, "merged_repo": repo}, indent=1))


def logs(a) -> None:
    for line in HfApi(token=_token()).fetch_job_logs(job_id=a.job_id, namespace=a.namespace, follow=a.follow):
        print(line, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--namespace", default=None, help="who owns (and pays for) the job; default $HF_JOBS_NAMESPACE")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sft")
    s.add_argument("train_args", nargs="*", help="train.py options (after --)")
    s.add_argument("--name", required=True)
    s.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    s.add_argument("--bucket", default=os.environ.get("DESIGNWORLD_SFT_BUCKET"),
                   help="optional HF bucket mirror of the SFT data, mounted at /data; default $DESIGNWORLD_SFT_BUCKET")
    s.add_argument("--runs-repo", default=None, help="dataset for code bundles; default $DESIGNWORLD_SFT_RUNS_REPO")
    s.add_argument("--hub-model-id", default=None)
    s.add_argument("--flavor", default="h200")
    s.add_argument("--timeout", default="2h")
    s.add_argument("--image", default="ghcr.io/astral-sh/uv:python3.12-bookworm")
    s.set_defaults(fn=sft)
    m = sub.add_parser("merge")
    m.add_argument("--adapter", required=True, help="the run's checkpoint repo")
    m.add_argument("--checkpoint", default=None, help="checkpoint-<step> (default: the repo root, i.e. final)")
    m.add_argument("--repo", default=None)
    m.add_argument("--base", default="Qwen/Qwen3.6-35B-A3B")
    m.add_argument("--flavor", default="cpu-performance")
    m.add_argument("--timeout", default="2h")
    m.set_defaults(fn=merge)
    lg = sub.add_parser("logs")
    lg.add_argument("job_id")
    lg.add_argument("--follow", action="store_true")
    lg.set_defaults(fn=logs)
    a = ap.parse_args()
    _defaults(a)
    a.fn(a)


if __name__ == "__main__":
    main()
