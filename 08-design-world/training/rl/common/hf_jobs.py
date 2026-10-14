"""Launch Miles / SkyRL DesignGym RL jobs on Hugging Face Jobs, and follow them.

    python -m rl.common.hf_jobs launch miles --flavor h200x8 --name smoke --model Qwen/Qwen3.6-35B-A3B -- --gpus 8
    python -m rl.common.hf_jobs launch skyrl --flavor h200x8 --name smoke --model Qwen/Qwen3.8-27B
    python -m rl.common.hf_jobs logs <job_id> [--follow] [--grep PATTERN]
    python -m rl.common.hf_jobs status <job_id>
    python -m rl.common.hf_jobs cancel <job_id>

(run from the training folder). The job runs a bundle of this checkout's code, uploaded to the runs dataset
(`--runs-repo`, default $RL_RUNS_REPO, else <namespace>/designworld-rl-runs) as <framework>/<run>/code.tar.gz: the
shared core and the adapters (as rl/), the DesignGym env package (as designgym/; the job pulls the task pack) and the task list (tasks/train.jsonl). DesignGym runs inside the GPU job (common/job_common.sh); logs, metrics and
checkpoints go back to <framework>/<run>/ in the same repo. Jobs run under `--namespace` (default $HF_JOBS_NAMESPACE,
else the token's user). HF_TOKEN comes from the environment or the nearest .env up the tree and is passed as a Job
secret; it is never printed.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import tarfile
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import HfApi

HERE = Path(__file__).resolve().parent
RLF = HERE.parent
EXP = RLF.parents[1]
ENV_PKG = EXP / "envs" / "designgym"
TASKS = RLF / "tinker" / "tasks" / "rl-train-gui.jsonl"
SKINS = ("figma", "canva", "photoshop", "slides", "powerpoint", "inkscape")  # the mocked editors
SKIP = {".venv", "__pycache__", ".pytest_cache", ".ruff_cache", "tests", "artifacts", "runs"}
IMAGES = {
    "miles": "radixark/miles:latest",
    "skyrl": "novaskyai/skyrl-train-ray-2.57.0-py3.12-cu13.0-megatron",
}


def secret(name: str) -> str:
    """From the environment, else the nearest .env up the tree (read by name, never printed)."""
    if os.environ.get(name):
        return os.environ[name]
    for d in [HERE, *HERE.parents]:
        env = d / ".env"
        if not env.is_file():
            continue
        for line in env.read_text().splitlines():
            k, _, v = line.strip().removeprefix("export ").partition("=")
            if k == name:
                return v.strip().strip("'\"")
    raise SystemExit(f"{name} not found in the environment or a .env up the tree")


def _defaults(a) -> None:
    if not a.namespace:
        a.namespace = os.environ.get("HF_JOBS_NAMESPACE") or HfApi(token=secret("HF_TOKEN")).whoami()["name"]
    if getattr(a, "runs_repo", None) is None:
        a.runs_repo = os.environ.get("RL_RUNS_REPO") or f"{a.namespace}/designworld-rl-runs"


def _keep(info: tarfile.TarInfo):
    parts = set(Path(info.name).parts)
    return None if parts & SKIP or info.name.endswith((".pyc", ".egg-info")) else info


def _tasks(path: Path | None, n: int) -> bytes:
    """The task list (one reset kwargs dict per line): `--tasks-file`, else the first N GUI tasks of the env's `rl`
    split, one mocked editor each in turn."""
    if path is not None:
        return path.read_bytes()
    rows = [{"split": "rl", "index": i, "mode": "reference", "surface": "gui", "skin": SKINS[i % len(SKINS)],
             "coordinates": "normalized"} for i in range(n)]
    return "".join(json.dumps(r) + "\n" for r in rows).encode()


def bundle(tasks: bytes) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name in ("__init__.py", "common", "miles", "skyrl"):
            tar.add(RLF / name, arcname=f"rl/{name}", filter=_keep)
        tar.add(ENV_PKG, arcname="designgym", filter=_keep)
        info = tarfile.TarInfo("tasks/train.jsonl")
        info.size = len(tasks)
        tar.addfile(info, io.BytesIO(tasks))
    return buf.getvalue()


def launch(a) -> None:
    token = secret("HF_TOKEN")
    api = HfApi(token=token)
    run = f"{datetime.now(UTC):%m%d-%H%M}-{a.name}"
    api.create_repo(a.runs_repo, repo_type="dataset", private=a.private, exist_ok=True)
    path = f"{a.framework}/{run}/code.tar.gz"
    tasks_file = a.tasks_file or (TASKS if TASKS.is_file() else None)
    api.upload_file(path_or_fileobj=bundle(_tasks(tasks_file, a.tasks)), path_in_repo=path, repo_id=a.runs_repo, repo_type="dataset",
                    commit_message=f"{a.framework} code {run}")
    script = (RLF / a.framework / "job.sh").read_text()
    env = {"BUNDLE_REPO": a.runs_repo, "BUNDLE_PATH": path, "RESULTS_REPO": a.runs_repo, "FRAMEWORK": a.framework,
           "RUN_NAME": run, "MODEL": a.model, "N_TASKS": str(a.tasks), "ENV_WORKERS": str(a.env_workers),
           "ENV_URL": a.env_url or "", "TRAIN_ARGS": " ".join(a.train_args)}
    env.update(dict(kv.split("=", 1) for kv in a.env))
    volumes = []
    if a.cache_bucket:
        from huggingface_hub import Volume

        volumes.append(Volume(type="bucket", source=a.cache_bucket, mount_path="/cache", read_only=False))
        env["CACHE_DIR"] = "/cache"
    job = api.run_job(image=a.image or IMAGES[a.framework], command=["bash", "-c", script], env=env,
                      secrets={"HF_TOKEN": token}, flavor=a.flavor, timeout=a.timeout, namespace=a.namespace,
                      labels={"project": "designgym", "kind": f"rl-{a.framework}", "run": run},
                      **({"volumes": volumes} if volumes else {}))
    print(json.dumps({"job": job.id, "url": job.url, "flavor": a.flavor, "run": run, "model": a.model,
                      "results": f"https://huggingface.co/datasets/{a.runs_repo}/tree/main/{a.framework}/{run}"},
                     indent=1))


def logs(a) -> None:
    api = HfApi(token=secret("HF_TOKEN"))
    for line in api.fetch_job_logs(job_id=a.job_id, namespace=a.namespace, follow=a.follow):
        if a.grep and a.grep not in line:
            continue
        print(line, flush=True)


def status(a) -> None:
    job = HfApi(token=secret("HF_TOKEN")).inspect_job(job_id=a.job_id, namespace=a.namespace)
    print(json.dumps({"id": job.id, "status": str(job.status.stage), "message": job.status.message,
                      "flavor": job.flavor, "created": str(job.created_at)}, indent=1))


def cancel(a) -> None:
    HfApi(token=secret("HF_TOKEN")).cancel_job(job_id=a.job_id, namespace=a.namespace)
    print(f"cancelled {a.job_id}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--namespace", default=None, help="who owns (and pays for) the job; default $HF_JOBS_NAMESPACE")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("launch")
    s.add_argument("framework", choices=sorted(IMAGES))
    s.add_argument("train_args", nargs="*", help="trainer options (after --)")
    s.add_argument("--name", required=True)
    s.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    s.add_argument("--flavor", default="h200x8")
    s.add_argument("--timeout", default="2h")
    s.add_argument("--tasks", type=int, default=8, help="first N tasks of the task list")
    s.add_argument("--tasks-file", type=Path, default=None,
                   help="JSONL of reset kwargs (default: rl/tinker/tasks/rl-train-gui.jsonl if present, else the "
                        "env's rl split, GUI, one mocked editor per task in turn)")
    s.add_argument("--env-workers", type=int, default=8)
    s.add_argument("--env-url", default=None, help="use this DesignGym server instead of one in the job")
    s.add_argument("--env", action="append", default=[], help="extra KEY=VALUE for the job")
    s.add_argument("--image", default=None)
    s.add_argument("--cache-bucket", default=None, help="HF bucket mounted read-write at /cache (uv/venv cache)")
    s.add_argument("--runs-repo", default=None, help="dataset for bundles and results; default $RL_RUNS_REPO")
    s.add_argument("--private", action="store_true", help="create the runs repo private (when it does not exist)")
    s.set_defaults(fn=launch)
    lg = sub.add_parser("logs")
    lg.add_argument("job_id")
    lg.add_argument("--follow", action="store_true")
    lg.add_argument("--grep", default=None)
    lg.set_defaults(fn=logs)
    st = sub.add_parser("status")
    st.add_argument("job_id")
    st.set_defaults(fn=status)
    c = sub.add_parser("cancel")
    c.add_argument("job_id")
    c.set_defaults(fn=cancel)
    a = ap.parse_args()
    _defaults(a)
    a.fn(a)


if __name__ == "__main__":
    main()
