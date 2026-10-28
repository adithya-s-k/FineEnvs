"""Launch DesignGym jobs on Hugging Face Jobs: export a Tinker checkpoint to the Hub, then evaluate base vs RL.

    # 1. Tinker LoRA -> PEFT adapter in a private model repo (CPU job; downloads the base weights there)
    .venv/bin/python hf_jobs/launch.py export --tinker-path tinker://<run>/sampler_weights/final \
        --repo <namespace>/designgym-qwen3.6-35b-a3b-rl-tools --merged

    # 2. base vs base+adapter on core30, GUI first (one H200: vLLM + the env server + Chromium)
    .venv/bin/python hf_jobs/launch.py eval --rl-model <namespace>/designgym-qwen3.6-35b-a3b-rl-tools \
        --surfaces gui,tools --subset core30
    .venv/bin/python hf_jobs/launch.py eval --base-model Qwen/Qwen3.8-27B --surfaces gui --subset core30

    # Tinker RL driven from a CPU job next to its own env server (training itself runs on Tinker)
    .venv/bin/python hf_jobs/launch.py rl -- tasks_path=tasks/rl-reference-gui.jsonl model_name=Qwen/Qwen3.6-35B-A3B

    .venv/bin/python hf_jobs/launch.py logs <job_id> --follow
    .venv/bin/python hf_jobs/launch.py status <job_id>

The eval job runs the env from a bundle of this checkout's code (the DesignGym package and the DesignWorldBench
evalsets, uploaded to the private results repo), so it scores exactly what is local. Results land in the results repo
(`--results-repo`, default $DESIGNWORLD_RESULTS_REPO, else <namespace>/designworld-evals) under runs/<run name>/
(rollouts/summary.json, trace.jsonl, frames, and the vLLM / env / rollout logs). Jobs run under `--namespace`
(default $HF_JOBS_NAMESPACE, else the token's user). Tokens come from the environment or the nearest .env up the tree
and are passed as Job secrets; they are never printed.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import tarfile
import time
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import HfApi

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[3]
ENV_PKG = EXP / "envs" / "designgym"
BENCH = EXP / "bench"
SKIP = {".venv", "__pycache__", ".pytest_cache", ".ruff_cache", "tests", "artifacts"}


def _secret(name: str) -> str:
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
        a.namespace = os.environ.get("HF_JOBS_NAMESPACE") or HfApi(token=_secret("HF_TOKEN")).whoami()["name"]
    if getattr(a, "results_repo", "") is None:
        a.results_repo = os.environ.get("DESIGNWORLD_RESULTS_REPO") or f"{a.namespace}/designworld-evals"


def _bundle() -> bytes:
    """The env package (as designgym/) and the frozen evalsets (bench/), as a tar.gz."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        def keep(info: tarfile.TarInfo):
            parts = set(Path(info.name).parts)
            return None if parts & SKIP or info.name.endswith((".pyc", ".egg-info")) else info
        tar.add(ENV_PKG, arcname="designgym", filter=keep)
        tar.add(BENCH, arcname="bench", filter=keep)
    return buf.getvalue()


def _rl_bundle() -> bytes:
    """The eval bundle plus this training project (as tinker/) with the exact package versions installed here."""
    import subprocess

    buf = io.BytesIO()
    with tarfile.open(fileobj=io.BytesIO(_bundle()), mode="r:gz") as base, tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for m in base.getmembers():
            tar.addfile(m, base.extractfile(m) if m.isfile() else None)
        skip = SKIP | {"runs", "evals", "tasks"}

        def keep(info: tarfile.TarInfo):
            return None if set(Path(info.name).parts) & skip or info.name.endswith(".pyc") else info
        tar.add(HERE.parent, arcname="tinker", filter=keep)
        if (HERE.parent / "tasks").is_dir():
            tar.add(HERE.parent / "tasks", arcname="tinker/tasks")
        freeze = subprocess.run(["uv", "pip", "freeze", "--python", str(HERE.parent / ".venv" / "bin" / "python")],
                                capture_output=True, text=True, check=True).stdout
        data = "\n".join(l for l in freeze.splitlines() if not l.startswith("-e ")).encode()
        info = tarfile.TarInfo("tinker/requirements.lock")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def rl(a) -> None:
    token = _secret("HF_TOKEN")
    api = HfApi(token=token)
    run = a.name or datetime.now(UTC).strftime("%Y%m%d-%H%M") + "-rl"
    api.create_repo(a.results_repo, repo_type="dataset", private=True, exist_ok=True)
    bundle_path = f"inputs/{run}/rl-bundle.tar.gz"
    api.upload_file(path_or_fileobj=_rl_bundle(), path_in_repo=bundle_path, repo_id=a.results_repo,
                    repo_type="dataset", commit_message=f"rl bundle {run}")
    job = api.run_job(
        image=a.image, command=["bash", "-c", (HERE / "rl_job.sh").read_text()],
        env={"BUNDLE_REPO": a.results_repo, "BUNDLE_PATH": bundle_path, "RESULTS_REPO": a.results_repo,
             "RUN_NAME": run, "ENV_WORKERS": str(a.env_workers), "SYNC_MINUTES": str(a.sync_minutes),
             "TRAIN_ARGS": " ".join(a.train_args)},
        secrets={"HF_TOKEN": token, "TINKER_API_KEY": _secret("TINKER_API_KEY")},
        flavor=a.flavor, timeout=a.timeout, namespace=a.namespace,
        labels={"project": "designgym", "kind": "rl", "run": run},
    )
    print(f"rl job {job.id} ({a.flavor}): {job.url}\nrun dir -> https://huggingface.co/datasets/{a.results_repo}/tree/main/rl/{run}")


def export(a) -> None:
    api = HfApi(token=_secret("HF_TOKEN"))
    job = api.run_uv_job(
        str(HERE / "export_adapter.py"),
        script_args=["--tinker-path", a.tinker_path, "--base-model", a.base_model, "--repo", a.repo,
                     "--notes", a.notes] + (["--merged"] if a.merged else []),
        flavor=a.flavor, timeout=a.timeout, namespace=a.namespace,
        secrets={"HF_TOKEN": _secret("HF_TOKEN"), "TINKER_API_KEY": _secret("TINKER_API_KEY")},
        labels={"project": "designgym", "kind": "export"},
    )
    print(f"export job {job.id}: {job.url}")


def evaluate(a) -> None:
    token = _secret("HF_TOKEN")
    api = HfApi(token=token)
    run = a.name or datetime.now(UTC).strftime("%Y%m%d-%H%M") + "-" + (a.rl_model or "base").split("/")[-1]
    api.create_repo(a.results_repo, repo_type="dataset", private=True, exist_ok=True)
    bundle_path = f"inputs/{run}/bundle.tar.gz"
    api.upload_file(path_or_fileobj=_bundle(), path_in_repo=bundle_path, repo_id=a.results_repo, repo_type="dataset",
                    commit_message=f"eval bundle {run}")
    env = {"BASE_MODEL": a.base_model, "RL_MODEL": a.rl_model or "", "EVALSET": a.evalset,
           "SURFACES": a.surfaces, "SUBSET": a.subset, "LIMIT": str(a.limit or ""), "SKINS": a.skins or "",
           "CONCURRENCY": str(a.concurrency), "BUNDLE_REPO": a.results_repo, "BUNDLE_PATH": bundle_path,
           "RESULTS_REPO": a.results_repo, "RUN_NAME": run, "VLLM_VERSION": a.vllm, "MAX_MODEL_LEN": str(a.max_model_len),
           "ENV_ONLY": "1" if a.env_only else "", "RESUME_FROM": a.resume_from or "", "ENV_URL": "" if a.local_env or a.real_apps else a.env_url}
    secrets = {"HF_TOKEN": token}
    if a.real_apps:  # the env server in the job runs the real apps in its own Daytona pool, one desktop per episode
        env.update(DESIGN_CANVAS_APPS="daytona", DESIGN_CANVAS_DAYTONA_POOL=f"eval-{run}"[:60],
                   DESIGN_CANVAS_DAYTONA_DESKTOPS=str(a.concurrency))
        secrets.update(DAYTONA_API_KEY=_secret("DAYTONA_API_KEY"), DAYTONA_API_URL=_secret("DAYTONA_API_URL"))
    # --split: one single-GPU job per model, in parallel (same results folder; each writes rollouts-<model>)
    parts = [("base", "h200"), ("rl", "h200")] if a.split and a.rl_model else [("", a.flavor)]
    for only, flavor in parts:
        job = api.run_job(
            image=a.image, command=["bash", "-c", (HERE / "eval_job.sh").read_text()],
            env={**env, "ONLY": only, "RUN_NAME": run}, secrets=secrets, flavor=flavor,
            timeout=a.timeout, namespace=a.namespace,
            labels={"project": "designgym", "kind": "eval", "run": run, "model": only or "all"},
        )
        print(f"eval job {job.id} ({flavor}{', ' + only if only else ''}): {job.url}")
    print(f"results -> https://huggingface.co/datasets/{a.results_repo}/tree/main/runs/{run}")


def logs(a) -> None:
    api = HfApi(token=_secret("HF_TOKEN"))
    for line in api.fetch_job_logs(job_id=a.job_id, namespace=a.namespace, follow=a.follow):
        print(line, flush=True)


def status(a) -> None:
    info = HfApi(token=_secret("HF_TOKEN")).inspect_job(job_id=a.job_id, namespace=a.namespace)
    print(json.dumps({"id": info.id, "status": str(info.status.stage), "message": info.status.message,
                      "flavor": str(info.flavor), "created": str(info.created_at)}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--namespace", default=None, help="who owns (and pays for) the job; default $HF_JOBS_NAMESPACE")
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("export", help="Tinker checkpoint -> PEFT adapter -> private Hub repo")
    e.add_argument("--tinker-path", required=True)
    e.add_argument("--repo", required=True)
    e.add_argument("--base-model", default="Qwen/Qwen3.6-35B-A3B")
    e.add_argument("--notes", default="", help="JSON object of training details for the model card")
    e.add_argument("--merged", action="store_true", help="publish merged full weights instead of a PEFT adapter")
    e.add_argument("--flavor", default="cpu-performance")
    e.add_argument("--timeout", default="2h")
    e.set_defaults(fn=export)

    v = sub.add_parser("eval", help="vLLM (base + adapter) + DesignGym + rollout.py on one GPU node")
    v.add_argument("--rl-model", default=None, help="merged RL model repo (export --merged); omit for base only")
    v.add_argument("--base-model", default="Qwen/Qwen3.6-35B-A3B")
    v.add_argument("--surfaces", default="gui,tools")
    v.add_argument("--evalset", default="bench/evalsets/recreation-cua-v1.json",
                   help="frozen evalset, relative to the DesignWorld root (bundled with bench/)")
    v.add_argument("--subset", default="core30")
    v.add_argument("--limit", type=int, default=None, help="pilot: only this many tasks")
    v.add_argument("--skins", default=None, help="GUI: run each task in each of these editors (comma list)")
    v.add_argument("--concurrency", type=int, default=12, help="episodes at once (each GUI one drives a Chromium)")
    v.add_argument("--results-repo", default=None,
                   help="dataset for bundles and results; default $DESIGNWORLD_RESULTS_REPO")
    v.add_argument("--name", default=None)
    v.add_argument("--flavor", default="h200x2", help="h200x2 evaluates base and RL in parallel; h200 one after the other")
    v.add_argument("--timeout", default="6h")
    v.add_argument("--image", default="ghcr.io/astral-sh/uv:python3.12-bookworm")
    v.add_argument("--vllm", default="0.30.0")
    v.add_argument("--max-model-len", type=int, default=65536)
    v.add_argument("--split", action="store_true", help="one single-H200 job per model, in parallel")
    v.add_argument("--env-only", action="store_true", help="debug: start only the env server and open one episode")
    v.add_argument("--env-url", default="https://fineenvs-designgym.hf.space",
                   help="the DesignGym server to play against (default: the hosted Space; reward v3)")
    v.add_argument("--local-env", action="store_true", help="run the env inside the job instead of using --env-url")
    v.add_argument("--resume-from", default=None, help="a previous run name: keep its clean episodes, rerun the rest")
    v.add_argument("--real-apps", action="store_true",
                   help="real apps (--skins impress,draw,inkscape-desktop): a local env with its own Daytona pool "
                        "(eval-<run>)")
    v.set_defaults(fn=evaluate)

    r = sub.add_parser("rl", help="host the env + drive Tinker RL from one CPU job (training itself runs on Tinker)")
    r.add_argument("train_args", nargs="+", help="train_designgym.py key=value options (env_url/log_path are set)")
    r.add_argument("--results-repo", default=None,
                   help="dataset for bundles and run dirs; default $DESIGNWORLD_RESULTS_REPO")
    r.add_argument("--name", default=None)
    r.add_argument("--flavor", default="cpu-performance")
    r.add_argument("--env-workers", type=int, default=8)
    r.add_argument("--sync-minutes", type=int, default=10)
    r.add_argument("--timeout", default="24h")
    r.add_argument("--image", default="ghcr.io/astral-sh/uv:python3.12-bookworm")
    r.set_defaults(fn=rl)

    lg = sub.add_parser("logs")
    lg.add_argument("job_id")
    lg.add_argument("--follow", action="store_true")
    lg.set_defaults(fn=logs)
    st = sub.add_parser("status")
    st.add_argument("job_id")
    st.set_defaults(fn=status)

    a = ap.parse_args()
    _defaults(a)
    t0 = time.time()
    a.fn(a)
    if a.cmd in ("export", "eval", "rl"):
        print(f"launched in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
