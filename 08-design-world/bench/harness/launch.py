"""Run DesignWorldBench on Hugging Face Jobs: serve Hub models with vLLM, play the frozen set on DesignGym, publish.

    # one model, GUI, the 30-task core subset, against the hosted DesignGym Space
    python bench/harness/launch.py --namespace <org> eval --models Qwen/Qwen3.6-35B-A3B --subset core30

    # two models side by side on a 2-GPU node, GUI and tool calls, the 60-task lite subset
    python bench/harness/launch.py --namespace <org> eval --models <repo-a>,<repo-b> --flavor h200x2 \
        --surfaces gui,tools --subset lite

    # finish an interrupted run: its clean episodes are kept, the rest are played
    python bench/harness/launch.py --namespace <org> eval --models <repo> --resume-from runs/<earlier run>

    # the real apps (LibreOffice Impress, LibreOffice Draw, Inkscape): an env server in the job with a Daytona pool
    python bench/harness/launch.py --namespace <org> eval --models <repo> --real-apps impress,draw,inkscape-desktop

    python bench/harness/launch.py --namespace <org> logs <job_id> --follow
    python bench/harness/launch.py --namespace <org> status <job_id>

The job checks out the public repo (--repo-url, --ref; default branch design-world) and runs
08-design-world/envs/designgym/rollout.py on 08-design-world/bench/evalsets/recreation-cua-v1.json, so it scores
exactly what is on that ref (pass a commit SHA to pin it). Raw results (summary.json, trace.jsonl, frames, renders,
vLLM / env / rollout logs, job.json) land in the results repo under runs/<run>/; the run as cua-v1 rows
(bench/harness/to_cua.py) under data/<run>/. The namespace that owns and pays for the job comes from --namespace or
DESIGNWORLD_JOBS_NAMESPACE; the results repo from --results-repo or DESIGNWORLD_RESULTS_REPO. HF_TOKEN is read from
the environment or the local Hugging Face login and passed as a Job secret.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import HfApi

HERE = Path(__file__).resolve().parent
REPO_URL = "https://github.com/adithya-s-k/FineEnvs"
REF = "design-world"
RESULTS_REPO = os.environ.get("DESIGNWORLD_RESULTS_REPO", "FineEnvs/designworld-bench-runs")
NAMESPACE = os.environ.get("DESIGNWORLD_JOBS_NAMESPACE")
SPACE_URL = "https://fineenvs-designgym.hf.space"
REAL_APPS = ("impress", "draw", "inkscape-desktop")  # the public real apps: LibreOffice Impress, Draw, Inkscape


def _secret(name: str, required: bool = True) -> str | None:
    if os.environ.get(name):
        return os.environ[name]
    if name == "HF_TOKEN":
        from huggingface_hub import get_token

        if token := get_token():
            return token
    if required:
        raise SystemExit(f"{name} is not set (export it, or `hf auth login` for HF_TOKEN)")
    return None


def _slug(model: str) -> str:
    return model.replace("/", "__")


def _real_apps(value: str | None) -> list[str]:
    if not value:
        return []
    apps = [a.strip() for a in value.split(",") if a.strip()]
    bad = [a for a in apps if a not in REAL_APPS]
    if bad:
        raise SystemExit(f"--real-apps: {', '.join(bad)} not available; choose from {', '.join(REAL_APPS)}")
    return apps


def _resume_source(api: HfApi, a, run: str) -> str:
    """A local run directory is uploaded (traces, frames, renders) next to the run's inputs; a Hub path is passed on."""
    src = Path(a.resume_from)
    if not src.is_dir():
        return a.resume_from
    path = f"inputs/{run}/resume"
    api.upload_folder(repo_id=a.results_repo, repo_type="dataset", folder_path=str(src), path_in_repo=path,
                      allow_patterns=["trace.jsonl", "*/trace.jsonl", "frames/**", "renders/**", "*/frames/**",
                                      "*/renders/**"], commit_message=f"resume input for {run}")
    return f"{a.results_repo}/{path}"


def evaluate(a) -> None:
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    apps = _real_apps(a.real_apps)
    if a.skins and apps:
        raise SystemExit("give either --skins (built-in editors) or --real-apps, not both")
    token = _secret("HF_TOKEN")
    api = HfApi(token=token)
    run = a.name or datetime.now(UTC).strftime("%Y%m%d-%H%M") + "-" + models[0].split("/")[-1]
    api.create_repo(a.results_repo, repo_type="dataset", private=True, exist_ok=True)
    env = {"MODELS": ",".join(models), "THINKING": a.thinking, "TP": str(a.tensor_parallel),
           "SURFACES": "gui" if apps else a.surfaces, "MODES": a.modes, "SUBSET": a.subset,
           "LIMIT": str(a.limit or ""), "SKINS": ",".join(apps) or (a.skins or ""), "CONCURRENCY": str(a.concurrency),
           "MAX_TOKENS": str(a.max_tokens), "IMAGE_WINDOW": a.image_window, "RESULTS_REPO": a.results_repo,
           "RUN_NAME": run, "REPO_URL": a.repo_url, "REPO_REF": a.ref, "VLLM_VERSION": a.vllm,
           "MAX_MODEL_LEN": str(a.max_model_len), "TOOL_PARSER": a.tool_call_parser, "PATCH_TOOL_LISTING": "1" if a.patch_tool_listing else "",
           "ENV_ONLY": "1" if a.env_only else "", "PUBLISH_CUA": "" if a.no_cua else "1",
           "RESUME_FROM": _resume_source(api, a, run) if a.resume_from else "",
           "ENV_URL": "" if a.local_env or apps else a.env_url}
    secrets = {"HF_TOKEN": token}
    if apps:  # the env server in the job runs the real apps in its own Daytona pool, one desktop per episode
        env.update(DESIGN_CANVAS_APPS="daytona", DESIGN_CANVAS_DAYTONA_POOL=f"eval-{run}"[:60],
                   DESIGN_CANVAS_DAYTONA_DESKTOPS=str(a.concurrency))
        secrets.update(DAYTONA_API_KEY=_secret("DAYTONA_API_KEY"))
        if url := _secret("DAYTONA_API_URL", required=False):
            secrets["DAYTONA_API_URL"] = url
    # --split: one job per model, in parallel (same run folder; each writes rollouts-<model>)
    parts = [[m] for m in models] if a.split and len(models) > 1 else [models]
    for part in parts:
        job = api.run_job(
            image=a.image, command=["bash", "-c", (HERE / "eval_job.sh").read_text()],
            env={**env, "MODELS": ",".join(part)}, secrets=secrets, flavor=a.flavor,
            timeout=a.timeout, namespace=a.namespace,
            labels={"project": "designworld", "kind": "bench", "run": run,
                    "model": _slug(part[0])[:60] if len(part) == 1 else "all"},
        )
        print(f"bench job {job.id} ({a.flavor}, {', '.join(part)}): {job.url}")
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
    ap.add_argument("--namespace", default=NAMESPACE,
                    help="org or user that owns (and pays for) the job; default DESIGNWORLD_JOBS_NAMESPACE")
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("eval", help="vLLM + DesignGym + rollout.py on one GPU node, then publish the run")
    v.add_argument("--models", required=True, help="comma list of Hub model repos, each served by vLLM")
    v.add_argument("--thinking", default="off", choices=("off", "low", "default"), help="rollout.py thinking mode")
    v.add_argument("--tensor-parallel", type=int, default=1, help="GPUs per model")
    v.add_argument("--surfaces", default="gui", help="comma list of tools, html, gui")
    v.add_argument("--modes", default="reference", help="comma list of reference, description")
    v.add_argument("--subset", default="core30", help="full, lite or core30")
    v.add_argument("--limit", type=int, default=None, help="pilot: only this many tasks, balanced across tiers")
    v.add_argument("--skins", default=None, help="GUI: run each task in each of these built-in editors (comma list)")
    v.add_argument("--real-apps", default=None, metavar="APPS",
                   help=f"GUI in real apps instead ({', '.join(REAL_APPS)}): a local env with its own Daytona pool")
    v.add_argument("--concurrency", type=int, default=12, help="episodes at once (each GUI one drives a Chromium)")
    v.add_argument("--max-tokens", type=int, default=4096)
    v.add_argument("--image-window", default="8,3", help="rollout.py --image-window MAX,KEEP")
    v.add_argument("--results-repo", default=RESULTS_REPO)
    v.add_argument("--name", default=None, help="run name (default: <UTC time>-<first model>)")
    v.add_argument("--flavor", default="h200", help="h200x2 serves two models (or one with TP 2) at once")
    v.add_argument("--timeout", default="6h")
    v.add_argument("--image", default="ghcr.io/astral-sh/uv:python3.12-bookworm")
    v.add_argument("--vllm", default="0.30.0")
    v.add_argument("--max-model-len", type=int, default=65536)
    v.add_argument("--tool-call-parser", default="qwen3_coder", help="vLLM tool call parser for these models")
    v.add_argument("--patch-tool-listing", action="store_true",
                   help="list tools in the chat template as bare function objects (as the Tinker renderer does)")
    v.add_argument("--split", action="store_true", help="one job per model, in parallel")
    v.add_argument("--env-only", action="store_true", help="debug: start only the env and open one episode")
    v.add_argument("--env-url", default=SPACE_URL, help="the DesignGym server to play against (default: the Space)")
    v.add_argument("--local-env", action="store_true", help="run the env inside the job instead of using --env-url")
    v.add_argument("--resume-from", default=None,
                   help="a previous run (local dir, runs/<name>, <name> or <org>/<repo>/<path>): keep its clean "
                        "episodes, play the rest")
    v.add_argument("--no-cua", action="store_true", help="don't publish the run as cua-v1 rows")
    v.add_argument("--repo-url", default=REPO_URL)
    v.add_argument("--ref", default=REF, help="branch, tag or commit SHA to check out")
    v.set_defaults(fn=evaluate)

    lg = sub.add_parser("logs")
    lg.add_argument("job_id")
    lg.add_argument("--follow", action="store_true")
    lg.set_defaults(fn=logs)
    st = sub.add_parser("status")
    st.add_argument("job_id")
    st.set_defaults(fn=status)

    a = ap.parse_args()
    if not a.namespace:
        ap.error("--namespace (or DESIGNWORLD_JOBS_NAMESPACE) is required")
    t0 = time.time()
    a.fn(a)
    if a.cmd == "eval":
        print(f"launched in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
