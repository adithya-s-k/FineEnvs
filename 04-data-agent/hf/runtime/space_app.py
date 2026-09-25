"""Run a private environment Space with the frozen native OpenEnv services."""
import os
import hmac
from common import ROOT, RUN, configure

configure()
arm = os.environ["COMPARISON_ARM"]
mode = "shared"
owner = os.environ.get("SPACE_ID", f"{arm}-{mode}").replace("/", "-")
os.environ["RUN_OWNER"] = owner
trials = ROOT / "space-trials" / owner
trials.mkdir(parents=True, exist_ok=True)
os.environ["OPENENV_HARBOR_TRIALS_DIR"] = str(trials)
os.environ["DAYTONA_WHITEBOX_TRIALS"] = str(trials)
os.environ["WHITE_BOX_BASH_TASK_SOURCE"] = "harbor-frozen"
os.environ["OPENENV_CAPTURE_TRANSPORT"] = "tunnel"
os.environ["OPENENV_DATASETS"] = ",".join(str(RUN / "datasets" / s) for s in ["train", "test"])
os.environ.setdefault("OPENENV_MODEL", "Qwen/Qwen3.5-2B")
os.environ.setdefault("OPENENV_MAX_OUTPUT_TOKENS", "16384")
os.environ.setdefault("OPENENV_EXPOSE", "gradio")
os.environ.setdefault("MAX_CONCURRENT_ENVS", "1024")
os.environ.setdefault("WHITE_BOX_BASH_MAX_CONCURRENT_ENVS", os.environ["MAX_CONCURRENT_ENVS"])
os.environ.setdefault("WHITE_BOX_BASH_MAX_SESSIONS", os.environ.get("SANDBOX_CAPACITY", "128"))
os.environ["ENABLE_WEB_INTERFACE"] = "false"

if arm == "blackbox":
    from harbor_service import install
    install()
    from harbor_env.server.app import app
elif arm == "whitebox":
    from whitebox_bash.server.app import app
else:
    raise ValueError("Unknown comparison arm")


@app.middleware("http")
async def protect_run_artifacts(request, call_next):
    # A protected HF Space has a public app. Keep cross-session run artifacts
    # behind the same bearer credential already sent by the training/eval bridge.
    if request.url.path == "/diagnostics" or request.url.path.startswith("/trial/"):
        from fastapi.responses import JSONResponse
        secret = os.environ.get("HF_TOKEN", "")
        provided = request.headers.get("Authorization", "")
        if not secret or not hmac.compare_digest(provided, "Bearer " + secret):
            return JSONResponse({"detail": "Authentication required"}, status_code=401)
    return await call_next(request)


@app.get("/deployment")
def deployment():
    from service_policy import admission
    return {"arm": arm, "mode": mode, "owner": owner, "train_tasks": 1000, "test_tasks": 250,
            "sandbox": "daytona", "bundle_sha256": os.environ.get("BUNDLE_SHA256"),
            "max_concurrent_envs": int(os.environ["MAX_CONCURRENT_ENVS"]),
            "admission": admission.snapshot(), "output_tokens": {"train": 16384, "eval": 4096},
            "interactive_ui": True, "trackio": False}


@app.get("/trial/{name}/result")
def trial_result(name: str):
    # Bearer authentication is enforced above. Expose only the native result metadata
    # needed to verify harness versions; arbitrary filesystem access is intentionally absent.
    import json
    from pathlib import Path
    from fastapi import HTTPException
    if Path(name).name != name or name in {".", ".."}:
        raise HTTPException(400)
    path = trials / name / "result.json"
    if not path.is_file():
        raise HTTPException(404)
    return json.loads(path.read_text())


@app.get("/diagnostics")
async def diagnostics():
    # No frame locals, prompts, credentials, or answer files. Useful for distinguishing
    # remote execution from response delivery when a long RPC stops making progress.
    import asyncio
    from service_policy import admission
    tasks = []
    for task in asyncio.all_tasks():
        tasks.append({"name": task.get_name(), "stack": [f"{f.f_code.co_name}:{f.f_lineno}" for f in task.get_stack(limit=4)]})
    rows = []
    for path in trials.iterdir():
        if path.is_dir():
            rows.append({"trial": path.name, "result_ready": (path / "result.json").exists(),
                         "cleanup_ready": (path / "cleanup.json").exists()})
    return {"admission": admission.snapshot(), "async_tasks": tasks, "trials": rows[-100:]}


@app.on_event("startup")
async def configure_thread_capacity():
    import anyio.to_thread
    anyio.to_thread.current_default_thread_limiter().total_tokens = 512


from environment_ui import mount_ui
app = mount_ui(app, arm)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860, ws_ping_interval=20, ws_ping_timeout=None,
                timeout_keep_alive=120, log_level="info")
