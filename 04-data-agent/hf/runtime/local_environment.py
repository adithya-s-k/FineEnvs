"""Bind the frozen native OpenEnv service to loopback inside its Slurm allocation."""
import os
from common import RUN, ROOT, configure

configure()
arm = os.environ["COMPARISON_ARM"]
os.environ.update(ENABLE_WEB_INTERFACE="false", MAX_CONCURRENT_ENVS="128")
if arm == "opencode":
    os.environ.update(DATA_AGENT_SPLITS="train,test", DATA_AGENT_SANDBOX="daytona",
        DATA_AGENT_FROZEN_TASKS_DIR=str(RUN / "datasets"), DATA_AGENT_CAPTURE_EXPOSE="gradio",
        DATA_AGENT_MAX_CONCURRENT=os.environ.get("SANDBOX_CAPACITY", "64"))
    from data_agent_env.server.app import app
    from data_agent_env.tasks import rows_for
    counts = {split: len(rows_for(split)) for split in ("train", "test")}
else:
    os.environ.update(WHITE_BOX_BASH_TASK_SOURCE="harbor-frozen",
        DAYTONA_WHITEBOX_TRIALS=str(ROOT / "outputs" / os.environ["RUN_OWNER"] / "trials"),
        WHITE_BOX_BASH_MAX_SESSIONS="16", WHITE_BOX_BASH_MAX_CONCURRENT_ENVS="128")
    from whitebox_bash.server.app import app
    # The same frozen manifest-backed provider used by the Space.
    from whitebox_bash.tasks import num_tasks
    counts = {split: num_tasks(split) for split in ("train", "test")}


@app.get("/deployment")
def deployment():
    return {"arm": arm, "implementation": "standalone-opencode" if arm == "opencode" else "whitebox-seta",
        "bundle_sha256": os.environ["BUNDLE_SHA256"], "owner": os.environ["RUN_OWNER"],
        "train_tasks": counts["train"], "test_tasks": counts["test"], "local": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ["LOCAL_ENV_PORT"]),
                ws_ping_interval=20, ws_ping_timeout=None, timeout_keep_alive=120)
