"""A job-local OpenEnv service. Sandboxes remain isolated on the selected provider."""
import os
import json
from pathlib import Path


def credentials():
    from huggingface_hub import get_token
    token = get_token()
    if not token:
        raise ValueError("Log in with hf auth login; Harbor task staging requires HF_TOKEN")
    os.environ["HF_TOKEN"] = token


def serve(cfg, data, vllm, port=8200, capture_port=8300):
    import uvicorn

    credentials()
    data = Path(data).resolve()
    output = Path(cfg["output"]).resolve()
    os.environ.update(OPENENV_LLM_URL=vllm, OPENENV_MODEL=cfg["profile"]["id"],
        OPENENV_MAX_OUTPUT_TOKENS=str(cfg["max_output_tokens"]), ENABLE_WEB_INTERFACE="false", RUN_OWNER=cfg["run_name"],
        MAX_CONCURRENT_ENVS=str(max(cfg["max_inflight"], cfg["eval_concurrency"]) + 8),
        OPENENV_HARBOR_AGENT_VERSIONS=json.dumps(cfg["harness_versions"]),
        OPENENV_CAPTURE_PORT=str(capture_port), OPENENV_EXPOSE="gradio")
    if cfg["mode"] == "opencode":
        os.environ.update(DATA_AGENT_SPLITS="train,test", DATA_AGENT_SANDBOX=cfg["sandbox"],
            DATA_AGENT_OPENCODE_VERSION=cfg["harness_versions"]["opencode"],
            DATA_AGENT_FROZEN_TASKS_DIR=str(data / "datasets"), DATA_AGENT_CAPTURE_EXPOSE="gradio",
            DATA_AGENT_CAPTURE_PORT=str(capture_port),
            DATA_AGENT_MAX_CONCURRENT=str(max(cfg["max_inflight"], cfg["eval_concurrency"])))
        from data_agent_env.server.app import app
    elif cfg["mode"] == "whitebox":
        if cfg["sandbox"] != "daytona":
            raise ValueError("The SETA adapter currently requires Daytona")
        os.environ.update(DAYTONA_COMPARISON_RUN=str(data), WHITE_BOX_BASH_TASK_SOURCE="harbor-frozen",
            DAYTONA_WHITEBOX_TRIALS=str(output / "trials"), WHITEBOX_RECORDS=str(output / "whitebox-records"),
            EFFICIENCY_WEIGHT=str(cfg["efficiency_weight"]), TOOL_BUDGET=str(cfg["tool_budget"]),
            WHITE_BOX_BASH_MAX_SESSIONS=str(max(cfg["max_inflight"], cfg["eval_concurrency"]) + 8),
            WHITE_BOX_BASH_MAX_CONCURRENT_ENVS=str(max(cfg["max_inflight"], cfg["eval_concurrency"]) + 8))
        from whitebox_bash.server.app import app
    else:
        os.environ.update(OPENENV_DATASETS=",".join(str(data / "datasets" / s) for s in ("train", "test")),
                          OPENENV_HARBOR_TRIALS_DIR=str(output / "trials"))
        from harbor_env.server.app import app
    uvicorn.run(app, host="127.0.0.1", port=port, ws_ping_timeout=None, timeout_keep_alive=120)
