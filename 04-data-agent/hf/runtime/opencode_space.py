"""Space entry point for envs/blackbox-opencode, independent of Harbor execution."""
import hmac
import os
from common import ROOT, RUN, configure

configure()
os.environ.update(DATA_AGENT_SPLITS="train,test", DATA_AGENT_SANDBOX="daytona",
    DATA_AGENT_FROZEN_TASKS_DIR=str(RUN / "datasets"), DATA_AGENT_CAPTURE_EXPOSE="gradio",
    DATA_AGENT_MAX_CONCURRENT=os.environ.get("SANDBOX_CAPACITY", "32"),
    HF_SANDBOX_NAMESPACE="HuggingEnvs", ENABLE_WEB_INTERFACE="false",
    RUN_OWNER=os.environ.get("SPACE_ID", "standalone-opencode").replace("/", "-"))
from data_agent_env.server.app import app
from data_agent_env.server.environment import DataAgentEnvironment
from data_agent_env.tasks import task_at, rows_for, _public
from data_agent_env.sandbox import BACKENDS, available
from data_agent_env.server.rollout import OPENCODE_VERSION


@app.get("/deployment")
def deployment():
    import hashlib
    return {"arm": "opencode", "implementation": "standalone-opencode", "mode": "shared",
        "source": "HuggingEnvs/04-data-agent/envs/blackbox-opencode",
        "bundle_sha256": os.environ.get("BUNDLE_SHA256"),
        "train_tasks": len(rows_for("train")), "test_tasks": len(rows_for("test")),
        "test_manifest_sha256": hashlib.sha256((RUN / "test_manifest.json").read_bytes()).hexdigest(),
        "sandboxes": {"supported": list(BACKENDS), "usable": available()},
        "sandbox_capacity": int(os.environ["DATA_AGENT_MAX_CONCURRENT"]),
        "opencode_version": OPENCODE_VERSION, "output_tokens": {"train": 16384, "test": 4096},
        "interactive_ui": True, "trackio": False}


class ServiceAuth:
    """Native execution RPCs use the owner credential; the public UI supplies its own inference."""
    def __init__(self, app): self.app = app
    async def __call__(self, scope, receive, send):
        path = scope.get("path", "")
        protected = (scope["type"] == "websocket" or path in {"/step", "/reset", "/state", "/mcp"}
                     or path.startswith("/mcp/"))
        if protected:
            headers = dict(scope.get("headers", []))
            token = os.environ.get("HF_TOKEN", "")
            if not token or not hmac.compare_digest(headers.get(b"authorization", b""), ("Bearer " + token).encode()):
                if scope["type"] == "websocket":
                    await send({"type": "websocket.close", "code": 1008})
                else:
                    from starlette.responses import JSONResponse
                    await JSONResponse({"detail": "Authentication required"}, status_code=401)(scope, receive, send)
                return
        await self.app(scope, receive, send)

app.add_middleware(ServiceAuth)

@app.on_event("startup")
async def startup():
    import anyio.to_thread
    anyio.to_thread.current_default_thread_limiter().total_tokens = 256
    for split in ("train", "test"): rows_for(split)


def preview(split, index):
    task = task_at(split, int(index))
    return task.instruction, f"**{task.difficulty_tier.capitalize()}** · {task.task_id}"


def rollout(split, index, backend, url, model, key):
    import json
    if not url.strip() or not model.strip():
        raise gr.Error("Enter an inference endpoint and model to run this task.")
    result = json.loads(DataAgentEnvironment()._run_rollout(split, int(index), url.strip(),
        model.strip(), backend, 17, 600, False, key))
    if result["reward"] is None:
        return "The rollout could not be graded. Check the endpoint and try again.", {}
    dialogue = "\n\n".join(f"### Turn {i+1}\n{t['text']}\n" +
        ("```json\n" + json.dumps(t['tool_calls'], indent=2) + "\n```" if t['tool_calls'] else "")
        for i,t in enumerate(result['turns']))
    return dialogue, {k:result[k] for k in ('correctness','reward','answer','answer_source','n_tool_calls','timed_out')}

import gradio as gr
from environment_ui import UI_CSS
with gr.Blocks(title="Data Agent Blackbox OpenCode Env") as demo:
    gr.HTML('''<div id="agent-hero"><div class="agent-eyebrow">HUGGINGENVS · DATA AGENT</div>
    <h1>Blackbox OpenCode</h1><p>Give OpenCode a real data-analysis task. It explores the tables,
    runs its own tools, and submits an answer in an isolated sandbox.</p>
    <div class="agent-chips"><span class="agent-chip">Daytona · HF · E2B</span>
    <span class="agent-chip">1,000 training tasks</span><span class="agent-chip">250 test tasks</span></div></div>''')
    with gr.Row():
        with gr.Column(scale=5, elem_id="task-panel"):
            split = gr.Radio(["train", "test"], value="test", label="Task set")
            index = gr.Number(value=2, precision=0, minimum=0, maximum=249, label="Task index")
            badge = gr.Markdown()
            instruction = gr.Textbox(label="Task", lines=16, interactive=False, elem_id="task-instructions")
        with gr.Column(scale=4, elem_id="workspace-panel"):
            backend = gr.Dropdown(["daytona", "hf", "e2b"], value="daytona", label="Sandbox")
            url = gr.Textbox(label="OpenAI-compatible inference URL", placeholder="https://…/v1")
            model = gr.Textbox(label="Model", value="Qwen/Qwen3.5-2B")
            key = gr.Textbox(label="Inference API key", type="password")
            run = gr.Button("Run OpenCode", variant="primary")
            score = gr.JSON(label="Result")
    transcript = gr.Markdown(label="Agent activity")
    split.change(lambda s: gr.update(maximum=999 if s == "train" else 249, value=2), split, index)
    for event in (split.change, index.change): event(preview, [split,index], [instruction,badge], api_name=False)
    demo.load(preview, [split,index], [instruction,badge], api_name=False)
    run.click(rollout, [split,index,backend,url,model,key], [transcript,score], concurrency_limit=2, api_name=False)
app = gr.mount_gradio_app(app, demo, path="/", css=UI_CSS)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860, ws_ping_interval=20, ws_ping_timeout=None,
                timeout_keep_alive=120)
