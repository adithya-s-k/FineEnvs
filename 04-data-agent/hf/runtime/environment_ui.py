"""Interactive task clients, mounted after the native OpenEnv API as in environment-101."""
from dataclasses import dataclass, field
import json
import os
import secrets
import threading
import time

import gradio as gr
import httpx
from common import RUN

LOCAL = "http://127.0.0.1:7860"
TITLES = {"whitebox": "Data Agent SETA Whitebox Env",
          "blackbox": "Data Agent Blackbox Harbor Env"}

UI_CSS = """
.gradio-container {max-width: 1440px !important; margin: auto !important;}
#agent-hero {padding: 30px 32px; border: 1px solid var(--border-color-primary);
  border-radius: 20px; margin-bottom: 18px;
  background: linear-gradient(120deg, rgba(16,185,129,.10), rgba(59,130,246,.05));}
#agent-hero h1 {font-size: clamp(26px, 3vw, 38px); line-height: 1.2; margin: 12px 0;}
#agent-hero p {max-width: 820px; line-height: 1.6; opacity: .85; margin: 0;}
.agent-eyebrow {font-size: 12px; letter-spacing: .13em; font-weight: 700;}
.agent-chips {display: flex; flex-wrap: wrap; gap: 8px; margin-top: 18px;}
.agent-chip {font-size: 12px; border: 1px solid var(--border-color-primary);
  border-radius: 999px; padding: 5px 11px; background: var(--background-fill-primary);}
#task-panel {border: 1px solid var(--border-color-primary); border-radius: 16px; padding: 20px;}
#workspace-panel {border: 1px solid var(--border-color-primary); border-radius: 16px; padding: 20px;}
#task-instructions textarea {font-size: 14px; line-height: 1.65;}
#task-badge {font-size: 13px; opacity: .8;}
#tool-console {min-height: 290px;}
.agent-footnote {font-size: 12px; opacity: .75;}
"""


def task_badge(metadata):
    difficulty = str(metadata.get("difficulty", "Task ready")).capitalize()
    name = metadata.get("task_name", metadata.get("task_id", ""))
    return f"**{difficulty}** · {name}" if name else f"**{difficulty}**"


def tool_inputs(tool):
    return (gr.update(visible=tool in {"bash", "grep"},
                      label="Search pattern" if tool == "grep" else "Shell command"),
            gr.update(visible=tool != "bash"), gr.update(visible=tool in {"write", "edit"}),
            gr.update(visible=tool == "edit"))


def split_spec(arm, split):
    if split not in ("train", "test"):
        raise ValueError("Choose train or test")
    return str(RUN / "datasets" / split) if arm == "blackbox" else split


def preview(arm, split, index):
    index = validate_index(split, index)
    name = "harbor_env" if arm == "blackbox" else "white_box_bash"
    response = httpx.post(LOCAL + f"/{name}/task", json={"split": split_spec(arm, split),
                          "index": int(index)}, timeout=30)
    response.raise_for_status()
    task = response.json()["task"]
    instruction = task.get("instruction", task.get("prompt", ""))
    return instruction, {key: task[key] for key in ("task_name", "task_id", "difficulty") if key in task}


def validate_index(split, index):
    limit = {"train": 1000, "test": 250}.get(split)
    if limit is None or index is None or int(index) != index or not 0 <= index < limit:
        raise gr.Error(f"Choose a whole-number task index from 0 to {(limit or 1) - 1}.")
    return int(index)


@dataclass
class BrowserSession:
    lock: threading.Lock = field(default_factory=threading.Lock)
    env: object = None
    used_at: float = field(default_factory=time.monotonic)


class WhiteboxUI:
    """One isolated native MCP client/sandbox per browser; explicit and idle cleanup."""
    def __init__(self):
        self.sessions = {}
        self.lock = threading.Lock()
        self.stop = threading.Event()
        threading.Thread(target=self.reap, daemon=True, name="ui-sandbox-cleanup").start()

    def session(self, request):
        if not request.session_hash:
            raise gr.Error("A browser session is required")
        with self.lock:
            return self.sessions.setdefault(request.session_hash, BrowserSession())

    def dispose(self, s):
        if s.env is not None:
            env = s.env
            if env._session and env._reward is None:
                env._mcp.call("close_episode", session_id=env._session)
            env._mcp.close()
            s.env = None

    def reap(self):
        while not self.stop.wait(30):
            with self.lock:
                items = list(self.sessions.items())
            for key, s in items:
                if time.monotonic() - s.used_at > 1200 and s.lock.acquire(blocking=False):
                    try:
                        # Check again after locking: an active tool may have refreshed it.
                        if time.monotonic() - s.used_at <= 1200:
                            continue
                        self.dispose(s)
                        with self.lock:
                            self.sessions.pop(key, None)
                    except Exception:
                        pass  # Retry on the next sweep; no user inputs or credentials in logs.
                    finally:
                        s.lock.release()

    def start(self, split, index, request: gr.Request):
        index = validate_index(split, index)
        from whitebox_bash import white_box_bash_env
        s = self.session(request)
        with s.lock:
            self.dispose(s)
            with self.lock:
                active = sum(x.env is not None for x in self.sessions.values())
                if active >= 8:
                    raise gr.Error("All interactive workspaces are in use. Try again after a session finishes.")
                env = white_box_bash_env(LOCAL, toolsets="bash,seta", step_limit=100)()
                s.env = env
            try:
                instruction = env.reset(split=split, index=int(index))
                s.used_at = time.monotonic()
                return instruction, "Sandbox ready. Working directory: /workdir", "Active"
            except Exception as exc:
                self.dispose(s)
                raise gr.Error(f"Could not start the task ({type(exc).__name__}).") from None

    def run(self, tool, command, path, content, old, request: gr.Request):
        s = self.session(request)
        with s.lock:
            if s.env is None:
                raise gr.Error("Start a task first")
            e = s.env
            args = {"bash": {"command": command}, "read": {"path": path},
                    "write": {"path": path, "content": content},
                    "edit": {"path": path, "old": old, "new": content},
                    "grep": {"pattern": command, "path": path},
                    "glob": {"pattern": path}, "ls": {"path": path or "."}}
            if tool not in args:
                raise gr.Error("Unknown tool")
            try:
                return getattr(e, tool)(**args[tool])
            finally:
                s.used_at = time.monotonic()

    def grade(self, answer, request: gr.Request):
        s = self.session(request)
        with s.lock:
            if s.env is None:
                raise gr.Error("Start a task first")
            try:
                if answer.strip():
                    s.env.submit_solution(answer=answer)
                reward = s.env.get_reward()
                return f"Reward: {reward}", "Finished"
            finally:
                self.dispose(s)
                s.used_at = time.monotonic()

    def close(self, request: gr.Request):
        s = self.session(request)
        with s.lock:
            self.dispose(s)
            s.used_at = time.monotonic()
        return "Sandbox closed", "Closed"


def blackbox_run(split, index, harness, url, model, key):
    from openenv.harbor.client import HarborEnv
    index = validate_index(split, index)
    if not url.strip():
        raise gr.Error("Enter the OpenAI-compatible inference endpoint to use")
    yield "Starting the agent in an isolated workspace…", {}
    try:
        with HarborEnv(base_url=LOCAL, websocket_ping_interval_s=None,
                       websocket_ping_timeout_s=None) as client:
            result = client.run_rollout(split=split_spec("blackbox", split), task_index=int(index),
                        harness=harness, sandbox="daytona", llm_url=url.strip(), model=model.strip(),
                        api_key=key, agent_timeout_sec=600, agent_step_limit=17)
        summary = {"reward": result.reward, "ok": result.ok, "seconds": result.wall_s,
                   "model_calls": result.n_turns, "captured_training_tokens": result.n_trainable_tokens,
                   "capture_level": result.capture_level, "trial": result.trial_name}
        # Human-readable captured messages only. Credentials and raw prompt-token arrays stay out of UI.
        turns = result.model_dump().get("conversations", [])
        yield json.dumps(turns, ensure_ascii=False, indent=2)[-120000:] or "Rollout finished", summary
    except Exception as exc:
        raise gr.Error(f"Rollout failed ({type(exc).__name__}); retry after checking the endpoint.") from None


def mount_ui(app, arm):
    from openenv.core.env_server.gradio_theme import OPENENV_GRADIO_CSS, OPENENV_GRADIO_THEME
    import provider_demo
    use_hf = provider_demo.enabled()
    if use_hf:
        from inference_providers import mount_provider_relay
        app = mount_provider_relay(app)
    title = TITLES[arm]
    description = ("Explore the data, run tools, and submit your answer in an isolated workspace."
                   if arm == "whitebox" else
                   "Connect a model, choose an agent harness, and watch it solve a data task.")
    capability = "Bash + SETA tools" if arm == "whitebox" else "4 agent harnesses"
    with gr.Blocks(title=title) as demo:
        gr.HTML(f'<section id="agent-hero"><span class="agent-eyebrow">HUGGINGENVS / DATA AGENT</span>'
                f'<h1>{title.removeprefix("Data Agent ").removesuffix(" Env")}</h1><p>{description} '
                'The same environment powers training and evaluation.</p><div class="agent-chips">'
                '<span class="agent-chip">1,000 training tasks</span><span class="agent-chip">250 test tasks</span>'
                f'<span class="agent-chip">{capability}</span><span class="agent-chip">Isolated workspaces</span>'
                '</div></section>')
        with gr.Tab("Explore the environment"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=4, min_width=310, elem_id="task-panel"):
                    gr.Markdown("### 1. Choose a task")
                    with gr.Row():
                        split = gr.Dropdown([("Test · 250 tasks", "test"), ("Train · 1,000 tasks", "train")],
                                            value="test", label="Dataset")
                        index = gr.Number(value=0, minimum=0, maximum=999, precision=0, label="Task index", info="0–249")
                    with gr.Row():
                        inspect = gr.Button("Load task", variant="primary")
                        random = gr.Button("Random task")
                    badge = gr.Markdown("Load a task to see its difficulty.", elem_id="task-badge")
                    instruction = gr.Textbox(label="Your task", lines=18, max_lines=30,
                                             interactive=False, elem_id="task-instructions")
                    with gr.Accordion("Task details", open=False):
                        metadata = gr.JSON(label="Task metadata")
                with gr.Column(scale=6, min_width=400, elem_id="workspace-panel"):
                    if arm == "whitebox":
                        ui = WhiteboxUI()
                        gr.Markdown("### 2. Work with the data\nStart a workspace, inspect the files, and use the tools below.")
                        with gr.Row():
                            begin = gr.Button("Start workspace", variant="primary")
                            close = gr.Button("Close workspace")
                        state = gr.Textbox(label="Workspace status", value="Not started", interactive=False)
                        tool = gr.Dropdown(["bash", "read", "write", "edit", "grep", "glob", "ls"],
                                           value="bash", label="Tool")
                        command = gr.Textbox(value="ls -la /workdir", label="Shell command", lines=3)
                        path = gr.Textbox(value="/workdir", label="File or directory path", visible=False)
                        content = gr.Textbox(label="File content / replacement text", lines=5, visible=False)
                        old = gr.Textbox(label="Text to replace", visible=False)
                        execute = gr.Button("Run tool", variant="primary")
                        output = gr.Code(label="Output", language=None, interactive=False, lines=16, elem_id="tool-console")
                        gr.Markdown("### 3. Submit your answer")
                        answer = gr.Textbox(label="Final answer", placeholder="Enter your answer, or leave blank to grade answer.txt.")
                        submit = gr.Button("Submit and grade", variant="primary")
                        gr.Markdown("Your workspace is isolated from other users. Close it when finished; idle sessions expire after 20 minutes.",
                                    elem_classes="agent-footnote")
                        tool.change(tool_inputs, tool, [command, path, content, old], api_visibility="private")
                        begin.click(ui.start, [split, index], [instruction, output, state], api_name="start_task")
                        execute.click(ui.run, [tool, command, path, content, old], output, api_name="run_tool")
                        submit.click(ui.grade, answer, [output, state], api_name="grade_task")
                        close.click(ui.close, outputs=[output, state], api_name="close_task")
                        def unload(request: gr.Request):
                            ui.close(request)
                        demo.unload(unload)
                        if use_hf:
                            with gr.Accordion("Let a model solve this task", open=False):
                                hf_provider, hf_model = provider_demo.controls()
                                hf_run = gr.Button("Run model on this task", variant="primary")
                                hf_summary = gr.JSON(label="Agent result")
                                hf_transcript = gr.Code(label="Agent conversation", language="json", lines=14)
                                def run_hf_whitebox(s, i, p, m, oauth_token: gr.OAuthToken, request: gr.Request):
                                    yield from provider_demo.whitebox(ui, s, i, p, m, oauth_token, request)
                                hf_run.click(run_hf_whitebox, [split, index, hf_provider, hf_model],
                                    [hf_transcript, hf_summary], concurrency_limit=4,
                                    concurrency_id="interactive-agents", api_name="run_hf_agent")
                    else:
                        gr.Markdown("### 2. Connect your agent\nUse an OpenAI-compatible model endpoint to run a task.")
                        with gr.Row():
                            harness = gr.Dropdown([("OpenCode", "opencode"), ("Claude Code", "claude-code"),
                                                   ("Codex", "codex"), ("Mini-SWE-Agent", "mini-swe-agent")],
                                                  value="opencode", label="Agent harness")
                            model = gr.Textbox(value="Qwen/Qwen3.5-2B", label="Model name")
                        url = gr.Textbox(label="Inference endpoint", placeholder="https://your-endpoint/v1")
                        key = gr.Textbox(label="Inference API key", type="password", placeholder="Only if your endpoint requires authentication")
                        run = gr.Button("Run agent on this task", variant="primary")
                        gr.Markdown("### 3. Inspect the result")
                        summary = gr.JSON(label="Score and run statistics")
                        with gr.Accordion("Agent conversation", open=True):
                            transcript = gr.Code(label="Conversation", language="json", interactive=False, lines=20)
                        run.click(blackbox_run, [split, index, harness, url, model, key], [transcript, summary],
                                  concurrency_limit=4, api_name="run_agent")
                        if use_hf:
                            with gr.Accordion("Use your Hugging Face account", open=True):
                                hf_provider, hf_model = provider_demo.controls()
                                hf_run = gr.Button("Run with HF Inference Providers", variant="primary")
                                hf_run.click(provider_demo.blackbox,
                                    [split, index, harness, hf_provider, hf_model], [transcript, summary],
                                    concurrency_limit=4, concurrency_id="interactive-agents", api_name="run_hf_agent")
            def load(s, i):
                return preview(arm, s, i)
            inspect.click(load, [split, index], [instruction, metadata], api_name="preview_task").then(
                task_badge, metadata, badge, api_visibility="private")
            split.change(lambda s: gr.update(value=0, info="0–999" if s == "train" else "0–249"),
                         split, index, api_visibility="private").then(load, [split, index], [instruction, metadata],
                         api_visibility="private").then(task_badge, metadata, badge, api_visibility="private")
            random.click(lambda s: secrets.randbelow(1000 if s == "train" else 250), split, index,
                         api_visibility="private").then(load, [split, index], [instruction, metadata], api_visibility="private").then(
                         task_badge, metadata, badge, api_visibility="private")
            demo.load(lambda: preview(arm, "test", 0), outputs=[instruction, metadata], api_visibility="private").then(
                task_badge, metadata, badge, api_visibility="private")
        with gr.Tab("Training & evaluation API"):
            name = "harbor_env" if arm == "blackbox" else "white_box_bash"
            gr.Markdown(f"### One environment, both datasets\nUse this Space's endpoint for training and evaluation. "
                        "Select the task split per request; run model inference on your own endpoint.\n\n"
                        f"| Task operation | Route |\n| --- | --- |\n| Available datasets | `GET /{name}/splits` |\n"
                        f"| Number of tasks | `POST /{name}/num_tasks` |\n| Task instructions | `POST /{name}/task` |\n"
                        f"| Task range | `POST /{name}/task_range` |\n\n"
                        "The native OpenEnv MCP client executes environment actions. Training and evaluation "
                        "share the task catalog while using separate model endpoints and sandbox sessions.")
    demo.queue(max_size=1024, default_concurrency_limit=8)
    return gr.mount_gradio_app(app, demo, path="/", theme=OPENENV_GRADIO_THEME, css=OPENENV_GRADIO_CSS + UI_CSS)
