"""Explore, solve and score a SmolDataEnvs task in the browser."""

import html
import logging

import gradio as gr

from .catalog import tasks
from .playground import PlaygroundSession, close_session

log = logging.getLogger(__name__)
HEADER = """
<style>
.gradio-container .smol-panel {
  padding:20px !important; border:1px solid var(--border-color-primary) !important;
  border-radius:14px !important; background:var(--block-background-fill) !important;
}
.gradio-container .smol-panel > .smol-panel { padding:0 !important; border:0 !important; }
.gradio-container .smol-panel .styler {
  background:transparent !important; display:flex; flex-direction:column; gap:14px;
}
.gradio-container .smol-panel .form { background:transparent !important; }
.gradio-container .smol-panel button { border-radius:8px !important; min-height:40px; }
.gradio-container .smol-panel h3 { font-size:18px; margin:0; }
.gradio-container .smol-panel .wrap:has(> textarea) { border-radius:8px !important; }
</style>
<div class="hero">
  <div class="eyebrow">FINEENVS / INTERACTIVE ENVIRONMENT</div>
  <h1>Small data. Real problems.</h1>
  <p>Explore a dataset, run a little code, and see how your answer scores.</p>
  <div class="tags"><span>SETA whitebox</span><span>Your own sandbox</span>
  <span>The same tools used in training</span></div>
</div>
"""
HEADER_CSS = """
.hero { background:#111e32; color:#f7fafc; border-radius:16px !important; padding:30px 32px; margin-bottom:12px; }
.eyebrow { color:#90bfff; font-size:11px; font-weight:650; letter-spacing:1.7px; }
.hero h1 { color:#fff; margin:12px 0 8px; font-size:34px; font-weight:650; }
.hero p { color:#c3cfdf; margin:0 0 20px; font-size:16px; }
.tags { display:flex; flex-wrap:wrap; gap:8px; }
.tags span { border:1px solid #42516a; padding:5px 10px; border-radius:30px !important; font-size:12px; color:#d6e6ff; }
@media (max-width:600px) { .hero { padding:22px; } .hero h1 { font-size:27px; } }
"""
CARD_CSS = """
.cards { display:flex; gap:12px; margin:8px 0 16px; flex-wrap:wrap; }
.card { flex:1; min-width:100px; border:1px solid var(--border-color-primary); border-radius:12px !important; padding:16px 18px; background:var(--block-background-fill); }
.card .label { font-size:12px; color:var(--body-text-color-subdued); }
.card .value { font-size:25px; font-weight:650; color:var(--body-text-color); margin-top:4px; }
.trace { border:1px solid var(--border-color-primary); border-radius:10px !important; margin:8px 0; padding:12px; }
.trace summary { cursor:pointer; font-weight:600; }
.trace pre { white-space:pre-wrap; overflow-wrap:anywhere; font-size:12px; }
.empty { padding:16px; color:var(--body-text-color-subdued); }
"""


def task_question(row):
    text = row["instruction"]
    return (
        text.partition("Question:")[2].split("\n\n")[0].strip() or text.splitlines()[0]
    )


def task_choices(split, difficulty):
    return [
        (
            f"{i + 1:03d} · {row['difficulty'].title()} · {task_question(row)[:90]}",
            row["name"],
        )
        for i, row in enumerate(tasks(split))
        if difficulty == "All" or row["difficulty"] == difficulty.lower()
    ]


def preview(split, name):
    if not name:
        return "No tasks in this filter. Try another difficulty."
    row = next(t for t in tasks(split) if t["name"] == name)
    files = (
        row["instruction"]
        .partition("Files (in /home/user/input, no subfolders):")[2]
        .split("\n\n")[0]
        .strip()
    )
    return f"**{row['difficulty'].title()} · {split.title()} split**\n\n{task_question(row)}\n\n**Input files**\n{files or 'Use List files to see the inputs.'}\n\nFiles are in `/home/user/input`. Run your code in `/workdir`."


def instructions(split, name):
    return next(
        (row["instruction"] for row in tasks(split) if row["name"] == name),
        "Choose a task first.",
    )


def metrics(session):
    values = [
        ("Tool calls", str(session.environment._calls)),
        (
            "Correctness",
            "Not graded"
            if session.correctness is None
            else f"{session.correctness:.0%}",
        ),
        (
            "Total reward",
            "Not graded" if session.reward is None else f"{session.reward:.3f}",
        ),
    ]
    return (
        '<div class="cards">'
        + "".join(
            f'<div class="card"><div class="label">{label}</div><div class="value">{value}</div></div>'
            for label, value in values
        )
        + "</div>"
    )


def trace(session):
    if not session.history:
        return (
            '<div class="empty">Your commands and their outputs will appear here.</div>'
        )
    return "".join(
        f'<details class="trace"><summary>Command {i + 1} · {item["seconds"]:.1f}s</summary><pre>$ {html.escape(item["command"])}</pre><pre>{html.escape(item["output"])}</pre></details>'
        for i, item in enumerate(session.history)
    )


def build_ui(manager, fields, metadata, is_chat, title, quick_start):
    choices = task_choices("test", "All")
    first = choices[0][1] if choices else None
    with gr.Blocks(
        title="SmolDataEnvs Multi-harness | SETA Whitebox", analytics_enabled=False
    ) as demo:
        session = gr.State(
            PlaygroundSession(), time_to_live=900, delete_callback=close_session
        )
        gr.HTML(HEADER, css_template=HEADER_CSS, apply_default_css=False)
        gr.Markdown(
            "**01 Choose a task** → **02 Explore the data** → **03 Submit and score**"
        )
        scores = gr.HTML(
            metrics(PlaygroundSession()), css_template=CARD_CSS, apply_default_css=False
        )
        with gr.Row():
            with gr.Column(scale=4, min_width=300):
                with gr.Group(elem_classes=["smol-panel"]):
                    gr.Markdown("### 01 / Your task")
                    with gr.Row():
                        split = gr.Dropdown(
                            ["test", "train"],
                            value="test",
                            label="Dataset split",
                            interactive=True,
                        )
                        difficulty = gr.Dropdown(
                            ["All", "Easy", "Medium", "Hard"],
                            value="All",
                            label="Difficulty",
                            interactive=True,
                        )
                    picker = gr.Dropdown(
                        choices, value=first, label="Choose a task", interactive=True
                    )
                    start = gr.Button(
                        "Start this task", variant="primary", elem_id="start-task"
                    )
                    question = gr.Markdown(preview("test", first))
                    with gr.Accordion("Full agent instructions", open=False):
                        full_instructions = gr.Markdown(instructions("test", first))
                with gr.Accordion("How scoring works", open=False):
                    gr.Markdown(
                        "Correctness comes from the task's held-out grader. A correct answer gets a small bonus for fewer tool calls.\n\n"
                        "`reward = correctness × (1 + 1.5 / (15 + tool calls))`\n\n"
                        "The bonus applies only after at least one tool call. Submitting your answer does not count as a tool call. A grading failure is shown as unscored.\n\n"
                        "You have 10 minutes per task. Starting a new task releases the previous sandbox."
                    )
            with gr.Column(scale=6, min_width=340):
                with gr.Group(elem_classes=["smol-panel"]):
                    gr.Markdown("### 02 / Explore the data")
                    status = gr.Markdown(
                        "Start a task to create your sandbox. No model or API key needed."
                    )
                    with gr.Row():
                        list_files = gr.Button("List files", size="sm")
                        inspect_data = gr.Button("Preview a CSV", size="sm")
                    command = gr.Code(
                        "ls -lh /home/user/input",
                        language="shell",
                        label="Bash command",
                        lines=6,
                    )
                    with gr.Row():
                        run = gr.Button(
                            "Run command",
                            variant="primary",
                            interactive=False,
                            elem_id="run-command",
                        )
                        stop = gr.Button("Release sandbox", interactive=False)
                    output = gr.Textbox(
                        label="Command output",
                        value="Start a task to open your workspace.",
                        lines=7,
                        max_lines=14,
                        interactive=False,
                    )
                with gr.Group(elem_classes=["smol-panel"]):
                    gr.Markdown("### 03 / Submit your answer")
                    answer = gr.Textbox(
                        label="Final answer",
                        placeholder="Write the answer itself, not a command.",
                        lines=2,
                    )
                    grade = gr.Button(
                        "Submit and score",
                        variant="primary",
                        interactive=False,
                        elem_id="grade-answer",
                    )
        with gr.Accordion("Execution history", open=False):
            history = gr.HTML(
                trace(PlaygroundSession()),
                css_template=CARD_CSS,
                apply_default_css=False,
            )
        with gr.Accordion("Use this environment in code", open=False):
            gr.Markdown(
                "This playground uses the same `BashEnvironment` as the whitebox training script. Each browser session has its own sandbox.\n\n"
                "[Source and local setup](https://huggingface.co/spaces/FineEnvs/smoldataenv-multi-harness-whitebox/tree/main) · "
                "[Training tutorial](https://github.com/adithya-s-k/FineEnvs/tree/main/05-multi-harness-rl) · "
                "[SmolDataEnvs collection](https://huggingface.co/collections/FineEnvs/smoldataenvs-6ab4f2f6e09b7cb872ebc867)"
            )

        def render(state):
            return (
                state,
                state.status,
                state.output,
                metrics(state),
                trace(state),
                *[gr.update(interactive=state.active)] * 3,
                *[gr.update(interactive=not state.active)] * 4,
            )

        def act(state, action, *args):
            try:
                getattr(state, action)(*args)
                if action == "close":
                    state.status = "Sandbox released. Choose a task to start again."
            except ValueError as exc:
                gr.Warning(str(exc))
            except Exception:
                log.exception("Whitebox playground action failed: %s", action)
                state.status = (
                    "The sandbox operation failed. Release it and start the task again."
                )
                if not state.active:
                    state.output = "No active sandbox."
            return render(state)

        def choose(split_name, level):
            rows = task_choices(split_name, level)
            name = rows[0][1] if rows else None
            return (
                gr.update(choices=rows, value=name),
                preview(split_name, name),
                instructions(split_name, name),
                gr.update(interactive=bool(rows)),
            )

        outputs = [
            session,
            status,
            output,
            scores,
            history,
            run,
            grade,
            stop,
            split,
            difficulty,
            picker,
            start,
        ]
        split.change(
            choose, [split, difficulty], [picker, question, full_instructions, start]
        )
        difficulty.change(
            choose, [split, difficulty], [picker, question, full_instructions, start]
        )
        picker.change(
            lambda s, n: (preview(s, n), instructions(s, n)),
            [split, picker],
            [question, full_instructions],
        )
        start.click(
            lambda s, p, n: (*act(s, "start", p, n), ""),
            [session, split, picker],
            outputs + [answer],
            concurrency_limit=None,
        )
        run.click(
            lambda s, c: act(s, "run", c),
            [session, command],
            outputs,
            concurrency_limit=None,
        )
        grade.click(
            lambda s, a: act(s, "grade", a),
            [session, answer],
            outputs,
            concurrency_limit=None,
        )
        stop.click(lambda s: act(s, "close"), session, outputs, concurrency_limit=None)
        list_files.click(lambda: "ls -lh /home/user/input", outputs=command)
        inspect_data.click(
            lambda: (
                "python3 - <<'PY'\nfrom pathlib import Path\nimport pandas as pd\np = next(Path('/home/user/input').glob('*.csv'))\ndf = pd.read_csv(p)\nprint(p.name, df.shape)\nprint(df.head().to_string(index=False))\nPY"
            ),
            outputs=command,
        )
    return demo
