"""The "Try Environment" tab of OpenEnv's web UI: play an episode by hand.

Same shape as the other FineEnvs environments (GeoGuesser, DesignGym): reset(split=) and reset(index=) on top, then
the environment itself. The embedded page (`/viewer/#/play`) is a UI over the env's own interfaces only: the Task API
to pick a task, and an OpenEnv WebSocket session (`/ws`) for reset and the MCP tools (get_situation, check_plan,
submit_plan), so the reward a person sees is the rubric's. OpenEnv's own Playground tab stays next to it. Model
rollouts and the eval live in a separate Space (FineEnvs/PortSimEnv-Eval).
"""

from __future__ import annotations

import html
import random
from urllib.parse import quote

import gradio as gr

from berth_core import TaskPack

DEFAULT_SPLIT = "eval"
DEFAULT_TASK = "dock-24B-w07x1-busy-0"


def _iframe(task_id: str, split: str) -> str:
    src = f"/viewer/?embed=1#/play?split={quote(split)}&task={quote(task_id)}&start=1"
    return (
        # OpenEnv's page caps the container width; the editor wants the full window.
        "<style>.gradio-container .main.fillable{max-width:none!important;padding-left:12px!important;"
        "padding-right:12px!important}.gradio-container .html-container:has(iframe){padding:0!important}</style>"
        f'<iframe src="{html.escape(src)}" title="PortSimEnv v1: {html.escape(task_id)}" '
        'style="width:100%;height:calc(100vh - 210px);min-height:720px;border:0;display:block"></iframe>'
    )


def build_ui(pack: TaskPack) -> gr.Blocks:
    ids = {s: [t.task_id for t in pack.tasks if t.split == s] for s in pack.splits()}
    names = [s for s in (DEFAULT_SPLIT, "train") if ids.get(s)] + [s for s in ids if s not in (DEFAULT_SPLIT, "train")]
    split0 = names[0]
    index0 = ids[split0].index(DEFAULT_TASK) if DEFAULT_TASK in ids[split0] else 0

    def _label(split: str) -> str:
        return f"reset(index=)  ·  0 to {len(ids[split]) - 1}"

    def _frame(index, split):
        split = split if split in ids else split0
        i = min(max(int(index or 0), 0), len(ids[split]) - 1)
        return _iframe(ids[split][i], split)

    with gr.Blocks(title="PortSimEnv v1") as ui:
        with gr.Row():
            split_box = gr.Dropdown(choices=names, value=split0, label="reset(split=)", scale=1,
                                    interactive=len(names) > 1)
            task_box = gr.Number(value=index0, minimum=0, maximum=len(ids[split0]) - 1, step=1, precision=0,
                                 label=_label(split0), scale=2)
            load_button = gr.Button("load episode", variant="primary", scale=1)
            random_button = gr.Button("random episode", scale=1)
        frame = gr.HTML(value=_frame(index0, split0), show_label=False)

        def _on_split(split: str):
            """Re-range the index box so it cannot address a missing task."""
            split = split if split in ids else split0
            return gr.update(maximum=len(ids[split]) - 1, value=0, label=_label(split))

        split_box.change(fn=_on_split, inputs=split_box, outputs=task_box)
        load_button.click(fn=_frame, inputs=[task_box, split_box], outputs=frame)
        random_button.click(fn=lambda split: _frame(random.randrange(len(ids.get(split) or ids[split0])), split),
                            inputs=split_box, outputs=frame)
    return ui
