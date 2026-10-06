"""The "Berth planner" tab of OpenEnv's web UI: play an episode by hand.

The embedded page (`/viewer/#/play`) is a UI over the env's own interfaces only: the Task API to pick a task, and an
OpenEnv WebSocket session (`/ws`) for reset and the MCP tools (get_situation, check_plan, submit_plan), so the
reward a person sees is the rubric's. Model rollouts and reference plans are not on it; they live in the Explorer
at `/viewer/`. OpenEnv's own Playground tab stays next to it.
"""

from __future__ import annotations

import gradio as gr

from berth_core import TaskPack


def build_ui(pack: TaskPack) -> gr.Blocks:
    with gr.Blocks(title="PortSimEnv v1") as ui:
        gr.HTML(
            # OpenEnv's page caps the container width; the viewer wants the full window.
            "<style>.gradio-container .main.fillable{max-width:none!important;padding-left:12px!important;"
            "padding-right:12px!important}.gradio-container .html-container:has(iframe){padding:0!important}</style>"
            '<iframe src="/viewer/?embed=1#/play" title="PortSimEnv v1" '
            'style="width:100%;height:calc(100vh - 140px);min-height:720px;border:0;display:block"></iframe>'
        )
    return ui
