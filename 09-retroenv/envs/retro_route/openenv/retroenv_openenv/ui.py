"""Browser playground at /web: pick a task, call the tools by hand, submit routes.

Each browser session owns one RetroRouteEnvironment, and every call goes
through the same MCP step path an agent uses, so scores match training and
evaluation exactly.
"""

from __future__ import annotations

import json
import threading
from typing import Any

import gradio as gr
from openenv.core.env_server.mcp_types import CallToolAction

from . import render
from .client import _tool_payload
from .config import shared_resources
from .environment import RetroRouteEnvironment


def _arguments_template(tool: str, target: str) -> dict[str, Any]:
    return {
        "inspect_molecule": {"smiles": target},
        "pubchem_lookup": {"query": target},
        "stock_retrieve": {"query": target, "mode": "exact", "limit": 10},
        "reaction_precedent_search": {"product_smiles": target, "limit": 5},
        "validate_disconnection": {"product_smiles": target, "reactants": ["", ""]},
        "reaction_class_lookup": {"product_smiles": target, "reactants": ["", ""]},
        "reaction_conditions_search": {"product_smiles": target, "reactants": [], "limit": 5},
        "search_literature": {"product_smiles": target, "limit": 5},
    }.get(tool, {})


def _submission_template(target: str) -> dict[str, Any]:
    return {
        "routes": [
            {
                "type": "mol",
                "smiles": target,
                "in_stock": False,
                "children": [
                    {
                        "type": "reaction",
                        "is_reaction": True,
                        "metadata": {
                            "explanation": "",
                            "reaction_class": "",
                            "confidence": 0.5,
                            "literature": [],
                            "precursor_roles": {},
                        },
                        "children": [{"type": "mol", "smiles": "REACTANT_SMILES", "in_stock": True, "children": []}],
                    }
                ],
            }
        ]
    }


def _summary(result: Any) -> str:
    if not isinstance(result, dict):
        return str(result)
    if "error" in result:
        return f"error: {result['error']}"
    if "returned" in result:
        return f"{result['returned']} result(s)"
    if "valid" in result:
        return f"valid={result['valid']} support={result.get('support')}"
    if "score" in result:
        return f"reward={result['score'].get('reward')}"
    return ", ".join(sorted(result))[:160]


class PlaygroundSession:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.env: RetroRouteEnvironment | None = None
        self.opening: dict[str, Any] | None = None
        self.history: list[dict[str, Any]] = []
        self.submission: Any = None
        self.score: dict[str, Any] | None = None

    def __deepcopy__(self, memo: dict) -> "PlaygroundSession":
        # Gradio copies the initial state per browser session; start each one empty.
        return type(self)()

    def start(self, split: str, index: int) -> None:
        with self.lock:
            self.close()
            self.env = RetroRouteEnvironment()
            self.opening = self.env.reset(split=split, index=int(index)).metadata
            self.history, self.submission, self.score = [], None, None

    def close(self) -> None:
        with self.lock:
            if self.env is not None:
                self.env.close()
                self.env = None

    def call(self, tool: str, arguments: dict[str, Any]) -> Any:
        with self.lock:
            if self.env is None:
                raise gr.Error("Start an episode first.")
            observation = self.env.step(CallToolAction(tool_name=tool, arguments=arguments))
            error = getattr(observation, "error", None)
            result = {"error": error.message} if error else _tool_payload(observation.result)
            self.history.append(
                {"tool": tool, "arguments": json.dumps(arguments, sort_keys=True), "summary": _summary(result)}
            )
            if tool == "emit_routes" and isinstance(result, dict) and result.get("score"):
                self.submission, self.score = arguments.get("submission"), result["score"]
            return result


def build_ui(*_: Any, **__: Any) -> gr.Blocks:
    store = shared_resources().store
    splits = store.splits()
    default_split = "test_id" if "test_id" in splits else splits[0]

    def task_choices(split: str) -> list[tuple[str, int]]:
        return [
            (
                f"{i:03d} · {task.variant} · depth ≤ {task.max_depth} · {task.min_routes} route{'s' if task.min_routes > 1 else ''} · "
                f"{task.target_smiles[:48]}",
                i,
            )
            for i, task in enumerate(store.tasks(split))
        ]

    def tool_names(session: PlaygroundSession) -> list[str]:
        names = list(session.opening["tools"]) if session.opening else []
        return [name for name in names if name != "emit_routes"]

    def panels(session: PlaygroundSession) -> tuple[str, str, str, str]:
        state = session.env.state if session.env else None
        return (
            render.target_panel(session.opening, state),
            render.history_panel(session.history),
            render.score_panel(session.score),
            render.routes_panel(session.submission),
        )

    with gr.Blocks(title="RetroEnv") as demo:
        session = gr.State(PlaygroundSession())
        gr.HTML(
            render.STYLE
            + render.wrap(
                "<h2 style='margin:0'>RetroEnv</h2><p class='muted' style='margin:2px 0 0'>Plan routes back to "
                "purchasable molecules. These are the tools and the verifier that agents use. "
                "API: <a href='/docs'>/docs</a> · tasks: <a href='/retro_route/splits'>/retro_route/splits</a></p>"
            ),
            apply_default_css=False,
        )
        with gr.Row(equal_height=False):
            with gr.Column(scale=4, min_width=320):
                split = gr.Dropdown(splits, value=default_split, label="Split")
                task = gr.Dropdown(task_choices(default_split), value=0, label="Task")
                start = gr.Button("Start episode", variant="primary")
                target = gr.HTML(render.target_panel(None), apply_default_css=False)
            with gr.Column(scale=7, min_width=360):
                with gr.Tab("Tools"):
                    tool = gr.Dropdown([], label="Tool", interactive=True)
                    arguments = gr.Code("{}", language="json", label="Arguments", lines=6)
                    run = gr.Button("Run tool")
                    result = gr.Code("", language="json", label="Result", lines=12, interactive=False)
                with gr.Tab("Submit"):
                    submission = gr.Code("", language="json", label="emit_routes submission", lines=16)
                    emit = gr.Button("Emit routes and score", variant="primary")
                    score = gr.HTML(render.score_panel(None), apply_default_css=False)
                    routes = gr.HTML(render.routes_panel(None), apply_default_css=False)
                with gr.Tab("History"):
                    history = gr.HTML(render.history_panel([]), apply_default_css=False)

        def on_split(name: str):
            return gr.update(choices=task_choices(name), value=0)

        def on_start(state: PlaygroundSession, split_name: str, index: int):
            state.start(split_name, index)
            names = tool_names(state)
            target_smiles = state.opening["target_smiles"]
            first = names[0] if names else None
            return (
                state,
                *panels(state),
                gr.update(choices=names, value=first),
                json.dumps(_arguments_template(first, target_smiles), indent=2) if first else "{}",
                "",
                json.dumps(_submission_template(target_smiles), indent=2),
            )

        def on_tool(state: PlaygroundSession, name: str):
            if not state.opening or not name:
                return "{}"
            return json.dumps(_arguments_template(name, state.opening["target_smiles"]), indent=2)

        def on_run(state: PlaygroundSession, name: str, text: str):
            try:
                args = json.loads(text or "{}")
            except json.JSONDecodeError as exc:
                raise gr.Error(f"Arguments are not valid JSON: {exc}") from exc
            value = state.call(name, args)
            return (state, json.dumps(value, indent=2, sort_keys=True), *panels(state))

        def on_emit(state: PlaygroundSession, text: str):
            try:
                value = json.loads(text or "{}")
            except json.JSONDecodeError as exc:
                raise gr.Error(f"Submission is not valid JSON: {exc}") from exc
            state.call("emit_routes", {"submission": value})
            return (state, *panels(state))

        split.change(on_split, split, task)
        start.click(
            on_start,
            [session, split, task],
            [session, target, history, score, routes, tool, arguments, result, submission],
        )
        tool.change(on_tool, [session, tool], arguments)
        run.click(on_run, [session, tool, arguments], [session, result, target, history, score, routes])
        emit.click(on_emit, [session, submission], [session, target, history, score, routes])
    return demo
