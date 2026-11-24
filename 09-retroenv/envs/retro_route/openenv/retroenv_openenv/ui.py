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
from retroenv.chemistry import ChemistryError, canonicalize_smiles
from retroenv.disconnections import describe, strategic_disconnections

from . import render
from .client import _tool_payload
from .config import shared_resources
from .environment import RetroRouteEnvironment

# Tasks the Task dropdown lists at once: every task of the evaluation splits; train (75k) is searched.
TASK_PAGE = 1500


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
    if "supported" in result:
        return f"supported={result['supported']} support={result.get('support')}"
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
        # The route being assembled by hand: product -> precursors, plus what the
        # stock tool has revealed so far. Both drive the Plan tab.
        self.plan: dict[str, tuple[str, ...]] = {}
        self.stock_known: dict[str, bool] = {}

    def __deepcopy__(self, memo: dict) -> "PlaygroundSession":
        # Gradio copies the initial state per browser session; start each one empty.
        return type(self)()

    def start(self, split: str, index: int) -> None:
        with self.lock:
            self.close()
            self.env = RetroRouteEnvironment()
            self.opening = self.env.reset(split=split, index=int(index)).metadata
            self.history, self.submission, self.score = [], None, None
            self.plan, self.stock_known = {}, {}

    def frontier(self) -> list[str]:
        """Molecules still to disconnect: the target, then any precursor neither cut nor known in stock."""
        if not self.opening:
            return []
        out: list[str] = []
        queue, seen = [canonicalize_smiles(self.opening["target_smiles"])], set()
        while queue:
            smiles = queue.pop(0)
            if smiles in seen:
                continue
            seen.add(smiles)
            precursors = self.plan.get(smiles)
            if precursors:
                queue.extend(precursors)
            elif not self.stock_known.get(smiles):
                out.append(smiles)
        return out

    def build_submission(self) -> dict[str, Any]:
        """The plan as an emit_routes submission, claiming stock only where the stock tool confirmed it."""
        target = canonicalize_smiles(self.opening["target_smiles"]) if self.opening else ""

        def node(smiles: str, depth: int = 0) -> dict[str, Any]:
            precursors = self.plan.get(smiles) if depth <= 12 else None
            if not precursors:
                return {
                    "type": "mol",
                    "smiles": smiles,
                    "in_stock": self.stock_known.get(smiles, False),
                    "children": [],
                }
            reaction = {
                "type": "reaction",
                "is_reaction": True,
                "metadata": {
                    "explanation": "",
                    "reaction_class": "",
                    "confidence": 0.5,
                    "literature": [],
                    "precursor_roles": {},
                },
                "children": [node(p, depth + 1) for p in precursors],
            }
            return {"type": "mol", "smiles": smiles, "in_stock": False, "children": [reaction]}

        return {"routes": [node(target)]}

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
            if tool == "stock_retrieve" and isinstance(result, dict):
                self._note_stock(arguments, result)
            if tool == "emit_routes" and isinstance(result, dict) and result.get("score"):
                self.submission, self.score = arguments.get("submission"), result["score"]
            return result

    def _note_stock(self, arguments: dict[str, Any], result: dict[str, Any]) -> None:
        """Everything the stock returns is in stock; an exact miss proves the query is not."""
        for row in result.get("results") or []:
            self.stock_known[str(row["smiles"])] = True
        if result.get("mode") in {"exact", "inchikey"} and not result.get("results"):
            try:
                self.stock_known[canonicalize_smiles(str(arguments.get("query", "")))] = False
            except ChemistryError:
                pass


def build_ui(*_: Any, **__: Any) -> gr.Blocks:
    resources = shared_resources()
    store = resources.store
    snapshot_path = resources.benchmark.root / "corpus-manifest.json"
    snapshot = json.loads(snapshot_path.read_text()) if snapshot_path.exists() else None
    credits = render.credits_panel(resources.benchmark.manifest.get("source"), snapshot)
    splits = store.splits()
    default_split = "test_id" if "test_id" in splits else splits[0]

    def task_start(split: str, query: str) -> int | None:
        """The index a task number or task id names in ``split``; None when there is none."""
        query = query.strip()
        if query.isdigit():
            return int(query) if int(query) < len(store.tasks(split)) else None
        return store.position(split, query) if query else 0

    def task_choices(split: str, start: int = 0) -> list[tuple[str, int]]:
        """One page of the split from ``start``.

        Listing all 75k train tasks took seconds to build and to send to the browser; a page holds
        every task of the evaluation splits, and Go to task reaches the rest.
        """
        tasks = store.tasks(split)
        width = max(3, len(str(len(tasks) - 1)))
        return [
            (
                f"{i:0{width}d} · {task.variant} · depth ≤ {task.max_depth} · {task.min_routes} "
                f"route{'s' if task.min_routes > 1 else ''} · {task.target_smiles[:48]}",
                i,
            )
            for i, task in enumerate(tasks[start : start + TASK_PAGE], start)
        ]

    def task_info(split: str) -> str:
        count = len(store.tasks(split))
        return f"{count:,} tasks" + (f"; {TASK_PAGE:,} listed at a time" if count > TASK_PAGE else "")

    def tool_names(session: PlaygroundSession) -> list[str]:
        names = list(session.opening["tools"]) if session.opening else []
        return [name for name in names if name != "emit_routes"]

    def panels(
        session: PlaygroundSession, keep: str = "", keep_cut: str = ""
    ) -> tuple[str, str, str, str, str, Any, Any, Any]:
        """Every live panel. ``keep`` holds the molecule under study while it still needs cutting,
        and ``keep_cut`` the proposed cut, so checking one does not make you choose it again."""
        state = session.env.state if session.env else None
        target = canonicalize_smiles(session.opening["target_smiles"]) if session.opening else ""
        frontier = session.frontier()
        chosen = keep if keep in frontier else (frontier[0] if frontier else None)
        cuts = cut_choices(chosen or "")
        still_offered = any(keep_cut == value for _, value in cuts)
        return (
            render.target_panel(session.opening, state),
            render.history_panel(session.history),
            render.score_panel(session.score),
            render.routes_panel(session.submission),
            render.plan_panel(target, session.plan, session.stock_known),
            gr.update(choices=frontier, value=chosen),
            # Gradio fires .change only on user input, so the cuts are refreshed here too.
            gr.update(choices=cuts, value=keep_cut if still_offered else None),
            # Redrawing an unchanged molecule would reload its 3D view on every tool call.
            render.molecule_views(chosen) if chosen != keep else gr.update(),
        )

    def cut_choices(product: str) -> list[tuple[str, str]]:
        """Label each proposed cut with the forward reaction; the value carries the precursors."""
        if not product:
            return []
        out = []
        for cut in strategic_disconnections(product)[:8]:
            phrase = describe(cut.family, cut.reactants, product)
            out.append((f"{cut.bond} · {phrase}", ".".join(cut.reactants)))
        return out

    with gr.Blocks(title="RetroEnv") as demo:
        session = gr.State(PlaygroundSession())
        gr.HTML(
            render.STYLE
            + render.wrap(
                "<h2 style='margin:0'>RetroEnv</h2><p class='muted' style='margin:2px 0 0'>Plan routes back to "
                "purchasable molecules. These are the tools and the verifier that agents use. "
                "API: <a href='/docs'>/docs</a> · tasks: <a href='/retro_route/splits'>/retro_route/splits</a></p>"
                f"<p class='muted' style='margin:2px 0 0'>{render.credit_line()}</p>"
            ),
            apply_default_css=False,
        )
        with gr.Row(equal_height=False):
            with gr.Column(scale=4, min_width=320):
                split = gr.Dropdown(splits, value=default_split, label="Split")
                task = gr.Dropdown(task_choices(default_split), value=0, label="Task", info=task_info(default_split))
                find = gr.Textbox("", label="Go to task", placeholder="task number or task id, then Enter", max_lines=1)
                start = gr.Button("Start episode", variant="primary")
                with gr.Tabs():
                    with gr.Tab("2D"):
                        target_2d = gr.HTML(render.structure_2d(None), apply_default_css=False)
                    with gr.Tab("3D"):
                        target_3d = gr.HTML(render.structure_3d(None), apply_default_css=False)
                target = gr.HTML(render.target_panel(None), apply_default_css=False)
            with gr.Column(scale=7, min_width=360):
                with gr.Tab("Plan"):
                    gr.HTML(
                        render.wrap(
                            "<p class='muted' style='margin:0 0 6px'>Cut the target back to molecules the stock "
                            "holds. Checking a cut or a precursor spends a tool call, exactly as it would for a "
                            "model.</p>"
                        ),
                        apply_default_css=False,
                    )
                    frontier = gr.Dropdown([], label="Molecule to disconnect", interactive=True)
                    selected = gr.HTML(render.molecule_views(None), apply_default_css=False)
                    candidate = gr.Dropdown([], label="Proposed cut", interactive=True)
                    custom = gr.Textbox("", label="Or precursors by hand", placeholder="SMILES.SMILES", max_lines=1)
                    with gr.Row():
                        check_cut = gr.Button("Check this cut")
                        add_cut = gr.Button("Add to plan", variant="primary")
                        check_stock = gr.Button("Is it in stock?")
                    verdict = gr.HTML(render.wrap(""), apply_default_css=False)
                    plan_view = gr.HTML(render.plan_panel("", {}, {}), apply_default_css=False)
                with gr.Tab("Tools"):
                    tool = gr.Dropdown([], label="Tool", interactive=True)
                    arguments = gr.Code("{}", language="json", label="Arguments", lines=6)
                    run = gr.Button("Run tool")
                    result = gr.Code("", language="json", label="Result", lines=12, interactive=False)
                with gr.Tab("Submit"):
                    from_plan = gr.Button("Fill from the plan")
                    submission = gr.Code("", language="json", label="emit_routes submission", lines=16)
                    emit = gr.Button("Emit routes and score", variant="primary")
                    score = gr.HTML(render.score_panel(None), apply_default_css=False)
                    routes = gr.HTML(render.routes_panel(None), apply_default_css=False)
                with gr.Tab("History"):
                    history = gr.HTML(render.history_panel([]), apply_default_css=False)
        gr.HTML(credits, apply_default_css=False)

        def on_split(name: str):
            return gr.update(choices=task_choices(name), value=0, info=task_info(name))

        def on_find(split_name: str, query: str):
            start = task_start(split_name, query)
            if start is None:
                gr.Warning(f"No task {query.strip()!r} in {split_name}")
                return gr.update()
            return gr.update(choices=task_choices(split_name, start), value=start)

        def on_start(state: PlaygroundSession, split_name: str, index: int):
            state.start(split_name, index)
            names = tool_names(state)
            target_smiles = state.opening["target_smiles"]
            first = names[0] if names else None
            return (
                state,
                render.structure_2d(target_smiles),
                render.structure_3d(target_smiles),
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

        def on_run(state: PlaygroundSession, name: str, text: str, keep: str):
            try:
                args = json.loads(text or "{}")
            except json.JSONDecodeError as exc:
                raise gr.Error(f"Arguments are not valid JSON: {exc}") from exc
            value = state.call(name, args)
            return (state, json.dumps(value, indent=2, sort_keys=True), *panels(state, keep))

        def on_emit(state: PlaygroundSession, text: str, keep: str):
            try:
                value = json.loads(text or "{}")
            except json.JSONDecodeError as exc:
                raise gr.Error(f"Submission is not valid JSON: {exc}") from exc
            state.call("emit_routes", {"submission": value})
            return (state, *panels(state, keep))

        def _precursors(chosen: str, typed: str) -> list[str]:
            raw = (typed or chosen or "").strip()
            parts = [piece.strip() for piece in raw.split(".") if piece.strip()]
            if not parts:
                raise gr.Error("Choose a proposed cut, or type the precursors.")
            return parts

        def on_frontier(product: str):
            return gr.update(choices=cut_choices(product), value=None), render.molecule_views(product)

        def on_check_cut(state: PlaygroundSession, product: str, chosen: str, typed: str):
            if not product:
                raise gr.Error("Start an episode first.")
            value = state.call(
                "validate_disconnection", {"product_smiles": product, "reactants": _precursors(chosen, typed)}
            )
            return (state, render.verdict_panel(value), *panels(state, product, chosen))

        def on_check_stock(state: PlaygroundSession, product: str, chosen: str, typed: str):
            if not product:
                raise gr.Error("Start an episode first.")
            found = []
            for smiles in _precursors(chosen, typed):
                found.append((smiles, state.call("stock_retrieve", {"query": smiles, "mode": "exact", "limit": 1})))
            return (state, render.stock_panel(found), *panels(state, product, chosen))

        def on_add_cut(state: PlaygroundSession, product: str, chosen: str, typed: str):
            if not product:
                raise gr.Error("Start an episode first.")
            try:
                state.plan[canonicalize_smiles(product)] = tuple(
                    canonicalize_smiles(piece) for piece in _precursors(chosen, typed)
                )
            except ChemistryError as exc:
                raise gr.Error(f"Not a readable SMILES: {exc}") from exc
            return (state, render.wrap(""), *panels(state))

        def on_from_plan(state: PlaygroundSession):
            if not state.opening:
                raise gr.Error("Start an episode first.")
            return json.dumps(state.build_submission(), indent=2)

        plan_outputs = [target, history, score, routes, plan_view, frontier, candidate, selected]
        split.change(on_split, split, task)
        find.submit(on_find, [split, find], task)
        start.click(
            on_start,
            [session, split, task],
            [session, target_2d, target_3d, *plan_outputs, tool, arguments, result, submission],
        )
        tool.change(on_tool, [session, tool], arguments)
        run.click(on_run, [session, tool, arguments, frontier], [session, result, *plan_outputs])
        emit.click(on_emit, [session, submission, frontier], [session, *plan_outputs])
        frontier.input(on_frontier, frontier, [candidate, selected])
        check_cut.click(on_check_cut, [session, frontier, candidate, custom], [session, verdict, *plan_outputs])
        check_stock.click(on_check_stock, [session, frontier, candidate, custom], [session, verdict, *plan_outputs])
        add_cut.click(on_add_cut, [session, frontier, candidate, custom], [session, verdict, *plan_outputs])
        from_plan.click(on_from_plan, session, submission)
    return demo
