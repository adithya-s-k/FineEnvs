"""The OpenEnv environment: MCP tools over one shared episode.

`reset(split=, index=, mode=, surface=)` picks a design template and how it is
played; the tools a model sees depend on the surface:

    tools  add_image, add_text, add_shape, update_element, ... (structured edits)
    html   write_html, read_html                               (one page of markup)
    gui    screenshot, left_click, left_click_drag, type, key  (a real editor)

On the gui surface the editor is the built-in one in one of six layouts, or a real app (skin="impress", "draw",
"inkscape-desktop", or one a plugin registers; realapps/), which the server runs locally or in Daytona sandboxes it
manages.

and every surface shares get_task, view_reference and submit. `submit` ends the
episode and its reward is the score of the final document (see core/score.py).
Pure MCP clients, which have no `reset`, start with `start_task`.

Tasks are served through OpenEnv's Task API (`list_splits`, `num_tasks`,
`get_task`, `get_task_range`), so a trainer can enumerate a split and then
reset onto any index.
"""

from __future__ import annotations

import base64
import json
import os
import random
from typing import Annotated, Any, Literal

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.transforms.visibility import Visibility
from fastmcp.utilities.types import Image
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.mcp_types import CallToolAction
from openenv.core.env_server.types import Action, Observation
from pydantic import Field

try:
    from ..core import ops
    from ..core import workspace as ws
    from ..core.contact import contact_sheet
    from ..core.html_profile import HTMLProfileError, html_to_doc
    from ..core.pack import TaskPack, load_pack
    from ..core.prompts import PROMPT_VERSION, instructions
    from ..models import DesignState
    from .episodes import MODES, REAL_SKINS, SKINS, STORE, SURFACES, Episode
    from .gui import VIEWPORT, GuiDriver, self_url
except ImportError:  # flat layout inside the Docker image
    from core import ops
    from core import workspace as ws
    from core.contact import contact_sheet
    from core.html_profile import HTMLProfileError, html_to_doc
    from core.pack import TaskPack, load_pack
    from core.prompts import PROMPT_VERSION, instructions
    from models import DesignState
    from server.episodes import MODES, REAL_SKINS, SKINS, STORE, SURFACES, Episode
    from server.gui import VIEWPORT, GuiDriver, self_url

EDIT_TOOLS = {
    "add_image", "add_text", "add_shape", "update_element", "delete_element",
    "duplicate_element", "reorder_element", "align_elements", "set_background", "undo", "redo",
    "write_html",
}
DEFAULT_MAX_STEPS = {"tools": 60, "html": 20, "gui": 200}
# Tools that only look (and the lifecycle call) don't spend the action budget.
OBSERVATION_TOOLS = {"start_task", "get_task", "view_reference", "view_canvas", "view_assets", "list_fonts",
                     "list_elements", "read_html", "screenshot", "wait"}
OBSERVATION_FACTOR = 3

Num = float | int
FILE_DIALOGS = ("mac", "windows", "linux")
COORDINATES = ("pixels", "normalized")
Coord = Annotated[list[int], Field(description="[x, y] on the screenshot: pixels, or a 0-1000 grid when "
                                               "your instructions say so", min_length=2, max_length=2)]


def _key_sequence(keys) -> list[str]:
    """'enter', ['ctrl+a', 'delete'] or the JSON string '["enter"]' -> key combos."""
    if isinstance(keys, str):
        text = keys.strip()
        if text.startswith("["):
            try:
                keys = json.loads(text)
            except json.JSONDecodeError:
                keys = [text.strip("[]\"' ")]
        else:
            return [text] if text else []
    return [str(k).strip() for k in keys if str(k).strip()]


def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


class DesignCanvasEnvironment(MCPEnvironment):
    """Recreate a graphic design on a real canvas, through tools, HTML or a GUI."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, pack: TaskPack | None = None):
        self.pack = pack or load_pack()
        self._episode: Episode | None = None
        self._state = DesignState()
        self._reward_mode = "final"
        self._last_total = 0.0
        mcp = FastMCP("design_canvas")
        self._mcp = mcp
        self._register_common(mcp)
        self._register_tools_surface(mcp)
        self._register_html_surface(mcp)
        self._register_gui_surface(mcp)
        super().__init__(mcp)
        self._set_surface("tools")

    # ============================================================ Task API

    def list_splits(self):
        return [{"name": s, "type": s, "num_tasks": self.pack.count(s)}
                for s in self.pack.splits()]

    def num_tasks(self, split):
        return self.pack.count(split)

    def get_task(self, split, index):
        return self.pack.public(self.pack.at(split, int(index)))

    def get_task_range(self, split, start=None, stop=None):
        return self.pack.task_range(split, start, stop)

    def list_tasks(self, split):
        return self.get_task_range(split)

    # ============================================================ lifecycle

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        split: str | None = None,
        index: int | None = None,
        task_id: str | None = None,
        mode: str = "reference",
        surface: str = "tools",
        skin: str | None = None,
        theme: str | None = None,
        layout: str | None = None,
        max_steps: int | None = None,
        reward_mode: str = "final",
        coordinates: str = "pixels",
        file_dialog: str | None = None,
        workspace: str | dict | None = None,
        budget_note: str = "count",
        **kwargs: Any,
    ) -> Observation:
        if coordinates not in COORDINATES:
            raise ValueError(f"coordinates must be one of {COORDINATES}")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if surface not in SURFACES:
            raise ValueError(f"surface must be one of {SURFACES}")
        if reward_mode not in ("final", "dense"):
            raise ValueError("reward_mode must be final or dense")
        if budget_note not in ("count", "efficient"):
            raise ValueError("budget_note must be count or efficient")
        if skin in REAL_SKINS and surface != "gui":  # an app is driven by mouse and keyboard only
            raise ValueError(f"skin {skin!r} is a real app: use surface='gui' (tools and html edit the canvas "
                             "directly, with no app)")
        rng = random.Random(seed)
        if task_id is not None:
            task = self.pack.get(task_id)
        else:
            split = split or ("train" if "train" in self.pack.splits() else self.pack.splits()[0])
            count = self.pack.count(split)
            index = rng.randrange(count) if index is None else int(index)
            task = self.pack.at(split, index)
        # A seeded reset randomises the look (layout and theme), so a policy has to read
        # the screen rather than memorise where controls live. Mirrored panels
        # (layout="swap") exist but are opt-in: no real editor flips its UI.
        skin = skin or (rng.choice(SKINS) if seed is not None else "figma")
        theme = theme or (rng.choice(("light", "dark")) if seed is not None else "light")
        layout = layout or "default"
        if layout not in ("default", "swap"):
            raise ValueError("layout must be default or swap")
        # The file window images are opened from looks like macOS, Windows or Linux (GTK).
        # Drawn after skin and theme so seeded looks from earlier versions are unchanged.
        file_dialog = file_dialog or (rng.choice(FILE_DIALOGS) if seed is not None else None)
        if file_dialog is not None and file_dialog not in FILE_DIALOGS:
            raise ValueError(f"file_dialog must be one of {FILE_DIALOGS}")
        # Where the task's images live (GUI only): None keeps a1.png, ... in Pictures; "realistic" puts them under
        # everyday names in an everyday folder among other files (core.workspace); a dict is a caller's own layout.
        if workspace is not None and surface != "gui":
            raise ValueError("a workspace is for the gui surface (the files are opened in a file window)")
        if workspace == "realistic":
            workspace = ws.realistic(task, self.pack, seed if seed is not None else rng.randrange(1 << 30))
        elif workspace is not None and not isinstance(workspace, dict):
            raise ValueError("workspace must be 'realistic' or a workspace spec (see core.workspace)")
        self._close_gui()
        ep = Episode(
            pack=self.pack, task=task, mode=mode, surface=surface, skin=skin, theme=theme,
            max_steps=int(max_steps or DEFAULT_MAX_STEPS[surface]), workspace=workspace, budget_note=budget_note,
        )
        if episode_id:
            ep.episode_id = str(episode_id)[:64]
        STORE.add(ep)
        self._episode = ep
        self._reward_mode = reward_mode
        # GUI coordinates: screenshot pixels, or the 0-1000 grid many open VLMs (Qwen, Gemma)
        # are trained to point with. Converted to pixels at the tool, so the editor never knows.
        self._coordinates = coordinates
        self._last_total = 0.0
        self._set_surface(surface)
        editor = f"/editor/?episode={ep.episode_id}&skin={skin}&theme={theme}" + (
            "&layout=swap" if layout == "swap" else "") + (f"&os={file_dialog}" if file_dialog else "") + (
            # A seeded reset also fixes where the file window opens and how it lists files.
            f"&fs={rng.randrange(1 << 30)}" if seed is not None else "")
        if skin in REAL_SKINS:  # a real app behind the same interface: its own editor, set up per episode
            from ..realapps import available, get_backend, registry

            ok, why = available()
            if not ok:
                raise RuntimeError(f"{REAL_SKINS[skin][0]} is not available on this server: {why}")

            backend = get_backend(skin, fonts_dir=self.pack.root / "fonts")
            gt = self.pack.gt_doc(task)
            if registry.get(skin).kind == "web":  # a web app (a plugin's): it opens itself in the env's browser
                session = backend.open_episode(self, ep, gt)
            else:  # a desktop app on its own X display: the GUI tools drive that screen instead of a page
                session = backend.new_episode(skin, gt.width, gt.height, ep.assets, fonts=self.pack.fonts,
                                              workspace=ep.workspace)
                GuiDriver.get().attach_remote(ep.episode_id, session.client)
            ep.realapp = (backend, session)
            # the app lives behind the server (often in a sandbox): what a client can open is its live screen
            editor = f"/api/episodes/{ep.episode_id}/screen.png"
        elif surface == "gui":
            GuiDriver.get().open(ep.episode_id, self_url() + editor + "&agent=1").result(45)
        self._state = DesignState(
            episode_id=ep.episode_id, step_count=0, task_id=task["task_id"], split=task["split"],
            mode=mode, surface=surface, skin=skin, editor_url=editor,
        )
        public = os.environ.get("DESIGN_CANVAS_PUBLIC_URL", "").rstrip("/")
        return self._observation(ep, public + editor)

    def _observation(self, ep: Episode, editor: str) -> Observation:
        info = ep.info()
        meta = {
            **info,
            "instructions": instructions(ep.surface, ep.mode, info,
                                         (VIEWPORT["width"], VIEWPORT["height"]),
                                         coordinates=getattr(self, "_coordinates", "pixels"), skin=ep.skin),
            "coordinates": getattr(self, "_coordinates", "pixels"),
            "prompt_version": PROMPT_VERSION,
            "skin": ep.skin,
            "theme": ep.theme,
            "layout": "swap" if "layout=swap" in editor else "default",
            "file_dialog": next((d for d in FILE_DIALOGS if f"&os={d}" in editor), None),
            "editor_url": editor,
            "tools": sorted(self._surface_tools(ep.surface)),
        }
        if ep.skin in REAL_SKINS:  # which real app the episode runs in (its tools and reward are the usual ones)
            meta["app"] = {"skin": ep.skin, "name": REAL_SKINS[ep.skin][0], "real": True}
        if ep.mode == "reference":
            meta["reference_jpeg_b64"] = base64.b64encode(
                self.pack.target_jpeg(ep.task, 768)).decode()
        return Observation(done=False, reward=None, metadata=meta)

    def _set_surface(self, surface: str) -> None:
        # One FastMCP server per session; swap the visibility transform so
        # list_tools only advertises this surface's tools.
        self._mcp._transforms = [t for t in self._mcp._transforms
                                 if not isinstance(t, Visibility)]
        self._mcp.enable(tags={"common", surface}, only=True)

    @staticmethod
    def _surface_tools(surface: str) -> set[str]:
        return {name for name, tags in _TOOL_TAGS.items() if tags & {"common", surface}}

    def _close_gui(self):
        if self._episode is not None and self._episode.surface == "gui":
            try:
                GuiDriver.get().close(self._episode.episode_id).result(10)
            except Exception:
                pass
            if self._episode.realapp is not None and hasattr(self._episode.realapp[0], "release"):
                GuiDriver.get().detach_remote(self._episode.episode_id)
                backend, session = self._episode.realapp
                self._episode.realapp = None  # released once: the app (a desktop, a sandbox) is free again
                backend.release(session.url)

    def close(self):
        self._close_gui()
        super().close()

    @property
    def state(self) -> DesignState:
        ep = self._episode
        if ep is not None:
            self._state.step_count = ep.step_count
            self._state.n_elements = len(ep.doc.elements)
            self._state.done = ep.done
            if ep.result is not None:
                self._state.reward = ep.result.total
                self._state.grade = ep.result.as_dict()
        return self._state

    # ============================================================ stepping

    def _after_step(self, action: Action, obs: Observation) -> Observation:
        ep = self._episode
        if ep is None or not isinstance(action, CallToolAction):
            return obs
        tool = action.tool_name
        # The budget counts actions, not looking: a harness that screenshots after every turn,
        # or an agent that checks its render, spends nothing. Looking has its own generous cap
        # so an agent can't loop on it forever.
        if tool in OBSERVATION_TOOLS:
            ep.observation_count += 1
            if not ep.done and ep.observation_count >= OBSERVATION_FACTOR * ep.max_steps:
                ep.submit("env", reason="observation limit")
        else:
            ep.step_count += 1
        if not ep.done and ep.step_count >= ep.max_steps:
            ep.submit("env", reason="step limit")
        if ep.done and ep.result is not None:
            obs.done = True
            final = ep.result.total
            obs.reward = final - self._last_total if self._reward_mode == "dense" else final
            self._last_total = final
            obs.metadata = {**(obs.metadata or {}), "grade": ep.result.as_dict(),
                            "episode_id": ep.episode_id}
        elif self._reward_mode == "dense" and tool in EDIT_TOOLS:
            total = ep.score().total
            obs.reward = total - self._last_total
            self._last_total = total
        else:
            obs.reward = 0.0
        return obs

    def step(self, action: Action, timeout_s: float | None = None, **kwargs: Any) -> Observation:
        obs = super().step(action, timeout_s=timeout_s, **kwargs)
        return self._after_step(action, obs)

    async def step_async(self, action: Action, timeout_s: float | None = None,
                         **kwargs: Any) -> Observation:
        obs = await super().step_async(action, timeout_s=timeout_s, **kwargs)
        return self._after_step(action, obs)

    def _step_impl(self, action: Action, timeout_s: float | None = None, **kwargs) -> Observation:
        return Observation(done=False, reward=0.0, metadata={
            "error": f"unsupported action {type(action).__name__}; use CallToolAction"})

    # ============================================================ helpers

    def _ep(self) -> Episode:
        if self._episode is None:
            raise ToolError("no task is loaded: call reset() or the start_task tool first")
        return self._episode

    def _edit(self, op: str, args: dict) -> str:
        ep = self._ep()
        if ep.done:
            raise ToolError("the episode is finished; start a new task")
        try:
            result = ep.apply(op, {k: v for k, v in args.items() if v is not None}, "agent")
        except ops.OpError as exc:
            raise ToolError(str(exc)) from None
        return _json(result)

    # ============================================================ tools

    def _register_common(self, mcp: FastMCP) -> None:
        env = self

        @mcp.tool(tags={"common"})
        def start_task(
            split: Annotated[str, Field(description="train, validation or test")] = "validation",
            index: Annotated[int | None, Field(description="task index; random if omitted")] = None,
            mode: Annotated[Literal["reference", "description"], Field(
                description="reference: recreate a target image; description: build from a brief"
            )] = "reference",
            surface: Annotated[Literal["tools", "html"], Field(
                description="tools: structured edit calls; html: write one page of markup"
            )] = "tools",
        ) -> str:
            """Load a design task (for MCP clients without reset). Returns the instructions.

            After starting, open the returned editor_url in a browser to watch the canvas live.
            """
            obs = env.reset(split=split, index=index, mode=mode, surface=surface)
            meta = obs.metadata
            return _json({k: meta[k] for k in ("episode_id", "task_id", "instructions",
                                                "editor_url", "tools")})

        @mcp.tool(tags={"common"})
        def get_task() -> str:
            """The current task: canvas size, assets (id, kind, aspect ratio), fonts used, and,
            in description mode, the design brief."""
            return _json(env._ep().info())

        @mcp.tool(tags={"common"})
        def view_reference() -> Image:
            """The target design you are recreating (reference mode only)."""
            ep = env._ep()
            if ep.mode != "reference":
                raise ToolError("description mode has no target image; read the brief in get_task")
            return Image(data=env.pack.target_jpeg(ep.task, 768), format="jpeg")

        @mcp.tool(tags={"common"})
        def submit() -> str:
            """Finish the episode and score the current canvas. Irreversible."""
            ep = env._ep()
            grade = ep.submit("agent")
            return grade.report()

        @mcp.tool(tags={"tools", "html"})
        def view_canvas() -> Image:
            """Render your current canvas, exactly as it will be scored."""
            ep = env._ep()
            return Image(data=ep.render_jpeg(768), format="jpeg")

        @mcp.tool(tags={"tools", "html"})
        def view_assets() -> Image:
            """One image showing every asset of this task, labelled with its id, kind and
            aspect ratio. Checkerboard = transparent."""
            ep = env._ep()
            font = env.pack.fonts.file_path("Montserrat", "regular")
            return Image(data=contact_sheet(ep.task["assets"], ep.assets, font), format="png")

        @mcp.tool(tags={"tools", "html"})
        def list_fonts(query: Annotated[str | None, Field(
                description="optional case-insensitive filter")] = None) -> str:
            """The font families available. Any other family is rejected."""
            fams = env.pack.fonts.families()
            if query:
                fams = [f for f in fams if query.lower() in f.lower()]
            return _json({"count": len(fams), "fonts": fams})

    def _register_tools_surface(self, mcp: FastMCP) -> None:
        env = self
        tags = {"tools"}

        @mcp.tool(tags=tags)
        def list_elements() -> str:
            """Everything on the canvas, back to front, with ids, boxes and styles."""
            ep = env._ep()
            return ep.doc.summary() + "\n\n" + _json(ep.doc.public()["elements"])

        @mcp.tool(tags=tags)
        def add_image(
            asset_id: Annotated[str, Field(description="asset id, e.g. a3")],
            x: Num | None = None, y: Num | None = None,
            width: Num | None = None, height: Num | None = None,
            angle: Annotated[Num, Field(description="degrees clockwise")] = 0,
            opacity: Annotated[Num, Field(ge=0, le=1)] = 1,
        ) -> str:
            """Place an asset on top of the canvas. Give width or height alone to keep the
            asset's aspect ratio; omit both for a default size. Returns the new element id."""
            return env._edit("add_image", dict(asset_id=asset_id, x=x, y=y, width=width,
                                               height=height, angle=angle, opacity=opacity))

        @mcp.tool(tags=tags)
        def add_text(
            text: Annotated[str, Field(description="the copy; use \\n for line breaks")],
            x: Num | None = None, y: Num | None = None,
            width: Annotated[Num | None, Field(
                description="box width for alignment; omit to fit the text")] = None,
            font_family: str = "Montserrat",
            font_size: Annotated[Num, Field(description="px")] = 48,
            color: Annotated[str, Field(description="#rrggbb")] = "#000000",
            align: Literal["left", "center", "right"] = "left",
            bold: bool = False, italic: bool = False, underline: bool = False,
            uppercase: bool = False,
            line_height: Annotated[Num, Field(description="multiple of font size")] = 1.0,
            letter_spacing: Annotated[Num, Field(description="px added after each letter")] = 0,
            angle: Num = 0, opacity: Annotated[Num, Field(ge=0, le=1)] = 1,
            runs: Annotated[list[dict] | None, Field(
                description="optional styled pieces [{text, color?, bold?, italic?}] that "
                            "concatenate to the text")] = None,
        ) -> str:
            """Add a text box on top of the canvas. Height follows the number of lines."""
            return env._edit("add_text", dict(
                text=text, x=x, y=y, width=width, font_family=font_family, font_size=font_size,
                color=color, align=align, bold=bold, italic=italic, underline=underline,
                uppercase=uppercase, line_height=line_height, letter_spacing=letter_spacing,
                angle=angle, opacity=opacity, runs=runs))

        @mcp.tool(tags=tags)
        def add_shape(
            shape: Literal["rect", "ellipse"] = "rect",
            x: Num | None = None, y: Num | None = None,
            width: Num | None = None, height: Num | None = None,
            fill: Annotated[str, Field(description="#rrggbb or none")] = "#d9d9d9",
            stroke: str | None = None, stroke_width: Num = 0,
            radius: Annotated[Num, Field(description="corner radius (rect)")] = 0,
            angle: Num = 0, opacity: Annotated[Num, Field(ge=0, le=1)] = 1,
        ) -> str:
            """Draw a rectangle or ellipse on top of the canvas."""
            return env._edit("add_shape", dict(
                shape=shape, x=x, y=y, width=width, height=height, fill=fill, stroke=stroke,
                stroke_width=stroke_width, radius=radius, angle=angle, opacity=opacity))

        @mcp.tool(tags=tags)
        def update_element(
            element_id: str,
            properties: Annotated[dict, Field(
                description="fields to change, e.g. {\"x\": 40, \"font_size\": 72, "
                            "\"color\": \"#ff0000\"}. Any field from add_image/add_text/"
                            "add_shape, plus visible, locked, flip_x, flip_y")],
        ) -> str:
            """Change an element's position, size, rotation or style."""
            if not isinstance(properties, dict):
                raise ToolError("properties must be an object")
            return env._edit("update_element", {"element_id": element_id, **properties})

        @mcp.tool(tags=tags)
        def delete_element(element_id: str) -> str:
            """Remove an element."""
            return env._edit("delete_element", {"element_id": element_id})

        @mcp.tool(tags=tags)
        def duplicate_element(element_id: str, dx: Num = 20, dy: Num = 20) -> str:
            """Copy an element just above itself, offset by (dx, dy)."""
            return env._edit("duplicate_element", {"element_id": element_id, "dx": dx, "dy": dy})

        @mcp.tool(tags=tags)
        def reorder_element(
            element_id: str,
            to: Annotated[str, Field(
                description="front, back, forward, backward, or a layer index (0 = back)")],
        ) -> str:
            """Change an element's place in the stacking order."""
            return env._edit("reorder_element", {"element_id": element_id, "to": to})

        @mcp.tool(tags=tags)
        def align_elements(
            element_ids: list[str],
            alignment: Literal["left", "center", "right", "top", "middle", "bottom"],
            relative_to: Literal["canvas", "selection"] = "canvas",
        ) -> str:
            """Align elements to the canvas or to each other."""
            return env._edit("align_elements", {"element_ids": element_ids,
                                                "alignment": alignment,
                                                "relative_to": relative_to})

        @mcp.tool(tags=tags)
        def set_background(color: Annotated[str, Field(description="#rrggbb")]) -> str:
            """Set the canvas background colour."""
            return env._edit("set_background", {"color": color})

        @mcp.tool(tags=tags)
        def undo() -> str:
            """Undo the last change."""
            ok = env._ep().undo("agent")
            return "undone" if ok else "nothing to undo"

        @mcp.tool(tags=tags)
        def redo() -> str:
            """Redo the last undone change."""
            ok = env._ep().redo("agent")
            return "redone" if ok else "nothing to redo"

    def _register_html_surface(self, mcp: FastMCP) -> None:
        env = self

        @mcp.tool(tags={"html"})
        def write_html(html: Annotated[str, Field(
                description='the full page: <div id="canvas" ...> with one child per element')],
        ) -> str:
            """Replace the canvas with this HTML. Returns what was parsed and any warnings."""
            ep = env._ep()
            if ep.done:
                raise ToolError("the episode is finished; start a new task")
            try:
                document, warnings = html_to_doc(html, ep.doc.width, ep.doc.height)
                ep.apply("replace_document", {"document": document}, "agent")
            except (HTMLProfileError, ops.OpError, ValueError) as exc:
                raise ToolError(f"HTML rejected: {exc}") from None
            out = {"elements": len(ep.doc.elements), "summary": ep.doc.summary()}
            if warnings:
                out["warnings"] = sorted(set(warnings))
            return _json(out)

        @mcp.tool(tags={"html"})
        def read_html() -> str:
            """The current canvas as HTML in the profile, ready to edit and write back."""
            return env._ep().html()

    def _register_gui_surface(self, mcp: FastMCP) -> None:
        env = self
        tags = {"gui"}
        driver = GuiDriver.get

        def gui_ep() -> Episode:
            ep = env._ep()
            if ep.surface != "gui":
                raise ToolError("GUI actions need reset(surface='gui')")
            if ep.done:
                raise ToolError("the episode is finished")
            return ep

        def mods(text: str | None) -> list[str]:
            return [m for m in (text or "").split("+") if m]

        def px(c) -> tuple[int, int]:
            """The agent's [x, y] as screenshot pixels."""
            x, y = int(c[0]), int(c[1])
            if getattr(env, "_coordinates", "pixels") == "normalized":
                x = round(x * (VIEWPORT["width"] - 1) / 1000)
                y = round(y * (VIEWPORT["height"] - 1) / 1000)
            return x, y

        @mcp.tool(tags=tags)
        async def screenshot() -> Image:
            """Capture the editor window as the next observation."""
            ep = env._ep()
            return Image(data=await driver().screenshot(ep.episode_id), format="png")

        @mcp.tool(tags=tags)
        async def left_click(coordinate: Coord, text: Annotated[str | None, Field(
                description="modifier keys held during the click, e.g. shift or ctrl")] = None,
        ) -> str:
            """Left-click at [x, y]."""
            ep = gui_ep()
            await driver().click(ep.episode_id, *px(coordinate), modifiers=mods(text))
            return f"clicked {coordinate}"

        @mcp.tool(tags=tags)
        async def double_click(coordinate: Coord) -> str:
            """Double-click at [x, y] (edits text, opens a field)."""
            ep = gui_ep()
            await driver().click(ep.episode_id, *px(coordinate), clicks=2)
            return f"double-clicked {coordinate}"

        @mcp.tool(tags=tags)
        async def triple_click(coordinate: Coord) -> str:
            """Triple-click at [x, y] (selects all text in a field)."""
            ep = gui_ep()
            await driver().click(ep.episode_id, *px(coordinate), clicks=3)
            return f"triple-clicked {coordinate}"

        @mcp.tool(tags=tags)
        async def right_click(coordinate: Coord) -> str:
            """Right-click at [x, y] (context menu)."""
            ep = gui_ep()
            await driver().click(ep.episode_id, *px(coordinate), button="right")
            return f"right-clicked {coordinate}"

        @mcp.tool(tags=tags)
        async def mouse_move(coordinate: Coord) -> str:
            """Move the pointer to [x, y] without clicking (shows hover states)."""
            ep = gui_ep()
            await driver().move(ep.episode_id, *px(coordinate))
            return f"moved to {coordinate}"

        @mcp.tool(tags=tags)
        async def left_click_drag(start_coordinate: Coord, coordinate: Coord,
                                  text: str | None = None) -> str:
            """Press at start_coordinate, drag to coordinate, release. Moves, resizes and
            rotates elements via their handles, and drops assets onto the canvas."""
            ep = gui_ep()
            await driver().drag(ep.episode_id, *px(start_coordinate), *px(coordinate),
                                modifiers=mods(text))
            return f"dragged {start_coordinate} -> {coordinate}"

        @mcp.tool(tags=tags)
        async def scroll(coordinate: Coord,
                         scroll_direction: Literal["up", "down", "left", "right"],
                         scroll_amount: Annotated[int, Field(ge=1, le=20)] = 3,
                         text: str | None = None) -> str:
            """Scroll at [x, y]. Hold ctrl (text="ctrl") to zoom the canvas."""
            ep = gui_ep()
            step = 100 * scroll_amount
            dx = {"left": -step, "right": step}.get(scroll_direction, 0)
            dy = {"up": -step, "down": step}.get(scroll_direction, 0)
            await driver().scroll(ep.episode_id, *px(coordinate), dx, dy, modifiers=mods(text))
            return f"scrolled {scroll_direction} {scroll_amount}"

        @mcp.tool(name="type", tags=tags)
        async def type_text(text: str) -> str:
            """Type text at the keyboard focus."""
            ep = gui_ep()
            await driver().type_text(ep.episode_id, text)
            return f"typed {len(text)} characters"

        @mcp.tool(tags=tags)
        async def key(keys: Annotated[str | list[str], Field(
                description="a key or chord, e.g. enter, escape, delete, ctrl+z, shift+up; "
                            "a list presses each in turn")],
        ) -> str:
            """Press a key or key combination."""
            ep = gui_ep()
            sequence = _key_sequence(keys)
            if not sequence:
                raise ToolError("keys is empty")
            for combo in sequence:
                await driver().key(ep.episode_id, combo)
            return f"pressed {' then '.join(sequence)}"

        @mcp.tool(tags=tags)
        async def wait(duration: Annotated[Num, Field(ge=0, le=10)] = 1) -> str:
            """Wait for the given seconds."""
            ep = gui_ep()
            await driver().wait(ep.episode_id, float(duration))
            return f"waited {duration}s"


_TOOL_TAGS: dict[str, set[str]] = {
    "start_task": {"common"}, "get_task": {"common"}, "view_reference": {"common"},
    "submit": {"common"},
    "view_canvas": {"tools", "html"}, "view_assets": {"tools", "html"},
    "list_fonts": {"tools", "html"},
    **{n: {"tools"} for n in ("list_elements", "add_image", "add_text", "add_shape",
                              "update_element", "delete_element", "duplicate_element",
                              "reorder_element", "align_elements", "set_background", "undo",
                              "redo")},
    "write_html": {"html"}, "read_html": {"html"},
    **{n: {"gui"} for n in ("screenshot", "left_click", "double_click", "triple_click",
                            "right_click", "mouse_move", "left_click_drag", "scroll", "type",
                            "key", "wait")},
}
