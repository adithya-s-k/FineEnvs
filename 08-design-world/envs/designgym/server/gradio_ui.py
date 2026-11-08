"""The /web playground: pick a design, work on it through any surface, score it.

Same shape as the other FineEnvs playgrounds (task controls on the left, the
action and the score on the right), with one addition: the real editor is
embedded, live-synced to the episode, so whatever you do in the Tool call, HTML
or Expert tabs appears on the canvas as it happens, and whatever you drag in the
editor shows up in the trajectory.

Every button here goes through the same `DesignCanvasEnvironment` a trainer
talks to, via `CallToolAction`, so the playground exercises the real MCP path.
"""

from __future__ import annotations

import base64
import io
import json
import random
import threading
from collections import OrderedDict

import gradio as gr
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction
from PIL import Image

try:
    from ..core import expert
    from ..core.pack import TaskPack
    from ..core.score import CONFIGS
    from ..realapps import available as apps_available
    from ..realapps import status as apps_status
    from .environment import DesignCanvasEnvironment
    from .episodes import REAL_SKINS, SKIN_INFO, STORE
except ImportError:
    from core import expert
    from core.pack import TaskPack
    from core.score import CONFIGS
    from realapps import available as apps_available
    from realapps import status as apps_status
    from server.environment import DesignCanvasEnvironment
    from server.episodes import REAL_SKINS, SKIN_INFO, STORE

SIM_CHOICES = [(f"{name} ({kind})", sid) for sid, (name, kind) in SKIN_INFO.items()]
SKIN_CHOICES = SIM_CHOICES + [(f"{name} (real app)", sid) for sid, (name, _) in REAL_SKINS.items()]


def skin_choices(surface: str) -> list:
    """The editors a surface can use: real apps only on computer use (the env rejects them elsewhere), and marked
    when this server cannot reach them."""
    if surface != "gui":
        return SIM_CHOICES
    ok, _ = apps_available()
    # an app this server cannot reach is not offered at all, rather than failing on Load
    return SIM_CHOICES + ([(f"{name} (real app)", sid) for sid, (name, _) in REAL_SKINS.items()] if ok else [])
MODE_CHOICES = [("Recreate from reference", "reference"), ("Build from brief", "description")]
SURFACE_CHOICES = [("Tool calls", "tools"), ("HTML", "html"), ("Computer use", "gui")]
CLICK_CHOICES = [("Click", "left_click"), ("Double-click", "double_click"), ("Right-click", "right_click")]

INTRO = """
Recreate a real graphic design from its parts. Each task is a design
template taken apart into its pre-rendered assets and its typed copy; the agent rebuilds it on a canvas through
**tool calls**, **one page of HTML**, or **an editor driven by mouse and keyboard** (computer use). All three edit
the same document and are scored by the same renderer. On computer use the editor is the built-in one in one of six
layouts, or a real app (LibreOffice Impress, LibreOffice Draw, Inkscape) that this server runs, locally or in Daytona
sandboxes. A real app's own document is read back on submit and scored the same way.

Reward (reference mode) `= 0.60·structure + 0.30·visual + 0.05·order + 0.05·background`, where *structure*
matches elements one-to-one (content, position, size, typography) and *visual* is multi-scale SSIM + colour
layout against `render(ground truth)`, normalised so an empty canvas scores 0. Description mode loosens geometry and adds
coverage of every asset and every line of copy.
"""

SCORING = """
**Structure (0.60).** Submitted and target elements are matched one-to-one with the Hungarian algorithm. A pair
scores `content × geometry` for images and `content × (0.62·geometry + 0.38·typography)` for text. Content is 1 for
the right asset (a look-alike asset earns partial credit by visual similarity) and 1 − normalised edit distance for
copy. Geometry averages a tolerant IoU, centre distance (σ = 2.5 % of the canvas diagonal) and log size ratio, on the
box that is actually drawn: for text, the extent of the glyphs, so an invisible box width is never scored.
Typography covers font, size, colour, alignment, weight, case, letter spacing and line height. The sum is divided by
`max(#target, #submitted)`, so extra elements cost as much as missing ones.

**Visual (0.30).** Both documents are rendered at 256 px by the same skia renderer and compared on an area-averaged
pyramid: SSIM at 128, 64 and 32 px plus colour-layout error at 32 px, each rescaled so a blank canvas is 0 and an exact
match is 1. Weighting the coarse scales keeps a few-pixel nudge from reading as a different design; precision is the
structure term's job. The target is `render(ground truth)`, never the source's own preview, so a perfect reconstruction
scores exactly 1.0.

**Order (0.05)** agreement of stacking order among overlapping matched pairs. **Background (0.05)** colour
distance of the page. Replaying the expert trajectory scores 1.000 on every task in the pack.
"""


def _b64_image(data: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")


def _png(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


class Playground:
    def __init__(self, pack: TaskPack):
        self.pack = pack
        self.envs: OrderedDict[str, DesignCanvasEnvironment] = OrderedDict()
        self.lock = threading.Lock()

    # ------------------------------------------------------------ sessions

    def env(self, episode_id: str | None) -> DesignCanvasEnvironment | None:
        if not episode_id:
            return None
        with self.lock:
            env = self.envs.get(episode_id)
            if env is not None:
                self.envs.move_to_end(episode_id)
            return env

    def start(self, split, index, mode, surface, skin, previous=None):
        with self.lock:  # this tab's last episode is done with: free its app (a desktop, a browser page) first
            old = self.envs.pop(previous, None) if previous else None
        if old is not None:
            old.close()
        env = DesignCanvasEnvironment(self.pack)
        obs = env.reset(split=split, index=int(index), mode=mode, surface=surface, skin=skin)
        ep_id = obs.metadata["episode_id"]
        with self.lock:
            self.envs[ep_id] = env
            while len(self.envs) > 64:
                _, old = self.envs.popitem(last=False)
                old.close()
        return env, obs

    def call(self, episode_id, tool, args):
        env = self.env(episode_id)
        if env is None:
            raise gr.Error("Start a task first.")
        obs = env.step(CallToolAction(tool_name=tool, arguments=args))
        return obs

    def tools(self, episode_id):
        env = self.env(episode_id)
        if env is None:
            return []
        obs = env.step(ListToolsAction())
        return [t.name for t in obs.tools if t.name != "start_task"]


def build_ui(pack: TaskPack) -> gr.Blocks:
    pg = Playground(pack)
    splits = pack.splits()

    def index_range(s: str) -> str:
        return f"{s}: 0 – {pack.count(s) - 1:,}"
    default_split = "validation" if "validation" in splits else splits[0]

    def editor_frame(ep_id, skin):
        # Inline styles: OpenEnv mounts this Blocks inside its own app, which
        # drops a nested Blocks' css.
        if not ep_id:
            return ("<div style='height:220px;display:flex;align-items:center;justify-content:center;"
                    "opacity:.6;border:1px dashed #555;border-radius:8px'>Start a task to open the "
                    "editor.</div>")
        src = f"/editor/?episode={ep_id}&skin={skin}"
        # Sized by its width at the editor's own 1280x800 shape, never by 100vh: on a Space the page sits in an
        # iframe the Hub grows to fit its content, so vh there is that iframe's height and the editor ran away.
        return (f"<div style='width:100%'><iframe src='{src}' title='DesignGym editor' "
                f"style='display:block;width:100%;aspect-ratio:16/10;min-height:560px;max-height:880px;"
                f"border:1px solid #444;border-radius:8px;"
                f"background:#fff' allow='clipboard-write'></iframe>"
                f"<div style='text-align:right;margin-top:6px;font-size:12px'><a href='{src}' "
                f"target='_blank' rel='noopener'>Open the editor in a new tab ↗</a></div></div>")

    def task_info(task, obs_meta):
        app = obs_meta.get("app")
        where = f" · in {app['name']} (real app, {apps_status()['runs_on']})" if app else ""
        return (f"**{task['title']}** · {task['format']} · {task['canvas']['width']}×{task['canvas']['height']} px · "
                f"{task['stats']['n_elements']} layers ({task['stats']['n_text']} text) · {task['difficulty']} · "
                f"`{task['task_id']}` · episode `{obs_meta['episode_id']}`{where}")

    def screen_now(ep_id):
        ep = STORE.maybe(ep_id)
        if ep is None or ep.surface != "gui":
            return None
        obs = pg.call(ep_id, "screenshot", {})
        content = (obs.result.get("content") if isinstance(obs.result, dict)
                   else getattr(obs.result, "content", None)) or []
        for c in content:
            c = c if isinstance(c, dict) else c.model_dump()
            if c.get("type") == "image":
                return _b64_image(c["data"])
        return None

    def load(split, index, mode, surface, skin, previous=None, progress=gr.Progress()):
        count = pack.count(split)
        index = max(0, min(int(index or 0), count - 1))
        real = skin in REAL_SKINS
        if real and surface != "gui":
            raise gr.Error(f"{REAL_SKINS[skin][0]} is a real app: it runs on Computer use only.")
        if real and not apps_available()[0]:
            raise gr.Error(f"{REAL_SKINS[skin][0]} is not available on this server: {apps_available()[1]}")
        if real:
            progress(0, desc=f"Opening {REAL_SKINS[skin][0]} (a real app: the first one can take a minute)")
        env, obs = pg.start(split, index, mode, surface, skin, previous)
        meta = obs.metadata
        ep = STORE.get(meta["episode_id"])
        task = ep.task
        gallery = [(str(p), aid) for aid, p in ep.assets.items()]
        target = _png(pack.target_png(task, 768)) if mode == "reference" else None
        brief = task["description"] if mode == "description" else ""
        tools = pg.tools(meta["episode_id"])
        return (
            meta["episode_id"], index,
            task_info(task, meta),
            gr.update(value=target, visible=mode == "reference"),
            gr.update(value=brief, visible=mode == "description"),
            gallery,
            gr.update(value=editor_frame(meta["episode_id"], skin), visible=not real),
            _png(ep.render_png(768)),
            meta["instructions"],
            gr.update(choices=tools, value=tools[0] if tools else None),
            "", None, "", "*Not submitted yet.*", {},
            gr.update(visible=surface == "html"),
            gr.update(visible=real),
            screen_now(meta["episode_id"]) if real else None,
            gr.update(visible=real),
        )

    def act_on_screen(ep_id, tool, args):
        if not ep_id or STORE.maybe(ep_id) is None:
            raise gr.Error("Start a task first.")
        text, _, _ = call_tool(ep_id, tool, json.dumps(args))
        return screen_now(ep_id), text

    def click_screen(ep_id, kind, evt: gr.SelectData):
        x, y = (int(v) for v in evt.index)
        return act_on_screen(ep_id, kind, {"coordinate": [x, y]})

    def read_from_app(ep_id):
        ep = STORE.maybe(ep_id)
        if ep is None:
            raise gr.Error("Start a task first.")
        if not ep.read_app():
            raise gr.Error("Only a running real-app episode has a document to read.")
        return _png(ep.render_png(768))

    def apps_table(check):
        st = apps_status(check=check)
        rows = ["| App | Skin |", "|---|---|"] + [f"| {a['name']} | `{a['skin']}` |" for a in st["apps"]]
        md = [f"Real apps run **{st['runs_on']}**" + (f" (up to {st['desktops']} desktop episodes at once)"
                                                       if st["runs_on"] == "daytona" else "") + ".", "", *rows]
        if "sandboxes" in st:
            md += ["", "| Sandbox | Role | State | CPU | Memory | Episodes now |", "|---|---|---|---:|---:|---:|"]
            md += [f"| `{b['id']}` | {b['role']} | {b['state']} | {b['cpu']} | {b['memory_gb']} GB | {b['episodes']} |"
                   for b in st["sandboxes"]] or ["| none yet | | | | | |"]
        if "endpoints" in st:
            md += ["", "| Endpoint | Role | Up |", "|---|---|---|"]
            md += [f"| `{e['url']}` | {e['role']} | {('yes' if e['up'] else 'no') if 'up' in e else 'not checked'} |"
                   for e in st["endpoints"]]
        return "\n".join(md)

    def shuffle(split, mode, surface, skin, previous=None):
        return load(split, random.randrange(pack.count(split)), mode, surface, skin, previous)

    def step_index(split, index, delta, mode, surface, skin, previous=None):
        return load(split, (int(index or 0) + delta) % pack.count(split), mode, surface, skin, previous)

    def render_now(ep_id):
        ep = STORE.maybe(ep_id)
        return _png(ep.render_png(768)) if ep else None

    def call_tool(ep_id, tool, args_text):
        if not tool:
            raise gr.Error("Pick a tool.")
        try:
            args = json.loads(args_text) if args_text and args_text.strip() else {}
        except json.JSONDecodeError as exc:
            raise gr.Error(f"Arguments are not JSON: {exc}") from None
        obs = pg.call(ep_id, tool, args)
        text, image = "", None
        if obs.error is not None:
            text = f"❌ {obs.error.message}"
        else:
            res = obs.result
            content = (res.get("content") if isinstance(res, dict) else getattr(res, "content", None)) or []
            for c in content:
                c = c if isinstance(c, dict) else c.model_dump()
                if c.get("type") == "image":
                    image = _b64_image(c["data"])
                elif c.get("type") == "text":
                    text += c["text"]
            if isinstance(res, dict) and res.get("is_error"):
                text = "❌ " + text
        if obs.done:
            text += f"\n\n**Episode done · reward {obs.reward:.3f}**"
        return text, image, render_now(ep_id)

    def tool_template(tool):
        examples = {
            "add_image": {"asset_id": "a1", "x": 0, "y": 0, "width": 400},
            "add_text": {"text": "Summer\nSale", "x": 80, "y": 120, "font_family": "Montserrat",
                         "font_size": 96, "color": "#1d1d1f", "align": "left"},
            "add_shape": {"shape": "rect", "x": 100, "y": 100, "width": 300, "height": 120,
                          "fill": "#e63946", "radius": 24},
            "update_element": {"element_id": "e1", "properties": {"x": 40, "y": 40}},
            "delete_element": {"element_id": "e1"},
            "duplicate_element": {"element_id": "e1", "dx": 20, "dy": 20},
            "reorder_element": {"element_id": "e1", "to": "front"},
            "align_elements": {"element_ids": ["e1"], "alignment": "center"},
            "set_background": {"color": "#f4efe6"},
            "list_fonts": {"query": "sans"},
            "write_html": {"html": "<div id=\"canvas\">…</div>"},
        }
        return json.dumps(examples.get(tool, {}), indent=2)

    def read_html(ep_id):
        ep = STORE.maybe(ep_id)
        if ep is None:
            raise gr.Error("Start a task first.")
        return ep.html()

    def write_html(ep_id, html):
        text, _, img = call_tool(ep_id, "write_html", json.dumps({"html": html}))
        return text, img

    def replay_expert(ep_id, progress=gr.Progress()):
        ep = STORE.maybe(ep_id)
        if ep is None:
            raise gr.Error("Start a task first.")
        if ep.done:
            raise gr.Error("This episode is finished; load the task again.")
        gt = pack.gt_doc(ep.task)
        lines = []
        if ep.realapp is not None:  # a real app: the ground truth is written into the app's own document
            backend, session = ep.realapp
            backend.write_doc(session, gt)
            lines.append(f"Wrote the {len(gt.elements)} target elements into {REAL_SKINS[ep.skin][0]}. "
                         "Submit to read the app's document back and score it.")
            return "\n".join(lines), render_now(ep_id)
        if ep.surface == "gui":  # the built-in editor on computer use: the editor shows the document as it changes
            ep.apply("replace_document", {"document": gt.model_dump(mode="json")}, "expert")
            return f"Set the editor's document to the {len(gt.elements)} target elements.", render_now(ep_id)
        if ep.surface == "html":
            text, _, _ = call_tool(ep_id, "write_html", json.dumps({"html": expert.html_page(gt)}))
            lines.append(f"write_html → {text[:120]}")
        else:
            calls = [c for c in expert.tool_calls(gt) if c["name"] != "submit"]
            for c in progress.tqdm(calls, desc="Replaying"):
                text, _, _ = call_tool(ep_id, c["name"], json.dumps(c["arguments"]))
                lines.append(f"{c['name']}({json.dumps(c['arguments'], ensure_ascii=False)[:90]}) → {text[:60]}")
        return "\n".join(lines), render_now(ep_id)

    def submit(ep_id):
        ep = STORE.maybe(ep_id)
        if ep is None:
            raise gr.Error("Start a task first.")
        pg.call(ep_id, "submit", {})
        g = ep.result
        rows = [("Structure", g.structure, CONFIGS[g.mode].w_structure),
                ("Visual", g.visual, CONFIGS[g.mode].w_visual)]
        if g.mode == "description":
            rows.append(("Coverage", g.coverage, CONFIGS[g.mode].w_coverage))
        rows += [("Order", g.order, CONFIGS[g.mode].w_order),
                 ("Background", g.background, CONFIGS[g.mode].w_background)]
        table = "\n".join(f"| {n} | {v:.3f} | × {w:.2f} |" for n, v, w in rows)
        md = (f"### Reward {g.total:.3f}\n\nMatched **{g.n_matched}/{g.n_gt}** target elements, "
              f"{g.n_submitted} submitted.\n\n| Term | Score | Weight |\n|---|---|---|\n{table}")
        return md, g.as_dict(), render_now(ep_id)

    def trajectory(ep_id):
        ep = STORE.maybe(ep_id)
        return {"episode_id": ep_id, "steps": len(ep.log), "log": ep.log} if ep else {}

    with gr.Blocks() as demo:
        # OpenEnv wraps this Blocks in its own app, which drops a nested css=;
        # a style tag still applies. It lets the playground (and the editor in
        # it) use the full window width instead of Gradio's centred column.
        gr.HTML("<style>.gradio-container{max-width:100% !important;padding-left:16px !important;"
                "padding-right:16px !important}.gradio-container .main,.gradio-container .wrap"
                "{max-width:100% !important}</style>", padding=False)
        gr.Markdown("# DesignGym — try it")
        gr.Markdown(INTRO)
        episode = gr.State(None)
        with gr.Row():
            split = gr.Dropdown(splits, value=default_split, label="Split", scale=1)
            index = gr.Number(value=0, precision=0, label="Task index", minimum=0, scale=1,
                              maximum=pack.count(default_split) - 1, info=index_range(default_split))
            mode = gr.Radio(MODE_CHOICES, value="reference", label="Mode", scale=2)
            surface = gr.Radio(SURFACE_CHOICES, value="tools", label="Surface", scale=2)
            skin = gr.Dropdown(skin_choices("tools"), value="figma", label="Editor", scale=2,
                               info="Real apps (Impress, Draw, Inkscape) are listed under Computer use"
                               if apps_available()[0] else "Real apps are not set up on this server")
        with gr.Row():
            prev_btn = gr.Button("← Previous")
            load_btn = gr.Button("Load index")
            next_btn = gr.Button("Next →")
            shuffle_btn = gr.Button("🎲 Shuffle task", variant="primary")
        info = gr.Markdown("*No task yet — click “Shuffle task”.*")

        # The editor gets the whole width; everything else sits around it.
        with gr.Tabs():
            with gr.Tab("Editor"):
                editor = gr.HTML(editor_frame(None, "figma"))
                # a real app runs behind the server: its screen, driven through the same GUI tools the agent has
                with gr.Column(visible=False) as app_box:
                    screen = gr.Image(label="App screen (what the agent sees; click it to act there)", type="pil",
                                      interactive=False, buttons=[])
                    with gr.Row(equal_height=False):
                        click_kind = gr.Radio(CLICK_CHOICES, value="left_click", label="Clicking the screen does",
                                              scale=2)
                        type_box = gr.Textbox(label="Text", placeholder="typed at the keyboard focus", scale=3)
                        type_btn = gr.Button("Type", scale=1, size="md")
                        key_box = gr.Textbox(label="Key", placeholder="enter, escape, ctrl+z, shift+up", scale=2)
                        key_btn = gr.Button("Press", scale=1, size="md")
                        screen_btn = gr.Button("Refresh screen", scale=1, size="md")
                    screen_out = gr.Markdown()
            with gr.Tab("Tool call"):
                with gr.Row():
                    tool = gr.Dropdown([], label="Tool", scale=3)
                    run_btn = gr.Button("Call tool", variant="primary", scale=1)
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1):
                        args = gr.Code(label="Arguments (JSON)", language="json", value="{}",
                                       lines=10)
                        tool_out = gr.Markdown()
                    tool_img = gr.Image(label="Tool image output", type="pil", height=420,
                                        interactive=False, scale=1)
            with gr.Tab("HTML"):
                html_hint = gr.Markdown("Load a task with **Surface = HTML** to use `write_html`.",
                                        visible=True)
                html_code = gr.Code(label="Page", language="html", lines=22)
                with gr.Row():
                    read_btn = gr.Button("Read current page")
                    write_btn = gr.Button("write_html", variant="primary")
                html_out = gr.Markdown()
            with gr.Tab("Expert"):
                gr.Markdown("Replays the ground truth as the tool calls (or the one HTML page) that build it: the SFT "
                            "trajectory, which scores 1.000. On computer use it sets the editor's document; in a real "
                            "app it writes the design into the app's own document, which Submit then reads back.")
                expert_btn = gr.Button("Replay expert trajectory", variant="primary")
                expert_out = gr.Textbox(label="Calls", lines=16)
            with gr.Tab("Trajectory"):
                traj_btn = gr.Button("Refresh")
                traj = gr.JSON()

        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=260):
                target = gr.Image(label="Target (render of the ground truth)", type="pil",
                                  height=360, interactive=False)
                brief = gr.Textbox(label="Design brief", lines=16, visible=False,
                                   interactive=False)
            with gr.Column(scale=1, min_width=260):
                canvas = gr.Image(label="Current canvas (as scored)", type="pil", height=360,
                                  interactive=False)
            with gr.Column(scale=1, min_width=260):
                refresh_btn = gr.Button("↻ Re-render canvas")
                read_app_btn = gr.Button("Read the document from the app", visible=False)
                submit_btn = gr.Button("✅ Submit & score", variant="primary")
                result = gr.Markdown("*Not submitted yet.*")
                with gr.Accordion("Score breakdown (per element)", open=False):
                    details = gr.JSON()
        with gr.Accordion("Assets", open=False):
            assets = gr.Gallery(label="Assets (id under each)", columns=8, height=240,
                                object_fit="contain", allow_preview=True)
        with gr.Accordion("Real apps on this server", open=False):
            apps_md = gr.Markdown(apps_table(False))
            apps_btn = gr.Button("Check what is up")
        with gr.Accordion("Instructions the agent receives", open=False):
            instructions = gr.Markdown()
        with gr.Accordion("How the reward is computed", open=False):
            gr.Markdown(SCORING)

        outputs = [episode, index, info, target, brief, assets, editor, canvas, instructions, tool,
                   tool_out, tool_img, html_out, result, details, html_hint, app_box, screen, read_app_btn]
        load_btn.click(load, [split, index, mode, surface, skin, episode], outputs)
        shuffle_btn.click(shuffle, [split, mode, surface, skin, episode], outputs)
        prev_btn.click(lambda s, i, m, su, sk, e: step_index(s, i, -1, m, su, sk, e),
                       [split, index, mode, surface, skin, episode], outputs)
        next_btn.click(lambda s, i, m, su, sk, e: step_index(s, i, 1, m, su, sk, e),
                       [split, index, mode, surface, skin, episode], outputs)
        split.change(lambda s: gr.update(maximum=pack.count(s) - 1, info=index_range(s)), [split], [index])
        tool.change(tool_template, [tool], [args])
        run_btn.click(call_tool, [episode, tool, args], [tool_out, tool_img, canvas]).then(
            screen_now, [episode], [screen])
        read_btn.click(read_html, [episode], [html_code])
        write_btn.click(write_html, [episode, html_code], [html_out, canvas])
        expert_btn.click(replay_expert, [episode], [expert_out, canvas])
        traj_btn.click(trajectory, [episode], [traj])
        refresh_btn.click(render_now, [episode], [canvas])
        submit_btn.click(submit, [episode], [result, details, canvas])
        def pick_skin(ep_id, sk):
            # the built-in editor's layout can change on a live episode; a real app needs a new one (Load)
            ep = STORE.maybe(ep_id)
            live = ep is not None and ep.skin not in REAL_SKINS and sk not in REAL_SKINS
            return editor_frame(ep_id, sk) if live else editor_frame(None, sk)

        def pick_surface(su, sk):
            # only offer the editors this surface can use; a real app falls back to the default layout
            return gr.update(choices=skin_choices(su), value=sk if su == "gui" or sk not in REAL_SKINS else "figma")

        skin.change(pick_skin, [episode, skin], [editor])
        surface.change(pick_surface, [surface, skin], [skin])
        screen.select(click_screen, [episode, click_kind], [screen, screen_out])
        type_btn.click(lambda e, t: act_on_screen(e, "type", {"text": t}), [episode, type_box], [screen, screen_out])
        key_btn.click(lambda e, k: act_on_screen(e, "key", {"keys": k}), [episode, key_box], [screen, screen_out])
        screen_btn.click(lambda e: (screen_now(e), ""), [episode], [screen, screen_out])
        read_app_btn.click(read_from_app, [episode], [canvas])
        apps_btn.click(lambda: apps_table(True), [], [apps_md])
    return demo
