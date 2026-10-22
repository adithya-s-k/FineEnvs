---
title: DesignGym
emoji: 🎨
colorFrom: indigo
colorTo: pink
sdk: docker
app_port: 8000
base_path: /web
pinned: false
tags: [openenv, mcp, computer-use, graphic-design, designworld]
short_description: Recreate real graphic designs via MCP, HTML or a real editor
---

# DesignGym

An [OpenEnv](https://github.com/huggingface/OpenEnv) environment for rebuilding real graphic designs from
their parts, part of [DesignWorld](../../README.md). Each task is a design template taken apart into its
pre-rendered layers and its typed copy. The agent puts it back together on a canvas and is scored against a
render of the original.

The package is `designgym` (CLI `designgym`). It used to be `design_canvas`: `import design_canvas` still works
(an alias of `designgym`), the `design-canvas` command still exists, and the OpenEnv name (Task API
`/design_canvas/...`) and the `DESIGN_CANVAS_*` settings are unchanged, so clients, datasets and traces keep
working.

- **Two tasks.** `mode="reference"`: here is the target image, recreate it. `mode="description"`: here is a
  written brief naming every asset and line of copy, build it.
- **Three surfaces over one document.** `surface="tools"`: structured MCP edits. `surface="html"`: one page of
  restricted HTML. `surface="gui"`: mouse and keyboard on a real browser editor, seeing only screenshots.
- **One renderer, one reward.** Every surface edits the same document, and every score comes from the same
  skia renderer, so a GUI episode and a tool-call episode are directly comparable.

## Run it

```bash
uv sync --extra dev
designgym pull                # the task pack from the Hub (DESIGNGYM_PACK_REPO); --splits test for the eval set only
designgym serve --port 8008   # OpenEnv server + editor + playground
```

- Editor: `http://127.0.0.1:8008/editor/` · Playground: `/web/` · MCP: `/mcp` · Task API: `/design_canvas/...`
- Or `docker build -t designgym . && docker run -p 8000:8000 designgym` (pulls the pack at first start).

The task pack has 4k tasks in three splits: `sft` (2k designs, the ones DesignWorld-SFT is recorded on), `rl` (2k
more, disjoint) and `test` (the frozen DesignWorldBench set). Each task holds its ground truth, assets, brief and
difficulty; the data licence is in [`NOTICE.md`](NOTICE.md) and travels with every export.

```bash
designgym evalset verify ../../bench/evalsets/recreation-cua-v1.json                      # the set against the pack
designgym evalset export ../../bench/evalsets/recreation-cua-v1.json --out artifacts/evalpack  # offline copy
```

Trajectories in `cua.jsonl` form (eval runs: [`bench/harness/to_cua.py`](../../bench/harness/to_cua.py)) are
written as a Hub-ready `cua-v1` dataset by `cua_format.py`: OpenAI-style messages with pixel tool calls, embedded
screenshots, per-message `trainable`; `windows()` derives SFT rows with the same screenshot window the RL adapters
use.

```bash
designgym export-hf --cua run/cua.jsonl --out artifacts/hf/cua    # cua-v1 Parquet + dataset card
```

Check a running server end to end (HTTP routes, target gating, expert replays scoring 1.0 on tools and HTML in both
modes, an empty submission scoring 0, concurrent episodes; the client must load the same task pack as the server):

```bash
.venv/bin/python -m designgym.verify_server --server http://127.0.0.1:8008
```

Watch Claude work, live:

```bash
claude mcp add designgym -- designgym mcp
```

then ask it to "start a design task and recreate it", and open the `editor_url` that `start_task` returns.

For more throughput, `designgym serve --workers N` (or `DESIGN_CANVAS_WORKERS=N`) runs N server processes behind
one URL ([`server/router.py`](server/router.py)). Don't use uvicorn's own `--workers`: OpenEnv's client creates a
session over HTTP and then attaches a WebSocket to it, and the built-in editor's page calls back for its episode,
so both must reach the process that holds the state. The router keeps every session and episode on its worker.
Measured locally (a load test, 4 workers, no errors):

- tools: about 2,900 episodes per minute, 3× one process;
- built-in GUI editor: 128 concurrent episodes at about 190 actions per second, 2× one process.

The router itself saturates one core at about 300–450 tool calls per second, so 4–6 workers is the useful range.

On the Space [FineEnvs/DesignGym](https://huggingface.co/spaces/FineEnvs/DesignGym) (cpu-upgrade, 8 vCPU / 32 GB,
4 workers × 48 sessions), tested from India, where the network round trip is about 0.22 s:

| Load | Result | Throughput | Step p50 / p95 |
|---|---|---|---|
| Tools, 192 concurrent | 575/576 ok | 643 episodes/min, 110 calls/s | 0.26 / 0.8 s |
| Built-in GUI editor, 128 concurrent | 256/256 ok | 75 actions/s | 0.46 / 1.3 s |

- The one failure arrived at the capacity edge, just before a closing session was freed.
- At capacity, cold resets take 7–10 s; each new design is fetched once.
- More than 192 sessions are refused cleanly ("Server at capacity") after every worker has been tried.
- OpenEnv's client has no option for auth headers: a client of a private Space needs to add them itself.

## Talk to it

```python
from openenv.core.mcp_client import MCPToolClient
from openenv.core.env_server.mcp_types import CallToolAction

with MCPToolClient(base_url="http://127.0.0.1:8007").sync() as env:
    obs = env.reset(split="test", index=3, mode="reference", surface="tools")
    print(obs.observation.metadata["instructions"])     # the task, as the model should see it
    tools = env.list_tools()                           # depends on the surface
    env.step(CallToolAction(tool_name="add_image", arguments={"asset_id": "a2", "x": 0, "y": 0, "width": 1080}))
    out = env.step(CallToolAction(tool_name="submit", arguments={}))
    print(out.reward, out.observation.metadata["grade"])
```

`reset` arguments: `split`, `index` or `task_id`, `mode` (`reference` | `description`), `surface`
(`tools` | `html` | `gui`), `skin` (`figma` | `canva` | `photoshop` | `slides` | `powerpoint` | `inkscape` editor layouts; random per `seed`;
or a real app, below),
`theme`, `file_dialog` (`mac` | `windows` | `linux`: the look of the file window images are opened from; random per `seed`),
`coordinates` (`pixels`, or `normalized` for a 0-1000 grid), `max_steps`,
`reward_mode` (`final`, or `dense` for per-edit score deltas that sum to the final score).

The Task API lists tasks without giving them away: `GET /design_canvas/splits`,
`POST /design_canvas/num_tasks`, `/task`, `/task_range`.

## Tools

| Surface | Tools |
|---|---|
| all | `get_task`, `view_reference`, `submit`, `start_task` (for MCP clients with no `reset`) |
| tools, html | `view_canvas`, `view_assets` (a labelled contact sheet), `list_fonts` |
| tools | `add_image`, `add_text`, `add_shape`, `update_element`, `delete_element`, `duplicate_element`, `reorder_element`, `align_elements`, `set_background`, `list_elements`, `undo`, `redo` |
| html | `write_html`, `read_html` |
| gui | `screenshot`, `left_click`, `double_click`, `triple_click`, `right_click`, `mouse_move`, `left_click_drag`, `scroll`, `type`, `key`, `wait` |

Images come back as MCP image content blocks, so a vision model sees pixels, not base64.

## Real apps

The GUI surface can also put the model in a real design app instead of the built-in editor: `skin="impress"`
(LibreOffice Impress), `"draw"` (LibreOffice Draw) or `"inkscape-desktop"` (Inkscape), each on a Linux desktop. The
tools, the prompt's task and the reward are the same. At `reset` the env sets the app up with a blank canvas at the
task's size and the task's assets loaded. On `submit` it reads the app's own document back (the slide, the page,
the SVG) as a design document and scores it like any other.

The client always talks to one env server, and that server runs the apps:

- **Locally.** Run the desktop image from `realapps/desktop/`. The server finds desktops at
  `DESIGN_CANVAS_DESKTOP_URLS` (one episode per desktop at a time).
- **On Daytona.** Set `DESIGN_CANVAS_APPS=daytona` and `DAYTONA_API_KEY`. The server keeps its own pool of
  private sandboxes (`realapps/daytona.py`): one per running desktop episode (`DESIGN_CANVAS_DAYTONA_DESKTOPS`,
  default 2, built from a snapshot of the desktop image). It reaches them through their preview URLs with the
  access token in a header. Sandboxes are reused across server restarts and stop when idle.
  `python -m designgym.realapps.daytona up | status | down` starts, lists or deletes them.

A real-app episode's `reset` observation carries `app` (`{"skin", "name", "real": true}`), and its `editor_url` is
the app's live screen (`/api/episodes/{id}/screen.png`). `GET /api/apps` lists the apps and where they run
(`?check=1` also asks each endpoint or sandbox whether it is up). In the `/web` playground, pick a real app under
**Editor**: the Editor tab shows its screen, and a click on the screen is a click in the app. The Expert tab
writes the target into the app, and **Read the document from the app** shows what Submit will score.

Expert round trips (the ground truth written into the app and read back) score 1.000 in Inkscape and 0.975 to
0.999 in Impress.

### More apps: plugins

Other app backends register from outside, through [`realapps/registry.py`](realapps/registry.py): a package
declares an entry point in the group `designgym.apps` (or is named in `DESIGNGYM_APP_PLUGINS=module:function`)
whose function calls `register(App(...))`. A desktop app names the desktop agent's app it uses and, if the public
image lacks it, registers its own desktop image and agent-side modules with `register_desktop(...)` (the agent
loads `*.py` from its `apps/` folder). A web app gives a backend that opens its page in the env's browser. The
registered skins then work in `reset(skin=...)`, the prompts, the playground and `/api/apps` like the built-in
ones.

## Reward

`0.60·structure + 0.30·visual + 0.05·order + 0.05·background` in reference mode. Structure matches elements
one-to-one (Hungarian) on content, geometry and typography; visual is multi-scale SSIM and colour layout against
`render(ground truth)`, normalised so an empty canvas scores 0. Description mode loosens geometry and adds
coverage. Replaying the ground truth scores exactly 1.000 on every task. Details:
[`core/score.py`](core/score.py) and the project README.

`DESIGN_CANVAS_REWARD=3` (or `grade(..., version=3)`) selects reward v3 for reference mode, calibrated against
a pairwise visual judge (Claude Opus 5.5, both orders) on frontier and Qwen attempts across all three surfaces:

- **Coarse-to-fine geometry.** Half the tight kernel, half a loose one (15% of the diagonal), scaled by √position,
  so a roughly placed element in the right region earns credit, but 1.0 still needs exact placement.
- **Typography that multiplies.** Font and colour scale the rest of the style match, and style scales placement.
- **Copy counts triple in structure.** Every unmatched submitted element counts against the score.
- **Gated pixel terms.** Visual, order and background count in proportion to min(1, structure/0.3), so colour
  patches and scattered dumps (~0.25 under v1) score ~0.

Results:

- **Validation:** agrees with the judge on 89.9% of 556 pairs vs 85.4% for v1 (+33/−8, p<0.001), and on 91.4%
  of Qwen pairs.
- **Held-out test tasks:** 88.4% vs 84.2% (p=0.005), including 271 GUI pairs.
- **Description mode:** unchanged.

## Layout

```
core/      document model, ops, fonts, skia renderer, reward, HTML profile, briefs, expert, task pack, eval set
server/    OpenEnv environment, editor API + live sync, GUI driver (Playwright), Gradio playground, stdio MCP
realapps/  real apps behind the GUI surface: registry (plugins), desktop agent (Impress, Draw, Inkscape), Daytona pool
web/       the editor: vanilla JS, six layouts (Studio / Template / Pixel / Deck / Present / Vector), no build step
scripts/   view_cua.py (browse a cua.jsonl run)
compat/    the `design_canvas` alias
tests/     core, reward invariants over every task, OpenEnv contract, browser E2E, GUI agent, registry
rollout.py real models on every surface (Anthropic, OpenAI, Hugging Face router, vLLM)
```

Code: Apache-2.0. Data: see [`NOTICE.md`](NOTICE.md).
