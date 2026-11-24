# Running DesignWorldBench v2 (Recreation-CUA, workspace)

v2 is the computer-use benchmark as the DesignWorld-SFT traces were recorded: the 30 core tasks, in all 14 editors,
each episode frozen in full so every model meets the same one.

- **Editors (14):** the 6 built-in (figma, canva, photoshop, slides, powerpoint, inkscape), the public apps
  (inkscape-desktop, impress, draw) and the private apps (penpot, drawio, scribus, onlyoffice, photocraft).
- **A lived-in computer:** the task's images have everyday names (`blue-diagonal-shape-background.png`,
  `IMG_4021.jpg`, `image (7).png`) in everyday folders (Downloads, Desktop, Documents, a project folder), next to
  decoy pictures from other designs and a few documents. The prompt names the files, not where they are; the model
  finds them, previews them in the file window, brings them in, and saves or exports before it submits.
- **No announced budget:** the prompt asks for efficient work and names no number. A limit of 1000 actions,
  400 turns and 60 minutes per episode is still enforced, as a guard.
- **Scored with reward v3**, reference mode. v1 (`recreation-cua-v1.json`) stays as it is.

The set is `bench/evalsets/recreation-cua-v2.json` (420 episodes = 30 tasks × 14 editors). The episodes need an
offline task pack with the set's tasks and the designs its decoys come from (60 tasks, about 70 MB).

## 1. Install

The private package installs the public DesignGym from a path next to it
(`../FineEnvs-worktrees/design-world/08-design-world/envs/designgym`), so clone both into one folder like this:

```bash
mkdir designworld && cd designworld
git clone -b design-world https://github.com/adithya-s-k/FineEnvs FineEnvs-worktrees/design-world
git clone https://github.com/adithya-s-k/designworld-private          # private: needs access
cd designworld-private && uv sync
```

Run every command below from `designworld-private/`.

With only the public code, install `envs/designgym` (`uv sync --extra rollout --extra daytona` there) and run
`--editors public`: the 6 built-in editors and Inkscape, Impress, Draw.

Keys go in a `.env` in `designworld-private/` (never commit it; `--env-file` points at another):

```
ANTHROPIC_API_KEY=...      OPENAI_API_KEY=...      HF_TOKEN=...          # models; HF_TOKEN also uploads results
DAYTONA_API_KEY=...        DAYTONA_API_URL=...                           # the real apps run in Daytona sandboxes
```

## 2. Get the eval pack

```bash
hf download perpetual-research/designworldbench-v2-pack --repo-type dataset --local-dir ~/designworldbench-v2-pack
```

## 3. Start the environment server

```bash
set -a; source .env; set +a
DESIGN_CANVAS_REWARD=3 DESIGN_CANVAS_APPS=daytona DESIGN_CANVAS_DAYTONA_POOL=eval-$USER DESIGN_CANVAS_DAYTONA_DESKTOPS=6 \
  .venv/bin/designgym serve --pack ~/designworldbench-v2-pack --port 8130
curl -s localhost:8130/api/info        # 14 skins listed, reward_version 3, the pack's evalset_id
```

The server starts real-app desktops in Daytona on demand (labelled `pool=eval-$USER`) and stops them when it exits.
Leftovers: `.venv/bin/python -m designgym.realapps.daytona down` with the same `DESIGN_CANVAS_DAYTONA_POOL`.

## 4. Run

```bash
B=../FineEnvs-worktrees/design-world/08-design-world/bench
# smoke first: one easy task in every editor
.venv/bin/python $B/rollout.py --base-url http://localhost:8130 --models anthropic:claude-sonnet-5-5 \
    --task-ids test-00209 --out runs/smoke-sonnet55
# the full set for a model (420 episodes); rerun the same command to resume after an interruption
.venv/bin/python $B/rollout.py --base-url http://localhost:8130 --models openai:gpt-6-astra@low \
    --editors all --concurrency 6 --app-concurrency 2 --out runs/v2-gpt6-astra-low
```

- `--models`: comma list of `provider:model[@thinking]`: `anthropic:<model>`, `openai:<model>` (Responses API),
  `openai-chat:<model>`, `hf:<org/model>:<provider>` (HF router), `vllm:<model>` (your own server at
  `VLLM_BASE_URL`). `@off` / `@low` set thinking where the provider supports it.
- `--editors`: `all`, `public`, `builtin`, `apps`, `private`, or a comma list of editors.
- `--task-ids` / `--tasks N`: a subset of the 30.
- `--concurrency` is episodes at once; `--app-concurrency` caps each real app (keep the total of real-app
  episodes within `DESIGN_CANVAS_DAYTONA_DESKTOPS`).
- A run folder holds `trace.jsonl` (every turn, with the system prompt and tool schemas), `frames/`, `renders/`,
  `run.json` and `summary.md` (mean reward per editor and tier, 95% intervals, tokens). Episodes that errored are
  run again on resume; finished ones are kept.

## 5. Publish (private, Perpetual org)

```bash
.venv/bin/python $B/publish.py runs/v2-gpt6-astra-low --dry-run     # counts
.venv/bin/python $B/publish.py runs/v2-gpt6-astra-low               # -> perpetual-research/designworldbench-runs
```

Each run becomes one config of `perpetual-research/designworldbench-runs` (private): `data/<run>/` holds one row
per episode in the cua-v1 format of the DesignWorld-SFT data (the full conversation with every screenshot, the
grade, the submitted document, the episode's editor look and workspace), `meta/<run>.json` the counts and summary,
and `runs/<run>/` the raw trace and settings.

## Costs and time (core30 × 14 editors = 420 episodes per model)

Frontier models take roughly 60–150 turns an episode here; a full model run is a few hours at concurrency 6, and
the real apps add about $0.33 an hour per Daytona desktop. Run the smoke first and read its `summary.md`: every
editor should show a reward, and `errors` should be 0.
