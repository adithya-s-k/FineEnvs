---
title: RetroEnv
emoji: ⚗️
colorFrom: gray
colorTo: green
sdk: docker
app_port: 8000
base_path: /web
pinned: false
tags: [openenv, chemistry, retrosynthesis, tool-use]
datasets:
  - LiteFold/RetroEnv
short_description: Plan retrosynthesis routes with verifier-backed rewards.
---

# RetroEnv: OpenEnv server

An agent gets a target molecule and works back to purchasable starting materials. It uses tools to inspect molecules, search a fixed stock and training precedents, and check disconnections. It ends the episode by submitting molecule/reaction trees with `emit_routes`, and the verifier judges every step against a frozen reaction library; matching the patent route only adds a bonus.

The same server is used for training, evaluation and the browser playground at `/web`.

## Read the code

- [`retroenv_openenv/environment.py`](retroenv_openenv/environment.py): the MCP tools, the Task API, `reset`, and the terminal reward.
- [`retroenv_openenv/client.py`](retroenv_openenv/client.py): `RetroEnvClient` for one episode per WebSocket session, and `RemoteRetroRouteEnv` for TRL's `environment_factory`.
- [`retroenv_openenv/agent.py`](retroenv_openenv/agent.py), [`agent_anthropic.py`](retroenv_openenv/agent_anthropic.py), [`agent_responses.py`](retroenv_openenv/agent_responses.py): model loops for OpenAI-compatible chat, the Claude Messages API and the OpenAI Responses API.
- [`retroenv_openenv/ui.py`](retroenv_openenv/ui.py): the playground.
- [`../core/retroenv`](../core/retroenv): chemistry, stock search, tasks and the verifier. Every front-end shares it.

## The episode

| Step | What happens |
|---|---|
| `reset(split, index)` | Returns the prompt, target, depth budget, route count, constraints and tool names. It never returns known routes, patents or the stock list. |
| tool calls | MCP `call_tool`. Up to 32 calls per episode (`RETROENV_MAX_TOOL_CALLS`); `emit_routes` always stays available. |
| `emit_routes(submission)` | Scores the trees. That step returns `done=True` with the dense reward, once. |

| Tool | Answers from |
|---|---|
| `inspect_molecule`, `pubchem_lookup` | RDKit on the given SMILES |
| `stock_retrieve` | The task's stock (excluded building blocks absent): exact, InChIKey, class, SMARTS or similarity search, capped at 20 results |
| `reaction_precedent_search`, `search_literature`, `reaction_conditions_search` | Train-visible corpus reactions only |
| `validate_disconnection` | Train-visible reactions and frequent templates, never this task's hidden routes. Removed by `RETROENV_TOOLSET=unaided`. |
| `reaction_class_lookup` | The verifier's own reaction classifier |
| `emit_routes` | The verifier |

The Task API lists splits and public task rows without references: `GET /retro_route/splits`, and `POST /retro_route/task`, `/tasks`, `/task_range` and `/num_tasks`. The API docs are at `/docs`.

## Data and credits

The tasks, known routes, difficulty labels, stock and reaction library are
[LiteFold/RetroEnv](https://huggingface.co/datasets/LiteFold/RetroEnv), the dataset built and
released by [LiteFold](https://huggingface.co/LiteFold). This server serves it unchanged.

It derives from PaRoutes v2: S. Genheden and E. Bjerrum, *PaRoutes: towards a framework for
benchmarking retrosynthesis route predictions*, Digital Discovery 2022
([doi:10.1039/D2DD00015F](https://doi.org/10.1039/D2DD00015F)); data on
[Zenodo 7341155](https://zenodo.org/records/7341155), CC BY 4.0. PaRoutes takes its reactions from
D. Lowe, *Chemical reactions from US patents (1976 to Sep 2016)*
([figshare](https://doi.org/10.6084/m9.figshare.5104873.v1)), CC0. Software: RDKit, rdchiral,
3Dmol.js and OpenEnv.

## Data and splits

The server serves a pinned snapshot of [LiteFold/RetroEnv](https://huggingface.co/datasets/LiteFold/RetroEnv)
from the public bucket [FineEnvs/retroenv-bucket](https://huggingface.co/buckets/FineEnvs/retroenv-bucket).
`corpus-manifest.json`, beside the server code, pins it: the release revision, the serving index and the
SHA-256 of every file a server reads.

| Split | Tasks | Use |
|---|---:|---|
| `train` | 75,224 | RL rollouts |
| `dev` | 1,079 | checkpoint selection |
| `test_id` | 1,083 | unseen targets, training distribution |
| `test_hard` | 1,119 | longer, novel or rarer chemistry |
| `final_eval` | 50 | 17 easy, 17 medium, 16 hard from the two test splits (`dataset/build_final_eval.py`) |

`final_eval` names tasks of the test splits, so they score exactly as there.

## Run locally

From the project folder (`09-retroenv`):

```bash
uv sync --extra dev --extra eval
uv run bash envs/retro_route/openenv/start.sh
```

`start.sh` runs `prepare.py`, which downloads the pinned snapshot from the bucket (about 115 MB,
once), checks every file's SHA-256, and starts the server; a restart reuses the checked files.
Startup then takes about 10 s. To serve a local release directory without an index instead:

```bash
RETROENV_BENCHMARK_DIR=data/release/RetroEnv-RL uv run uvicorn retroenv_openenv.server:app --port 8000
```

Open <http://localhost:8000/web/> for the playground. Its Plan tab works the way the task does: pick
the molecule still to make, take a cut the rule library proposes (or type one), check it and its
pieces with the real tools, and add it to a route drawn as it grows; "Fill from the plan" turns that
route into the `emit_routes` submission. Each molecule is drawn in 2D and, from an RDKit conformer,
in 3D (vendored [3Dmol.js](https://3dmol.csb.pitt.edu/), so it works offline). To run one model
episode against the server:

```bash
uv run python envs/retro_route/openenv/rollout.py --server http://127.0.0.1:8000 \
  --split final_eval --index 0 --provider anthropic --model claude-opus-5-5
```

## Docker and Spaces

The image holds code and `corpus-manifest.json` only. On a Space the bucket is mounted read-only at
`/data` and `prepare.py` copies the snapshot from there; with nothing mounted it downloads it.

```bash
python deploy.py --stage-only --stage-dir /tmp/retroenv-space
docker build -t retroenv /tmp/retroenv-space
docker run --rm -p 8000:8000 retroenv                       # downloads the pinned snapshot
```

[FineEnvs/retroenv](https://huggingface.co/spaces/FineEnvs/retroenv) is deployed with:

```bash
python deploy.py --repo FineEnvs/retroenv --public
```

`deploy.py` refuses to deploy unless every file the manifest names is already in the bucket, mounts
the bucket, and sets the variables below. A new snapshot comes from `dataset/publish_bucket.py`,
which also rewrites `corpus-manifest.json`.

| Variable | Default | Meaning |
|---|---|---|
| `RETROENV_CORPUS_MANIFEST` | `corpus-manifest.json` here | The serving snapshot to fetch and check |
| `RETROENV_BUCKET_ROOT` | `/data` in the image | A mounted copy of the bucket; files missing there are downloaded |
| `RETROENV_BUCKET_ID` | the manifest's | The bucket to download from, if it was renamed |
| `RETROENV_PREPARED_DIR` | `prepared` | Where the checked snapshot is kept between restarts |
| `RETROENV_BENCHMARK_DIR` | none | A local release or prepared directory, used as is |
| `RETROENV_TASKS_REPO` | none | A release dataset `org/name[@revision]`, served without an index |
| `RETROENV_TOOLSET` | `full` | `unaided` removes `validate_disconnection` (an ablation) |
| `RETROENV_MAX_TOOL_CALLS` | `32` | Tool budget per episode |
| `RETROENV_DEFAULT_SPLIT` | `train` | Split used by a reset that names none |
| `MAX_CONCURRENT_ENVS` | `64` | WebSocket sessions per process |
| `ENABLE_WEB_INTERFACE` | `true` | Mount the playground at `/web` |
