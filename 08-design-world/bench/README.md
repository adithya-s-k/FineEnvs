# DesignWorldBench

The DesignWorld benchmark: can an agent rebuild a real design in a design tool? An agent gets a target design and
rebuilds it in [DesignGym](../envs/designgym) through tool calls, HTML, or mouse and keyboard. The environment
grades the result against the design's layers.

**v1 is *Recreation-CUA*.** It has two parts: a frozen eval set and the harness that runs it. Every eval run is
published as a computer-use (CUA) dataset in the `cua-v1` format.

**v2 is *Recreation-CUA* as the DesignWorld-SFT traces were recorded:** the core30 tasks in all 14 editors, every
episode frozen in full (editor look, file window, and a lived-in workspace of everyday file names, decoy pictures
and documents), with no announced action budget. Run it with `bench/rollout.py` and publish with
`bench/publish.py`; the step-by-step is [RUN_V2.md](RUN_V2.md).

```
bench/
├── evalsets/recreation-cua-v1.json   the frozen set: 300 test designs, subsets lite and core30
├── evalsets/recreation-cua-v2.json   v2: 420 frozen episodes (core30 x 14 editors), each with its workspace
├── rollout.py     the v2 runner (models x episodes, resume, per-app concurrency, summary)
├── publish.py     a v2 run -> cua-v1 rows (+ raw trace) -> perpetual-research/designworldbench-runs (private)
├── RUN_V2.md      how to run v2: install, pack, server, run, publish
└── harness/
    ├── launch.py      run the bench on Hugging Face Jobs (vLLM + DesignGym + rollout.py)
    ├── eval_job.sh    what the job runs
    ├── resume.py      carry a previous run's episodes into a new run
    └── to_cua.py      a run directory -> cua-v1 rows -> FineEnvs/designworld-bench-runs
```

The episode loop is `envs/designgym/rollout.py`. It lives in the environment package, and the bench calls it.

## The eval set

`evalsets/recreation-cua-v1.json` holds 300 designs from the DesignGym `test` split, frozen by task id:

| | size | easy | medium | hard |
|---|---:|---:|---:|---:|
| `full` | 300 | 100 | 100 | 100 |
| `lite` | 60 | 20 | 20 | 20 |
| `core30` | 30 | 10 | 10 | 10 |

`core30` is a subset of `lite`. `lite` was picked from the full set by the same rules, so it is a smaller sample
of the same distribution.

- **Selection.** The set is stratified by difficulty tier and design group. It caps how many designs come from one
  appearance cluster, and it guarantees a quota of hard features: rotation, transparency, mixed-style text, dense
  layouts, long copy, tiny text, reused assets, three or more fonts, and no photo. Every tie is broken by a seeded
  hash.
- **Validation.** Each design's ground truth scores exactly 1.0 against itself.
- **Task entries.** Each entry has `task_id`, `design_id` (the source design), `tier`, a difficulty `score`,
  layout and content features, `doc_sha256` (the ground truth's digest) and `flags`.
- **Identity.** `evalset_id` is a hash of the snapshot, split, seed and task ids. `designgym.core.evalset.load`
  refuses a file whose ids don't match it.

## What is measured

| | |
|---|---|
| **Surfaces** | `tools`: structured edit calls over MCP. `html`: one page of restricted HTML. `gui`: mouse and keyboard on an editor, seeing only screenshots. |
| **Modes** | `reference`: the target image is given. `description`: a written brief names every asset and line of copy. |
| **Editors (gui)** | 6 built-in editors, modelled on common design tools: `figma`, `canva`, `photoshop`, `slides`, `powerpoint` and `inkscape` (Figma-, Canva-, Photoshop-, Google Slides-, PowerPoint- and Inkscape-like). Without `--skins`, each task gets one editor, chosen by its position. |
| **Real apps (gui)** | `impress` (LibreOffice Impress), `draw` (LibreOffice Draw) and `inkscape-desktop` (Inkscape), each on a Linux desktop. These run only when named with `--skins`, and they need an env server set up for them (see [Real apps](../envs/designgym/README.md)). |

Every surface edits the same document and is scored by the same renderer, so scores are comparable across
surfaces. Turn budgets follow `rollout.py`: 60 turns for tools, 20 for html and 200 for gui. The model is told
its action budget, and the environment enforces it.

## Run it locally

`rollout.py` takes models as `provider:model`, optionally with a thinking suffix (`@off`, `@low`, `@default`):

| Provider prefix | What it calls | Credentials |
|---|---|---|
| `anthropic:` | Anthropic | `ANTHROPIC_API_KEY` |
| `openai:`, `openai-chat:` | OpenAI | `OPENAI_API_KEY` |
| `hf:` | Hugging Face router | `HF_TOKEN` |
| `vllm:` | Your own OpenAI-compatible server | `VLLM_BASE_URL` (plus `VLLM_API_KEY` if the server needs one) |

Keys are read from the environment or the nearest `.env`.

```bash
cd envs/designgym
uv sync --extra rollout

# against the hosted DesignGym Space
uv run python rollout.py --base-url https://fineenvs-designgym.hf.space \
    --evalset ../../bench/evalsets/recreation-cua-v1.json --subset lite \
    --models anthropic:claude-sonnet-5 --surfaces gui --modes reference \
    --image-window 8,3 --concurrency 8 --out ../../results/rollouts/lite-sonnet5

# your own vLLM server, all three surfaces, both modes, on core30
VLLM_BASE_URL=http://127.0.0.1:8000/v1 uv run python rollout.py --base-url https://fineenvs-designgym.hf.space \
    --evalset ../../bench/evalsets/recreation-cua-v1.json --subset core30 \
    --models vllm:Qwen/Qwen3.6-35B-A3B@off --surfaces tools,html,gui --modes reference,description \
    --max-tokens 4096 --image-window 8,3 --out ../../results/rollouts/core30-qwen36

# GUI in each of the real apps (needs an env server set up for them, passed with --base-url)
uv run python rollout.py --base-url <env server> --evalset ../../bench/evalsets/recreation-cua-v1.json \
    --subset core30 --models <model> --surfaces gui --skins impress,draw,inkscape-desktop --out <dir>
```

Useful options:

- **No `--base-url`.** `rollout.py` starts a server in-process. That server needs a task source with the test split
  (`designgym pull --splits test` downloads it; or `designgym evalset export` an offline pack of this set).
- **Subsets of the set.** Choose `--subset full|lite|core30`. `--limit N` takes N tasks spread evenly across tiers.
  `--tiers hard` and `--task-ids a,b` narrow the run.
- **Resuming.** Run again with the same `--out`. Clean episodes are skipped, and errored ones are played again.
  To start a fresh directory from an earlier run, use
  `python bench/harness/resume.py <old dir> <new dir> --model <label>`.

The harness defaults are `--image-window 8,3` and `--max-tokens 4096`:

- **`--image-window 8,3`.** The target stays in context. Up to 8 screenshots build up, then the context drops to
  the newest 3.
- **`--max-tokens 4096`.** The cap on the model's output tokens per turn.

Report these settings with your scores.

## Run it on Hugging Face Jobs

`harness/launch.py` starts one GPU job. The job does the following:

1. Checks out this repo, `https://github.com/adithya-s-k/FineEnvs`, at `--ref` (default: branch `design-world`;
   pass a commit SHA to pin it).
2. Serves each model in `--models` with vLLM, one model per GPU, or `--tensor-parallel` GPUs per model.
3. Runs `08-design-world/envs/designgym/rollout.py` with
   `--evalset 08-design-world/bench/evalsets/recreation-cua-v1.json`.
4. Publishes the run.

The environment is the hosted Space unless you pass `--local-env` or `--real-apps`. Those start a server inside
the job.

```bash
export DESIGNWORLD_JOBS_NAMESPACE=<org or user that pays for the job>   # or --namespace
export DESIGNWORLD_RESULTS_REPO=FineEnvs/designworld-bench-runs         # the default

python bench/harness/launch.py eval --models Qwen/Qwen3.6-35B-A3B --subset core30
python bench/harness/launch.py eval --models <repo-a>,<repo-b> --flavor h200x2 --surfaces gui,tools --subset lite
python bench/harness/launch.py eval --models <repo> --real-apps impress,draw,inkscape-desktop   # needs DAYTONA_API_KEY
python bench/harness/launch.py eval --models <repo> --resume-from runs/<earlier run>
python bench/harness/launch.py logs <job_id> --follow
```

- **`--real-apps`.** Accepts only `impress`, `draw` and `inkscape-desktop`. The job's env server runs them in its
  own Daytona pool, one desktop per concurrent episode.
- **`--resume-from`.** Takes a local run directory (uploaded with the job's inputs), `runs/<name>` or `<name>` in
  the results repo, or `<org>/<repo>/<path>`. `resume.py` copies that run's records for the same model, surfaces,
  modes, skins and subset, with their frames and renders. `rollout.py` then skips every episode whose
  `(model, surface, mode, task, skin)` key already has a clean record. The new summary covers the carried records
  as well.
- **`--split`.** Starts one job per model.
- **`--patch-tool-listing`.** Rewrites a Qwen chat template so tools are listed the way the Tinker renderer lists
  them in training.

`HF_TOKEN` comes from the environment or your Hugging Face login and is passed as a Job secret. Each job writes
`runs/<run>/` in the results repo, on success or failure. The folder holds:

- `job.json`: the repo, ref, resolved commit and settings
- `rollouts-<model>/`: `trace.jsonl`, `summary.json`, `summary.md`, frames and renders
- the vLLM, env, rollout and `to_cua` logs

## Scores

Each episode's reward is the environment's grade of the submitted design, between 0 and 1. On the Space this is
reward v3 in reference mode. The reward has these components:

- **Structure.** Elements are matched one-to-one on content, geometry and typography.
- **Visual.** Multi-scale SSIM and colour layout against the rendered ground truth.
- **Order.**
- **Background.**

Replaying the ground truth scores 1.0. An episode that ends without a submit is submitted by the harness. An
episode that fails counts as 0.

`rollout.py` writes `summary.json` and `summary.md` with two tables:

- **Per model, surface and mode.** Episode count; mean, min and max reward; mean turns and tool calls; tool
  errors; failed episodes; tokens; and how much the model reasoned per turn.
- **Per tier.** The mean reward for `easy`, `medium`, `hard` and `all`, with a 95% bootstrap interval (2,000
  resamples of the episodes, seed 0, percentile method).

Report scores for a subset (`full`, `lite` or `core30`), with the surface, mode, editor or app choice, and the
settings above.

## Runs as CUA datasets

Every run is published to [`FineEnvs/designworld-bench-runs`](https://huggingface.co/datasets/FineEnvs/designworld-bench-runs).
Set `DESIGNWORLD_RESULTS_REPO` or pass `--push <repo>` to use another repo. The job does this itself unless
`--no-cua` is set. To publish a run by hand:

```bash
python bench/harness/to_cua.py results/rollouts/<run> --dry-run              # counts only, no extra deps
python bench/harness/to_cua.py results/rollouts/<run> --out /tmp/cua --config <run> --push
```

It needs `designgym` and `datasets`. Run it with the environment package's interpreter.

**Layout.** Each run is one config, split `test`, stored in `data/<run>/test-<model>-xxxxx-of-yyyyy.parquet`:

- **Shards per model.** Each model's shards are named apart, so the models of one run can be published by
  separate jobs.
- **Run metadata.** `meta/<run>.json` holds the rollout summaries, the mean reward per model, surface and mode,
  and the conversion counts.
- **Card.** `--push` rewrites the repo card's config list and runs table.

**Rows.** One row per episode, in the `trajectories` view of `designgym.cua_format`:

- **`messages`.** The OpenAI chat shape:
  - an opening user turn with the target (reference mode) and the first screenshot (GUI), then the opening
    text `rollout.py` sends
  - assistant turns with `tool_calls`, whose `arguments` are JSON strings in the coordinates the model used
  - one `tool` message per call, holding the call's result text and every image it returned
- **GUI screenshots.** When the model didn't call `screenshot`, the harness took one after the turn. It is
  attached to the turn's last tool message, as the harness returned it.
- **Images and training flags.** `images` are listed in the order their parts appear. `trainable` is true on
  assistant turns. `windows()` derives the same windowed SFT rows as for DesignWorld-SFT.
- **Bench columns.** `model`, `run`, `evalset_id`, `episode_id`, `tier`, `difficulty_score`, `grade` (JSON),
  `error`, `auto_submitted`, `seconds`, `usage` (JSON), `final_doc` (the submitted document, so a later reward
  version can re-score a run without the model), and `missing_images`.

**Trace gaps.** `trace.jsonl` doesn't hold everything a lossless row needs:

- **System prompt and tool schemas.** Not recorded. Only `prompt_version` is, so rows have no system message and
  `tools` is null. `to_cua.py` uses `instructions` and `tools` from a record if they are there, so recording them
  once per episode in `rollout.py` would close this gap.
- **Tool result text.** Cut at 400 characters.
- **Call arguments.** Arguments that weren't valid JSON are stored as `{}`. Only the parse error survives in the
  tool message.
- **Calls after submit.** Calls made in the same turn after the one that ended the episode aren't recorded.
- **Call ids.** Rebuilt as `call_<turn>_<k>`.
- **Screen and editor details.** For GUI rows, `screen` is measured from the first screenshot. `editor` has only
  `skin`, and `theme`, `layout` and `file_dialog` are null. `task_source` is null.
- **Images.** Images are saved only with frames on (the default; not with `--no-frames`). Older traces have no
  `image_paths`. Episodes with a missing image are skipped unless `--keep-incomplete` is set, and the dry run
  reports how many.

## Results

None are published yet. Runs will appear in `FineEnvs/designworld-bench-runs`, with a summary table here.
