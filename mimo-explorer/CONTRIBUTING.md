# Contributing to the MiMo RL Environment Explorer

Thanks for helping. This explorer makes the 7,780 environments in
[XiaomiMiMo/MiMo-V2.6-RL-oss](https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss) easy to read, and lets
anyone run a rollout on one and see it graded the way the model was trained. It is built by Hugging Face and the
[FineEnvs](https://huggingface.co/FineEnvs) team, and it is unofficial: not affiliated with Xiaomi.

Found a problem, a task that looks broken, or a rollout that was graded wrongly? Open a thread in the
[Space discussions](https://huggingface.co/spaces/FineEnvs/MiMo-RL-Envs-Explorer/discussions). A rollout link and
the version in the page footer make it much quicker to look into.

## Run it locally

```bash
uv sync
uv run uvicorn app.main:app --port 8000     # http://localhost:8000
```

Locally there is no OAuth app: your own Hugging Face token (`HF_TOKEN` or `hf auth login`) is the signed-in user,
rollouts run on your account, and traces go to `./.local-runs`. The same image the Space runs:
`docker build -t mimo . && docker run -p 127.0.0.1:7860:7860 -e HF_TOKEN -e MIMO_TRUST_NETWORK=1 mimo`.
Read [SECURITY.md](SECURITY.md) before exposing it beyond your own machine.

## How a rollout works

1. `POST /api/runs` checks the task, the model (tool calling required) and the judge, then starts a worker thread.
2. The domain adapter in `app/runner/domains.py` starts an HF Sandbox from the task's own image and sets it up the
   way Xiaomi's harness does (hidden tests stay out, the agent runs unprivileged where the reference does, MCP
   systems start for General tasks).
3. OpenCode runs the agent against HF Inference Providers or the user's own OpenAI-compatible endpoint
   (`app/runner/opencode.py`). Every event it prints becomes a trace event, live.
4. The task's own grader runs: hidden tests, the crash server, `run_verify.py`, the vision rubric, or the music
   scorer. The reward and every check land in the trace.

Each adapter cites the file and commit of XiaomiMiMo/verl or mimoagent it mirrors. Keep it that way: when you change
setup or grading, say which reference behaviour you are matching, or why this has to differ.

## The data

Everything a rollout produces is in one folder, `runs/<id>/`, in the store (`./.local-runs` locally, the private
bucket `FineEnvs/mimo-explorer-runs` on the Space).

### `run.json`: the record

```json
{
  "id": "20260926-173659-f02602",
  "task_id": "arvo_42480818", "domain": "cyber", "title": "heap-buffer-overflow in ...",
  "model": "zai-org/GLM-5.3-Flash", "provider": "novita", "judge": null,
  "endpoint": null,
  "params": {"thinking": "none", "steps": 80, "timeout_min": 20},
  "visibility": "public",
  "status": "done", "reward": 1.0,
  "tokens": {"input": 179739, "output": 1409, "reasoning": 0, "cache_read": 0, "cache_write": 0},
  "cost": {"model": 0.084, "judge": 0.0, "sandbox": 0.001, "total": 0.085},
  "flavor": "cpu-basic", "image": "docker.io/xiaomimimo/mimo-v2.6-rl-oss:arvo-v1-42480818",
  "provenance": {
    "app": {"version": "1.0.0", "source": "102c5523ca3c"},
    "harness": {"name": "opencode", "version": "1.18.32", "installed": "1.18.32"},
    "reference": {"XiaomiMiMo/verl": "a2ad9f6", "XiaomiMiMo/mimoagent": "467f0a1"},
    "dataset": {"repo": "XiaomiMiMo/MiMo-V2.6-RL-oss", "revision": "639865fd33..."}
  },
  "created_at": 1790422019.5, "started_at": 1790422019.6, "finished_at": 1790422499.1
}
```

`provenance.app.source` is a hash of the code that shipped, so two deploys of one version are still told apart.
With your own endpoint, `endpoint` holds its URL and the prices you gave; `provider` is empty.

### `events.jsonl`: the trace

One JSON object per line, in order, each with `i` (sequence), `t` (seconds since start) and `kind`:

| kind | fields | meaning |
|---|---|---|
| `phase` | `name`, `status` (`start`/`done`/`error`), `detail` | sandbox, setup, agent, verify, done |
| `thinking` | `text`, `chars` | the model's reasoning between steps |
| `text` | `text` | a message from the model |
| `tool` | `tool`, `title`, `input`, `output`, `status`, `ms`, `truncated` | one tool call (output capped at 8,000 characters) |
| `step` | `tokens`, `cost` | one model call finished |
| `diff` | `text`, `files` | Code: the agent's change against the base commit |
| `files` | `title`, `files` | Webdev: what was delivered in `dist/` |
| `image` | `name`, `width`, `height` | Webdev: the full-page render the judge saw |
| `log` | `text` | setup or grader output |
| `checks` | `reward`, `checks`, `summary`, `error`, `formula` | the grade, with one entry per check |
| `error` | `text` | something failed |

```json
{"i": 14, "t": 61.2, "kind": "tool", "tool": "bash", "title": "", "input": {"command": "ls /testbed"}, "output": "README.md\nsrc\n...", "status": "completed", "ms": 212}
{"i": 40, "t": 482.9, "kind": "checks", "reward": 1.0, "summary": "Crash reproduced in the expected function.", "checks": [{"id": "right_function", "passed": true, "message": "crashed in `extract_name`, expected `extract_name`"}]}
```

Artifacts sit next to the trace: `screenshot.jpg` (Webdev), `piece.abc` (Music).

### Repository snapshots (Code)

`repo-snapshots/<task_id>.json.gz` holds the file list, base commit, upstream remote, whether the image's git history
is truncated at the base, and the text of files up to 200 KB. They are built by `python -m app.snapshot_repos`,
which opens each task's image as a sandbox; nothing in them comes from the hidden tests.

## House rules

* **Reference code is vendored unmodified** (`app/vendor/`, with its license and source in `app/vendor/README.md`).
  If a grader needs a fix, fix it around the vendored file, not inside it.
* **"Not scored" is never 0.** If the testbed, the judge or the render fails, the reward is `None` and the page says
  why. Averaging infrastructure failures in as zeros misreads results; Xiaomi's harness masks them the same way.
* **Sandboxes hold no credential on the Space.** Model and judge calls go through `/api/llm/<capability>`; don't
  add code that puts a token or key into a sandbox when `config.PUBLIC_URL` is set.
* **Nothing secret reaches a trace.** Tokens and endpoint keys are held in memory for the rollout only, and every
  event passes through the redaction in `Rollout._redact`. Never log a request header or an environment dump.
* **Public means anonymous.** The public view of a rollout (`_public_view` in `app/main.py`) has no user, no sandbox
  id and no endpoint URL, and the owner's username is scrubbed from the trace. Add a field to it only if it is safe
  for anyone to see.
* **Frontend:** plain ES modules, no build step. Colours and spacing come from the tokens at the top of
  `web/app.css`, icons from `web/js/icons.js`. No emoji or glyph icons.

## Checking a change

```bash
uv run python -m app.validate --per-domain 12 --workers 8    # a stratified sample, no model in the loop
uv run python -m app.validate --ids format-code-task-000158  # specific tasks
```

This runs the real adapters with a fixed probe in place of the agent and checks the environment behaved as intended:
hidden tests fail on an untouched repo, the MCP systems answer the agent user and the databases stay hidden, the
Cyber answer is unreadable, a fixture page renders and gets judged, a valid tune passes the music gate. It needs
sandbox time on your account (a few cents for the sample). A real rollout with a cheap model (GLM-5.3-Flash is
a good one) is the final check.

## Adding a domain

Write an adapter class in `app/runner/domains.py` with `steps`, `timeout` and `run(r)`. It starts the sandbox,
sets it up, calls `opencode.run`, grades, emits one `checks` event and returns `{"reward": ...}`. Register it in
`ADAPTERS`, give the task page a `verify` kind in `app/catalog.py`, and add its health check to `app/validate.py`.

## Community data

Rollouts are public by default and can be switched to private at any time from the rollout page. Public rollouts
appear on their task and in Community for everyone, without who ran them. To help research on these environments,
public rollouts may later be released as an open dataset, for example as SFT or RL traces, with the same
anonymisation. Private rollouts are never included.

## License

The explorer's code is Apache-2.0. Vendored files keep their own licenses (MIT for mimoagent, Apache-2.0 for verl),
and the dataset is Apache-2.0.
