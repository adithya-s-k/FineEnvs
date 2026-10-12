# Evaluate a model

## One command

Name the models; `evaluate.py` runs them all at once on an evaluation set and writes one board.

```bash
uv sync --extra dev --extra eval
uv run python eval/evaluate.py claude-sonnet-5-5 gpt-6-sol "Qwen/Qwen3.8-27B:novita"
uv run python eval/evaluate.py --board eval/boards/core30-nothink.json   # the 13-model board
uv run python eval/evaluate.py "google/gemma-4-31B-it:novita" --tasks 1    # a one-task smoke test
```

* **Provider** follows from the id: `claude-*` on the Anthropic API, `gpt-*` on the OpenAI Responses API, `org/name` on the HF router. Pin a router provider (`org/name:novita`) so prices are stable. Keys come from `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` and `HF_TOKEN`, or the nearest `.env` above the working directory.
* **Set**: `--set core30` (default; 10 easy, 10 medium, 10 hard inside final_eval), `--set final_eval` (50), or a whole split (`--set dev`).
* **Protocol**: thinking off (`--thinking default` to keep it), malformed submissions graded as sent (`--repair` to repair them), the environment's budget of 16 turns and 32 tool calls, `--max-cost` 25 USD per model. Models that cannot turn thinking off run at their lowest effort: Claude Opus 5.5 and Fable 5.1, gpt-6.1-sol and gpt-6-astra.
* **Server**: `--server URL` (a local `start.sh`, or the Space `https://fineenvs-retroenv.hf.space`), or none, and a local server is started from the pinned bucket snapshot (`envs/retro_route/openenv/prepare.py`; the first start downloads it).
* **Output**: `runs/core30-nothink/<model>/` and `<model>.log`, and the board in `runs/core30-nothink/results/RESULTS.md`. A rerun resumes each model, so adding a model later costs only that model and it joins the same board. `--tasks 1` is a smoke test into `runs/core30-nothink-smoke`.

Any other `run_eval.py` option passes through to every model (`--attempts 3`, `--max-turns 24`).

Without repair, a submission sent as a broken JSON string scores 0, as in training. `eval/regrade.py runs/core30-nothink --server URL` re-grades those final submissions after the harness's repair, through the same server and with no model calls, and writes `results/REPAIR.md` with both scores side by side.

## One model, one split

`run_eval.py` runs a model on a split through the OpenEnv server. Every tool call and the reward come from the server, the same one used for training and the playground. Without `--server` it starts a local server for `--benchmark-dir` (default `data/release/RetroEnv-RL`; download it from [LiteFold/RetroEnv](https://huggingface.co/datasets/LiteFold/RetroEnv) or point `--server` at one started with `RETROENV_TASKS_REPO`). The release is fully open: every split, test_id and test_hard included, ships its known routes, and the runs on 30 test_id tasks are under `runs/`. Use dev to pick checkpoints during training and report test_id and test_hard, so test scores never steer training. For a quick board, `--split final_eval` runs the 50-task subset of the two test splits (17 easy, 17 medium, 16 hard); it is served by a server started from the bucket snapshot (`envs/retro_route/openenv/start.sh`, then `--server http://127.0.0.1:8000`).

```bash
uv sync --extra dev --extra eval

# Claude, on the Messages API
uv run python eval/run_eval.py --provider anthropic --model claude-opus-5-5 \
  --split test_id --output runs/test_id/claude-opus-5-5

# GPT-5.6, on the Responses API (chat completions rejects tools with reasoning)
uv run python eval/run_eval.py --provider openai --model gpt-5.6-sol --split test_id --output runs/test_id/gpt-5.6-sol

# Open models on the HF router; pin a provider so prices are stable
uv run python eval/run_eval.py --provider hf --model "deepseek-ai/DeepSeek-V4.1-Flash:novita" \
  --split test_id --output runs/test_id/deepseek-v4.1-flash

# A local vLLM server or any OpenAI-compatible endpoint
uv run python eval/run_eval.py --provider custom --endpoint http://127.0.0.1:8001/v1 \
  --model Qwen/Qwen3.5-4B --api-key-env VLLM_KEY --split test_id --output runs/test_id/qwen3.5-4b

# No model: a scripted chemist replays the private reference routes (SFT data; see train/README.md)
uv run python eval/run_eval.py --provider reference --split train --attempts 2 --output runs/sft/chemist

# Build the table from finished runs
uv run python eval/summarize.py runs/test_id/* --output runs/test_id/results
```

Each held-out split has 1,000 targets plus variants; run a dev slice first to estimate cost.

| Option | Default | Notes |
|---|---|---|
| `--split`, `--start`, `--tasks` | `dev`, 0, all | A stable slice of the split |
| `--attempts` | 1 | Above 1, the summary adds unbiased pass@k |
| `--max-turns` | 16 | The last turn exposes only `emit_routes` |
| `--toolset` | `full` | `unaided` starts the local server without `validate_disconnection` (an ablation) |
| `--concurrency` | 8 | One WebSocket session per episode |
| `--max-cost` | none | Stops scheduling episodes once spend reaches this many USD |
| `--provider` | from `--model` | `claude-*` anthropic, `gpt-*` openai, `org/name` hf |
| `--thinking` | `default` | `off`: no thinking, or the lowest effort where a model cannot turn it off |
| `--no-repair` | off | Grade a malformed submission as sent, as training does |
| `--tool-choice` | `required`; `auto` on hf | Some HF providers only accept `auto` (novita, and cerebras for Qwen3.8) |

Keys come from the environment: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `HF_TOKEN`, `OPENROUTER_API_KEY`, or `--api-key-env`.

## The protocol

Three loops implement the same protocol: `agent.py` for OpenAI-compatible chat, `agent_anthropic.py` for Claude and `agent_responses.py` for the OpenAI Responses API. All three live in `envs/retro_route/openenv/retroenv_openenv/`.

* The system prompt and task prompt are identical for every model. Each tool result reports the model turns left.
* The final turn exposes only `emit_routes`. Claude Opus 5.5 and Sonnet 5.5 reject a forced `tool_choice`, so that turn says so in text, as it does whenever `tool_choice` is `auto`.
* Two turns without a tool call trigger the final turn early. An episode that never calls `emit_routes` is closed with an empty route set, which scores 0.
* Claude runs at its default effort and thinking. GPT-5.6 runs at its default reasoning effort. Other models run at temperature 0.
* Some models send `submission` as a JSON string, sometimes with one stray closing bracket. The harness decodes that and records `submission_coerced`; the server's verifier stays strict, so training still sees the error.
* A refusal (Claude's `refusal` stop reason) counts as a failed episode. It is reported as `refusal_rate`, and requests are never re-routed to another model.

## Output

| File | Contents |
|---|---|
| `identity.json` | Model, provider, sampling, task IDs, toolset, and hashes of the tool schemas and agent code. A rerun must match. |
| `episodes/NNNN-<task>-aK.json` | Reward, components, failures, usage, cost, the submission and the full transcript |
| `progress.json`, `summary.json` | Coverage, pass@1 with a Wilson CI, exact route rate, reward and components, tool calls, no-emit and refusal rates, cost, and breakdowns by route count, step cap and heuristic tier |

Rerun the same command to resume. Graded episodes are kept, and those that hit a provider or transport error are retried. `runs/` is gitignored; `dataset/publish_release.py --runs runs` publishes them with the release.
