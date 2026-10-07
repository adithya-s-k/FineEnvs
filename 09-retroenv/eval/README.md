# Evaluate a model

`run_eval.py` runs a model on a split through the OpenEnv server. Every tool call and the reward come from the server, the same one used for training and the playground. Without `--server` it starts a local server for `--benchmark-dir` (default `data/release/RetroEnv-RL`; download it from [LiteFold/RetroEnv](https://huggingface.co/datasets/LiteFold/RetroEnv) or point `--server` at one started with `RETROENV_TASKS_REPO`). The release is fully open: every split, test_id and test_hard included, ships its known routes, and the runs on 30 test_id tasks are under `runs/`. Use dev to pick checkpoints during training and report test_id and test_hard, so test scores never steer training.

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
| `--tool-choice` | `required` | Some HF providers only accept `auto` (novita, and cerebras for Qwen3.8) |

Keys come from the environment: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `HF_TOKEN`, `OPENROUTER_API_KEY`, or `--api-key-env`.

## The protocol

Three loops implement the same protocol: `agent.py` for OpenAI-compatible chat, `agent_anthropic.py` for Claude and `agent_responses.py` for the OpenAI Responses API. All three live in `envs/retro_route/openenv/retroenv_openenv/`.

* The system prompt and task prompt are identical for every model. Each tool result reports the model turns left.
* The final turn exposes only `emit_routes`. Claude Opus 5.5 and Sonnet 5.5 reject a forced `tool_choice`, so that turn says so in text, as it does whenever `tool_choice` is `auto`.
* Two turns without a tool call trigger the final turn early. An episode that never calls `emit_routes` is closed with an empty route set, which scores the 0.05 floor.
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
