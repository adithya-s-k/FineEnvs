# SmolDataEnvs: learn RL through three agent interfaces

Fine-tune a small language model to solve data-analysis tasks, then measure whether it answers more questions correctly and uses fewer tools. This example compares three ways of connecting an agent to a trainer, using the same tasks and reward.

Start with [04: SmolDataEnvs](../04-smoldataenvs/) for the dataset, SFT and single-turn RL. This chapter takes those tasks into multi-turn agent harnesses. The companion article, [The ultimate guide to multi-harness RL](https://huggingface.co/spaces/AdithyaSK/multi-harness-rl), explains the approach and earlier experiments.

**Start here:** [run the tutorial](REPRODUCE.md). You can submit an HF Job from a CPU laptop, or run on a local two-GPU machine or Slurm cluster. [Current results](RESULTS.md) distinguish completed checks from ongoing experiments; [validation notes](VALIDATION.md) contain the detailed evidence.

## Choose an agent interface

A *harness* is the program that manages the model's conversation and tool calls. OpenEnv provides the environment interface. Harbor runs third-party harnesses and their task verifiers.

| Mode | What happens | Trainer | Evaluation |
|---|---|---|---|
| `whitebox` | TRL generates responses and directly invokes bash/SETA tools | Synchronous GRPO | Native bash/SETA |
| `opencode` | Native OpenCode owns the agent loop; OpenEnv captures its model calls | AsyncGRPO | All four Harbor harnesses |
| `multi-harness` | Harbor rotates OpenCode, Claude Code, Codex and Mini-SWE-Agent between task groups | AsyncGRPO | All four Harbor harnesses |

**Native OpenCode training does not run through Harbor.** Both blackbox policies use the same Harbor evaluation interfaces so you can measure transfer to other harnesses. Whitebox evaluates its native tools, which is a different evaluation interface.

Use `--model lfm` for **LiquidAI/LFM2.5-2.6B**, or `--model qwen` for **Qwen/Qwen3.5-2B**. Both use non-thinking prompts. The new runs need their own baseline; historical thinking-enabled results are not interchangeable.

## What happens in a run?

1. Load the fixed task list and checked dataset revision.
2. Start vLLM and OpenEnv inside the job. Daytona supplies isolated task containers.
3. Generate eight rollouts for each task group. Each rollout tries to solve the task with tools.
4. Grade the answer and count verified native tool calls.
5. Train on the model's generated tokens, masking prompts and tool outputs.
6. Save checkpoints and evaluate them on held-out tasks.

TiTO means *tokens in, tokens out*: training uses the inference engine's actual token IDs and logprobs rather than reconstructing them from text. A rewritten conversation can produce several training rows from one rollout. A task, rollout, training row and optimizer step are therefore different units.

Both blackbox modes implement OpenEnv's `fetch_training_trace()` contract. The producer selects agent calls and supplies explicit masks; TRL receives the engine IDs unchanged. Partial masks and zero-masked agent turns survive. The worker supplies sampling before the factory starts a session. Native tool counts are carried as structured reward metadata, separately from token eligibility.

Synchronous GRPO waits for its batch of tool rollouts. AsyncGRPO collects rollouts while training proceeds, with a bounded amount of policy staleness. A group whose rollouts all have the same reward supplies no relative learning signal. Successful execution alone does not establish useful learning.

## The shared experiment

| Setting | Default |
|---|---|
| Training set | 1,000 fixed tasks: 400 medium and 600 hard |
| Held-out set | 250 fixed tasks: 33 easy, 118 medium and 99 hard; no train/test task or notebook overlap |
| Learning rate / sampling | `3e-6`; temperature `0.8`; full-vocabulary sampling |
| Rollouts per task group | 8, using the same task and harness |
| Async limits | 32 inflight, 16 outstanding rollouts, maximum staleness 4 |
| Async batch | Batch 4 × accumulation 4; 40,960-token packing target |
| Whitebox batch | Microbatch 1 × accumulation 16 |
| Episode limits | 17 iterations, 600 seconds, 4,096 generated tokens per model call |
| Training limits | 16,384 completion tokens; 131,072 context; full BF16 fine-tuning |
| Checkpoints | Every 50 optimizer steps |
| Evaluation | Pass@1; default concurrency 35; TP1/DP2 inference |

Correctness is binary. A correct answer with a verified, positive tool count earns:

```text
reward = correctness × (1 + 0.1 × 15 / (15 + tool_calls))
```

For example, a correct answer using 15 calls earns `1.05`; an incorrect answer earns `0`. Zero or unverified tool counts receive no bonus. Infrastructure failures remain ungraded. Evaluation reports correctness separately from shaped reward and tool/token usage.

The full schedule contains two passes and stops at 1,000 updates or schedule exhaustion, whichever comes first. The 100-step pilot retains the same schedule and training settings but stops at 100 updates. Neither setting guarantees that every scheduled task is consumed. The committed-group audit records what async training actually used. For whitebox, 16 rollouts per update divided by 8 rollouts per task means 2 unique prompts per update. Its 2,000 scheduled prompts therefore allow 1,000 updates; custom step limits are capped at schedule exhaustion.

## Start small, then scale

After the [HF account, bucket and sandbox setup](REPRODUCE.md#2-submit-from-a-cpu-laptop-with-hf-jobs), preview a quick training and evaluation smoke:

```bash
python runtime/launch.py hf smoke --model lfm --mode opencode \
  --run-name smoke-lfm-opencode-v1 --bucket "$HF_BUCKET" \
  --flavor a100x4 --timeout 45m --smoke-eval --concurrency 4
```

Add `--submit` to launch. This runs two small updates, saves both checkpoints and reloads checkpoint 2 for evaluation on two tasks through all four harnesses. For the other modes, use `--mode multi-harness` or `--mode whitebox` and a new run name. Whitebox evaluates two native SETA episodes. The [tutorial](REPRODUCE.md) also covers local and Slurm execution.

| Action | Purpose |
|---|---|
| `smoke --smoke-eval` | Two small training updates, checkpoint saving, then two-task reload evaluation |
| `pilot` | Baseline, 100 updates, checkpoint-100 evaluation on 25 fixed tasks by default |
| `train` + evaluation watcher | Longer training with separate checkpoint-evaluation jobs |

The pilot evaluates 3 easy, 12 medium and 10 hard tasks. That is **100 pairs per blackbox evaluation** or **25 whitebox episodes**. Its phases run sequentially in one allocation, restarting services between phases. No watcher is needed. Full training uses separate evaluation allocations; shared sandbox capacity and storage can still affect throughput.

There is **no Space to deploy** for these runs. OpenEnv lives inside the GPU job; the task containers live in Daytona. A Trackio Space is optional for online charts.

## Find the code

| Path | What it does |
|---|---|
| `configs/default.json` | Inspect model pins and shared hyperparameters |
| `run.py`, `runtime/launch.py` | Run locally, submit to HF Jobs or inspect a Slurm command |
| `runtime/pilot.py` | Follow the baseline → train → checkpoint-eval sequence |
| `train/whitebox.py` | Understand synchronous native tool training |
| `train/blackbox.py`, `train/adapters.py` | Understand async training and the native/Harbor boundary |
| `eval/evaluate.py` | Inspect pass@1, coverage and per-harness results |
| `eval/watch.py` | Submit separate evaluations for full training checkpoints |
| `prepare.py`, `data/` | Inspect the exact task selection and dataset checks |
| `configs/runtime-lock.json`, `runtime/patches.py` | Inspect pinned dependencies and compatibility fixes |

`runtime/bootstrap.py` downloads the pinned TRL/OpenEnv sources and 45 existing runtime files into ignored `.runtime/`. It checks the archive file hashes. You do not need the original experiment workspace. Prepared tasks, checkpoints, caches and credentials are not committed.

## Earlier training results

![Historical LFM training reward, held-out pass@1 and tool-call savings](assets/historical-lfm-curves.svg)

These curves come from the [article](https://huggingface.co/spaces/AdithyaSK/multi-harness-rl), using 250 test tasks across four harnesses. Both historical LFM runs used Harbor, including the OpenCode-only run. This tutorial's `opencode` mode uses native OpenCode. The curves are historical context, not results from the new non-thinking pilots.

Training reward uses a trailing 50-update mean. Evaluation is unsmoothed; hollow markers indicate incomplete coverage. Tool-call savings compare pairs solved by both baseline and checkpoint. See [the data source and final scores](RESULTS.md#historical-article-curves), or explore the [Trackio dashboard](https://huggingface.co/spaces/FineEnvs/data-agent-training-comparison-trackio).

The full experiment remains on the [archived branch](https://github.com/adithya-s-k/FineEnvs/tree/archive/data-agent-experiments-20260930/04-data-agent). Corrected Harbor graders and dataset revisions are pinned here; [VALIDATION.md](VALIDATION.md) records what has been checked for this recipe.
