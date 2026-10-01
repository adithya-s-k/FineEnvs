# SmolDataEnvs: train an agent, then change its harness

A model can solve the same task through different agent programs. Does training through several programs help it transfer? This tutorial gives you three scripts to explore that question on data-analysis tasks.

Start with [04: SmolDataEnvs](../04-smoldataenvs/) for the dataset. The [multi-harness RL article](https://huggingface.co/spaces/AdithyaSK/multi-harness-rl) explains the earlier experiments; its [source and figures](../content/articles/multi-harness-rl/) live in this branch too.

Try the [whitebox playground](https://fineenvs-smoldataenv-multi-harness-whitebox.hf.space/web/) to solve a task yourself. Browse the environments and experiment artifacts in the [SmolDataEnv Multi-harness RL collection](https://huggingface.co/collections/FineEnvs/smoldataenv-multi-harness-rl-6abdfaaa8d74dacd481d5212).

If you want to run training first, follow the [HF Jobs quickstart](REPRODUCE.md#2-submit-from-a-laptop-with-hf-jobs). If you already have GPUs, use the [local and Slurm commands](REPRODUCE.md#3-run-the-same-commands-locally-or-with-slurm). Both launch the environment inside the allocation; the public Spaces are also available for interactive exploration.

## Open a training script first

Each script reads like a notebook: settings → tokenizer → environment → reward → TRL config → training. The data loading, reward and trainer call are visible in that file. The numbered sections keep the setup and training flow easy to follow. To change the learning rate or batch size, edit the visible TRL config in that script.

| Script | Who runs the tool loop? | Trainer |
|---|---|---|
| [train/whitebox.py](train/whitebox.py) | TRL calls Python bash/SETA tools | `GRPOTrainer` |
| [train/opencode.py](train/opencode.py) | Native OpenCode runs inside a sandbox | `AsyncGRPOTrainer` |
| [train/multi_harness.py](train/multi_harness.py) | Harbor runs OpenCode, Claude Code, Codex or Mini-SWE-Agent | `AsyncGRPOTrainer` |

Read whitebox first if you are new to tool-using RL. Its [environment](envs/whitebox/smoldataenv_whitebox/environment.py) shows exactly which tools the model can call and how an answer is graded. Then read OpenCode to see what changes when an existing agent owns the conversation. The multi-harness script assigns one harness to each task; its eight rollouts all use that assignment.

The installer uses **TRL main and OpenEnv main**, including the merged typed training-trace integration. `check_setup.py` verifies the required APIs before training. Each run records the resolved commits; see [VALIDATION.md](VALIDATION.md) for the tested versions and remaining GPU checks.

## Folder layout

```text
train/                  Three readable TRL training scripts
  whitebox.py
  opencode.py
  multi_harness.py
eval/evaluate.py        Fixed pass@1 evaluation and tool/token metrics
envs/                   Environment implementations and deployment
  whitebox/             Standalone SETA Space and Python package
  opencode/             Standalone native OpenCode Space and package
  harbor/               Standalone Harbor Space with its rollout UI
jobs/                   HF Jobs, Slurm and train/reload smoke commands
data/                   Fixed task identities
prepare.py              Download and check the task files
```

## What is being trained?

The model receives a question and can inspect data in a Daytona sandbox. Its answer is compared with a held-out answer using the task's deterministic grader. A correct answer earns a small extra reward when it uses fewer tools:

```python
bonus = 0.1 * 15 / (15 + tool_calls) if tool_calls > 0 else 0
reward = correctness * (1 + bonus)
```

A correct answer with 15 tool calls earns `1.05`; a wrong answer earns `0`. Missing tool-count evidence gets no bonus. A failed environment without a grade is excluded, not scored as wrong. The code counts native tool actions, not model requests or training rows.

Whitebox gives TRL the tools and lets it generate each turn. Blackbox gives TRL an OpenEnv session. OpenEnv returns the actual engine token IDs, behavior logprobs and loss masks through `fetch_training_trace()`. TRL trains on the eligible model tokens. Prompts and tool outputs are context.

```python
# Whitebox: TRL owns generation and tool execution.
trainer = GRPOTrainer(
    model=model,
    args=config,
    train_dataset=dataset,
    reward_funcs=[],
    environment_factory=BashEnvironment,
)

# Blackbox: the agent owns generation; OpenEnv captures its tokens.
worker = HarnessRolloutWorker(
    harness_session_factory=factory,
    harness_adapter=None,
    rollout_reward_fn=reward,
    # See either blackbox script for the model and sampling arguments.
)
trainer = AsyncGRPOTrainer(
    model=model,
    args=config,
    train_dataset=dataset,
    rollout_worker=worker,
)
```

These are the boundaries to compare. The complete calls, including every training parameter, are in the scripts above.

## Run a small experiment

Use Python 3.12 and two H100 or H200 GPUs: one for vLLM, one for training. HF Jobs can supply them; you do not need a GPU on your laptop. OpenEnv runs inside the job. Daytona supplies the task sandboxes. Each [environment folder](envs/README.md) is also a standalone CPU Space: its local Dockerfile and application files are uploaded unchanged.

Follow [REPRODUCE.md](REPRODUCE.md) for installation, a two-step smoke, a 100-step comparison, and checkpoint evaluation. From a prepared local environment:

```bash
# Two updates, save both checkpoints, then reload checkpoint 2 for evaluation.
python jobs/smoke.py --mode multi_harness --output runs/harbor-smoke
```

Use `--mode opencode` for native OpenCode, or `--mode whitebox` for bash/SETA. Blackbox evaluation uses all four Harbor harnesses for both policies. Whitebox evaluation uses its own bash/SETA tools.

## Shared settings

| Setting | Value |
|---|---|
| Models | LFM2.5-2.6B or Qwen3.5-2B, non-thinking |
| Training tasks | 1,000 fixed tasks: 400 medium, 600 hard |
| Test tasks | 250 fixed tasks: 33 easy, 118 medium, 99 hard |
| Learning rate | `3e-6`, constant, no warmup |
| Sampling | Temperature `0.8`, full vocabulary |
| Rollouts per task group | 8 |
| Async limits | 32 inflight, maximum staleness 4 |
| Async batching | Batch 4, accumulation 4, packed row budget 40,960 tokens |
| Whitebox batching | Batch 1, accumulation 16 |
| Training length | Up to 1,000 optimizer updates |
| Save / evaluate | Save every 50; evaluate checkpoints 100, 200, … separately |
| Evaluation | Pass@1, default concurrency 35, TP1/DP2 |

A step is an optimizer update, not a task. Async sampling and packing can use a variable number of prompts per step. This tutorial uses the upstream sampler and does not reproduce the archive's custom finite curriculum, exact committed-group resume, or whole-rollout weighting. Whitebox consumes 16 rollouts, or two groups of eight, per update. Blackbox histories can fork into multiple rows; long rows beyond the packing budget are dropped by upstream TRL. Inspect its drop and staleness metrics.

Both models use a 131,072-token serving context and up to 4,096 tokens per model call. Whitebox and Harbor cap the agent at 17 model turns. The public native OpenCode API has a 600-second timeout and per-call token cap, but no equivalent strict turn limit. These limits are not interchangeable. LFM's packed convolution boundary handling is explicit in each async script and checked on the GPU before training.

## Where to look next

- [envs/](envs/README.md): actual environments, OpenEnv servers and clients, Docker image and Space deployment.
- [eval/evaluate.py](eval/evaluate.py): individual pass@1 results, coverage, correctness, tool calls and token usage. Failed pairs remain visible and can be retried without rerunning graded pairs.
- [jobs/](jobs/): installation, service startup, HF submission and a Slurm template. No hidden training configuration.
- [prepare.py](prepare.py) and [data/](data/): fixed task selection and checked dataset revisions.
- [RESULTS.md](RESULTS.md): earlier experiment results and artifact locations.

Training keeps local Trackio data and checkpoints. Add `--space-id your-org/your-trackio-space` to also publish charts. It does not automatically write to the historical dashboard.

## Earlier experiment

![Historical LFM training and evaluation curves](assets/historical-lfm-curves.svg)

These curves belong to the [article](https://huggingface.co/spaces/AdithyaSK/multi-harness-rl), not this rewrite. Both historical LFM policies used Harbor, including its OpenCode-only policy. The native OpenCode tutorial is a different interface. [RESULTS.md](RESULTS.md#historical-article-curves) links the plotted source data and the [historical dashboard](https://huggingface.co/spaces/FineEnvs/data-agent-training-comparison-trackio).

The full experimental implementation remains on the [archived branch](https://github.com/adithya-s-k/FineEnvs/tree/archive/data-agent-experiments-20260930/04-data-agent). Earlier recipe code is also recoverable from this branch's git history.
