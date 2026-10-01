# Evaluate a checkpoint

[Read the article](https://huggingface.co/spaces/FineEnvs/multi-harness-rl) · [Collection](https://huggingface.co/collections/FineEnvs/smoldataenvs-multi-harness-rl-6abdfaaa8d74dacd481d5212) · [Tutorial](../README.md) · [Setup](../REPRODUCE.md)

Use the same tasks, sampling settings and evaluation interfaces for the base model and every checkpoint. The shared evaluator measures pass@1, correctness plus tool-efficiency reward, tool calls and token use.

## Which environment does it use?

The launcher accepts the same mode names as training:

| Training mode | Evaluation environment | Full test set |
|---|---|---|
| `whitebox` | SETA bash tools | 250 tasks × one attempt |
| `opencode` | Harbor: OpenCode, Claude Code, Codex and Mini-SWE-Agent | 250 tasks × four harnesses |
| `multi_harness` | The same four Harbor harnesses | 250 tasks × four harnesses |

An OpenCode-trained checkpoint is tested across four harnesses to measure transfer. This evaluates OpenCode through Harbor; it does not invoke the separate native OpenCode environment. Whitebox has its own tool interface, so report that result separately.

## Start with a small comparison

After [installation and task preparation](../REPRODUCE.md#1-install-and-prepare-the-tasks), run these commands from `05-multi-harness-rl` on a two-GPU machine:

```bash
# Base model: 25 fixed tasks across four harnesses, 100 attempts total.
python jobs/run.py eval --mode multi_harness --tasks 25 \
  --concurrency 35 --output runs/baseline

# After checkpoint 100 has finished saving, repeat the same measurement.
python jobs/run.py eval --mode multi_harness --tasks 25 \
  --checkpoint runs/harbor-pilot/checkpoint-100 --step 100 \
  --concurrency 35 --output runs/checkpoint-100-eval
```

Use `--mode opencode` for a native OpenCode-trained policy or `--mode whitebox` for SETA. Add `--model Qwen/Qwen3.5-2B` for Qwen; the default is LFM2.5-2.6B. Omit `--tasks 25` to evaluate all 250 tasks. A tiny subset is useful for checking that a run works; use the full split for the final comparison.

The launcher starts vLLM with TP1/DP2 and the appropriate OpenEnv server inside the allocation. The evaluator uses a different allocation from training. Follow the [HF Jobs and Slurm recipes](../REPRODUCE.md) to submit the same commands remotely.

## Already serving the model and environment?

Run the evaluator directly. Its two execution modes are `blackbox` and `whitebox`:

```bash
python -m eval.evaluate --mode blackbox \
  --model LiquidAI/LFM2.5-2.6B --checkpoint YOUR_IMMUTABLE_CHECKPOINT_ID \
  --server http://127.0.0.1:8200 --vllm-url http://127.0.0.1:8000 \
  --tasks 25 --concurrency 35 --output runs/custom-eval
```

`--server` can point to your local server or the matching deployed Space: Harbor for `blackbox`, SETA for `whitebox`. For a remote Harbor server, also pass `--env-llm-url` with an inference endpoint reachable from that server. Its localhost is different from your machine's localhost. The model endpoint must support the token capture checks used by Harbor.

## Read the result, then retry failures

| File | What to look for |
|---|---|
| `summary.json` | Pass@1 and coverage, plus reward, tool and token averages; harness and difficulty breakdowns |
| `pairs/*.json` | Each task/harness result, elapsed time and any ungraded error |
| `identity.json` | Model/checkpoint, task identities, sampling and evaluator version |
| `progress.json` | Results completed so far |
| `trackio/` | Local aggregate evaluation metrics |

Always read **coverage alongside pass@1**. An infrastructure failure has no grade and is excluded from the observed score. A graded wrong answer counts as zero. Incomplete coverage exits with status 2.

To recover, rerun the same command with the same output directory. It retains both correct and incorrect graded attempts and retries only ungraded or missing pairs. Use a fresh directory for a different model, checkpoint or evaluator version.

Add `--space-id your-org/your-trackio-space` to publish aggregate evaluation metrics. Per-harness and difficulty details are retained in `summary.json`. The [article](https://huggingface.co/spaces/FineEnvs/multi-harness-rl) shows how to interpret answer quality and tool savings together.
