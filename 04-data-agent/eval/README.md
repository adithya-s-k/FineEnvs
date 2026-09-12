# Evaluating a data-agent environment

```bash
python eval_env.py --env opencode --server http://127.0.0.1:8200 \
    --llm-url $VLLM/v1 --model Qwen/Qwen3.5-2B --split eval --k 4 \
    --step-limit 25 --concurrency 16
```

## Concurrency is the setting that decides whether the numbers mean anything

Measured on base Qwen3.5-2B over the 144-task `eval` split:

| concurrency | step limit | rollout length | excluded |
| --- | --- | --- | --- |
| 64 | 10 | ~7 turns | **1 / 576** |
| 64 | 25 | ~16 turns | **257 / 576 (44.6%)** |
| 16 | 25 | ~14 turns (training) | **0 over ~450 rollouts** |

The failure is PROGRESSIVE, not a constant rate -- 1%, 3%, 35%, 89%, 40%, 100% across the run, with
1,625 server-side WebSocket errors and `ConnectionClosedError: sent 1011 (internal error) keepalive
ping timeout` dominating. One MCP client per rollout, a single-process uvicorn behind them, and
longer rollouts holding their sessions longer: the same 64 clients overlap far more at 16 turns than
at 7, until the server cannot service keepalives at all.

So a concurrency that was fine yesterday is not fine after any change that lengthens a rollout.
**Read the exclusion count before the score.** At 44.6% excluded there is no score worth reading, and
the number it printed (pass@1 0.0638) looked entirely plausible next to the 0.104 reference.

## The step limit is part of the measurement, not a safety setting

At `--step-limit 10`, 197 of 224 rollouts (88%) were cut off mid-task and turns pinned at exactly 9;
pass@1 came out 0.036 against a 0.104 reference. At 25 -- which is where it belongs, at or below the
reward's `step_budget` of 30 -- the same model scores 0.1058 with turns at 14.2. A cap tight enough
to truncate the work measures the cap, not the model.

## What the driver reports, and in what order

Exclusions first, then the score, because a reward of `None` is an UNGRADED rollout and not a zero:
one is infrastructure failing, the other says the policy was wrong, and averaging them together
flatters or damns a model for no reason.

It also prints WHICH tasks were measured rather than only how many, and dispatches in a fixed
shuffled order so a partial run is an unbiased sample of the split instead of its easy prefix. And it
prints mean turns next to the score, which diagnoses an agent faster than the score does: 9-11 is
working, under 7 means it quit before reading the data, over 25 means it thrashed.

Passes run with a barrier between them. E2B builds a template per task on first use and concurrent
first use races that build, so running all k samples of a task at once makes k-1 of them fail in ~7 s
with `404: tag 'default' does not exist`. Pass 0 warms every template; later passes cannot race it.
