# Evaluation

Predictions are JSONL. Each row identifies a task and carries ranked attempts:

```json
{
  "task_id": "retro_...",
  "attempts": [
    {
      "submission": {"schema_version": "retro-route-graph-v1", "routes": []},
      "tool_calls": 6,
      "invalid_proposals": 1
    }
  ]
}
```

Missing tasks fail Pass@k rather than disappearing from the denominator.
Duplicate prediction rows and unknown task IDs are errors. All rewards are
recomputed from private tasks and the pinned stock; submitted rewards are never
trusted.

`run_baselines.py` checks the reward endpoints. `run_model.py` drives any
OpenAI-compatible tool-calling endpoint, stores raw episodes, and writes a
separate offline report. It checkpoints every attempt atomically, supports
`--resume`, applies a per-request timeout, records model resolution and usage,
and can stop scheduling at a provider-reported cost cap. The last model turn
is restricted to `emit_routes`, ensuring every episode gets a scoreable final
graph attempt. Empty assistant turns receive one standardized recovery prompt;
after two empties, one terminal emit-only turn prevents unbounded provider
latency.

The initial frozen board is 20 tasks at temperature 0, one attempt per task,
and 16 model turns. Exact model IDs and the dated OpenRouter price snapshot are
in `benchmark_models.json`; `estimate_board_cost.py` projects cost from measured
episode tokens. `dataset/audit_benchmark.py` must pass before a board run.
Large boards can use non-overlapping `--start-index`/`--limit` task shards;
`merge_model_shards.py` rejects duplicate, missing, or configuration-mismatched
shards and recomputes the merged report. `summarize_board.py` then creates the
cross-model JSON and Markdown table while flagging configured models that are
still missing.
