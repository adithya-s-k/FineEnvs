# Training handoff

RetroEnv is an OpenEnv HTTP/MCP environment. Start it against
`sample/tasks-private` and `sample/stocks`, then give the trainer deterministic
`reset(split="train", index=i)` rows. The six-task sample is intended for a
rollout and overfit test, not a headline checkpoint.

Before spending GPU time:

1. Run `uv run pytest` and `uv run python eval/run_baselines.py`.
2. Use `eval/run_model.py` on the base model and inspect its `.episodes.jsonl`.
3. Verify every task has nonzero within-group reward variance across 4–8
   samples; flat groups produce no GRPO advantage.
4. Overfit 2–4 fixed task indices and confirm reward, valid graphs, exact stock
   claims, and `emit_routes` completion improve together.
5. Train against the same server image and private bundle used for evaluation.

The policy sees tool outputs and the public task only. It must never load
`tasks-private`, `normalized-routes.jsonl`, baseline oracle files, or the stock
file directly. `stock_retrieve` is the sole stock surface.

`grpo_smoke.py` is the minimal TRL 1.12 recipe. From a current TRL environment:

```bash
uv run --with 'trl[vllm]>=1.12,<1.13' --with datasets --with peft \
  python train/grpo_smoke.py
```

It uses `RetroRouteTrainingEnv` as `environment_factory`, four generations per
indexed task, Dr. GRPO, ten optimizer steps, and optional JSONL traces via
`RETROENV_TRACE_PATH=outputs/episodes.jsonl`. Treat it as a wiring/overfit run;
do not publish its six-task metric as model quality.
