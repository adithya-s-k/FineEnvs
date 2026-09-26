# Training

Three scripts. The first two run on a laptop. The third is the GPU recipe.

```bash
# 1. The estimator, on synthetic GeoGuesser groups built from LEARNINGS.md.
python train/analyse_geoguesser.py

# 2. Four-arm GRPO on a 32-d pointer over the 2,309-word list. No GPU.
python train/tiny_grpo.py --steps 200 --out results/tiny-ablation.json

# 3. The same reward and dead-group bookkeeping, behind TRL, against Qwen.
hf jobs uv run --flavor a100-large --timeout 6h --image huggingface/trl \
  --secrets HF_TOKEN \
  -e MODEL=Qwen/Qwen3.5-4B -e REWARD_SHAPE=process -e SCALE_REWARDS=none \
  -e MAX_STEPS=200 -e NUM_GENERATIONS=8 \
  train/grpo_wordle.py
```

Step 3 has not been run in this contribution. The numbers in the project README are from steps 1 and 2. If you run step 3, the metric to watch is `alive/frac_dead` against TRL's `frac_reward_zero_std` — they should agree; both count every zero-std group — and `alive/frac_collapse`, which TRL does not report.

## Files

| file | does |
|---|---|
| `alive.py` | classify a group as live / cliff / collapse; rank / loo / GRPO advantages; failure replay |
| `tiny_grpo.py` | CPU trainer used for the ablation |
| `analyse_geoguesser.py` | within-task Monte Carlo on GeoGuesser's published medians |
| `geoguesser_run4.py` | drop-in classifier + recommended env for a GeoGuesser follow-up run |
| `grpo_wordle.py` | TRL `environment_factory` recipe, in-process Wordle |

## What "alive" actually changes

TRL will still compute `(r - mean) / (std + 1e-4)` if you leave `scale_rewards=group`. Run 1 of GeoGuesser measured a typical 60× on that term and a 10,000× ceiling when a group tied. `tiny_grpo.py --arm process-alive` uses centered ranks instead, skips a backward pass on a dead group, and oversamples tasks that produced a cliff.

Dynamic sampling — keep drawing until the group is live — is in the tiny trainer and not in `grpo_wordle.py`. TRL marks it unsupported. Forking `GRPOTrainer` to add it is a TRL patch, not an environment recipe.
