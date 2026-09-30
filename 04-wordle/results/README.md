# Results

Committed artifacts, all produced on CPU.

| file | produced by | what it is |
|---|---|---|
| `geoguesser-dead-groups.json` | `train/analyse_geoguesser.py` | 20,000 simulated GeoGuesser groups of 8, from the medians in `03-geoguesser/LEARNINGS.md` |
| `tiny-ablation.json` | `train/tiny_grpo.py --steps 200` | four arms, same seed, greedy eval on the 200-word holdout |
| `tiny-ablation.csv` | same run | the eval curve, one row per (arm, step) |

Nothing here is a Qwen run. `train/grpo_wordle.py` is the command for that; it writes `alive_stats.json` next to the adapter.

Regenerate:

```bash
python train/analyse_geoguesser.py
python train/tiny_grpo.py --steps 200 --out results/tiny-ablation.json
python -m pytest -q
```
