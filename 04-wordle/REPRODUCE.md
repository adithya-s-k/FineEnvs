# Reproducing

Every number in README.md comes from the commands below. None of them need a GPU.

```bash
cd 04-wordle

python -m pytest -q
# 24 tests. Domain, advantages, GeoGuesser classifier, tiny-policy smoke.

python envs/wordle/rollout.py zonal
# One readable episode. The answer must not appear except as a guess
# the policy itself typed.

python train/analyse_geoguesser.py
# writes results/geoguesser-dead-groups.json

python train/tiny_grpo.py --steps 200 --seed 0 --out results/tiny-ablation.json
# four arms, ~minutes on CPU. The JSON is the table in the README.
```

The Qwen recipe, which this contribution has not spent a GPU on:

```bash
hf jobs uv run --flavor a100-large --timeout 6h --image huggingface/trl \
  --secrets HF_TOKEN \
  -e MODEL=Qwen/Qwen3.5-4B \
  -e REWARD_SHAPE=process \
  -e SCALE_REWARDS=none \
  -e MAX_STEPS=200 \
  -e NUM_GENERATIONS=8 \
  -e ACCUM=2 \
  train/grpo_wordle.py
```

Wordle is in-process. There is no Space to deploy and no `ENV_URL` to get wrong.

A GeoGuesser follow-up run that uses the classifier in `train/geoguesser_run4.py` is the env in that file on top of `03-geoguesser/train/grpo_geoguesser.py`. It is not a fourth GeoGuesser run until someone launches it; LEARNINGS.md in this folder says what we expect it to change, and what we would treat as a failure.
