# Training handoff

RetroEnv is an OpenEnv HTTP/MCP environment. Start it against a release
(`data/release/RetroEnv-RL`), then give the trainer deterministic
`reset(split="train", index=i)` rows. Download the release with
`uv run hf download LiteFold/RetroEnv --repo-type dataset --local-dir data/release/RetroEnv-RL`.
It is fully open and carries every split's known routes, which the environment needs: they
feed the reward and hide a train task's own reactions from the tools.

Before spending GPU time:

1. Run `uv run pytest` and `uv run python -m dataset.audit_release`.
2. Run `eval/run_eval.py` on the base model for a few dev tasks and read its episodes.
3. Verify every task has nonzero within-group reward variance across 4–8
   samples; flat groups produce no GRPO advantage.
4. Overfit 2–4 fixed task indices and confirm reward, valid graphs, exact stock
   claims, and `emit_routes` completion improve together.
5. Train against the same server image and private bundle used for evaluation.

The policy sees tool outputs and the public task only. It must never load
`tasks-private`, the `library/` files, or the stock file directly. `stock_retrieve` is the sole stock surface.

`grpo_smoke.py` is the minimal TRL 1.12 recipe. From a current TRL environment:

```bash
uv run --with 'trl[vllm]>=1.12,<1.13' --with datasets --with peft \
  python train/grpo_smoke.py
```

To train against the OpenEnv server instead, the same one evaluation uses, start it
and point the script at it. Each rollout then holds one WebSocket session:

```bash
RETROENV_BENCHMARK_DIR=data/release/RetroEnv-RL uv run uvicorn retroenv_openenv.server:app --port 8000 &
RETROENV_SERVER=http://127.0.0.1:8000 MAX_TASKS=8 uv run --with 'trl[vllm]>=1.12,<1.13' \
  --with datasets --with peft python train/grpo_smoke.py
```

`MAX_TASKS` limits the train split for an overfit run. In-process, it uses
`RetroRouteTrainingEnv` as `environment_factory`, four generations per
indexed task, Dr. GRPO, ten optimizer steps, and optional JSONL traces via
`RETROENV_TRACE_PATH=outputs/episodes.jsonl`. Treat it as a wiring/overfit run;
do not publish its metric as model quality.

## GRPO on HF Jobs

Two developer scripts train the same experiment, LoRA on Qwen3.8-27B (or Qwen3.6-35B-A3B)
against the RetroEnv server, with the evaluation board's system prompt and non-thinking chat
template. Read their numbered sections in order:

| Script | Trainer | Batch |
|---|---|---|
| [sync_grpo.py](sync_grpo.py) | TRL `GRPOTrainer`: generate a batch, then one optimizer step | 16 episodes per step (2 tasks × 8) |
| [async_grpo.py](async_grpo.py) | TRL `AsyncGRPOTrainer`: generation keeps running while it trains | the same, at most 4 policy versions stale |

Both use `RemoteRetroRouteEnv` as `environment_factory`: its public methods are the tools, each
call goes to the server over one WebSocket session per rollout, and the `emit_routes` grade is
the reward. An episode that never submits scores 0, as in evaluation; one whose session broke
scores NaN and stays out of its group's baseline. The default curriculum is the 51,358 standard
train tasks with a depth budget of 4 (`--max-depth`).

Evaluation tells the model its turns left and forces `emit_routes` on the last one; TRL's tool
loop does neither. Without a substitute, most rollouts researched until the loop ended and never
submitted, so whole groups tied at 0 and gave no gradient. So every tool result carries the
calls left in the server's 32-call budget and, with four left, a reminder to submit; the prompt
keeps the board's 16 turns and training allows 20, so an overrun can still submit. Each graded
episode is appended to `episodes.jsonl` in the run directory (task, reward, submitted, tool calls).

### What makes the gradient

| Choice | Why |
|---|---|
| Tool results carry the budget left; 20 turns against the prompt's 16 | Without them most rollouts never submitted, and groups tied at 0 |
| A failed submission's grade × 0.5 (`--partial-credit`) | One accepted cut with no stock leaf earns 0.55 against ~0.88 for a solve; halving it keeps solving clearly ahead |
| `vllm_importance_sampling_mode="token_truncate"`, cap 2 (sync) | TRL's default is one ratio per sequence; over a 16-turn episode it drifts far from 1 and zeroes whole episodes |
| `epsilon_high=0.28` | DAPO's clip-higher, against entropy collapse |
| `mask_truncated_completions` (sync) | An episode cut off by the length limit is not a verdict |
| NaN for a broken session | Infrastructure failure stays out of the group baseline instead of scoring 0 |
| `solved` and `submitted` logged with weight 0 (sync); `episodes.jsonl` (both) | Pass rate and submission rate beside the reward, so a rising reward can be told apart from reward hacking |

Watch `rewards/solved/mean` and `rewards/submitted/mean` (sync), `reward_std` and
`frac_reward_zero_std` (how many groups gave no gradient), `entropy`, the clip ratios, and the
`sampling/importance_sampling_ratio` range. Evaluate checkpoints on core30 with the same harness
as the board; the base model scores pass@1 0.100 and reward 0.335 there (`base-eval`).

[jobs/](jobs) runs them on HF Jobs the way `05-multi-harness-rl/jobs` does, on `h200x2`: vLLM
serves the model on one GPU, the trainer runs on the other, and the RetroEnv server runs on CPU
from the project's locked environment, preparing its serving snapshot from the public bucket
mounted at `/data`. Checkpoints, logs and boards are written to an output bucket as they happen.

```bash
B=FineEnvs/retroenv-rl-runs
python train/jobs/hf_job.py check --name check --bucket $B --submit            # CPU: installs and setup checks
python train/jobs/hf_job.py smoke --mode sync --name sync-smoke --bucket $B --submit
python train/jobs/hf_job.py train --mode async --name async-100 --bucket $B --steps 100 --submit
python train/jobs/hf_job.py eval --name async-100-eval --bucket $B \
    --checkpoint /outputs/async-100/checkpoint-100 --submit                       # the core30 board
python train/jobs/hf_job.py eval --name base-eval --bucket $B --submit           # the base model, same harness
```

Without `--submit` the launcher prints the job plan. `check_setup.py` runs first in every job and
fails before the GPUs load anything if TRL cannot train through the model's chat template, a
tool call does not round-trip through it, or a tool's docstring does not become a schema. The
smoke test runs two updates of four rollouts each, requires a non-zero gradient and episode
traces, checks both checkpoints, then merges checkpoint 2 and evaluates it on two tasks. Evaluation merges the adapter into the base weights (`jobs/merge_lora.py`),
serves them with vLLM and runs `eval/evaluate.py` with the board's protocol, so its RESULTS.md
lines up with `runs/core30-nothink`.

## Scaling RL

The release's train split holds every eligible target that shares no leakage key with a
held-out task (see `manifest.json` for counts by shortest-route depth and tier), plus
constraint variants. Train against it, watch dev, and report on test_id and test_hard.

- **Curriculum.** `MAX_ROUTE_STEPS` keeps only tasks whose depth budget is at most that
  value (a public field). Start at 3 or 4 and raise it once groups stop being flat.
- **Precedents mirror evaluation.** Tools search only train-visible corpus reactions, and
  a train task's search also hides its own reactions, patents, scaffolds and
  near-duplicates (`PrecedentIndex.hidden`), so RL cannot learn to copy an answer that
  evaluation never offers. `validate_disconnection` likewise never confirms a train task's
  own reactions by corpus lookup.
- **Flat groups.** A base model that never passes gets no GRPO signal; warm-start it with
  SFT first.

### Training and evaluation tasks

Train on a fixed 2,000-task mix, 60% easy and 40% medium, drawn deterministically from
LiteFold/RetroEnv-Curated's training set (`data/eval-train12k.json`: 13,200 train-split tasks,
none sharing a leakage key with an evaluation set). Report on `core150` (the curated test board:
50/50/50 easy/medium/hard from test_id and test_hard, with core30, final_eval and core100 as its
prefixes); `val150` (50/50/50 from dev) is the dev set for checks along the way, and `heldout300`
stays sealed until the end. `train/sweeps/train-mix2k.txt` holds 1,200 easy tasks (600 each with
shortest route 2 and 3) and 800 medium (routes 2 to 8, visited round-robin), drawn with the
evaluation sets' own tiered draw:

```bash
python train/difficulty_sweep.py select --name train-mix2k --seed retroenv-train-mix-v1 \
  --quotas easy=1200,medium=800,hard=0 --shards 1 --from-set train12k
python train/jobs/hf_job.py train --mode async --bucket <owner>/<bucket> --flavor h200x8 --vllm-gpus 3 \
  --task-pool train/sweeps/train-mix2k.txt --reward solved --lr 1e-4 --steps 200 --save-steps 25 --submit
python train/jobs/hf_job.py eval --bucket <owner>/<bucket> --set core150 --harness training --attempts 4 \
  --temperature 0.8 --checkpoint /outputs/<run>/checkpoint-<n> --name <run>-<n>-core150 --submit
```

### Difficulty sweep

`difficulty_sweep.py` measures where the base model has signal before a run spends GPUs on
flat groups. `select` draws 250 standard train tasks the way the evaluation sets are drawn
(equal tiers, shortest-route depths visited round-robin) into `train/sweeps/`, split into two
shards; each shard is one evaluation job sampling every task 8 times at the trainer's
temperature:

```bash
python train/difficulty_sweep.py select
for shard in 0 1; do
  python train/jobs/hf_job.py eval --name sweep250-$shard --bucket <owner>/<bucket> --set train \
    --task-ids train/sweeps/train-sweep250-$shard.txt --attempts 8 --temperature 0.8 --concurrency 64 --submit
done
python train/difficulty_sweep.py report runs/sweep250 --server http://127.0.0.1:8010
```

`report` writes pass@1 and pass@8 by tier and by depth, what each episode ended in, a label per
task (`never`, `learnable` or `always` solved, with broken submissions re-graded as repaired),
`learnable.txt` for the next run's task list, and the solved transcripts as `sft.jsonl`.

### Overfitting run: does the reward climb on fixed tasks?

Before spending on the full pool, train on a handful of tasks over and over and check that the
solve rate climbs. `train/sweeps/overfit8-index.json` holds eight easy train tasks the base model
solves a few times in sixteen (the sweep's learnable tasks, one per reaction type);
`overfit4-index.json` the four of them with the most signal without thinking (hydrogenation,
benzyl deprotection, O-alkylation, Suzuki coupling); `think3-index.json` the three that thinking
still finds hard (carbonyl reduction, Boc protection, ether cleavage).

The run that climbs, non-thinking (Qwen3.8-27B, 2026-11-08, `ab3-n4g32` in
`AdithyaSK/retroenv-rl-runs`): four tasks, 32 attempts each, learning rate 1e-4, four updates per
round, three vLLM GPUs and five trainer GPUs. The solve rate per round went
0.27, 0.27, 0.27, 0.28, 0.41, 0.43, 0.33, 0.49, 0.47, 0.57, 0.63, 0.53, 0.60, 0.66, 0.66, 0.71,
0.73, 0.74, 0.68, 0.76, 0.78 over its 48 updates (about 65 minutes of training, checkpoints every
8), every task improving (per task, of 32: 10→19, 7→18, 15→31, 7→29), submissions at 98-100%,
tool calls per episode falling from 27 to 19, and KL near 0.001. At 2e-4 it climbed no faster (0.60 by round 11)
and entropy fell faster; eight tasks of sixteen attempts climbed later and less.

What it learned, checked by replaying checkpoint-48 in the training harness at learning rate 0
(`--init-adapter`, which also records each graded submission and the tools called) and re-scoring
every submission with the verifier:

- The grades hold: re-scoring agrees with the logged grade (5 of 249 differed only because the
  trace kept a later, ignored call; the server grades `emit_routes` once).
- It solves 0.86 in the training harness, about as it ended training. Every solve is a one-step
  route from template-supported reactions to in-stock precursors, the kind the base model already
  found; per task it settles on one route (86 of 91, 60 of 66, 27 of 28, 16 of 34): memorizing
  these tasks' answers, which is what an overfitting run is for. It still looks stock up
  (3-16 `stock_retrieve` calls per solved episode).
- One pattern to tighten in the verifier: half the Suzuki solves couple two aryl bromides through
  a broad "other cross-coupling" template, which a chemist would not accept as a Suzuki route.
- Evaluated from its merged weights it first scored like the base model, under either harness.
  The merge was at fault: transformers' `save_pretrained` of the merged Qwen3.5 model wrote the base
  weights for every layer past the eighth, so every merged-checkpoint evaluation before this measured
  a mostly untrained model. `jobs/merge_lora.py` now merges in the checkpoint files (weight +
  scale x B @ A per tensor) and checks the saved weights against the base layer by layer. With it,
  under the training harness (below) checkpoint-48 solves 0.78 on these tasks (25 of 32; per task
  8, 8, 5 and 4 of 8) and the base 0.25, as in training. Under the board's agent (`eval/evaluate.py`)
  it solves 0.62 (20 of 32) and the base 0.50: part of the gain carries over. The board's agent
  lifts the base model on its own (the server's tool schemas, a forced final submission), and it
  does not hand broken JSON back, which fails a quarter of checkpoint-48's board episodes.

Evaluate in the context RL trains in with `--harness training` (`eval/run_eval.py`, passed through
by `evaluate.py` and the eval job): the training environment's tools as TRL renders them, its tool
results (calls remaining, the reminder, malformed JSON handed back twice), several calls a turn,
20 turns and no forced or automatic submission. It runs over any OpenAI-compatible endpoint, so
base, trained, open and frontier models are compared in one harness. The curated boards (`val150`,
`core100`, `core150`, `heldout300`, LiteFold/RetroEnv-Curated) are served in jobs as their own sets:

```bash
python train/jobs/hf_job.py eval --bucket <owner>/<bucket> --set val150 --harness training \
  --temperature 0.8 --checkpoint /outputs/<run>/checkpoint-<n> --name <run>-val150
```

```bash
python train/jobs/hf_job.py train --mode async --bucket <owner>/<bucket> --flavor h200x8 \
  --vllm-gpus 3 --train-tasks train/sweeps/overfit4-index.json --generations 32 --reward solved \
  --lr 1e-4 --updates-per-round 4 --inflight 128 --steps 48 --save-steps 8 --name overfit4-nothink --submit
# thinking: --thinking --reasoning-effort low --turn-tokens 32768, the default 4/4 GPU split, and
# tasks thinking does not already solve (--train-tasks train/sweeps/think3-index.json, with
# --updates-per-round 3 so an update is still 32 samples)
```

Thinking (low reasoning effort) climbs too, about five times slower per update. On the three tasks
it still finds hard (`think3-index.json`: carbonyl reduction, Boc protection, ether cleavage; the
other overfit tasks it already solves 70-90% of the time), 32 attempts, learning rate 1e-4, three
updates per round, four vLLM GPUs and four trainer GPUs (`ab5-t3g32`): episodes started under
the base policy solved 0.27; under updates 1-4, 5-8, 9-12, 13-16 and 17-20 they solved 0.35,
0.25, 0.30, 0.40 and 0.45, broken JSON falling from 9% to under 5% and unsubmitted episodes to
about zero, KL near 0.001-0.002. Twenty updates took about 2.5 hours: a group of 32 waits on its
slowest episode (10-25 minutes, ~20-30k generated tokens), and an update trains ~30 long rows.
At 2e-4 it did not climb (0.23 to 0.26). Run thinking with the reasoning trim removed from the
training template (`tokenizer_for`), which this run predates: trimmed, a turn whose thinking began
or ended with whitespace re-rendered differently from its first token and forked a new row, ~2
rows per episode; untrimmed, drifts fell from 20k to 7k tokens and forks from ~10% to ~5% of turns.
A 5/3 vLLM/trainer split made the trainer the bottleneck (352 s an update); keep 4/4.

Continued from that run's checkpoint-24 with the trim removed (`ab7-t3g32-from24`), episodes
started under the checkpoint solved 0.31 and under updates 1-3, 4-6 and 7-9 0.44, 0.44 and 0.42:
one step up, then flat. About 13% of its episodes ran out of context before submitting and scored
0, and the loss is averaged over every token in the batch, so these long failures carried more of
each update than their share of episodes. Count a thinking run's solve rate by the policy an
episode started under, once that policy's slow episodes are in: the first episodes back are the
quick ones, mostly solves (updates 7-9 read 0.76 at 33 episodes and 0.42 at 119).

Masking those episodes out of the loss (`--mask-truncated`, DAPO's overlong filtering: the reward
is NaN, so the episode leaves its group's mean and std and gets advantage 0) did not change the
shape from the base model either: 0.23 to 0.30, 0.40 and 0.33 over nine updates at 1e-4, 0.24 to
0.34, 0.40 and 0.34 at 2e-4. The trainer was throwing most of the data away. A thinking episode
trains as about 2.2 rows (a later turn that re-renders differently from what was generated
starts a new row), while an update was sized for one row per episode: three updates took 96 of a
round's ~210 rows, each update saw about 14 episodes, often of one task, and the backlog aged to
the staleness limit and was dropped (356 rows at once, then 14 minutes idle). Updates are now
sized to a round's rows (`--rows-per-episode`, 2.2 by default with `--thinking`: 72 rows an update).
With both changes, from the base model at 2e-4 (`ab9-t3g32-rows-lr2e4`), every update trained 72
rows with nothing dropped, staleness rising one a version while the base policy's backlog drained.
The run was stopped after 8 updates (about 1.7 hours): episodes started under updates 1-3 solved
0.22 task-balanced against 0.27 under the base policy, where the non-thinking run was still flat
at the same point. Whether thinking climbs as non-thinking does is still open; a thinking round
takes about 28 minutes against non-thinking's 3, so measure it over at least 20 updates.

- **Same tasks every round.** The tasks stay in a fixed order and a round is a group (`--generations`)
  of every task, one sample per row per GPU. `--updates-per-round` splits a round into that many
  optimizer steps: with one update per round the adapter barely moved (KL 0.001 after three), and
  Adam moves each weight by about the learning rate per step whatever the batch. A sample may be
  `2 + 2 x updates-per-round` policy versions stale, so a round's samples are not dropped while the
  trainer is a round behind. Each task's r-th group gives a per-task curve.
- **Reward `solved`:** 1 for a solve, 0 otherwise. The grade's partial-credit band can be gamed
  (one accepted cut scores 0.6 of a solve), so a rising reward here can only mean more solves.
- **Thinking** keeps the template as shipped, so earlier turns' thinking stays in the history and
  an episode trains as one row (Qwen, MiniMax, Kimi, GLM and DeepSeek all keep it). Reasoning
  effort is `--reasoning-effort` (default `medium`; the template's `xhigh` asks for long
  deliberation, and at `medium` some tasks' first turn thought past the 16k cap and never called a
  tool), sampling is untruncated at temperature 1, and each turn's token cap is clamped to the
  context left.
- **Turn speed.** Every turn resends the episode so far. vLLM 0.25 leaves prefix caching off for
  hybrid models like Qwen3.8 and its data-parallel server sends each request to the least-busy
  replica, so turns took ~33 s re-reading their history. Training asks for prefix caching and pins
  each episode to one replica (`X-data-parallel-rank`).
- **Malformed submissions** come back ungraded, up to twice, with where the JSON broke and the
  brackets left to close; only a tree the grader can read spends the single graded call.
- **Environment.** By default each job serves episodes itself (256 sessions). `--env-url`
  points it at a hosted Space instead; the Space holds 64 sessions across every job using it.
- **Fixed on the way:** an unpinned `kernels` broke model loading, two jobs exhausted the
  Space's sessions, an EIO on the bucket mount ended the rollout process (traces now go to local
  disk and are copied each minute), the turn cap patched TRL's controller instead of
  `_AsyncRolloutLoop`, and a one-version staleness limit dropped 112 of a step's samples.

## SFT warm start

The scripted chemist (`retroenv_openenv/agent_reference.py`, `--provider reference` in
`eval/run_eval.py`) replays each task's compliant known routes through the server, so its
episodes are graded like evaluation episodes. The previous SFT exporter and the v3 SFT
datasets are archived with the old benchmarks; regenerate trajectories from this release's
train split before training on them.

`sft_smoke.py` is the matching TRL recipe: LoRA on Qwen3.5-4B, loss on assistant turns
only, and a merged checkpoint that `grpo_smoke.py` can start from
(`MODEL=outputs/retroenv-sft-smoke/merged`).
