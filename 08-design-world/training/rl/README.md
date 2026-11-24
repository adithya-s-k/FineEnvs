# Multi-turn vision RL on DesignGym (Tinker, Miles, SkyRL)

Three RL stacks train on DesignGym episodes. Tinker (`tinker/`) runs sampling and LoRA training on Thinking Machines'
service, with the rollout loop and the env server on a CPU machine or job; see `tinker/README.md`. The rest of this
page covers Miles and SkyRL, which run everything on GPU nodes. Smoke results, timings and the framework
recommendation are at the end.

Miles and SkyRL drive the same DesignGym GUI episodes (computer use with screenshots, mocked editors, reward v3 at
submit) on one Hugging Face Jobs node, with the environment running inside the GPU job:

| | Miles (`radixark/miles`) | SkyRL (`NovaSky-AI/SkyRL`) |
|---|---|---|
| engine | SGLang `/generate`, `input_ids` + `image_data` | vLLM `/inference/v1/generate`, token ids + `mm_features` |
| trainer | Megatron-Bridge (LoRA), colocated | Megatron (LoRA, merged into vLLM), colocated |
| hook | custom generate function | custom generator (`get_generator`) |
| episode → samples | one `Sample` per chunk, shared `rollout_id` | one step-wise row per chunk, `is_last_step` |
| image | `radixark/miles:latest` | `novaskyai/skyrl-train-ray-2.57.0-py3.12-cu13.0-megatron` + `uv sync --extra megatron` |

## Layout

```
tinker/           Tinker RL + eval (openenv_rl adapter, DesignGym glue, HF Jobs launchers): see tinker/README.md
common/
  env_client.py   async OpenEnv WebSocket client for one episode (needs only `websockets`)
  qwen_chat.py    token segments for Qwen3.5/3.6/3.8 VLMs (patched tool listing, thinking off), qwen3_coder parser
  episode.py      the episode loop (rollout.py's), image window 8,3 -> chunks, chunk cap, forced submit
  test_core.py    offline tests (tokenizer/processor only) + a live scripted episode against a DesignGym server
  hf_jobs.py      bundle this checkout -> the runs dataset, run/follow/cancel jobs
  job_common.sh   in-job steps: bundle, DesignGym server (Chromium), model download, token-scrubbed upload
miles/
  designgym_generate.py   --custom-generate-function-path rl.miles.designgym_generate.generate
  train_designgym.py      composes the Miles command (bridge mode, LoRA, colocate) per model
  job.sh                  the HF Job
skyrl/
  designgym_generator.py  DesignGymVLMGenerator (step-wise VLM rows; images via vLLM's render endpoint)
  entrypoint.py           BasePPOExp with our generator; registers a no-op `designgym` env class
  job.sh                  the HF Job (clones SkyRL at SKYRL_REF, builds the megatron venv)
```

## The shared rollout core

`common/episode.py` plays one episode exactly as the evaluation harness (`envs/designgym/rollout.py`) does: reset
with `max_steps = max_turns`, `start_task` hidden, opening user turn = target image + editor screenshot + intro,
tool calls in order with text as tool responses and images in a following user turn ("Image returned by <tool>."),
an automatic screenshot when the turn took none, the nudge for turns without calls (at most twice), and a forced
`submit` at the turn cap. The reward is the env's grade (`DESIGN_CANVAS_REWARD=3` on the server).

Tokens are rendered once and never re-rendered (token-in/token-out): the opening and every observation are rendered
with the model's chat template (tool listing patched to bare function objects as in eval and SFT, `enable_thinking=
False`), and each sampled turn is kept exactly as the engine returned it, so the trainer sees the engine's tokens.
Each image is its own segment (`<|vision_start|>` + N `<|image_pad|>` + `<|vision_end|>`, N from the model's image
processor: 1000 tokens for a 1280x800 screenshot).

Image window (`8,3`): the opening images never drop; past 8 screenshots all but the newest 3 become
`[earlier screenshot omitted]`. A drop rewrites the prompt, so the current chunk closes and the next starts from the
windowed history: every chunk is a training sample whose context is exactly what the engine saw, with loss only on
the turns sampled inside it, and all chunks carry the episode reward. `max_chunks` ends an episode (forced submit)
rather than open one chunk more; Miles needs that (fixed samples per episode, see below).

Thinking off and history: the template's generation prompt puts an empty `<think>\n\n</think>\n\n` before each
sampled turn, and TITO keeps it in history. Qwen3.8 (`preserve_thinking`) renders history the same way; Qwen3.6's
template would drop it from turns before the last user message, so a re-rendering eval differs from training by
those empty blocks (as in the Tinker runs, which also kept them).

```bash
cd 08-design-world/training
uv run --with torch --with torchvision --with "transformers==5.19.0" --with pillow --with websockets --with pytest \
    python -m pytest rl/common/test_core.py -q
DESIGNGYM_URL=https://fineenvs-designgym.hf.space uv run --with ... python -m pytest rl/common/test_core.py -q -k live
```

The offline tests download only the main models' tokenizer and processor files (no weights); `TEST_MODELS` picks the
models (default `Qwen/Qwen3.6-35B-A3B,Qwen/Qwen3.8-27B`). The live test plays scripted episodes against the public
DesignGym Space (`FineEnvs/DesignGym`) or any other server.

## Running

From `08-design-world/training`. The launcher bundles the shared core and the adapters (as `rl/`), the env package
(`envs/designgym`, as `designgym/`) and the task list, uploads them to the runs dataset (`--runs-repo`, or
`$RL_RUNS_REPO`, default `<namespace>/designworld-rl-runs`) as `<framework>/<run>/code.tar.gz`, and starts the job
under `--namespace` (or `$HF_JOBS_NAMESPACE`, default the token's user); results (train log, step metrics, GPU log, env
log, checkpoint tree / LoRA adapters) land next to it. `HF_TOKEN` comes from the environment or the nearest `.env` up
the tree. `--tasks N` takes the first N rows of `--tasks-file` (JSONL of reset kwargs; default
`tinker/tasks/rl-train-gui.jsonl` if present, else GUI tasks of the env's `rl` split, one mocked editor each in turn).

```bash
# Miles: Qwen3.6-35B-A3B LoRA r32 on 8xH200 (TP2 EP8, SGLang TP8), 4 tasks x group 4, 20 turns, 2 rollouts x 2 steps
python -m rl.common.hf_jobs launch miles --flavor h200x8 --timeout 2h --name smoke-35b \
    --model Qwen/Qwen3.6-35B-A3B --tasks 4 -- --gpus 8 --rollout-batch-size 4 --group-size 4 --max-turns 20

# Miles: Qwen3.8-27B on 4xH200 (TP2, SGLang 2xTP2)
python -m rl.common.hf_jobs launch miles --flavor h200x4 --timeout 75m --name smoke-27b \
    --model Qwen/Qwen3.8-27B --tasks 4 -- --gpus 4 --rollout-batch-size 4 --group-size 4 --max-turns 20 --sglang-mem 0.45

# SkyRL: Qwen3.8-27B LoRA r32 on 8xH200 (Megatron TP4, 2 vLLM engines x TP4)
python -m rl.common.hf_jobs launch skyrl --flavor h200x8 --timeout 2h --name smoke-27b \
    --model Qwen/Qwen3.8-27B --tasks 8 -- generator.inference_engine.num_engines=2 \
    generator.inference_engine.tensor_parallel_size=4 trainer.policy.megatron_config.tensor_model_parallel_size=4 \
    trainer.train_batch_size=4 trainer.policy_mini_batch_size=4 generator.n_samples_per_prompt=4 \
    trainer.max_training_steps=2 trainer.ckpt_interval=2 generator.max_turns=20 trainer.max_prompt_length=32768 \
    generator.max_input_length=32768 generator.sampling_params.max_generate_length=1024

# SkyRL: Qwen3.6-35B-A3B on 4xH200 (Megatron TP2 EP4, vLLM 1xTP4): as above plus
#   generator.inference_engine.num_engines=1 trainer.policy.megatron_config.tensor_model_parallel_size=2 \
#   trainer.policy.megatron_config.expert_model_parallel_size=4 trainer.policy.megatron_config.expert_tensor_parallel_size=1 \
#   trainer.policy.megatron_config.lora_config.experts_shared_outer_loras=false

python -m rl.common.hf_jobs logs <job_id> [--follow] [--grep "step "]
python -m rl.common.hf_jobs cancel <job_id>
```

## Framework notes

### Miles
- The generate function sets `sample.reward`, so no reward model runs.
- Every episode returns exactly `--designgym-samples-per-episode` samples (default: the window's chunk bound for the
  turn cap), padding short episodes with inert samples (`remove_sample`, same `rollout_id` and reward, one 64x64
  image so every micro-batch runs the vision tower). Reason: for multi-sample ("compact") rollouts Miles skips its
  sample trimming, and with multimodal data it skips the rollout-side scheduler; the trainer then trains
  floor(samples / global batch) x global batch samples while the rewards list keeps all of them
  (`IndexError` in `get_grpo_returns`). `train_designgym.py` sets `--global-batch-size` = samples per rollout /
  `--steps-per-rollout`.
- Qwen3.8-27B uses Megatron TP2 + sequence parallel: TP4 hung in the first backward (NCCL all-reduce timeout).
- Never pass secrets in `execute_train(extra_env_vars=...)`: Miles prints the Ray runtime env into the log. Ray
  workers inherit `HF_TOKEN` from the job environment.

### SkyRL
- SkyRL's VLM generator re-renders the conversation every turn and keeps only the last render's images, so it
  cannot carry an image window and refuses step-wise output. `DesignGymVLMGenerator` emits one step-wise row per
  chunk with that chunk's `pixel_values` / `image_grid_thw` (`generator.step_wise_trajectories=true`,
  `merge_stepwise_output=false`, `batched=false`, `trainer.remove_microbatch_padding=false`,
  `language_model_only=false` on trainer and engine).
- Images reach vLLM as SkyRL's Tinker renderer does it: each new image is rendered once alone through
  `/v1/chat/completions/render` (placeholder tokens, `mm_hash`, `kwargs_data`), spliced into `mm_features` at the
  image's position in our tokens; the same `kwargs_data` decode into the trainer's pixels.
- `generator.chat_template_kwargs` is not needed: rendering happens in the shared core with thinking off.

## Smoke results (October 2026)

Both frameworks ran the same DesignGym GUI episodes (mocked editors, reward v3, env server inside the GPU job) through
the shared core on HF Jobs. Settings: easy GUI training tasks (first 4 or 8), group 4, at most 20 turns, image window
8,3, LoRA r32 / alpha 32, lr 1e-5, thinking off, temperature 1.0, 1024 new tokens per turn, 32k context.

A smoke run passes when it goes env reset -> multi-turn screenshot rollouts served by the framework's engine -> tool
calls in DesignGym -> reward at submit (or forced submit at the cap) -> at least 2 training steps with images ->
trainer-vs-rollout logprob mismatch logged -> weights synced to the engine -> checkpoint saved.

| framework | model | flavor | result |
|---|---|---|---|
| Miles | Qwen3.6-35B-A3B | h200x8 | pass (4 steps, 2 rollouts, 2 LoRA checkpoints) |
| Miles | Qwen3.8-27B | h200x4 | pass with TP2 (4 steps, 2 rollouts, 2 LoRA checkpoints); TP4 hung in the first backward |
| SkyRL | Qwen3.6-35B-A3B | h200x4 | pass (2 steps, checkpoint) |
| SkyRL | Qwen3.8-27B | h200x8 | pass (2 steps, checkpoint) |

Trainer-vs-rollout logprob mismatch (mean |Δ logprob| over trained tokens, image chunks):

| | Qwen3.6-35B-A3B | Qwen3.8-27B |
|---|---|---|
| Miles (`train/train_rollout_logprob_abs_diff`) | 0.0108–0.0117 (KL 0.0015–0.0018) | 0.0059–0.0064 (KL 0.0008–0.0010) |
| SkyRL (`policy/rollout_train_logprobs_abs_diff_mean`) | 0.046, 0.044 | 0.022, 0.025 |

Both are small enough to show that engine and trainer use the vision tower identically (a trainer ignoring the
images would be off by whole nats after every screenshot). SkyRL is consistently 2-4x Miles; the step-1 value (LoRA
B = 0, merged weights equal to the base) rules out merge rounding, which leaves kernel differences or the vision path.

Time per step (all colocated: engine and trainer share the GPUs and sleep in turn):

| run | layout | rollout (16 episodes × ≤20 turns) | logprob fwd | train | weight sync | step |
|---|---|---|---|---|---|---|
| Miles 35B-A3B, 8×H200 | Megatron-Bridge TP2+SP, EP8; SGLang 1×TP8 | 138–154 s | 25 s | 58 s | 8 s | 283 s per rollout (2 optimizer steps) |
| Miles 27B, 4×H200 | Megatron-Bridge TP2+SP; SGLang 2×TP2 | 153–154 s | 41 s | 114 s | 1.4–1.8 s | 376 s per rollout (2 optimizer steps) |
| SkyRL 35B-A3B, 4×H200 | Megatron TP2, EP4; vLLM 1×TP4 | 163–173 s | 34 s | 79 s | 12 s | 300 s |
| SkyRL 27B, 8×H200 | Megatron TP4; vLLM 2×TP4 | 162–164 s | 50 s | 96 s | 4.6–5 s | 331 s |

The first step of each run is 1.5-2x slower (JIT). Rollouts are bound by the env and the 20 sequential turns, not
the engine: token-in/token-out keeps prompts stable between window drops, so prefixes stay cached.

Recommendation: Miles as the primary stack, SkyRL as the second. Miles' trainer-vs-rollout mismatch is 2-7x lower
on both models, it serves the LoRA in SGLang without merging and saves PEFT adapters per rollout (which slot into the
merge + vLLM eval path), and its image has everything installed. Its costs are the fixed-samples-per-episode
workaround and TP2 for the 27B. SkyRL's step-wise rows are the cleaner abstraction for the image window; its higher
mismatch is the thing to investigate before long runs.
