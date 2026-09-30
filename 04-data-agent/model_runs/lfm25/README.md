# LFM2.5 multi-harness training

Prepared for `LiquidAI/LFM2.5-2.6B`, revision `654f9463ce32b05d0429d76fe1f580b27d4c1ac0`. Qualification is recorded in `experiments/lfm25-multi4-20260921/`; a submitted smoke is not proof that training is ready.

| Setting | Value |
| --- | --- |
| Training data | Original Qwen 1,000 tasks: 150 easy, 600 medium, 250 hard |
| Routing | One harness per task/group; original frozen rotation; eight rollouts per group |
| Harnesses | OpenCode, Claude Code, Codex, Mini-SWE-Agent; same pinned versions |
| Optimizer | LR 3e-6, paged AdamW 8-bit, BF16, batch 4, accumulation 4 |
| Async | 32 in flight, maximum staleness 4, outstanding rollout admission 16 |
| Budget | 40,960 packed tokens, 131,072 context; 1,000 optimizer-step ceiling |
| Capture | Exact tokens/logprobs, lossless capture, all agent turns |
| Output | 4,096 tokens per model call; trainer trajectory cap 16,384, as in the original run |
| Save/eval | Save every 50 steps; independent eval every 100 |
| Eval | Same fixed 250 tasks × four harnesses, pass@1, concurrency 50 |
| Infrastructure | Local vLLM and OpenEnv, E2B task sandboxes; dedicated eval GPUs |
| Logging | Local metrics/captures/checkpoints and separate Trackio project `lfm25-multi4-20260921` |

The 1,000 tasks describe the training pool, not a guarantee that all are consumed in 1,000 optimizer steps. A task group may produce several updates. The finite scheduler records actual admitted groups and avoids replaying committed groups on resume.

Model-specific differences: native `lfm2` tool parsing, reasoning enabled, and `preserve_thinking=true` to retain prior reasoning in prompts. Sampling remains temperature 0.8/top-p 1/top-k disabled/repetition penalty 1 for comparison; it intentionally overrides the model card's inference defaults. `packing.py` isolates convolution state at packed sequence boundaries. Its test compares separate versus packed outputs and gradients.

## Commands

From the repository workspace, prepare the frozen source and verify manifests:

```bash
.venv312/bin/python HuggingEnvs/04-data-agent/model_runs/lfm25/prepare.py
.venv312/bin/python -m pytest HuggingEnvs/04-data-agent/model_runs/lfm25/test_*.py -q
.venv312/bin/python HuggingEnvs/04-data-agent/model_runs/lfm25/launch.py
```

`launch.py` prints the baseline and training commands. `--submit` submits the trainer and its independent CPU logging/eval controller only after the optimizer/save/resume/checkpoint-eval qualification passes. The full run has not been launched.

`job.sh eval-smoke` runs two fixed test tasks through all four harnesses. `job.sh train-smoke` normally needs two GPUs and performs two updates, saves, resumes through update four, then reloads checkpoint four and evaluates all four harnesses. Development-node qualification uses separately allocated inference/training GPUs on the same node because that partition allows one GPU per job. Production uses a two-GPU allocation.

`job.sh eval --model-path /absolute/checkpoint/path` evaluates saved weights. Without that argument it evaluates the pinned pretrained model. Complete scores require every requested cell to be graded and audited; failed infrastructure calls are not counted as incorrect answers.

Inputs and dependency versions are in `provenance.json`. This setup reuses the existing local virtual environments and archived source bundle; it does not install or modify shared packages. Credentials are read from the existing experiments `.env` and are never written into submission scripts.

Sources: [model card](https://huggingface.co/LiquidAI/LFM2.5-2.6B), [vLLM tool parser](https://docs.vllm.ai/en/stable/api/vllm/tool_parsers/lfm2_tool_parser/).

The development-node smoke needed `NCCL_P2P_DISABLE=1`, `NCCL_SHM_DISABLE=1`, and `NCCL_CUMEM_ENABLE=0` for weight transfer between separate Slurm allocations. NCCL then initialized successfully over the network transport. These are smoke-allocation overrides, not changes to the optimizer. See [NCCL transport controls](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html). `train_entry.py` also assigns TRL's existing LFM2.5 response template explicitly, because the pinned model's newer chat-template text is not recognized by TRL's exact template matcher.

After the GPU smoke completes, run `qualify.py /absolute/path/to/train-smoke-JOB`. It verifies all four optimizer updates, checkpoint integrity, no replay of committed groups, and eight graded/TiTO-audited rollouts from the reloaded checkpoint. This binds the launch proof to `config.json`.
