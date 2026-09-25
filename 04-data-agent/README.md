# Data Agent: compare who owns the agent loop

Train and evaluate **Qwen3.5-2B** on the same fixed data-analysis tasks, using three OpenEnv implementations.

| Recipe | Agent loop | Trainer | Environment |
| --- | --- | --- | --- |
| `harbor-multi` | OpenCode, Claude Code, Codex and Mini-SWE-Agent via Harbor | AsyncGRPO | [Blackbox Harbor Space](https://huggingface.co/spaces/HuggingEnvs/data-agent-blackbox-harbor-env) |
| `harbor-opencode` | OpenCode via Harbor | AsyncGRPO | Same Harbor Space |
| `native-opencode` | Original standalone OpenCode adapter | AsyncGRPO | [Blackbox OpenCode Space](https://huggingface.co/spaces/HuggingEnvs/data-agent-blackbox-opencode-env) |
| `seta` | Model calls native bash/SETA tools through TRL | Synchronous GRPO | [SETA Whitebox Space](https://huggingface.co/spaces/HuggingEnvs/data-agent-seta-whitebox-env) |

Start with **[reproduce.md](reproduce.md)** for local/Slurm and HF Jobs instructions. See **[results.md](results.md)** for measured pass@1, checkpoint curves, difficulty breakdowns and the limits of the comparison.

```bash
cd 04-data-agent
python reproduce.py prepare --recipe harbor-opencode --env-file .env
python reproduce.py --help
```

Training uses 1,000 fixed tasks (150 easy, 600 medium, 250 hard), eight rollouts per selected task, LR `3e-6`, checkpoint saves every 50 optimizer steps and independent evaluations every 100. The four-harness recipe assigns one harness to each task per pass. Task count, rollout count and optimizer steps are different quantities.

Checkpoint evaluation is pass@1 on 250 held-out tasks: 1,000 cells through the four Harbor harnesses for either async trainer; 250 native bash/SETA cells for the sync trainer. Hub evaluation defaults to concurrency **35**. Each Space serves both training and evaluation; reserved sandbox slots protect training. Evaluation uses separate inference GPUs and never swaps the trainer's active weights.

Exact engine token IDs, processed log probabilities and loss masks are retained. Prompt rewrites can produce multiple training rows; the async recipe consumes complete rollout groups before updating. This prevents partial-group admission but does not establish that different harnesses receive identical gradient weighting.

## Validated environments

<!-- BEGIN:matrix -->
| Env | Tools | Backend | `openenv` |
|---|---|---|---|
| **blackbox-opencode** | agent-owned | `e2b / hf / daytona` | ✅ |
| **blackbox-harbor** | agent-owned | `Harbor / Daytona / E2B` | ✅ |
| **whitebox-bash** | bash, read, write, edit, grep, glob, ls, submit_solution | `e2b / daytona` | ✅ |
<!-- END:matrix -->

## Folder map

| Path | Purpose |
| --- | --- |
| `reproduce.py` | Main command: prepare, upload, deploy, smoke, evaluate, train |
| `hf/configs/` | Shared recipe, immutable model/source/task pins |
| `hf/locks/` | Separate hash-locked training and environment dependencies |
| `hf/runtime/` | Shared Space/Job runtime, artifact upload, eval coordinator, TiTO audits |
| `envs/` | Standalone OpenCode and whitebox implementations; Harbor uses OpenEnv |
| `train/` | Trainer recipes, deterministic schedule, atomic rollout batching, save/resume |
| `eval/` | Evaluation client, first-graded ledger and scoring checks |
| `serve/` | Validated vLLM launcher |
| `tools/` | Shared native evaluators, capture audits and logging utilities |
| `results/` | Committed score tables, plots and dated provenance |
| `temp/` | Ignored bundles, local runs and preserved superseded material |

The baseline cohorts and recipe history differ in infrastructure; the recorded curves are observational, not a controlled causal comparison. Do not reinterpret incomplete evaluations as scores. Credentials, private grading data, raw captures, model weights and Trackio databases stay out of Git.
