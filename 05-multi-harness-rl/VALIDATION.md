# Validation of the tutorial

This is the validation record for the `train/`, `eval/` and `envs/` rewrite. Historical pilots in [RESULTS.md](RESULTS.md) and the [article](../content/articles/multi-harness-rl/) used an earlier implementation.

## Runtime and CPU checks

- **35 tests passed** against OpenEnv main `7ee88d590d36ae1ac3daf2cc11bc551bca7a804f` on 1 October 2026. Tests cover task identity, grading, cleanup, service routing, tokenizer templates and the typed training contract, including partial and zero loss masks.
- A clean Python 3.12 CI environment passed 10 lightweight tests and skipped 25 runtime tests. Lint, formatting, shell syntax and the repository index also passed.
- A separate CPU installation, with **neither Torch nor TRL installed**, started all three environment servers. Their `/health`, `/web/` and Task API requests succeeded. Whitebox and native OpenCode exposed the 250 test tasks. Harbor's UI check used an empty registry; GPU checks use the prepared task registry.
- All 1,000 training and 250 test tasks were prepared with instruction/grader hash checks and no notebook overlap.
- LFM's GPU check produced identical packed and separate-sequence logits, then completed backward. The scripts run this guard before async training.
- A real Harbor/Daytona upload passed from the cluster's user ID with `TAR_OPTIONS=--no-same-owner`.

TRL's typed consumer is still in [PR #6947](https://github.com/huggingface/trl/pull/6947). Qualification explicitly uses `910138e9392dda75d3a9766e7c494d44f1d82723`; the installer defaults to upstream main and reports the missing API until that PR merges. OpenEnv always comes from upstream main. The local checks before the refresh used `86a180e`; the diff to `7ee88d5` changes release metadata and lockfiles only, with identical UI, rollout and contract code.

The fresh GPU environment uses Python 3.12, Torch 2.11.0, vLLM 0.25.1, Transformers main `d6c1e71bd717bf092f8293f0c3c9bd4a5ac5401a`, and Harbor 0.23.0. `dependencies.json` and `packages.txt` record what each installation resolved.

```bash
python -m pytest tests -q             # Lightweight checks
python -m pytest tests -q --contract  # Installed runtime and tokenizer checks
python jobs/smoke.py --mode multi_harness --output runs/harbor-smoke
```

## GPU smoke progress

Snapshot: **1 October 2026, 07:47 UTC**. Each smoke requests two optimizer updates, saves both checkpoints, and reloads checkpoint 2 for two test tasks. Blackbox evaluation covers all four harnesses, so it requires eight graded pairs. These tiny samples check execution, not benchmark quality.

| Platform | Model / mode | Training | Checkpoint reload evaluation |
|---|---|---|---|
| Slurm, two H100s | LFM / whitebox | Two updates and checkpoints verified | 2/2 graded on latest main |
| Slurm, two H100s | Qwen / whitebox | Two updates; persistent Trackio verified | 2/2 graded |
| Slurm, two H100s | LFM / native OpenCode | Two updates; persistent Trackio verified | 8/8 graded on latest main |
| Slurm, two H100s | Qwen / native OpenCode | Two updates and checkpoints verified | 8/8 graded on latest main |
| Slurm, two H100s | LFM / Harbor | Two updates and checkpoints verified | 8/8 graded on latest main |
| Slurm, two H100s | Qwen / Harbor | Two updates and checkpoints verified | 8/8 graded on latest main |
| HF Jobs, two H200s | Qwen / whitebox | Two updates; persistent Trackio present | Reload timed out during bucket weight loading; sequential-loading retry submitted |
| HF Jobs, two H200s | Qwen / native OpenCode | Earlier attempt saved both checkpoints; replacement queued | Pending |
| HF Jobs, two H200s | LFM / whitebox, native OpenCode, Harbor; Qwen / Harbor | Queued for hardware | Pending |

The native OpenCode reloads were separate retries against their saved checkpoints; they were not single uninterrupted smoke invocations. Their `checkpoint-reload-verification.json` records both source snapshots. The whitebox and Harbor checks completed through `jobs/smoke.py`.

The local two-update groups had no reward contrast and zero gradient. An earlier HF native Qwen attempt had reward 0.2574 and gradient norm 3.672 at update 2, but failed its logging-location assertion after training. It is not counted as a completed smoke. No claim of reward improvement follows from these checks.

## Issues caught and corrected

- **Packed LFM convolutions:** the default kernel path ignored sequence boundaries. The scripts explicitly enable the compatible version-2 kernel on the GPU and check packed logits before training.
- **Qwen whitebox context:** the current GRPO tool loop reads the outer config's context limit. The script copies it from Qwen's text config using public model configuration.
- **Trackio persistence:** set `TRACKIO_DIR` before importing TRL/Trackio. Accept SQLite or append-only JSONL storage; FSx uses the latter. Replacement jobs keep logs beside their checkpoints.
- **Harbor routing:** register task metadata routes outside Harbor's root app, while preserving its lifespan, UI and capture routes.
- **Reward selection:** the pinned tasks include JSON `correctness` and scalar `reward` outputs. OpenEnv's public `correctness,reward` fallback reads both, so valid grades are not discarded.
- **HF checkpoint reload:** both DP workers stalled in memory-mapped bucket weight loading. Checkpoint serving now requests vLLM's public `eager` loader. A local two-replica reload loaded each weight shard in about seven seconds and graded 2/2 tasks. The HF retry is queued; its saved checkpoints are retained.
- **Daytona archive ownership:** cluster file owners may be outside the sandbox's user namespace. Task preparation sets tar to retain the sandbox owner. The grader and questions are unchanged; installed libraries are not patched.

## Evidence and remaining checks

Local evidence is under `experiments/tutorial-gpu-smoke-20261001/`: immutable `source-v*` snapshots, `local/` checkpoints and evaluation results, CPU UI checks, the packed-kernel probe and the real Daytona upload check. Repository-local test logs are in ignored `runs/qualification-20261001/`.

HF outputs use bucket `FineEnvs/data-agent-daytona-artifacts`. Qwen whitebox checkpoints are in `tutorial-main-smoke-qwen-whitebox-v4`; their reload retry is `tutorial-main-eval-qwen-whitebox-v8`. Other refreshed source snapshots use `tutorial-main-smoke-lfm-whitebox-v8`, `tutorial-main-smoke-{lfm,qwen}-opencode-v8`, and `tutorial-main-smoke-{lfm,qwen}-multi_harness-v8`. Each smoke keeps `train/`, `reload-eval/` and, only after success, `smoke.json`.

Before a long run, finish the outstanding HF GPU/reload checks, repeat against TRL main after merge, and verify reward contrast over a larger sample. The Docker recipe was reviewed and the CPU server installation was exercised separately; a new public Space deployment has not been qualified by this rewrite.

Native `opencode_env` remains deprecated upstream. Its timeout is not Harbor's strict turn cap. Upstream async packing may drop rows above 40,960 tokens and split one rollout into several rows. This tutorial does not reproduce the archive's exact committed-group resume or whole-rollout weighting.

Earlier code remains in git history and in ignored `temp/pre-tutorial-rewrite/`. Credentials, prepared tasks and run artifacts are excluded from the PR.
