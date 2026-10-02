# Validation of the tutorial

[Read the article](https://huggingface.co/spaces/FineEnvs/multi-harness-rl) · [Collection](https://huggingface.co/collections/FineEnvs/smoldataenvs-multi-harness-rl-6abdfaaa8d74dacd481d5212) · [Tutorial](README.md)

This is the validation record for the `train/`, `eval/` and `envs/` rewrite. Historical pilots in [RESULTS.md](RESULTS.md) and the [article](../content/articles/multi-harness-rl/) used an earlier implementation.

## Runtime and CPU checks

The final review passed 44 runtime/contract tests, including two new checks that Whitebox and Harbor preflight do not import deprecated native OpenCode. The article build, 105 external article links, 69 tutorial file links, section anchors and desktop/mobile rendering also passed.

- **42 tests passed** against OpenEnv main `7ee88d590d36ae1ac3daf2cc11bc551bca7a804f` on 1 October 2026. Tests cover task identity, grading, cleanup, service routing, tokenizer templates and the typed training contract, including partial and zero loss masks.
- A clean Python 3.12 CI environment passed 12 lightweight tests and skipped 30 runtime tests. Lint, formatting, shell syntax and the repository index also passed.
- A separate CPU installation, with **neither Torch nor TRL installed**, started all three environment servers. Their `/health`, `/web/` and Task API requests succeeded. Whitebox and native OpenCode exposed the 250 test tasks. Harbor's UI check used an empty registry; GPU checks use the prepared task registry.
- Evaluation tests exercise both execution modes, verify routing from all three training modes, retain graded wrong answers, retry only ungraded pairs and reject a different checkpoint in the same output directory. The latest test logs and documentation previews are in `experiments/smoldataenv-docs-20261001/`.
- All 1,000 training and 250 test tasks were prepared with instruction/grader hash checks and no notebook overlap.
- LFM's GPU check produced identical packed and separate-sequence logits, then completed backward. The scripts run this guard before async training.
- A real Harbor/Daytona upload passed from the cluster's user ID with `TAR_OPTIONS=--no-same-owner`.

TRL's typed consumer merged in [PR #6947](https://github.com/huggingface/trl/pull/6947). All 42 runtime/contract tests also passed against merged TRL main `52fb144e7ece97e26aabd3c025740022838e46b5` and OpenEnv main `7ee88d590d36ae1ac3daf2cc11bc551bca7a804f`. Local and HF installers now use both upstream main branches, without a PR revision override. Earlier GPU checks below used the reviewed TRL commit `910138e9392dda75d3a9766e7c494d44f1d82723`; they are not fresh GPU tests of the merge commit.

The fresh GPU environment uses Python 3.12, Torch 2.11.0, vLLM 0.25.1, Transformers main `d6c1e71bd717bf092f8293f0c3c9bd4a5ac5401a`, and Harbor 0.23.0. `dependencies.json` and `packages.txt` record what each installation resolved.

```bash
python -m pytest tests -q             # Lightweight checks
python -m pytest tests -q --contract  # Installed runtime and tokenizer checks
python jobs/smoke.py --mode multi_harness --output runs/harbor-smoke
```

## Standalone Space deployments

Each environment was copied outside this tutorial and started in the CPU-only installation. All three passed health, UI and API checks without importing a sibling environment. The uploaded source files were also compared byte for byte with their local folders.

| Environment | Public UI | Hardware | Task API |
|---|---|---|---|
| SETA whitebox | [Open](https://fineenvs-smoldataenv-multi-harness-whitebox.hf.space/web/) | CPU Basic | 1,000 train / 250 test |
| Native OpenCode | [Open](https://fineenvs-smoldataenv-multi-harness-opencode.hf.space/web/) | CPU Basic | 1,000 train / 250 test |
| Harbor multi-harness | [Open](https://fineenvs-smoldataenv-multi-harness-harbor.hf.space/web/) | CPU Basic | 1,000 train / 250 test |

Browser checks on 1 October 2026 verified the three UIs and their root redirects, with no JavaScript errors. Harbor uses the current OpenEnv task browser, rollout controls and trace UI. Whitebox has a task picker, Bash workspace, execution history and score cards. Native OpenCode uses OpenEnv's playground with environment-specific connection examples. A real whitebox Space session staged task data in Daytona, executed one tool call, graded an intentionally incorrect answer as zero and released the sandbox.

The existing 1,024-session limits are preserved. This is a configuration limit, not a measured throughput claim. Native OpenCode needs a reachable token-capturing inference endpoint before model rollouts; the Space browser check does not establish GPU training readiness. Local evidence is in `experiments/standalone-env-spaces-20261001/`, including source hashes, browser checks and the sandbox probe. The whitebox UI tests cover separate browser sessions, restart and expiry cleanup, failed submissions, unscored attempts and escaped tool output. The deployed browser smoke computed a task answer from its CSV with one command and received correctness `1.0` and reward `1.09375`. A second browser remained isolated, and the 390px mobile layout had no horizontal overflow. Screenshots and logs are in `experiments/whitebox-playground-20261001/`. Previous Space bundles are retained in ignored local archives and the Space's `.archive/` directory.

The three Spaces were renamed to `smoldataenv-multi-harness-*` and grouped in the [SmolDataEnvs Multi-harness RL collection](https://huggingface.co/collections/FineEnvs/smoldataenvs-multi-harness-rl-6abdfaaa8d74dacd481d5212). Their history, public visibility, CPU Basic hardware and concurrency settings were preserved. Checks of the new URLs passed, including the whitebox task above, browser isolation and mobile layout. The merged-main installation, test results and deployment checks are recorded in `experiments/smoldataenv-migration-main-20261001/`.

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

The tables above summarize the checks for readers. The underlying logs are internal maintainer records, not part of this repository. Local evidence is under `experiments/tutorial-gpu-smoke-20261001/`: immutable `source-v*` snapshots, `local/` checkpoints and evaluation results, CPU UI checks, the packed-kernel probe and the real Daytona upload check. Repository-local test logs are in ignored `runs/qualification-20261001/`.

Internal HF outputs use the **private** bucket `FineEnvs/data-agent-daytona-artifacts`. Qwen whitebox checkpoints are in `tutorial-main-smoke-qwen-whitebox-v4`; their reload retry is `tutorial-main-eval-qwen-whitebox-v8`. Other refreshed source snapshots use `tutorial-main-smoke-lfm-whitebox-v8`, `tutorial-main-smoke-{lfm,qwen}-opencode-v8`, and `tutorial-main-smoke-{lfm,qwen}-multi_harness-v8`. Each smoke keeps `train/`, `reload-eval/` and, only after success, `smoke.json`.

Before a long run, finish the outstanding HF GPU/reload checks on merged TRL main and verify reward contrast over a larger sample. The public Space deployments and contract tests above validate the environment and trainer interfaces; the separate HF GPU smoke matrix remains pending hardware.

Native `opencode_env` remains deprecated upstream. Its timeout is not Harbor's strict turn cap. Upstream async packing may drop rows above 40,960 tokens and split one rollout into several rows. This tutorial does not reproduce the archive's exact committed-group resume or whole-rollout weighting.

Earlier code remains in git history and in ignored `temp/pre-tutorial-rewrite/`. Credentials, prepared tasks and run artifacts are excluded from the PR.

The tutorial intentionally installs TRL and OpenEnv from main and records the resolved commits. It is not a dependency lockfile. Native OpenCode can stop building after upstream removal. The standalone sandbox image also uses a mutable tag in a personal Docker Hub repository; moving it to an organization registry with a fixed digest remains a reproducibility follow-up.
