# Data Agent comparisons on Hugging Face

Three distinct environment Spaces serve both training and evaluation. Their apps are public on CPU Upgrade; protected visibility keeps source and grading material private.

| Space | Implementation | Baseline |
| --- | --- | --- |
| [Blackbox Harbor](https://huggingface.co/spaces/HuggingEnvs/data-agent-blackbox-harbor-env) | OpenEnv Harbor adapter; historical CLI arm `blackbox` | Four harnesses ×250tasks |
| [Blackbox OpenCode](https://huggingface.co/spaces/HuggingEnvs/data-agent-blackbox-opencode-env) | Original `envs/blackbox-opencode` / `data_agent_env`; CLI arm `opencode` | Native OpenCode ×250tasks, separately on Daytona and HF |
| [SETA Whitebox](https://huggingface.co/spaces/HuggingEnvs/data-agent-seta-whitebox-env) | Native bash/SETA; CLI arm `whitebox` |250tasks; completed47/250=18.8% pass@1 on HF |

**The Harbor arm is not the standalone OpenCode implementation.** No historical score is relabelled. The standalone adapter supports Daytona, HF and E2B; the requested baseline uses Daytona and HF. Token IDs, logprobs and authoritative loss masks are preserved. Pass@1 counts fully correct submissions, separately from the native shaped training reward.

Trackio belongs to training Jobs. Environment Spaces contain no Trackio dashboard. Captures and checkpoints persist in a private artifact bucket. Automatic training launches are held until the corrected standalone baseline and optimizer smoke are validated. The older two-arm setup pipeline below targets Harbor/SETA and must not be used to launch the standalone comparison.

Run from the workspace root, using an existing Python environment with
`huggingface_hub`, `python-dotenv` and Trackio 0.33.0. The submitted Job creates
its own two hash-locked environments; local environments are never modified.

```bash
python HuggingEnvs/04-data-agent/hf/build.py \
  --out experiments/daytona_harness_comparison/logs/hf-20260915/bundle

python HuggingEnvs/04-data-agent/hf/deploy.py upload --env-file experiments/.env
python HuggingEnvs/04-data-agent/hf/deploy.py spaces --env-file experiments/.env
python HuggingEnvs/04-data-agent/hf/deploy.py job --role preflight --env-file experiments/.env

python HuggingEnvs/04-data-agent/hf/deploy.py job --role eval --arm blackbox \
  --phase smoke --flavor a100-large --env-file experiments/.env
python HuggingEnvs/04-data-agent/hf/deploy.py job --role eval --arm whitebox \
  --phase smoke --flavor a100-large --env-file experiments/.env

python HuggingEnvs/04-data-agent/hf/deploy.py status --env-file experiments/.env
```

Use `build.py --refresh-runtime` after editing this directory to avoid copying
the frozen task tree again. Upload and deploy the resulting bundle together:
Jobs reject a Space serving a different bundle hash. Wait for `/deployment` to
report the new hash before submitting GPU work. Never update an environment
Space while its trainer/evaluator is using it.

The credential file is read only by the launcher. Its explicitly selected
`HF_API_KEY` becomes the Job/Space secret `HF_TOKEN`; credentials are never
packaged into the source archive or launch metadata.

Actual integration checks:

- `runtime/check_task_schedule.py`: all 1,250 task identities, instructions and
  both training schedules match the native Harbor and whitebox catalogs.
- `runtime/space_smoke.py`: private task discovery and eight real Daytona
  bash/SETA/verifier cases, with correct and incorrect submissions.
- `runtime/logging_sync.py`: training metrics enter native Trackio storage and
  SQLite backups, persisted by the asynchronous artifact uploader.
- `tests/ui_smoke.py`: two real browser sessions use the same file path in separate
  train/test sandboxes, receive their own content, grade and release their resources.
- `tests/test_shared_service.py`: concurrent capture requests preserve split-specific
  output limits; eval saturation cannot consume reserved training slots.
- `tests/test_auth_bridge.py`: real HTTP streaming and WebSocket forwarding.

`--phase ramp` measures 8, 32 and the quota-bounded arm concurrency; `--phase baseline` uses the
fixed pass@1 test set. These must pass before production training. The existing
Slurm baselines are retained as historical evidence, not relabeled HF results.

Long training requires a completed HF baseline with the same task/sampling/harness
protocol and a successful optimizer/save/remote-resume smoke on the exact runtime bundle.
The launcher starts an independent CPU checkpoint coordinator with each long training job.
It admits evaluations only after the complete checkpoint manifest is uploaded, uses
one separate A100 eval Job per arm, and verifies model hashes before serving them.

The two-arm setup pipeline waits for both baselines, updates idle Spaces, tests the
train/test UI, runs both GPU optimizer smokes and starts both long runs. It monitors
startup every minute and training every ten minutes after each arm reaches ten
optimizer steps. Failed gates are recorded as `needs_attention`; the pipeline does
not silently allocate duplicate jobs or treat a failed eval as a score.

```bash
python HuggingEnvs/04-data-agent/hf/deploy.py job --role coordinator --phase setup \
  --flavor cpu-upgrade --timeout 36h --env-file experiments/.env \
  --baseline-job-map '{"blackbox":"COMPLETED_OR_RUNNING_BASELINE_JOB","whitebox":"COMPLETED_OR_RUNNING_BASELINE_JOB"}'

python HuggingEnvs/04-data-agent/hf/status.py HF_JOB_ID \
  --env-file experiments/.env \
  --out experiments/daytona_harness_comparison/logs/hf-20260915
```

`--phase setup` may start while the baselines run. It leaves the environment Spaces
and training GPUs alone until both baselines pass. Update/upload the runtime bundle
first, then submit the pipeline. A long run can also be launched directly with
`--role train --phase long --flavor h200x2 --timeout 24h --baseline-job ID --smoke-job ID`.

Training Trackio replay runs every 30 seconds in a separate process, deduplicates
events and backs up SQLite. Completed checkpoint eval scores join the training run's
metrics while it is alive. Evaluations finishing after the trainer exits remain in
the coordinator's canonical score artifacts for later replay.

WebSocket keepalive is enabled between HF Jobs and Spaces. The regression test
uses a real TCP proxy that drops idle sockets: a delayed result fails with keepalive
disabled and survives with it enabled.

Shared-server capacity and budgets:

- Native transport ceiling: 1,024 sessions per Space. This is distinct from active sandboxes.
- Verified Daytona EU quota on 2026-09-15: 250 vCPU, 500 GiB RAM, 2,000 GiB disk.
  Each frozen task requests 1 vCPU, 4 GiB RAM and 5 GiB disk: RAM bounds the joint pool to 125.
- Blackbox admits 64 sandboxes, reserving 16 for training; whitebox admits 61,
  reserving 8. Parallel eval limits are 48 and 53 respectively. Update the deployment
  config only after verifying a larger provider quota.
- Blackbox capture uses per-session dataset metadata: train gets a 16,384-token
  per-call ceiling and test gets 4,096. Separate inference endpoints remain per rollout.
- Whitebox generation occurs in the training/eval Job; the shared Space supplies
  tools and grading, preserving the caller's native generation budgets.
- The four superseded train/eval Spaces and separate Daytona Trackio Space were
  deleted with explicit user approval on 2026-09-15. Their repository snapshots
  are backed up locally; task data, checkpoints and evaluation artifacts remain
  in their separate repositories/Buckets. Only the two shared environment Spaces
  above are deployment targets. Do not pause the original multi-harness service.

## Standalone OpenCode baseline

Use a separate output directory to preserve the historical Harbor bundle and Jobs:

```bash
python HuggingEnvs/04-data-agent/hf/deploy.py spaces --only opencode --env-file experiments/.env --out PATH_TO_STANDALONE_BUNDLE_PARENT
python HuggingEnvs/04-data-agent/hf/deploy.py job --role eval --arm opencode --phase baseline --flavor a100-large --timeout 4h --env-file experiments/.env --out PATH_TO_STANDALONE_BUNDLE_PARENT
```

Wait for the Space's `/deployment` SHA to match the uploaded bundle before submitting. The Job serves Qwen3.5-2B with exact token/logprob capture, runs two real smoke tasks on each backend, then runs fresh250-task pass@1 cohorts on Daytona and HF. Daytona ramps8→32→100and HF8→16→32concurrency; a coverage or TiTO failure blocks progression. Previously graded cells, including zeros, are immutable within each cohort. Only ungraded infrastructure attempts are retried. `daytona/scores.json` and `hf/scores.json` report progress and difficulty scores; `canonical_scores.json` records completed backend cohorts.

The evaluator can also use an existing endpoint through `runtime/eval_opencode.py --server OPENENV_URL --vllm-url VLLM_URL --api-key-env HF_TOKEN --backends daytona,hf --daytona-concurrency 100 --hf-concurrency 32 --out OUTPUT`. An authenticated local bridge is required when the environment's native API is protected.
