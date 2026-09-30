**Current status (2026-09-15 20:41 UTC):** Native local baseline80514 completed at21/250 (8.4% pass@1). Optimizer80549 and independent checkpoint eval80565 passed. Main native training is prepared for hopper-prod with save50/eval100; no long comparison run has been submitted. See [COMPARISON_READINESS.md](COMPARISON_READINESS.md). The HF ramp descriptions below are historical, superseded by the completed fixed50 local cohort.

# Standalone OpenCode comparison

This is `04-data-agent/envs/blackbox-opencode` (`data_agent_env`), where OpenCode owns the rollout loop. It uses the shared OpenEnv capture contract and direct sandbox adapters. Harbor does not launch these rollouts. Reading the same frozen task files is data reuse, not Harbor execution.

The retained Harbor application is [Blackbox Harbor](https://huggingface.co/spaces/HuggingEnvs/data-agent-blackbox-harbor-env). The new application is [Blackbox OpenCode](https://huggingface.co/spaces/HuggingEnvs/data-agent-blackbox-opencode-env). [SETA Whitebox](https://huggingface.co/spaces/HuggingEnvs/data-agent-seta-whitebox-env) remains separate. Each application serves its own train and test sets through one environment endpoint.

| Setting | Standalone baseline |
| --- | --- |
| Model | Qwen/Qwen3.5-2B, revision15852e8c16360a2fea060d615a32b45270f8a8fc |
| Harness | OpenCode1.18.31, original standalone rollout loop |
| Sandbox backends | Daytona and HF, separate cohorts; E2B adapter retained |
| Test set |250fixed tasks:33easy,118medium,99hard |
| Manifest SHA256 |38943d89f5bb0fec79db8c7a2680c4a8353cf5c53c6eabffaaef8971cb2eb964 |
| Metric | pass@1: fully correct submissions /250; shaped training reward reported separately |
| Inference | HF Job A10080GB, TP1/DP1, BF16, context131072, processed logprobs and engine token IDs |
| Sampling | temperature0.8, top_p1.0, top_k−1, thinking disabled |
| Limits |17model calls,600seconds agent time,4096output tokens/call |
| Admission |100rollouts per standalone deployment; Daytona ramps8→32→100, HF8→16→32 |
| Start gate |2real tasks on each backend, with TiTO audit, before fresh full cohorts |
| Retries | Only ungraded infrastructure attempts; first graded result including0 is immutable |
| Artifacts | Separate backend results, private captures, difficulty scores and scalability measurements |
| Training | Held until baseline and real optimizer/save/resume validation |

Native pass@1 tests the original standalone implementation on each backend. Four-harness checkpoint comparisons still use the retained Harbor evaluation service; these are distinct measurements.

Original deployment bundle:02781ae29b4a2c05d8c9613bf001810946ab29c342da527ff87494251cd64fd7. Private reproduction revision:0a38ef6849fea5e7cad8e886e748321b3cb76060. [Baseline HF Job6aa985baf76d6a098a70e01a](https://huggingface.co/jobs/HuggingEnvs/6aa985baf76d6a098a70e01a), owner`eval-opencode-1789494714`, submitted2026-09-15. See STATUS.md for execution results; submission is not a completed baseline.

Validated before submission: real Daytona/HF sandbox file, exec, temporary environment, background exit-code and deletion checks; four token/task regression checks; native server import, public task API, authenticated execution and browser render. Native packaging explicitly includes source files only, excluding experiment logs and model checkpoints.

Relevant code: `envs/blackbox-opencode/sandbox/daytona.py`, `server/capture.py`, `server/rollout.py`, `harness.py`, `tasks.py`; deployment and baseline entry points `hf/runtime/opencode_space.py` and `hf/runtime/eval_opencode.py`.

## Daytona100 replacement

The user requested100concurrent Daytona containers. The original Job `6aa985baf76d6a098a70e01a` had both limits fixed at process startup and was canceled to start a fresh cohort at the new capacity. Its last downloaded partial ledger contains29graded tasks,5correct, TiTO valid and zero ungraded attempts at that snapshot. These partial results stay archived and are not imported into the replacement cohort. Seven remaining owned Daytona sandboxes were deleted, and zero remaining was verified.

The replacement uses bundle3dbab07f0abad40dadd556262cbfa5d19730d02c0a3c5e0c2b3f8cf8aba58c3e, reproduction revision9f78cac05ca61e99483ab7030040856e966868c8. Space admission100, Daytona eval ceiling100, HF eval ceiling32. Backend limits are explicit Job parameters and reported in each backend's configuration/progress artifacts. The recipe, model, tasks and pass@1 definition are unchanged. Before the swap, the Daytona API verified EU quota250vCPU/500GiB/2000GiB;100sandboxes consume100vCPU/400GiB/500GiB. Re-budget joint capacity before starting other Daytona trainers/evals.

Replacement [HF Job `6aa98b715527934177ee5e73`](https://huggingface.co/jobs/HuggingEnvs/6aa98b715527934177ee5e73), owner `eval-opencode-1789496177`, was submitted at 18:16 UTC and is RUNNING. The live Space reports the replacement bundle and 100-slot cap. Successful admission, TiTO and throughput at the 100-way stage remain to be measured. Evidence is under `experiments/daytona_harness_comparison/logs/hf-20260915/standalone-c100/`.
