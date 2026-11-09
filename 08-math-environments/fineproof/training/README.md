# Training on FineProofEnvs with the FineProof harness

This trains a policy on the [FineProofEnvs tasks](../envs/fineproof/harbor/) with OpenEnv's Harbor integration and
TRL's AsyncGRPO, the path [05-multi-harness-rl](../../../05-multi-harness-rl/) built for stock coding agents. The
difference is the harness: instead of OpenCode or Claude Code, every rollout runs the **FineProof harness**, a Harbor agent of
our own.

## The harness: write, check, rewrite

QED-Nano's strongest test-time scaffolds end in the same move: the model reads its own proof as a grader would and
writes a better one. The FineProof harness makes that move the episode, so RL trains it directly:

```
system     "write a proof; a strict grader gives credit only for fully justified steps"
user       the task: the problem
assistant  a first proof
user       "grade it step by step, list every gap, then write the corrected proof under ## Final proof"
assistant  the critique and the final proof   ──►  /workspace/proof.md  ──►  verifier: rubric judge, score / 7
```

Every call the harness makes goes through OpenEnv's capture proxy and is trained. With Qwen3.5 in non-thinking mode
the two calls are two training rows, not one. The chat template drops the empty `<think></think>` block from an
assistant turn once a user message follows it, so the second call's prompt re-renders the first proof instead of
extending the first call's tokens. OpenEnv records each call with the engine's own token ids and logprobs, and both
rows get the rollout's reward. The cost is the first proof, repeated as context in the second row. The harness is a
host-side Harbor agent
([`harness/fineproof_agent/agent.py`](harness/fineproof_agent/agent.py)): it calls the policy over an OpenAI chat
endpoint and writes the final proof into the sandbox. The loop itself is
[`harness/fineproof_agent/loop.py`](harness/fineproof_agent/loop.py), and the number of check-and-rewrite rounds is
`FINEPROOF_ROUNDS` (default 1). Every option can be set as a `FINEPROOF_<OPTION>` environment variable, for launchers
that cannot pass agent kwargs.

## How a run is wired

```
GPU 0: vLLM (policy)  ◄── capture proxy ◄── FineProof harness ► Daytona sandbox ──► collect proof.md
GPU 1: AsyncGRPO trainer                       (in the Harbor server)                        │
        ▲                                                                       separate verifier sandbox
        └──── tokens, logprobs, reward ◄── OpenEnv Harbor server ◄── reward.txt ◄── gpt-oss-20b judge (HF Inference Providers)
```

- The Harbor server is OpenEnv's own (`harbor_env.server.app`, or `openenv harbor serve`). Rollouts name the
  harness by import path, `fineproof_agent.agent:FineProofAgent`; OpenEnv sees a host-side agent and hands it the
  capture proxy's loopback URL and the session id as `api_base`/`api_key`. That needs an OpenEnv with
  `module:Class` support (see `jobs/install.sh`).
- [`train.py`](train.py) is 05's `multi_harness.py` with one harness and the judge's fractional reward used as is.
- [`prepare.py`](prepare.py) writes the tasks: by default 1,024 training problems on which QED-Nano's base model
  scored between 0.25 and 0.9, and all 128 FineProofs-RL-test problems for evaluation, which are removed from the
  training pool (they are a subset of it).
- [`evaluate.py`](evaluate.py) scores a base model or a checkpoint on the test problems through the same harness and
  judge: mean rubric reward, share of full-marks proofs, coverage.
- The judge's key reaches each verifier through the task's `[verifier] env`, resolved from the server's environment;
  the agent never sees it. The judge is gpt-oss-20b on HF Inference Providers, with three providers in
  `JUDGE_MODEL` (deepinfra, then groq, then featherless-ai): when one rate-limits, the next grades at once, and each
  grade records which one gave it.

Pinned stack: 05's validated one (TRL `52fb144e`, vLLM 0.25.1, Transformers `d6c1e71b`), with OpenEnv at `main` plus
[openenv/OpenEnv#1573](https://github.com/openenv/OpenEnv/pull/1573) (`0bd1a89d`: custom harnesses, loopback, one
step-limit warning per harness), and Harbor 0.24 for separate verifiers. Models: `Qwen/Qwen3.5-2B`, `-4B` and `-9B`
in non-thinking mode, at pinned revisions. The two-update smoke (`jobs/smoke.py`) passed on HF Jobs (2× H200) and on
Slurm (2× H100).

## Run it

You need an HF token (model, Jobs and the judge on Inference Providers) and an existing bucket for outputs; with
`--sandbox daytona` (the default), also a Daytona key. With `--sandbox hf-sandbox` the task sandboxes are HF Jobs too
(`cpu-basic`, shut down after 10 idle minutes), so the whole run stays on Hugging Face. `prepare.py` then gives each
task its Dockerfile's base image and leaves the agent's network on, which HF Sandbox needs and the FineProof harness
does not mind: it calls the model from outside the sandbox and only writes the proof into it.

```bash
cd 08-math-environments/fineproof/training
python jobs/hf_job.py smoke --name fp-smoke --bucket <owner>/<bucket> --prepare-args "--train-tasks 16" --submit
python jobs/hf_job.py smoke --name fp-smoke-hf --sandbox hf-sandbox --bucket <owner>/<bucket> --submit  # all on HF
python jobs/hf_job.py eval  --name fp-eval-base --bucket <owner>/<bucket> --submit
python jobs/hf_job.py train --name fp-train --bucket <owner>/<bucket> --steps 100 --submit
python jobs/hf_job.py eval  --name fp-eval-step100 --bucket <owner>/<bucket> \
  --checkpoint /outputs/fp-train/checkpoint-100 --step 100 --submit
```

The MVP run puts everything on Hugging Face: HF Sandboxes, a public Trackio Space, and an eval Job per checkpoint
on the fixed benchmark step 0 was scored on ([`results/fineproof-bench-4b-step0.json`](../results/fineproof-bench-4b-step0.json)):

```bash
JUDGE_MODEL=openai/gpt-oss-20b:groq,openai/gpt-oss-20b:featherless-ai \
python jobs/hf_job.py train --name fp-rl-4b --model Qwen/Qwen3.5-4B --bucket <owner>/<bucket> \
  --sandbox hf-sandbox --rounds 0 --max-tokens 16384 --steps 100 --save-steps 20 --timeout 28h \
  --space-id <owner>/fineproof-rl --trackio-bucket <owner>/fineproof-trackio --eval-checkpoints --submit
```

Without `--submit` each command prints the job it would run. On your own two-GPU machine, run `bash jobs/install.sh`
and `python prepare.py`, then `python jobs/run.py {train,eval} ...` with the same options. The GPUs must be A100s,
H100s or H200s: the pinned TRL's AsyncGRPO trains with `kernels-community/flash-attn3`, which is built for those
architectures (sm80, sm90) and refuses a B200.

On a shared cluster whose user ids a Daytona sandbox cannot map (ids above 65535 are common), every rollout fails
before grading with `tar: .: Cannot change ownership to uid …`. Harbor's Daytona backend copies the agent's proof into
the verifier's sandbox as an archive and unpacks it as root, keeping the host's owner
([harbor-framework/harbor#1959](https://github.com/harbor-framework/harbor/issues/1959)). Until
[harbor-framework/harbor#2043](https://github.com/harbor-framework/harbor/pull/2043) is released, add
`--no-same-owner` to the `tar -xzf` command in the installed `harbor/environments/tar_transfer.py`.

## What a run records

In its output folder (copied to the bucket every 5 minutes, when the Job is stopped, and at the end):

| File | What is in it |
|---|---|
| `train.log`, `eval.log` | The trainer's or evaluator's console, with each update's metrics |
| `vllm.log` | The policy engine |
| `harbor.log` | OpenEnv's Harbor server at INFO: budget stops, token clamps, upstream retries |
| `trials/<trial>/` | Harbor's files for each rollout: the conversation, the proof, `result.json` (phase timings, tokens) and `verifier/grading.json` (every judge grade: provider, score, written assessment, tokens, cost) |
| `rollouts.jsonl` | One line per rollout ([`rollout_log.py`](rollout_log.py)): the reward, OpenEnv's result (errors, each model call's finish reason and tokens, phase timings, findings) and the judge's summary |
| `checkpoint_evals.jsonl` | With `--eval-checkpoints`: the eval Job submitted for each checkpoint |
| `trackio/` | The Trackio project, also on the Space when `--space-id` is set |

In Trackio (project `fineproof-rl`), the training run has TRL's metrics for each update (loss, KL, importance ratio,
reward, gradient norm, staleness, throughput), plus `train/rollouts/*` and `train/judge/*` over the rollouts that
finished since the last update: graded share, zero and full-marks shares, truncation, timings, tokens, judge
disagreement and cost. Every graded rollout is a trace (`graded_rollouts`), with spans for the sandbox, each policy
call and each judge grade, and the judge's written assessment. The run also has GPU and CPU metrics, its console
output, and alerts when many rollouts go ungraded, the judge fails, or proofs hit the token cap. Each checkpoint's
eval logs to one run, `<name>-evals`, at the checkpoint's step: `eval/*` (mean reward with its standard error over
problems, full marks, pass@4, best-of-4, learnable share), `eval_rollouts/*`, `eval_judge/*`, and a trace per attempt.
`python evaluate.py --report-only --output <eval folder> ...` charts an eval that has already run.

## Tests

```bash
python -m unittest discover -s harness/tests                       # the loop, against a stub model
python -m unittest discover -s ../envs/fineproof/harbor/tests       # the verifier's grade.py
```
