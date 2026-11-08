# Training a small model on QED-Nano proofs, through our own Harbor harness

This trains a policy on the [QED-Nano Harbor tasks](../envs/qed_nano/harbor/) with OpenEnv's Harbor integration and
TRL's AsyncGRPO, the path [05-multi-harness-rl](../../../05-multi-harness-rl/) built for stock coding agents. The
difference is the harness: instead of OpenCode or Claude Code, every rollout runs **qed-proof**, a Harbor agent of
our own.

## The harness: write, check, rewrite

QED-Nano's strongest test-time scaffolds end in the same move: the model reads its own proof as a grader would and
writes a better one. `qed-proof` makes that move the episode, so RL trains it directly:

```
system     "write a proof; a strict grader gives credit only for fully justified steps"
user       the task: the problem
assistant  a first proof
user       "grade it step by step, list every gap, then write the corrected proof under ## Final proof"
assistant  the critique and the final proof   ──►  /workspace/proof.md  ──►  verifier: rubric judge, score / 7
```

The conversation is append-only, so OpenEnv's capture proxy records it as one training row, and every call the
harness makes is trained. The harness is a host-side Harbor agent
([`harness/qed_proof_agent/agent.py`](harness/qed_proof_agent/agent.py)): it calls the policy over an OpenAI chat
endpoint and writes the final proof into the sandbox. The loop itself is
[`harness/qed_proof_agent/loop.py`](harness/qed_proof_agent/loop.py), and the number of check-and-rewrite rounds is
`QED_PROOF_ROUNDS` (default 1). Every option can be set as a `QED_PROOF_<OPTION>` environment variable, for launchers
that cannot pass agent kwargs.

## How a run is wired

```
GPU 0: vLLM (policy)  ◄── capture proxy ◄── qed-proof agent ──► Daytona sandbox ──► collect proof.md
GPU 1: AsyncGRPO trainer                       (in the Harbor server)                        │
        ▲                                                                       separate verifier sandbox
        └──── tokens, logprobs, reward ◄── OpenEnv Harbor server ◄── reward.txt ◄── gpt-oss-20b judge (HF Inference Providers)
```

- The Harbor server is OpenEnv's own (`harbor_env.server.app`, or `openenv harbor serve`). Rollouts name the
  harness by import path, `qed_proof_agent.agent:QedProofAgent`; OpenEnv sees a host-side agent and hands it the
  capture proxy's loopback URL and the session id as `api_base`/`api_key`. That needs an OpenEnv with
  `module:Class` support (see `jobs/install.sh`).
- [`train.py`](train.py) is 05's `multi_harness.py` with one harness and the judge's fractional reward used as is.
- [`prepare.py`](prepare.py) writes the tasks: by default 1,024 training problems on which QED-Nano's base model
  scored between 0.25 and 0.9, and all 128 FineProofs-RL-test problems for evaluation, which are removed from the
  training pool (they are a subset of it).
- [`evaluate.py`](evaluate.py) scores a base model or a checkpoint on the test problems through the same harness and
  judge: mean rubric reward, share of full-marks proofs, coverage.
- The judge's key reaches each verifier through the task's `[verifier] env`, resolved from the server's environment;
  the agent never sees it.

Pinned stack: 05's validated one (TRL `52fb144e`, vLLM 0.25.1, Transformers `d6c1e71b`), with OpenEnv at `main` plus
[openenv/OpenEnv#1573](https://github.com/openenv/OpenEnv/pull/1573) (`6fb632a4`, custom harnesses and loopback; the
training path on it is not yet re-tested with vLLM), and
Harbor 0.24 for separate verifiers. Model: `Qwen/Qwen3.5-2B` in non-thinking mode, at the revision 05 pinned.

## Run it

You need an HF token (model, Jobs and the judge on Inference Providers), a Daytona key, and an existing bucket for
outputs.

```bash
cd 08-math-environments/qed-nano/training
python jobs/hf_job.py smoke --name qed-smoke --bucket <owner>/<bucket> --prepare-args "--train-tasks 16" --submit
python jobs/hf_job.py eval  --name qed-eval-base --bucket <owner>/<bucket> --submit
python jobs/hf_job.py train --name qed-train --bucket <owner>/<bucket> --steps 100 --submit
python jobs/hf_job.py eval  --name qed-eval-step100 --bucket <owner>/<bucket> \
  --checkpoint /outputs/qed-train/checkpoint-100 --step 100 --submit
```

Without `--submit` each command prints the job it would run. On your own two-GPU machine, run `bash jobs/install.sh`
and `python prepare.py`, then `python jobs/run.py {train,eval} ...` with the same options.

## Tests

```bash
python -m unittest discover -s harness/tests                       # the loop, against a stub model
python -m unittest discover -s ../envs/qed_nano/harbor/tests       # the verifier's grade.py
```
