<div align="center">

<h1>QED-Nano proofs</h1>

<h3>Write an olympiad proof in plain language, graded 0–7 by a rubric judge</h3>

<p>QED-Nano's training problems, rubrics and judge as Harbor tasks, and a small model trained on them through our own harness.</p>

<a href="https://github.com/harbor-framework/harbor"><img src="https://img.shields.io/badge/framework-Harbor-3B82F6?style=for-the-badge&labelColor=1a1a1a" alt="Harbor" height="32"></a>
<a href="https://huggingface.co/spaces/lm-provers/qed-nano-blogpost"><img src="https://img.shields.io/badge/%F0%9F%93%84%20Source-QED--Nano%20blog%20post-10B981?style=for-the-badge&labelColor=1a1a1a" alt="The QED-Nano blog post" height="32"></a>

</div>

---

> **Work in progress.** The tasks and the verifier are built and validated: the judge's agreement on QED-Nano's
> grader benchmarks, cheating probes, a sweep grading every reference proof, and end-to-end runs on Daytona with
> Harbor 0.25. The train and test sets are staged as the Hub datasets `FineEnvs/qed-nano-envs-train` and
> `FineEnvs/qed-nano-envs-test` (not uploaded yet). The training recipe is built but has not run.
> [DESIGN.md](DESIGN.md) has the reasoning, [training/](training/) the recipe.

[QED-Nano](https://huggingface.co/lm-provers/QED-Nano) is a 4B model trained by CMU, Hugging Face, ETH Zurich and
Numina to write olympiad proofs. Its reward is an LLM judge: gpt-oss-20b scores each proof 0–7 against a
problem-specific rubric. The team released the problems, the rubrics, the judge's prompt and benchmarks for the
judge itself. This project packages them as [Harbor](https://github.com/harbor-framework/harbor) tasks any agent can
attempt: write a proof, and a separate sandbox grades it with the same judge, hardened against injected grades.

It is the other end of [openai-math](../openai-math/): there a proof must pass the Lean kernel and match the
statement exactly; here a natural-language proof earns partial credit for the steps it gets right.

<!-- BEGIN:matrix -->
| Env | Tools | Backend | `harbor` |
|---|---|---|---|
| **olympiad_proofs** | shell (Python) | `daytona` | ✅ |
<!-- END:matrix -->

## What's here

- [`envs/qed_nano/harbor/`](envs/qed_nano/harbor/): the generator (`python -m qed_nano_harbor.adapter`), the verifier
  (`templates/tests/grade.py`, QED-Nano's judge plus safeguards against injected grades) and its unit tests.
- [`training/`](training/): **qed-proof**, our own Harbor harness (write a proof, grade it, rewrite it), and the
  AsyncGRPO recipe that trains through it with OpenEnv's Harbor integration, as in 05-multi-harness-rl.

## Task sets

| Set | Tasks | From |
|---|---:|---|
| train | 5,099 | [lm-provers/FineProofs-RL](https://huggingface.co/datasets/lm-provers/FineProofs-RL), with QED-Nano's difficulty labels |
| test | 128 | `lm-provers/FineProofs-RL-test`; these 128 are also in FineProofs-RL, so train leaves them out |
| IMO-ProofBench | 60 | [lm-provers/IMOProofBench](https://huggingface.co/datasets/lm-provers/IMOProofBench) |
| ProofBench | 145 | [lm-provers/ProofBench](https://huggingface.co/datasets/lm-provers/ProofBench), splits `24_25` and `other` |

## Through OpenEnv

The tasks and qed-proof run through stock `openenv harbor serve`, the server training uses, with the harness named
by import path (`--harness qed_proof_agent.agent:QedProofAgent`). That needs
[openenv/OpenEnv#1573](https://github.com/openenv/OpenEnv/pull/1573), which makes `module:Class` work and gives a
host-side agent the capture proxy's loopback URL. Before it, a public tunnel cut long proofs with a 502 and the
retries were recorded as extra conversations.

Eval mode, on the 64 validated test tasks: Qwen3.5-9B on deepinfra (thinking off), qed-proof, Daytona, judge
gpt-oss-20b. The two paths agree within noise ([results/openenv-parity-v1.json](results/openenv-parity-v1.json)):

| Path | Graded | Mean reward | Full marks |
|---|---:|---:|---:|
| `openenv harbor serve` + `training/evaluate.py --hosted` | 64/64 | 0.667 | 34 |
| `harbor run` | 64/64 | 0.717 | 40 |

Per task, OpenEnv minus Harbor averages −0.049 with a standard error of 0.046. Every rollout made exactly 2 model
calls on both paths.

Training tier, on the same 64 tasks: Qwen3.5-9B on our own vLLM (one B200, serving token ids and processed
logprobs) with the recipe's training sampling (temperature 0.8, top_p 1). 63 of 64 rollouts were graded; the judge's
provider rate-limited the last one. Mean reward 0.753, 42 full marks
([results/openenv-training-tier-v1.json](results/openenv-training-tier-v1.json)). Every graded rollout gave a valid
`TrainingTrace`, the object TRL's `HarnessRolloutWorker` trains on:
- both calls are trainable;
- there is one logprob per sampled token;
- the prompt is masked out;
- the engine's prompt tokens match a local re-render on all 126 turns.

## Credits

Problems, rubrics, benchmarks and the judge's prompt are from the
[QED-Nano](https://huggingface.co/spaces/lm-provers/qed-nano-blogpost) release by the LM-Provers team (CMU, Hugging
Face, ETH Zurich, Numina; [code](https://github.com/CMU-AIRe/QED-Nano), Apache-2.0). The problems come from
[AI-MO/olympiads](https://huggingface.co/datasets/AI-MO/olympiads) and [AI-MO/aops](https://huggingface.co/datasets/AI-MO/aops)
(Art of Problem Solving). DESIGN.md lists the open licensing questions.
