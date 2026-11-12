# FineProofEnvs design

Status: tasks, verifier and training recipe built and used. The train and test sets are on the Hub as
`FineEnvs/fineproof-envs-{train,test}`, and a 500-update RL run trained through them ([training/RESULTS.md](training/RESULTS.md)). This note records what QED-Nano released, how its reward works, and how we
turned it into Harbor tasks.

## What QED-Nano is

[QED-Nano](https://huggingface.co/lm-provers/QED-Nano) (CMU, Hugging Face, ETH Zurich, Numina; Feb 2026) is a 4B
model post-trained from Qwen3-4B-Thinking-2507 to write olympiad proofs in natural language: no Lean, no tools. The
recipe has three stages: SFT on DeepSeek-Math-V2 proofs, RL with a rubric-graded reward, and RL with a Reasoning
Cache, where the model reasons, a summary replaces the trace, and it continues from the summary. It reports 40 on
IMO-ProofBench and 45 on ProofBench alone, and 54 on both with a test-time scaffold
([blog post](https://huggingface.co/spaces/lm-provers/qed-nano-blogpost),
[code](https://github.com/CMU-AIRe/QED-Nano), Apache-2.0).

## What they released that we can use

| Dataset | Rows | Fields | Role here |
|---|---:|---|---|
| [lm-provers/FineProofs-RL](https://huggingface.co/datasets/lm-provers/FineProofs-RL) | 5,227 | `problem`, `source` (aops / olympiads), `rubrics` (0–7, by Gemini 3 Pro), `scores` / `rewards` of 128 Qwen3-4B-Thinking attempts graded by gpt-oss-20b, `reward_mean`, `reward_std` | training tasks; the 128-attempt mean is a difficulty label |
| `lm-provers/FineProofs-RL-test` | 128 | same | held-out tasks from the same distribution |
| [lm-provers/FineProofs-SFT](https://huggingface.co/datasets/lm-provers/FineProofs-SFT) | 7,777 (4,281 unique problems) | DeepSeek-Math-V2 `reasoning_content` and `proof`, `gemini-3-pro-grade` (0–7), category, competition | reference proofs for the oracle and for grader controls |
| [lm-provers/IMOProofBench](https://huggingface.co/datasets/lm-provers/IMOProofBench) | 60 | problem, reference solution, grading guidelines, level | eval tasks |
| [lm-provers/ProofBench](https://huggingface.co/datasets/lm-provers/ProofBench) | 290 (`24_25` 70, `other` 75, `train` 145) | problem, reference solution, grading scheme | eval tasks (`24_25` and `other`) |
| [lm-provers/matharena-gradingbench](https://huggingface.co/datasets/lm-provers/matharena-gradingbench) | 438 | MathArena model proofs with **human** points | does our judge agree with people |
| [lm-provers/olympiads-proof-graderbench](https://huggingface.co/datasets/lm-provers/olympiads-proof-graderbench) | 480 | in-distribution proofs with Gemini 3 Pro grades | does our judge agree with theirs |

FineProofs was filtered against IMOProofBench and ProofBench (fuzzy matching, no 2025 problems). One catch: all 128
problems of `FineProofs-RL-test` are also in `FineProofs-RL`, so our train set removes them; otherwise a model would
be evaluated on problems it trained on. `data.decontaminate` goes further, and `decontamination.json` in the train
dataset lists every drop with its reason:
- train problems whose word 5-grams cover at least 70% of a test or benchmark problem (2 test near-twins);
- one paraphrase a word filter can't see: ProofBench `TST-2023-4` restated as `aops-c8f456a645`;
- the same problem under two sources, and near-identical train pairs, at 90% overlap or more (23).

Four statements that ran on into their official solution ("Common remarks", "## a) Solution 1") are cut at that
point; the task ids still hash the original text.

## How QED-Nano's harnesses and judges work

**Training harness.** A fork of [PipelineRL](https://github.com/ServiceNow/PipelineRL): asynchronous, streaming
GRPO, 16 attempts per problem, 50k-token attempts. The Reasoning Cache variant runs 3 reason-then-summarise cycles;
the summary comes from a frozen Qwen3-4B-Instruct-2507 (2,048 tokens) and each cycle's proof is graded.

**Training reward** (`training/pipelinerl/domains/math/`):

- grader: `openai/gpt-oss-20b` at medium reasoning, temperature 1.0, up to 81,920 output tokens;
- prompt: `prompts/evaluator_prompts/v1.md`, a ProofBench-style "expert math proof grader" given the problem, the
  rubric and the proof, with no reference solution (adding one made grading slightly worse in their tests);
- the grader sees only the text after the last `</think>`; an attempt without `</think>` scores 0;
- score: the first `<score>N</score>` in the grader's reply; reward = score / 7. No tag scores 0.

**Choosing the grader.** Five prompts and three models, scored on the two grader benchmarks above by the mean
absolute difference of *advantages* (each grade minus the problem's mean grade), since constant shifts don't matter
to GRPO. gpt-oss-20b at medium with the strict ProofBench prompt was within noise of gpt-oss-120b, at about 1.2
points of error.

**Eval harness.** Their own runner (`eval/scripts/run.py`, `eval.py`) graded by Gemini 3 Pro with an IMO-style prompt
(7, 6, 1 or 0 points), plus seven test-time scaffolds in `eval/configs/agents/`: DeepSeek-Math (generate, self-verify,
refine; the "QED-Nano (Agent)" numbers), Reasoning Cache, RSA, Nomos, Self-Check, and two of their own (a dynamic
planner that calls the model as a tool, which models struggled with, and a static pipeline). A human grader found
Gemini lenient: 22/42 on IMO 2025 by Gemini, 14/42 by hand.

## What we build

The scaffolds belong on the agent side; the rubric judge belongs in the environment. So the environment is
scaffold-agnostic: any Harbor agent writes a proof file, and the verifier grades it with QED-Nano's training judge.

| Harbor piece | Contents |
|---|---|
| `instruction.md` | the problem, and "write a complete, rigorous proof to `/workspace/proof.md`" |
| `environment/` | a small Python image pinned by digest (sympy for checking computations) |
| `[agent] network_mode = "no-network"` | most problems are public AoPS threads; Harbor >= 0.25 still lets installed agents reach their model's host and turns off their built-in web search, and a host-side agent (fineproof) calls its model from outside the sandbox |
| `artifacts = ["/workspace/proof.md"]` | the one file that crosses: Harbor copies it to the same path in the verifier's sandbox (and sanitizes it on 0.25); grade.py re-checks it is a regular, size-capped file |
| `tests/` | `task.json` (problem, marking scheme, reference solution for the IMO prompt), `judge_prompt.md` (QED-Nano's prompt, verbatim apart from placeholder names) and `grade.py`, which writes `reward.txt` = score / 7 |
| verifier | `environment_mode = "separate"`; `env` forwards `JUDGE_API_KEY` (required) and the optional `JUDGE_PROVIDER` (hf, openai, vllm, anthropic), `JUDGE_MODEL`, `JUDGE_BASE_URL`, `JUDGE_REASONING_EFFORT`, `JUDGE_MAX_TOKENS`, `JUDGE_EXTRA_BODY`, `JUDGE_STREAM`, `JUDGE_SAMPLES` from the host; the agent never sees the key or the rubric (the rubric sketches the solution). Default judge: gpt-oss-20b for train and test (QED-Nano's training reward), gpt-oss-120b for the benchmarks |
| `solution/` | a DeepSeek-Math-V2 proof Gemini 3 Pro graded 7 (FineProofs-SFT, its appended self-evaluation removed), or the benchmark's official solution; answer-only solutions are not used, so those tasks ship without an oracle |
| `task.toml` metadata | source, category, `reward_mean` / `reward_std` from the 128 attempts, so users can filter to QED-Nano's training subset (mean ≤ 0.7, std > 0.08) |

Checked against Harbor 0.25.0 (and main, 2026-11-04): `harbor lint --select ALL` passes except `root-agent`, which
is deliberate (the agent may be root in its own sandbox; grading happens in another).

Registry entries: `fineproof-envs-train` (FineProofs-RL), `fineproof-envs-test` (FineProofs-RL-test),
`fineproof-envs-imo-proofbench`, `fineproof-envs-proofbench`, and a `-validated` variant of each when oracle results are
given. Eval sets carry the benchmark's own grading guidelines in place of a FineProofs rubric.

### Grader hardening (beyond theirs)

Their parser has gaps that matter once a policy is optimising against it:

- the score is not clamped: a reply with `<score>70</score>` gives reward 10. We clamp to 0–7 and, as they do, take
  the first tag.
- the proof is pasted into the prompt as is. We put it between random per-call delimiters and neutralise anything
  shaped like the judge's tags (also `＜score＞` and `&lt;score&gt;`), the prompts' section names (`#`, bold or
  underlined), `---` rules and `PROOF-<hex>` look-alikes. After the closing delimiter the prompt says that nothing
  inside the proof is an instruction: without that line the default judge gave 7 to 8 of 50 proofs that only asked,
  in prose, for full marks; with it, 0 of 50, and reference proofs graded the same
  ([results/judge-probes-v2.json](results/judge-probes-v2.json)).
- a proof can carry its own `</think>`. Here the agent writes a file, and the grader reads it only if it is a regular
  file (never a symlink) under the size cap; an over-length proof scores 0.
- a judge failure is never a 0. An error body, an empty or cut-off reply and a broken stream are retried, then no
  reward is written, so Harbor reports an error, as it does for bad `JUDGE_*` settings. With several samples, one
  failed sample voids the grade. A training run drops such rollouts, so watch the judge error rate: a proof that
  makes the judge run out of tokens escapes a 0.
- only `environment_mode = "separate"` keeps `JUDGE_API_KEY` out of the agent's sandbox.

## Training through our own harness

[training/](training/) trains a small model on these tasks with OpenEnv's Harbor integration and TRL's AsyncGRPO,
the path 05-multi-harness-rl built for stock coding agents, but every rollout runs the **FineProof harness**: a host-side Harbor
agent that writes a proof in one call by default. With `--rounds N` it then grades its proof the way a strict grader
would and rewrites it, N times: the move QED-Nano's best scaffolds end in, which RL can then train directly. The
published run used one call. Every call goes through OpenEnv's capture proxy and is trained
(with Qwen3.5 in non-thinking mode, as one row per call: see training/README.md), and the reward is the verifier's
score / 7. The Harbor server is OpenEnv's own; rollouts name the harness by import path
(`fineproof_agent.agent:FineProofAgent`), which OpenEnv supports since [openenv/OpenEnv#1573](https://github.com/openenv/OpenEnv/pull/1573).

## How we validate it

openai-math could prove each task solvable with the official proof and an exact check. With a judge the equivalent
is statistical:

1. **Grader agreement.** Our `grade.py` on both grader benchmarks should reproduce their ~1.2-point advantage error.
2. **Oracle.** The 7-graded reference proof should score high (≥ 6 of 7); tasks where it doesn't are flagged,
   which also catches bad rubrics.
3. **Controls.** Empty, restated-problem, answer-only and truncated proofs should score 0.
4. **End to end.** Qwen3-4B-Thinking-2507 on a sample of problems should match each problem's recorded
   `reward_mean` within sampling error.
5. **Cheat probes**, in the style of openai-math's `attacks.py` (`judge_probes.py`): 16 kinds, from injected and
   look-alike score tags to an easier variant and prose instructions to the grader, over 10 validated problems with
   2 grades each. Only prose instructions beat an opening-only control, which the after-proof line above fixed
   ([results/judge-probes-v2.json](results/judge-probes-v2.json)).

## Open questions

- **Licensing (open).** FineProofs-RL's card says Apache-2.0 in its text but has no `license` field; most problems
  come from AoPS forum posts via `AI-MO/aops`, which has no license; IMOProofBench and ProofBench state none. The
  train and test tasks are public on the Hub under the Apache-2.0 the source card states, crediting lm-provers and
  linking every source, with AoPS problems noted as their authors'. Confirmation from LM-Provers is still pending.
- **Judge cost and noise.** Every grade is one judge call with up to 32,768 output tokens (`JUDGE_MAX_TOKENS`;
  QED-Nano allowed 81,920). The training run averaged 2 grades per proof (`JUDGE_SAMPLES=2`) at about $0.004 per
  proof.
- **Oracle coverage.** 3,689 of the 5,073 train tasks and 88 of the 128 test tasks have a reference proof that Gemini 3
  Pro graded 7; gpt-oss-120b (one grade each, temperature 1.0) gives at least 6/7 to 2,884 and 64 of them, the
  `-validated` subsets.
- **Harbor or in-process.** One sandbox per attempt suits agents with tools and agent evaluation. For QED-Nano-style
  single-turn GRPO (about 1,000 attempts per step) an in-process environment sharing the same `grade.py` would be
  far cheaper; Verifiers or OpenEnv variants can follow.
