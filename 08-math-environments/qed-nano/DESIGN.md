# QED-Nano proofs as Harbor tasks: design

Status: tasks, verifier and training recipe built; a local end-to-end run (our harness, Docker sandboxes, separate
verifier) works up to the judge call. This note records what QED-Nano released, how its reward works, and how we
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
be evaluated on problems it trained on.

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
| `environment/` | a small Python image (sympy available for checking computations); no network for the agent, since most problems are public AoPS threads |
| collect hook | packs `/workspace/proof.md` (regular file, size-capped) into `/logs/artifacts/` |
| `tests/` | the rubric and `grade.py`: calls gpt-oss-20b through Hugging Face Inference Providers with their v1 prompt and settings, writes `reward.txt` = score / 7 |
| verifier | `environment_mode = "separate"`, network allowed only to the judge's host, `env = { HF_TOKEN = "${HF_TOKEN}" }`; the agent never sees the key or the rubric (the rubric sketches the solution) |
| `solution/` | a proof that Gemini graded 7 in FineProofs-SFT, or the official solution for olympiad-sourced problems; tasks with neither ship without an oracle |
| `task.toml` metadata | source, category, `reward_mean` / `reward_std` from the 128 attempts, so users can filter to QED-Nano's training subset (mean ≤ 0.7, std > 0.08) |

Registry entries: `qed-nano-train` (FineProofs-RL), `qed-nano-test` (FineProofs-RL-test), `imo-proofbench`,
`proofbench`. Eval sets carry the benchmark's own grading guidelines in place of a FineProofs rubric.

### Grader hardening (beyond theirs)

Their parser has gaps that matter once a policy is optimising against it:

- the score is not clamped: a reply with `<score>70</score>` gives reward 10. We clamp to 0–7.
- the proof is pasted into the prompt as is. We put it between random per-call delimiters and strip anything shaped
  like a grader tag (`<score>`, `<points>`, `### INPUT DATA`, …) before grading.
- a proof can carry its own `</think>`. Here the agent writes a file, so the grader reads the file, never a chat
  transcript.
- proofs are capped in length; an over-length proof scores 0.
- if the judge call fails after retries, no reward is written, so Harbor reports an error rather than a 0, as the
  openai-math grader does for a broken setup.

## Training through our own harness

[training/](training/) trains a small model on these tasks with OpenEnv's Harbor integration and TRL's AsyncGRPO,
the path 05-multi-harness-rl built for stock coding agents, but every rollout runs **qed-proof**: a host-side Harbor
agent that writes a proof, grades it the way a strict grader would, and rewrites it. That is the move QED-Nano's best
scaffolds end in, and here RL trains it directly. Every call goes through OpenEnv's capture proxy, the conversation is
append-only (one training row), and the reward is the verifier's score / 7. OpenEnv documents `module:Class`
harnesses but resolves only registered names, so `training/server.py` registers a `qed-proof` seam in the Harbor
server process.

## How we validate it

openai-math could prove each task solvable with the official proof and an exact check. With a judge the equivalent
is statistical:

1. **Grader agreement.** Our `grade.py` on both grader benchmarks should reproduce their ~1.2-point advantage error.
2. **Oracle.** The 7-graded reference proof should score high (≥ 6 of 7); tasks where it doesn't are flagged,
   which also catches bad rubrics.
3. **Controls.** Empty, restated-problem, answer-only and truncated proofs should score 0.
4. **End to end.** Qwen3-4B-Thinking-2507 on a sample of problems should match each problem's recorded
   `reward_mean` within sampling error.
5. **Cheat probes**, in the style of openai-math's `attacks.py`: injected score tags, "ignore the rubric"
   instructions, fake grader output, a proof of a different problem, very long proofs. All must score 0.

## Open questions

- **Licensing.** FineProofs-RL's card says Apache-2.0 in its text but has no `license` field; most problems come
  from AoPS forum posts via `AI-MO/aops`, which has no license; IMOProofBench and ProofBench state none. Plan: the
  tasks credit lm-provers and link every source, and we ask before re-hosting anything on the Hub.
- **Judge cost and noise.** Every trial is one judge call with up to 82k output tokens. Averaging 2–3 calls cuts
  noise at extra cost; to be measured on the grader benchmarks.
- **Oracle coverage.** How many of the 5,227 problems have a 7-graded proof in FineProofs-SFT is not yet known; it
  can be read from two columns without downloading the full ~350 MB.
- **Harbor or in-process.** One sandbox per attempt suits agents with tools and agent evaluation. For QED-Nano-style
  single-turn GRPO (about 1,000 attempts per step) an in-process environment sharing the same `grade.py` would be
  far cheaper; Verifiers or OpenEnv variants can follow.
