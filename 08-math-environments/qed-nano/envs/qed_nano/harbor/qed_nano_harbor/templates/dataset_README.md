---
license: apache-2.0
task_categories:
- text-generation
language:
- en
tags:
- harbor
- rl-environment
- reinforcement-learning
- math
- olympiad
- proof
- llm-judge
- qed-nano
pretty_name: "QED-Nano Envs"
size_categories:
- 1K<n<10K
---

# QED-Nano Envs

> **Experimental.** The reward is an LLM judge, not a proof checker: it can be wrong, and it is noisy (regrading
> the same proof moves its score by about 1 point out of 7). Read any proof before trusting its score.

The olympiad proof problems [QED-Nano](https://huggingface.co/spaces/lm-provers/qed-nano-blogpost) was trained on,
as [Harbor](https://github.com/harbor-framework/harbor) environments. An agent writes a proof in plain language; a
separate sandbox grades it 0-7 against the problem's marking scheme with the judge QED-Nano trained with, and the
reward is the score / 7. Partial credit counts: a proof that gets the key lemma right but not the whole argument
earns the lemma's points.

| Set | Tasks | With a reference proof | Validated (reference proof ≥ 6/7) |
|---|---:|---:|---:|
| `train` | $n_train | $n_train_oracle | $n_train_validated |
| `test` | $n_test | $n_test_oracle | $n_test_validated |

`test` is QED-Nano's held-out set (`FineProofs-RL-test`). Its 128 problems are also in `FineProofs-RL`, so `train`
leaves them out.

## How a task works

The agent gets a sandbox with Python (sympy included, for checking computations), no internet, and an olympiad
problem. It writes its proof to `/workspace/proof.md`. That one file is then copied to a fresh sandbox that never ran
the agent's code, where `tests/grade.py`:

1. sends the problem, the problem's marking scheme and the proof to the judge, using
   [QED-Nano's grader prompt](https://github.com/CMU-AIRe/QED-Nano/blob/main/training/prompts/evaluator_prompts/v1.md);
2. reads the judge's score (0-7) and writes `reward.txt` = score / 7.

The marking scheme (written for each problem by Gemini 3 Pro in QED-Nano's release) lists the checkpoints worth
points, the mistakes worth zero and the deductions. The agent never sees it, or the judge's key.

Because a policy will be trained against the judge, the grader adds safeguards QED-Nano's own did not have: the
proof sits between random per-call markers and the judge is told everything between them is the proof; anything in
the proof shaped like the judge's own verdict tags is neutralised; the score is the judge's last verdict, clamped
to 0-7; and if the judge cannot be reached, no reward is written, so Harbor reports an error rather than a 0.

## The judge

Any model behind one of these APIs, set in the environment you run `harbor run` from:

| `JUDGE_PROVIDER` | Endpoint | Default `JUDGE_MODEL` |
|---|---|---|
| `hf` (default) | Hugging Face Inference Providers | `openai/gpt-oss-20b:deepinfra`, QED-Nano's training judge |
| `openai` | OpenAI | set your own |
| `anthropic` | Anthropic Messages API | set your own |
| `vllm` | any OpenAI-compatible server, e.g. your own vLLM (`JUDGE_BASE_URL`) | set your own |

`JUDGE_API_KEY` is required. Optional: `JUDGE_MODEL`, `JUDGE_BASE_URL`, `JUDGE_REASONING_EFFORT` (medium),
`JUDGE_MAX_TOKENS`, `JUDGE_EXTRA_BODY` (JSON added to each request) and `JUDGE_SAMPLES` (average several grades to
cut noise). A different judge gives a different reward; the numbers below are for gpt-oss.

## Run it with the Harbor CLI

You need Harbor 0.25+ and a sandbox backend (Daytona below; Docker works too).

```bash
pip install "harbor[daytona]>=0.25"
export JUDGE_API_KEY=hf_...            # an HF token that can call Inference Providers
export DAYTONA_API_KEY=...

hf download FineEnvs/qed-nano-envs --repo-type dataset --local-dir qed-nano-envs
cd qed-nano-envs

# Any agent Harbor supports, on a few test problems:
harbor run -p test/tasks -e daytona -a claude-code -m anthropic/claude-sonnet-5-5 -l 8 -n 4

# The oracle: submit each task's reference proof and grade it.
harbor run -p test/tasks -e daytona -a oracle -n 8
```

Harbor's registry is in `registry.json`: `qed-nano-envs-train`, `qed-nano-envs-test` and their `-validated`
subsets. `manifest.parquet` has one row per task (source, category, QED-Nano's difficulty, oracle score) for
picking subsets.

### Our harness: write, grade yourself, rewrite

`harness/` is a Harbor agent of our own, **qed-proof**. The model writes a proof, grades it the way a strict grader
would, then writes the corrected proof, which is what QED-Nano's best test-time scaffolds end in. It calls any
OpenAI-compatible endpoint from the host, so the sandbox needs no network:

```bash
pip install -e harness
harbor run -p test/tasks -e daytona -a qed_proof_agent.agent:QedProofAgent \
  -m "Qwen/Qwen3.5-9B:deepinfra" --ak api_base=https://router.huggingface.co/v1 --ak api_key_env=JUDGE_API_KEY \
  --ak stream=true -n 8
```

The same harness runs in RL training through OpenEnv's Harbor integration; the recipe is in
[FineEnvs](https://github.com/adithya-s-k/FineEnvs/tree/main/08-math-environments/qed-nano/training).

## Validation

| Check | Result |
|---|---|
| Judge vs Gemini 3 Pro grades, 120 in-distribution proofs (QED-Nano's graderbench) | advantage error 1.11 points (QED-Nano reports 1.19) |
| Judge vs human grades, 60 MathArena proofs | advantage error 1.36 (QED-Nano reports 1.21-1.43) |
| 18 cheating submissions (restated problem, bare claim, injected score tags, instructions to the grader, a fake verdict, another problem's proof) | all 0; the reference proofs scored 7/7 |
| Oracle over every reference proof, graded by $oracle_judge | mean $oracle_mean/7; $oracle_full of $oracle_graded at 7/7, $oracle_validated at ≥ 6 |
| End to end on Daytona, Harbor 0.25.0 and 0.24.0 (oracle, and Qwen3.5-9B through qed-proof) | every trial graded; Qwen3.5-9B, thinking off, averaged 0.52 on 16 test problems |

The advantage error is QED-Nano's grader metric: the mean absolute difference between the judge's and the
reference's grades after subtracting each problem's mean, since a constant offset does not change GRPO.

A low oracle score flags a task for one of two reasons, and both make it a noisier training signal:

- **the reference proof is wrong**: one DeepSeek-Math-V2 proof misses a solution of the problem that its Gemini
  grade let through, and our judge gives it 0-2;
- **the marking scheme follows one solution path**: several geometry rubrics award points for specific steps
  ("define the auxiliary points S and R", "apply the law of cosines"), and the judge under-credits a correct proof
  that takes another route, such as coordinates, although its prompt asks it to map other methods onto the
  checkpoints.

The `-validated` registries keep the tasks whose reference proof scored at least 6/7, where the scheme and a
correct proof agree; they are the safer default for training.

## What's included

- `train/tasks/`, `test/tasks/`: one Harbor task per problem (instruction, task.toml, sandbox image, verifier, and
  the oracle when the task has a reference proof)
- `registry.json`, `manifest.parquet`
- `generator/`: the code that builds every task from QED-Nano's release, the judge checks and the cheat probes
- `harness/`: the qed-proof agent
- `LICENSE`, `NOTICE`, `LICENSE-QED-Nano`

## Licensing and credits

- **Problems and marking schemes** are from [lm-provers/FineProofs-RL](https://huggingface.co/datasets/lm-provers/FineProofs-RL)
  (its card states Apache-2.0), curated by the LM-Provers team (CMU, Hugging Face, ETH Zurich, Numina) from
  [AI-MO/olympiads](https://huggingface.co/datasets/AI-MO/olympiads) and Art of Problem Solving forum posts via
  [AI-MO/aops](https://huggingface.co/datasets/AI-MO/aops). AoPS problems remain their authors'.
- **Reference proofs** are DeepSeek-Math-V2 proofs from
  [lm-provers/FineProofs-SFT](https://huggingface.co/datasets/lm-provers/FineProofs-SFT) (Apache-2.0), with the
  model's appended self-evaluation removed.
- **The judge's prompt** is from [CMU-AIRe/QED-Nano](https://github.com/CMU-AIRe/QED-Nano) (Apache-2.0; see
  `LICENSE-QED-Nano`).
- **The generator, verifier and harness** are Apache-2.0 (`LICENSE`).
- If you use these environments, cite QED-Nano:

```bibtex
@misc{qednano2026,
  title  = {QED-Nano: Teaching a Tiny Model to Prove Hard Theorems},
  author = {LM-Provers and Yuxiao Qu and Amrith Setlur and Jasper Dekoninck and Edward Beeching and Jia Li and Ian Wu and Lewis Tunstall and Aviral Kumar},
  year   = {2026},
  howpublished = {\url{https://huggingface.co/spaces/lm-provers/qed-nano-blogpost}}
}
```
