---
license: apache-2.0
language:
- en
task_categories:
- other
tags:
- rl-environment
- reinforcement-learning
- harbor
- math
- olympiad
- proof
- llm-judge
- fineproof
- qed-nano
pretty_name: "FineProofEnvs ($split)"
size_categories:
- $size_category
viewer: false
---

[![View tasks in Harbor Visualiser](https://img.shields.io/badge/%F0%9F%A4%97%20Harbor%20Visualiser-View%20tasks-FFD21F?style=for-the-badge)](https://huggingface.co/spaces/HuggingFaceH4/harbor-visualiser?dataset=$repo)

# FineProofEnvs: $split (Harbor)

**Write an olympiad proof, graded against a marking scheme.** $n_tasks Harbor tasks from
[`$source_repo`](https://huggingface.co/datasets/$source_repo), $source_desc
[QED-Nano](https://huggingface.co/spaces/lm-provers/qed-nano-blogpost), converted so every one runs as a standard
Harbor task.

The agent gets a problem and a sandbox with Python (no internet), and writes its proof to `/workspace/proof.md`.
A separate sandbox grades the proof 0-7 against the problem's marking scheme with QED-Nano's judge prompt, and the
reward is score / 7, so a partial proof earns partial credit. The reward comes from an LLM judge, not a verified
proof checker.

| | |
|---|---|
| Tasks | $n_tasks ($n_validated in the `-validated` subset) |
| Graded by | an LLM judge, 0-7 against the problem's marking scheme |
| Default judge | `openai/gpt-oss-20b`, QED-Nano's training judge, through HF Inference Providers |
| Reference proofs | $n_oracle, from [`lm-provers/FineProofs-SFT`](https://huggingface.co/datasets/lm-provers/FineProofs-SFT) |
| Source | [`$source_repo`](https://huggingface.co/datasets/$source_repo) at `$source_rev` |
| Adapter | `fineproof_harbor $version` |

Part of a pair: [train](https://huggingface.co/datasets/$org/fineproof-envs-train) ·
[test](https://huggingface.co/datasets/$org/fineproof-envs-test). $overlap_note

Trained on these tasks: [`$org/Qwen3.5-2B-FineProof-RL`](https://huggingface.co/$org/Qwen3.5-2B-FineProof-RL).
That is Qwen3.5-2B after 500 RL updates from the base model. It went from 0.217 to 0.407 mean reward on the 64
validated test tasks. [Training logs](https://huggingface.co/spaces/$org/fineproof-rl-trackio) ·
[every eval attempt](https://huggingface.co/datasets/$org/fineproof-rl-runs) ·
[recipe](https://github.com/adithya-s-k/FineEnvs/tree/main/08-math-environments/fineproof).

## Layout

```text
tasks/<task_id>/
├── task.toml         # provenance, QED-Nano's difficulty, the reference proof's score, judge settings
├── instruction.md    # the problem, and where to write the proof
├── environment/      # Dockerfile: Python 3.12 with sympy, pinned by digest
├── solution/         # the reference proof (Harbor's oracle), when the task has one
└── tests/            # grade.py, the judge prompt and the marking scheme; only the verifier sees them
registry.json         # fineproof-envs-$split and fineproof-envs-$split-validated
manifest.parquet      # one row per task, for filtering without walking the tree
harness/              # the FineProof harness, our Harbor agent
generator/            # the adapter that builds these tasks
```

Example: [task.toml](https://huggingface.co/datasets/$repo/blob/main/tasks/$example/task.toml) ·
[instruction](https://huggingface.co/datasets/$repo/blob/main/tasks/$example/instruction.md) ·
[verifier](https://huggingface.co/datasets/$repo/blob/main/tasks/$example/tests/grade.py).

## Run

```bash
pip install "harbor[daytona]>=0.25"
hf download $repo --repo-type dataset --local-dir fineproof-envs-$split
cd fineproof-envs-$split
export JUDGE_API_KEY=hf_...    # an HF token that can call Inference Providers
export DAYTONA_API_KEY=...
harbor run -p tasks -e daytona -a claude-code -m anthropic/claude-sonnet-5-5 -l 8 -n 4
```

`-a oracle` submits the reference proofs instead. For another judge, set `JUDGE_PROVIDER` (`hf`, `openai`,
`anthropic` or `vllm`), `JUDGE_MODEL`, and `JUDGE_BASE_URL` for vLLM or any OpenAI-compatible server. A different
judge gives a different reward.

`harness/` is our own Harbor agent, the **FineProof harness**: the model writes a proof, grades it the way a strict grader
would, then writes the corrected proof. It calls the model from the host, so the sandbox stays offline:

```bash
pip install -e harness
harbor run -p tasks -e daytona -a fineproof_agent.agent:FineProofAgent -m Qwen/Qwen3.5-9B:deepinfra \
  --ak api_base=https://router.huggingface.co/v1 --ak api_key_env=JUDGE_API_KEY --ak stream=true -n 8
```

## How it was converted

With the `fineproof_harbor` adapter (`generator/`), from QED-Nano's release at pinned revisions:

- Each problem becomes one task. Its marking scheme (written per problem by Gemini 3 Pro in QED-Nano's release)
  goes to `tests/`, where the agent never sees it. QED-Nano's difficulty for the problem, the mean score of its
  128 Qwen3-4B attempts, is kept in `task.toml` as `reward_mean`.
- The reference proof is the DeepSeek-Math-V2 proof from FineProofs-SFT that Gemini 3 Pro graded 7/7, without
  the self-evaluation the model appends. We graded each one once with gpt-oss-120b (high reasoning); the score is
  `oracle_score` in `task.toml`, and the `-validated` subset keeps the tasks whose reference proof scored at
  least 6/7. That is one gpt-oss-120b grade per reference proof, at temperature 1.0, so a borderline proof can land
  on either side.
- Grading uses QED-Nano's judge prompt unchanged. Because a policy trains against it, the verifier also fences the
  proof between random markers, neutralises anything in it shaped like the judge's verdict, and writes no reward
  when the judge can't be reached, so Harbor records an error rather than a 0.

## Validation

- Every task loads with Harbor 0.25.
- The default judge agrees with reference grades as closely as QED-Nano reports for it (QED-Nano's advantage
  error: 1.11 points against Gemini 3 Pro grades, where they report 1.19; 1.36 against MathArena's human grades,
  where they report 1.21-1.43).
- 18 cheating submissions (a restated problem, injected score tags, instructions to the grader, another problem's
  proof, ...) all scored 0.
- The reference proofs here averaged $oracle_mean/7; $n_validated of $n_oracle scored at least 6.
- End to end on Daytona, with the oracle and with Qwen3.5-9B through the FineProof harness: every trial was graded.

## Credits

The problems, marking schemes and reference proofs are from QED-Nano's release by the
[LM-Provers](https://huggingface.co/lm-provers) team (CMU, Hugging Face, ETH Zurich, Numina):
[FineProofs-RL](https://huggingface.co/datasets/lm-provers/FineProofs-RL),
[FineProofs-RL-test](https://huggingface.co/datasets/lm-provers/FineProofs-RL-test) and
[FineProofs-SFT](https://huggingface.co/datasets/lm-provers/FineProofs-SFT). The problems were curated from
[AI-MO/olympiads](https://huggingface.co/datasets/AI-MO/olympiads) and Art of Problem Solving posts via
[AI-MO/aops](https://huggingface.co/datasets/AI-MO/aops). The judge prompt is from
[CMU-AIRe/QED-Nano](https://github.com/CMU-AIRe/QED-Nano). The task format and runner are
[Harbor](https://github.com/harbor-framework/harbor). This conversion is by the FineEnvs team at Hugging Face.

## License

Apache-2.0, as the source datasets (`LICENSE`). The judge prompt is Apache-2.0 from CMU-AIRe/QED-Nano
(`LICENSE-QED-Nano`). AoPS problems remain their authors'. `NOTICE` lists every source.

```bibtex
@misc{qednano2026,
  title  = {QED-Nano: Teaching a Tiny Model to Prove Hard Theorems},
  author = {LM-Provers and Yuxiao Qu and Amrith Setlur and Jasper Dekoninck and Edward Beeching and Jia Li and Ian Wu and Lewis Tunstall and Aviral Kumar},
  year   = {2026},
  howpublished = {\url{https://huggingface.co/spaces/lm-provers/qed-nano-blogpost}}
}
```
