<div align="center">

<h1>Math environments</h1>

<h3>From mental arithmetic to research-level proofs, as RL environments</h3>

<p>A catalog of math environments and their evals, from FineEnvs and the community. Each one says how it grades:
exact arithmetic, a kernel-checked Lean proof, or a rubric judge.</p>

<a href="https://huggingface.co/collections/FineEnvs/math-environments-6af40312bf237e3c32912840"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Collection-Math%20Environments-10B981?style=for-the-badge&labelColor=1a1a1a" alt="The Math Environments collection" height="32"></a>
<a href="#add-your-environment"><img src="https://img.shields.io/badge/Contribute-Add%20your%20environment-3B82F6?style=for-the-badge&labelColor=1a1a1a" alt="Add your environment" height="32"></a>

</div>

---

Maths makes good RL environments because an answer can be checked. How it is checked decides what the environment
teaches: an exact check is hard to fool but gives no credit for being nearly right, and a judge gives credit for
progress but can be fooled. The environments here sit at different points on that line, and each README is explicit
about which.

| Environment | What the agent produces | How it is graded | Partial credit |
|---|---|---|---|
| Scott Wu Math | one arithmetic expression | exact evaluation with fractions | shaped, for near misses |
| openai/math | a Lean 4 proof | Comparator: exact statement, standard axioms, kernel replay | none |
| FineProofEnvs | a proof in plain language | an LLM judge against a per-problem rubric, 0–7 | yes, by rubric step |

## The catalog

<table>
<tr>
<td width="33%" valign="top">
<h3><a href="scott-wu-math/">Scott Wu Math</a></h3>
<sub>by <a href="https://github.com/adithya-s-k">Adithya S Kolavi</a> (Hugging Face)</sub>
<p>Make the number from every card once with + − × ÷: the 24 game, then the current year from eight cards, then
any four-digit number from nine, as Scott Wu plays it in a pub. People and models play the same server.</p>
<sub>OpenEnv · 4,000 train / 200 test deals · 12 models</sub><br>
<a href="https://huggingface.co/spaces/FineEnvs/ScottWuMath">Play</a> ·
<a href="scott-wu-math/results/eval-v1.json">Eval card</a>
</td>
<td width="33%" valign="top">
<h3><a href="openai-math/">openai/math</a></h3>
<sub>by <a href="https://github.com/adithya-s-k">Adithya S Kolavi</a> (Hugging Face)</sub>
<p>Prove a research-level result from OpenAI's openai/math release in Lean 4. A separate sandbox runs Comparator:
reward 1 only for a kernel-checked proof of the exact statement.</p>
<sub>Harbor · 369 tasks, each validated with the official proof</sub><br>
<a href="https://huggingface.co/datasets/FineEnvs/openai-math">Dataset</a> ·
<a href="openai-math/envs/openai_math/harbor/README.md">Grading and threat model</a>
</td>
<td width="33%" valign="top">
<h3><a href="fineproof/">FineProofEnvs</a></h3>
<sub>by <a href="https://github.com/adithya-s-k">Adithya S Kolavi</a> (Hugging Face), from the
<a href="https://huggingface.co/lm-provers">LM-Provers</a> release</sub>
<p>Write an olympiad proof in plain language; gpt-oss-20b grades it 0–7 against the problem's rubric, the reward
QED-Nano was trained with. RL from the base model takes Qwen3.5-2B from 0.22 to 0.41 on held-out proofs.</p>
<sub>Harbor · 5,227 train problems, IMO-ProofBench, ProofBench</sub><br>
<a href="fineproof/training/RESULTS.md">Results</a> ·
<a href="https://huggingface.co/FineEnvs/Qwen3.5-2B-FineProof-RL">Model</a> ·
<a href="https://huggingface.co/spaces/FineEnvs/fineproof-rl-trackio">Training logs</a> ·
<a href="fineproof/DESIGN.md">Design</a>
</td>
</tr>
</table>

## The evals

Each row comes from an eval card.

| Environment | Eval tasks | 1.0 means | Judge | Models | Best mean reward | Eval card |
|---|---|---|---|---:|---|---|
| Scott Wu Math, at once | 20 test deals, 5 per level | the target, exactly | no | 12 | GPT-6.1 Sol 0.70\* | [eval-v1.json](scott-wu-math/results/eval-v1.json) |
| Scott Wu Math, reasoning | the same 20 deals | the target, exactly | no | 12 | 7 of 12 solve all 20 | [eval-v1.json](scott-wu-math/results/eval-v1.json) |
| openai/math | 369 tasks | a kernel-checked proof of the exact statement | no | | not run yet | |
| FineProofEnvs, held-out test | 64 validated test problems × 4 | 7 of 7 on the rubric | gpt-oss-20b | 2 | Qwen3.5-2B-FineProof-RL 0.41 (base 0.22) | [eval-v1.json](fineproof/results/eval-v1.json) |
| FineProofEnvs, benchmarks | IMO-ProofBench (60), ProofBench (145) | 7 of 7 on the rubric | gpt-oss-120b | | not run yet | |

\* In "at once" mode GPT-6.1 Sol and Claude Opus 5.5 still reason briefly (`effort: low`, the lowest their APIs allow), so their rows are not pure mental arithmetic. They are the only two above 5 of 20 in that mode.

## Add your environment

There are two ways in:

- **Code here.** Open a PR that adds a folder like [scott-wu-math](scott-wu-math/): `envs/<env>/` for the
  environment, `eval/` for the runner. [CONTRIBUTING](../CONTRIBUTING.md) has the layout.
- **Link only.** Your code, data and Hub repos stay where they are; the folder points to them.

Either way, the folder holds three files:

1. **`project.yaml`.** Title, tagline, `authors` (name, GitHub handle, affiliation), your Hub repos and frameworks.
   Copy [scott-wu-math/project.yaml](scott-wu-math/project.yaml), then run `python3 tools/build_index.py`. The root
   README lists the environment with your name on it.
2. **`README.md`.** The task, the data and its licences, how it's graded (and whether a judge is involved), and links
   to your code, Space and data.
3. **`results/eval*.json`.** One eval card per environment or version, in the format of
   [07's cards](../07-simulation-environments/README.md#add-your-environment); `grading.judge` says whether an LLM
   judge decides the reward.
