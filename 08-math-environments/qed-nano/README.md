<div align="center">

<h1>QED-Nano proofs</h1>

<h3>Write an olympiad proof in plain language, graded 0–7 by a rubric judge</h3>

<p>QED-Nano's training problems, rubrics and judge as Harbor tasks. Work in progress.</p>

<a href="https://github.com/harbor-framework/harbor"><img src="https://img.shields.io/badge/framework-Harbor-3B82F6?style=for-the-badge&labelColor=1a1a1a" alt="Harbor" height="32"></a>
<a href="https://huggingface.co/spaces/lm-provers/qed-nano-blogpost"><img src="https://img.shields.io/badge/%F0%9F%93%84%20Source-QED--Nano%20blog%20post-10B981?style=for-the-badge&labelColor=1a1a1a" alt="The QED-Nano blog post" height="32"></a>

</div>

---

> **Work in progress.** Nothing here is built yet; [DESIGN.md](DESIGN.md) is the plan.

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
| **olympiad_proofs** | shell (Python) | `daytona` | — |
<!-- END:matrix -->

## Planned task sets

| Set | Tasks | From |
|---|---:|---|
| train | 5,227 | [lm-provers/FineProofs-RL](https://huggingface.co/datasets/lm-provers/FineProofs-RL), with QED-Nano's difficulty labels |
| test | 128 | `lm-provers/FineProofs-RL-test` |
| IMO-ProofBench | 60 | [lm-provers/IMOProofBench](https://huggingface.co/datasets/lm-provers/IMOProofBench) |
| ProofBench | 145 | [lm-provers/ProofBench](https://huggingface.co/datasets/lm-provers/ProofBench), splits `24_25` and `other` |

## Credits

Problems, rubrics, benchmarks and the judge's prompt are from the
[QED-Nano](https://huggingface.co/spaces/lm-provers/qed-nano-blogpost) release by the LM-Provers team (CMU, Hugging
Face, ETH Zurich, Numina; [code](https://github.com/CMU-AIRe/QED-Nano), Apache-2.0). The problems come from
[AI-MO/olympiads](https://huggingface.co/datasets/AI-MO/olympiads) and [AI-MO/aops](https://huggingface.co/datasets/AI-MO/aops)
(Art of Problem Solving). DESIGN.md lists the open licensing questions.
