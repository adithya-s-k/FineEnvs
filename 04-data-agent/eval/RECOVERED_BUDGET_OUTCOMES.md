# Harbor checkpoint 900/1,000 evaluation completion

Both checkpoints have 250 tasks × four harnesses = 1,000 audited cells.

| Checkpoint | Overall | OpenCode | Claude Code | Codex | Mini-SWE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 900 | 22.7% | 2.4% | 36.4% | 19.2% | 32.8% |
| 1000 | 26.3% | 5.6% | 38.8% | 24.8% | 36.0% |

The evaluation driver discarded verifier rewards whenever an agent error was present and misclassified the agent deadline as a connection timeout. We recovered seven existing verifier zeros for checkpoint 900 and sixteen for 1,000, after checking native trial results, reward files, agent logs and exact-token captures. All previously accepted grades are unchanged. Full TiTO, fixed-task coverage and pinned harness-version checks passed; CPU audit job 81645 completed successfully.

The correction covers the missing cells only. Earlier accepted outcomes retain their historical retry policy. The full curve needs a uniform retrospective audit before a strict first-attempt pass@1 interpretation. Model weights, task sets, time/output limits and sampling were unchanged.

[Public results and scoring note](https://huggingface.co/spaces/HuggingEnvs/data-agent-training-comparison-trackio/blob/main/RECONCILIATION.md). Local evidence and reproducible reconciliation: `experiments/eval-finalize-20260917/` in the parent workspace.
