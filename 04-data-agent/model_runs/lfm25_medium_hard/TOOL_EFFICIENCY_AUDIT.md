# Tool-efficiency reward audit — 2026-09-22

The reward is correctly wired and active. No live training configuration was changed.

Policy: incorrect answers receive 0. Correct answers with a verified positive native action count receive `1 + 1.5/(15 + actions)`. Unknown/zero counts receive correctness only; missing verifier results stay unscorable. For a correct answer, 3 actions yield 1.0833 and 11 yield 1.0577.

## Live reward audit (10:16 UTC snapshot)

| Training arm | Graded reward records | Independently recounted native actions | Formula/count mismatches | Wrong answers with bonus | Optimizer-admitted rollouts missing evidence |
|---|---:|---:|---:|---:|---:|
| OpenCode | 1,670 | 1,670 | 0 | 0 | 0 |
| Multi-harness | 1,466 | 1,465 | 0 | 0 | 0 |

The remaining multi-harness record had no available native evidence and received no bonus. Mean positive bonuses were approximately 0.075 for OpenCode, 0.068 for Claude Code, 0.066 for Codex, and 0.064 for Mini-SWE-Agent. Native counts include recorded actions absent from retained capture after retries; these are not extra bonuses or double-counted rewards. Counts measure native recorded tool calls, not shell commands or total compute.

The worker uses the rollout reward callback, no extra reward functions, and no additional environment reward column in this finite loop. The resulting reward is normalized within its task/harness group. 27 OpenCode groups and 28 multi-harness groups admitted to optimizer work were all-correct with different tool counts: efficiency alone supplies their reward contrast. Therefore a small raw bonus does not guarantee a small gradient contribution. All-wrong groups still have no contrast.

## Early behavioral check: checkpoint 100 → 200

Compare only the same evaluation task/harness cells solved correctly at both checkpoints with valid native counts. Retain the first graded result per cell, as in the evaluator. Counts below are means on the paired subsets; checkpoint evals remain incomplete.

| Training arm / evaluation subset | Paired cells | Mean actions at 100 | At 200 | Change |
|---|---:|---:|---:|---:|
| OpenCode / all four harnesses | 374 | 6.89 | 7.04 | +2.1% |
| OpenCode / OpenCode harness | 93 | 4.57 | 4.20 | -8.0% |
| Multi-harness / all four harnesses | 390 | 6.74 | 6.35 | -5.8% |
| Multi-harness / OpenCode | 82 | 4.89 | 4.24 | -13.2% |
| Multi-harness / Claude Code | 62 | 6.63 | 6.06 | -8.5% |
| Multi-harness / Codex | 111 | 7.30 | 6.68 | -8.4% |
| Multi-harness / Mini-SWE-Agent | 135 | 7.47 | 7.50 | +0.4% |

This is encouraging for the multi-harness run and for the OpenCode control on its own harness. It is not uniform improvement and does not establish causality: there is no otherwise identical run without shaping, the samples are pass@1, and the paired-correct subset excludes regressions/newly solved tasks and missing results. Do not compare absolute native counts across different harnesses as interchangeable units of work.

Evidence and reproducible scripts: `experiments/lfm25-medium-hard1000-20260921/plots/tool_efficiency_audit.json`, `paired_eval_action_counts.json`, `audit_efficiency.py`, and `compare_eval_actions.py`.
