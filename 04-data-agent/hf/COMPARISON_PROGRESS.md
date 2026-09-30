# Comparison checkpoint evaluations

Checked: 2026-09-16T08:18:30.309899+00:00

Native OpenCode training uses Daytona; these checkpoint evaluations use its four Harbor adapters on the fixed 250 tests per harness. All entries below have 1,000 graded cells, exact TiTO and matching harness versions. The separate native 250-task baseline is 8.4%; it is not a four-harness average.

| Checkpoint | OpenCode | Claude Code | Codex | Mini-SWE-Agent | Overall pass@1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 100 | 19.6% | 20.4% | 18.0% | 20.8% | 19.7% |
| 200 | 17.2% | 24.0% | 25.6% | 21.6% | 22.1% |
| 300 | 20.8% | 29.2% | 16.8% | 19.6% | 21.6% |
| 400 | 20.4% | 32.4% | 25.6% | 27.2% | 26.4% |
| 500 | 18.0% | 26.4% | 18.8% | 29.2% | 23.1% |
| 600 | 16.8% | 32.4% | 18.4% | 32.8% | 25.1% |
| 700 | 18.4% | 29.6% | 14.4% | 30.4% | 23.2% |

Native training completed1,000steps. Checkpoint800 evaluator80993 has999/1000grades;900/1000 remain outstanding. SETA is running at132updates; its first checkpoint100 scored87/250=34.8% versus47/250=18.8% at baseline. Easy23/33=69.7%,medium45/118=38.1%,hard19/99=19.2%. The SETA evaluation completed before an artifact-upload HTTP500 caused the HFjob to report ERROR; the controller was repaired to recognize fully verified published results. See [STATUS.md](STATUS.md) for evidence and repair details. The original multi-harness learning curve and difficulty breakdown are in [train/PROGRESS.md](../train/PROGRESS.md).

Original checkpoint500/600 evaluations are now fully comparable at37.0% and31.8%. The retry scorer originally failed to follow measured harness-version records in the original trial directories. The corrected lookup passed12tests, including actual version mismatch and invalid/cyclic ancestry rejection; all2,000grades are unchanged. Independent later-eval recoveries80908/80910/80912/80914 preserve997/993/990/3grades respectively. Training weights and hyperparameters remain unchanged.
