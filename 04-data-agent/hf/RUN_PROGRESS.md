# Three-run training progress

Checked: 2026-09-17T07:57:03.596454+00:00

Automatic checks run every 10 minutes on CPU. Reward windows describe optimizer updates; fixed-test pass@1 measures checkpoint quality. Startup and changing task difficulty can change training reward.

| Run | Job | State | Step | Reward last 20 | Previous 20 | Alerts |
| --- | --- | --- | ---: | ---: | ---: | --- |
| multi4 | 80608 | COMPLETED | 1000 | 0.268 | 0.341 | controller:eval_failed:checkpoint-1000; controller:eval_failed:checkpoint-900; controller:tito_failure:claude-code; controller:tito_failure:codex; controller:tito_failure:mini-swe-agent; controller:tito_failure:opencode; tito_failed |
| opencode | 80626 | COMPLETED | 1000 | 0.208 | 0.171 | none |
| whitebox | 6aa9b6c9f76d6a098a70e786 | CANCELED | 150 | 0.394 | 0.512 | trainer_ended_CANCELED |

| Multi-harness checkpoint | Eval job | State | Audited pass@1 |
| --- | --- | --- | ---: |
| checkpoint-100 | 79057 | COMPLETED | 24.8% |
| checkpoint-200 | 79199 | COMPLETED | 26.3% |
| checkpoint-300 | 79305 | COMPLETED | 28.6% |
| checkpoint-400 | 80323 | COMPLETED | 33.3% |
| checkpoint-500 | 80639 | COMPLETED | 37.0% |
| checkpoint-600 | 80644 | COMPLETED | 31.8% |
| checkpoint-684 | 80908 | COMPLETED | 32.1% |
| checkpoint-700 | 80910 | COMPLETED | 28.8% |
| checkpoint-800 | 80912 | COMPLETED | 27.0% |
| checkpoint-900 | 81645 | COMPLETED | 22.7% |
| checkpoint-1000 | 81645 | COMPLETED | 26.3% |

Independent trainer/inference GPU pairs; independent eval GPUs. The Daytona checkpoint controllers reserve training capacity and admit one comparison eval at a time. The original multi-harness run uses E2B.

JSON snapshots retain checkpoint/eval state, TiTO evidence, logging health, numerical checks and recovery actions. Transient dead CPU controllers can be restarted twice. Training, TiTO or provenance failures are recorded for diagnosis; the monitor never resets weights in response to a reward dip.
