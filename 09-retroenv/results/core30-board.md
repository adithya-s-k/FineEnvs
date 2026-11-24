# RetroEnv v2 eval board

Split `final_eval`, 30 tasks, 1 attempt(s), 16 model turns, toolset `full`, tool budget 32. Every tool call and reward came from the OpenEnv server. Pass needs every required route verified against a hidden patent route; exact route is the share of tasks where a submitted route matches one step for step.

| Model | Tasks | Pass@1 (95% CI) | Exact route | Reward | Steps | Stock | Structure | Tool calls | No emit | Refused | Cost |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gpt-6-astra | 30/30 | 0.567 [0.39, 0.73] | 0.067 | 0.747 | 0.877 | 0.911 | 1.000 | 12.7 | 0.000 | 0.000 | $5.18 |
| gpt-6.1-sol | 30/30 | 0.567 [0.39, 0.73] | 0.033 | 0.732 | 0.844 | 0.939 | 1.000 | 13.3 | 0.000 | 0.000 | $1.02 |
| claude-opus-5-5 | 30/30 | 0.500 [0.33, 0.67] | 0.133 | 0.614 | 0.723 | 0.725 | 0.767 | 8.9 | 0.200 | 0.200 | $2.29 |
| claude-sonnet-5-5 | 30/30 | 0.400 [0.25, 0.58] | 0.067 | 0.459 | 0.531 | 0.530 | 0.600 | 13.6 | 0.000 | 0.000 | $2.17 |
| deepseek-ai/DeepSeek-V4.1-Flash:novita | 30/30 | 0.300 [0.17, 0.48] | 0.067 | 0.476 | 0.494 | 0.713 | 0.743 | 34.0 | 0.233 | 0.000 | $1.49 |
| moonshotai/Kimi-K2.6:deepinfra | 30/30 | 0.233 [0.12, 0.41] | 0.100 | 0.627 | 0.821 | 0.638 | 0.933 | 16.0 | 0.000 | 0.000 | $2.09 |
| deepseek-ai/DeepSeek-V4-Pro-0813:together | 30/30 | 0.133 [0.05, 0.30] | 0.000 | 0.191 | 0.228 | 0.258 | 0.267 | 15.8 | 0.333 | 0.000 | $4.53 |
| Qwen/Qwen3.8-27B:novita | 30/30 | 0.100 [0.03, 0.26] | 0.000 | 0.307 | 0.261 | 0.408 | 0.707 | 15.5 | 0.233 | 0.000 | $1.37 |
| Qwen/Qwen3.8-2.4T-A95B:together | 30/30 | 0.100 [0.03, 0.26] | 0.033 | 0.281 | 0.317 | 0.281 | 0.547 | 15.6 | 0.033 | 0.000 | $6.02 |
| Qwen/Qwen3.6-35B-A3B:deepinfra | 30/30 | 0.067 [0.02, 0.21] | 0.000 | 0.218 | 0.169 | 0.393 | 0.440 | 15.9 | 0.000 | 0.000 | $0.45 |
| claude-haiku-5-5 | 30/30 | 0.067 [0.02, 0.21] | 0.033 | 0.112 | 0.142 | 0.097 | 0.167 | 10.7 | 0.000 | 0.000 | $0.08 |
| zai-org/GLM-5.3:novita | 30/30 | 0.033 [0.01, 0.17] | 0.000 | 0.343 | 0.478 | 0.281 | 0.622 | 19.0 | 0.000 | 0.000 | $3.55 |
| google/gemma-4-31B-it:novita | 27/30 | 0.000 [0.00, 0.12] | 0.000 | 0.059 | 0.037 | 0.018 | 0.267 | 14.9 | 0.037 | 0.000 | $0.28 |

### By heuristic tier (exact route rate)

| Model | easy | hard | medium |
|---|---|---|---|
| gpt-6-astra | 0.200 (n=10) | 0.000 (n=10) | 0.000 (n=10) |
| gpt-6.1-sol | 0.100 (n=10) | 0.000 (n=10) | 0.000 (n=10) |
| claude-opus-5-5 | 0.200 (n=10) | 0.200 (n=10) | 0.000 (n=10) |
| claude-sonnet-5-5 | 0.200 (n=10) | 0.000 (n=10) | 0.000 (n=10) |
| deepseek-ai/DeepSeek-V4.1-Flash:novita | 0.200 (n=10) | 0.000 (n=10) | 0.000 (n=10) |
| moonshotai/Kimi-K2.6:deepinfra | 0.200 (n=10) | 0.000 (n=10) | 0.100 (n=10) |
| deepseek-ai/DeepSeek-V4-Pro-0813:together | 0.000 (n=10) | 0.000 (n=10) | 0.000 (n=10) |
| Qwen/Qwen3.8-27B:novita | 0.000 (n=10) | 0.000 (n=10) | 0.000 (n=10) |
| Qwen/Qwen3.8-2.4T-A95B:together | 0.100 (n=10) | 0.000 (n=10) | 0.000 (n=10) |
| Qwen/Qwen3.6-35B-A3B:deepinfra | 0.000 (n=10) | 0.000 (n=10) | 0.000 (n=10) |
| claude-haiku-5-5 | 0.100 (n=10) | 0.000 (n=10) | 0.000 (n=10) |
| zai-org/GLM-5.3:novita | 0.000 (n=10) | 0.000 (n=10) | 0.000 (n=10) |
| google/gemma-4-31B-it:novita | 0.000 (n=10) | 0.000 (n=9) | 0.000 (n=8) |

## Reading the table

**No emit** is the share of episodes the model never closed with `emit_routes`; the harness then submits an empty route set, which scores 0. **Refused** is the share the provider declined on safety grounds, which counts as a failed episode and is never re-routed to another model. Where the two columns match, every unclosed episode was a refusal, and the model's reward is held down by requests it did not answer rather than by its chemistry.

**Steps**, **Stock** and **Structure** are the mean `steps`, `stock` and `structure` reward components: steps the reaction library supports, leaves truthfully claimed as purchasable, and a well-formed tree. A model can score well on all three while failing the task: they do not measure whether the route matches the patent.
