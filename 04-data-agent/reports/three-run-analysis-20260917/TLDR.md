Qwen3.5-2B comparison is complete: three 1,000-step runs, evaluated on 250 fixed tasks × four harnesses, pass@1.

- Best scores: **Harbor multi-harness 37.0% @500**, **Harbor OpenCode-only 39.5% @700**, **native OpenCode 29.8% @1,000**. The Harbor runs finish lower, at 26.3% and 26.4%.
- Multi-harness training outputs grow **3.5k → 9.5k tokens/rollout**, while tool calls fall **15.9 → 11.2**. In the final window, 37.5% of admitted rollouts contain a response exceeding the eval's 4k cap; eval truncation rises sharply.
- Harbor OpenCode-only makes more calls but submits fewer answers. Native keeps shorter outputs and uses fewer training tokens, although late context duplication increases its compute cost.
- We found a resume-accounting bug: **1,575/1,579 post-resume rollouts revisit seen tasks**. Also, **35–58% of steps have zero fresh gradient** because groups have no reward contrast.

Next: fix resume accounting, test matched output budgets, and track completion, tokens, tool use and task coverage. Equal steps were not equal exposure; these results do not establish that multi-harness training is worse.
