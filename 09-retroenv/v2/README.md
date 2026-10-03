# RetroEnv v2 pilot

This isolated pilot tests an explicit, evidence-bearing retrosynthesis DAG on
three frozen evaluation tasks with Claude Opus 5.5. The diagnostic frontend
also reuses the saved v1 trajectories for Sonnet, Luna, Sol, Qwen Max, Qwen
27B, DeepSeek Flash, and DeepSeek Pro.

For a like-for-like native-v2 comparison, Task A was additionally rerun with
Sonnet, Luna, Sol, and Qwen Max. Luna and Sol emitted renderable evidence DAGs;
Sonnet emitted no recoverable graph, while Qwen Max omitted reaction nodes.
The frontend labels each of these outcomes directly and uses the saved v1
trajectory only where no v2 rollout was performed (Tasks B and C).

```bash
uv run python -m v2.prepare_pilot
uv run --extra eval python -m v2.run_pilot
uv run python -m v2.recompute
uv run python -m v2.web.build_data
python -m http.server 8081 --directory v2/web
```

The final schema is a flat shared graph: molecule and reaction nodes have
stable IDs; typed edges point `precursor -> reaction -> product`; route views
reference reaction IDs. Every chemical claim cites immutable tool evidence.
Private references and the full stock remain outside the prompt.

Legacy v1 nested trees are converted to flat DAGs for display only. Their
reaction explanations and precursor roles are preserved, and the UI labels
that they did not use the v2 evidence-citation contract. V1 and v2 rewards are
therefore shown with their own formulas and are not directly rank-comparable.

Reward weights: parse 5%, graph 10%, chemistry 25%, evidence 20%, reasoning
15%, stock 10%, route diversity 10%, and reference coverage 5%. Hard validity
also requires every deterministic check to pass.
