# RetroEnv explorer

A local browser for a built RetroEnv release: the environment, every task, its hidden
known routes, and live play through the real session.

```bash
uv run python -m dataset.build_release                 # once: writes data/release/RetroEnv-RL
uv run python explorer/server.py                        # then open http://127.0.0.1:8050
uv run python explorer/server.py --release tests/fixtures/mini-release   # the small test release
```

| View | What it shows |
|---|---|
| Environment | Tasks per split and what each split is for, one episode, the chosen split's shortest-route depth, variants, first reaction, tier and target size (click a bar to list those tasks), the tools and the reward components |
| RL tasks | Every task, filterable by split, variant, shortest route, first reaction, tier and size, or searched by ID or SMILES |
| Task | What the policy sees (target, depth budget, constraints, exact prompt), the known patent and witness routes as a pan-and-zoom route graph, and a **Try it** panel |

**Try it** plays the task through the same core session the server runs. Call any tool
with JSON arguments, validate the candidate cuts the rule library proposes, submit the
known routes or an empty route, and see the reward split into its components. The first
episode loads the release's reaction library and precedent index (about a minute).

Held-out tasks (dev, test_id, test_hard) hide their known routes until you choose to show
them, so the default view is what a model sees. Molecules are drawn by RDKit on the server
and follow the page's light or dark theme. The server listens on localhost only.
