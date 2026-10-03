# Quick OpenRouter benchmark

Run on 2026-09-24 against the two frozen `dev` tasks. Each model received one
attempt per task, 20 model turns, temperature 0.2, identical tools, stock, and
private verifier. This is a harness smoke test (`n=2`), not a model leaderboard.

| Rank | Model | Mean reward | Task rewards | Pass@1 | Graph | Calls | Tokens | Cost |
|---:|---|---:|---|---:|---:|---:|---:|---:|
| 1 | Claude Sonnet 5 | 0.4844 | 0.2438 / 0.7250 | 0/2 | 0.7188 | 36 | 290,292 | $0.6210 |
| 2 | GPT-5.6 Luna | 0.3625 | 0.4813 / 0.2438 | 0/2 | 0.6250 | 33 | 86,320 | $0.0177 |
| 3 | Qwen3.6 Max Preview | 0.2438 | 0.2438 / 0.2438 | 0/2 | 0.4375 | 61 | 261,394 | $0.3467 |
| 3 | DeepSeek V3.2 | 0.2438 | 0.2438 / 0.2438 | 0/2 | 0.4375 | 36 | 213,912 | $0.0363 |
| 5 | Qwen3.8 27B | 0.0500 | 0.0500 / 0.0500 | 0/2 | 0.0000 | 32 | 75,317 | $0.0699 |

The final comparable sweep cost $1.0916. Calibration/retry traffic—including a
first run whose visible turn budget was mismatched and a rejected Qwen forced
tool-choice request—brought total key usage for the day to $1.9896.

## What failed

- No model produced a fully supported, stock-closed two-route set.
- Claude produced the strongest single episode: structurally valid graphs,
  valid molecules, one supported step, and 0.5 reference similarity, but no
  complete valid route because terminal stock closure failed.
- GPT produced a well-formed first-task graph with valid molecules and truthful
  stock leaves, but its reaction steps had no hidden evidence support. Its
  second output failed the graph-node schema.
- Qwen and DeepSeek emitted on time but placed non-molecule wrapper objects in
  `routes`; both consequently earned only parse, route-count, and partial graph
  credit.
- Qwen3.8 27B was the slowest run. It made no tool call on the first task, then
  exhausted all 32 environment calls on the second without invoking
  `emit_routes`; both attempts therefore received the empty-submission floor.

The immediate environment improvement is to put a recursive molecule/reaction
schema directly into the `emit_routes` tool definition instead of relying on
the textual prompt plus generic `object` items. The chemistry result is also a
useful baseline: a six-reaction training precedent index is far too sparse for
held-out route discovery, so the next data milestone should build the retrieval
index from the larger leakage-safe training corpus.

Raw predictions, episode transcripts, online scores, and offline reports are
stored beside this file for each model.
