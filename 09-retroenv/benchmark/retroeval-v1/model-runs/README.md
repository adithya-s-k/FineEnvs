# Model-run artifacts

`board-v1/` is the only official result directory. Its root contains merged,
20-task prediction, episode, manifest, and report files. `board-v1/shards/`
contains the exact non-overlapping five-task source shards; the merged reports
were recomputed offline from private truth.

`board-v1/board.json` deliberately has `complete: false` until every model in
`eval/benchmark_models.json` is present. The only currently missing model is
GPT-5.5.

All official runs use required tool choice except Claude Opus 5.5, for which
OpenRouter rejects forced (`required`/`any`) tool choice across providers.
The official Opus run therefore uses `auto`; its terminal turn still exposes
only `emit_routes`. The failed, zero-token forced-tool probe is retained under
`board-v1/incompatible-required/opus/` and is excluded from the board.

The following are retained only as audit evidence and are excluded from the
board:

- `schema-pilot/`: one-task schema compatibility probe.
- `board-v1/partial-unsharded/`: interrupted pre-sharding runs.
- `board-v1-aborted-no-tool-recovery/`: run before empty-turn recovery.
- `board-v1-aborted-unbounded-empty-recovery/`: run before recovery was bounded.
- `board-v1-aborted-intermediate-leakage/`: run invalidated after the deeper
  hidden-intermediate leakage audit found a crossing.
- `board-v1/incompatible-required/opus/`: zero-token provider rejections from
  attempting required tool choice with Opus 5.5.

Do not compare or aggregate those directories with `board-v1/RESULTS.md`.
