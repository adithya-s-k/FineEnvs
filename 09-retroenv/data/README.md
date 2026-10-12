# Local raw data

`raw/` contains immutable upstream artifacts and is intentionally ignored by
Git. Recreate it with:

```bash
python dataset/download_raw.py --include-research-only
python dataset/inventory_raw.py
```

`raw/DOWNLOAD_RECEIPTS.json` records pinned revisions, upstream checksums, and
licenses. `raw/RAW_INVENTORY.json` records a SHA-256 digest for every local
artifact while excluding Git object stores. Neither file is a redistribution
license: consult `dataset/sources.json` before publishing an artifact.

Generated and non-commercial sources must remain separate from the default
observed/reusable corpus configuration.

# Serving snapshot

[LiteFold/RetroEnv](https://huggingface.co/datasets/LiteFold/RetroEnv) is the release and stays
canonical. [FineEnvs/retroenv-bucket](https://huggingface.co/buckets/FineEnvs/retroenv-bucket) is the
public copy a server reads, written by `dataset/publish_bucket.py`:

| Bucket path | What it is |
|---|---|
| `manifest.json`, `checksums.json`, `library/`, `stocks/`, `tasks-*/` | The release at a pinned revision, copied server-side by Xet hash; `runs/` is left out |
| `openenv/indexes/<snapshot_id>/serving.sqlite.gz` | The serving index: task rows, stock and precedents, precomputed; gzipped (196 to 82 MB), and `prepare.py` checks it before and after unpacking |
| `openenv/evalsets/final_eval.json` | The frozen 50-task `final_eval` subset (also [here](eval-final_eval.json)) |
| `openenv/indexes/<snapshot_id>/manifest.json` | What a server fetches, with each file's size and SHA-256; published last |

The index (`envs/retro_route/core/retroenv/serving.py`) holds what every server would otherwise
derive at startup: fingerprints and leakage keys for 177,822 train-visible precedents, the stock's
canonical SMILES and fingerprints, and every task row, compressed. It is written through the same
`PrecedentIndex` and `StockIndex` code a server would run, and `tests/test_serving.py` checks that
tasks, stock search, precedent search and scores are identical either way. It records the RDKit
version that built it; a server with another version recomputes the RDKit-derived parts instead.

`snapshot_id` hashes the index version, the source repository, revision and `checksums.json`, the
RDKit version and the index's SHA-256, so two servers agree only if they serve identical data.
`envs/retro_route/openenv/corpus-manifest.json` is the committed copy of the manifest; it pins what
`prepare.py` fetches, locally or on the Space. To publish a new snapshot:

```bash
uv run python -m dataset.build_final_eval      # only if the release or the design changed
uv run python -m dataset.publish_bucket        # mirror, upload the index, publish the manifest last
```

The bucket keeps PaRoutes' CC-BY-4.0 licence and attribution (Genheden & Bjerrum, *Digital
Discovery* 2022; Zenodo 7341155).
