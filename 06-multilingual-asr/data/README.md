# FLEURS data contract

Source: [google/fleurs](https://huggingface.co/datasets/google/fleurs), **CC BY 4.0**.
Pinned copy: [FineEnvs/fleurs-bucket](https://huggingface.co/buckets/FineEnvs/fleurs-bucket),
878,377,671,348 bytes across 1,004 files, holding both the raw `data/<lang>/` layout
(audio plus `train/dev/test.tsv`) and `parquet-data/<lang>/<split>-*.parquet`.

Columns used: `id`, `path`, `num_samples`, `audio` (struct of `bytes`/`path`, 16 kHz),
`transcription`, `raw_transcription`, `language`, `gender`. `lang_id` and `lang_group_id`
are not used: they are label indices that would pin this environment to one FLEURS release
ordering.

**`id` is a sentence id, not an utterance id.** Several speakers record the same sentence,
so hi_in test holds 418 rows under 265 distinct `id` values. Identity therefore comes from
`path`, the recording's own filename: keying on `id` would have collapsed 37% of the corpus
and given two different clips the same task id.

`data/corpus-manifest.json` is the committed index of the whole corpus: one SQLite index per
language, published in the bucket under `openenv/indexes/<snapshot_id>/`, recording every
eligible utterance and the parquet file, row group, and row that holds its audio. The
manifest carries each index's SHA-256 and per-language task counts, and `snapshot_id` hashes
the index version, the bucket inventory, and every index digest together, so two deployments
agree only if they address identical data. Audio is never copied: it is read from the bucket
on demand into a byte-bounded cache.

`asr-prepare` still writes a self-contained snapshot of `manifest.json`, `catalog.sqlite`,
and `assets/<sha256>` for a machine with no bucket access. Snapshots and prepared audio are
not committed: they carry redistributable but bulky source media. Keep attribution with any
copy you publish.
