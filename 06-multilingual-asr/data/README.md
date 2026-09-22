# FLEURS data contract

Source: [google/fleurs](https://huggingface.co/datasets/google/fleurs), **CC BY 4.0**.
Pinned copy: [FineEnvs/fleurs-bucket](https://huggingface.co/buckets/FineEnvs/fleurs-bucket),
878,377,671,348 bytes across 1,004 files, holding both the raw `data/<lang>/` layout
(audio plus `train/dev/test.tsv`) and `parquet-data/<lang>/<split>-*.parquet`.

Columns used: `id`, `num_samples`, `audio` (struct of `bytes`/`path`, 16 kHz),
`transcription`, `raw_transcription`, `language`, `gender`. `path`, `lang_id`, and
`lang_group_id` are not used: the first is a source-local filename and the other two are
label indices that would pin this environment to one FLEURS release ordering.

`asr-prepare` writes a snapshot of `manifest.json`, `catalog.sqlite`, and `assets/<sha256>`.
The manifest records the source, revision, languages, families, media bytes, per-group task
counts, and every exclusion reason with its count. `snapshot_id` hashes the settings
together with every task payload, so two snapshots agree only if their tasks do.

Snapshots and prepared audio are not committed: they carry redistributable but bulky source
media. Keep attribution with any copy you publish.
