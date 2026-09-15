# Nayana data contract

Source: [Cognitive-Lab/NayanaOCR_Corpus_2025](https://huggingface.co/datasets/Cognitive-Lab/NayanaOCR_Corpus_2025),
revision `b220b074a8c82bb90427051e856e4c4edc79885b`, CC BY-NC 4.0, CognitiveLab.
The complete card and attribution remain available at that pinned revision. Code in this
repository does not change the license of copied pages, annotations, or derived crops.

`source-manifest.json` is metadata, not a redistributed dataset. `snapshots/` is ignored by Git.
The Space publisher copies an explicitly selected window, including its provenance manifest.

| Source field | Use |
|---|---|
| `jpg` | Embedded JPEG bytes, accessed with `Image(decode=False)` |
| `image_id.txt` | Canonical page ID; derive document ID by removing `_page_<number>` |
| `regions.json` | Region ID, original pixel bbox, English/translated reference text |
| `vqa.json` | MCQ question and options; descriptive answers deferred |
| `font_used.txt` | Source font metadata, not needed in the serving projection |
| `__key__` | Not a globally stable ID |
| `__url__` | Original source archive provenance, not a media URL |

Task identity hashes schema version, dataset revision, language, canonical page ID, task
family, and region ID or question position. Split assignment hashes the document ID and seed
independently of language. Changing the task derivation or normalization contract requires a
schema version bump and a newly prepared window.

Prepared directories contain:

```text
catalog.sqlite    task/reference metadata, asset index, and page-level preparation checkpoints
assets/<sha256>   original JPEGs or lossless PNG crops, deduplicated by content
manifest.json    finalized configuration, counts, snapshot ID, source and audit information
```

The snapshot ID hashes configuration and ordered task records, including media hashes.
Finished directories are immutable. Use one writer per output directory and serve the same
directory to all replicas. Do not rebuild a mounted directory while it is being served.
Use a new versioned path and restart the server after its preparation completes.

This is a **bounded-window workflow**. It neither indexes all remote Parquet rows nor performs
training through the Dataset Viewer API. SQLite is the local task index; the preparation
input remains a Datasets Parquet stream. An ordinary downloaded Arrow Dataset is also disk
backed and memory mapped, but would require downloading the selected full config first.

Preparation is sequential without a shuffle buffer. `--num-shards` and `--shard-index`
partition physical stream shards for separate preparation jobs; excessive partitions are
rejected. The first training recipe supports one GPU and does not independently shard TRL's
rollouts. Combining windows or adding distributed live scheduling is future work.
