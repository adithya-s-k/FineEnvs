# Full-corpus data contract

Source: [CognitiveLab / NayanaOCR_Corpus_2025](https://huggingface.co/datasets/Cognitive-Lab/NayanaOCR_Corpus_2025/tree/b220b074a8c82bb90427051e856e4c4edc79885b),
revision `b220b074a8c82bb90427051e856e4c4edc79885b`, **CC BY-NC 4.0**.
Copy: [HuggingEnvs/NayanaOCR_Corpus_2025_bucket](https://huggingface.co/buckets/HuggingEnvs/NayanaOCR_Corpus_2025_bucket).
There are 45,735 pages per language, 22 languages, 1,784 Parquet shards, 1,807 source files,
and 813,665,107,295 bytes of pinned files. Sizes include repository metadata; they are not
image-only byte counts. Historical revisions are excluded.

`nayana-mirror` generates the ignored `corpus-source.json` inventory with every source path,
size, Xet identity, and LFS SHA-256 where available. Recreate it from the pinned revision
when auditing or rebuilding the index.
The mirror preserves source paths and uses server-side copying for Xet files. It refuses
conflicting existing objects and verifies every copied Parquet's Xet hash and size. Small
non-Xet metadata files are size-checked. Existing unrelated bucket objects are preserved.
The dataset card and attribution accompany the copied corpus.

`corpus-manifest.json` pins the published full index. The generated SQLite databases live at
`openenv/indexes/<snapshot_id>/<language>.sqlite` in the same bucket; `manifest.json` is published
last. The snapshot hashes the source inventory, index version, split seed, and each database
SHA-256. The server validates the identity and each fetched index before opening it locally.

| Source field | Use |
|---|---|
| `jpg` | Embedded JPEG bytes; read only when an image group is needed |
| `image_id.txt` | Canonical page ID; document ID removes `_page_<number>` |
| `regions.json` | Region IDs, pixel boxes, English and translated text |
| `vqa.json` | Multiple-choice and descriptive questions with source answers |
| `font_used.txt` | Source metadata, preserved in the bucket but not used by serving |
| `__key__`, `__url__` | Original provenance; neither is a global task ID or image endpoint |

## Index and source access

Indexing projects only `image_id.txt`, `regions.json`, and `vqa.json` from every Parquet file.
The footer supplies row-group boundaries and expected row counts. Each language has SQLite
`files`, `blocks`, `pages`, and `tasks` tables. Page annotations are compressed once; task rows
reference their page. Global split positions and language/family positions have direct indexes.
References are reconstructed from stored annotations. No JPEG decoding or crop generation occurs
during indexing. One transaction per shard makes interrupted builds resumable. A finalized
index is immutable; change versions or derivation policy in a new directory.

The source has only a train split. A seeded SHA-256 document partition creates train,
validation, and test with 80/10/10 hash buckets. All languages and tasks of one document share
its split. Language row orders differ and are not used to join translations. Counts reflect
the actual indexed annotations in each partition; they are not estimates from a sample.

A stable full-corpus task ID is `nayana-c1.<snapshot_id>.<language>.<split>.<position>`.
Metadata enumeration returns IDs, instructions, and source/task descriptors without reading
image bytes. Before materialization, dimensions and asset hashes are unavailable. Reset
fetches the physical JPEG row group and checks the requested page ID, then derives the image.

The original grouping is retained. Cold access generally reads 100 JPEGs (roughly 70–110 MB),
not one image. The server verifies bucket Xet identities before and after a cold read. A local
offline source uses LFS SHA-256 checks instead. Cached content remains tied to the original
identity. Never edit the pinned source or index prefix during a run; create a new snapshot.

## Caching and replay

Three local caches hold index files, image groups (Arrow IPC), and rendered task envelopes.
Atomic writes and striped file locks coalesce concurrent loads and pin files during reads.
LRU eviction skips leased entries. A failed load leaves no published partial entry. Prefetch
uses a bounded worker pool and queue; foreground loads share its concurrency limit. An evicted
asset is regenerated using the pinned task ID and checked against its content SHA-256.

Full-epoch training hash-shuffles row groups, optionally partitions entire groups across
ranks, and shuffles fixed-size metadata chunks inside each group. It visits each selected task
once per epoch. Its cursor records snapshot, seed, partition, plan, block, and consumed offset.
This provides exact iterator replay; optimizer checkpoint replay is a separate unverified
contract. The first GPU recipe is single-process/single-GPU and lets TRL repeat completions.

## Annotation and image policy

Schema 3 supplies `section_ocr`, `mcq_vqa`, `page_ocr`, `layout_detection`, and
`descriptive_vqa`. Layout includes all six source classes, including non-text regions;
descriptive answers use strict Gemma grading through HF Inference Providers. See [JUDGE.md](../JUDGE.md). Full-page annotations require unique
region IDs, nonempty language references, valid boxes, and no positive-area overlap. Geometric
reading order uses `whitespace-columns-v1`; Arabic columns run RTL while text code points
remain in logical order. Outside-region pixels are masked white. Section crops preserve source
pixels and VQA preserves the original JPEG. This is not official semantic reading order or
table-format ground truth.

Source annotations do not include image dimensions. Indexing validates their metadata; reset
validates their bounds against the actual image. Counts describe candidate tasks before this
lazy image validation. Exclusions and malformed annotations are reported per language in the
manifest. Rendering/font defects require separate quality audits; no automatic audit of all
one million source images is claimed.

`nayana-prepare` still builds small offline windows and fixtures using Datasets streaming;
that path is optional and is not the complete-corpus serving backend. The runtime
`corpus-manifest.json` is committed because it pins the published source and index identities.
Generated inventories, indexes, caches, windows, and run reports are ignored by Git.
The source license also covers derived indexes and crops.

## Measured layout

The complete index is **4,304,867,328 bytes** across 22 databases and addresses **11,052 physical
row groups**. Compressed image-column bytes per group have median **75,121,734**, 95th percentile
**115,700,835**, and maximum **188,043,822**. These are footer sizes, not application network
measurements. Partial groups at shard ends can contain fewer than 100 pages. The committed
[manifest](corpus-manifest.json) records task totals, exclusions, and per-language index sizes.
