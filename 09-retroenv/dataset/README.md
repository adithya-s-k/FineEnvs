# Dataset pipeline

The normalized JSONL schema is intentionally small and source-neutral. Use
`retroenv-build-corpus` for reaction-SMILES tables. It streams its inputs,
deduplicates by atom-map-free canonical reaction hash, merges provenance, and
writes bad rows to a rejects file instead of silently dropping them.

## Downloaded sources

`sources.json` is the machine-readable intake register. The default downloader
fetches approved sources; `--include-research-only` also fetches quarantined
benchmarks whose lineage still prevents redistribution in a combined release.

```bash
python dataset/download_raw.py --list
python dataset/download_raw.py --include-research-only
python dataset/inventory_raw.py
```

The current reviewed intake covers ORD, Lowe USPTO (1976–2016), CRD 1.37M,
USPTO-LLM, CREED/CREED-CCV, FREA/RxnVerif, PaRoutes v2, and SynRXN. CREED is
non-commercial; the PaRoutes Zenodo benchmark is CC-BY-4.0 while its software
repository is tracked separately; SynRXN remains per-artifact. Generated
examples stay labelled and are excluded from default task generation. Named
sources without an auditable artifact/license remain `pending_review` and are
not fetched.

The 3.84 GB PaRoutes `selected_reactions*.csv` pair is recorded as
`covered_by_upstream`: it is a derived subset of the already downloaded Lowe
USPTO raw release and is not duplicated under the workspace quota. Exact
filenames, sizes, and MD5 values remain pinned in `sources.json`.

Source adapters are available through `retroenv-prepare-source` after installing
`uv sync --extra sources`. They cover CRD zip text, Lowe USPTO 7z tables,
CREED Parquet, FREA CSV, USPTO-LLM zip files, PaRoutes v2 trees, and ORD
protobuf-in-Parquet. The ORD adapter preserves the explicit reaction roles from
the official schema instead of guessing them from a flat string. The PaRoutes
adapter walks the supplied trees; it never reconstructs routes by globally
joining reaction rows.

```bash
retroenv-prepare-source paroutes_v2 \
  data/raw/paroutes-v2-benchmark build/paroutes-n1.routes.jsonl \
  --release n1
```

The official ORD repository documents that Parquet rows contain serialized
protobuf reaction messages and gives conversion examples:
<https://github.com/open-reaction-database/ord-data#data-manipulation>.

Typical pipeline:

```text
raw immutable artifact + license manifest
  -> source-specific role extraction
  -> retroenv-build-corpus
  -> normalized/deduplicated corpus + rejects + report
  -> retroenv-build-tasks
  -> server-private tasks + split manifest + stock snapshot
  -> export_public.py
  -> reference-free public prompts
```

Do not infer multi-step routes by globally joining products and reactants.
Route planning records must preserve route-level provenance.

## Leakage contract

The task builder unions tasks sharing any exact target, Murcko scaffold, route
ID, reaction ID, source/patent group, or Morgan-fingerprint near-duplicate. A
whole connected component is assigned to one split, and output is rejected if
the post-split audit sees any group crossing a boundary. Because exhaustive
fingerprint comparison is quadratic, the built-in exact comparison refuses
corpora above 25,000 tasks. Large releases must pre-cluster targets and feed the
cluster identifier as a source group; disabling the comparison without an
external audit is not a release-quality configuration.
