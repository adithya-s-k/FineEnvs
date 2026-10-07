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

## Release build

`build_release.py` turns the PaRoutes v2 archive into the RL release (tasks, stock,
reaction library) in `data/release/RetroEnv-RL`, and `audit_release.py` checks it with the
runtime code. The stages live in `pipeline/`: flatten, annotate (class, rdchiral template,
atom-map flags), solve the reaction graph against the stock, choose targets, split by
blocklist, generate constraint variants with witnesses, and write the release.

Leakage contract: held-out tasks share no patent, route molecule (target or
intermediate), reaction, scaffold group or Tanimoto >= 0.90 near-duplicate with train or
with each other, including the molecules and reactions of variant witness routes. Library
reactions touching a held-out key are marked invisible and never reach the agent's tools.

Witness routes join corpus reactions from different patents through shared
intermediates. Each step is a recorded reaction, so the chemistry-first verifier accepts
them, but the route as a whole was never run; they only prove a task is solvable and add
a similarity bonus, while patent routes keep their own provenance.

