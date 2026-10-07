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
