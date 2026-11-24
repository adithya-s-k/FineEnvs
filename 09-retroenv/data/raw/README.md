# Raw data cache

This directory is intentionally ignored by Git. Run
`python dataset/download_raw.py --include-research-only` to populate pinned,
license-reviewed sources, then `python dataset/inventory_raw.py` to hash them.
Every source directory receives a `RETROENV_RECEIPT.json`; the root receives a
combined `DOWNLOAD_RECEIPTS.json` and `RAW_INVENTORY.json`.

Raw artifacts are not one redistributable dataset. Preserve their separate
licenses and evidence labels when publishing derived Hugging Face configs.
