"""Shared corpus fixture: a tiny synthetic bucket-parquet corpus with a real index."""

import hashlib

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from nayana_ocr.data.corpus import CorpusCatalog
from nayana_ocr.data.index import build_index, sha256_file
from nayana_ocr.data.schema import REVISION, canonical_json
from nayana_ocr.fixtures import fixture_rows


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    root = tmp_path / "source"
    files = []
    for language in ("en", "ar"):
        rows = list(fixture_rows(language))
        for index in range(2):
            path = root / language / f"train-{index:05d}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(
                pa.Table.from_pylist(rows[index * 3 : (index + 1) * 3]),
                path,
                row_group_size=2,
            )
            files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "xet_hash": sha256_file(path),
                }
            )
    inventory = {
        "bucket_id": "fixture/corpus",
        "source": "synthetic-fixture",
        "revision": REVISION,
        "source_license": "synthetic-fixture",
        "files": files,
        "source_bytes": sum(f["size"] for f in files),
        "parquet_files": len(files),
    }
    inventory["inventory_id"] = hashlib.sha256(
        canonical_json(inventory).encode()
    ).hexdigest()
    output = tmp_path / "index"
    with monkeypatch.context() as context:
        # A complete index must not decode or materialize any page image.
        context.setattr(
            "nayana_ocr.data.tasks.Image.open",
            lambda *a, **kw: pytest.fail("Index touched image decoder"),
        )
        manifest = build_index(
            inventory, output, languages=("en", "ar"), source_root=root, workers=2
        )
    catalog = CorpusCatalog(
        output,
        tmp_path / "cache",
        source_root=root,
        local_source=True,
        group_cache_bytes=2_000_000,
        max_group_bytes=1_000_000,
        asset_cache_bytes=100_000,
    )
    yield catalog, manifest, root, output, inventory
    catalog.close()
