# /// script
# requires-python = ">=3.12"
# dependencies = ["pyarrow==26.0.0", "huggingface_hub[hf_xet]"]
# ///
"""Trace-read throughput of the DesignWorld-SFT bucket, as train.py `--ids` reads it (one ranged read per trace):
from the mount at /data and over HTTP, at several concurrencies. Run as an HF Job with the bucket mounted:

    hf jobs uv run --flavor cpu-upgrade -v hf://buckets/perpetual-research/designworld-sft:/bucket:ro \\
        -s HF_TOKEN bench_io.py perpetual-research/designworld-sft /bucket
"""
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

BUCKET = sys.argv[1]
MOUNT = Path(sys.argv[2] if len(sys.argv) > 2 else "/data")
fs = HfFileSystem()


class Ranged:
    def __init__(self, size, data_start, data):
        self.size, self.start, self.data, self.pos = size, data_start, data, 0

    def read(self, n=-1):
        a = self.pos - self.start
        b = len(self.data) if n is None or n < 0 else min(len(self.data), a + n)
        self.pos = self.start + b
        return self.data[a:b]

    def seek(self, p, w=0):
        self.pos = p if w == 0 else self.pos + p if w == 1 else self.size + p
        return self.pos

    def tell(self):
        return self.pos

    def seekable(self):
        return True

    def readable(self):
        return True

    def close(self):
        pass

    closed = False


sizes = {f.path: f.size for f in HfApi().list_bucket_tree(BUCKET, prefix="data", recursive=True)
         if getattr(f, "path", "").endswith(".parquet")}
entries = [json.loads(x) for x in fs.cat_file(f"buckets/{BUCKET}/sft/train.jsonl").decode().splitlines() if x.strip()]
entries = [e for e in entries if e["shard"] in sizes]
print(f"{len(sizes)} shards, {len(entries)} readable traces", flush=True)
meta = {}


def fetch(src, rel, s, e):
    if src == "mount":
        with open(MOUNT / rel, "rb", buffering=0) as f:
            f.seek(s)
            return f.read(e - s)
    return fs.cat_file(f"buckets/{BUCKET}/{rel}", start=s, end=e)


def row(src, e):
    rel, i = e["shard"], e["row"]
    if rel not in meta:
        tail = min(sizes[rel], 1 << 20)
        meta[rel] = pq.ParquetFile(Ranged(sizes[rel], sizes[rel] - tail, fetch("http", rel, sizes[rel] - tail, sizes[rel]))).metadata
    md = meta[rel]
    rg = md.row_group(i)
    cols = [rg.column(k) for k in range(rg.num_columns)]
    s = min(c.dictionary_page_offset or c.data_page_offset for c in cols)
    end = max((c.dictionary_page_offset or c.data_page_offset) + c.total_compressed_size for c in cols)
    t = time.time()
    data = fetch(src, rel, s, end)
    dt = time.time() - t
    r = pq.ParquetFile(Ranged(sizes[rel], s, data), metadata=md).read_row_group(i).to_pylist()[0]
    assert r["trace_id"] == e["trace_id"]
    return len(data), dt


rng = random.Random(0)
for src in ("mount", "http"):
    for conc in (1, 8, 32, 96):
        sample = rng.sample(entries, min(len(entries), max(24, conc * 3)))
        errors, t0 = 0, time.time()
        with ThreadPoolExecutor(conc) as pool:
            res = []
            for f in [pool.submit(row, src, e) for e in sample]:
                try:
                    res.append(f.result())
                except Exception as ex:  # noqa: BLE001
                    errors += 1
                    print(f"  {src} error: {ex!r}"[:200], flush=True)
        wall = time.time() - t0
        mb = sum(b for b, _ in res) / 2**20
        lat = sorted(d for _, d in res) or [0]
        print(json.dumps({"source": src, "concurrency": conc, "traces": len(res), "errors": errors,
                          "MB": round(mb), "MB_s": round(mb / wall, 1), "traces_s": round(len(res) / wall, 1),
                          "p50_s": round(lat[len(lat) // 2], 2), "p95_s": round(lat[int(len(lat) * .95)], 2)}),
              flush=True)
