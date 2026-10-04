"""The source benchmark, read at one pinned revision so every conversion sees the same bytes."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

DATASET = "XiaomiMiMo/MiMo-V2.6-RL-oss"
REVISION = "639865fd3374018d6cb29b9fb82dd531406fcf5f"
IMAGE_REPO = "docker.io/xiaomimimo/mimo-v2.6-rl-oss"
PARQUETS = {"code": "code.parquet", "cyber": "cyber.parquet", "general": "general/train.parquet",
            "webdev": "webdev.parquet", "music": "music.parquet"}


def fetch(path: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(DATASET, path, repo_type="dataset", revision=REVISION))


@lru_cache(maxsize=1)
def load_rows() -> dict[str, dict]:
    """task id -> {"domain", "prompt", "instance", "extra"}; the same parsing as app/catalog.py rows()."""
    import pandas as pd

    out: dict[str, dict] = {}
    for domain, fname in PARQUETS.items():
        df = pd.read_parquet(fetch(fname))
        for n, r in enumerate(df.to_dict("records")):
            extra = r.get("extra_info") or {}
            inst = json.loads(extra["instance_json"]) if extra.get("instance_json") else {}
            tid = inst.get("instance_id") or f"music-{extra.get('src_id', n)}"
            if tid in out:
                raise ValueError(f"duplicate task id {tid}")
            out[tid] = {"domain": domain, "prompt": r["prompt"][-1]["content"], "instance": inst,
                        "extra": {k: _plain(v) for k, v in extra.items() if k != "instance_json"}}
    return dict(sorted(out.items()))


def _plain(v):
    """numpy scalars/arrays from parquet -> JSON-able values."""
    if hasattr(v, "tolist"):
        return v.tolist()
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    return v


def kind(row: dict) -> str:
    """Harbor dataset a row goes to: the five domains, with Terminal-Bench rows split out of General."""
    if row["domain"] == "general" and row["instance"].get("dataset_type") == "terminal_bench":
        return "terminal"
    return row["domain"]


def image_tag(row: dict) -> str | None:
    """app/catalog.py image_for, as a tag."""
    d, img = row["domain"], (row["instance"].get("docker_image") or "")
    if d == "music" or not img:
        return None
    if d == "cyber":                       # arvo-rl:v1-arvo-35858 -> arvo-v1-35858
        return "arvo-v1-" + img.rsplit("arvo-", 1)[-1]
    if d == "webdev":
        return "webdev-rl-opensource"
    return img.split(":", 1)[0]            # format-code-task-001457:latest, general-agent-env-3:oss


@lru_cache(maxsize=1)
def general_files() -> dict[str, list[dict]]:
    """General env id -> [{"path", "size", "sha256" | "git_sha1"}], from one listing of the dataset at REVISION.

    LFS files carry their sha256; small files are checked by their git blob id, so nothing has to be downloaded
    to know what the sandbox should receive."""
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFile

    out: dict[str, list[dict]] = {}
    for f in HfApi().list_repo_tree(DATASET, repo_type="dataset", path_in_repo="general/envs", recursive=True,
                                    revision=REVISION, expand=False):
        if not isinstance(f, RepoFile):
            continue
        env = f.path.split("/")[2]
        e = {"path": f.path, "size": f.size}
        if f.lfs:
            e["sha256"] = f.lfs.sha256
        else:
            e["git_sha1"] = f.blob_id
        out.setdefault(env, []).append(e)
    for v in out.values():
        v.sort(key=lambda e: e["path"])
    return dict(sorted(out.items()))
