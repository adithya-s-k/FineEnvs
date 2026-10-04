"""Harbor datasets on the Hub: which exist, where their tasks are, what each task asks and how it is graded.

A Harbor task is a folder holding `task.toml`. Datasets put those folders in different places (`tasks/<task>/`,
`tasks/<group>/<task>/`, `<task>/` at the root, `<benchmark>/tasks/<task>/`), so a task is found by where its
`task.toml` is, never by a fixed path. A dataset is indexed on first open, in the background:

    listing   one recursive listing of the repository: every task folder, its files and their sizes
    reading   one filtered download of the files an index needs (task.toml, instruction.md, tests/test.sh)
    parsing   one row per task: title, facets, environment, how it is graded

and the index is kept (STORAGE_DIR/indexes) until the dataset's revision changes.

Public datasets are read anonymously, with `token=False`: the server's own token can see its owner's private and
gated datasets, and a visitor naming one must not get it shown. A private or gated dataset is read with the
signed-in visitor's own token, and every request for it checks that token's access again before anything cached
is served. Nothing that holds an answer is shown: metadata keys like `gold_answer` are withheld by name, and
`solution/` files and data files beside the grader (rubrics, expected outputs) are listed but never sent.

Each indexed dataset also gets a pack: every task's instruction, task.toml, tests and Dockerfile as text, and
its file list. Task pages read the pack, so they never wait on the Hub; packs and indexes can live in a bucket.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import posixpath
import re
import threading
import time
import tomllib
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from . import config

# ── collections: featured environments, grouped ──────────────────────────────
# Admins edit these (app/settings.py); these are the defaults. An entry is a dataset id, or `space:<id>` for a Space.
# `color` names one of the page's validated identity colours (--c-<color> in the stylesheets).
DEFAULT_COLLECTIONS: list[dict[str, Any]] = [
    {"id": "fineenvs", "group": "FineEnvs on OpenEnv", "icon": "globe", "color": "amber", "about": "FineEnvs' live environment servers: tasks, tools, rewards",
     "ids": [f"space:FineEnvs/{n}" for n in ("geoguesser-env", "latex-ocr-env", "watercolour-env", "nayana-ocr-env", "fleurs-asr-env",
                                              "smoldataenv-multi-harness-harbor", "smoldataenv-multi-harness-whitebox", "wordle-openenv")]},
    {"id": "mimo", "group": "MiMo-V2.6 RL", "icon": "grid", "color": "webdev", "about": "Xiaomi's RL release, raw and in Harbor",
     "ids": ["XiaomiMiMo/MiMo-V2.6-RL-oss"] + [f"FineEnvs/MiMo-V2.6-RL-harbor-{d}" for d in ("code", "cyber", "general", "terminal", "webdev", "music")]},
    {"id": "terminal", "group": "Terminal-Bench", "icon": "terminal", "color": "cyber", "about": "terminal tasks, hand-written tests",
     "ids": ["harborframework/terminal-bench-2.1", "harborframework/terminal-bench-3.0", "harborframework/terminal-bench-2.0",
             "harborframework/terminal-bench-science", "IntelligenceLab/Long-Horizon-Terminal-Bench"]},
    {"id": "data", "group": "Data analysis", "icon": "table", "color": "general", "about": "questions over real data, checked answers",
     "ids": ["FineEnvs/data-agent-harbor-eval", "FineEnvs/data-agent-harbor-test", "FineEnvs/SmolDataEnvs-harbor-train",
             "AdithyaSK/dabstep-harbor", "FineEnvs/SmolDataEnvs"]},
    {"id": "swe", "group": "Software engineering", "icon": "code", "color": "code", "about": "repository tasks, graded by tests",
     "ids": ["FineEnvs/repo2rlenv-swe-flow", "FineEnvs/repo2rlenv-swe-smith", "FineEnvs/repo2rlenv-r2e", "FineEnvs/repo2rlenv-cli-gym",
             "FineEnvs/HF_ML_Tasksmith", "Lego-X/LegoFlow-SWE"]},
    {"id": "mixed", "group": "Mixed and other", "icon": "flask", "color": "music", "about": "agentic benchmarks across domains",
     "ids": ["internlm/WildClawBench-Harbor", "armin-aptura/skilltrainbench-public", "space:FineEnvs/wordle-openenv", "space:openenv/coding_env"]},
]
COLORS = ("code", "webdev", "cyber", "music", "general", "amber")
ICONS = ("grid", "terminal", "table", "code", "image", "flask", "globe", "brain", "music", "briefcase", "database", "layout", "target", "box")


def collections() -> list[dict[str, Any]]:
    """The collections, as admins last saved them, else the defaults."""
    from . import settings

    return settings.get("collections") or DEFAULT_COLLECTIONS


def collection_of(key: str) -> str | None:
    """The collection an environment is in: `key` is a dataset id or `space:<id>`."""
    return next((g["id"] for g in collections() if key in g["ids"]), None)


def featured_datasets() -> list[str]:
    return [k for g in collections() for k in g["ids"] if not k.startswith("space:")]


def collection(spec: str) -> dict[str, Any] | None:
    """The collection an environment is featured in (its id, name, icon, colour), or None."""
    cid = collection_of(spec)
    return next(({k: v for k, v in g.items() if k != "ids"} for g in collections() if g["id"] == cid), None)


def hidden() -> set[str]:
    """Environments admins took off the listings (spam, broken), by key."""
    from . import settings

    return set(settings.get("hidden", []))


# FineEnvs' flagship environment servers lead Trending until an admin saves pins of their own (an empty list included)
DEFAULT_PINNED = ["space:FineEnvs/geoguesser-env", "space:FineEnvs/latex-ocr-env"]


def pinned() -> list[str]:
    from . import settings

    return list(settings.get("pinned", DEFAULT_PINNED))


HUB_ID = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")
# Metadata that is an answer. Harbor puts no rules on `[metadata]`, and datasets keep gold answers there.
ANSWER_KEY = re.compile(r"(gold|answer|solution|expected|reference_output|ground_truth|oracle)", re.I)
CONTACT_KEY = re.compile(r"(email|e_mail|phone)", re.I)   # an author's contact details: not metadata to browse by
PACKED = re.compile(r"\.(parquet|tar|tar\.gz|tgz|zip|jsonl)$", re.I)
MAX_DEPTH = 8
# Bumped when what an index holds or how it is read changes, so older indexes are rebuilt rather than misread.
INDEX_VERSION = 13
# Benchmarks open their instructions with a canary string, there to catch training sets that leak them.
CANARY = re.compile(r"BENCHMARK DATA SHOULD NEVER APPEAR|canary GUID|canary string", re.I)
TEXT_LIMIT = 512 * 1024

# Metadata keys that mean the same thing across datasets, read in this order.
DIFFICULTY_KEYS = ("difficulty", "difficulty_tier", "difficulty_level", "task_complexity", "complexity", "level")
CATEGORY_KEYS = ("category", "domain", "task_type", "skill_type", "type", "subcategory")
TAG_KEYS = ("tags", "keywords")


def _api(token: str | None = None):
    from huggingface_hub import HfApi

    return HfApi(token=token or False)


def _who(token: str | None) -> str:
    """A token's stand-in in cache keys: never the token itself."""
    return hashlib.sha256(token.encode()).hexdigest()[:16] if token else ""


def check_spec(spec: str) -> str:
    spec = (spec or "").strip()
    if not HUB_ID.match(spec) or ".." in spec:
        raise ValueError("a Hub dataset id, like org/name")
    return spec


def _slug(spec: str) -> str:
    return spec.replace("/", "__")


# ── Hub search and dataset info ──────────────────────────────────────────────
_memo: dict[tuple, tuple[float, Any]] = {}
_big: "OrderedDict[tuple, tuple[float, Any]]" = OrderedDict()   # whole files, every row of a split: a few at a time
_memo_lock = threading.Lock()
_inflight: dict[tuple, threading.Lock] = {}
STALE_FOR = 24 * 3600   # a value whose refresh fails is served this much longer (the Hub down shouldn't take pages down)
BIG_ENTRIES = 3


def _cached(key: tuple, ttl: float, fn, big: bool = False):
    """A value kept `ttl` seconds. Concurrent misses fetch it once; a failed refresh serves the last good value (up to a
    day old) instead of failing; `big` values (whole files) are kept a few at a time, not 2,000."""
    store_ = _big if big else _memo
    now = time.time()
    with _memo_lock:
        hit = store_.get(key)
        if hit and now - hit[0] < ttl:
            return hit[1]
        lock = _inflight.setdefault(key, threading.Lock())
    with lock:
        with _memo_lock:   # someone else may have fetched it while we waited
            hit = store_.get(key)
            if hit and time.time() - hit[0] < ttl:
                return hit[1]
        try:
            value = fn()
        except Exception:
            if hit and time.time() - hit[0] < STALE_FOR:
                return hit[1]
            raise
        finally:
            with _memo_lock:
                _inflight.pop(key, None)
        with _memo_lock:
            store_[key] = (time.time(), value)
            if big:
                _big.move_to_end(key)
                while len(_big) > BIG_ENTRIES:
                    _big.popitem(last=False)
            elif len(_memo) > 2000:
                for k in sorted(_memo, key=lambda k: _memo[k][0])[:500]:
                    _memo.pop(k, None)
        return value


# What a dataset is, as the Hub's own badges say it.
BADGES = (("rl-environment", "RL Environment"), ("benchmark", "Benchmark"))
SORTS = {"trending": "trending_score", "downloads": "downloads", "updated": "last_modified", "likes": "likes", "new": "created_at"}
_EXPAND = ["downloads", "likes", "lastModified", "tags", "trendingScore", "gated", "private", "createdAt", "description"]


# card paragraphs that say where a dataset comes from rather than what it is
_NOT_A_BRIEF = re.compile(r"^(\(|warning|note\b|disclaimer|⚠|important)|mirror|primary source|open issues|pull requests|leaderboard above|"
                          r"how to run|https?://", re.I)


# a card's first line that names a section, not the dataset ("Dataset Description:", "Overview")
_GENERIC_HEADING = re.compile(r"^(dataset\s+(card|description|summary|overview|details|information)|description|overview|summary|"
                              r"introduction|about(\s+this\s+dataset)?|abstract|readme|contents?|table\s+of\s+contents)\s*:?$", re.I)


def _card_text(text: str | None) -> tuple[str | None, str]:
    """The dataset card's heading and its first paragraph that describes the dataset, from the Hub's plain-text
    `description` (the card's text, cut short by the Hub)."""
    text = re.sub(r"…\s*See the full description on the dataset page:.*$", "…", text or "", flags=re.S)
    paras = []
    for block in re.split(r"\n[ \t]*\n", text):
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        while lines and len(lines[0]) < 40 and ("·" in lines[0] or "|" in lines[0]):   # "Paper · Blog · Leaderboard"
            lines.pop(0)
        if lines:
            paras.append(re.sub(r"\s+", " ", " ".join(lines)))
    heading = paras[0] if paras and len(paras[0]) <= 80 else None
    named = re.sub(r"^dataset card for\s+", "", heading or "", flags=re.I).strip()
    if heading and (not named or _GENERIC_HEADING.match(named) or named.endswith(":")):
        named = ""
    brief = next((p for p in paras[1 if heading else 0:] if len(p) >= 60 and not _NOT_A_BRIEF.search(p) and not CANARY.search(p)), "")
    return named or None, (brief if len(brief) <= 280 else brief[:279].rsplit(" ", 1)[0] + "…")


def framework_of(tags: list[str]) -> str:
    """Which kind of environment a dataset is, from its Hub tags (the Hub's own convention: `rl-environment` plus a
    framework tag): Harbor task folders, NeMo Gym or Verifiers data, OpenEnv data, verl rows, or other RL rows. A
    dataset whose index finds Harbor task folders is Harbor whatever its tags say (environments() corrects it)."""
    t = set(tags)
    if t & {"harbor", "library:harbor"}:
        return "harbor"
    if t & {"nemo-gym", "library:nemo-gym"}:
        return "nemo-gym"
    if t & {"verifiers", "library:verifiers"}:
        return "verifiers"
    if t & {"openenv", "library:openenv"}:
        return "openenv-data"
    if t & {"verl", "skyrl"}:
        return "verl"
    return "rows"


def _summary(d: Any) -> dict[str, Any]:
    raw = d.tags or []
    tags = [t for t in raw if ":" not in t]
    heading, brief = _card_text(getattr(d, "description", None))
    return {"id": d.id, "key": d.id, "kind": "dataset", "collection": collection_of(d.id), "heading": heading, "brief": brief,
            "downloads": d.downloads or 0, "likes": d.likes or 0,
            "updated": d.last_modified.isoformat() if getattr(d, "last_modified", None) else None,
            "created": d.created_at.isoformat() if getattr(d, "created_at", None) else None,
            "trending": getattr(d, "trending_score", None) or 0,
            "badges": [label for tag, label in BADGES if tag in raw],
            "tags": [t for t in tags if t not in ("harbor",) + tuple(tag for tag, _ in BADGES)][:12],
            "gated": bool(d.gated), "private": bool(d.private), "framework": framework_of(raw)}


def _listing_fetch(full: bool) -> list[dict[str, Any]]:
    found = _api().list_datasets(filter="harbor", sort="trending_score", limit=None if full else 2000, expand=_EXPAND)
    rows = {d.id: _summary(d) for d in found if not d.private and not d.gated}
    # other RL environment formats: rows a processor reads (app/envs)
    for tag in ("rl-environment", "library:verifiers", "library:nemo-gym", "verifiers", "nemo-gym", "library:openenv", "verl", "skyrl"):
        try:
            for d in _api().list_datasets(filter=tag, sort="trending_score", limit=None if full else 1000, expand=_EXPAND):
                if d.id not in rows and not d.private and not d.gated:
                    rows[d.id] = _summary(d)
        except Exception:  # noqa: BLE001 - one tag failing leaves the rest
            continue
    missing = [s for s in featured_datasets() if s not in rows]
    with ThreadPoolExecutor(max_workers=12) as pool:   # one Hub call each: in parallel
        for spec, row in zip(missing, pool.map(_info_or_none, missing)):
            if row:
                rows[spec] = row
    for sp in spaces(full=full):
        rows[sp["key"]] = sp
    return list(rows.values())


LISTING_TTL = 900   # seconds a listing is served before it's refreshed (in the background)
# On the Spaces the indexer Job (app/indexer.py) writes listing.json.gz every hour: re-read it when it changes and
# don't fetch the Hub's listing in every process, unless it's older than LISTING_STALE (the Job stopped).
LISTING_FROM_STORE = os.environ.get("RLX_LISTING_FROM_STORE", "0") == "1"
LISTING_STALE = 3 * 3600
_env_list: dict[str, Any] = {"rows": None, "at": 0.0, "full": False, "refreshing": False, "mtime": None, "looked": 0.0}
_listing_lock = threading.Lock()


def _listing_path() -> Path:
    return config.STORAGE_DIR / "listing.json.gz"


def _listing_rows() -> list[dict[str, Any]]:
    """Every environment on the Hub, served at once: from memory, else from the last listing written to the store
    (so a restart doesn't wait on the Hub), else a quick listing (the 1,000 most trending per tag); a full one (every
    Space with its files, minutes) refreshes it in the background whenever it's older than LISTING_TTL."""
    with _listing_lock:
        rows, at = _env_list["rows"], _env_list["at"]
    if LISTING_FROM_STORE:
        if time.time() - _env_list["looked"] > 30:   # a stat every 30 s, a read when the file changed
            _env_list["looked"] = time.time()
            try:
                mtime = _listing_path().stat().st_mtime
                if mtime != _env_list["mtime"]:
                    doc = json.loads(gzip.decompress(_listing_path().read_bytes()))
                    rows, at = doc["rows"], doc["at"]
                    with _listing_lock:
                        _env_list.update(rows=rows, at=at, full=doc.get("full", False), mtime=mtime)
            except (OSError, ValueError, KeyError):
                pass
        if rows is not None and time.time() - at < LISTING_STALE:
            return rows   # the Job's listing: no fetch here
    if rows is None:
        try:
            doc = json.loads(gzip.decompress(_listing_path().read_bytes()))
            rows, at = doc["rows"], doc["at"]
            with _listing_lock:
                _env_list.update(rows=rows, at=at, full=doc.get("full", False))
        except (OSError, ValueError, KeyError):
            rows = None
    if rows is None:
        rows = _listing_fetch(full=False)
        at = time.time()
        with _listing_lock:
            _env_list.update(rows=rows, at=at, full=False)
        _refresh_listing(force=True)
    elif time.time() - at > LISTING_TTL or not _env_list.get("full"):
        _refresh_listing()
    return rows


def _refresh_listing(force: bool = False) -> None:
    with _listing_lock:
        if _env_list["refreshing"] or (not force and time.time() - _env_list["at"] < LISTING_TTL and _env_list.get("full")):
            return
        _env_list["refreshing"] = True

    def run() -> None:
        try:
            rows = _listing_fetch(full=True)
            if rows:
                with _listing_lock:
                    _env_list.update(rows=rows, at=time.time(), full=True)
                atomic_write(_listing_path(), gzip.compress(json.dumps({"at": time.time(), "full": True, "rows": rows}, default=str).encode()))
        except Exception:  # noqa: BLE001 - the last listing stays; the next request tries again
            pass
        finally:
            with _listing_lock:
                _env_list["refreshing"] = False

    threading.Thread(target=run, daemon=True, name="listing").start()


def environments(include_hidden: bool = False) -> list[dict[str, Any]]:
    """Every public RL environment on the Hub: datasets tagged `harbor` or as RL environments of other formats,
    environment Spaces (OpenEnv, ORS, others), and every featured one, with its counters and, for an indexed dataset,
    its task count. The page sorts and filters these itself."""
    gone, pins = (set() if include_hidden else hidden()), pinned()
    out = []
    for r in _listing_rows():
        if r["key"] in gone:
            continue
        indexed = headline(r["id"]) if r["kind"] == "dataset" else None
        fw = "harbor" if indexed and indexed.get("tasks") else r.get("framework")   # its index found task folders
        coll = collection_of(r["key"])
        if r["kind"] == "space" and coll == "fineenvs" and fw != "openenv":   # curated as OpenEnv servers, manifest or not
            r = {**r, "openenv": True, "badges": ["OpenEnv"]}
            fw = "openenv"
        out.append({**r, "framework": fw, "collection": coll, "pinned": r["key"] in pins, "indexed": indexed})
    return out


# ── OpenEnv and other environment Spaces ─────────────────────────────────────
_SPACE_EXPAND = ["likes", "trendingScore", "sdk", "runtime", "tags", "lastModified", "createdAt", "cardData", "private"]


def _space_summary(s: Any) -> dict[str, Any]:
    card = (s.card_data.to_dict() if s.card_data else {}) or {}
    raw = s.tags or []
    stage = getattr(s.runtime, "stage", None) if s.runtime is not None else None
    version = next((t[8:] for t in raw if re.match(r"^openenv-\d", t)), None)
    files = [f.rfilename for f in (getattr(s, "siblings", None) or [])]
    manifest = min((f for f in files if _MANIFEST.match(f) and not _VENDORED.match(f)), key=lambda f: f.count("/"), default=None)
    openenv = "openenv" in raw or bool(version) or manifest is not None
    ors = bool({"ors", "openreward"} & set(raw))
    return {"id": s.id, "key": f"space:{s.id}", "kind": "space", "openenv": openenv, "manifest": manifest,
            "framework": "openenv" if openenv else "ors" if ors else "space", "files": len(files) or None,
            "openenv_version": version, "mcp": any(t in ("mcp-server", "mcp") for t in raw),
            "hardware": getattr(s.runtime, "hardware", None) if s.runtime is not None else None,
            "heading": card.get("title"), "brief": (card.get("short_description") or "")[:280],
            "downloads": 0, "likes": s.likes or 0, "trending": getattr(s, "trending_score", None) or 0,
            "updated": s.last_modified.isoformat() if getattr(s, "last_modified", None) else None,
            "created": s.created_at.isoformat() if getattr(s, "created_at", None) else None,
            "sdk": s.sdk, "stage": str(stage) if stage else None,
            "badges": ["OpenEnv"] if openenv else ["ORS"] if ors else [],
            "tags": [t for t in raw if ":" not in t and t not in ("docker", "openenv", "rl-environment", "mcp-server", "ors", "openreward") and not t.startswith("openenv-")][:12],
            "private": bool(s.private), "gated": False}


SPACE_TAGS = ("openenv", "rl-environment", "ors", "openreward")
# a manifest a Space's own environment declares (not one vendored from OpenEnv's source, or its templates)
_MANIFEST = re.compile(r"^(?:[^/]+/)?openenv\.ya?ml$")
_VENDORED = re.compile(r"^(src/(openenv|core)/|build/|.*\.egg-info/|envs/[^/]+/)")


def spaces(full: bool = True) -> list[dict[str, Any]]:
    """Environment Spaces: OpenEnv servers, ORS servers and others tagged as RL environments (Docker Spaces, not
    articles or demos). `full` lists every one with its files (thousands; a minute or two, so done in the background)
    and tells OpenEnv apart by its manifest; otherwise the most trending 1,000 per tag, by tags alone."""
    featured = [k[6:] for g in collections() for k in g["ids"] if k.startswith("space:")]
    rows: dict[str, dict[str, Any]] = {}
    expand = _SPACE_EXPAND + (["siblings"] if full else [])
    for tag in SPACE_TAGS:
        try:
            for sp in _api().list_spaces(filter=tag, sort="trending_score", limit=None if full else 1000, expand=expand):
                tags = sp.tags or []
                if sp.private or sp.sdk != "docker" or "research-article-template" in tags:
                    continue
                rows.setdefault(sp.id, _space_summary(sp))
        except Exception:  # noqa: BLE001 - Spaces failing leaves the datasets
            continue
    for sid in featured:
        if sid not in rows:
            try:
                rows[sid] = _space_summary(_api().space_info(sid, expand=expand))   # with its files: its manifest
            except Exception:  # noqa: BLE001
                continue
    return list(rows.values())


def space(spec: str) -> dict[str, Any]:
    """A Space's page: its card and its README. What its server offers (and its stage now) is spaces_live.probe."""

    spec = check_spec(spec)

    def fetch():
        sp = _api().space_info(spec, expand=_SPACE_EXPAND + ["subdomain", "siblings"])   # its files: its openenv.yaml
        if sp.private:
            raise PermissionError("this Space is private")
        row = _space_summary(sp)
        readme = ""
        try:
            from huggingface_hub import hf_hub_download

            readme = _read(Path(hf_hub_download(spec, "README.md", repo_type="space", token=False,
                                                local_dir=str(config.CACHE_DIR / "spaces" / _slug(spec)))), 256 * 1024)
            readme = re.sub(r"\A---\n.*?\n---\n", "", readme, flags=re.S)   # the YAML header
        except Exception:  # noqa: BLE001
            pass
        sub = getattr(sp, "subdomain", None) or ""
        host = f"https://{sub}.hf.space" if sub else ""
        if not re.match(r"^https://[a-z0-9-]+\.hf\.space$", host or ""):   # only ever call the Space's own hf.space host
            host = ""
        server: dict[str, Any] = {}   # what the server offers comes live from /api/spaces/<id>/live (spaces_live.probe)
        return {**row, "readme": readme[:200_000], "host": host, "server": server, "collection": collection_of(row["key"]),
                "license": (sp.card_data.to_dict() if sp.card_data else {}).get("license")}

    return _cached(("space", spec), 120, fetch)


def _info_or_none(spec: str) -> dict[str, Any] | None:
    try:
        return info(spec)
    except Exception:  # noqa: BLE001 - a deleted or gated dataset drops out, it doesn't break the page
        return None


_INFO_EXPAND = ["sha", "downloads", "likes", "lastModified", "createdAt", "tags", "cardData", "gated", "private", "trendingScore",
                "description"]


def _fetch_info(spec: str, token: str | None) -> dict[str, Any]:
    # only these fields: by default the Hub also returns every file, seconds for a big dataset
    d = _api(token).dataset_info(spec, expand=_INFO_EXPAND)
    restricted = bool(d.private or d.gated)
    if restricted and not token:
        raise PermissionError("this dataset is private or gated: sign in with an account that can read it")
    card = (d.card_data.to_dict() if d.card_data else {}) or {}
    return {**_summary(d), "sha": d.sha, "pretty_name": card.get("pretty_name"), "license": card.get("license"),
            "harbor": "harbor" in (d.tags or []), "restricted": restricted, "all_tags": list(d.tags or [])}


def info(spec: str, token: str | None = None) -> dict[str, Any]:
    """A dataset's revision, card fields and counters. A public one is read anonymously (and cached for everyone);
    a private or gated one only with a token that can read it, and cached per token. Raises otherwise."""
    from huggingface_hub.errors import RepositoryNotFoundError

    spec = check_spec(spec)
    try:
        return _cached(("info", spec), 300, lambda: _fetch_info(spec, None))
    except (RepositoryNotFoundError, PermissionError):
        if not token:
            raise
    return _cached(("info", spec, _who(token)), 120, lambda: _fetch_info(spec, token))


def mine(token: str) -> list[dict[str, Any]]:
    """The visitor's own Harbor datasets and their organizations', private ones included, read with their token."""
    from huggingface_hub import whoami

    def fetch():
        me = whoami(token=token)
        owners = [me["name"]] + [o["name"] for o in (me.get("orgs") or [])][:10]

        def one(owner: str) -> list[dict[str, Any]]:
            try:
                return [_summary(d) for d in _api(token).list_datasets(author=owner, filter="harbor", limit=300, expand=_EXPAND)]
            except Exception:  # noqa: BLE001 - one organization failing leaves the rest
                return []

        with ThreadPoolExecutor(max_workers=6) as pool:
            rows = {r["id"]: r for batch in pool.map(one, owners) for r in batch}
        return list(rows.values())

    return [{**r, "mine": True, "indexed": headline(r["id"])} for r in _cached(("mine", _who(token)), 120, fetch)]


def random_task(spec: str | None = None, token: str | None = None) -> dict[str, str]:
    """A task to open: from `spec`, or from a featured dataset already indexed (each dataset equally likely, so the
    biggest doesn't crowd out the rest)."""
    import random

    if spec:
        info(spec, token)   # may this visitor read it?
    pool = [check_spec(spec)] if spec else featured_datasets()
    random.shuffle(pool)
    for s in pool:
        idx = _read_index(s)
        if idx and idx.get("tasks"):
            return {"dataset": s, "path": random.choice(idx["tasks"])["path"]}
    raise LookupError("no indexed tasks yet" if not spec else "this dataset isn't indexed yet")


def warm() -> None:
    """Index the featured datasets in the background, so their task counts are ready. One from each collection in
    turn, so every collection has some early rather than the first all done before the second."""
    from itertools import chain, zip_longest

    try:
        environments()   # the home page's one fetch, ready before the first visitor asks
    except Exception:  # noqa: BLE001 - the page fetches it again
        pass
    order = [s for s in chain.from_iterable(zip_longest(*([k for k in g["ids"] if not k.startswith("space:")] for g in collections()))) if s]
    for spec in order:
        try:
            meta = info(spec)
            if _read_index(spec, meta["sha"]):
                continue
            job = {"state": "listing", "done": 0, "total": 0, "sha": meta["sha"], "at": time.time()}
            with _jobs_lock:
                if spec in _jobs and _jobs[spec].get("state") not in ("done", "error"):
                    continue
                _jobs[spec] = job
            _build(spec, meta, job)
        except Exception:  # noqa: BLE001 - warming is best effort; opening the dataset retries
            continue


# ── indexing ─────────────────────────────────────────────────────────────────
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def atomic_write(p: Path, data: bytes) -> None:
    """Write a file so a reader sees the old one or the new one, never half of either, even with two processes (the
    explorer, its admin, an indexer job) writing it: a temp name of this process's own, then a rename."""
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_bytes(data)
    tmp.replace(p)


def _index_path(spec: str) -> Path:
    return config.INDEX_DIR / f"{_slug(spec)}.json.gz"


_indexes: "OrderedDict[tuple, tuple[int, dict]]" = OrderedDict()   # (path, mtime) -> (bytes, parsed index)
_indexes_lock = threading.Lock()
INDEX_CACHE_BYTES = int(os.environ.get("RLX_INDEX_CACHE_MB", 400)) * 1024 * 1024


def _read_index(spec: str, sha: str | None = None) -> dict[str, Any] | None:
    """A dataset's index, parsed once per version of its file (a big one takes a quarter second to parse), within a
    memory budget. Callers must not change what it returns."""
    p = _index_path(spec)
    try:
        key = (str(p), p.stat().st_mtime)
    except OSError:
        return None
    with _indexes_lock:
        hit = _indexes.get(key)
        if hit:
            _indexes.move_to_end(key)
    if hit:
        idx = hit[1]
    else:
        try:
            raw = gzip.decompress(p.read_bytes())
            idx = json.loads(raw)
        except (OSError, ValueError):
            return None
        with _indexes_lock:
            for k in [k for k in _indexes if k[0] == key[0]]:   # an older version of the same file
                _indexes.pop(k, None)
            _indexes[key] = (len(raw), idx)
            while len(_indexes) > 1 and sum(n for n, _ in _indexes.values()) > INDEX_CACHE_BYTES:
                _indexes.popitem(last=False)
    if (sha and idx.get("sha") != sha) or idx.get("version") != INDEX_VERSION:
        return None
    return idx


def _write_index(idx: dict[str, Any]) -> None:
    atomic_write(_index_path(idx["spec"]), gzip.compress(json.dumps(idx, separators=(",", ":")).encode()))
    # a few numbers beside it, cheap to read for every card in a listing
    atomic_write(_head_path(idx["spec"]), json.dumps({**_headline(idx), "sha": idx["sha"], "version": idx.get("version")}).encode())
    _heads.pop(idx["spec"], None)


def _head_path(spec: str) -> Path:
    return config.INDEX_DIR / f"{_slug(spec)}.head.json"


_heads: dict[str, tuple[float, dict[str, Any] | None]] = {}


def headline(spec: str) -> dict[str, Any] | None:
    """What a listing shows of an indexed dataset (its task count and how it is graded), or None. Re-read from the
    store at most once a minute per dataset: a listing asks for hundreds."""
    hit = _heads.get(spec)
    if hit and time.time() - hit[0] < 60:
        return hit[1]
    try:
        h = json.loads(_head_path(spec).read_text())
        h = h if h.get("version") == INDEX_VERSION else None
    except (OSError, ValueError):
        h = None
    _heads[spec] = (time.time(), h)
    return h


def _headline(idx: dict[str, Any]) -> dict[str, Any]:
    s = idx.get("summary") or {}
    return {"tasks": len(idx.get("tasks") or []), "packed": idx.get("packed"), "graded": s.get("verifier"),
            "image": s.get("image"), "built": idx.get("built")}


def index_status(spec: str, token: str | None = None) -> dict[str, Any]:
    """The index if it is current, else the indexing job's progress (starting one if none runs)."""
    spec = check_spec(spec)
    meta = info(spec, token)
    idx = _read_index(spec, meta["sha"])
    if idx and not idx.get("partial"):
        return {"state": "done", "index": idx}
    if idx:   # the tasks are known, their files are being read: the page opens now and refreshes when it's whole
        with _jobs_lock:
            job = _jobs.get(spec)
            alive = job and job.get("sha") == meta["sha"] and job["state"] not in ("done", "error")
            running = sum(1 for j in _jobs.values() if j.get("state") not in ("done", "error"))
            if not alive and running < config.MAX_INDEX_JOBS and not (job and job["state"] == "error" and time.time() - job.get("at", 0) < 300):
                job = {"state": "listing", "done": 0, "total": 0, "sha": meta["sha"], "at": time.time(), "t": {"listing": time.time()}}
                _jobs[spec] = job
                threading.Thread(target=_build, args=(spec, meta, job, token if meta.get("restricted") else None), daemon=True,
                                 name=f"index-{spec}").start()
        prog = {k: job.get(k) for k in ("state", "done", "total", "error")} if job else None
        return {"state": "done", "index": idx, "progress": prog}
    with _jobs_lock:
        job = _jobs.get(spec)
        if job and job.get("sha") == meta["sha"] and job["state"] not in ("error",):
            return {**{k: v for k, v in job.items() if k != "sha"}, "now": time.time()}
        if job and job["state"] == "error" and time.time() - job.get("at", 0) < 60:
            return {k: v for k, v in job.items() if k != "sha"}
        running = sum(1 for j in _jobs.values() if j.get("state") not in ("done", "error"))
        if running >= config.MAX_INDEX_JOBS:
            raise RuntimeError(f"{running} datasets are being indexed right now; try again in a minute")
        job = {"state": "listing", "done": 0, "total": 0, "sha": meta["sha"], "at": time.time(), "t": {"listing": time.time()}}
        _jobs[spec] = job
    threading.Thread(target=_build, args=(spec, meta, job, token if meta.get("restricted") else None), daemon=True,
                     name=f"index-{spec}").start()
    return {**{k: v for k, v in job.items() if k != "sha"}, "now": time.time()}


def _build(spec: str, meta: dict[str, Any], job: dict[str, Any], token: str | None = None) -> None:
    try:
        idx, pack = build_index(spec, meta, job, token, on_partial=lambda early: (_write_index(early), job.update(partial=True)))
        job.setdefault("t", {})["saving"] = time.time()
        job["state"] = "saving"
        if pack is not None:
            _write_pack(spec, meta["sha"], pack)
        _write_index(idx)
        job.update(state="done")
    except Exception as exc:  # noqa: BLE001 - reported to the page
        job.update(state="error", error=f"{type(exc).__name__}: {str(exc)[:300]}", at=time.time())


def _task_dirs(paths: list[str]) -> list[str]:
    """Folders holding a `task.toml`, shallowest first; a task folder inside another task's folder is not a task."""
    found = sorted({posixpath.dirname(p) for p in paths if posixpath.basename(p) == "task.toml"
                    and p.count("/") <= MAX_DEPTH and not any(seg.startswith(".") for seg in p.split("/"))},
                   key=lambda d: (d.count("/"), d))
    kept: list[str] = []
    for d in found:
        if not any(d == k or d.startswith(k + "/") for k in kept if k):
            kept.append(d)
    return sorted(kept)


# what a task's page shows, fetched for every task when the dataset is indexed
CORE_FILES = ("task.toml", "instruction.md", "tests/test.sh", "environment/Dockerfile", "tests/Dockerfile")
PACK_BUDGET = 512 * 1024 * 1024   # bytes to fetch for a dataset's pack before falling back to CORE_FILES
SMALL = 64 * 1024
# a multi-step task's own files: steps/<name>/instruction.md, tests/, solution/, workdir/ (Harbor's TaskPaths)
STEP_FILE = re.compile(r"^steps/([^/]+)/(.+)$")
MAX_STEPS = 20


def _in_step(rel: str) -> str:
    """A step's file as the task-level file it stands for (`steps/s1/tests/x.py` is `tests/x.py`), else `rel`."""
    m = STEP_FILE.match(rel)
    return m.group(2) if m else rel


def _wanted(rel: str, size: int) -> bool:
    if rel in ("task.toml", "instruction.md") or (STEP_FILE.match(rel) and _in_step(rel) == "instruction.md"):
        return size <= TEXT_LIMIT
    if size > SMALL or _binary_name(rel):
        return False
    return (rel.startswith("tests/") or (rel.startswith("steps/") and _in_step(rel).startswith("tests/"))
            or bool(re.match(r"environment/(Dockerfile|(docker-)?compose\.ya?ml)$", rel)))


def _pick(files: dict[str, int], budget_ok: bool = True) -> list[str]:
    """A task's files worth keeping as text: its toml and instruction, the verifier's scripts, the environment, and
    each step's instruction and tests (a few per step)."""
    picked = [r for r, n in files.items() if _wanted(r, n) and (budget_ok or r in CORE_FILES or _in_step(r) in ("instruction.md", "tests/test.sh"))]
    tests = sorted((r for r in picked if r.startswith("tests/")), key=lambda r: (r != "tests/test.sh", r.count("/"), r))[:24]
    steps: dict[str, list[str]] = {}
    for r in sorted((r for r in picked if r.startswith("steps/")), key=lambda r: (_in_step(r) not in ("instruction.md", "tests/test.sh"), r.count("/"), r)):
        own = steps.setdefault(STEP_FILE.match(r).group(1), [])
        if len(own) < 7 and (len(steps) <= MAX_STEPS or own):
            own.append(r)
    return [r for r in picked if not r.startswith(("tests/", "steps/"))] + tests + [r for own in steps.values() for r in own]


def build_index(spec: str, meta: dict[str, Any], job: dict[str, Any] | None = None,
                token: str | None = None, on_partial=None) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """The dataset's index (one row per task) and its pack (every task's text and file list). A dataset walked
    folder by folder gets a first index as soon as its tasks are found (`on_partial`), with their names only, so
    its page opens while the tasks' files are read."""
    from huggingface_hub.hf_api import RepoFile

    job = job if job is not None else {}
    job.setdefault("t", {})["listing"] = time.time()
    lister = token or (config.INDEX_TOKEN if not meta.get("restricted") else None)
    files, walked = _listing(spec, meta["sha"], lister, job)
    dirs = walked["tasks"] if walked else _task_dirs(list(files))
    job["tasks"] = len(dirs)
    base = {"spec": spec, "sha": meta["sha"], "built": time.time(), "info": meta, "version": INDEX_VERSION}
    if not dirs:
        packed = sorted(p for p in files if PACKED.search(p))[:20]
        return {**base, "tasks": [], "packed": packed or None, "summary": {},
                "note": "Tasks are packed into archives or tables, which this explorer can't open yet."
                if packed else "No task.toml anywhere in this repository: it isn't in Harbor's format."}, None
    if len(dirs) > config.MAX_INDEX_TASKS:
        raise ValueError(f"{len(dirs):,} tasks: more than the {config.MAX_INDEX_TASKS:,} this explorer indexes")

    # every file under each task folder, in one pass over the listing
    if walked:
        per_task = walked["files"]
    else:
        per_task = {d: {} for d in dirs}
        for path, size in files.items():
            parent = posixpath.dirname(path)
            while True:   # up the folders until one is a task's (a few steps, not a scan of every task)
                if parent in per_task:
                    per_task[parent][path[len(parent) + 1:] if parent else path] = size
                    break
                if not parent:
                    break
                parent = posixpath.dirname(parent)
    if walked and on_partial:
        early = [_row(i, d, Path("/nonexistent"), per_task[d], [], {})[0] for i, d in enumerate(dirs)]
        for r in early:
            r.update(title=r["name"].replace("__", " · ").replace("_", " ").replace("-", " ").strip() or r["name"], pending=True, invalid=False)
        _group(early)
        on_partial({**base, "tasks": early, "summary": summarize(early), "packed": None, "walked": True, "partial": True})
    picks = {d: _pick(per_task[d]) for d in dirs}
    if sum(per_task[d][r] for d in dirs for r in picks[d]) > PACK_BUDGET:
        picks = {d: _pick(per_task[d], budget_ok=False) for d in dirs}
    paths = [f"{d}/{r}" if d else r for d in dirs for r in picks[d]]
    job["t"]["reading"] = time.time()
    job.update(state="reading", done=0, total=len(paths))
    paths.sort(key=lambda p: (posixpath.basename(p) not in ("task.toml", "instruction.md"), p))   # what a row needs most, first
    local = config.CACHE_DIR / "index-files" / f"{_slug(spec)}@{meta['sha'][:12]}"
    # a public dataset may be fetched with the operator's own token (higher rate limits), never a private one
    _download(spec, meta["sha"], token or (config.INDEX_TOKEN if not meta.get("restricted") else None), paths, local, job)
    job["t"]["parsing"] = time.time()
    job.update(state="parsing", done=0, total=len(dirs))
    with ThreadPoolExecutor(max_workers=config.INDEX_WORKERS) as pool:
        out = list(pool.map(lambda pair: _row(pair[0], pair[1], local, per_task[pair[1]], picks[pair[1]], job), enumerate(dirs)))
    rows = [r for r, _ in out]
    _group(rows)
    _briefs(rows)
    # texts by content: the grader's helpers are often the same file in every task, kept once
    blobs: dict[str, str] = {}
    tasks: dict[str, Any] = {}
    for d, (_, texts) in zip(dirs, out):
        refs = {}
        for rel, text in texts.items():
            h = hashlib.sha1(text.encode()).hexdigest()[:16]
            blobs.setdefault(h, text)
            refs[rel] = h
        tasks[d] = {"files": dict(sorted(per_task[d].items())[:2000]), "more": max(0, len(per_task[d]) - 2000), "texts": refs,
                    **({"lazy": walked["lazy"].get(d, [])} if walked else {})}
    return ({**base, "tasks": rows, "summary": summarize(rows), "packed": None, **({"walked": True} if walked else {})},
            {"blobs": blobs, "tasks": tasks})


def looks_harbor(spec: str, sha: str, token: str | None = None) -> bool:
    """Whether a dataset not tagged `harbor` is laid out as Harbor tasks anyway (PrimeIntellect/Terminal-Lego-15k is
    tagged `verifiers`): a task.toml one, two or three folders down, looking at the first few folders of each level.
    At most eight Hub calls, cached."""
    from huggingface_hub.hf_api import RepoFile, RepoFolder

    def folders(path: str | None, n: int) -> list[str]:
        out = []
        for e in api.list_repo_tree(spec, repo_type="dataset", revision=sha, path_in_repo=path):
            if isinstance(e, RepoFolder) and not e.path.rsplit("/", 1)[-1].startswith("."):
                out.append(e.path)
                if len(out) >= n:
                    break
        return out

    def found(cand: list[str]) -> bool:
        return any(isinstance(e, RepoFile) for e in api.get_paths_info(spec, cand[:500], repo_type="dataset", revision=sha))

    def check() -> bool:
        try:
            level = folders(None, 40)
            if found(["task.toml"] + [f"{d}/task.toml" for d in level]):
                return True
            for depth in (2, 3):   # tasks/<task>/, then tasks/<split>/<task>/
                level = [c for d in level[:3] for c in folders(d, 10)]
                if not level:
                    return False
                if found([f"{d}/task.toml" for d in level]):
                    return True
            return False
        except Exception:  # noqa: BLE001 - can't tell: read it as rows
            return False

    api = _api(token)

    return _cached(("looks-harbor", spec, sha), 3600, check)


LIST_BUDGET = float(os.environ.get("RLX_LIST_BUDGET", 25))   # seconds for one recursive listing before walking instead
# a task's files read for the index when it is walked folder by folder: checked in batches, never listed
KNOWN_FILES = ("task.toml", "instruction.md", "README.md", "tests/test.sh", "environment/Dockerfile",
               "environment/docker-compose.yaml", "environment/docker-compose.yml", "environment/compose.yaml", "solution/solve.sh",
               "tests/Dockerfile", "trajectory.json")   # a separate grader's build; a prior conversation


class _TooBig(Exception):
    pass


def _listing(spec: str, sha: str, token: str | None, job: dict[str, Any]) -> tuple[dict[str, int], dict[str, Any] | None]:
    """Every file, from one recursive listing; or, when that is too slow or too big (tasks that each ship a whole
    app or repository), the tasks found folder by folder (see _walk). Returns (files, None) or ({}, walked)."""
    from huggingface_hub.hf_api import RepoFile

    files: dict[str, int] = {}
    t0 = time.time()
    try:
        for n, entry in enumerate(_api(token).list_repo_tree(spec, repo_type="dataset", revision=sha, recursive=True)):
            if isinstance(entry, RepoFile):
                files[entry.path] = entry.size
            if n % 500 == 0:
                job["files"] = len(files)
                if time.time() - t0 > LIST_BUDGET or n > config.MAX_LISTING_FILES:
                    raise _TooBig
    except _TooBig:
        job.update(files=0, mode="folders")
        return {}, _walk(spec, sha, token, job)
    job["files"] = len(files)
    return files, None


def _walk(spec: str, sha: str, token: str | None, job: dict[str, Any]) -> dict[str, Any]:
    """Find the tasks folder by folder: list a level, ask which of its folders hold a task.toml (one batched call per
    thousand folders), descend into the rest. Inside a task only the files the index reads are looked up (batched
    too); the task's folder is listed when its page opens, and each subfolder when someone opens it (`task_folder`)."""
    from huggingface_hub.hf_api import RepoFile, RepoFolder

    api = _api(token)

    def ls(path: str) -> list[Any]:
        return list(api.list_repo_tree(spec, repo_type="dataset", revision=sha, path_in_repo=path or None, recursive=False))

    def info_of(paths: list[str]) -> list[Any]:
        out: list[Any] = []
        chunks = [paths[i:i + 500] for i in range(0, len(paths), 500)]   # the Hub takes up to ~1,000 form fields a call
        with ThreadPoolExecutor(max_workers=4) as pool:
            for got in pool.map(lambda c: api.get_paths_info(spec, c, repo_type="dataset", revision=sha), chunks):
                out += got
        return out

    tasks: list[str] = []
    found: dict[str, int] = {}
    root = ls("")
    calls, depth = 1, 0
    if any(isinstance(e, RepoFile) and e.path == "task.toml" for e in root):
        tasks, found, frontier = [""], {"task.toml": next(e.size for e in root if e.path == "task.toml")}, []
    else:
        frontier = [e.path for e in root if isinstance(e, RepoFolder) and not posixpath.basename(e.path).startswith(".")]
    while frontier and depth < MAX_DEPTH:
        hits = {}
        for e in info_of([f"{d}/task.toml" for d in frontier]):
            if isinstance(e, RepoFile):
                hits[posixpath.dirname(e.path)] = e.size
        calls += (len(frontier) + 499) // 500
        tasks += sorted(hits)
        found.update({f"{d}/task.toml": n for d, n in hits.items()})
        if len(tasks) > config.MAX_INDEX_TASKS:
            raise ValueError(f"over {config.MAX_INDEX_TASKS:,} tasks: more than this explorer indexes")
        rest = [d for d in frontier if d not in hits]
        if len(rest) > 3000:
            raise ValueError(f"{len(rest):,} folders without a task.toml at one level: not a Harbor layout this explorer can walk")
        job.update(tasks=len(tasks), files=calls)
        nxt: list[str] = []
        with ThreadPoolExecutor(max_workers=8) as pool:
            for entries in pool.map(ls, rest):
                nxt += [e.path for e in entries if isinstance(e, RepoFolder) and not posixpath.basename(e.path).startswith(".")]
        calls += len(rest)
        frontier, depth = nxt, depth + 1
    tasks.sort(key=lambda d: (d.count("/"), d))
    # the files the index reads, looked up for every task at once (five hundred paths a call); each task's own folder
    # is listed when its page opens
    files: dict[str, dict[str, int]] = {d: {"task.toml": found.get(f"{d}/task.toml".lstrip("/"), 0)} for d in tasks}
    job.update(state="listing", done=0, total=len(tasks))
    for e in info_of([f"{d}/{k}".lstrip("/") for d in tasks for k in KNOWN_FILES if k != "task.toml"]):
        if not isinstance(e, RepoFile):
            continue
        for d in (posixpath.dirname(e.path), posixpath.dirname(posixpath.dirname(e.path))):
            if d in files:
                files[d][e.path[len(d) + 1:] if d else e.path] = e.size
                break
    job["files"] = sum(len(f) for f in files.values())
    return {"tasks": tasks, "files": files, "lazy": {d: ["*"] for d in tasks}}


def _folder(spec: str, sha: str, path: str, rel: str, token: str | None) -> list[dict[str, Any]]:
    """One folder of a task (rel "" is the task's own), its entries relative to the task, cached for an hour."""
    from huggingface_hub.hf_api import RepoFile

    target = "/".join(x for x in (path, rel) if x)

    def fetch():
        out = []
        for e in _api(token).list_repo_tree(spec, repo_type="dataset", revision=sha, path_in_repo=target or None, recursive=False):
            sub = e.path[len(path) + 1:] if path else e.path
            if isinstance(e, RepoFile):
                out.append({"path": sub, "size": e.size, "withheld": _withheld(sub)})
            elif not posixpath.basename(sub).startswith("."):
                out.append({"path": sub + "/", "dir": True, "lazy": True, "withheld": _withheld(sub + "/")})
            if len(out) >= 3000:
                break
        return out

    return _cached(("folder", spec, sha, target), 3600, fetch)


def _lister(idx: dict[str, Any], token: str | None) -> str | None:
    return token or (config.INDEX_TOKEN if not idx["info"].get("restricted") else None)


def task_folder(spec: str, path: str, rel: str, token: str | None = None) -> dict[str, Any]:
    """One folder inside a task, listed when someone opens it in the file viewer: its files and subfolders."""
    spec = check_spec(spec)
    _, idx = _task_row(spec, path, token)
    rel = posixpath.normpath(rel or "").strip("/")
    if not rel or rel.startswith("..") or rel == ".":
        raise ValueError("not a folder in this task")
    entries = _folder(spec, idx["sha"], path, rel, _lister(idx, token))
    return {"entries": entries, "truncated": len(entries) >= 3000}


def _walked_root(spec: str, idx: dict[str, Any], path: str, token: str | None) -> tuple[dict[str, int], list[str]]:
    """A walked task's own folder: its files, and its folders (listed when opened). Empty for a listed task."""
    pack = _read_pack(spec, idx["sha"])
    entry = pack and pack["tasks"].get(path)
    if not idx.get("walked") and (not entry or "*" not in (entry.get("lazy") or [])):
        return {}, sorted(entry.get("lazy") or []) if entry else []
    entry = entry or {"files": {}}
    try:
        entries = _folder(spec, idx["sha"], path, "", _lister(idx, token))
    except Exception:  # noqa: BLE001 - the index's own files still show
        return {}, sorted({k.split("/")[0] for k in entry["files"] if "/" in k})
    return ({e["path"]: e["size"] for e in entries if not e.get("dir")},
            sorted(e["path"].rstrip("/") for e in entries if e.get("dir")))


def _download(spec: str, sha: str, token: str | None, paths: list[str], local: Path, job: dict[str, Any]) -> None:
    """The files, into a folder for this revision, many at once: one GET each (no metadata round trip), so a
    dataset of thousands of tasks reads in minutes. A file already there isn't fetched again."""
    import httpx
    from urllib.parse import quote

    headers = {"User-Agent": "hf-rl-explorer"} | ({"Authorization": f"Bearer {token}"} if token else {})
    base = f"https://huggingface.co/datasets/{spec}/resolve/{sha}/"
    with httpx.Client(headers=headers, follow_redirects=True, timeout=httpx.Timeout(60, connect=15),
                      limits=httpx.Limits(max_connections=48, max_keepalive_connections=48)) as client:
        def one(path: str) -> None:
            dst = local / path
            try:
                if not dst.exists():
                    for attempt in range(4):
                        r = client.get(base + quote(path))
                        if r.status_code == 429 or r.status_code >= 500:
                            time.sleep(1.5 * (attempt + 1))
                            continue
                        if r.status_code == 200:
                            dst.parent.mkdir(parents=True, exist_ok=True)
                            tmp = dst.with_name(dst.name + ".part")
                            tmp.write_bytes(r.content)
                            tmp.replace(dst)
                        break
            except Exception:  # noqa: BLE001 - a file that won't download is a gap on one task's page, not a failed index
                pass
            job["done"] = job.get("done", 0) + 1

        with ThreadPoolExecutor(max_workers=48) as pool:
            list(pool.map(one, paths))


# ── packs: every task's text, one file per dataset ───────────────────────────
_packs: OrderedDict[tuple[str, str], dict[str, Any]] = OrderedDict()
_pack_bytes: dict[tuple[str, str], int] = {}
_packs_lock = threading.Lock()
PACK_CACHE_BYTES = int(os.environ.get("RLX_PACK_CACHE_MB", 700)) * 1024 * 1024   # about one big pack (LegoFlow) and a few small


def _pack_path(spec: str) -> Path:
    return config.PACK_DIR / f"{_slug(spec)}.json.gz"


def _write_pack(spec: str, sha: str, pack: dict[str, Any]) -> None:
    atomic_write(_pack_path(spec), gzip.compress(json.dumps({"spec": spec, "sha": sha, "version": INDEX_VERSION, **pack},
                                                             separators=(",", ":")).encode(), compresslevel=6))


def _read_pack(spec: str, sha: str) -> dict[str, Any] | None:
    """A dataset's pack, kept in memory for the few datasets read most recently."""
    key = (spec, sha)
    with _packs_lock:
        if key in _packs:
            _packs.move_to_end(key)
            return _packs[key]
    try:
        raw = gzip.decompress(_pack_path(spec).read_bytes())
        pack = json.loads(raw)
    except (OSError, ValueError):
        return None
    if pack.get("sha") != sha or pack.get("version") != INDEX_VERSION or "blobs" not in pack:
        return None
    size = len(raw) * 2   # parsed, it takes about twice its text
    with _packs_lock:
        _packs[key] = pack
        _pack_bytes[key] = size
        while len(_packs) > 1 and sum(_pack_bytes.get(k, 0) for k in _packs) > PACK_CACHE_BYTES:
            old, _ = _packs.popitem(last=False)
            _pack_bytes.pop(old, None)
    return pack


# what may hold an answer: the reference solution, files named for answers, data beside the grader
ANSWER_FILE = re.compile(r"(^|[/_.\-])(gold|answers?|solutions?|expected|ground_truth|oracle|rubrics?|verifier_meta|references?)"
                         r"([/_.\-]|$)", re.I)
ANSWER_TEXT = re.compile(r"\b(pass_anchor|gold_answer|expected_answer|ground_truth|reference_answer)\b")
CODE_FILE = re.compile(r"\.(sh|bash|py|js|mjs|ts|rb|go|rs|java|pl|r)$", re.I)


ANSWER_NAME = r"[A-Za-z_]*(?:expected|answer|gold|solution|ground_truth|oracle|reference)[A-Za-z_0-9]*"
_HEREDOC = re.compile(r"(<<-?\s*['\"]?(\w+)['\"]?[^\n]*\n)(.*?)(\n[ \t]*\2\b)", re.S)
_ASSIGN = re.compile(rf"^(\s*(?:export\s+|local\s+|readonly\s+)?({ANSWER_NAME})\s*[:]?=\s*)(['\"0-9\[{{-].*)$", re.I | re.M)


def _mask_answers(rel: str, text: str) -> str:
    """A grader script with the answer it compares against taken out: a heredoc whose name or target looks like an
    answer (`expected=$(cat <<'ANSWER_EOF' ... ANSWER_EOF)`), and literal values given to answer-like names
    (`EXPECTED="42"`). The logic stays readable; the answer doesn't. A step's tests (`steps/<name>/tests/`) too."""
    if not _in_step(rel).startswith("tests/"):
        return text

    def heredoc(m: re.Match) -> str:
        line_start = text.rfind("\n", 0, m.start()) + 1
        head = text[line_start:m.start()]
        if re.search(ANSWER_NAME, m.group(2), re.I) or re.search(ANSWER_NAME, head, re.I):
            return f"{m.group(1)}‹withheld: the expected answer›{m.group(4)}"
        return m.group(0)

    text = _HEREDOC.sub(heredoc, text)
    return _ASSIGN.sub(lambda m: f"{m.group(1)}\"‹withheld›\"", text)


# keys of a data file that hold an answer or the reference command (e.g. a manifest's "original_bash")
ANSWER_DATA_KEY = re.compile(r"(gold|answers?$|solutions?|expected|ground_?truth|oracle|reference|_bash$|^bash$|_cmd$|_command$|"
                             r"_commands$|solve|_output$|^output$|target_)", re.I)


def _answer_data(rel: str, text: str) -> bool:
    """A JSON or YAML data file whose keys hold an answer: a task manifest naming the reference command, say."""
    if not re.search(r"\.(json|jsonl|ya?ml)$", rel, re.I) or len(text) > 2_000_000:
        return False
    keys: list[str] = []
    try:
        docs = [json.loads(text)] if rel.lower().endswith(".json") else [json.loads(x) for x in text.splitlines()[:200] if x.strip()] if rel.lower().endswith(".jsonl") else []
    except ValueError:
        docs = []
    if not docs and re.search(r"\.ya?ml$", rel, re.I):
        keys = re.findall(r"^\s*-?\s*([\w.\-]+)\s*:", text, re.M)

    def walk(v: Any, depth: int = 0) -> None:
        if depth > 6:
            return
        if isinstance(v, dict):
            for k, x in v.items():
                keys.append(str(k))
                walk(x, depth + 1)
        elif isinstance(v, list):
            for x in v[:200]:
                walk(x, depth + 1)

    for d in docs:
        walk(d)
    return any(ANSWER_DATA_KEY.search(k) and not k.endswith(("_path", "_file", "_dir")) for k in keys)


def _withheld(rel: str, text: str | None = None) -> bool:
    """Whether a task file is kept off the page. Graders' scripts are shown; what they compare against isn't, nor
    a data file whose keys hold an answer or the reference command. A multi-step task's steps (`steps/<name>/`) hold
    the same folders, read by the same rules: each step's `solution/` is withheld, its `tests/` like the task's."""
    base = _in_step(rel)
    if rel.startswith("solution/") or base.startswith("solution/") or ANSWER_FILE.search(rel):
        return True
    if base.startswith("tests/") and base != "tests/test.sh" and not CODE_FILE.search(base):
        return True
    if text and _answer_data(rel, text):
        return True
    return bool(text and (rel.startswith(("tests/", "environment/")) or base.startswith(("tests/", "workdir/"))) and ANSWER_TEXT.search(text))


def _briefs(rows: list[dict[str, Any]]) -> None:
    """Each task's brief: its first paragraph that most of the dataset's tasks don't share. Datasets open every
    instruction with the same preamble ("You are an agent working in a sandbox..."), which tells two tasks apart
    no better than nothing."""
    seen = Counter(p for r in rows for p in set(r["_paras"]))
    common = max(2, int(0.3 * len(rows)))
    titles = Counter(r["title"] for r in rows)
    for r in rows:
        own = [p for p in r.pop("_paras") if seen[p] <= common and not p.startswith(r["title"][:60])]
        r["brief"] = own[0] if own else ""
        # a heading every task shares ("Development Requirements Document") names none of them: its own words do
        if titles[r["title"]] > common and own:
            r["title"] = own[0] if len(own[0]) <= 140 else own[0][:139].rsplit(" ", 1)[0] + "…"
            r["brief"] = own[1] if len(own) > 1 else ""


# ── one task ─────────────────────────────────────────────────────────────────
def _read(path: Path, limit: int = TEXT_LIMIT) -> str:
    try:
        with path.open("rb") as fh:
            return fh.read(limit).decode("utf-8", errors="replace")
    except OSError:
        return ""


def _first_line(text: str, limit: int = 160) -> str:
    """The instruction's first line of prose: not a code fence or what is inside one, not markup or data."""
    fenced = False
    for line in text.splitlines():
        t = line.strip()
        if t.startswith("```"):
            fenced = not fenced
            continue
        t = t.lstrip("#").strip()
        if fenced or not t or t.startswith(("<", "|", "---", "{", "[", '"', "'", "$ ")) or CANARY.search(t):
            continue
        if sum(c.isalpha() or c == " " for c in t) < 0.6 * len(t):   # mostly symbols: code or data, not a sentence
            continue
        return t[:limit]
    return ""


def _paragraphs(text: str, limit: int = 320, most: int = 5) -> list[str]:
    """The instruction's first few prose paragraphs, flattened, for a task's brief (chosen per dataset later)."""
    out = []
    for p in re.split(r"\n\s*\n", text):
        p = p.strip()
        if not p or p.startswith(("#", "```", "|", "<")) or len(p) < 30 or CANARY.search(p):
            continue
        if (p.endswith(":") and len(p) < 120) or re.match(r"([-*+]|\d+[.)])\s", p):   # a list, or its lead-in
            continue
        flat = re.sub(r"\s+", " ", p)
        out.append(flat if len(flat) <= limit else flat[: limit - 1].rsplit(" ", 1)[0] + "…")
        if len(out) >= most:
            break
    return out


def _scalar(v: Any, limit: int = 80) -> str | None:
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, float, str)) and str(v).strip():
        t = " ".join(str(v).split())
        return t if len(t) <= limit else t[: limit - 1].rstrip() + "…"
    return None


def _strings(v: Any) -> list[str]:
    if isinstance(v, str):
        return [x.strip() for x in v.split(",") if x.strip()] if "," in v and len(v) < 200 else [v.strip()]
    if isinstance(v, (list, tuple)):
        return [str(x).strip()[:60] for x in v if isinstance(x, (str, int, float)) and str(x).strip()]
    return []


def _first(meta: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for k in keys:
        if k in meta and not ANSWER_KEY.search(k):
            v = _scalar(meta[k])
            if v:
                return v
    return None


# ── task.toml, read the way Harbor reads it ──────────────────────────────────
# the sections Harbor's TaskConfig knows (harbor/models/task/config.py); anything else at the top level (LegoFlow's
# [scoring]) is the dataset's own, kept as metadata
HARBOR_KEYS = ("schema_version", "version", "task", "metadata", "verifier", "agent", "environment", "solution", "source",
               "multi_step_reward_strategy", "steps", "artifacts")
_quiet = threading.Event()


def _dict(v: Any) -> dict[str, Any]:
    return v if isinstance(v, dict) else {}


def _num(v: Any) -> int | float | None:
    """A number from task.toml, or None (a file Harbor rejects may say `gpus = "1"`)."""
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _size_mb(v: Any) -> int | None:
    """A legacy size ("4G", "512M", "1024K") in MB, as Harbor's EnvironmentConfig converts `memory` and `storage`."""
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([GMK])\s*", v, re.I) if isinstance(v, str) else None
    if not m:
        return None
    n, unit = float(m.group(1)), m.group(2).upper()
    return int(n * 1024) if unit == "G" else int(n) if unit == "M" else int(n / 1024)


def _as_written(doc: dict[str, Any]) -> dict[str, Any]:
    """task.toml as written, with Harbor's legacy fields converted the way TaskConfig converts them (for a file it
    rejects, still worth reading)."""
    def env_of(e: Any) -> dict[str, Any]:
        e = dict(_dict(e))
        for old, new in (("memory", "memory_mb"), ("storage", "storage_mb")):
            if old in e:
                mb = _size_mb(e.pop(old))
                if mb is not None:
                    e.setdefault(new, mb)
        if e.get("allow_internet") is not None and "network_mode" not in e and e.get("allowed_hosts") is None:
            e["network_mode"] = "public" if e["allow_internet"] else "no-network"
        e.pop("allow_internet", None)
        e.setdefault("network_mode", "public")
        if isinstance(e.get("os"), str):
            e["os"] = e["os"].lower()
        return e

    def ver_of(v: Any) -> dict[str, Any]:
        v = dict(_dict(v))
        if isinstance(v.get("environment"), dict):
            v["environment"] = env_of(v["environment"])
        return v

    cfg = {k: v for k, v in doc.items() if k != "_invalid"}
    if "version" in cfg:
        cfg.setdefault("schema_version", cfg.pop("version"))
    cfg.update(environment=env_of(doc.get("environment")), verifier=ver_of(doc.get("verifier")), agent=dict(_dict(doc.get("agent"))))
    steps = doc.get("steps")
    cfg["steps"] = [dict(s, verifier=ver_of(s.get("verifier"))) for s in steps if isinstance(s, dict)] if isinstance(steps, list) else None
    return cfg


def _task_config(doc: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """task.toml as Harbor's own TaskConfig reads it, as plain JSON: legacy `memory = "4G"`, `storage`, `allow_internet`
    and `version` converted, Harbor's defaults filled in. A file TaskConfig rejects (so Harbor won't run it) is read as
    written, legacy fields converted the same way, with TaskConfig's reason."""
    import copy
    import warnings

    if not doc or doc.get("_invalid"):
        return {}, None
    try:
        from harbor.models.task.config import TaskConfig
    except ImportError:   # no Harbor here: as written
        return _as_written(doc), None
    if not _quiet.is_set():   # one per legacy field per task: thousands while indexing
        warnings.filterwarnings("ignore", message=r"The '\w+' field is deprecated", category=DeprecationWarning)
        _quiet.set()
    try:
        return TaskConfig.model_validate(copy.deepcopy(doc)).model_dump(mode="json"), None
    except Exception as exc:  # noqa: BLE001 - pydantic's ValidationError, or a validator's own error
        why = f"{type(exc).__name__}: {exc}"
        try:
            e = exc.errors()[0]   # type: ignore[attr-defined]
            where = ".".join(str(x) for x in e.get("loc") or ())
            msg = re.sub(r"^(Value|Assertion) error, ", "", str(e.get("msg") or ""))
            why = f"`{where}`: {msg}" if where else msg
        except Exception:  # noqa: BLE001 - not a ValidationError: its own words
            pass
        return _as_written(doc), " ".join(why.split())[:240]


def _verifier_mode(ver: Any) -> str | None:
    """Harbor's rule for one [verifier]: its `environment_mode`, else "separate" when it defines an environment, else
    unset (a step inherits the task's; the task's default is "shared")."""
    ver = _dict(ver)
    if ver.get("environment_mode") in ("shared", "separate"):
        return ver["environment_mode"]
    return "separate" if isinstance(ver.get("environment"), dict) else None


def _phases(cfg: dict[str, Any]) -> dict[str, str]:
    """The network each phase sets for itself (Harbor's [agent] and [verifier] `network_mode` overrides, a separate
    grader's own container, each step's), by phase. The sandbox's own is the row's env["network"]."""
    out: dict[str, str] = {}

    def put(key: str, sec: Any) -> None:
        mode = _dict(sec).get("network_mode")
        if isinstance(mode, str) and mode:
            out[key] = mode

    ver = _dict(cfg.get("verifier"))
    put("agent", cfg.get("agent"))
    put("verifier", ver)
    if _verifier_mode(ver) == "separate":
        put("verifier.environment", ver.get("environment"))
    for s in (cfg.get("steps") or [])[:MAX_STEPS]:
        s = _dict(s)
        name = str(s.get("name") or "step")
        put(f"steps.{name}.agent", s.get("agent"))
        put(f"steps.{name}.verifier", s.get("verifier"))
        if _verifier_mode(s.get("verifier")) == "separate":
            put(f"steps.{name}.verifier.environment", _dict(s.get("verifier")).get("environment"))
    return out


def phase_label(key: str) -> str:
    """A phase key from `_phases` in words: "verifier" is "the grader", "steps.s1.agent" is "step `s1`'s agent"."""
    m = re.match(r"steps\.(.+)\.(agent|verifier|verifier\.environment)$", key)
    own = {"agent": "agent", "verifier": "grader", "verifier.environment": "grader's own container"}
    return f"step `{m.group(1)}`'s {own[m.group(2)]}" if m else f"the {own.get(key, key)}"


def _hosts(cfg: dict[str, Any]) -> list[str]:
    """Every host a network allowlist names, in any phase."""
    secs = [cfg.get("environment"), cfg.get("agent"), cfg.get("verifier"), _dict(cfg.get("verifier")).get("environment")]
    for s in cfg.get("steps") or []:
        secs += [_dict(s).get("agent"), _dict(s).get("verifier"), _dict(_dict(s).get("verifier")).get("environment")]
    out: list[str] = []
    for sec in secs:
        for h in _dict(sec).get("allowed_hosts") or []:
            if isinstance(h, str) and h not in out:
                out.append(h)
    return out


def _separate(cfg: dict[str, Any], names: Any, texts: dict[str, str]) -> dict[str, Any] | None:
    """A grader that runs in a container of its own (`[verifier] environment_mode = "separate"`, or a
    `[verifier.environment]`), as Harbor builds it: from `[verifier.environment]` (else a copy of `[environment]`),
    its image, else `tests/Dockerfile`; and whether an HF Sandbox can start it (an image, or a Dockerfile this
    explorer can replay)."""
    from . import dockerfile

    ver = _dict(cfg.get("verifier"))
    task_mode = _verifier_mode(ver) or "shared"
    steps = [_dict(s) for s in cfg.get("steps") or []]
    modes = [(_verifier_mode(s.get("verifier")) or task_mode) for s in steps] if steps else [task_mode]
    if "separate" not in modes:
        return None
    own = isinstance(ver.get("environment"), dict)
    venv = _dict(ver.get("environment")) if own else _dict(cfg.get("environment"))
    image = _scalar(venv.get("docker_image"), 300)
    df = "tests/Dockerfile" in names
    if image:
        ok, why = True, ""
    elif df:
        text = texts.get("tests/Dockerfile")
        ok, why = dockerfile.check(text) if text else (False, "couldn't be read")
        why = f"its `tests/Dockerfile` can't be replayed here ({why})" if why else ""
    else:
        ok, why = False, "it gives that container no image or Dockerfile of its own"
    return {k: v for k, v in {"own": own, "image": image, "dockerfile": df, "ok": ok, "why": why, "gpus": _num(venv.get("gpus")) or 0,
                              "steps": modes.count("separate") if steps else None}.items() if v not in (None, "", 0, False) or k == "ok"}


def _flat(table: Any, prefix: str = "", depth: int = 0) -> list[tuple[str, str]]:
    """A metadata table's short values by dotted key (`repo2env.recipe`), nested tables two levels down; answers and
    contact details left out."""
    out: list[tuple[str, str]] = []
    for k, v in _dict(table).items():
        if ANSWER_KEY.search(str(k)) or CONTACT_KEY.search(str(k)):
            continue
        if isinstance(v, dict):
            if depth < 2:
                out += _flat(v, f"{prefix}{k}.", depth + 1)
        else:
            s = _scalar(v)
            if s is not None and len(s) <= 60:
                out.append((f"{prefix}{k}", s))
    return out


def _answer_keys(v: Any, prefix: str = "", depth: int = 0) -> list[str]:
    """Every answer-like key in a table, at any depth (and in arrays of tables), by dotted path."""
    out: list[str] = []
    if depth > 8:
        return out
    if isinstance(v, dict):
        for k, x in v.items():
            if ANSWER_KEY.search(str(k)):
                out.append(f"{prefix}{k}")
            else:
                out += _answer_keys(x, f"{prefix}{k}.", depth + 1)
    elif isinstance(v, list):
        for x in v[:200]:
            out += [k for k in _answer_keys(x, prefix, depth + 1) if k not in out]
    return out


def _clean(v: Any, depth: int = 0) -> Any:
    """A table as JSON with its answer-like and contact keys left out at every depth (dates as text)."""
    if depth > 8:
        return None
    if isinstance(v, dict):
        return {str(k): _clean(x, depth + 1) for k, x in v.items() if not ANSWER_KEY.search(str(k)) and not CONTACT_KEY.search(str(k))}
    if isinstance(v, list):
        return [_clean(x, depth + 1) for x in v[:200]]
    return v if v is None or isinstance(v, (str, int, float, bool)) else str(v)


# a grader file that calls a model itself: an LLM SDK imported, and its completion / messages / responses API called
_SDK = re.compile(r"\bfrom\s+(?:openai|anthropic|litellm)(?:\.[\w.]+)?\s+import\b|\bimport\s+(?:openai|anthropic|litellm)\b|"
                  r"\brequire\(\s*['\"](?:openai|@anthropic-ai/sdk)['\"]\s*\)|\bfrom\s+['\"](?:openai|@anthropic-ai/sdk)['\"]")
_CALL = re.compile(r"\.(?:chat\.completions|completions|messages|responses)\.(?:create|parse|stream)\s*\(|\blitellm\.a?completion\s*\(")


def _calls_model(text: str) -> bool:
    """Whether a grader file calls a model: it imports openai, anthropic or litellm AND calls a completion, messages or
    responses API. Mentioning a judge (a `judge: bool` parameter, a key's name) isn't enough."""
    if not text or not _SDK.search(text):
        return False
    if _CALL.search(text):
        return True
    return bool(re.search(r"\bfrom\s+litellm\s+import\s+[^\n]*\ba?completion\b", text)
                and re.search(r"^(?!\s*(?:async\s+)?def\s)[^\n#]*(?<![\w.])a?completion\s*\(", text, re.M))


def verifier_kind(test_sh: str, extra: str = "", env_keys: list[str] | None = None, *, toml_text: str | None = None,
                  sources: dict[str, str] | None = None) -> dict[str, Any]:
    """How a task is graded, read from its verifier script: one score (`reward.txt`) or named scores
    (`reward.json`, with the keys a literal in the script names), tests (pytest), and whether a model grades it: its
    task.toml hands the grader a model's key or URL (app/judge.py's plan), or a grader file imports an LLM SDK and calls
    it (`sources`: the grader's other files by path; without them, `extra` is read as one). `env_keys` is accepted for
    old callers and no longer counts: a key's name alone isn't a judge."""
    text = test_sh + "\n" + extra
    kind = "json" if "reward.json" in text else "txt" if "reward.txt" in text else None
    keys: list[str] = []
    if kind == "json":
        for line in text.splitlines():
            if "reward" in line and "{" in line:
                keys += re.findall(r"""["'](\w{2,40})["']\s*:""", line)
    judged = False
    if toml_text:
        from . import judge as judges

        judged = judges.plan(toml_text).get("judge") is not None
    files = [test_sh, *sources.values()] if sources is not None else [test_sh, extra]
    return {"kind": kind, "keys": sorted(set(keys))[:12], "pytest": "pytest" in text,
            "judge": judged or any(_calls_model(t) for t in files)}


def _compact(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v not in (None, "", [], {}, False, 0) or k == "ok"}


def _facts(cfg: dict[str, Any], error: str | None, doc: dict[str, Any], names: Any,
           texts: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """What the index keeps of a task.toml beyond the basics, compactly (empty values left out): for env, verifier,
    and the row's `spec`."""
    env, ver, agent = _dict(cfg.get("environment")), _dict(cfg.get("verifier")), _dict(cfg.get("agent"))
    steps = [_dict(s) for s in cfg.get("steps") or []]
    tpu = _dict(env.get("tpu"))
    env_more = _compact({
        "storage_mb": _num(env.get("storage_mb")), "os": env.get("os") if env.get("os") not in (None, "linux") else None,
        "gpu_types": [str(g) for g in env.get("gpu_types") or []][:8], "tpu": " ".join(str(tpu[k]) for k in ("type", "topology") if tpu.get(k)),
        "workdir": _scalar(env.get("workdir"), 200), "skills_dir": _scalar(env.get("skills_dir"), 200),
        "healthcheck": isinstance(env.get("healthcheck"), dict) or any(isinstance(s.get("healthcheck"), dict) for s in steps),
        "mcp": [str(_dict(m).get("name")) for m in env.get("mcp_servers") or [] if _dict(m).get("name")][:10],
        "hosts": _hosts(cfg)[:20], "phases": _phases(cfg)})
    ver_more = _compact({
        "separate": _separate(cfg, names, texts), "user": _scalar(ver.get("user")), "collect": len(ver.get("collect") or []),
        "bat": "tests/test.bat" in names})
    if steps:
        mins = {str(s.get("name")): s.get("min_reward") for s in steps if s.get("min_reward") is not None}
        multi = _compact({"strategy": cfg.get("multi_step_reward_strategy") or "mean", "min_reward": dict(list(mins.items())[:MAX_STEPS])})
    else:
        multi = None
    pkg = _dict(cfg.get("task"))
    arts = [a if isinstance(a, str) else _dict(a).get("source") for a in cfg.get("artifacts") or []]
    spec = _compact({
        "schema": _scalar(cfg.get("schema_version")), "harbor_error": error,
        "package": _compact({"name": _scalar(pkg.get("name"), 120), "version": _scalar(pkg.get("version"), 40)}),
        "authors": [str(_dict(a).get("name"))[:80] for a in pkg.get("authors") or [] if _dict(a).get("name")][:8],
        "agent_user": _scalar(agent.get("user")), "artifacts": [str(a)[:160] for a in arts if a][:8], "multi": multi,
        "step_names": [str(s.get("name")) for s in steps][:MAX_STEPS],
        "trajectory": "trajectory.json" in names or any(_in_step(n) == "trajectory.json" for n in names if n.startswith("steps/")),
        "tables": [k for k, v in doc.items() if k not in HARBOR_KEYS and k != "_invalid" and isinstance(v, dict)][:8]})
    return env_more, ver_more, spec


def _row(i: int, d: str, local: Path, files: dict[str, int], picks: list[str],
         job: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """One task's index row, and its texts for the pack (answers left out, task.toml masked). task.toml is read by
    Harbor's own TaskConfig (`_task_config`); a multi-step task's instructions come from steps/<name>/instruction.md."""
    folder = local / d if d else local
    texts = {r: _read(folder / r, TEXT_LIMIT if _in_step(r) in ("task.toml", "instruction.md") else SMALL) for r in picks}
    toml_text = texts.get("task.toml", "")
    try:
        doc = tomllib.loads(toml_text) if toml_text else {}
    except tomllib.TOMLDecodeError:
        doc = {"_invalid": True}
    cfg, error = _task_config(doc)
    task, meta = _dict(cfg.get("task")), _dict(doc.get("metadata"))
    raw_env, env, ver, agent = _dict(doc.get("environment")), _dict(cfg.get("environment")), _dict(cfg.get("verifier")), _dict(cfg.get("agent"))
    steps = cfg.get("steps") if isinstance(cfg.get("steps"), list) else None
    first = str(_dict(steps[0]).get("name")) if steps else None
    instruction = texts.get("instruction.md", "") or (texts.get(f"steps/{first}/instruction.md", "") if first else "")
    test_sh = texts.get("tests/test.sh", "")
    name = posixpath.basename(d) or "task"
    title = (_scalar(meta.get("title"), 200) or _scalar(task.get("description"), 200) or _first_line(instruction)
             or name).strip()
    tags: list[str] = []
    for k in TAG_KEYS:
        for t in _strings(meta.get(k)) + (_strings(task.get(k)) if k == "keywords" else []):
            if t not in tags and len(tags) < 12:
                tags.append(t)
    names = list(files)
    job["done"] = job.get("done", 0) + 1
    # the scripts test.sh hands the grading to (`python3 /tests/grade.py`) say which reward file it writes; a step's
    # tests too
    sources = {r: t for r, t in texts.items() if _in_step(r).startswith("tests/") and r != "tests/test.sh"}
    tables = {k: v for k, v in doc.items() if k not in HARBOR_KEYS and isinstance(v, dict)}
    flat = _flat(meta)[:48] + [kv for k, v in tables.items() for kv in _flat(v, f"{k}.", 1)][:16]
    env_more, ver_more, spec = _facts(cfg, error, doc, set(names), texts)
    row = {
        "i": i, "path": d, "name": name, "title": title[:240],
        "_paras": _paragraphs(instruction),
        "difficulty": _first(meta, DIFFICULTY_KEYS), "category": _first(meta, CATEGORY_KEYS), "tags": tags,
        "meta": dict(flat),
        "withheld": sorted(set(_answer_keys(meta) + [k for t, v in tables.items() for k in _answer_keys(v, f"{t}.")])),
        "env": {"image": _scalar(env.get("docker_image"), 300), "dockerfile": "environment/Dockerfile" in files,
                "compose": any(re.match(r"environment/(docker-)?compose\.ya?ml$", n) for n in names),
                "cpus": _num(env.get("cpus")), "memory_mb": _num(env.get("memory_mb")), "gpus": _num(env.get("gpus")) or 0,
                "internet": False if _network(raw_env) in ("no-network", "allowlist") else raw_env.get("allow_internet"),
                "network": _network(raw_env), "build_timeout": env.get("build_timeout_sec"), **env_more},
        "verifier": {**verifier_kind(test_sh, "\n".join(sources.values()), toml_text=toml_text, sources=sources),
                     "timeout": ver.get("timeout_sec"), "files": sorted(n for n in names if n.startswith("tests/"))[:40],
                     "env": sorted(_dict(ver.get("env")).keys()), **ver_more},
        "agent_timeout": agent.get("timeout_sec"),
        "steps": len(steps) if isinstance(steps, list) else None,
        "solution": any(_in_step(n).startswith("solution/") for n in names),
        "reads_env": "${" in toml_text,
        "files": len(files), "bytes": sum(files.values()),
        "invalid": bool(doc.get("_invalid")) or not toml_text,
        "spec": spec,
    }
    row["run"] = runnable(row, texts)["how"]   # how it runs on an HF Sandbox: "image", "dockerfile", or not at all
    kept = {r: (_masked_toml(t) if r == "task.toml" else _mask_answers(r, t)) for r, t in texts.items() if t and not _withheld(r, t)}
    return row, kept


def _group(rows: list[dict[str, Any]]) -> None:
    """A task's group: the folders between the dataset's common root and the task, when there are any."""
    parents = [posixpath.dirname(r["path"]) for r in rows]
    common = posixpath.commonpath(parents) if parents and all(parents) else ""
    for r, p in zip(rows, parents):
        rel = p[len(common):].strip("/") if common else p
        rel = re.sub(r"(^|/)tasks$", "", rel).strip("/")   # `hle/tasks` is the group `hle`
        r["group"] = rel or None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """A dataset at a glance: environments, how it is graded, and the facets worth filtering by."""
    n = len(rows) or 1
    kinds = Counter(r["verifier"]["kind"] or "unknown" for r in rows)
    keys = Counter(k for r in rows for k in r["verifier"]["keys"])
    meta_keys = Counter(k for r in rows for k in r["meta"])
    facets = {}
    for name in ("difficulty", "category", "group"):
        c = Counter(r.get(name) for r in rows if r.get(name))
        if 1 < len(c) <= 60 or (len(c) == 1 and sum(c.values()) < n):
            facets[name] = c.most_common(40)
    tags = Counter(t for r in rows for t in r["tags"])
    if tags:
        facets["tags"] = tags.most_common(40)
    # other metadata worth filtering by: a few distinct values, on enough of the tasks
    for k, seen in meta_keys.most_common(30):
        if k in DIFFICULTY_KEYS + CATEGORY_KEYS + TAG_KEYS or seen < 0.2 * n:
            continue
        c = Counter(r["meta"].get(k) for r in rows if r["meta"].get(k))
        repeated = sum(v for v in c.values() if v > 1)
        if 2 <= len(c) <= 25 and repeated >= 0.5 * sum(c.values()):   # categories, not one-off names
            facets[f"meta:{k}"] = c.most_common(25)
    return {
        "tasks": len(rows),
        "image": sum(1 for r in rows if r["env"]["image"]),
        "dockerfile_only": sum(1 for r in rows if not r["env"]["image"] and r["env"]["dockerfile"]),
        "compose": sum(1 for r in rows if r["env"]["compose"]),
        "gpu": sum(1 for r in rows if r["env"]["gpus"]),
        "solution": sum(1 for r in rows if r["solution"]),
        "runnable": sum(1 for r in rows if r.get("run")),
        "replayed": sum(1 for r in rows if r.get("run") == "dockerfile"),
        "multi_step": sum(1 for r in rows if r["steps"]),
        "reads_env": sum(1 for r in rows if r["reads_env"]),
        "invalid": sum(1 for r in rows if r["invalid"]),
        "verifier": dict(kinds), "reward_keys": keys.most_common(20),
        "pytest": sum(1 for r in rows if r["verifier"]["pytest"]),
        "judge": sum(1 for r in rows if r["verifier"]["judge"]),
        "withheld": sorted({k for r in rows for k in r["withheld"]}),
        "facets": facets,
        "meta_keys": [k for k, _ in meta_keys.most_common(40)],
    }


# ── a task's page ────────────────────────────────────────────────────────────
def _task_row(spec: str, path: str, token: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    meta = info(spec, token)   # checks this visitor may read it, every time
    idx = _read_index(spec, meta["sha"])
    if not idx:
        raise LookupError("this dataset is not indexed yet")
    for r in idx["tasks"]:
        if r["path"] == path:
            return r, idx
    raise LookupError("no such task")


def _task_files(spec: str, sha: str, path: str, token: str | None) -> dict[str, int]:
    from huggingface_hub.hf_api import RepoFile

    def fetch():
        out = {}
        for e in _api(token).list_repo_tree(spec, repo_type="dataset", revision=sha, path_in_repo=path or None, recursive=True):
            if isinstance(e, RepoFile):
                out[e.path[len(path) + 1:] if path else e.path] = e.size
        return out

    return _cached(("files", spec, sha, path), 3600, fetch)


def _fetch(spec: str, sha: str, path: str, rel: str, token: str | None) -> Path:
    from huggingface_hub import hf_hub_download

    target = f"{path}/{rel}".lstrip("/")
    return Path(hf_hub_download(spec, target, repo_type="dataset", revision=sha, token=token or False,
                                local_dir=str(config.CACHE_DIR / _slug(spec))))


_TOML_KEY = re.compile(r"^(\s*)([\w.\-\"']+)(\s*=\s*)(.*)$")
_TOML_TABLE = re.compile(r"^\s*\[\[?\s*([^\[\]]+?)\s*\]\]?\s*(#.*)?$")


def _value_end(lines: list[str], i: int, value: str) -> int:
    """The last line of the TOML value that starts on line `i` (`value`: its text there): line `i` itself, unless it
    opens an array, an inline table or a multi-line string that closes further down."""
    if not re.match(r"\s*(\[|\{|\"\"\"|''')", value):
        return i
    acc = value
    for j in range(i, min(len(lines), i + 5000)):
        if j > i:
            acc += "\n" + lines[j]
        if re.search(r"(\]|\}|\"\"\"|''')\s*(#.*)?$", lines[j] if j > i else value):   # it may close here: try it
            try:
                tomllib.loads("x = " + acc)
                return j
            except tomllib.TOMLDecodeError:
                pass
    return i


def _masked_toml(text: str) -> str:
    """task.toml as written, with any answer-like metadata value replaced: a key named like an answer (its whole
    value, over every line it spans), and every key of a table named like one (`[metadata.gold]`)."""
    lines = text.splitlines()
    out: list[str] = []
    hidden, i = False, 0
    while i < len(lines):
        line = lines[i]
        m = _TOML_KEY.match(line)
        t = None if m else _TOML_TABLE.match(line)
        if t:   # a table: its keys are hidden when its name is an answer's (not Harbor's own [solution.env])
            name = t.group(1)
            hidden = bool(ANSWER_KEY.search(name)) and not re.fullmatch(r"solution(\.env)?", name.strip())
            out.append(line)
            i += 1
            continue
        if m:
            end = _value_end(lines, i, m.group(4))
            if hidden or ANSWER_KEY.search(m.group(2)):
                out.append(f"{m.group(1)}{m.group(2)}{m.group(3)}\"‹withheld›\"")
            else:
                out += lines[i:end + 1]
            i = end + 1
            continue
        out.append(line)
        i += 1
    return "\n".join(out)


def _task_texts(spec: str, sha: str, path: str, token: str | None) -> tuple[dict[str, int], dict[str, str], set[str]]:
    """A task's file list, its texts (answers left out) and which files were read: from the dataset's pack when it
    has one, else from the Hub."""
    pack = _read_pack(spec, sha)
    entry = pack and pack["tasks"].get(path)
    if entry:
        files = entry["files"]
        return files, {rel: pack["blobs"][h] for rel, h in entry["texts"].items()}, set(_pick(files))
    idx = _read_index(spec, sha)
    if idx and idx.get("walked"):   # a task too big to list: its own folder and the files a page reads
        from huggingface_hub.hf_api import RepoFile

        files = {e["path"]: e["size"] for e in _folder(spec, sha, path, "", _lister(idx, token)) if not e.get("dir")}
        want = [f"{path}/{k}".lstrip("/") for k in KNOWN_FILES if "/" in k]
        for e in _api(_lister(idx, token)).get_paths_info(spec, want, repo_type="dataset", revision=sha):
            if isinstance(e, RepoFile):
                files[e.path[len(path) + 1:] if path else e.path] = e.size
    else:
        files = _task_files(spec, sha, path, token)
    picks = _pick(files)[:16]
    texts: dict[str, str] = {}

    def one(rel: str) -> None:
        try:
            text = _read(_fetch(spec, sha, path, rel, token), TEXT_LIMIT if rel in ("task.toml", "instruction.md") else SMALL)
        except Exception:  # noqa: BLE001 - one missing file doesn't sink the page
            return
        if not _withheld(rel, text):
            texts[rel] = _masked_toml(text) if rel == "task.toml" else _mask_answers(rel, text)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(one, picks))
    return files, texts, set(picks)


def _steps_detail(cfg: dict[str, Any], doc: dict[str, Any], files: dict[str, int], texts: dict[str, str]) -> list[dict[str, Any]]:
    """A multi-step task's steps as Harbor runs them: each one's instruction (steps/<name>/instruction.md), its own
    tests, the files staged before it (workdir/), whether it has a solution (never shown) and its settings."""
    raw = {str(_dict(s).get("name")): _dict(s) for s in doc.get("steps") or [] if isinstance(s, dict)}
    out = []
    for s in (cfg.get("steps") or [])[:MAX_STEPS]:
        s = _dict(s)
        name = str(s.get("name") or "")
        own = f"steps/{name}/"
        ins = own + "instruction.md"
        inline = raw.get(name, {}).get("instruction")
        out.append({
            "name": name,
            # Harbor reads the file; an older task.toml wrote it inline (Harbor ignores that, and the step can't run)
            "instruction": texts.get(ins) if ins in texts else None, "inline": inline[:20000] if isinstance(inline, str) else None,
            "found": ins in files, "withheld": ins in files and _withheld(ins),
            "tests": sorted(k[len(own):] for k in files if k.startswith(own + "tests/"))[:40],
            "texts": {k[len(own):]: v for k, v in texts.items() if k.startswith(own + "tests/")},
            "workdir": sorted(k[len(own) + 8:] for k in files if k.startswith(own + "workdir/"))[:40],
            "solution": any(k.startswith(own + "solution/") for k in files),
            "config": {k: v for k, v in s.items() if k != "name"},
        })
    return out


def task(spec: str, path: str, token: str | None = None) -> dict[str, Any]:
    """What the task page shows: the row, the instruction in full (each step's, for a multi-step task), the settings as
    Harbor reads them, the verifier's own files, the environment's recipe and the file list. Answers stay out (see the
    module docstring)."""
    spec = check_spec(spec)
    row, idx = _task_row(spec, path, token)
    sha = idx["sha"]
    files, texts, read = _task_texts(spec, sha, path, token)
    root_files, folders = _walked_root(spec, idx, path, token)
    files = {**root_files, **files}
    toml_text = texts.get("task.toml", "")
    try:
        doc = tomllib.loads(toml_text) if toml_text else {}
    except tomllib.TOMLDecodeError:
        doc = {}
    cfg, _ = _task_config(doc)   # from the masked task.toml: answer-like values are already out
    test_files = {k: v for k, v in texts.items() if k.startswith("tests/")}
    sources = {k: v for k, v in texts.items() if _in_step(k).startswith("tests/") and k != "tests/test.sh"}
    verifier = {**row["verifier"], **verifier_kind(texts.get("tests/test.sh", ""), "\n".join(sources.values()),
                                                   toml_text=toml_text, sources=sources)}
    if row["verifier"]["kind"] and not verifier["kind"]:   # the index read files the page withholds
        verifier.update(kind=row["verifier"]["kind"], keys=row["verifier"]["keys"])
    verifier["judge"] = bool(row["verifier"].get("judge") or verifier["judge"])
    return {
        **row, "dataset": spec, "sha": sha, "verifier": verifier, "collection": collection(spec), "runnable": runnable(row, texts),
        "restricted": bool(idx["info"].get("restricted")),
        "instruction": texts.get("instruction.md", ""),
        "toml": toml_text,
        "config": {k: v for k, v in cfg.items() if k != "metadata"},
        "metadata": _clean(_dict(doc.get("metadata"))),
        "tables": _clean({k: v for k, v in doc.items() if k not in HARBOR_KEYS}),
        "steps_detail": _steps_detail(cfg, doc, files, texts),
        "dockerfile": texts.get("environment/Dockerfile"),
        "compose": next((v for k, v in texts.items() if re.match(r"environment/(docker-)?compose\.ya?ml$", k)), None),
        "tests": test_files,
        "tree": [{"path": k, "size": v, "withheld": _withheld(k, texts.get(k)) or (k in read and k not in texts)}
                 for k, v in sorted(files.items())][:2000]
                + [{"path": f"{d}/", "dir": True, "lazy": True, "withheld": _withheld(f"{d}/")} for d in folders],
        "walked": bool(folders),
        "tree_truncated": len(files) > 2000,
    }


def _network(env: dict[str, Any]) -> str | None:
    """The network the task asks for: Harbor's `network_mode`, or the older `allow_internet`."""
    mode = env.get("network_mode")
    if isinstance(mode, str):
        return mode
    return "no-network" if env.get("allow_internet") is False else None


def _spec_row(row: dict[str, Any], texts: dict[str, str]) -> dict[str, Any]:
    """A row read outside an index (a Harbor task packed into a dataset row) completed from its task.toml with what
    runnable() checks."""
    try:
        doc = tomllib.loads(texts["task.toml"])
    except tomllib.TOMLDecodeError:
        return {**row, "spec": {}}
    cfg, error = _task_config(doc)
    env_more, ver_more, spec = _facts(cfg, error, doc, set(texts), texts)
    return {**row, "env": {"gpus": _num(_dict(cfg.get("environment")).get("gpus")) or 0, **(row.get("env") or {}), **env_more},
            "verifier": {**(row.get("verifier") or {}), **ver_more}, "spec": spec}


def runnable(row: dict[str, Any], texts: dict[str, str]) -> dict[str, Any]:
    """Whether the task can run on an HF Sandbox: from its prebuilt image, or by replaying its Dockerfile. Not when
    Harbor itself rejects its task.toml, nor when it asks for what Harbor's HF Sandbox environment can't give or
    enforce: a Windows container, a GPU or a TPU, or no internet or a network allowlist for the sandbox or for any one
    phase (Harbor refuses rather than run it more open than its author meant). A grader in a container of its own needs
    an image, or a `tests/Dockerfile` this explorer can replay. (`${VAR}`s nothing here fills: app/judge.py.)"""
    from . import dockerfile

    if "spec" not in row and texts.get("task.toml"):   # not from an index: its task.toml says the rest
        row = _spec_row(row, texts)
    e, spec = row.get("env") or {}, row.get("spec") or {}
    sep = (row.get("verifier") or {}).get("separate") or {}

    def no(why: str) -> dict[str, Any]:
        return {"ok": False, "how": None, "why": why}

    if spec.get("harbor_error"):
        return no(f"Harbor itself rejects its task.toml ({spec['harbor_error']})")
    if e.get("os") == "windows":
        return no("it targets a Windows container, and HF Sandboxes run Linux")
    gpus = _num(e.get("gpus")) or _num(sep.get("gpus")) or 0
    if gpus > 0:
        return no(f"it asks for {'a GPU' if gpus == 1 else f'{gpus} GPUs'}, which Harbor can't attach to an HF Sandbox")
    if e.get("tpu"):
        return no(f"it asks for a TPU slice ({e['tpu']}), which Harbor can't attach to an HF Sandbox")
    net = e.get("network")
    if net in ("no-network", "allowlist"):
        return no("it asks for a sandbox without internet access, which HF Sandboxes can't enforce"
                  if net == "no-network" else "it asks for a network allowlist, which HF Sandboxes can't enforce")
    for phase, mode in (e.get("phases") or {}).items():
        if mode in ("no-network", "allowlist"):
            return no(f"{phase_label(phase)} asks for {'no internet access' if mode == 'no-network' else 'a network allowlist'}, "
                      "which HF Sandboxes can't enforce")
    if sep and not sep.get("ok"):
        return no(f"its grader runs in a container of its own, and {sep.get('why') or 'that container cannot start here'}")
    if row["env"].get("image"):
        return {"ok": True, "how": "image"}
    if row["env"].get("compose"):
        return {"ok": False, "how": None, "why": "it runs several containers from a compose file"}
    text = texts.get("environment/Dockerfile")
    ok, why = dockerfile.check(text)
    return {"ok": ok, "how": "dockerfile" if ok else None, "why": why,
            "base": dockerfile.plan(text).base if ok else None}


def task_dir(spec: str, path: str, token: str | None = None) -> Path:
    """A task's folder on local disk, every file but the reference solution, for the runner."""
    from huggingface_hub import snapshot_download

    meta = info(spec, token)
    local = config.CACHE_DIR / _slug(spec)
    prefix = f"{path}/" if path else ""
    snapshot_download(spec, repo_type="dataset", revision=meta["sha"], local_dir=str(local),
                      token=(token if meta.get("restricted") else None) or False,
                      allow_patterns=[f"{prefix}*"], ignore_patterns=[f"{prefix}solution/*", f"{prefix}steps/*/solution/*"], max_workers=8)
    return local / path if path else local


def _binary_name(name: str) -> bool:
    return bool(re.search(r"\.(png|jpe?g|gif|webp|pdf|zip|tar|gz|tgz|bz2|xz|parquet|arrow|bin|pt|pkl|npy|npz|so|o|exe|"
                          r"whl|mp3|wav|mp4|db|sqlite|xlsx?|docx?|pptx?)$", name, re.I))


def task_file(spec: str, path: str, rel: str, token: str | None = None) -> dict[str, Any]:
    """One file from a task, for the file viewer: a text file inside the task's folder that holds no answer."""
    spec = check_spec(spec)
    row, idx = _task_row(spec, path, token)
    files, texts, _ = _task_texts(spec, idx["sha"], path, token)
    rel = posixpath.normpath(rel or "").lstrip("/")
    if rel.startswith("..") or rel == ".":
        return {"path": rel, "error": "not a file in this task"}
    root_files, folders = _walked_root(spec, idx, path, token)
    files = {**root_files, **files}
    if rel not in files:   # in a folder the index didn't list: look it up in that folder's listing
        if "/" not in rel or rel.split("/")[0] not in folders:
            return {"path": rel, "error": "not a file in this task"}
        hit = next((e for e in task_folder(spec, path, posixpath.dirname(rel), token)["entries"] if e["path"] == rel and not e.get("dir")), None)
        if hit is None:
            return {"path": rel, "error": "not a file in this task"}
        files = {**files, rel: hit["size"]}
    if _withheld(rel):
        return {"path": rel, "size": files[rel], "withheld": True}
    if rel in texts:
        if _withheld(rel, texts[rel]):   # packs built before a rule was added are checked again as they're served
            return {"path": rel, "size": files[rel], "withheld": True}
        return {"path": rel, "size": files[rel], "text": texts[rel], "truncated": files[rel] > len(texts[rel].encode())}
    if _binary_name(rel):
        return {"path": rel, "size": files[rel], "binary": True}
    text = _read(_fetch(spec, idx["sha"], path, rel, token))
    if "\x00" in text[:8192]:
        return {"path": rel, "size": files[rel], "binary": True}
    if _withheld(rel, text):
        return {"path": rel, "size": files[rel], "withheld": True}
    if rel == "task.toml":
        text = _masked_toml(text)
    return {"path": rel, "size": files[rel], "text": _mask_answers(rel, text), "truncated": files[rel] > TEXT_LIMIT}
