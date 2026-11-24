"""A frozen, diverse, difficulty-ranked evaluation set over the test split (DesignWorldBench, Recreation-CUA v1).

Random sampling is a poor eval set here: the test split is 40 appearance
clusters of very uneven size (1 to 227 designs), so a random 300 is dominated
by a handful of looks. Selection is instead:

1. **Stratified** by difficulty tier x the designs' use group (social post, cover,
   marketing, ad, event, blog), with each tier held to an equal share.
2. **Spread**: at most `cluster_cap` tasks per appearance cluster, and within a
   stratum the next pick is the candidate whose category, format and canvas
   shape are least represented so far.
3. **Covering**: until each hard feature (rotation, transparency, mixed-style
   text, dense layouts, long copy, tiny text, reused assets, no photo) has its
   quota, candidates that add an under-covered feature go first.
4. **Seeded**: every tie breaks on a hash of (seed, snapshot, task id), so the
   same corpus and seed give the same set, and nothing depends on file order.

The lite subset is chosen from the full set by the same rules, so lite scores
are a smaller sample of the same distribution rather than a different test.

`build(validate=True)` fetches every selected design and checks its ground
truth scores exactly 1.0 against itself. A task that fails is recorded, never
swapped for another; a set with failures refuses to load. Building needs a
catalog with per-design index features (`rows()`, `snapshot_id`); scoring,
subsets and offline export need only the frozen file and a task pack.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from .difficulty import TIERS

EVALSET_VERSION = 1
DEFAULT_SIZE = 300
DEFAULT_LITE = 60
DEFAULT_SPLIT = "test"

# Hard features the set must contain, with how to detect each from index features.
FLAGS = {
    "rotated": lambda f: f["rotated"] > 0,
    "transparent": lambda f: f["semi"] > 0,
    "mixed_style_text": lambda f: f["styled"] > 0,
    "dense": lambda f: f["n_elements"] >= 20,
    "long_copy": lambda f: f["chars"] >= 175,
    "tiny_text": lambda f: f["tiny_text"] > 0,
    "reused_asset": lambda f: f["reuse"] > 0,
    "three_fonts": lambda f: f["n_fonts"] >= 3,
    "no_photo": lambda f: f["n_images"] == 0,
}


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _h(seed, snapshot, *parts) -> str:
    # the salt is part of the frozen sets' tie-breaks: changing it would change which tasks a rebuild picks
    return hashlib.sha256(canonical(["design-canvas-evalset-v1", seed, snapshot, *parts]).encode()).hexdigest()


def doc_digest(doc: dict) -> str:
    """The ground truth's fingerprint, as frozen in each entry's `doc_sha256`."""
    return hashlib.sha256(canonical(doc).encode()).hexdigest()


def flags_of(row: dict) -> list[str]:
    if "flags" in row:  # a frozen entry carries its flags; index rows carry the raw features
        return list(row["flags"])
    return [name for name, test in FLAGS.items() if test(row)]


def _allocate(pool: list[dict], size: int) -> dict[tuple[str, str], int]:
    """Per-(tier, group) quotas: tiers equal, groups equal within a tier, capped by supply."""
    groups = sorted({r["group"] for r in pool})
    supply = Counter((r["tier"], r["group"]) for r in pool)
    quotas = {}
    per_tier, extra = divmod(size, len(TIERS))
    for ti, tier in enumerate(TIERS):
        want = per_tier + (ti < extra)
        need = {g: 0 for g in groups}
        # Water-fill: hand out one at a time to the group with the fewest so far that has supply left.
        for _ in range(want):
            open_ = [g for g in groups if need[g] < supply[tier, g]]
            if not open_:
                raise ValueError(f"the {tier} tier cannot supply {want} tasks")
            g = min(open_, key=lambda g: (need[g], g))
            need[g] += 1
        for g in groups:
            quotas[tier, g] = need[g]
    return quotas


def select(pool: list[dict], size: int, *, seed: int, snapshot: str, cluster_cap: int | None = None,
           flag_quota: int | None = None) -> list[dict]:
    """Choose `size` rows from `pool` (index rows with features, tier, group, cluster)."""
    if size > len(pool):
        raise ValueError(f"asked for {size} tasks from a pool of {len(pool)}")
    n_clusters = len({r["cluster"] for r in pool})
    cap = cluster_cap or max(1, math.ceil(2 * size / n_clusters))
    fq = flag_quota if flag_quota is not None else max(3, size // 20)
    quotas = _allocate(pool, size)
    cells: dict[tuple, list[dict]] = {}
    for r in sorted(pool, key=lambda r: _h(seed, snapshot, r["task_id"])):
        cells.setdefault((r["tier"], r["group"]), []).append(r)
    counts = {k: Counter() for k in ("cluster", "category", "format", "shape", "flag")}
    chosen, taken = [], Counter()

    def pick(cell):
        best, best_key = None, None
        for r in cells[cell]:
            if counts["cluster"][r["cluster"]] >= cap:
                continue
            gain = sum(1 for f in flags_of(r) if counts["flag"][f] < fq)
            key = (-gain, counts["category"][r["category"]], counts["format"][r["format"]],
                   counts["shape"][r["shape"]])
            if best_key is None or key < best_key:
                best, best_key = r, key
        return best

    order = [c for c in sorted(quotas) if quotas[c]]
    while len(chosen) < size:
        progressed = False
        for cell in order:  # round robin, so balancing counters evolve evenly across strata
            if taken[cell] >= quotas[cell]:
                continue
            r = pick(cell)
            if r is None:
                continue
            cells[cell].remove(r)
            taken[cell] += 1
            chosen.append(r)
            progressed = True
            counts["cluster"][r["cluster"]] += 1
            for k in ("category", "format", "shape"):
                counts[k][r[k]] += 1
            for f in flags_of(r):
                counts["flag"][f] += 1
        if not progressed:
            short = {f"{t}/{g}": quotas[t, g] - taken[t, g] for t, g in order if taken[t, g] < quotas[t, g]}
            raise ValueError(f"cluster cap {cap} leaves strata short: {short}; raise cluster_cap")
    return chosen


ENTRY_KEYS = ("task_id", "design_id", "tier", "score", "group", "shape", "cluster", "category",
              "format", "n_elements", "n_text", "chars", "doc_sha256")


def _entry(r: dict) -> dict:
    return {**{k: r[k] for k in ENTRY_KEYS}, "flags": flags_of(r)}


def _identity(snapshot: str, split: str, seed: int, ids: list[str], lite: list[str]) -> str:
    return hashlib.sha256(canonical([EVALSET_VERSION, snapshot, split, seed, sorted(ids),
                                     sorted(lite)]).encode()).hexdigest()


def build(pack, *, split: str = DEFAULT_SPLIT, size: int = DEFAULT_SIZE, lite: int = DEFAULT_LITE,
          seed: int = 42, validate: bool = True, log=print) -> dict:
    """Select, validate, and return the frozen record. `pack` is a catalog with index features (`rows()`)."""
    pool = pack.rows(split)
    snapshot = pack.snapshot_id
    full = select(pool, size, seed=seed, snapshot=snapshot)
    lite_rows = select(full, lite, seed=seed, snapshot=snapshot, flag_quota=max(2, lite // 30))
    tasks = sorted((_entry(r) for r in full), key=lambda e: (TIERS.index(e["tier"]), e["task_id"]))
    lite_ids = sorted(r["task_id"] for r in lite_rows)
    failures = []
    if validate:
        pack.prefetch([pack.get(e["task_id"]) for e in tasks])
        for i, e in enumerate(tasks, 1):
            try:
                task = pack.get(e["task_id"])
                grade = pack.grade(task, pack.gt_doc(task))
                if abs(grade.total - 1.0) > 1e-6:
                    raise ValueError(f"ground truth scores {grade.total:.4f}, not 1.0")
                if not task.get("description"):
                    raise ValueError("no written brief for description mode")
            except Exception as err:  # recorded, never substituted
                failures.append({"task_id": e["task_id"], "error": f"{type(err).__name__}: {err}"})
            if i % 25 == 0:
                log(f"  validated {i}/{len(tasks)} ({len(failures)} failures)")
    record = {
        "evalset_version": EVALSET_VERSION,
        "name": f"recreation-cua-{split}-{size}",
        "snapshot_id": snapshot,
        "difficulty": pack.manifest["difficulty"],
        "split": split,
        "seed": seed,
        "size": len(tasks),
        "validated": validate,
        "failures": failures,
        "subsets": {"lite": lite_ids},
        "tasks": tasks,
    }
    record["evalset_id"] = _identity(snapshot, split, seed, [e["task_id"] for e in tasks], lite_ids)
    record["summary"] = summarize(record)
    return record


def summarize(record: dict, subset: str | None = None) -> dict:
    tasks = subset_tasks(record, subset)
    c = lambda key: dict(sorted(Counter(e[key] for e in tasks).items()))  # noqa: E731
    flags = Counter(f for e in tasks for f in e["flags"])
    return {
        "size": len(tasks),
        "tiers": {t: sum(e["tier"] == t for e in tasks) for t in TIERS},
        "groups": c("group"),
        "shapes": c("shape"),
        "clusters": len({e["cluster"] for e in tasks}),
        "max_per_cluster": max(Counter(e["cluster"] for e in tasks).values()),
        "categories": len({e["category"] for e in tasks}),
        "formats": len({e["format"] for e in tasks}),
        "flags": {f: flags[f] for f in FLAGS},
        "median_elements_by_tier": {
            t: sorted(e["n_elements"] for e in tasks if e["tier"] == t)[
                len([e for e in tasks if e["tier"] == t]) // 2] for t in TIERS
            if any(e["tier"] == t for e in tasks)},
    }


def add_subset(record: dict, name: str, size: int, within: str = "lite") -> list[str]:
    """Nest a smaller subset inside an existing one (by the same selection rules), so its scores are a smaller
    sample of the same distribution and a larger run later extends them. The set's identity is unchanged."""
    if name in ("full", "lite"):
        raise ValueError(f"{name} is built with the set; pick another name")
    pool = subset_tasks(record, within)
    rows = select(pool, size, seed=record["seed"], snapshot=record["snapshot_id"], flag_quota=max(2, size // 30))
    ids = sorted(r["task_id"] for r in rows)
    record["subsets"][name] = ids
    return ids


def subset_tasks(record: dict, subset: str | None = None) -> list[dict]:
    if not subset or subset == "full":
        return list(record["tasks"])
    ids = set(record["subsets"][subset])
    return [e for e in record["tasks"] if e["task_id"] in ids]


def save(record: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n")
    return path


def load(path: str | Path, snapshot_id: str | None = None) -> dict:
    """Read a frozen set and refuse one that is inconsistent or built for another corpus."""
    record = json.loads(Path(path).read_text())
    if record.get("evalset_version") != EVALSET_VERSION:
        raise ValueError("rebuild this evaluation set for the current evalset version")
    if snapshot_id and record["snapshot_id"] not in (snapshot_id, record.get("corpus_snapshot_id")):
        raise ValueError("evaluation set was built for a different corpus snapshot; rebuild it")
    if record.get("failures"):
        raise ValueError(f"{len(record['failures'])} task(s) failed validation; fix them before scoring")
    ids = [e["task_id"] for e in record["tasks"]]
    if len(set(ids)) != len(ids) or len(ids) != record["size"]:
        raise ValueError("evaluation set size or task uniqueness is inconsistent")
    for name, members in record["subsets"].items():
        if not set(members) <= set(ids):
            raise ValueError(f"{name} subset is not inside the full set")
    snap = record.get("corpus_snapshot_id", record["snapshot_id"])
    if _identity(snap, record["split"], record["seed"], ids, record["subsets"]["lite"]) != record["evalset_id"]:
        raise ValueError("evaluation set identity does not match its task ids")
    return record


def check_against(record: dict, pack) -> None:
    """Every task still resolves to the same ground truth in the served catalog."""
    for e in record["tasks"]:
        task = pack.get(e["task_id"])
        if task["design_id"] != e["design_id"]:
            raise ValueError(f"{e['task_id']} is design {task['design_id']} here, not {e['design_id']}")
        if doc_digest(task["doc"]) != e["doc_sha256"]:
            raise ValueError(f"{e['task_id']}: ground truth differs from the frozen set")


def export_pack(pack, record: dict, out: str | Path) -> dict:
    """Write the set as an ordinary offline task pack (no bucket, no network needed).

    Task ids are kept, so `reset(task_id=...)` and results line up with runs
    against the full corpus. Positions in the pack follow the frozen order.
    """
    import shutil

    if record.get("failures"):
        raise ValueError("refusing to export a set with recorded failures")
    out = Path(out)
    (out / "tasks").mkdir(parents=True, exist_ok=True)
    if not (out / "fonts").exists():
        shutil.copytree(pack.fonts.root, out / "fonts")
    lines = []
    for pos, e in enumerate(record["tasks"]):
        task = dict(pack.get(e["task_id"]))
        # a catalog that fetches designs on demand fills them in first
        src = pack.materialize(task) if hasattr(pack, "materialize") else pack.root / "designs" / task["design_id"]
        pack.preview_path(task)
        dest = out / "designs" / task["design_id"]
        if not dest.exists():
            shutil.copytree(src, dest, ignore=shutil.ignore_patterns(".ok"))
        task["index"] = pos  # position in this pack; task_id keeps the corpus identity
        lines.append(json.dumps(task, separators=(",", ":")))
    (out / "tasks" / f"{record['split']}.jsonl").write_text("\n".join(lines) + "\n")
    manifest = {
        "version": 2,
        "created": record.get("created"),
        "source": pack.manifest.get("source"),
        "splits": {record["split"]: len(lines)},
        "corpus_snapshot_id": record["snapshot_id"],
        "evalset_id": record["evalset_id"],
        "max_elements": pack.manifest.get("config", pack.manifest).get("max_elements"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if (pack.root / "NOTICE.md").exists():  # the data licence travels with the data
        shutil.copy(pack.root / "NOTICE.md", out / "NOTICE.md")
    save({**record, "corpus_snapshot_id": record["snapshot_id"]}, out / "evalset.json")
    return manifest
