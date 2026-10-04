"""Environments whose tasks are dataset rows (Verifiers, NeMo Gym, verl, MiMo's raw release, Harbor tasks packed into
parquet, traces, any table of prompts). Rows come from the Hub's dataset viewer, or straight from the dataset's files
when the viewer can't read it (direct.py); a row reader (processors.py) turns each into a task. `RowsAdapter`, at the
end, puts this behind the environment contract (contract.py): a task's ref is `<split>/<row>`, or
`<config>/<split>/<row>` for a dataset with several configs.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from typing import Any

from . import contract as c
from . import direct, viewer
from .base import ANSWER, FACET_SKIP, Dataset, Processor, parse_json, retitle
from .direct import DirectError
from .processors import HIDDEN, PROCESSORS
from .viewer import ViewerError

__all__ = ["describe", "rows", "task", "task_file", "packed", "materialize", "ViewerError", "DirectError", "pick"]
PAGE = 24


def pick(ds: Dataset) -> Processor:
    return max(PROCESSORS, key=lambda p: p.match(ds))


# ── where rows come from ─────────────────────────────────────────────────────
class Viewer:
    kind = "viewer"

    def __init__(self, spec: str, tok: str | None):
        self.spec, self.tok = spec, tok

    def splits(self) -> list[dict[str, str]]:
        return viewer.splits(self.spec, self.tok)

    def page(self, config: str, split: str, offset: int, length: int) -> dict[str, Any]:
        return viewer.rows(self.spec, config, split, offset, length, self.tok)

    def search(self, config, split, q, offset):
        try:
            if _broken.get((self.spec, "search"), 0) > time.time():
                raise ViewerError("not ready")
            return _visible_hits(viewer.search(self.spec, config, split, q, offset, PAGE, self.tok), q, offset), None
        except ViewerError:   # its search index isn't ready (or is off): a small split is searched here instead
            _broken[(self.spec, "search")] = time.time() + 600
            ql = q.lower()
            return self._local(config, split, offset, lambda r: ql in json.dumps(_searchable(r), ensure_ascii=False, default=str).lower())

    def filter(self, config, split, filters, columns, offset):
        try:
            if _broken.get((self.spec, "filter"), 0) > time.time():
                raise ViewerError("not ready")
            return viewer.filter_rows(self.spec, config, split, _where(filters, columns), offset, PAGE, self.tok), None
        except ViewerError:
            _broken[(self.spec, "filter")] = time.time() + 600
            return self._local(config, split, offset, lambda r: _matches(r, filters))

    def both(self, config, split, q, filters, columns, offset):
        """A search within a filter: the viewer does one or the other, so a small split is read here and matched on
        both; a big one is searched, with the filter left off (and a note saying so)."""
        ql = q.lower()
        try:
            return self._local(config, split, offset, lambda r: _matches(r, filters) and ql in json.dumps(_searchable(r), ensure_ascii=False, default=str).lower())
        except ViewerError:
            page, note = self.search(config, split, q, offset)
            return page, "This dataset is too big to search within a filter here, so the filter is left off this search."

    def _local(self, config, split, offset, match):
        rows = _all_rows(self.spec, config, split, self.tok)
        hits = [(i, r, []) for i, r in rows if match(r)]
        return {"rows": hits[offset:offset + PAGE], "total": len(hits), "features": []}, None

    def facets(self, config, split, roles, sample) -> list[dict[str, Any]]:
        out = []
        skip = {roles.get("task"), roles.get("id"), roles.get("title"), *roles.get("answer", [])}
        for st in viewer.statistics(self.spec, config, split, self.tok):
            col, kind, stats = st.get("column_name"), st.get("column_type"), st.get("column_statistics") or {}
            if col in skip or ANSWER.search(col or "") or FACET_SKIP.search(col or ""):
                continue
            freq = stats.get("frequencies")
            if kind in ("string_label", "bool", "class_label") and isinstance(freq, dict) and 2 <= len(freq) <= 60:
                out.append({"column": col, "values": sorted(([str(k), v] for k, v in freq.items()), key=lambda x: -x[1])[:40], "sampled": False})
        return out[:8] or _sample_facets(sample, roles)

    def can_search(self) -> bool:
        return bool(viewer.valid(self.spec, self.tok).get("search"))


class Files:
    kind = "files"

    def __init__(self, spec: str, meta: dict[str, Any], tok: str | None):
        self.spec, self.meta, self.tok = spec, meta, tok
        self.configs = direct.data_files(spec, meta, tok)
        if not self.configs:
            raise DirectError("no data files (JSON Lines, JSON, parquet or CSV) in this dataset")

    def splits(self) -> list[dict[str, str]]:
        return [{"config": c, "split": s} for c, ss in self.configs.items() for s in ss]

    def source(self, config: str, split: str) -> direct.Source:
        return direct.Source(self.spec, self.meta, self.configs.get(config, {}).get(split) or [], self.tok)

    def page(self, config, split, offset, length):
        src = self.source(config, split)
        got = src.read(offset, length)
        return {"rows": [(i, r, []) for i, r in got], "total": src.total(), "features": _features([r for _, r in got])}

    def _scan(self, config, split, offset, match, what):
        hits, cut = direct.scan(self.source(config, split), match)
        note = f"{what} looked through the first {direct.SCAN:,} rows only." if cut else None
        return {"rows": [(i, r, []) for i, r in hits[offset:offset + PAGE]], "total": len(hits), "features": []}, note

    def search(self, config, split, q, offset):
        ql = q.lower()
        return self._scan(config, split, offset, lambda r: ql in json.dumps(_searchable(r), ensure_ascii=False, default=str).lower(), "Search")

    def filter(self, config, split, filters, columns, offset):
        return self._scan(config, split, offset, lambda r: _matches(r, filters), "This filter")

    def both(self, config, split, q, filters, columns, offset):
        ql = q.lower()
        return self._scan(config, split, offset, lambda r: _matches(r, filters) and ql in json.dumps(_searchable(r), ensure_ascii=False, default=str).lower(), "Search")

    def facets(self, config, split, roles, sample):
        return _sample_facets(sample, roles)

    def can_search(self) -> bool:
        return True


LOCAL_MAX = 20_000   # rows of a split worth reading whole, to search and filter here when the viewer can't
_broken: dict[tuple[str, str], float] = {}
_choice: dict[tuple[str, str], tuple[str, float]] = {}   # (dataset, revision) -> (viewer|files, until)   # (dataset, "search"|"filter") -> until when the viewer's is skipped


def _word_in(w: str, text: str) -> bool:
    """The viewer's search stems words ("graders" finds "grader"): the same allowance here, and no looser."""
    if w in text:
        return True
    for suf in ("ing", "es", "ed", "s"):
        if len(w) > len(suf) + 2 and w.endswith(suf) and w[: -len(suf)] in text:
            return True
    return False


def _visible_hits(page: dict[str, Any], q: str, offset: int) -> dict[str, Any]:
    """The dataset viewer's search looks at every column, the withheld ones too: a row it found only through its answer
    would tell anyone (an agent over MCP, say) which rows hold that answer. Only rows whose visible fields have every
    word are kept, and when any was dropped the total stops at what's shown, so the count says nothing either."""
    words = [w for w in q.lower().split() if w]
    kept = [h for h in page["rows"] if all(_word_in(w, json.dumps(_searchable(h[1]), ensure_ascii=False, default=str).lower()) for w in words)]
    if len(kept) == len(page["rows"]):
        return page
    return {**page, "rows": kept, "total": offset + len(kept)}


def _searchable(row: dict[str, Any]) -> dict[str, Any]:
    """What search looks at: the row without its answers (searching for an answer mustn't find its rows)."""
    from .base import withhold

    return withhold(row, hide=HIDDEN)[0]


def _all_rows(spec: str, config: str, split: str, tok: str | None) -> list[tuple[int, dict[str, Any]]]:
    """Every row of a small split, through the viewer's row pages (8 at a time), kept for half an hour."""
    from concurrent.futures import ThreadPoolExecutor

    from .. import catalog

    def fetch():
        first = viewer.rows(spec, config, split, 0, 100, tok)
        total = first["total"] or 0
        if total > LOCAL_MAX:
            raise ViewerError(f"the dataset viewer can't filter this dataset right now, and its {total:,} rows are too many to read here")
        pages = [first] + list(ThreadPoolExecutor(8).map(lambda o: viewer.rows(spec, config, split, o, 100, tok), range(100, total, 100)))
        return [(i, r) for pg in pages for i, r, _ in pg["rows"]]

    return catalog._cached(("all-rows", spec, config, split), 1800, fetch, big=True)


def _features(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    names: list[str] = []
    for r in rows:
        names += [k for k in r if k not in names]
    return [{"name": n, "type": {}} for n in names]


def _sample_facets(sample: list[dict[str, Any]], roles: dict[str, Any]) -> list[dict[str, Any]]:
    """Facets from the first rows, when there are no statistics: short labels that repeat."""
    skip = {roles.get("task"), roles.get("id"), roles.get("title"), *roles.get("answer", [])}
    out = []
    for col in _features(sample):
        c = col["name"]
        if c in skip or ANSWER.search(c) or FACET_SKIP.search(c):
            continue
        vals = [r.get(c) for r in sample if isinstance(r.get(c), (str, bool)) and len(str(r.get(c))) <= 40]
        cnt = Counter(str(v) for v in vals)
        if len(vals) >= len(sample) * 0.8 and 2 <= len(cnt) <= 20 and max(cnt.values()) >= 2:
            out.append({"column": c, "values": [[k, v] for k, v in cnt.most_common(20)], "sampled": True})
    return out[:6]


def _where(filters: dict[str, Any], columns: list[str]) -> str:
    """A filter for the viewer from facet picks ({column: value or [values]}): only known columns, values quoted;
    several values of one column are any of them, columns are all of them."""
    parts = []
    for col, vals in filters.items():
        vals = [vals] if isinstance(vals, str) else [v for v in vals if isinstance(v, str)] if isinstance(vals, list) else []
        vals = [v for v in vals if len(v) <= 300][:20]
        if col not in columns or not vals:
            continue
        name = '"%s"' % col.replace('"', '""')
        ors = " OR ".join("%s='%s'" % (name, v.replace("'", "''")) for v in vals)
        parts.append(ors if len(vals) == 1 else f"({ors})")
    return " AND ".join(parts)


def _matches(row: dict[str, Any], filters: dict[str, Any]) -> bool:
    for k, vals in filters.items():
        vals = [vals] if isinstance(vals, str) else vals
        if str(row.get(k)) not in vals:
            return False
    return True


def _info(spec: str, token: str | None) -> tuple[dict[str, Any], str | None]:
    """The dataset's Hub record (checking the visitor may read it) and the token to read its rows with."""
    from .. import catalog

    meta = catalog.info(spec, token)
    return meta, (token if meta.get("restricted") else None)


def _backend(spec: str, token: str | None):
    """The dataset viewer when it can read this dataset, else its files. Checks the visitor may read it."""
    from .. import catalog

    meta = catalog.info(spec, token)
    tok = token if meta.get("restricted") else None

    def choose():
        v = Viewer(spec, tok)
        for attempt in range(2):   # one slow answer shouldn't send a dataset to its files
            try:
                sp = v.splits()
                if sp:
                    v.page(sp[0]["config"], sp[0]["split"], 0, 1)
                    return "viewer"
                break
            except ViewerError as e:
                if e.status == 404:   # the viewer can't read it at all
                    break
        return "files"

    key = (spec, meta["sha"])
    hit = _choice.get(key)
    if not hit or hit[1] < time.time():
        which = choose()
        _choice[key] = (which, time.time() + (1800 if which == "viewer" else 120))   # files: try the viewer again soon
    which = _choice[key][0]
    return meta, (Viewer(spec, tok) if which == "viewer" else Files(spec, meta, tok))


_LEFTOVER = re.compile(r"(dropped|discarded|excluded|removed|failed|invalid|rejected|unsolved)", re.IGNORECASE)


def _first(splits: list[dict[str, str]]) -> dict[str, str]:
    """The split a dataset opens on: its first config's train split, else its first that isn't a leftover (rows
    dropped from the dataset proper)."""
    first = [s for s in splits if s["config"] == splits[0]["config"]]
    return next((s for s in first if s["split"] == "train"), None) or next((s for s in first if not _LEFTOVER.search(s["split"])), None) or splits[0]


_split_files: dict[tuple[str, str, str, str], float] = {}   # (dataset, revision, config, split) -> until: read from files


def _first_page(spec: str, meta: dict[str, Any], be, pair: dict[str, str]):
    """The split's first rows, and the backend that reads that split. The viewer reads a dataset config by config, and
    can fail on one ("The dataset generation failed": a schema it can't cast) while reading the rest: that split is
    then read from its files, and remembered for a while, so its pages, tasks and search come from there too."""
    key = (spec, str(meta.get("sha")), pair["config"], pair["split"])
    if be.kind == "viewer" and _split_files.get(key, 0) > time.time():
        try:
            fb = Files(spec, meta, be.tok)
            return fb, fb.page(pair["config"], pair["split"], 0, 20)
        except DirectError:
            _split_files.pop(key, None)
    try:
        return be, be.page(pair["config"], pair["split"], 0, 20)
    except ViewerError as e:
        if be.kind != "viewer" or e.status == 404:
            raise
        try:
            fb = Files(spec, meta, be.tok)
            if not any(s["config"] == pair["config"] and s["split"] == pair["split"] for s in fb.splits()):
                raise e from None
            page = fb.page(pair["config"], pair["split"], 0, 20)
        except DirectError:
            raise e from None
        _split_files[key] = time.time() + 1800
        return fb, page


def _dataset(spec: str, config: str | None, split: str | None, token: str | None):
    meta, be = _backend(spec, token)
    splits = be.splits()
    if not splits:
        raise ViewerError("this dataset has no rows to read", 404)
    pair = next((s for s in splits if s["config"] == config and s["split"] == split), None) \
        or next((s for s in splits if s["config"] == config), None) or _first(splits)
    be, page = _first_page(spec, meta, be, pair)
    ds = Dataset(spec=spec, tags=meta.get("all_tags") or meta.get("tags") or [], card=meta, splits=splits, features=page["features"] or _features([r for _, r, _ in page["rows"]]),
                 sample=[r for _, r, _ in page["rows"]])
    proc = pick(ds)
    roles = proc.roles(ds)
    roles["_config"], roles["_split"] = pair["config"], pair["split"]
    return be, ds, pair["config"], pair["split"], proc, roles, page["total"]


def describe(spec: str, token: str | None = None, config: str | None = None, split: str | None = None, overview: bool = False) -> dict[str, Any]:
    """The dataset as read: its reader, configs, roles and facets; with `overview`, the reader's own sections for the
    dataset's page (only the page asks: lists and random picks don't pay for them)."""
    be, ds, config, split, proc, roles, total = _dataset(spec, config, split, token)
    configs: dict[str, list[str]] = {}
    for s in ds.splits:
        configs.setdefault(s["config"], []).append(s["split"])
    sections: list[dict[str, Any]] = []
    if overview:
        try:   # from what's read already: the first rows and the viewer's statistics (a framework's grading, its run commands)
            stats = viewer.statistics(spec, config, split, be.tok) if be.kind == "viewer" else []
            sections = proc.overview(ds, roles, stats, config, split)
        except Exception:  # noqa: BLE001 - an overview that can't be worked out isn't shown; the rows still are
            sections = []
    fw = proc.framework_for(ds) if hasattr(proc, "framework_for") else proc.framework
    return {
        "processor": {"id": proc.id, "name": proc.name, "framework": fw, "about": proc.about},
        "configs": [{"config": c, "splits": ss} for c, ss in configs.items()],
        "config": config, "split": split, "total": total, "source": be.kind,
        "roles": {k: v for k, v in roles.items() if not k.startswith("_")},
        "columns": ds.columns,
        "facets": be.facets(config, split, roles, ds.sample),
        "search": be.can_search(),
        "overview": sections,
    }


def rows(spec: str, config: str, split: str, offset: int = 0, q: str = "", filters: dict[str, str] | None = None,
         token: str | None = None) -> dict[str, Any]:
    be, ds, config, split, proc, roles, total = _dataset(spec, config, split, token)
    filters = {k: v for k, v in (filters or {}).items() if k in ds.columns}
    note = None
    try:
        if q.strip() and filters:
            page, note = be.both(config, split, q.strip()[:200], filters, ds.columns, offset)
            mode = "search"
        elif q.strip():
            page, note = be.search(config, split, q.strip()[:200], offset)
            mode = "search"
        elif filters:
            page, note = be.filter(config, split, filters, ds.columns, offset)
            mode = "filter"
        else:
            page, mode = be.page(config, split, offset, PAGE), "rows"
    except (ViewerError, DirectError) as e:
        if not (q.strip() or filters):
            raise
        page = be.page(config, split, offset, PAGE)
        mode, note = "rows", f"{'Search' if q.strip() else 'Filtering'} isn't available for this dataset right now ({e}); showing every row."
    cards = []
    for i, row, _ in page["rows"]:
        try:
            cards.append(proc.card(row, i, roles))
        except Exception:  # noqa: BLE001 - one odd row shows as a plain row, it doesn't sink the page
            cards.append({"i": i, "id": None, "title": f"Row {i}", "snippet": "", "chips": []})
    retitle(cards)
    return {"rows": cards, "total": page["total"], "offset": offset, "page": PAGE, "mode": mode, "note": note,
            "config": config, "split": split}


def _row(spec: str, config: str, split: str, index: int, token: str | None):
    be, ds, config, split, proc, roles, total = _dataset(spec, config, split, token)
    if index < 0 or (total is not None and index >= total):
        raise LookupError("no such row")
    page = be.page(config, split, index, 1)
    if not page["rows"]:
        raise LookupError("no such row")
    _, row, truncated = page["rows"][0]
    return ds, config, split, proc, roles, total, row, truncated


def task(spec: str, config: str, split: str, index: int, token: str | None = None) -> dict[str, Any]:
    ds, config, split, proc, roles, total, row, truncated = _row(spec, config, split, index, token)
    view = proc.view(row, index, ds, roles)
    view.update(spec=spec, config=config, split=split, index=index, total=total, truncated=truncated,
                processor={"id": proc.id, "name": proc.name, "framework": proc.framework, "about": proc.about},
                restricted=bool(_info(spec, token)[0].get("restricted")))
    view.setdefault("run", None)
    return view


def task_file(spec: str, config: str, split: str, index: int, rel: str, token: str | None = None) -> dict[str, Any]:
    ds, config, split, proc, roles, total, row, _ = _row(spec, config, split, index, token)
    if not hasattr(proc, "file"):
        raise LookupError("this dataset's rows have no files")
    return proc.file(row, ds, re.sub(r"^/+", "", rel or ""))


# ── running a Harbor task packed into a row ──────────────────────────────────
def packed(spec: str, config: str, split: str, index: int, token: str | None = None) -> dict[str, Any]:
    """A packed Harbor task as the runner needs it: its files, title, and whether (and how) it can run here."""
    import tomllib

    from .. import catalog
    from .processors import PackedHarbor, unpack

    ds, config, split, proc, roles, total, row, _ = _row(spec, config, split, index, token)
    if not isinstance(proc, PackedHarbor):
        raise LookupError("this dataset's rows aren't Harbor tasks")
    files = unpack(row.get(proc.blob(ds)))
    if "task.toml" not in files:
        raise LookupError("this row's archive holds no Harbor task")
    texts = {n: b.decode("utf-8", "replace") for n, b in files.items() if b"\x00" not in b[:4096]}
    try:
        doc = tomllib.loads(texts.get("task.toml", ""))
    except tomllib.TOMLDecodeError:
        raise LookupError("this task's task.toml can't be read") from None
    env = doc.get("environment") or {}
    look = {"env": {"image": env.get("docker_image"), "network": catalog._network(env),
                    "compose": any(re.match(r"environment/(docker-)?compose\.ya?ml$", n) for n in files)}}
    meta = _info(spec, token)[0]
    view = proc.view(row, index, ds, roles)
    return {"title": view["title"], "files": files, "runnable": catalog.runnable(look, texts), "image": env.get("docker_image"),
            "restricted": bool(meta.get("restricted")), "sha": meta["sha"], "bytes": sum(len(b) for b in files.values()),
            "verifier_env": sorted(((doc.get("verifier") or {}).get("env") or {}).keys())}


def materialize(spec: str, config: str, split: str, index: int, token: str | None = None):
    """The packed task written out as a folder for the runner, without its reference solution."""
    import shutil

    from .. import catalog, config as cfg

    task = packed(spec, config, split, index, token)
    root = cfg.CACHE_DIR / "packed" / catalog._slug(spec) / re.sub(r"[^\w.-]", "_", config) / re.sub(r"[^\w.-]", "_", split) / str(int(index))
    shutil.rmtree(root, ignore_errors=True)
    for rel, data in task["files"].items():
        if rel.startswith("solution/"):
            continue
        dst = (root / rel).resolve()
        if not str(dst).startswith(str(root.resolve()) + "/"):
            continue   # unpack() already drops these; belt and braces
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        if rel.endswith(".sh"):
            dst.chmod(0o755)
    return root


# ── the contract ─────────────────────────────────────────────────────────────
def ref_of(config: str, split: str, i: int) -> str:
    """A row's task ref: `<split>/<row>` in the default config, `<config>/<split>/<row>` otherwise."""
    return f"{split}/{i}" if config == "default" else f"{config}/{split}/{i}"


def parse_ref(ref: str) -> tuple[str, str, int]:
    parts = [p for p in str(ref).split("/") if p]
    if len(parts) < 2 or not parts[-1].isdigit():
        raise LookupError("not a row of this dataset")
    return ("/".join(parts[:-2]) or "default"), parts[-2], int(parts[-1])


def subset_id(config: str, split: str) -> str:
    return split if config == "default" else f"{config}/{split}"


def parse_subset(s: str | None) -> tuple[str | None, str | None]:
    if not s:
        return None, None
    return tuple(s.rsplit("/", 1)) if "/" in s else ("default", s)  # type: ignore[return-value]


_ICONS = {"task": "message", "trace": "message", "grading": "target", "tests": "flask", "rubric": "list", "environment": "box",
          "files": "folder", "metadata": "list", "tools": "wrench", "result": "trophy", "music": "music", "request": "send",
          "run": "play", "data": "list", "reward": "target"}


class RowsAdapter(c.Adapter):
    """Rows datasets behind the environment contract. One adapter for the family; which row reader (processor) reads a
    dataset is picked from its first rows, and names the framework."""

    id = "rows"
    name = "Rows"
    framework = "Rows"
    about = ""
    kinds = ("dataset",)
    run_note = ("No harness runs this format here yet: its rows hold the task, not the environment to run it in. A row "
                "reader can name a runnable form (a Harbor task, packed or converted) to run it.")

    def detect(self, kind: str, meta: dict[str, Any]) -> float:
        return 0.3 if kind == "dataset" else 0.0      # anything with rows the viewer (or its files) can read

    def summary(self, env, subset: str | None = None) -> dict[str, Any]:
        config, split = parse_subset(subset)
        d = describe(env.id, env.token, config, split, overview=True)
        subsets = [{"id": subset_id(x["config"], s), "label": s if x["config"] == "default" else f"{x['config']} · {s}"}
                   for x in d["configs"] for s in x["splits"]]
        roles = {k: v for k, v in d["roles"].items() if v}
        return {"state": "ready", "total": d["total"], "inline": False, "search": d["search"], "source": d["source"],
                "subsets": subsets, "subset": subset_id(d["config"], d["split"]),
                "facets": [c.facet(f["column"], f["column"].replace("_", " "), "level" if re.search(r"difficult|tier|level", f["column"], re.I) else "count",
                                   values=f["values"], sampled=f.get("sampled", False)) for f in d["facets"]],
                "how": {"framework": d["processor"]["framework"], "name": d["processor"]["name"], "about": d["processor"]["about"], "roles": roles,
                        "id": d["processor"]["id"]},
                "overview": d.get("overview") or []}

    def tasks(self, env, *, subset=None, q="", filters=None, offset=0, everything=False) -> dict[str, Any]:
        config, split = parse_subset(subset)
        if config is None:
            d = describe(env.id, env.token)
            config, split = d["config"], d["split"]
        r = rows(env.id, config, split, offset, q, filters or {}, env.token)
        cards = [c.card(ref_of(r["config"], r["split"], x["i"]), x["title"], id=x.get("id"), brief=x.get("snippet") or "", chips=x.get("chips") or [],
                        lead=x.get("lead"), stats=x.get("stats"))
                 for x in r["rows"]]
        return {"cards": cards, "total": r["total"], "offset": r["offset"], "page": r["page"], "note": r.get("note"), "mode": r["mode"],
                "subset": subset_id(r["config"], r["split"])}

    def task(self, env, ref: str) -> dict[str, Any]:
        config, split, i = parse_ref(ref)
        v = task(env.id, config, split, i, env.token)
        sections = []
        for s in v["sections"]:
            k, body = s["kind"], s["body"]
            blocks = (list(body or []) if k == "blocks"
                      else [c.markdown(body if isinstance(body, str) else "")] if k == "markdown"
                      else [c.messages(body)] if k == "messages"
                      else [c.code(body.get("text", ""), body.get("path", ""), body.get("path"))] if k == "code"
                      else [c.files(body.get("tree") or [])] if k == "files"
                      else [c.value(body)])
            sections.append(c.section(s["id"], s["title"], blocks, icon=_ICONS.get(s["id"], "list"), note=s.get("note") or ""))
        total = v.get("total")
        sub = subset_id(config, split)
        links = []
        twin = v.get("run")
        if twin:   # the same task in a Harbor dataset (MiMo's conversion): its folder's files are this task's files too
            links.append(c.link(f"In Harbor's format: {twin['dataset']}", f"/t/{twin['dataset']}/{twin['path']}",
                                "the same task, converted; its rollouts are this task's too", rel="same"))
            if not any(s["id"] == "files" for s in sections):
                try:
                    from .. import catalog

                    t = catalog.task(twin["dataset"], twin["path"])
                    sections.append(c.section("files", "Files", [c.files(t["tree"], t.get("tree_truncated"), c.hub_folder(twin["dataset"], t["sha"], twin["path"]))], icon="folder",
                                              note=f"its Harbor folder, {twin['path']}"))
                except Exception:  # noqa: BLE001 - the twin's index isn't here: no files section
                    pass
        return {"ref": ref, "title": v["title"], "id": v.get("id"), "chips": [x for x in v.get("chips") or [] if x], "sections": sections,
                "glance": [[a, b] for a, b in v.get("glance") or []], "withheld": v.get("withheld") or [], "links": links,
                "framework": v.get("framework") or v["processor"]["framework"], "restricted": v.get("restricted"),
                "hub": f"https://huggingface.co/datasets/{env.id}/viewer/{config}/{split}?row={i}",
                "nav": {"prev": ref_of(config, split, i - 1) if i > 0 else None,
                        "next": ref_of(config, split, i + 1) if total is None or i + 1 < total else None,
                        "index": i, "total": total, "subset": sub},
                "truncated": v.get("truncated") or [], "summary": v.get("summary") or None,
                "_packed_run": v.get("packed_run"), "_twin": v.get("run"), "_framework_run": v.get("framework_run")}

    def _twin(self, env, ref: str) -> dict[str, Any] | None:
        config, split, i = parse_ref(ref)
        return task(env.id, config, split, i, env.token).get("run")

    def file(self, env, ref: str, path: str) -> dict[str, Any]:
        config, split, i = parse_ref(ref)
        twin = self._twin(env, ref)
        if twin:   # its files are its Harbor twin's
            from .. import catalog

            return catalog.task_file(twin["dataset"], twin["path"], path)
        return task_file(env.id, config, split, i, path, env.token)

    def folder(self, env, ref: str, path: str) -> dict[str, Any]:
        twin = self._twin(env, ref)
        if twin:
            from .. import catalog

            return catalog.task_folder(twin["dataset"], twin["path"], path)
        raise LookupError("a row's files are listed up front")

    def random(self, env, subset: str | None = None) -> str:
        import random as _r

        config, split = parse_subset(subset)
        d = describe(env.id, env.token, config, split)
        return ref_of(d["config"], d["split"], _r.randrange(max(1, d["total"] or 1)))

    # running: what the row's reader can name as runnable (a packed Harbor task, a Harbor twin)
    def run_options(self, env, ref: str, view: dict[str, Any]) -> list[dict[str, Any]]:
        from .. import catalog
        from .harbor import harbor_option

        if view.get("_packed_run"):
            p = view["_packed_run"]
            return [harbor_option(p["runnable"], p.get("toml") or "")]
        if view.get("_framework_run"):   # runs with its own framework (NeMo Gym, Verifiers, verl): the page shows how
            return [view["_framework_run"]]
        twin = view.get("_twin")
        if twin:
            try:
                t = catalog.task(twin["dataset"], twin["path"])
            except Exception as e:  # noqa: BLE001 - its twin isn't readable now
                return [c.run_option("harbor", "Harbor, on an HF Sandbox", ok=False, why=f"its Harbor form can't be read ({type(e).__name__})")]
            opt = harbor_option(t["runnable"], t.get("toml") or "")
            opt["notes"] = [f"Runs `{twin['dataset']}` · `{twin['path']}`: this task in Harbor's format."] + opt["notes"]
            return [opt]
        return []

    def materialize(self, env, ref: str):
        from .. import catalog

        config, split, i = parse_ref(ref)
        view = self.task(env, ref)
        if view.get("_packed_run"):
            return materialize(env.id, config, split, i, env.token)
        if view.get("_twin"):
            return catalog.task_dir(view["_twin"]["dataset"], view["_twin"]["path"])
        raise LookupError("this row has no runnable form")

    def run_task(self, env, ref: str) -> dict[str, Any]:
        from .. import catalog

        config, split, i = parse_ref(ref)
        view = self.task(env, ref)
        if view.get("_packed_run"):
            p = packed(env.id, config, split, i, env.token)
            return {"title": p["title"], "restricted": p["restricted"], "sha": p["sha"], "bytes": p["bytes"], "runnable": p["runnable"],
                    "image": p["image"] or p["runnable"].get("base")}
        if view.get("_twin"):
            t = catalog.task(view["_twin"]["dataset"], view["_twin"]["path"])
            return {"title": view["title"], "restricted": bool(env.meta.get("restricted")), "sha": env.meta.get("sha"), "bytes": t.get("bytes") or 0,
                    "runnable": t["runnable"], "image": t["env"]["image"] or t["runnable"].get("base")}
        raise LookupError("this row has no runnable form")

    def aliases(self, env, ref: str) -> list[tuple[str, str]]:
        try:
            twin = self.task(env, ref).get("_twin")
        except Exception:  # noqa: BLE001
            return []
        return [(twin["dataset"], twin["path"])] if twin else []

    def mcp_tools(self, env) -> list[dict[str, Any]]:
        return []

    def mcp_call(self, env, name: str, args: dict[str, Any]) -> Any:
        raise LookupError(f"no tool named {name!r}")
