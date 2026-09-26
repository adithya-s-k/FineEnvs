"""The dataset, as the app needs it.

Two views of every task:
  * `view(task_id)`: everything a person needs to understand it (brief, environment, how it is
    graded), shaped per domain. Answers are withheld: rubric `pass_anchor` / `check_code`,
    `gold_answer`, the Cyber PoC. Hidden tests *are* shown: they are how the task is verified.
  * `runtime(task_id)`: what the runner needs to set up and grade a rollout.

The browsing index (web/data/*.json.gz, built by build_data.py) and the raw parquet rows are both
loaded once. A General environment's folder (tools, databases, workspace files, verifier) is fetched
from the dataset the first time it is opened, then cached.
"""

from __future__ import annotations

import ast
import base64
import gzip
import json
import re
import sqlite3
import threading
from functools import lru_cache
from pathlib import Path

from . import config

_lock = threading.Lock()
_env_locks: dict[str, threading.Lock] = {}

PARQUETS = {"code": "code.parquet", "cyber": "cyber.parquet", "general": "general/train.parquet",
            "webdev": "webdev.parquet", "music": "music.parquet"}


# ── loading ──────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def index() -> dict:
    doc = json.loads(gzip.decompress((config.WEB_DIR / "data" / "index.json.gz").read_bytes()))
    doc["by_id"] = {e["id"]: e for e in doc["envs"]}
    return doc


@lru_cache(maxsize=8)
def detail(domain: str) -> dict:
    return json.loads(gzip.decompress((config.WEB_DIR / "data" / f"{domain}.json.gz").read_bytes()))


@lru_cache(maxsize=1)
def rows() -> dict[str, dict]:
    """task id -> {"domain", "prompt", "instance" (parsed instance_json), "extra"}."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    out: dict[str, dict] = {}
    for domain, fname in PARQUETS.items():
        df = pd.read_parquet(hf_hub_download(config.DATASET, fname, repo_type="dataset"))
        for n, r in enumerate(df.to_dict("records")):
            extra = r.get("extra_info") or {}
            inst = json.loads(extra["instance_json"]) if extra.get("instance_json") else {}
            tid = inst.get("instance_id") or f"music-{extra.get('src_id', n)}"
            out[tid] = {"domain": domain, "prompt": r["prompt"][-1]["content"], "instance": inst,
                        "extra": {k: v for k, v in extra.items() if k != "instance_json"}}
    return out


@lru_cache(maxsize=1)
def _repo_files() -> dict[str, list[str]]:
    """General env id -> its files, from one listing of the dataset (41k files; listing per env is slow)."""
    from huggingface_hub import HfApi

    out: dict[str, list[str]] = {}
    for f in HfApi().list_repo_files(config.DATASET, repo_type="dataset"):
        if f.startswith("general/envs/"):
            out.setdefault(f.split("/")[2], []).append(f)
    return out


def env_dir(task_id: str) -> Path | None:
    """A General (simulated workplace) environment's folder, fetched on first use (files in parallel)."""
    if not task_id.startswith("s3k_"):
        return None
    local = config.CACHE_DIR / "general" / task_id
    if (local / ".complete").exists():
        return local
    with _lock:
        lk = _env_locks.setdefault(task_id, threading.Lock())
    with lk:
        if not (local / ".complete").exists():
            from concurrent.futures import ThreadPoolExecutor
            from huggingface_hub import hf_hub_download

            files = _repo_files().get(task_id) or []
            if not files:
                raise FileNotFoundError(task_id)
            prefix = f"general/envs/{task_id}/"

            def get(f: str) -> None:
                dst = local / f[len(prefix):]
                dst.parent.mkdir(parents=True, exist_ok=True)
                src = hf_hub_download(config.DATASET, f, repo_type="dataset", cache_dir=str(config.CACHE_DIR / "hub"))
                dst.write_bytes(Path(src).read_bytes())

            with ThreadPoolExecutor(16) as ex:
                list(ex.map(get, files))
            (local / ".complete").touch()
    return local


def warm() -> None:
    """Load the dataset rows and the file listing in the background at startup."""
    rows()
    _repo_files()


def record(task_id: str) -> dict | None:
    return index()["by_id"].get(task_id)


# ── per-domain facts ─────────────────────────────────────────────────────────
DESC = re.compile(r"^(?P<san>\S+):\s+(?P<kind>\S+)\s+in function `(?P<fn>[^`]+)`\s+in file `(?P<file>[^`]+)`")


def cyber_expected(desc: str) -> dict:
    """Parsed exactly as mimoagent's `_parse_description` does, for server_arvo's expected_func.json."""
    m = DESC.match(desc.strip())
    if not m:
        raise ValueError(f"cannot parse crash description: {desc!r}")
    return {"function": m["fn"], "file": m["file"], "sanitizer": m["san"], "error_type": m["kind"], "max_submits": 0}


def patch_files(patch: str) -> list[dict]:
    """Files a unified diff touches, with +/- line counts."""
    files, cur = [], None
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            cur = {"path": line.split(" b/", 1)[-1].strip(), "added": 0, "removed": 0, "new": False}
            files.append(cur)
        elif cur is not None:
            if line.startswith("new file mode"):
                cur["new"] = True
            elif line.startswith("+") and not line.startswith("+++"):
                cur["added"] += 1
            elif line.startswith("-") and not line.startswith("---"):
                cur["removed"] += 1
    return files


def patch_file_text(patch: str, path: str) -> str | None:
    """Content of a file the patch creates, rebuilt from its '+' lines (for small scripts)."""
    blocks = re.split(r"(?=^diff --git )", patch, flags=re.M)
    for b in blocks:
        if b.startswith("diff --git") and b.splitlines()[0].endswith("b/" + path) and "new file mode" in b[:400]:
            body = b.split("@@", 2)[-1].split("\n", 1)[-1] if "@@" in b else ""
            return "\n".join(l[1:] for l in body.splitlines() if l.startswith("+"))
    return None


def parse_tools(py: str) -> list[dict]:
    """Public functions of an MCP tools module: name, parameters, first paragraph of the docstring."""
    try:
        tree = ast.parse(py)
    except SyntaxError:
        return []
    tools = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            args = node.args
            defaults = [None] * (len(args.args) - len(args.defaults)) + list(args.defaults)
            params = []
            for a, d in zip(args.args, defaults):
                params.append({"name": a.arg, "type": ast.unparse(a.annotation) if a.annotation else "",
                               "default": ast.unparse(d) if d is not None else None})
            doc = (ast.get_docstring(node) or "").strip().split("\n\n")[0].replace("\n", " ")
            tools.append({"name": node.name, "params": params, "doc": doc[:400]})
    return tools


def db_summary(db: Path) -> list[dict]:
    """Tables of a system's SQLite state: columns and row count (rows come from db_rows)."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        out = []
        for (name,) in con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"):
            cols = [{"name": c[1], "type": c[2]} for c in con.execute(f'PRAGMA table_info("{name}")')]
            n = con.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
            out.append({"table": name, "columns": cols, "rows": n})
        return out
    finally:
        con.close()


def db_rows(db: Path, table: str, limit: int = 50) -> dict:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if table not in tables:
            raise KeyError(table)
        cur = con.execute(f'SELECT * FROM "{table}" LIMIT ?', (limit,))
        return {"columns": [d[0] for d in cur.description], "rows": [list(r) for r in cur.fetchall()]}
    finally:
        con.close()


KIND = {"xlsx": "spreadsheet", "xls": "spreadsheet", "csv": "spreadsheet", "docx": "document", "doc": "document",
        "pdf": "pdf", "pptx": "slides", "png": "image", "jpg": "image", "jpeg": "image", "svg": "image", "gif": "image",
        "html": "web", "htm": "web", "md": "text", "txt": "text", "json": "text", "xml": "text", "zip": "archive"}


from .names import pretty_system  # noqa: E402


# ── the task view ────────────────────────────────────────────────────────────
WEBDEV_DIMS = [
    ("layout_integrity", "Layout integrity", "Nothing overlaps, clips or overflows; the page holds together."),
    ("typography_hierarchy", "Typography", "A clear type scale and hierarchy; text is readable."),
    ("color_harmony", "Colour", "A coherent palette with enough contrast."),
    ("whitespace", "Whitespace", "Spacing and rhythm; not cramped, not empty."),
    ("content_richness", "Content richness", "As much content as a complete site of this kind needs."),
    ("query_fulfillment", "Brief fulfilment", "Everything the brief asked for is visibly there."),
    ("premium_assets", "Asset quality", "Real imagery and icons rather than placeholders."),
]


def view(task_id: str) -> dict | None:
    rec = record(task_id)
    if not rec:
        return None
    d = rec["d"]
    det = detail(d).get(task_id, {})
    raw = rows().get(task_id, {})
    inst = raw.get("instance", {})
    v = {"id": task_id, "domain": d, "title": rec["t"], "facets": rec["f"], "brief": det.get("brief") or raw.get("prompt", ""),
         "meta": det.get("meta", []), "image": image_for(task_id), "runnable": runnable(task_id)}

    if d == "code":
        patch = inst.get("test_patch") or ""
        files = patch_files(patch)
        v["verify"] = {
            "kind": "tests",
            "summary": "The agent's edits are kept; the hidden tests below are applied on top and the test command runs. "
                       "Reward 1 if it exits 0, else 0.",
            "steps": ["Reset every file the hidden tests touch to its original version",
                      "Apply the hidden test patch (git apply)", f"Run `{inst.get('test_command', '')}`",
                      "Reward = 1 if the command exits 0, otherwise 0"],
            "test_command": inst.get("test_command"), "timeout_s": inst.get("verifier_timeout_sec"),
            "files": files, "script": patch_file_text(patch, "mimo_test_command.sh"),
            "patch": patch if len(patch) < 400_000 else None,
        }
        v["environment"] = {"cwd": inst.get("cwd"), "image": image_for(task_id)}

    elif d == "cyber":
        try:
            exp = cyber_expected(inst.get("description") or raw.get("prompt", ""))
        except ValueError:
            exp = {}
        num = re.sub(r"\D", "", task_id)
        v["verify"] = {
            "kind": "crash",
            "summary": "The agent writes an input file (a proof of concept) and submits it. A root-owned server runs the "
                       "real fuzz target on it as a separate user. Reward 1 only if it crashes in the expected function.",
            "expected": exp,
            "steps": ["Agent explores the source in /home/agent/src and writes a PoC file",
                      "Agent runs `bash submit.sh <poc>` (as many times as it likes)",
                      "The server runs /home/agent/binary on the PoC under the sanitizer, as user `verify`",
                      f"Reward = 1 if it crashes and the first application frame is `{exp.get('function', '?')}`"],
        }
        v["links"] = [{"label": "OSS-Fuzz issue", "url": f"https://issues.oss-fuzz.com/issues/{num}"}] if num else []
        v["environment"] = {"cwd": "/home/agent", "image": image_for(task_id)}

    elif d == "webdev":
        v["verify"] = {
            "kind": "visual",
            "summary": "The agent builds a static site into dist/. It is rendered full-page in Chromium and a vision model "
                       "scores it: the mean of visual quality, brief fulfilment and asset quality.",
            "dims": [{"key": k, "label": l, "desc": s, "group": "visual" if k in
                      ("layout_integrity", "typography_hierarchy", "color_harmony", "whitespace", "content_richness") else k}
                     for k, l, s in WEBDEV_DIMS],
            "formula": "score = mean(visual, brief fulfilment, asset quality); visual = mean of the first five",
            "note": "This is Xiaomi's evaluation-mode grader. Training used a group-relative ranking, which has no meaning for a single rollout.",
            "needs_judge": "vision",
        }
        v["environment"] = {"cwd": inst.get("cwd"), "image": image_for(task_id), "deliver": f"{inst.get('cwd', '')}/dist"}

    elif d == "music":
        from .vendor.music_scorer.score import SPEC
        ref = json.loads((Path(__file__).parent / "vendor" / "music_scorer" / "baselines" / "ref_full4k.json").read_text())
        e = raw.get("extra", {})
        v["verify"] = {
            "kind": "music",
            "summary": "The model writes one piece in ABC notation. It is rendered to MIDI (abc2midi) and 18 features are "
                       "compared with the range human music occupies. No model judges it.",
            "spec": {"bpm": e.get("bpm"), "meter": e.get("meter"), "bars": e.get("length"), "voices": e.get("nvoice_want"),
                     "style": rec["f"].get("style"), "language": e.get("lang")},
            "features": [{"name": f, "rule": rule, "group": grp,
                          "band": [ref[f].get("p10"), ref[f].get("p90")] if f in ref else None} for f, rule, grp in SPEC],
        }
        v["environment"] = {"sandbox": False}

    elif d == "general" and inst.get("dataset_type") == "terminal_bench":
        tf = json.loads(inst.get("tests_files") or "{}")
        dec = {k: base64.b64decode(b).decode("utf-8", "replace") for k, b in tf.items()}
        v["verify"] = {
            "kind": "terminal",
            "summary": "After the agent finishes, the hidden tests are copied to /tests and `sh /tests/test.sh` runs: an "
                       "anti-tamper check, then pytest. Reward 1 if every test passes.",
            "files": [{"path": k, "size": len(tf[k]) * 3 // 4} for k in tf],
            "script": dec.get("test.sh"), "tests": dec.get("test_outputs.py"),
        }
        v["environment"] = {"cwd": inst.get("cwd"), "image": image_for(task_id), "cpus": inst.get("cpus"),
                            "memory_mb": inst.get("memory_mb"), "internet": inst.get("allow_internet"),
                            "tags": inst.get("tags"), "timeout_s": inst.get("agent_timeout_sec")}

    elif d == "general":
        v.update(general_view(task_id))
    return v


def general_view(task_id: str) -> dict:
    root = env_dir(task_id)
    man = json.loads((root / "manifest.json").read_text())
    systems = []
    for i, s in enumerate(man.get("mcp_servers", [])):
        name = s["name"]
        py = root / "tools" / f"{name}.py"
        db = root / "system" / name / "state.db"
        systems.append({"name": name, "label": pretty_system(name),
                        "tools": parse_tools(py.read_text()) if py.exists() else [],
                        "tables": db_summary(db) if db.exists() else []})
    ws = root / "workspace"
    files = sorted(({"path": str(f.relative_to(ws)), "size": f.stat().st_size,
                     "kind": KIND.get(f.suffix.lower().lstrip("."), "other")}
                    for f in ws.rglob("*") if f.is_file()), key=lambda x: (x["kind"], x["path"]))
    meta = json.loads((root / "verifier_meta.json").read_text())
    checks = [{"id": it.get("id"), "tier": it.get("tier"), "method": it.get("method"), "weight": it.get("weight"),
               "question": it.get("question", "")} for it in meta.get("items", [])]
    return {
        "systems": systems, "files": files,
        "verify": {
            "kind": "rubric",
            "summary": "When the agent finishes, its final answer and the systems' databases are checked. Rule checks run "
                       "code against the data; LLM checks ask a judge model one yes/no question each. Reward = weighted "
                       "share of checks passed.",
            "checks": checks, "needs_judge": "text" if any(c["method"] == "llm" for c in checks) else None,
        },
        "environment": {"cwd": man.get("cwd"), "image": image_for(task_id), "ports": man.get("wait_ports")},
    }


def workspace_path(task_id: str, rel: str) -> Path:
    root = (env_dir(task_id) / "workspace").resolve()
    p = (root / rel).resolve()
    if root not in p.parents and p != root:
        raise PermissionError(rel)
    return p


def system_db(task_id: str, system: str) -> Path:
    root = env_dir(task_id)
    if not re.fullmatch(r"[A-Za-z0-9_]+", system):
        raise PermissionError(system)
    return root / "system" / system / "state.db"


# ── what can run where ───────────────────────────────────────────────────────
def image_for(task_id: str) -> str | None:
    raw = rows().get(task_id, {})
    d, inst = raw.get("domain"), raw.get("instance", {})
    img = inst.get("docker_image") or ""
    if d == "music" or not img:
        return None
    if d == "cyber":                                    # arvo-rl:v1-arvo-35858 -> arvo-v1-35858
        tag = "arvo-v1-" + img.rsplit("arvo-", 1)[-1]
    elif d == "webdev":
        tag = "webdev-rl-opensource"
    else:                                               # format-code-task-001457:latest, general-agent-env-3:oss
        tag = img.split(":", 1)[0]
    return f"{config.IMAGE_REPO}:{tag}"


def runnable(task_id: str) -> bool:
    raw = rows().get(task_id)
    return bool(raw) and (raw["domain"] == "music" or bool(image_for(task_id)))
