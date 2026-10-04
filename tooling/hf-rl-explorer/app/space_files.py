"""An environment Space as its files: what its repository declares, read without waking it.

An OpenEnv environment is a Space repository as much as a running server: `openenv.yaml` (its manifest, sometimes in
a subfolder: the shallowest one counts), `models.py` (its Action, Observation and State), `server/app.py` (how the
server is made: `create_app(..., env_name=, max_concurrent_envs=, gradio_builder=)`), `server/*_environment.py` (the
episode and its reward), `Dockerfile`, `pyproject.toml` (the openenv-core it needs), `discovery.json`, the README.
The Hub lists a Space's files without waking it, so a sleeping Space (most of them) still shows what it is.

    GET /api/spaces/<org>/<name>/files      the curated file tree and what the files declare, pinned to one revision
    GET /api/spaces/<org>/<name>/file?f=    one text file from that tree, at that revision

Nothing here runs a Space's code: Python is read with `ast`, YAML with `safe_load` (and a bounded walk, so anchors
can't blow up), TOML with tomllib. Files are read anonymously (never with a visitor's token), pinned to the revision
listed, only files in the curated listing are served, and binaries aren't. Private Spaces are refused. Files that may
hold answers (a solution folder, task data) are listed but not served, as everywhere else in the explorer.
"""

from __future__ import annotations

import ast
import json
import math
import posixpath
import re
import shlex
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException

from . import catalog, config
from .spaces_live import ANSWER

router = APIRouter()

MAX_TEXT = 400 * 1024            # bytes of one file shown; a longer text file is shown up to here
MAX_FETCH = 4 * 1024 * 1024      # bytes a file may have and still be fetched to show its start
MAX_LISTED = 5000                # files in one listing (after the junk is dropped)
INFO_TTL = 60                    # seconds the revision and listing are trusted before asking the Hub again
PARSE_TTL = 6 * 3600             # what a revision declares never changes; kept a while, then recomputed if asked

# Vendored copies of OpenEnv itself, build output, caches: not the environment.
JUNK = re.compile(
    r"(^|/)(__pycache__|\.git|\.venv|venv|node_modules|\.ipynb_checkpoints|\.pytest_cache|\.mypy_cache|\.ruff_cache)/"
    r"|(^|/)[^/]+\.egg-info/"
    r"|^build/|(^|/)build/(lib|bdist)[^/]*/"
    r"|^src/(openenv|openenv_core|core)/"
    r"|(^|/)\.gitattributes$|(^|/)\.DS_Store$|\.py[co]$")
BINARY = re.compile(
    r"\.(png|jpe?g|gif|webp|bmp|ico|tiff?|avif|heic|pdf|zip|tar|gz|tgz|bz2|xz|zst|7z|rar|whl|egg|so|dylib|dll|exe|bin|"
    r"pt|pth|ckpt|safetensors|onnx|gguf|pkl|pickle|joblib|npy|npz|h5|hdf5|parquet|arrow|feather|db|sqlite3?|mp3|wav|"
    r"flac|ogg|m4a|mp4|webm|mov|avi|mkv|ttf|otf|woff2?|eot|jar|class|wasm)$", re.I)
CODE = re.compile(r"\.(py|pyi|sh|bash|js|mjs|ts|tsx|jsx|rb|go|rs|java|c|cc|cpp|h|hpp|md|html|css|ipynb)$|(^|/)Dockerfile[^/]*$", re.I)
# manifests whose keys are declarations, not data: never withheld for an answer-like key name
MANIFESTS = re.compile(r"(^|/)(openenv\.ya?ml|discovery\.json|package(-lock)?\.json|pyproject\.toml|tsconfig\.json)$", re.I)
DATA = re.compile(r"\.(jsonl|json|csv|tsv|parquet|arrow|ndjson)$", re.I)
# a folder of reference solutions or answers; task data files (which hold the answers the grader checks)
ANSWER_DIR = re.compile(r"(^|/)(solutions?|gold|answers?|oracle|ground_?truth)/", re.I)
TASK_DATA_DIR = re.compile(r"(^|/)(tasks?|splits?|evals?|test_?cases|episodes|answers?|labels?)/", re.I)


# ── the Space's record and files ─────────────────────────────────────────────
def _info(spec: str) -> dict[str, Any]:
    """The Space's revision and every file in it with its size (one Hub call; the Space isn't touched)."""
    spec = catalog.check_spec(spec)

    def fetch():
        sp = catalog._api().space_info(spec, files_metadata=True)
        if getattr(sp, "private", False) or getattr(sp, "gated", False):
            raise PermissionError("this Space is private")
        card = (sp.card_data.to_dict() if getattr(sp, "card_data", None) else {}) or {}
        rt = getattr(sp, "runtime", None)
        files = []
        for s in getattr(sp, "siblings", None) or []:
            name = getattr(s, "rfilename", None)
            if isinstance(name, str) and name and len(name) < 512:
                files.append((name, int(getattr(s, "size", None) or 0), getattr(s, "lfs", None) is not None))
        sub = getattr(sp, "subdomain", None) or ""
        return {"id": spec, "sha": str(sp.sha or ""), "files": files, "card": card,
                "subdomain": sub if re.match(r"^[a-z0-9-]{1,80}$", sub) else "",
                "stage": str(getattr(rt, "stage", "") or "") if rt is not None else "",
                "hardware": getattr(rt, "hardware", None) if rt is not None else None,
                "requested_hardware": getattr(rt, "requested_hardware", None) if rt is not None else None,
                "sdk": getattr(sp, "sdk", None), "tags": list(getattr(sp, "tags", None) or []),
                "updated": sp.last_modified.isoformat() if getattr(sp, "last_modified", None) else None}

    info = catalog._cached(("space-files-info", spec), INFO_TTL, fetch)
    if not re.match(r"^[0-9a-f]{40}$", info["sha"]):
        raise LookupError("the Hub didn't say which revision this Space is at")
    return info


def junk(path: str) -> bool:
    return bool(JUNK.search(path))


def _withheld_by_name(path: str) -> bool:
    """Listed but never served: reference solutions, and task data (it holds the answers the grader checks)."""
    if ANSWER_DIR.search(path):
        return True
    if DATA.search(path) and (TASK_DATA_DIR.search(path) or catalog.ANSWER_FILE.search(path)):
        return True
    return not CODE.search(path) and not MANIFESTS.search(path) and bool(catalog.ANSWER_FILE.search(path))


def listing(info: dict[str, Any]) -> tuple[list[dict[str, Any]], int, bool]:
    """The files worth reading: junk dropped, binaries marked, answer files marked withheld."""
    out, dropped = [], 0
    for path, size, lfs in sorted(info["files"]):
        if junk(path) or ".." in path.split("/") or path.startswith("/") or "\\" in path:
            dropped += 1
            continue
        row: dict[str, Any] = {"path": path, "size": size}
        if lfs or BINARY.search(path):
            row["binary"] = True
        if _withheld_by_name(path):
            row["withheld"] = True
        if path == "uv.lock" or path.endswith("/uv.lock"):
            row["collapsed"] = True
        out.append(row)
    return out[:MAX_LISTED], dropped, len(out) > MAX_LISTED


def _download(spec: str, sha: str, path: str, limit: int = MAX_TEXT + 1) -> bytes:
    """The first `limit` bytes of one file of a public Space at a revision: anonymous, and cached on disk by revision
    (a file at a commit never changes, so it is fetched once)."""
    import huggingface_hub

    p = huggingface_hub.hf_hub_download(spec, path, repo_type="space", revision=sha, token=False,
                                        cache_dir=str(config.CACHE_DIR / "space-files"))
    with open(p, "rb") as f:
        return f.read(limit)


def _decode(raw: bytes) -> str | None:
    """Text, or None for a binary (a NUL byte, or bytes that aren't UTF-8 before the cut)."""
    if b"\x00" in raw[:8192]:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as e:
        if e.start >= len(raw) - 4:   # cut mid-character at the size cap
            return raw[:e.start].decode("utf-8", "replace")
        return None


def _fetch_text(spec: str, sha: str, path: str, size: int) -> str | None:
    if size > MAX_TEXT:
        return None
    try:
        return _decode(_download(spec, sha, path)[:MAX_TEXT])
    except Exception:  # noqa: BLE001 - one unreadable file is a gap in what's declared, not a failed page
        return None


# ── what the files declare ───────────────────────────────────────────────────
class _Budget:
    def __init__(self, nodes: int = 6000):
        self.left = nodes


def plain(v: Any, depth: int = 0, budget: _Budget | None = None, drop_answers: bool = True, gone: list | None = None) -> Any:
    """A YAML/TOML value as bounded JSON: depth, sizes and the number of nodes visited are capped (YAML anchors
    can make a small file a huge graph), and answer-like keys are left out."""
    budget = budget or _Budget()
    budget.left -= 1
    if budget.left < 0 or depth > 7:
        return None
    if v is None or isinstance(v, (bool, int)):
        return v
    if isinstance(v, float):
        return v if math.isfinite(v) else str(v)
    if isinstance(v, str):
        return v[:2000]
    if isinstance(v, dict):
        out = {}
        for k, x in list(v.items())[:120]:
            key = str(k)[:120]
            if drop_answers and ANSWER.search(key):
                if gone is not None:
                    gone.append(key)
                continue
            out[key] = plain(x, depth + 1, budget, drop_answers, gone)
        return out
    if isinstance(v, (list, tuple, set)):
        return [plain(x, depth + 1, budget, drop_answers, gone) for x in list(v)[:200]]
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return str(v)[:200]


def _str(v: Any, n: int = 400) -> str | None:
    if v is None or isinstance(v, (dict, list)):
        return None
    s = str(v).strip()
    return s[:n] if s else None


def _num(v: Any) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None


MANIFEST_KEYS = ("spec_version", "name", "version", "type", "runtime", "app", "port", "description", "author", "license",
                 "display_name")


def _validation(v: Any) -> dict[str, Any] | None:
    """OpenEnv's `validation:` block (openenv.validation.manifest), read as data: reward contract, judge pin,
    resources, network, capabilities, type tags."""
    if not isinstance(v, dict):
        return None
    out: dict[str, Any] = {}
    rw = v.get("reward")
    if isinstance(rw, dict):
        rng = rw.get("range")
        out["reward"] = {"range": [_num(x) for x in rng[:2]] if isinstance(rng, (list, tuple)) and len(rng) >= 2 else None,
                         **{k: _num(rw.get(k)) for k in ("oracle_tolerance", "floor_margin", "variance_tolerance") if _num(rw.get(k)) is not None}}
    j = v.get("judge")
    if isinstance(j, dict):
        out["judge"] = {"model": _str(j.get("model"), 200), "version": _str(j.get("version"), 100),
                        "params": plain(j.get("params"), drop_answers=False) if isinstance(j.get("params"), dict) else None}
    r = v.get("resources")
    if isinstance(r, dict):
        out["resources"] = {k: (_num(r.get(k)) if k != "gpu_types" else [str(x)[:60] for x in (r.get(k) or [])[:8]] if isinstance(r.get(k), list) else None)
                            for k in ("cpu", "memory_mb", "disk_mb", "episode_timeout_s", "gpus", "gpu_types") if r.get(k) is not None}
    n = v.get("network")
    if isinstance(n, dict):
        out["network"] = {"mode": _str(n.get("mode"), 40) or "public",
                          "allowed_hosts": [str(x)[:120] for x in (n.get("allowed_hosts") or [])[:30]] if isinstance(n.get("allowed_hosts"), list) else []}
    c = v.get("capabilities")
    if isinstance(c, dict):
        ver = c.get("verifier") if isinstance(c.get("verifier"), dict) else {}
        orc = c.get("oracle") if isinstance(c.get("oracle"), dict) else None
        counts = c.get("declared_task_count") if isinstance(c.get("declared_task_count"), dict) else {}
        out["capabilities"] = {
            "verifier": {"kind": _str(ver.get("kind"), 40), "entry": _str(ver.get("entry"), 200)} if ver else None,
            "oracle": {"form": _str(orc.get("form"), 40), "location": _str(orc.get("location"), 200)} if orc else None,
            **{k: bool(c.get(k)) for k in ("set_state", "llm_judged", "rubric_tree", "task_api") if k in c},
            "declared_tools": [str(x)[:120] for x in (c.get("declared_tools") or [])[:100]] if isinstance(c.get("declared_tools"), list) else [],
            "declared_task_count": {str(k)[:80]: int(x) for k, x in list(counts.items())[:40] if isinstance(x, int) and not isinstance(x, bool)},
        }
    t = v.get("types")
    if isinstance(t, dict) and isinstance(t.get("tags"), list):
        out["types"] = [str(x)[:60] for x in t["tags"][:20]]
    extra = sorted(str(k) for k in v if k not in ("reward", "judge", "resources", "network", "capabilities", "types"))
    if extra:
        out["other"] = extra[:20]
    return out


TASK_FIELDS = ("id", "name", "title", "difficulty", "level", "description", "max_steps", "split", "grader", "score_range")


def _tasks(v: Any) -> list[dict[str, Any]] | None:
    """A free-form `tasks:` key (hackathon manifests list them): each as {id, name, difficulty, description, …}."""
    items: list[tuple[Any, Any]] = []
    if isinstance(v, list):
        items = [(None, x) for x in v[:500]]
    elif isinstance(v, dict):
        items = list(v.items())[:500]
    else:
        return None
    out = []
    for key, x in items:
        if isinstance(x, (str, int, float)) and not isinstance(x, bool):
            out.append({"id": str(x)[:120]} if key is None else {"id": str(key)[:120], "description": str(x)[:600]})
        elif isinstance(x, dict):
            row: dict[str, Any] = {"id": str(key)[:120]} if key is not None else {}
            for k in TASK_FIELDS:
                if k in x and not ANSWER.search(k):
                    val = x[k]
                    row[k] = (_str(val, 600) if not isinstance(val, (list, dict)) else plain(val, drop_answers=True))
            if "level" in row and "difficulty" not in row:
                row["difficulty"] = row.pop("level")
            if row:
                out.append(row)
    return out


def parse_manifest(text: str, path: str) -> dict[str, Any]:
    """openenv.yaml: its canonical keys, its `validation:` block, and the free-form keys authors add (tasks, rubric,
    reward, environment, tools; the legacy `action:`/`observation:` class names)."""
    import yaml

    out: dict[str, Any] = {"path": path}
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError as e:
        mark = getattr(e, "problem_mark", None)
        out["error"] = f"not valid YAML{f' (line {mark.line + 1})' if mark is not None else ''}"
        return out
    except Exception:  # noqa: BLE001 - recursion, memory: a broken manifest is said so, never a failed page
        out["error"] = "couldn't be read"
        return out
    if not isinstance(doc, dict):
        out["error"] = "not a YAML mapping"
        return out
    keys = {k: plain(doc[k]) if isinstance(doc[k], (int, float, bool)) else _str(doc[k], 1200) for k in MANIFEST_KEYS if k in doc}
    out["keys"] = {k: v for k, v in keys.items() if v is not None}
    tags = doc.get("tags")
    if isinstance(tags, list):
        out["tags"] = [str(t)[:60] for t in tags[:20]]
    if "validation" in doc:
        out["validation"] = _validation(doc.get("validation"))
    tasks = _tasks(doc.get("tasks"))
    if tasks is not None:
        out["tasks"] = tasks
    tools = doc.get("tools")
    if isinstance(tools, list):
        out["tools"] = [({"name": str(t.get("name"))[:120], "description": _str(t.get("description"), 300)} if isinstance(t, dict) and t.get("name")
                         else {"name": str(t)[:120]}) for t in tools[:100] if isinstance(t, (str, dict))]
    gone: list[str] = []
    for k in ("rubric", "rubrics", "reward", "rewards", "reward_range", "grader", "scoring", "environment", "action_space",
              "observation_space", "endpoints"):
        if k in doc and doc[k] is not None:
            out.setdefault("blocks", {})[k] = plain(doc[k], gone=gone)
    for k in ("action", "observation"):   # the legacy manifest named the classes; some name fields instead
        if isinstance(doc.get(k), str):
            out.setdefault("legacy", {})[k] = doc[k][:120]
        elif isinstance(doc.get(k), dict):
            out.setdefault("blocks", {})[k] = plain(doc[k], gone=gone)
    if gone:
        out["withheld"] = sorted(set(gone))
    known = set(MANIFEST_KEYS) | {"tags", "validation", "tasks", "tools", "action", "observation"} | set(out.get("blocks", {}))
    other = [str(k) for k in doc if k not in known]
    if other:
        out["other"] = other[:30]
    return out


# ── Python, read and never run ───────────────────────────────────────────────
ROLES = {"Action": "action", "Observation": "observation", "State": "state", "CallToolAction": "action",
         "CallToolObservation": "observation", "ListToolsAction": "action"}
IMAGE = re.compile(r"((^|_)(image|img|frame|screenshot|pixels|png|jpe?g|picture|photo|render)s?(_?(b64|data|url|bytes))?$|base64|(^|_)b64$)", re.I)


def _src(node: ast.AST | None, n: int = 160) -> str | None:
    if node is None:
        return None
    try:
        s = ast.unparse(node)
    except Exception:  # noqa: BLE001
        return None
    return s if len(s) <= n else s[: n - 1] + "…"


def _first_doc(node: ast.AST) -> str | None:
    try:
        d = ast.get_docstring(node)  # type: ignore[arg-type]
    except Exception:  # noqa: BLE001
        return None
    return d.strip().split("\n\n")[0].strip()[:500] if d else None


def _parse_py(text: str) -> ast.Module | None:
    try:
        return ast.parse(text)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None


def _field(stmt: ast.AnnAssign) -> dict[str, Any] | None:
    if not isinstance(stmt.target, ast.Name):
        return None
    name = stmt.target.id
    ann = _src(stmt.annotation, 200) or ""
    if name.startswith("_") or name == "model_config" or ann.startswith(("ClassVar", "typing.ClassVar")):
        return None
    row: dict[str, Any] = {"name": name, "type": ann, "required": stmt.value is None}
    v = stmt.value
    if isinstance(v, ast.Call) and (_src(v.func) or "").split(".")[-1] == "Field":
        kw = {k.arg: k.value for k in v.keywords if k.arg}
        first = v.args[0] if v.args else kw.get("default")
        if isinstance(first, ast.Constant) and first.value is Ellipsis:
            row["required"] = True
        elif first is not None:
            row["default"] = _src(first, 80)
        elif "default_factory" in kw:
            row["default"] = f"{_src(kw['default_factory'], 60)}()"
        else:
            row["required"] = True
        desc = kw.get("description")
        if isinstance(desc, ast.Constant) and isinstance(desc.value, str):
            row["description"] = desc.value[:400]
        limits = [f"{k} {_src(kw[k], 40)}" for k in ("ge", "gt", "le", "lt", "min_length", "max_length", "pattern") if k in kw]
        if limits:
            row["limits"] = limits[:4]
    elif v is not None:
        row["default"] = _src(v, 80)
    if IMAGE.search(name) or re.search(r"\b(Image|bytes)\b", ann):
        row["image"] = True
    return row


def parse_models(text: str, path: str) -> dict[str, Any]:
    """models.py's classes: which are the Action, Observation and State (by their bases, followed within the file),
    and each one's fields with type, default and description. Parsed, never imported."""
    tree = _parse_py(text)
    if tree is None:
        return {"path": path, "error": "couldn't be parsed as Python"}
    defs = [n for n in tree.body if isinstance(n, ast.ClassDef)][:80]
    bases = {c.name: [(_src(b, 80) or "").split(".")[-1].split("[")[0] for b in c.bases] for c in defs}

    def role(name: str, seen: frozenset = frozenset()) -> str | None:
        if name in ROLES:
            return ROLES[name]
        if name in seen or name not in bases:
            return None
        for b in bases[name]:
            r = role(b, seen | {name})
            if r:
                return r
        return None

    classes = []
    for c in defs:
        kind = "enum" if any(b.endswith("Enum") for b in bases[c.name]) else None
        r = role(c.name)
        row: dict[str, Any] = {"name": c.name, "bases": bases[c.name][:6], "role": r, "doc": _first_doc(c)}
        if kind == "enum":
            row["kind"] = "enum"
            row["values"] = [_src(s.value, 60) for s in c.body if isinstance(s, ast.Assign) and s.value is not None][:40]
        else:
            row["fields"] = [f for f in (_field(s) for s in c.body if isinstance(s, ast.AnnAssign)) if f][:80]
        classes.append(row)
    return {"path": path, "classes": classes}


def _consts(tree: ast.Module) -> dict[str, ast.AST]:
    """Module-level `NAME = value`, to read `max_concurrent_envs=MAX_CONCURRENT` back to what it is."""
    out = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            out[n.targets[0].id] = n.value
    return out


def _literal_of(node: ast.AST | None, consts: dict[str, ast.AST]) -> Any:
    """A literal, a module constant, or the default in `int(os.getenv("X", "4"))`: its value; else None."""
    for _ in range(3):
        if isinstance(node, ast.Name) and node.id in consts:
            node = consts[node.id]
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, int, float, bool)):
        return node.value
    if isinstance(node, ast.Call) and _src(node.func) in ("int", "float", "str") and node.args:
        inner = node.args[0]
        if isinstance(inner, ast.Call) and (_src(inner.func) or "").endswith(("getenv", "environ.get")) and len(inner.args) >= 2:
            v = _literal_of(inner.args[1], consts)
            try:
                return int(v) if _src(node.func) == "int" else v
            except (TypeError, ValueError):
                return None
        return _literal_of(inner, consts)
    return None


def parse_app(text: str, path: str) -> dict[str, Any]:
    """server/app.py: the `create_app(...)` call (its environment, Action and Observation classes, env_name,
    max_concurrent_envs, a Gradio tab of its own, the mode) and routes it adds itself."""
    tree = _parse_py(text)
    if tree is None:
        return {"path": path, "error": "couldn't be parsed as Python"}
    consts = _consts(tree)
    out: dict[str, Any] = {"path": path}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and (_src(node.func) or "").split(".")[-1] in ("create_app", "create_fastapi_app", "create_web_interface_app"):
            args = [(_src(a, 80) or "") for a in node.args[:3]]
            kw = {k.arg: k.value for k in node.keywords if k.arg}
            out["factory"] = (_src(node.func) or "").split(".")[-1]
            for i, (key, name) in enumerate((("env", "env"), ("action", "action_cls"), ("observation", "observation_cls"))):
                val = args[i] if i < len(args) else _src(kw.get(name), 80)
                if val:
                    out[key] = val.split(".")[-1]
            for key in ("env_name", "max_concurrent_envs", "custom_tab_name", "title_override", "mode"):
                if key in kw:
                    lit = _literal_of(kw[key], consts)
                    out[key] = lit if lit is not None else _src(kw[key], 80)
            if "gradio_builder" in kw:
                out["gradio_builder"] = _src(kw["gradio_builder"], 80)
            if "state_cls" in kw:
                out["state"] = (_src(kw["state_cls"]) or "").split(".")[-1]
            break
    routes = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for d in node.decorator_list:
                if isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute) and d.func.attr in ("get", "post", "put", "delete", "websocket", "api_route") \
                        and d.args and isinstance(d.args[0], ast.Constant) and isinstance(d.args[0].value, str):
                    routes.append(f"{d.func.attr.upper()} {d.args[0].value[:120]}")
    if routes:
        out["routes"] = routes[:30]
    mounts = [_src(n.args[0], 60) for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
              and n.func.attr == "mount" and n.args]
    if mounts:
        out["mounts"] = [m for m in mounts if m][:10]
    out["mcp"] = out.get("action") in ("CallToolAction", "ListToolsAction") or None
    return out


LLM_MODULES = {"openai", "anthropic", "litellm", "groq", "mistralai", "cohere", "together", "google.generativeai", "google.genai", "vertexai"}
REWARD_FN = re.compile(r"(reward|score|grade|grader|judge|verify|verifier|penalt|bonus|rubric)", re.I)


def parse_environment(text: str, path: str) -> dict[str, Any]:
    """server/*_environment.py: the environment class, the MCP tools it registers (`@mcp.tool`), its reward and
    grading functions, and whether an LLM judges (a client library imported, OpenEnv's LLMJudge) or a Rubric scores."""
    tree = _parse_py(text)
    if tree is None:
        return {"path": path, "error": "couldn't be parsed as Python"}
    imports: set[str] = set()
    names: set[str] = set()
    tools, fns, envs, local = [], [], [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)
            names |= {a.name for a in node.names}
            if (node.level or REWARD_FN.search(node.module or "")) and (REWARD_FN.search(node.module or "") or any(REWARD_FN.search(a.name) and a.name[:1].islower() for a in node.names)):
                local.append([node.level or 0, node.module or "", [a.name for a in node.names][:6]])
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.ClassDef):
            b = [(_src(x, 80) or "").split(".")[-1].split("[")[0] for x in node.bases]
            if any(x.endswith("Environment") for x in b):
                envs.append({"name": node.name, "bases": b[:4], "doc": _first_doc(node)})
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if any(re.search(r"(^|\.)tool$", (_src(d.func if isinstance(d, ast.Call) else d) or "")) for d in node.decorator_list):
                params = [a.arg for a in node.args.args if a.arg not in ("self", "cls", "ctx")][:12]
                tools.append({"name": node.name, "params": params, "description": (_first_doc(node) or "").split("\n")[0][:300] or None})
            elif REWARD_FN.search(node.name) and not node.name.startswith("__"):
                fns.append(node.name)
    llm = sorted({m for m in imports if m in LLM_MODULES or m.split(".")[0] in LLM_MODULES})
    judge = bool(llm) or bool({"LLMJudge", "InferenceClient", "chat_completion"} & names) or ("completions" in names and "create" in names)
    rubric = any("rubric" in m.lower() for m in imports) or bool({"Rubric", "RubricDict", "WeightedSum"} & names)
    out: dict[str, Any] = {"path": path, "classes": envs[:6], "tools": tools[:60], "functions": sorted(set(fns))[:24], "reward_imports": local[:8],
                           "llm_judge": judge, "llm_modules": llm[:6], "rubric": rubric,
                           "mcp": any(e["bases"] and any(x.startswith("MCP") for x in e["bases"]) for e in envs) or bool(tools)}
    return out


# ── Dockerfile, pyproject, discovery.json, uv.lock ───────────────────────────
def parse_dockerfile(text: str, path: str) -> dict[str, Any]:
    """The image's base, whether OpenEnv's web interface is on (ENABLE_WEB_INTERFACE), the port, the command."""
    from .dockerfile import _lines, _subst

    out: dict[str, Any] = {"path": path, "from": []}
    env: dict[str, str] = {}
    args: dict[str, str] = {}
    for line in _lines(text)[:400]:
        word, _, rest = line.partition(" ")
        word = word.upper()
        rest = rest.strip()
        if word == "FROM":
            parts = [p for p in rest.split() if not p.startswith("--")]
            if parts:
                out["from"].append(_subst(parts[0], args)[:160])
        if word == "ARG":
            k, _, val = rest.partition("=")
            if k.strip() and val:
                args.setdefault(k.strip(), val.strip().strip("'\""))
        elif word in ("ENV", "ARG"):
            try:
                parts = shlex.split(rest)
            except ValueError:
                parts = rest.split()
            if parts and "=" not in parts[0] and word == "ENV":
                env[parts[0]] = " ".join(parts[1:])
            else:
                for p in parts:
                    k, _, val = p.partition("=")
                    if word == "ENV" or k not in env:
                        env[k] = val
        elif word == "EXPOSE":
            m = re.match(r"(\d{2,5})", rest)
            if m:
                out["expose"] = int(m.group(1))
        elif word in ("CMD", "ENTRYPOINT"):
            out["cmd"] = rest[:400]
    web = env.get("ENABLE_WEB_INTERFACE")
    if web is not None:
        out["web_interface"] = web.strip().strip("'\"").lower() in ("true", "1", "yes")
    m = re.search(r"--port[ =\"',]+(\d{2,5})", out.get("cmd", ""))
    if m:
        out["port"] = int(m.group(1))
    if env.get("OPENENV_MODE"):
        out["mode"] = env["OPENENV_MODE"].strip("'\"")[:20]
    out["from"] = list(dict.fromkeys(out["from"]))[:6]
    return out


OPENENV_REQ = re.compile(r"^\s*(openenv-core|openenv_core|openenv)(?![\w.-])\s*(\[[^\]]*\])?\s*(.*)$", re.I)


def parse_pyproject(text: str, path: str) -> dict[str, Any]:
    """The package, its version, Python, and the openenv-core it requires (a version range or a git ref)."""
    try:
        doc = tomllib.loads(text)
    except (tomllib.TOMLDecodeError, ValueError, RecursionError):
        return {"path": path, "error": "not valid TOML"}
    proj = doc.get("project") if isinstance(doc.get("project"), dict) else {}
    deps = [d for d in proj.get("dependencies") or [] if isinstance(d, str)]
    out: dict[str, Any] = {"path": path, "package": _str(proj.get("name"), 120), "version": _str(proj.get("version"), 40),
                           "requires_python": _str(proj.get("requires-python"), 40)}
    for d in deps:
        m = OPENENV_REQ.match(d)
        if not m:
            continue
        rest = (m.group(3) or "").strip()
        git = re.search(r"git\+[^@#\s]+@v?([\w.\-]+)", rest)
        ver = re.search(r"(==|>=|~=|<=|>|<|!=)\s*v?(\d+(?:\.\d+)*\w*)", rest)
        out["openenv"] = d.strip()[:200]
        out["openenv_package"] = m.group(1).lower().replace("_", "-")
        out["openenv_version"] = git.group(1) if git else (ver.group(2) if ver.group(1) == "==" else ver.group(1) + ver.group(2)) if ver else None
        break
    scripts = proj.get("scripts") if isinstance(proj.get("scripts"), dict) else {}
    if isinstance(scripts.get("server"), str):
        out["server_script"] = scripts["server"][:160]
    out["dependencies"] = len(deps)
    return out


def parse_discovery(text: str, path: str) -> dict[str, Any]:
    """discovery.json: the environment card's declarations (description, license, tags, capabilities, queries)."""
    try:
        doc = json.loads(text)
    except ValueError:
        return {"path": path, "error": "not valid JSON"}
    if not isinstance(doc, dict):
        return {"path": path, "error": "not a JSON object"}
    out: dict[str, Any] = {"path": path}
    for k in ("description", "license", "license_url", "display_name", "displayName"):
        if _str(doc.get(k)):
            out["display_name" if k == "displayName" else k] = _str(doc.get(k), 1200)
    for k in ("tags", "capabilities", "representative_queries"):
        if isinstance(doc.get(k), list):
            out[k] = [str(x)[:200] for x in doc[k][:20] if isinstance(x, (str, int, float))]
    return out


def collapse_lock(text: str) -> str:
    """uv.lock, collapsed to what it pins: one `name version` line per package."""
    try:
        doc = tomllib.loads(text)
    except (tomllib.TOMLDecodeError, ValueError, RecursionError):
        return text
    pkgs = [p for p in doc.get("package") or [] if isinstance(p, dict) and p.get("name")]
    lines = [f"# uv.lock, collapsed: {len(pkgs)} locked package{'s' if len(pkgs) != 1 else ''} (name, version)", ""]
    width = max((len(str(p["name"])) for p in pkgs), default=0)
    lines += [f"{str(p['name']):<{width}}  {p.get('version') or (p.get('source') or {}).get('git', '') or ''}" for p in pkgs]
    return "\n".join(lines) + "\n"


# ── which files are the environment's ────────────────────────────────────────
def _dockerfile_copies(text: str | None) -> set[str]:
    """Folders a Dockerfile COPYs or ADDs from its build context."""
    if not text:
        return set()
    from .dockerfile import _lines

    out = set()
    for line in _lines(text)[:400]:
        word, _, rest = line.partition(" ")
        if word.upper() not in ("COPY", "ADD"):
            continue
        parts = [p for p in rest.split() if not p.startswith("--")]
        for src in parts[:-1]:
            out.add(src.strip("'\"").removeprefix("./").rstrip("/"))
    return out


def manifest_path(paths: list[str], dockerfile: str | None, spec: str) -> str | None:
    """The environment's openenv.yaml: the shallowest one; among several as shallow, the folder the Dockerfile copies,
    then the one named like the Space, then the first by name."""
    cands = [p for p in paths if (p == "openenv.yaml" or p.endswith("/openenv.yaml")) and "/templates/" not in f"/{p}" and not junk(p)]
    if not cands:
        return None
    depth = min(p.count("/") for p in cands)
    top = sorted(p for p in cands if p.count("/") == depth)
    if len(top) == 1:
        return top[0]
    copied = _dockerfile_copies(dockerfile)
    for p in top:
        if posixpath.dirname(p) in copied:
            return p
    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", s.lower())

    for p in top:
        if norm(posixpath.dirname(p)) == norm(spec.split("/")[-1]):
            return p
    return top[0]


def _first(paths: set[str], *cands: str) -> str | None:
    return next((c for c in cands if c in paths), None)


def key_files(paths: list[str], root: str, manifest: str | None) -> dict[str, str]:
    """Where the environment's own files are: under the manifest's folder first, then the repository's root."""
    have = set(paths)
    pre = f"{root}/" if root else ""
    out: dict[str, str | None] = {"manifest": manifest}
    out["models"] = _first(have, f"{pre}models.py", "models.py") or next(
        (p for p in sorted(paths, key=lambda p: (p.count("/"), p)) if p.endswith("/models.py") and p.startswith(pre) and "test" not in p), None)
    out["app"] = _first(have, f"{pre}server/app.py", f"{pre}app.py", "server/app.py") or next(
        (p for p in sorted(paths, key=lambda p: (p.count("/"), p)) if p.endswith("server/app.py")), None)
    envs = sorted((p for p in paths if p.startswith(f"{pre}server/") and re.search(r"(^|/|_)environment\.py$", p) and p.count("/") == pre.count("/") + 1),
                  key=lambda p: (len(p), p))
    out["environment"] = envs[0] if envs else next((p for p in sorted(paths, key=lambda p: (p.count("/"), p)) if p.endswith("_environment.py")), None)
    out["client"] = _first(have, f"{pre}client.py", "client.py")
    out["pyproject"] = _first(have, f"{pre}pyproject.toml", "pyproject.toml")
    out["dockerfile"] = _first(have, "Dockerfile", f"{pre}server/Dockerfile", f"{pre}Dockerfile", "server/Dockerfile")
    out["discovery"] = _first(have, f"{pre}discovery.json", "discovery.json")
    out["inference"] = _first(have, f"{pre}inference.py", "inference.py")
    out["readme"] = _first(have, "README.md", f"{pre}README.md")
    out["lock"] = _first(have, f"{pre}uv.lock", "uv.lock")
    return {k: v for k, v in out.items() if v}


def reward_files(env_path: str, imports: list, paths: set[str], root: str) -> list[str]:
    """The repository files an environment imports its reward from (`from .scoring import compute_reward`,
    `from ..rewards.reward_function import …`, `from my_env.grader import …`), found in the listing."""
    here = posixpath.dirname(env_path)
    out: list[str] = []
    for level, module, _names in imports:
        rel = module.replace(".", "/")
        bases = []
        if level:
            base = here
            for _ in range(level - 1):
                base = posixpath.dirname(base)
            bases.append(base)
        else:
            bases += [root, "", posixpath.dirname(root)]
            _, _, rest = rel.partition("/")
            if rest:   # `my_env.grader`: the package is the environment's own folder
                bases += [root]
                rel_alt = rest
            else:
                rel_alt = None
        for base in bases:
            for r in filter(None, [rel, None if level else rel_alt]):
                for cand in (f"{base}/{r}.py" if base else f"{r}.py", f"{base}/{r}/__init__.py" if base else f"{r}/__init__.py"):
                    cand = posixpath.normpath(cand)
                    if cand in paths and cand != env_path and cand not in out:
                        out.append(cand)
    return out[:6]


def _env_root(paths: list[str], manifest: str | None) -> str:
    if manifest:
        return posixpath.dirname(manifest)
    cands = [posixpath.dirname(posixpath.dirname(p)) for p in paths if p.endswith("server/app.py")] + \
            [posixpath.dirname(p) for p in paths if p.endswith("models.py") and "test" not in p]
    return min(cands, key=lambda d: (d.count("/") if d else -1, d)) if cands else ""


PARSERS = {"manifest": parse_manifest, "models": parse_models, "app": parse_app, "environment": parse_environment,
           "dockerfile": parse_dockerfile, "pyproject": parse_pyproject, "discovery": parse_discovery}


def declared(spec: str) -> dict[str, Any]:
    """The curated tree and what the environment's files declare, at the Space's current revision. Never wakes it."""
    info = _info(spec)
    sha = info["sha"]

    def build():
        files, dropped, truncated = listing(info)
        sizes = {f["path"]: f["size"] for f in files if not f.get("binary")}
        paths = [f["path"] for f in files]
        docker = _fetch_text(spec, sha, "Dockerfile", sizes["Dockerfile"]) if "Dockerfile" in sizes else None
        manifest = manifest_path(paths, docker, spec)
        root = _env_root(paths, manifest)
        keys = key_files(paths, root, manifest)
        want = {role: p for role, p in keys.items() if role in PARSERS and p in sizes}
        texts: dict[str, str | None] = {}
        with ThreadPoolExecutor(6) as pool:
            for role, text in zip(want, pool.map(lambda p: docker if p == "Dockerfile" and docker is not None else _fetch_text(spec, sha, p, sizes[p]), want.values())):
                texts[role] = text
        out: dict[str, Any] = {}
        for role, p in want.items():
            if texts.get(role) is None:
                out[role] = {"path": p, "error": "too large to read here" if sizes[p] > MAX_TEXT else "couldn't be read"}
                continue
            try:
                out[role] = PARSERS[role](texts[role], p)
            except Exception:  # noqa: BLE001 - whatever a file holds, the page still opens
                out[role] = {"path": p, "error": "couldn't be read"}
        env = out.get("environment")
        if isinstance(env, dict) and env.get("reward_imports"):
            env["reward_files"] = reward_files(env["path"], env.pop("reward_imports"), set(paths), root)
            more = [p for p in env["reward_files"] if p in sizes][:2]
            for p, text in zip(more, ThreadPoolExecutor(2).map(lambda p: _fetch_text(spec, sha, p, sizes[p]), more)):
                got = parse_environment(text, p) if text else None
                if got and not got.get("error"):
                    env["llm_judge"] = env["llm_judge"] or got["llm_judge"]
                    env["rubric"] = env["rubric"] or got["rubric"]
                    env["llm_modules"] = sorted(set(env["llm_modules"]) | set(got["llm_modules"]))[:6]
                    env["functions"] = sorted(set(env["functions"]) | {f"{posixpath.basename(p)[:-3]}.{f}" for f in got["functions"][:12]})[:30]
        elif isinstance(env, dict):
            env.pop("reward_imports", None)
        card = info["card"]
        out["card"] = {k: (card.get(k) if isinstance(card.get(k), (int, float)) else _str(card.get(k), 400))
                       for k in ("title", "short_description", "base_path", "app_port", "license", "sdk") if card.get(k) is not None}
        return {"id": spec, "sha": sha, "short": sha[:7], "subdomain": info["subdomain"], "root": root, "keys": keys,
                "files": files, "total": len(files), "dropped": dropped, "truncated": truncated,
                "hardware": info["hardware"] or info["requested_hardware"], "stage": info["stage"], "updated": info["updated"],
                "declared": out, "read_at": time.time(),
                "partial": any(v.get("error") == "couldn't be read" for v in out.values() if isinstance(v, dict))}

    key = ("space-files", spec, sha)
    res = catalog._cached(key, PARSE_TTL, build)
    if res.get("partial") and time.time() - res["read_at"] > 60:   # a file the Hub didn't give us: ask again
        catalog._memo.pop(key, None)
        res = catalog._cached(key, PARSE_TTL, build)
    return res


def file(spec: str, f: str) -> dict[str, Any]:
    """One file of the curated listing at the listed revision: text (the first MAX_TEXT bytes of a longer one),
    uv.lock collapsed to its pins, withheld files and binaries refused."""
    f = (f or "").strip()
    if not f or len(f) > 512 or f.startswith("/") or "\\" in f or "\x00" in f or ".." in f.split("/") or posixpath.normpath(f) != f:
        raise ValueError("not a file of this Space")
    info = _info(spec)
    files, _, _ = listing(info)
    row = next((x for x in files if x["path"] == f), None)
    if row is None:
        raise LookupError("not a file of this Space")
    out: dict[str, Any] = {"path": f, "size": row["size"], "sha": info["sha"]}
    if row.get("withheld"):
        return {**out, "withheld": True}
    if row.get("binary"):
        raise BinaryFile("a binary file: open it on the Hub")
    if row["size"] > MAX_FETCH:
        raise TooLarge(f"too large to show here ({row['size'] / 1048576:.1f} MB): open it on the Hub")

    raw = _download(info["id"], info["sha"], f, MAX_FETCH + 1)
    text = _decode(raw if row.get("collapsed") else raw[:MAX_TEXT])
    if text is None:
        raise BinaryFile("a binary file: open it on the Hub")
    if row.get("collapsed"):
        return {**out, "text": collapse_lock(text), "collapsed": True}
    if not MANIFESTS.search(f) and catalog._answer_data(f, text):
        return {**out, "withheld": True}
    return {**out, "text": text, "truncated": len(raw) > MAX_TEXT}


class BinaryFile(Exception):
    pass


class TooLarge(Exception):
    pass


# ── routes ───────────────────────────────────────────────────────────────────
def _errors(fn, *args):
    from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

    try:
        return fn(*args)
    except (RepositoryNotFoundError, PermissionError):
        raise HTTPException(404, "No such Space, or it is private.")
    except ValueError as e:
        raise HTTPException(400, str(e))
    except LookupError as e:
        raise HTTPException(404, str(e))
    except BinaryFile as e:
        raise HTTPException(415, str(e))
    except TooLarge as e:
        raise HTTPException(413, str(e))
    except HfHubHTTPError as e:
        code = getattr(getattr(e, "response", None), "status_code", 0)
        if code in (401, 403, 404):
            raise HTTPException(404, "No such Space, or it is private.")
        raise HTTPException(502, f"could not reach the Hub (HTTP {code})")
    except (OSError, httpx.HTTPError) as e:
        raise HTTPException(502, f"could not reach the Hub: {type(e).__name__}")


@router.get("/api/spaces/{org}/{name}/files")
def space_files(org: str, name: str):
    """The Space's files (vendored OpenEnv, build output and caches left out) and what they declare: openenv.yaml,
    models.py's classes, server/app.py, the environment's reward code, the Dockerfile and pyproject. Never wakes it."""
    return _errors(declared, f"{org}/{name}")


@router.get("/api/spaces/{org}/{name}/file")
def space_file(org: str, name: str, f: str = ""):
    """One text file of the Space at the listed revision (only files in the listing; no binaries)."""
    return _errors(file, f"{org}/{name}", f)
