#!/usr/bin/env python3
"""Local explorer for RetroEnv releases: the environment, every task, its known routes, eval runs, and live play.

    uv run python explorer/server.py                                   # http://127.0.0.1:8050
    uv run python explorer/server.py --release tests/fixtures/mini-release

It reads release directories in place (every ``data/release/*`` with ``tasks-private/``,
or the ones given with ``--release``) and eval runs under ``runs/`` (``--runs``), live
while they are still being written. Known routes of held-out tasks are sent only
when the page asks with ``reveal=1``, so the default view shows what a model sees.
Live play loads the release's reaction library on first use. It binds to localhost.
"""

from __future__ import annotations

import argparse
import json
import re
import socket
import sys
import threading
import time
import uuid
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from retroenv.benchmark import load_benchmark
from retroenv.chemistry import canonicalize_smiles
from retroenv.classes import CONSTRAINABLE_CLASSES
from retroenv.disconnections import describe, reaction_phrase, step_family, strategic_disconnections
from retroenv.environment import RetroRouteSession, task_prompt
from retroenv.models import RetroTask
from retroenv.store import load_stock
from retroenv.tools import ASSIST_TOOLS, TOOLS, openai_tools
from retroenv.verifier import WEIGHTS, known_routes_submission

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / "static"
SPLITS = ("train", "dev", "test_id", "test_hard")
HELD_OUT = ("dev", "test_id", "test_hard")
# A run without a final summary counts as live while its files changed this recently.
LIVE_WINDOW_S = 20 * 60


def size_bin(heavy: int) -> str:
    return "<=20" if heavy <= 20 else "21-30" if heavy <= 30 else "31-40" if heavy <= 40 else "41+"


def first_class(task: dict[str, Any]) -> str:
    for route in task["reference_routes"]:
        for step in route["steps"]:
            if step["product"] == task["target_smiles"]:
                return step.get("reaction_class") or "other"
    return "other"


def constraint_text(task: dict[str, Any]) -> str:
    c = task["constraints"]
    if c["forbidden_classes"]:
        return "no " + ", ".join(c["forbidden_classes"])
    if c["excluded_stock"]:
        return f"{len(c['excluded_stock'])} building block unavailable"
    if task["variant"] == "max_depth":
        return f"depth at most {task['max_depth']}"
    if task["min_routes"] > 1:
        return f"{task['min_routes']} distinct routes"
    return ""


class Release:
    def __init__(self, root: Path):
        self.name = root.name
        self.root = root
        self.manifest = json.loads((root / "manifest.json").read_text()) if (root / "manifest.json").exists() else {}
        self.tasks: dict[str, dict[str, Any]] = {}
        for split in SPLITS:
            path = root / "tasks-private" / f"{split}.jsonl"
            if path.exists():
                for line in path.open(encoding="utf-8"):
                    task = json.loads(line)
                    self.tasks[task["task_id"]] = task
        self.rows = sorted(
            (self._row(t) for t in self.tasks.values()), key=lambda r: (SPLITS.index(r["split"]), r["id"])
        )
        self._stock: frozenset[str] | None = None
        self._stock_lock = threading.Lock()

    @property
    def stock(self) -> frozenset[str]:
        # Canonicalizing a full stock takes about a minute; concurrent requests must wait for one load.
        with self._stock_lock:
            if self._stock is None:
                stock_id = next(iter(self.tasks.values()))["stock_id"]
                self._stock = load_stock(self.root / "stocks" / f"{stock_id}.smi")
        return self._stock

    @staticmethod
    def _row(task: dict[str, Any]) -> dict[str, Any]:
        d = task.get("difficulty", {})
        heavy = d.get("heavy_atoms", 0)
        return {
            "id": task["task_id"],
            "parent": task["parent_id"],
            "split": task["split"],
            "variant": task["variant"],
            "smiles": task["target_smiles"],
            "min_depth": d.get("constrained_min_depth") or d.get("min_depth"),
            "max_depth": task["max_depth"],
            "routes": task["min_routes"],
            "heavy": heavy,
            "size": size_bin(heavy),
            "family": first_class(task),
            "tier": d.get("tier"),
            "nn": d.get("nn_similarity"),
            "stereo": d.get("stereocentres", 0) > 0,
            "convergent": bool(d.get("convergent")),
            "constraint": constraint_text(task),
        }


def discover(paths: list[Path]) -> dict[str, Release]:
    roots = paths or sorted(p.parent for p in (ROOT / "data" / "release").glob("*/tasks-private"))
    found = {}
    for root in roots:
        print(f"loading {root} ...", file=sys.stderr)
        found[root.name] = Release(root)
    return found


app = FastAPI(title="RetroEnv explorer")
STATE: dict[str, Any] = {}


def release(name: str) -> Release:
    try:
        return STATE["releases"][name]
    except KeyError:
        raise HTTPException(404, f"unknown release {name!r}") from None


@app.get("/api/meta")
def meta() -> dict[str, Any]:
    return {
        "benchmarks": [
            {
                "name": n,
                "splits": dict(Counter(r["split"] for r in b.rows)),
                "manifest": {k: b.manifest.get(k) for k in ("source", "stock", "library")},
            }
            for n, b in STATE["releases"].items()
        ],
        "tools": [
            {
                "name": t["function"]["name"],
                "description": t["function"]["description"],
                "assist": t["function"]["name"] in ASSIST_TOOLS,
            }
            for t in TOOLS
        ],
        "weights": WEIGHTS,
        "constrainable": list(CONSTRAINABLE_CLASSES),
        "split_use": next(iter(STATE["releases"].values())).manifest.get("design", {}).get("split_use", {}),
    }


@app.get("/api/overview")
def overview(benchmark: str) -> dict[str, Any]:
    b = release(benchmark)
    out: dict[str, Any] = {"splits": {}}
    for split in SPLITS:
        rows = [r for r in b.rows if r["split"] == split]
        if not rows:
            continue
        standard = [r for r in rows if r["variant"] == "standard"]
        out["splits"][split] = {
            "tasks": len(rows),
            "parents": len(standard),
            "variant": dict(Counter(r["variant"] for r in rows)),
            "min_depth": dict(Counter(str(r["min_depth"]) for r in standard)),
            "tier": dict(Counter(str(r["tier"]) for r in standard)),
            "family": dict(Counter(r["family"] for r in standard)),
            "size": dict(Counter(r["size"] for r in standard)),
            "convergent": sum(r["convergent"] for r in standard),
            "stereo": sum(r["stereo"] for r in standard),
        }
    return out


@app.get("/api/tasks")
def tasks(
    benchmark: str,
    split: str = "",
    variant: str = "",
    min_depth: str = "",
    family: str = "",
    tier: str = "",
    size: str = "",
    q: str = "",
    offset: int = 0,
    limit: int = Query(50, le=200),
) -> dict[str, Any]:
    b = release(benchmark)
    needle = q.strip()
    canonical = None
    if needle and not needle.startswith("retro_"):
        try:
            canonical = canonicalize_smiles(needle)
        except Exception:
            canonical = None
    filters = {"split": split, "variant": variant, "min_depth": min_depth, "family": family, "tier": tier, "size": size}
    out = [
        row
        for row in b.rows
        if all(not value or str(row[key]) == value for key, value in filters.items())
        and (not needle or needle in row["id"] or needle in row["smiles"] or row["smiles"] == canonical)
    ]
    return {"total": len(out), "rows": out[offset : offset + limit]}


@app.get("/api/disconnections")
def disconnections(smi: str) -> list[dict[str, Any]]:
    """Rule-based candidate cuts of a molecule, to try in live play."""
    return [
        {
            "reactants": list(d.reactants),
            "bond": d.bond,
            "family": d.family,
            "score": d.score,
            "text": describe(d.family, d.reactants, smi),
        }
        for d in strategic_disconnections(smi)[:8]
    ]


def _route(b: Release, route: dict[str, Any]) -> dict[str, Any]:
    steps = []
    for step in route["steps"]:
        family = step_family(step["reactants"], step["product"])
        steps.append(
            {
                "product": step["product"],
                "reactants": step["reactants"],
                "family": step.get("reaction_class") or family,
                "phrase": reaction_phrase(family, step["reactants"], step["product"]),
                "text": describe(family, step["reactants"], step["product"]),
            }
        )
    molecules = {m for s in route["steps"] for m in [s["product"], *s["reactants"]]}
    return {
        "kind": route["kind"],
        "steps": steps,
        "patents": [s["patent"] for s in route.get("source", [])][:5],
        "in_stock": {m: m in b.stock for m in molecules},
    }


@app.get("/api/task")
def task(benchmark: str, id: str, reveal: int = 0) -> dict[str, Any]:
    b = release(benchmark)
    if id not in b.tasks:
        raise HTTPException(404, f"no task {id}")
    t = b.tasks[id]
    hidden = t["split"] in HELD_OUT and not reveal
    excluded = set(t["constraints"]["excluded_stock"])
    return {
        "row": next(r for r in b.rows if r["id"] == id),
        "task": {k: v for k, v in t.items() if k not in {"reference_routes", "difficulty"}},
        "prompt": task_prompt(RetroTask.from_dict(t)),
        "difficulty": t.get("difficulty", {}),
        "hidden": hidden,
        "target_in_stock": t["target_smiles"] in b.stock and t["target_smiles"] not in excluded,
        "routes": None if hidden else [_route(b, r) for r in t["reference_routes"]],
        "siblings": [r["id"] for r in b.rows if r["parent"] == t["parent_id"] and r["id"] != id],
    }


# --- live play through the same core session the server runs ---------------------------

SESSIONS: dict[str, dict[str, Any]] = {}


class NewSession(BaseModel):
    benchmark: str
    task: str
    toolset: str = "full"


class ToolCall(BaseModel):
    tool: str
    arguments: dict[str, Any] = {}


@app.post("/api/session")
def new_session(request: NewSession) -> dict[str, Any]:
    b = release(request.benchmark)
    if request.task not in b.tasks:
        raise HTTPException(404, f"no task {request.task}")
    core = load_benchmark(str(b.root.resolve()))
    session = core.session(toolset=request.toolset)
    observation = session.reset(RetroTask.from_dict(b.tasks[request.task]), b.stock)
    for key in sorted(SESSIONS, key=lambda k: SESSIONS[k]["at"])[:-40]:
        del SESSIONS[key]  # keep the 40 most recent
    sid = uuid.uuid4().hex
    SESSIONS[sid] = {"session": session, "release": b, "task": request.task, "at": time.time()}
    return {"session": sid, "observation": observation, "tools": openai_tools(request.toolset)}


def _session(sid: str) -> dict[str, Any]:
    if sid not in SESSIONS:
        raise HTTPException(404, "session expired; start a new one")
    SESSIONS[sid]["at"] = time.time()
    return SESSIONS[sid]


def _outcome(session: RetroRouteSession, tool: str, result: Any) -> dict[str, Any]:
    reward = result.get("score", {}).get("reward") if tool == "emit_routes" and isinstance(result, dict) else None
    return {
        "tool": tool,
        "result": result,
        "done": session.done,
        "reward": reward,
        "tool_calls_used": session.tool_calls,
        "tool_calls_remaining": max(0, session.max_tool_calls - session.tool_calls),
    }


@app.post("/api/session/{sid}/call")
def session_call(sid: str, request: ToolCall) -> dict[str, Any]:
    session = _session(sid)["session"]
    if request.tool not in session.tool_names:
        return {"tool": request.tool, "result": {"error": f"{request.tool} is not in the {session.toolset!r} toolset"}}
    try:
        result = getattr(session, request.tool)(**request.arguments)
    except TypeError as exc:
        result = {"error": f"bad arguments: {exc}"}
    return _outcome(session, request.tool, result)


@app.post("/api/session/{sid}/reference")
def session_reference(sid: str, reveal: int = 0) -> dict[str, Any]:
    """Submit the shortest compliant known routes, to see what a perfect episode scores."""
    entry = _session(sid)
    task = entry["release"].tasks[entry["task"]]
    if task["split"] in HELD_OUT and not reveal:
        raise HTTPException(403, "reveal the known routes first")
    submission = known_routes_submission(RetroTask.from_dict(task), entry["session"].stock)
    out = _outcome(entry["session"], "emit_routes", entry["session"].emit_routes(submission))
    return {**out, "arguments": {"submission": submission}}


# --- eval runs written by eval/run_eval.py: <runs>/<set>/<run>/{identity,progress,summary}.json, episodes/ ---


def read_json(path: Path) -> Any:
    return _read_json(path, path.stat().st_mtime_ns)


@lru_cache(maxsize=256)
def _read_json(path: Path, mtime_ns: int) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def failure_kind(text: str) -> str:
    """Group a verifier failure across tasks: drop route numbers, tree paths, SMILES lists and counts."""
    text = re.sub(r"^route \d+: ", "", text)
    if text.startswith("routes["):
        text = text.split(": ", 1)[-1]
    return re.sub(r"\d+", "N", text.split(": [")[0].split(": '")[0])


def failure_kinds(episode: dict[str, Any]) -> list[str]:
    """Distinct causes, without the pass verdict that every failed episode carries."""
    kinds = dict.fromkeys(failure_kind(f) for f in episode.get("hard_failures") or [])
    return [k for k in kinds if not k.startswith("N valid route")]


def headline_failure(episode: dict[str, Any]) -> str:
    kinds = failure_kinds(episode)
    if kinds:
        return kinds[0]
    return next((e for e in episode.get("errors") or [] if e.startswith("API")), "")


def episode_row(path: Path) -> dict[str, Any]:
    return _episode_row(path, path.stat().st_mtime_ns)


@lru_cache(maxsize=8192)
def _episode_row(path: Path, mtime_ns: int) -> dict[str, Any]:
    e = json.loads(path.read_text(encoding="utf-8"))
    usage = e.get("usage") or {}
    return {
        "task_id": e["task_id"],
        "attempt": e.get("attempt", 0),
        "smiles": e.get("target_smiles"),
        "tier": e.get("tier"),
        "max_depth": e.get("max_depth"),
        "variant": e.get("variant"),
        "graded": e.get("graded", True),
        "reward": e.get("reward"),
        "valid": bool(e.get("valid")),
        "exact": bool(e.get("exact_match")),
        "tool_calls": e.get("tool_calls"),
        "turns": e.get("turns"),
        "latency": usage.get("latency_seconds"),
        "tokens": (usage.get("prompt_tokens") or 0) + (usage.get("completion_tokens") or 0),
        "coerced": bool(e.get("submission_coerced")),
        "auto_emitted": bool(e.get("auto_emitted")),
        "failure": "" if e.get("valid") else headline_failure(e),
        "failures": failure_kinds(e),
    }


def run_dirs() -> dict[str, Path]:
    root = STATE["runs"]
    return {p.parent.relative_to(root).as_posix(): p.parent for p in sorted(root.glob("**/identity.json"))}


def run_dir(run: str) -> Path:
    try:
        return run_dirs()[run]
    except KeyError:
        raise HTTPException(404, f"unknown run {run!r}") from None


def run_episodes(root: Path) -> list[dict[str, Any]]:
    return [episode_row(p) for p in sorted((root / "episodes").glob("*.json"))]


def run_summary(run: str, root: Path) -> dict[str, Any]:
    identity = read_json(root / "identity.json")
    progress, summary = root / "progress.json", root / "summary.json"
    files = (root / "identity.json", progress, *(root / "episodes").glob("*.json"))
    updated = max(p.stat().st_mtime for p in files if p.exists())
    if summary.exists() and summary.stat().st_mtime >= updated:
        status, stats = "done", read_json(summary)
    else:
        status = "running" if time.time() - updated < LIVE_WINDOW_S else "stopped"
        stats = read_json(progress) if progress.exists() else {}
    return {
        "run": run,
        "folder": root.name,
        "label": identity.get("label") or identity.get("model"),
        "model": identity.get("model"),
        "provider": identity.get("provider"),
        "sampling": identity.get("sampling", {}),
        "max_turns": identity.get("max_turns"),
        "max_tool_calls": identity.get("max_tool_calls"),
        "status": status,
        "updated": updated,
        "expected": stats.get("episodes_expected") or len(identity.get("task_ids") or []),
        "graded": stats.get("episodes_graded", 0),
        **{
            key: stats.get(key)
            for key in (
                "pass_at_1",
                "pass_at_1_ci95",
                "exact_route_rate",
                "mean_reward",
                "mean_tool_calls",
                "mean_latency_seconds",
                "cost_usd",
                "coerced_submission_rate",
                "no_emit_rate",
                "components",
                "by_max_depth",
            )
        },
    }


def release_for(task_id: str) -> Release | None:
    return next((b for b in STATE["releases"].values() if task_id in b.tasks), None)


def tree_smiles(node: Any) -> set[str]:
    if isinstance(node, list):
        return set().union(*map(tree_smiles, node)) if node else set()
    if not isinstance(node, dict):
        return set()
    own = {node["smiles"]} if node.get("type") == "mol" and isinstance(node.get("smiles"), str) else set()
    return own | tree_smiles(node.get("children") or [])


def stock_status(task_id: str, smiles: set[str]) -> dict[str, bool]:
    b = release_for(task_id)
    if b is None:
        return {}
    excluded = set(b.tasks[task_id]["constraints"]["excluded_stock"])
    out = {}
    for s in smiles:
        try:
            canonical = canonicalize_smiles(s)
        except Exception:
            out[s] = False
            continue
        out[s] = canonical in b.stock and canonical not in excluded
    return out


@app.get("/api/evals")
def evals() -> dict[str, Any]:
    sets = Counter(Path(run).parent.as_posix() for run in run_dirs())
    return {"sets": [{"name": name, "runs": n} for name, n in sorted(sets.items())]}


@app.get("/api/evals/board")
def evals_board(name: str) -> dict[str, Any]:
    runs = {run: root for run, root in run_dirs().items() if Path(run).parent.as_posix() == name}
    if not runs:
        raise HTTPException(404, f"no runs in {name!r}")
    summaries, cells, tasks = [], {}, {}
    for run, root in runs.items():
        summaries.append(run_summary(run, root))
        for task_id in read_json(root / "identity.json").get("task_ids") or []:
            tasks.setdefault(task_id, {"id": task_id})
        cells[run] = {}
        for row in run_episodes(root):
            tasks.setdefault(row["task_id"], {"id": row["task_id"]}).update(
                smiles=row["smiles"], tier=row["tier"], max_depth=row["max_depth"]
            )
            cells[run][row["task_id"]] = {k: row[k] for k in ("reward", "valid", "exact", "graded", "failure")}
    return {"set": name, "runs": summaries, "tasks": list(tasks.values()), "cells": cells}


@app.get("/api/evals/run")
def evals_run(run: str) -> dict[str, Any]:
    root = run_dir(run)
    episodes = run_episodes(root)
    failed = [e for e in episodes if e["graded"] and not e["valid"]]
    return {
        "summary": run_summary(run, root),
        "episodes": episodes,
        "failures": Counter(kind for e in failed for kind in e["failures"]).most_common(),
        "failed": len(failed),
    }


def same_task_elsewhere(run: str, task: str, attempt: int) -> list[dict[str, Any]]:
    """This task's result for every model in the same eval set: its newest run, or the viewed run."""
    chosen: dict[str, tuple[tuple[bool, float], dict[str, Any]]] = {}
    for other, root in run_dirs().items():
        if Path(other).parent != Path(run).parent:
            continue
        episode = next((root / "episodes").glob(f"*-{task}-a{attempt}.json"), None)
        if episode is None:
            continue
        identity = read_json(root / "identity.json")
        rank = (other == run, (root / "identity.json").stat().st_mtime)
        if identity["model"] not in chosen or rank > chosen[identity["model"]][0]:
            row = episode_row(episode)
            entry = {"run": other, "label": identity.get("label") or identity["model"], "current": other == run}
            chosen[identity["model"]] = (rank, {**entry, **{k: row[k] for k in ("reward", "valid", "exact", "graded")}})
    return sorted((entry for _, entry in chosen.values()), key=lambda e: (-(e["reward"] or 0), e["label"]))


@app.get("/api/evals/episode")
def evals_episode(run: str, task: str, attempt: int = 0) -> dict[str, Any]:
    root = run_dir(run)
    paths = sorted((root / "episodes").glob(f"*-{task}-a{attempt}.json"))
    if not paths:
        raise HTTPException(404, f"no episode for {task} in {run}")
    episode = read_json(paths[0])
    submission = episode.get("submission")
    routes = submission.get("routes") if isinstance(submission, dict) else None
    order = [row["task_id"] for row in run_episodes(root)]
    at = order.index(task)
    return {
        "summary": run_summary(run, root),
        "episode": episode,
        "stock": stock_status(task, tree_smiles(routes or [])),
        "has_task": release_for(task) is not None,
        "same_task": same_task_elsewhere(run, task, attempt),
        "prev": order[at - 1] if at > 0 else None,
        "next": order[at + 1] if at + 1 < len(order) else None,
    }


@lru_cache(maxsize=20000)
def _svg(smiles: str, width: int, height: int) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'></svg>"
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    options = drawer.drawOptions()
    options.clearBackground = False
    options.bondLineWidth = 1.3
    options.padding = 0.05
    options.minFontSize = 10
    options.useBWAtomPalette()
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    text = drawer.GetDrawingText().replace("#000000", "currentColor")
    return text[text.index("<svg") :]


@app.get("/api/mol.svg")
def mol_svg(smi: str, w: int = Query(200, le=800), h: int = Query(110, le=600)) -> Response:
    return Response(_svg(smi, w, h), media_type="image/svg+xml", headers={"Cache-Control": "max-age=86400"})


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--release", type=Path, action="append", default=[], help="a release directory (repeatable)")
    parser.add_argument("--runs", type=Path, default=ROOT / "runs", help="eval output root (eval/run_eval.py --output)")
    args = parser.parse_args()
    STATE["runs"] = args.runs.resolve()
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", args.port)) == 0:
            raise SystemExit(
                f"port {args.port} is in use; open http://127.0.0.1:{args.port} or pass --port {args.port + 1}"
            )
    STATE["releases"] = discover(args.release)
    if not STATE["releases"]:
        raise SystemExit("no release found: build one with `uv run python -m dataset.build_release` or pass --release")
    for b in STATE["releases"].values():
        threading.Thread(target=lambda b=b: b.stock, daemon=True).start()
    print(f"ready: {', '.join(STATE['releases'])} on http://127.0.0.1:{args.port}", file=sys.stderr)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
