"""Publish a DesignWorldBench run as a cua-v1 computer-use dataset.

Reads rollout run directories (`trace.jsonl` plus the frames and renders rollout.py saves next to it) and writes
the `trajectories` view of cua-v1 (`designgym.cua_format`): one row per episode, `messages` in the OpenAI chat
shape (the opening user turn with the target and the first screenshot, assistant turns with `tool_calls`, one tool
message per call carrying the images it returned), `images` in the order their parts appear, `trainable` on the
assistant turns, and the bench's own columns (model, tier, grade, error, final document, ...).

    python bench/harness/to_cua.py results/rollouts/<run> --dry-run
    python bench/harness/to_cua.py results/rollouts/<run> --out /tmp/cua --config <run>
    python bench/harness/to_cua.py results/rollouts/<run> --out /tmp/cua --config <run> --push

Repo layout: one config per run, split `test`: `data/<run>/test-<model>-xxxxx-of-yyyyy.parquet` (each model's
shards are named apart, so the models of one run can be published by separate jobs), and `meta/<run>.json` (the
rollout summaries and the conversion counts). `--push` uploads both to DESIGNWORLD_RESULTS_REPO (default
FineEnvs/designworld-bench-runs) and rewrites the repo card's config list.

The trace does not hold everything a lossless row would (see bench/README.md): the system prompt and the tool
schemas are taken from the record only when present; tool result text is cut at 400 characters; arguments that were
not valid JSON are recorded as {}; calls after the one that ended the episode are not recorded; call ids are rebuilt
as call_<turn>_<k>. Episodes with a missing screenshot are skipped unless --keep-incomplete.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVALSET = HERE.parent / "evalsets" / "recreation-cua-v1.json"
RESULTS_REPO = os.environ.get("DESIGNWORLD_RESULTS_REPO", "FineEnvs/designworld-bench-runs")
SPLIT = "test"
# The opening and nudge messages, word for word as rollout.py sends them.
INTRO = {"reference": "This is the target design. Recreate it.",
         "description": "Build the design from the brief in your instructions."}
GUI_INTRO = {"reference": " The second image is the editor as it is now.",
             "description": " The image is the editor as it is now."}
NUDGE = ("Continue working with the tools. When the design matches, call the submit tool; "
         "do not stop without submitting.")
GAPS = ["system prompt: not in the trace (prompt_version only)", "tool schemas: not in the trace",
        "tool result text: cut at 400 characters", "invalid JSON arguments: recorded as {}",
        "calls after the episode-ending call: not recorded", "call ids: rebuilt as call_<turn>_<k>"]


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def _key(r: dict) -> tuple:  # the runners' episode key: a v2 frozen episode by its id, a v1 one by task and skin
    if r.get("evalset_episode"):
        return r["model"], r["evalset_episode"]
    return r["model"], r["surface"], r["mode"], r.get("evalset_task") or r["index"], r.get("skin_forced")


def load(runs: list[Path]) -> list[tuple[dict, Path]]:
    """The newest record per episode across the run directories, as rollout.py summarises them."""
    latest: dict[tuple, tuple[dict, Path]] = {}
    for run in runs:
        for line in (run / "trace.jsonl").read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                latest[_key(rec)] = (rec, run)
    return list(latest.values())


def trajectory(rec: dict, run: Path, entry: dict | None) -> tuple[dict, Counter]:
    """One trace record as a recorder-style trajectory (image parts point at files under `run`), plus counts."""
    stats: Counter = Counter()
    gui = rec["surface"] == "gui"

    def parts(paths: list, expected: int | None = None) -> list[dict]:
        out = []
        for p in paths:
            if p and (run / p).is_file():
                out.append({"type": "image", "image": p})
                stats["images"] += 1
            else:
                stats["missing_images"] += 1
        stats["missing_images"] += max(0, (expected or 0) - len(paths))  # returned but not saved (--no-frames)
        return out

    messages: list[dict] = []
    if rec.get("instructions") or rec.get("system"):  # (v1 runs: "instructions" when present; v2: "system")
        messages.append({"role": "system", "content": rec.get("instructions") or rec["system"]})
    opening = [f"renders/{rec['task_id']}__target.jpg"] if rec["mode"] == "reference" else []
    if gui:
        opening.append(rec.get("frame0"))
    intro = INTRO[rec["mode"]] + (GUI_INTRO[rec["mode"]] if gui else "")
    messages.append({"role": "user", "content": parts(opening) + [{"type": "text", "text": intro}]})
    turns, nudges = rec.get("turns") or [], 0
    for i, turn in enumerate(turns):
        calls = turn.get("calls") or []
        ids = [f"call_{turn['turn']}_{k}" for k in range(len(calls))]
        messages.append({"role": "assistant", "content": turn.get("text") or "", "tool_calls": [
            {"id": cid, "type": "function",
             "function": {"name": c["name"], "arguments": json.dumps(c.get("args") or {}, ensure_ascii=False)}}
            for cid, c in zip(ids, calls)]})
        if not calls:
            nudges += 1
            if nudges <= 2 and i < len(turns) - 1:  # a nudge is only seen if another turn follows
                messages.append({"role": "user", "content": NUDGE})
            continue
        # GUI: without a screenshot call the harness takes one after the turn and returns it with the last result
        auto = gui and "frame" in turn and not any(c["name"] == "screenshot" for c in calls)
        for k, (cid, c) in enumerate(zip(ids, calls)):
            content = [{"type": "text", "text": c.get("result", "")}] + parts(c.get("image_paths") or [],
                                                                             c.get("images", 0))
            if auto and k == len(calls) - 1:
                content += parts([turn.get("frame")])
            messages.append({"role": "tool", "tool_call_id": cid, "name": c["name"], "content": content})
    stats["tool_calls"] += rec.get("tool_calls", 0)
    meta = {"task_id": rec.get("task_id") or rec.get("evalset_task"), "design_id": (entry or {}).get("design_id"),
            "task_source": None, "mode": rec["mode"], "surface": rec["surface"],
            "skin": rec.get("skin") if gui else None, "theme": rec.get("theme"), "layout": rec.get("layout"),
            "file_dialog": rec.get("file_dialog"), "seed": rec.get("seed", 0), "reward": rec.get("reward") or 0.0,
            "turns": rec.get("n_turns", len(turns)), "actions": rec.get("tool_calls", 0),
            "n_elements": (entry or {}).get("n_elements", 0), "masked_turns": 0, "mistakes": [],
            "recorder_version": 0, "prompt_version": rec.get("prompt_version")}
    return {"meta": meta, "messages": messages, "tools": rec.get("tools")}, stats


BENCH_COLUMNS = {"model": "string", "run": "string", "evalset_id": "string", "episode_id": "string",
                 "tier": "string", "difficulty_score": "float32", "grade": "string", "error": "string",
                 "auto_submitted": "bool", "seconds": "float32", "usage": "string", "final_doc": "string",
                 "missing_images": "int32", "evalset_episode": "string", "real_app": "bool", "workspace": "string"}


def features():
    """cua-v1 `trajectories` features plus the bench columns."""
    from datasets import Features, Value

    from designgym.cua_format import features as cua_features

    feats = dict(cua_features()[0])
    feats.setdefault("design_id", Value("string"))
    feats.update({k: Value(v) for k, v in BENCH_COLUMNS.items()})
    return Features(feats)


def to_row(rec: dict, run: Path, traj: dict, stats: Counter, config: str, evalset_id: str | None,
           image_format: str) -> dict:
    from designgym.cua_format import standardize

    row = standardize(traj, run, image_format=image_format)
    gui = rec["surface"] == "gui"
    screen = None
    if gui and traj["messages"]:
        first = next((p["image"] for m in traj["messages"] if isinstance(m["content"], list)
                      for p in m["content"] if p["type"] == "image" and "renders/" not in p["image"]), None)
        if first:
            from PIL import Image

            with Image.open(run / first) as im:
                screen = {"width": im.width, "height": im.height}
    row.update(
        id=f"{_slug(rec['model'])}__{rec['surface']}__{rec['mode']}__{traj['meta']['task_id']}"
           + (f"__{rec.get('skin')}" if gui else ""),
        design_id=traj["meta"]["design_id"], coordinates=rec.get("coordinates") or "pixels",
        reward=rec.get("reward"), screen=screen, tools=json.dumps(rec["tools"]) if rec.get("tools") else None,
        model=rec["model"], run=config, evalset_id=evalset_id, episode_id=rec.get("episode_id"),
        tier=rec.get("tier"), difficulty_score=rec.get("difficulty_score"),
        grade=json.dumps(rec["grade"]) if rec.get("grade") else None, error=rec.get("error"),
        auto_submitted=bool(rec.get("auto_submitted")), seconds=rec.get("seconds"),
        usage=json.dumps(rec.get("usage") or {}),
        final_doc=json.dumps(rec["final_doc"], ensure_ascii=False) if rec.get("final_doc") is not None else None,
        missing_images=stats["missing_images"], evalset_episode=rec.get("evalset_episode"),
        real_app=bool(rec.get("real_app")),
        workspace=json.dumps(rec["workspace"], ensure_ascii=False) if rec.get("workspace") else None)
    return row


def write(rows: list[dict], feats, folder: Path, prefix: str, shard_mb: int) -> int:
    """`folder/<prefix>-xxxxx-of-yyyyy.parquet` shards of about `shard_mb` each (as cua_format._write)."""
    from datasets import Dataset

    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob(f"{prefix}-*.parquet"):  # a previous conversion of the same model
        old.unlink()
    ds = Dataset.from_list([{k: r.get(k) for k in feats} for r in rows], features=feats)
    size = sum(len(img["bytes"]) for r in rows for img in r["images"]) + sum(len(json.dumps(r["messages"])) for r in rows)
    n = max(1, math.ceil(size / (shard_mb * 1e6)))
    for k in range(n):
        ds.shard(num_shards=n, index=k, contiguous=True).to_parquet(str(folder / f"{prefix}-{k:05d}-of-{n:05d}.parquet"))
    return n


CARD = """---
license: other
task_categories: [image-text-to-text]
tags: [computer-use, gui-agent, agent-traces, design, openenv, benchmark]
configs:
{configs}
---

# DesignWorldBench runs

Every evaluation run of DesignWorldBench (*Recreation-CUA*), as `cua-v1` computer-use trajectories, the format of
the DesignWorld-SFT data: one config per run, split `test`, one row per episode. v2 runs were played by
`bench/rollout.py` on the frozen episodes of `bench/evalsets/recreation-cua-v2.json` (core30 x 14 editors, each in a
lived-in workspace); v1 runs by `envs/designgym/rollout.py` on `recreation-cua-v1.json`. Rows were converted by
`bench/harness/to_cua.py`; v2 rows also carry `evalset_episode`, `real_app` and the episode's `workspace` (JSON). Raw traces,
frames and logs are under `runs/<run>/`, conversion counts under `meta/<run>.json`. The designs keep the licence
of the source designs (see the DesignGym card).

Columns: the cua-v1 `trajectories` schema (`messages` in the OpenAI chat shape with `tool_calls`, `images` in the
order of their parts, `trainable`, `reward`, `screen`, `coordinates`, `editor`, `task_id`, `design_id`, ...) plus
`model`, `run`, `evalset_id`, `episode_id`, `tier`, `difficulty_score`, `grade` (JSON), `error`, `auto_submitted`,
`seconds`, `usage` (JSON), `final_doc` (the submitted document, JSON, so a later reward version can re-score the
run) and `missing_images`. Not recoverable from the trace: {gaps}.

## Runs

| run | model | surface | mode | episodes | mean reward |
|---|---|---|---|---:|---:|
{table}
"""


def push(out: Path, config: str, repo: str, prefixes: list[str]) -> None:
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    # upload_large_folder resumes chunk by chunk, so a slow or flaky uplink gets there in the end
    api.upload_large_folder(repo_id=repo, repo_type="dataset", folder_path=str(out), num_workers=2,
                            allow_patterns=[f"data/{config}/*.parquet", f"meta/{config}.json", f"runs/{config}/*"])
    # then drop shards of these models left from an earlier, longer publish
    local = {f.relative_to(out).as_posix() for f in (out / "data" / config).glob("*.parquet")}
    stale = [f for f in api.list_repo_files(repo, repo_type="dataset")
             if f.startswith(f"data/{config}/") and f not in local and any(f.split("/")[-1].startswith(f"{p}-") for p in prefixes)]
    if stale:
        api.delete_files(repo, stale, repo_type="dataset", commit_message=f"drop stale shards of {config}")
    files = api.list_repo_files(repo, repo_type="dataset")
    configs = sorted({f.split("/")[1] for f in files if f.startswith("data/") and f.endswith(".parquet")})
    blocks, table = [], []
    for c in configs:
        blocks.append(f"- config_name: {c}\n  data_files:\n  - split: {SPLIT}\n    path: data/{c}/*.parquet")
        if f"meta/{c}.json" in files:
            meta = json.loads(Path(hf_hub_download(repo, f"meta/{c}.json", repo_type="dataset")).read_text())
            table += [f"| {c} | {g['model']} | {g['surface']} | {g['mode']} | {g['n']} | {g['mean_reward']:.3f} |"
                      for g in meta.get("groups", [])]
    card = CARD.format(configs="\n".join(blocks), gaps="; ".join(GAPS), table="\n".join(table))
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=repo, repo_type="dataset",
                    commit_message=f"card: {len(configs)} runs")
    print(f"pushed {config} -> https://huggingface.co/datasets/{repo}/tree/main/data/{config}")


def groups(records: list[dict]) -> list[dict]:
    """Mean reward per (model, surface, mode), failures counted as 0, as rollout.py's summary."""
    g: dict[tuple, list[float]] = defaultdict(list)
    for r in records:
        g[r["model"], r["surface"], r["mode"]].append(r["reward"] if r.get("reward") is not None else 0.0)
    return [{"model": m, "surface": s, "mode": mode, "n": len(v), "mean_reward": round(statistics.fmean(v), 4)}
            for (m, s, mode), v in sorted(g.items())]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", type=Path, help="rollout run directories (each with trace.jsonl)")
    ap.add_argument("--out", type=Path, default=None, help="Hub-ready folder to write (data/, meta/)")
    ap.add_argument("--config", default=None, help="config name (default: the first run directory's name)")
    ap.add_argument("--evalset", type=Path, default=EVALSET, help="frozen set, for design_id and n_elements")
    ap.add_argument("--image-format", default="keep", choices=("keep", "png", "jpeg"))
    ap.add_argument("--shard-mb", type=int, default=500)
    ap.add_argument("--keep-incomplete", action="store_true", help="keep episodes with missing screenshots")
    ap.add_argument("--push", nargs="?", const=RESULTS_REPO, default=None, metavar="REPO",
                    help=f"upload to this dataset repo (default {RESULTS_REPO})")
    ap.add_argument("--dry-run", action="store_true", help="print counts only; needs neither designgym nor datasets")
    a = ap.parse_args()
    for run in a.runs:
        if not (run / "trace.jsonl").is_file():
            ap.error(f"{run} has no trace.jsonl")
    config = _slug(a.config or a.runs[0].name)
    record = json.loads(a.evalset.read_text())
    entries = {t["task_id"]: t for t in record["tasks"]}

    by_model: dict[str, list] = defaultdict(list)
    counts: dict[str, Counter] = defaultdict(Counter)
    loaded = load(a.runs)
    for rec, run in loaded:
        c = counts[rec["model"]]
        c["episodes"] += 1
        if not rec.get("turns"):
            c["skipped_no_turns"] += 1
            continue
        entry = entries.get(rec.get("evalset_task") or rec.get("task_id"))
        traj, stats = trajectory(rec, run, entry)
        if stats["missing_images"] and not a.keep_incomplete:
            c["skipped_missing_images"] += 1
            c["missing_images"] += stats["missing_images"]
            continue
        c.update(rows=1, errored=bool(rec.get("error")), not_in_evalset=entry is None, **stats)
        by_model[rec["model"]].append((rec, run, traj, stats))
    has = lambda key: sum(1 for rows in by_model.values() for rec, *_ in rows if rec.get(key))  # noqa: E731
    print(f"config {config}: {sum(c['episodes'] for c in counts.values())} episodes in {len(a.runs)} run dir(s), "
          f"evalset {record['name']} ({record['evalset_id'][:12]})")
    for model, c in sorted(counts.items()):
        print(f"  {model}: " + ", ".join(f"{k} {v}" for k, v in sorted(c.items()) if v))
    rows_total = sum(len(v) for v in by_model.values())
    print(f"  rows {rows_total}; with system prompt {has('instructions') + has('system')}, with tool schemas {has('tools')}")
    if a.dry_run:
        return 0
    if not a.out:
        ap.error("--out is required unless --dry-run")
    if not rows_total:
        print("nothing to write")
        return 1

    feats = features()
    folder = a.out / "data" / config
    prefixes = []
    for model, items in sorted(by_model.items()):
        rows = [to_row(rec, run, traj, stats, config, record["evalset_id"], a.image_format)
                for rec, run, traj, stats in items]
        prefix = f"{SPLIT}-{_slug(model)}"
        n = write(rows, feats, folder, prefix, a.shard_mb)
        prefixes.append(prefix)
        print(f"  wrote {len(rows)} rows of {model} in {n} shard(s)")
    summaries = {}
    for run in a.runs:
        if (run / "summary.json").is_file():
            summaries[run.name] = json.loads((run / "summary.json").read_text())
    meta = {"schema_version": "cua-v1", "config": config, "split": SPLIT, "evalset": record["name"],
            "evalset_id": record["evalset_id"], "groups": groups([rec for rec, _ in loaded]),
            "counts": {m: dict(c) for m, c in counts.items()}, "gaps": GAPS, "rollout_summaries": summaries}
    (a.out / "meta").mkdir(parents=True, exist_ok=True)
    (a.out / "meta" / f"{config}.json").write_text(json.dumps(meta, indent=1))
    # the raw run next to its rows: the trace as written, the run's settings and summary
    import shutil

    raw = a.out / "runs" / config
    raw.mkdir(parents=True, exist_ok=True)
    for run in a.runs:
        for name in ("trace.jsonl", "run.json", "summary.json", "summary.md"):
            if (run / name).is_file():
                shutil.copy(run / name, raw / (name if len(a.runs) == 1 else f"{run.name}__{name}"))
    if a.push:
        push(a.out, config, a.push, prefixes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
