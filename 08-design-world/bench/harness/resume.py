"""Seed a new rollout directory with a previous run's episodes, so rollout.py plays only what is missing.

rollout.py already resumes in place: an episode whose (model, surface, mode, task, skin) key has a clean record
in `<out>/trace.jsonl` is skipped, one that errored is played again, and the newest record per key goes into the
summary. This copies the matching records of an earlier run (and the frames and renders they point to) into a
fresh `<out>`, so the new run is self-contained: its summary and its CUA export cover the carried episodes too.

    python bench/harness/resume.py results/rollouts/<earlier run> results/rollouts/<new run> \
        --model 'vllm:Qwen/Qwen3.6-35B-A3B@off' --evalset bench/evalsets/recreation-cua-v1.json --subset lite

The source is a local run directory or a run on the Hub: `runs/<name>` or a bare `<name>` in the results repo,
or `<org>/<repo>/<path>` for any dataset repo. A run directory holds `trace.jsonl`, or `rollouts-*/trace.jsonl`
(the HF Jobs layout); every one found is read.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

RESULTS_REPO = os.environ.get("DESIGNWORLD_RESULTS_REPO", "FineEnvs/designworld-bench-runs")


def parse_hub(src: str, repo: str) -> tuple[str, str]:
    """'<name>', 'runs/<name>', 'inputs/...' (in `repo`) or '<org>/<repo>/<path>' -> (repo id, path in repo)."""
    src = src.removeprefix("hf://").removeprefix("datasets/").strip("/")
    if "/" not in src:
        return repo, f"runs/{src}"
    if src.startswith(("runs/", "inputs/")):
        return repo, src
    org, name, *rest = src.split("/")
    if not rest:
        raise SystemExit(f"{src!r}: give a path inside the repo, e.g. {org}/{name}/runs/<name>")
    return f"{org}/{name}", "/".join(rest)


def locate(src: str, repo: str = RESULTS_REPO) -> list[Path]:
    """Every trace.jsonl of the source run, downloading the run's traces, frames and renders from the Hub."""
    local = Path(src)
    if not local.is_dir():
        from huggingface_hub import snapshot_download

        repo_id, path = parse_hub(src, repo)
        traces = [f"{path}/trace.jsonl", f"{path}/*/trace.jsonl"]
        root = Path(snapshot_download(repo_id, repo_type="dataset", allow_patterns=traces)) / path
        dirs = [t.parent.relative_to(root) for t in _traces(root)]
        media = [f"{path}/{d}/{kind}/**".replace("/./", "/") for d in dirs for kind in ("frames", "renders")]
        if media:
            snapshot_download(repo_id, repo_type="dataset", allow_patterns=media)
        local = root
    return _traces(local)


def _traces(root: Path) -> list[Path]:
    return [p for p in [root / "trace.jsonl", *sorted(root.glob("*/trace.jsonl"))] if p.is_file()]


def _media(rec: dict) -> list[str]:
    """Files under the run directory a record points to."""
    paths = [rec.get("frame0"), rec.get("render")]
    for turn in rec.get("turns") or []:
        paths.append(turn.get("frame"))
        for call in turn.get("calls") or []:
            paths += call.get("image_paths") or []
    if rec.get("mode") == "reference" and rec.get("task_id"):
        paths.append(f"renders/{rec['task_id']}__target.jpg")
    return [p for p in paths if p]


def seed(sources: list[Path], out: Path, *, model: str | None = None, surfaces: set | None = None,
         modes: set | None = None, skins: set | None = None, task_ids: set | None = None) -> dict:
    """Append the matching records of `sources` to `out/trace.jsonl` and copy their files; returns counts."""
    out.mkdir(parents=True, exist_ok=True)
    kept, clean, files, seen = 0, 0, 0, 0
    with open(out / "trace.jsonl", "a") as trace:
        for src in sources:
            for line in src.read_text().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                seen += 1
                if ((model and rec.get("model") != model) or (surfaces and rec.get("surface") not in surfaces)
                        or (modes and rec.get("mode") not in modes)
                        or (skins and rec.get("surface") == "gui" and rec.get("skin_forced") not in skins)
                        or (task_ids is not None and rec.get("evalset_task") not in task_ids)):
                    continue
                trace.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1
                clean += not rec.get("error")
                for rel in _media(rec):
                    a, b = src.parent / rel, out / rel
                    if a.is_file() and not b.exists():
                        b.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(a, b)
                        files += 1
    return {"records_read": seen, "carried": kept, "clean": clean, "errored": kept - clean, "files": files}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("source", help="previous run: local directory, runs/<name>, <name>, or <org>/<repo>/<path>")
    ap.add_argument("out", help="the new run directory (rollout.py --out)")
    ap.add_argument("--repo", default=RESULTS_REPO, help="results repo for runs/<name> sources")
    ap.add_argument("--model", default=None, help="only this model label, as in the trace (e.g. vllm:<repo>@off)")
    ap.add_argument("--surfaces", default=None, help="only these surfaces (comma list)")
    ap.add_argument("--modes", default=None, help="only these modes (comma list)")
    ap.add_argument("--skins", default=None, help="GUI: only episodes run with these --skins (comma list)")
    ap.add_argument("--evalset", default=None, help="with --subset: only tasks of that subset")
    ap.add_argument("--subset", default=None)
    a = ap.parse_args()
    task_ids = None
    if a.evalset and a.subset:
        record = json.loads(Path(a.evalset).read_text())
        task_ids = ({t["task_id"] for t in record["tasks"]} if a.subset == "full"
                    else set(record["subsets"][a.subset]))
    sources = locate(a.source, a.repo)
    if not sources:
        print(f"no trace.jsonl under {a.source}")
        return 1
    split = lambda v: set(v.split(",")) if v else None  # noqa: E731
    counts = seed(sources, Path(a.out), model=a.model, surfaces=split(a.surfaces), modes=split(a.modes),
                  skins=split(a.skins), task_ids=task_ids)
    print(f"resume from {a.source}: {json.dumps(counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
