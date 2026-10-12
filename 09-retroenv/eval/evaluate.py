#!/usr/bin/env python3
"""Evaluate models on a RetroEnv evaluation set: name the models, get one board.

    uv run --extra eval python eval/evaluate.py claude-sonnet-5-5 "Qwen/Qwen3.8-27B:novita"
    uv run --extra eval python eval/evaluate.py --board eval/boards/core30-nothink.json

The provider follows from the model id: claude-* runs on the Anthropic API, gpt-* on the OpenAI
Responses API, and a Hub id on the HF router (pin a provider, org/name:provider, so prices are
stable). Keys come from ANTHROPIC_API_KEY, OPENAI_API_KEY and HF_TOKEN, read from the environment
or from the nearest .env file above the working directory.

--set is an evaluation set in data/ (core30, the default, or final_eval) or a whole split such as
dev or test_id. By default thinking is off (lowest effort where a model cannot turn it off), a
malformed submission is graded as sent, and the environment's own budget applies: 16 model turns
and 32 tool calls. Other run_eval.py options pass through, e.g. --attempts 3.

Every model runs at once, each in its own run_eval.py process writing <output>/<model>/ and
<output>/<model>.log, against one server: --server, or a local one serving the pinned bucket
snapshot (envs/retro_route/openenv/prepare.py; the first start downloads it). A rerun resumes each
model where it stopped, so a finished model costs nothing, and the board in <output>/results
covers every model in <output>: a model evaluated later joins the same table.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path

from run_eval import infer_provider, local_server

ROOT = Path(__file__).resolve().parents[1]
OPENENV = ROOT / "envs" / "retro_route" / "openenv"


def slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9.]+", "_", model).strip("_")


def load_env_file() -> None:
    """Fill unset variables from the nearest .env above the working directory."""
    for directory in (Path.cwd(), *Path.cwd().parents):
        path = directory / ".env"
        if path.is_file():
            for line in path.read_text().splitlines():
                key, sep, value = line.strip().removeprefix("export ").partition("=")
                if sep and key and not key.startswith("#"):
                    os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
            return


def resolve_set(name: str) -> tuple[str, list[str] | None]:
    """(split to request, task ids) for an evaluation set in data/; a plain split runs whole."""
    path = ROOT / "data" / f"eval-{name}.json"
    if not path.exists():
        return name, None
    record = json.loads(path.read_text())
    # core30 is served inside final_eval; final_eval is served as itself.
    return record["design"].get("nested_in", name), [task["task_id"] for task in record["tasks"]]


def prepared_snapshot() -> Path:
    """Fetch or check the pinned serving snapshot as start.sh does; the directory to serve."""
    printed = subprocess.run(
        [sys.executable, "prepare.py"], cwd=OPENENV, check=True, stdout=subprocess.PIPE, text=True
    ).stdout
    return (OPENENV / printed.strip().splitlines()[-1]).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="*", help="model ids; with --board, a subset of its models")
    parser.add_argument("--board", type=Path, help="a board file naming models and their options (eval/boards/)")
    parser.add_argument("--set", help="evaluation set or split (default: the board's, else core30)")
    parser.add_argument("--thinking", choices=("off", "default"), help="default: the board's, else off")
    parser.add_argument("--repair", action="store_true", help="repair malformed submissions instead")
    parser.add_argument("--server", help="URL of a running RetroEnv server; default starts a local one")
    parser.add_argument("--output", type=Path, help="default: runs/<set>-nothink, or runs/<set> with thinking")
    parser.add_argument("--parallel", type=int, help="models at once (default: all)")
    parser.add_argument("--concurrency", type=int, default=3, help="episodes at once per model")
    parser.add_argument("--max-cost", type=float, default=25.0, help="USD per model, then it stops scheduling")
    parser.add_argument("--tasks", type=int, help="only the first N tasks, a smoke test (default output gets -smoke)")
    args, passthrough = parser.parse_known_args()

    board = json.loads(args.board.read_text()) if args.board else {}
    specs = [entry if isinstance(entry, dict) else {"model": entry} for entry in board.get("models", [])]
    if specs and args.models:
        listed = {spec["model"] for spec in specs}
        specs = [spec for spec in specs if spec["model"] in args.models]
        specs += [{"model": model} for model in args.models if model not in listed]
    elif args.models:
        specs = [{"model": model} for model in dict.fromkeys(args.models)]
    if not specs:
        parser.error("name at least one model, or a --board")
    unknown = [
        spec["model"]
        for spec in specs
        if not infer_provider(spec["model"]) and "--provider" not in (*spec.get("args", []), *passthrough)
    ]
    if unknown:
        parser.error(f"no provider follows from {', '.join(unknown)}; pass --provider (and --endpoint) through")
    load_env_file()

    name = args.set or board.get("set", "core30")
    thinking = args.thinking or board.get("thinking", "off")
    split, task_ids = resolve_set(name)
    output = args.output or ROOT / "runs" / (f"{name}-nothink" if thinking == "off" else name)
    if args.tasks and not args.output:
        output = output.with_name(f"{output.name}-smoke")
    output.mkdir(parents=True, exist_ok=True)
    if task_ids is None:
        selection = ["--tasks", str(args.tasks)] if args.tasks else []
    else:
        ids_file = output / "task_ids.txt"
        ids_file.write_text("\n".join(task_ids[: args.tasks]) + "\n")
        selection = ["--task-ids", str(ids_file)]

    def run(spec: dict, url: str) -> int:
        out = output / slug(spec["model"])
        command = [
            sys.executable, str(ROOT / "eval" / "run_eval.py"),
            "--model", spec["model"],
            "--split", split, *selection,
            "--server", url,
            "--thinking", thinking,
            *([] if args.repair else ["--no-repair"]),
            "--max-tokens", "16000",
            "--concurrency", str(spec.get("concurrency", args.concurrency)),
            "--max-cost", str(spec.get("max_cost", args.max_cost)),
            "--output", str(out),
            *spec.get("args", []),
            *passthrough,
        ]  # fmt: skip
        with output.joinpath(f"{out.name}.log").open("w") as log:
            code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT).returncode
        print(f"{'done' if code == 0 else f'exit {code} (see {log.name})'}: {spec['model']}", flush=True)
        return code

    sessions = sum(spec.get("concurrency", args.concurrency) for spec in specs)
    server = (
        nullcontext(args.server)
        if args.server
        else local_server(prepared_snapshot(), "full", sessions, output / "server.log")
    )
    with server as url:
        size = len(task_ids[: args.tasks]) if task_ids is not None else args.tasks or "all"
        print(f"{len(specs)} model(s) on {name} ({split}, {size} tasks) via {url} -> {output}", flush=True)
        with ThreadPoolExecutor(max_workers=args.parallel or len(specs)) as pool:
            codes = list(pool.map(lambda spec: run(spec, url), specs))

    runs = sorted(path.parent for path in output.glob("*/summary.json"))
    if runs:
        subprocess.run(
            [sys.executable, str(ROOT / "eval" / "summarize.py"), *map(str, runs), "--output", str(output / "results")],
            cwd=ROOT,
        )
        print(f"board: {output / 'results' / 'RESULTS.md'}")
    return 1 if any(codes) else 0


if __name__ == "__main__":
    raise SystemExit(main())
