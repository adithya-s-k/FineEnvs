"""`designgym` command line (also installed as `design-canvas`, its earlier name).

    designgym pull [--splits test]                           download the task pack from the Hub
    designgym serve [--port 8000] [--workers N]              OpenEnv server + editor (N processes, one URL)
    designgym mcp                                            stdio MCP server for Claude & co.
    designgym export-hf --cua run/cua.jsonl --out hf/cua     cua-v1 dataset (Hub-ready)
    designgym smoke                                          offline end-to-end check

    designgym evalset verify FILE             check a frozen set against the task pack
    designgym evalset export FILE --out DIR   write it as an offline task pack
    designgym evalset subset FILE --name N --size K [--within lite]   nest a smaller subset in the set
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _pack_arg(p: argparse.ArgumentParser):
    p.add_argument("--pack", default=os.environ.get("DESIGN_CANVAS_PACK"),
                   help="task pack directory (default ~/.cache/designgym/pack)")


def cmd_pull(args) -> int:
    from .core.pack import DEFAULT_PACK, pull_pack

    out = pull_pack(Path(args.pack or DEFAULT_PACK).expanduser(), repo=args.repo, revision=args.revision,
                    splits=args.splits.split(",") if args.splits else None)
    print(f"pack ready at {out}")
    return 0


def cmd_serve(args) -> int:
    if args.pack:
        os.environ["DESIGN_CANVAS_PACK"] = str(Path(args.pack).expanduser())
    os.environ["PORT"] = str(args.port)
    from .server.app import main

    print(f"DesignGym on http://{args.host}:{args.port}  "
          f"(editor /editor/, playground /web/, MCP /mcp)")
    main(host=args.host, port=args.port, workers=args.workers)
    return 0


def cmd_mcp(args) -> int:
    if args.pack:
        os.environ["DESIGN_CANVAS_PACK"] = str(Path(args.pack).expanduser())
    from .server.stdio_mcp import run

    run(editor_port=args.editor_port)
    return 0


def cmd_export_hf(args) -> int:
    from .cua_format import Window, export_hf

    summary = export_hf(Path(args.cua), Path(args.out), image_format=args.image_format, min_reward=args.min_reward,
                        window=Window(args.max_images, args.keep_images), shard_mb=args.shard_mb,
                        with_windows=args.windows)
    print(json.dumps(summary, indent=2))
    return 0


def cmd_smoke(args) -> int:
    from .smoke import run

    return run(args.pack)


def cmd_evalset(args) -> int:
    from .core import evalset as ev
    from .core.pack import load_pack

    record = ev.load(args.file)
    if args.action == "subset":  # the set alone, no pack needed
        ev.add_subset(record, args.name, args.size, args.within)
        ev.save(record, args.file)
        print(json.dumps({"evalset_id": record["evalset_id"], "subset": args.name,
                          **ev.summarize(record, args.name), "task_ids": record["subsets"][args.name]}, indent=2))
        return 0
    pack = load_pack(args.pack)
    ev.check_against(record, pack)
    if args.action == "export":
        print(json.dumps(ev.export_pack(pack, record, args.out), indent=2))
    else:
        print(json.dumps({"evalset_id": record["evalset_id"], "ok": True, **record["summary"]}, indent=2))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="designgym", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pull", help="download the task pack from the Hub")
    _pack_arg(p)
    p.add_argument("--repo", default=None, help="Hub dataset repo (default DESIGNGYM_PACK_REPO)")
    p.add_argument("--revision", default=None)
    p.add_argument("--splits", default=None, help="comma list (e.g. test): only these splits' designs")
    p.set_defaults(fn=cmd_pull)

    p = sub.add_parser("serve", help="run the OpenEnv server with the editor")
    _pack_arg(p)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8007")))
    p.add_argument("--workers", type=int, default=int(os.environ.get("DESIGN_CANVAS_WORKERS", "1")),
                   help="server processes behind one URL (a session-aware router); 1 = a single process")
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("mcp", help="stdio MCP server (e.g. claude mcp add designgym -- ...)")
    _pack_arg(p)
    p.add_argument("--editor-port", type=int, default=8007,
                   help="also serve the live editor on this port (0 to disable)")
    p.set_defaults(fn=cmd_mcp)

    p = sub.add_parser("export-hf", help="write recorded trajectories as a cua-v1 Hugging Face dataset folder")
    p.add_argument("--cua", required=True, help="trajectories in cua.jsonl form (images next to it)")
    p.add_argument("--out", required=True)
    p.add_argument("--image-format", default="keep", choices=["keep", "png", "jpeg"],
                   help="keep = the screenshots' own format (WebP), smallest")
    p.add_argument("--min-reward", type=float, default=0.98)
    p.add_argument("--max-images", type=int, default=8)
    p.add_argument("--keep-images", type=int, default=3)
    p.add_argument("--shard-mb", type=int, default=500)
    p.add_argument("--windows", action="store_true", help="also store the SFT windows view (duplicates screenshots)")
    p.set_defaults(fn=cmd_export_hf)

    p = sub.add_parser("smoke", help="offline end-to-end check")
    _pack_arg(p)
    p.set_defaults(fn=cmd_smoke)

    p = sub.add_parser("evalset", help="verify, export or subset the frozen evaluation set")
    p.add_argument("action", choices=["verify", "export", "subset"])
    p.add_argument("file", help="frozen set, e.g. bench/evalsets/recreation-cua-v1.json")
    _pack_arg(p)
    p.add_argument("--name", help="subset name (subset)")
    p.add_argument("--within", default="lite", help="subset to nest inside (subset)")
    p.add_argument("--out", help="pack directory (export)")
    p.add_argument("--size", type=int, default=30, help="subset size (subset)")
    p.set_defaults(fn=cmd_evalset)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
