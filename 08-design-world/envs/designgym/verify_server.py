"""End-to-end check of a running DesignGym server (what training and evaluation rely on).

    DESIGN_CANVAS_PACK=<same as the server> .venv/bin/python -m designgym.verify_server

The client loads the same task pack as the server, so ground truth matches. Checks: HTTP routes and target gating;
expert replays over the network on the tools and HTML surfaces in both modes (each must score ~1.0); an empty
submission scores 0; and concurrent episodes.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from openenv.core.env_server.mcp_types import CallToolAction
from openenv.core.mcp_client import MCPToolClient

try:
    from .core import expert
    from .core.pack import load_pack
except ImportError:
    from designgym.core import expert
    from designgym.core.pack import load_pack

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)
    return ok


def status(url: str) -> int:
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


def episode(server: str, task: dict, gt, *, surface: str, mode: str, empty: bool = False) -> tuple[float, str]:
    """Replay the expert (or nothing) over the WebSocket step transport; returns (reward, episode_id)."""
    with MCPToolClient(base_url=server, message_timeout_s=300).sync() as env:
        r = env.reset(task_id=task["task_id"], mode=mode, surface=surface)
        eid = r.observation.metadata.get("episode_id", "")
        if not empty:
            if surface == "tools":
                for call in [c for c in expert.tool_calls(gt) if c["name"] != "submit"]:
                    out = env.step(CallToolAction(tool_name=call["name"], arguments=call["arguments"]))
                    if out.observation.error is not None:
                        raise RuntimeError(f"{call['name']}: {out.observation.error}")
            else:
                env.step(CallToolAction(tool_name="write_html", arguments={"html": expert.html_page(gt)}))
        out = env.step(CallToolAction(tool_name="submit", arguments={}))
        return float(out.reward or 0.0), eid


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--server", default="http://127.0.0.1:8007")
    ap.add_argument("--split", default="train")
    ap.add_argument("--index", type=int, default=3)
    args = ap.parse_args(argv)
    t0 = time.time()
    s = args.server.rstrip("/")
    pack = load_pack()
    task = pack.at(args.split, args.index)
    gt = pack.gt_doc(task)
    print(f"server {s} | task {task['task_id']} '{task['title'][:50]}' ({len(gt.elements)} elements)", flush=True)

    print("http", flush=True)
    check("health", status(f"{s}/health") == 200)
    check("playground /web/", status(f"{s}/web/") == 200)
    check("editor /editor/", status(f"{s}/editor/") == 200)
    check("no design browser (/api/tasks gone)", status(f"{s}/api/tasks") in (404, 405))

    print("expert replays over the network", flush=True)
    for surface in ("tools", "html"):
        for mode in ("reference", "description"):
            reward, eid = episode(s, task, gt, surface=surface, mode=mode)
            check(f"{surface}/{mode} expert scores ~1.0", reward >= 0.999, f"reward {reward:.4f}")
    with MCPToolClient(base_url=s).sync() as env:
        r = env.reset(task_id=task["task_id"], mode="reference", surface="tools")
        eid = r.observation.metadata["episode_id"]
        tools = {t.name for t in env.list_tools(use_cache=False)}
        check("tools surface lists its tools", {"add_text", "add_image", "submit"} <= tools, f"{len(tools)} tools")
        check("target hidden before submit", status(f"{s}/api/episodes/{eid}/reference.jpg") == 403)
    reward, _ = episode(s, task, gt, surface="tools", mode="reference", empty=True)
    check("empty submission scores 0", reward <= 0.01, f"reward {reward:.4f}")

    print("concurrency", flush=True)
    with cf.ThreadPoolExecutor(4) as ex:
        rewards = list(ex.map(lambda _: episode(s, task, gt, surface="tools", mode="reference")[0], range(4)))
    check("4 concurrent expert episodes", all(r >= 0.999 for r in rewards), f"rewards {[round(r, 4) for r in rewards]}")

    failed = [n for n, ok, _ in CHECKS if not ok]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed in {time.time() - t0:.0f}s")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
