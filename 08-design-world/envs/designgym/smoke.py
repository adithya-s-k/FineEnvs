"""Offline smoke test: the environment works end to end, no server, no network.

Replays the expert on one task through the in-process environment on the tools
and HTML surfaces, in both modes, and checks every episode scores 1.0; then
checks an empty submission scores 0.
"""

from __future__ import annotations

import time

from openenv.core.env_server.mcp_types import CallToolAction

try:
    from .core import expert
    from .core.pack import load_pack
    from .server.environment import DesignCanvasEnvironment
except ImportError:
    from core import expert
    from core.pack import load_pack
    from server.environment import DesignCanvasEnvironment


def run(pack_dir=None) -> int:
    pack = load_pack(pack_dir)
    split = pack.splits()[-1]
    task = pack.at(split, 0)
    gt = pack.gt_doc(task)
    ok = True
    for surface in ("tools", "html"):
        for mode in ("reference", "description"):
            env = DesignCanvasEnvironment(pack)
            t = time.time()
            env.reset(task_id=task["task_id"], surface=surface, mode=mode)
            calls = expert.tool_calls(gt) if surface == "tools" else [
                {"name": "write_html", "arguments": {"html": expert.html_page(gt)}},
                {"name": "submit", "arguments": {}}]
            out = None
            for c in calls:
                out = env.step(CallToolAction(tool_name=c["name"], arguments=c["arguments"]))
            good = out.done and abs(out.reward - 1.0) < 1e-4
            ok &= good
            print(f"{'ok ' if good else 'BAD'} {task['task_id']} {surface:5s} {mode:11s} "
                  f"reward {out.reward:.4f} in {len(calls)} calls, {time.time() - t:.2f}s")
    env = DesignCanvasEnvironment(pack)
    env.reset(task_id=task["task_id"])
    out = env.step(CallToolAction(tool_name="submit", arguments={}))
    good = out.done and out.reward == 0.0
    ok &= good
    print(f"{'ok ' if good else 'BAD'} empty submission reward {out.reward:.4f}")
    counts = ", ".join(f"{s} {pack.count(s)}" for s in pack.splits())
    print(("smoke passed" if ok else "smoke FAILED") + f" · pack: {counts}")
    return 0 if ok else 1


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
