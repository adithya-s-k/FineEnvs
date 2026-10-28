"""DesignGym glue for the openenv_rl recipe: opening messages and task lists per RL stage.

Stage 1 is `mode="reference"` (the target image is in the prompt); stage 2 is `mode="description"` (the brief
is in the instructions). Stage 3 (vague briefs, group-relative judging) needs a new env mode first.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _screen(episode_id: str) -> str | None:
    """The editor as the agent first sees it (GUI episodes), as a data URL, from the env server's REST API."""
    import base64
    import urllib.request

    base = os.environ.get("DESIGNGYM_ENV_URL")
    if not base:
        return None
    req = urllib.request.Request(f"{base.rstrip('/')}/api/episodes/{episode_id}/screen.png")
    if os.environ.get("HF_TOKEN") and ".hf." in base:
        req.add_header("Authorization", f"Bearer {os.environ['HF_TOKEN']}")
    with urllib.request.urlopen(req, timeout=60) as r:
        return "data:image/png;base64," + base64.b64encode(r.read()).decode()


def initial_messages(result, task) -> list[dict]:
    """The env's instructions as the system prompt; the target image (reference mode) in the user turn.

    GUI episodes open exactly as the eval harness (`rollout.py`) opens them: the target, then the editor as it is
    now, then the text, so a policy trained here is evaluated on the prompt layout it learned.
    """
    meta = result.observation.metadata or {}
    if (task or {}).get("surface") == "gui":
        images = []
        if meta.get("reference_jpeg_b64"):
            images.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + meta["reference_jpeg_b64"]}})
        screen = _screen(meta["episode_id"])
        if screen:
            images.append({"type": "image_url", "image_url": {"url": screen}})
        text = ("This is the target design. Recreate it. The second image is the editor as it is now."
                if meta.get("reference_jpeg_b64") else
                "Build the design from the brief in your instructions. The image is the editor as it is now.")
        return [{"role": "system", "content": meta["instructions"]}, {"role": "user", "content": [*images, {"type": "text", "text": text}]}]
    if meta.get("reference_jpeg_b64"):
        user = [
            {"type": "text", "text": "This is the target design. Recreate it, then call submit."},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + meta["reference_jpeg_b64"]}},
        ]
    else:
        user = "Build the design described in your instructions, then call submit."
    return [{"role": "system", "content": meta["instructions"]}, {"role": "user", "content": user}]


def write_tasks(out: Path, split: str, count: int, mode: str, surface: str, start: int = 0) -> Path:
    """One `reset()` kwargs line per task."""
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"split": split, "index": i, "mode": mode, "surface": surface} for i in range(start, start + count)]
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="rl", help="the env's task split (sft, rl, test)")
    ap.add_argument("--count", type=int, default=224)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--mode", choices=["reference", "description"], default="reference")
    ap.add_argument("--surface", choices=["tools", "html", "gui"], default="tools")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    path = a.out or Path(f"tasks/{a.split}-{a.mode}-{a.surface}.jsonl")
    print(write_tasks(path, a.split, a.count, a.mode, a.surface, a.start))
