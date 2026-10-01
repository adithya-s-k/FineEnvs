#!/usr/bin/env python3
"""Pull every review thread from the Space into one readable file.

    uv run --with huggingface_hub python review/pull_comments.py                 # from https://fineenvs-multi-harness-rl.hf.space
    uv run --with huggingface_hub python review/pull_comments.py --all           # include resolved threads in comments.md
    uv run --with huggingface_hub python review/pull_comments.py --url http://localhost:4331

Uses the owner-only /api/review/export endpoint with your local Hugging Face login (the token only
proves to the Space who is asking). Writes review/.pulled/threads/*.json, one per thread, and
review/.pulled/comments.md, grouped by chapter and section with the quoted text each comment points
at. review/.pulled/ is git-ignored, so the comments never reach the repository or the Space.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from collections import defaultdict
from pathlib import Path

from huggingface_hub import get_token

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="https://fineenvs-multi-harness-rl.hf.space")
    ap.add_argument("--out", default=str(HERE / ".pulled"))
    ap.add_argument("--all", action="store_true", help="include resolved threads in comments.md")
    args = ap.parse_args()

    token = get_token()
    if not token:
        raise SystemExit("Log in first: hf auth login")
    req = urllib.request.Request(f"{args.url.rstrip('/')}/api/review/export", headers={"Authorization": f"Bearer {token}"})
    threads = json.load(urllib.request.urlopen(req))

    out = Path(args.out)
    (out / "threads").mkdir(parents=True, exist_ok=True)
    for t in threads:
        (out / "threads" / f"{t['id']}.json").write_text(json.dumps(t, ensure_ascii=False, indent=1))

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for t in sorted(threads, key=lambda t: t["created_at"]):
        if t.get("status") == "resolved" and not args.all:
            continue
        a = t["anchor"]
        groups[(a.get("chapter") or "(no chapter)", a.get("section") or "")].append(t)

    lines = [f"# Review comments ({sum(len(v) for v in groups.values())} shown, {len(threads)} in total)", ""]
    for (chapter, section), ts in groups.items():
        lines += [f"## {chapter}" + (f" › {section}" if section else ""), ""]
        for t in ts:
            a = t["anchor"]
            target = f"Figure: {a.get('title') or a.get('figure')}" if a["type"] == "figure" else f"“{a['quote']}”"
            state = t.get("decision") or t.get("status", "open")
            lines += [f"### {t['id']} [{state}]", f"> {target}", ""]
            sug = t.get("suggestion")
            if sug:
                lines += [f"**Suggestion:** delete “{a['quote']}”" if sug["action"] == "delete"
                          else f"**Suggestion:** replace “{a['quote']}” with “{sug['text']}”", ""]
            for m in t["messages"]:
                lines.append(f"- **{m['author']['name']}** ({m['author']['username']}, {m['created_at']}): {m['body']}")
            lines.append("")
    (out / "comments.md").write_text("\n".join(lines))
    print(f"{len(threads)} threads from {args.url} -> {out}/comments.md")


if __name__ == "__main__":
    main()
