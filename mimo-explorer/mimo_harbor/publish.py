"""Stage each converted dataset as a Hub dataset repo, laid out like the other FineEnvs Harbor sets.

    uv run python -m mimo_harbor.publish ../mimo-v2.6-rl-harbor OUT_STAGE      # stage only
    uv run python -m mimo_harbor.publish ../mimo-v2.6-rl-harbor OUT_STAGE --push

Each repo: tasks/<id>/ (the Harbor tasks), registry.json, data/tasks.jsonl (a metadata index), manifest.json
(sha256 per task), jobs/<name>.yaml and agents/mimo_opencode.py (the reference agent), README.md, LICENSE, LICENSE-mimoagent.md and NOTICE.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tomllib
from pathlib import Path

from . import adapter, jobs, source

ORG = "FineEnvs"
EXPLORER = "https://huggingface.co/spaces/FineEnvs/MiMo-RL-Envs-Explorer"
VISUALISER = "https://huggingface.co/spaces/HuggingFaceH4/harbor-visualiser?dataset="
ADAPTER_SRC = "https://github.com/adithya-s-k/FineEnvs/tree/mimo-explorer-rollouts/mimo-explorer/mimo_harbor"

INFO = {
    "code": ("Code", "Fix a real issue in a real repository",
             "The agent gets an issue and the repository at its base commit. After it finishes, the files the hidden test "
             "patch touches are reset, the patch is applied, and the task's own test command decides the reward: 1 when "
             "it passes, 0 when it doesn't.", ["code", "swe", "software-engineering"]),
    "cyber": ("Cyber", "Reproduce a real memory-safety crash (ARVO)",
              "The agent gets a sanitizer report and the project's source and fuzzer binary, and has to submit an input "
              "that crashes it in the expected function with the expected error. The image's own verify server re-runs the "
              "last submission: reward 1 for the right crash, 0 otherwise.", ["security", "cyber", "arvo", "vulnerability"]),
    "general": ("General", "Work in a simulated company through its MCP systems",
                "Each task is a workplace with 5 to 15 business systems (finance, legal, HR, operations, ...) served over MCP, "
                "a workspace of documents, and a brief. The agent works through the systems as an unprivileged user; the "
                "task's own verifier grades the outcome with rule checks and an LLM judge.", ["mcp", "tool-use", "enterprise"]),
    "terminal": ("Terminal", "Terminal-Bench style tasks",
                 "Self-contained command-line tasks in the Terminal-Bench format, graded by each task's own pytest suite and "
                 "anti-hack guard.", ["terminal", "terminal-bench", "cli"]),
    "webdev": ("Webdev", "Build a website from a design brief",
               "The agent builds the site described in the brief and delivers it to dist/. The grader renders the page in "
               "a headless browser and a vision model scores the full-page screenshot on layout, typography, colour, "
               "whitespace, content, brief fulfilment and assets.", ["web-development", "frontend", "vision-judge"]),
    "music": ("Music", "Compose a piece in ABC notation",
              "The agent writes one piece of music, in ABC notation, to the brief. Xiaomi's scorer turns it into MIDI and "
              "measures 18 human-likeness features; a validity gate (notation errors, bar lengths) zeroes invalid pieces.",
              ["music", "abc-notation", "creative"]),
}
GRADER = {"code": "hidden tests (deterministic)", "cyber": "PoC verify server (deterministic)",
          "general": "rule checks + LLM judge (Inkling)", "terminal": "pytest + anti-hack guard (deterministic)",
          "webdev": "vision judge (Llama-4-Maverick)", "music": "feature scorer + validity gate (deterministic)"}


NOTICE = f"""MiMo-V2.6-RL in Harbor format
Converted by Hugging Face and the FineEnvs team. Not affiliated with or endorsed by Xiaomi.

This work is derived from, and redistributes material of:

  MiMo-V2.6-RL-oss (https://huggingface.co/datasets/{source.DATASET}, revision {source.REVISION})
  Copyright Xiaomi Corporation. Licensed under the Apache License, Version 2.0 (LICENSE).
  The task prompts, test patches, test commands, Terminal-Bench test files, General-domain verifier files,
  and the Docker images referenced by digest (docker.io/xiaomimimo/mimo-v2.6-rl-oss) are Xiaomi's.

  XiaomiMiMo/verl (https://github.com/XiaomiMiMo/verl, commit a2ad9f6), Apache License 2.0 (LICENSE):
  tests/music_scorer/ (recipes/design/music/scorer), tests/webdev/shot.py, tests/webdev/eval_rubric.py,
  tests/webdev/verdict.py (parse_verdict from eval_mode.py), the Webdev agent prompt, and the step limits.

  XiaomiMiMo/mimoagent (https://github.com/XiaomiMiMo/mimoagent, commit 467f0a1), MIT License
  (LICENSE-mimoagent.md; Copyright (c) 2026 Xiaomi Corporation, Copyright (c) 2025 Kilian A. Lieret and
  Carlos E. Jimenez (mini-swe-agent)): tests/server_arvo.py, and the Code-domain residue scrub, cache scrub,
  git-clean keep list, test-file reset and ARVO description parsing, ported into the setup and test scripts.

Vendored files are unmodified. Changes made in this conversion (Apache-2.0 section 4(b)):
  - Each environment is repackaged as a Harbor task: task.toml, instruction.md, environment/, tests/.
  - Setup that Xiaomi's harness runs before the agent is written as environment/setup/setup.sh and run from
    the task's healthcheck. In the Code residue scrub, /logs is emptied rather than deleted (Harbor owns it).
  - Grading entry points (tests/test.sh, grade.py, verify.py) wrap the original graders. Cyber re-runs the
    last submitted PoC with server_arvo.py's own logic; General redacts judge-key echoes from its output.
  - Music: "Write your complete reply to /app/answer.md." is appended to each prompt, because a Harbor task
    is an agent session rather than a single completion.
  - Images are referenced by digest instead of by tag; the MCP sidecar pins mcp==1.26.0.
  - agents/mimo_opencode.py extends Harbor's OpenCode agent (Apache-2.0, harbor-framework/harbor).

Harbor (https://github.com/harbor-framework/harbor) is the task format and runner; it is not redistributed here.
"""


def repo_id(kind: str) -> str:
    return f"{ORG}/MiMo-V2.6-RL-harbor-{kind}"


def index_row(d: Path, kind: str) -> dict:
    t = tomllib.loads((d / "task.toml").read_text())
    m, env = t["metadata"], t["environment"]
    row = {"task_id": d.name, "source_id": m["source_id"], "domain": kind, "task_path": f"tasks/{d.name}",
           "docker_image": env["docker_image"], "step_limit": m["reference_step_limit"],
           "agent_timeout_sec": t["agent"]["timeout_sec"], "verifier_timeout_sec": t["verifier"]["timeout_sec"],
           "instruction": (d / "instruction.md").read_text()[:4000]}
    if kind == "general":
        row["mcp_systems"] = len(env.get("mcp_servers") or [])
    if kind == "cyber":
        row["expected_crash"] = m.get("expected_crash")
    return row


def card(kind: str, n: int, rows: list[dict]) -> str:
    title, short, body, tags = INFO[kind]
    rid = repo_id(kind)
    ex = rows[0]["task_id"]
    tag_lines = "\n".join(f"- {x}" for x in ["rl-environment", "reinforcement-learning", "harbor", "agent", "mimo",
                                            "mimo-v2.6-rl", kind] + tags)
    size = "1K<n<10K" if n >= 1000 else "n<1K"
    return f"""---
license: apache-2.0
language:
- en
- zh
task_categories:
- other
tags:
{tag_lines}
size_categories:
- {size}
viewer: false
---

[![View tasks in Harbor Visualiser](https://img.shields.io/badge/%F0%9F%A4%97%20Harbor%20Visualiser-View%20tasks-FFD21F?style=for-the-badge)]({VISUALISER}{rid})
[![MiMo RL Environment Explorer](https://img.shields.io/badge/%F0%9F%A4%97%20Explorer-Run%20a%20rollout-FFD21F?style=for-the-badge)]({EXPLORER})

# MiMo-V2.6-RL {title} (Harbor)

**{short}.** {n:,} Harbor tasks from the {title} domain of Xiaomi's
[MiMo-V2.6-RL-oss](https://huggingface.co/datasets/{source.DATASET}), the RL environments MiMo-V2.6 was trained
on, converted so every one runs as a standard Harbor task.

{body}

| | |
|---|---|
| Tasks | {n:,} |
| Graded by | {GRADER[kind]} |
| Reference step limit | {rows[0]['step_limit']} |
| Source | [`{source.DATASET}`](https://huggingface.co/datasets/{source.DATASET}) at `{source.REVISION[:12]}` |
| Reference harness | XiaomiMiMo/verl `a2ad9f6` + XiaomiMiMo/mimoagent `467f0a1` |

Part of a set of six: {", ".join(f"[{k}](https://huggingface.co/datasets/{repo_id(k)})" for k in INFO)}.

## Layout

```text
tasks/<task_id>/
├── task.toml         # image pinned by digest, one-time setup (healthcheck), judge settings, provenance
├── instruction.md    # the prompt the agent sees, word for word what Xiaomi's harness gives it
├── environment/      # Dockerfile (FROM the same digest) and setup/, a readable copy of the setup
└── tests/            # test.sh and everything grading needs; uploaded only after the agent finishes
registry.json         # Harbor registry entry
data/tasks.jsonl      # one metadata row per task, for filtering without walking the tree
manifest.json         # sha256 of every task directory
jobs/{kind}.yaml      # the reference agent settings
agents/mimo_opencode.py
```

Example: [task.toml]({f"https://huggingface.co/datasets/{rid}/blob/main/tasks/{ex}/task.toml"}) ·
[instruction]({f"https://huggingface.co/datasets/{rid}/blob/main/tasks/{ex}/instruction.md"}) ·
[verifier]({f"https://huggingface.co/datasets/{rid}/blob/main/tasks/{ex}/tests/test.sh"}).

## Run

```bash
hf download {rid} --repo-type dataset --local-dir mimo-{kind}
cd mimo-{kind}
PYTHONPATH=agents HF_TOKEN=hf_... harbor run -y -c jobs/{kind}.yaml
```

The job runs `agents/mimo_opencode.py` (OpenCode 1.18.32) on HF Sandbox with the reference settings: the step
limit above, replies of up to {jobs.OUTPUT.get(kind, 65536):,} tokens, thinking low, no web fetch or search.
Change `model_name` and the provider block for another model. Any Harbor agent can run these tasks; this one
also applies two steps of Xiaomi's harness a task file can't express: the answer-leak blocklist after install,
and the unprivileged `agent` user on sandboxes that run everything as root.

Serving them to a trainer works like any Harbor dataset, for example with OpenEnv:

```bash
openenv harbor serve --dataset {rid} --llm-url http://127.0.0.1:8000/v1 --model <your-model>
```

## How it was converted

With the [mimo_harbor adapter]({ADAPTER_SRC}). It reads the source at one pinned revision and every image through
a digest lock, so a rerun produces byte-identical tasks (`manifest.json`). Setup and grading follow Xiaomi's
harness step by step, as the explorer's runner does, using the same vendored graders:

- Nothing that grades a task (hidden tests, rubrics, verifier scripts) is reachable while the agent works.
- A testbed failure (a patch that won't apply, a system that died, a judge that never answered) writes no reward,
  so Harbor records an error rather than a 0.
- Images are pinned by digest; HF Sandbox and other prebuilt-image backends run them without a build step.

Deviations from the reference, and why, are listed in the adapter's README.

## Validation

- All 7,780 tasks across the six datasets load with Harbor's task loader and pass the adapter's static checks
  (digest-pinned images, setup payload equal to its readable copy, rubrics only under `tests/`).
- With Harbor's no-op agent every dataset scores 0 and every verifier runs to completion: no free rewards.
- One task per dataset was run end to end with GLM-5.3 both in Harbor and in the
  [explorer]({EXPLORER}), which runs Xiaomi's harness; the results match (see the adapter's `parity.md`).
  A full parity study over repeated runs has not been done yet.

## Credits

The environments, their prompts, tests, verifiers, images and graders are the work of the
[Xiaomi MiMo](https://huggingface.co/XiaomiMiMo) team, released as
[{source.DATASET}](https://huggingface.co/datasets/{source.DATASET}) together with their training code
[XiaomiMiMo/verl](https://github.com/XiaomiMiMo/verl) and agent harness
[XiaomiMiMo/mimoagent](https://github.com/XiaomiMiMo/mimoagent). mimoagent builds on
[mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) (Kilian A. Lieret and Carlos E. Jimenez).
The task format and runner are [Harbor](https://github.com/harbor-framework/harbor).
This conversion is by Hugging Face and the FineEnvs team, and is not affiliated with or endorsed by Xiaomi.

## License

Apache-2.0, as the source dataset (`LICENSE`). Files vendored from XiaomiMiMo/mimoagent (`tests/server_arvo.py`
in Cyber tasks, and the ported setup commands) are MIT (`LICENSE-mimoagent.md`). `NOTICE` lists every source,
its license, and the changes this conversion made. Vendored grader files are unmodified.
"""


def stage(src: Path, dst: Path, kind: str) -> None:
    out = dst / kind
    if out.exists():
        shutil.rmtree(out)
    (out / "tasks").mkdir(parents=True)
    dirs = sorted(p for p in (src / kind).iterdir() if p.is_dir())
    rows = []
    for d in dirs:
        shutil.copytree(d, out / "tasks" / d.name)
        rows.append(index_row(d, kind))
    (out / "data").mkdir()
    (out / "data" / "tasks.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    (out / "registry.json").write_text(json.dumps([{
        "name": f"mimo-v2.6-rl-harbor-{kind}", "version": adapter.ADAPTER_VERSION,
        "description": f"MiMo-V2.6-RL {INFO[kind][0]}: {len(rows)} Harbor tasks",
        "tasks": [{"name": r["task_id"], "path": r["task_path"]} for r in rows]}], indent=1) + "\n")
    shutil.copy2(src / kind / "MANIFEST.json", out / "manifest.json")
    (out / "jobs").mkdir()
    import yaml
    head = (f"# MiMo-V2.6-RL {kind} on Harbor with the reference agent settings.\n"
            f"# From the repo root:  PYTHONPATH=agents HF_TOKEN=... harbor run -y -c jobs/{kind}.yaml\n")
    (out / "jobs" / f"{kind}.yaml").write_text(head + yaml.safe_dump(jobs.config(kind, "tasks"), sort_keys=False, width=120))
    (out / "agents").mkdir()
    shutil.copy2(Path(__file__).with_name("agent.py"), out / "agents" / "mimo_opencode.py")
    (out / "README.md").write_text(card(kind, len(rows), rows))
    vendor = Path(__file__).resolve().parents[1] / "app" / "vendor"
    (out / "LICENSE").write_text((vendor / "LICENSE-verl.txt").read_text())          # the Apache-2.0 text
    (out / "LICENSE-mimoagent.md").write_text((vendor / "LICENSE-mimoagent.md").read_text())
    (out / "NOTICE").write_text(NOTICE)
    print(f"{kind}: staged {len(rows)} tasks at {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("stage", type=Path)
    ap.add_argument("--kinds", nargs="*", default=list(INFO))
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()
    for kind in a.kinds:
        stage(a.src, a.stage, kind)
    if a.push:
        from huggingface_hub import HfApi
        api = HfApi()
        for kind in a.kinds:
            rid = repo_id(kind)
            api.create_repo(rid, repo_type="dataset", exist_ok=True, private=False)
            api.upload_large_folder(repo_id=rid, repo_type="dataset", folder_path=str(a.stage / kind))
            print(f"pushed https://huggingface.co/datasets/{rid}")


if __name__ == "__main__":
    main()
