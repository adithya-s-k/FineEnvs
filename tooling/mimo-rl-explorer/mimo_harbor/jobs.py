"""Per-dataset Harbor job configs carrying the reference agent settings (the parts a task.toml cannot hold).

Every value matches what the explorer's runner gives OpenCode (app/runner/opencode.py config_for and run), so a
Harbor run and an explorer run of one task are the same experiment: OpenCode 1.18.32, the reference step limit,
65,536 output tokens per reply (mimoagent's opencode.yaml), thinking low, web fetch and search denied.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .adapter import STEPS

OPENCODE_VERSION = "1.18.32"
MODEL = "zai-org/GLM-5.3"
PROVIDER = "deepinfra"
FLAVOR = {"code": "cpu-basic", "cyber": "cpu-basic", "general": "cpu-basic", "terminal": "cpu-basic",
          "webdev": "cpu-upgrade", "music": "cpu-basic"}   # app/config.py FLAVORS
OUTPUT = {"music": 100000}                                  # verl recipes/design/config/music.yaml response_length
MUSIC_TOOLS = {t: False for t in ("bash", "read", "edit", "glob", "grep", "list", "patch", "todowrite", "todoread",
                                  "webfetch", "websearch", "task")}   # a reply, not a session: only `write` stays


def config(kind: str, dataset_path: str, model: str = MODEL, provider: str | None = PROVIDER, thinking: str = "low",
           n_concurrent: int = 4) -> dict:
    mid = f"{model}:{provider}" if provider else model
    out = OUTPUT.get(kind, 65536)
    build: dict = {"steps": STEPS[kind]}
    if kind == "music":
        build["tools"] = MUSIC_TOOLS
    oc = {
        "autoupdate": False, "share": "disabled",
        "provider": {"hf": {"npm": "@ai-sdk/openai-compatible", "name": "Hugging Face Inference Providers",
                            "options": {"baseURL": "https://router.huggingface.co/v1", "apiKey": "{env:HF_TOKEN}"},
                            "models": {mid: {"name": model, "tool_call": True, "limit": {"context": 1000000, "output": out},
                                             **({"options": {"reasoningEffort": thinking}} if thinking != "default" else {})}}}},
        "agent": {"build": build},
        "snapshot": True, "lsp": True, "formatter": True,
        "permission": {"webfetch": "deny", "websearch": "deny", "bash": "allow", "edit": "allow"},
    }
    return {
        "jobs_dir": f"jobs/{kind}", "n_attempts": 1, "timeout_multiplier": 1.0,
        "orchestrator": {"type": "local", "n_concurrent_trials": n_concurrent},
        "environment": {"type": "hf-sandbox", "delete": True, "kwargs": {"flavor": FLAVOR[kind], "job_timeout": "1h"}},
        "agents": [{
            "import_path": "mimo_opencode:MimoOpenCode", "model_name": f"hf/{mid}",
            "kwargs": {"version": OPENCODE_VERSION, "opencode_config": oc},
            "env": {"HF_TOKEN": "${HF_TOKEN}", "OPENCODE_DISABLE_MODELS_FETCH": "1", "OPENCODE_DISABLE_AUTOUPDATE": "1",
                    "OPENCODE_DISABLE_LSP_DOWNLOAD": "1", "OPENCODE_DISABLE_SHARE": "1"},
        }],
        "datasets": [{"path": dataset_path}],
    }


def write(out: Path, kinds: list[str]) -> None:
    import yaml

    (out / "jobs").mkdir(parents=True, exist_ok=True)
    (out / "agents").mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).with_name("agent.py"), out / "agents" / "mimo_opencode.py")
    for kind in kinds:
        head = (f"# MiMo-V2.6-RL {kind} on Harbor with the reference agent settings (step limit {STEPS[kind]}).\n"
                f"# Run from the dataset root:  PYTHONPATH=agents HF_TOKEN=... harbor run -y -c jobs/{kind}.yaml\n")
        (out / "jobs" / f"{kind}.yaml").write_text(head + yaml.safe_dump(config(kind, kind), sort_keys=False, width=120))
