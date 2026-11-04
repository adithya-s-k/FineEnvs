"""Write QED-Nano's problems as Harbor tasks.

    python -m qed_nano_harbor.adapter --out out/qed-nano            # every set
    python -m qed_nano_harbor.adapter --out out/qed-nano --sets test,imo --limit 20

Each task:
    instruction.md          the problem and what to submit
    task.toml               limits, network policy, verifier settings, metadata (difficulty, source)
    environment/Dockerfile  a small Python image; the agent writes /workspace/proof.md
    tests/                  task.json (problem, marking scheme, reference solution), judge_prompt.md, grade.py
    solution/solve.sh       the oracle: a DeepSeek-Math-V2 proof Gemini 3 Pro graded 7 (FineProofs-SFT), when one exists

FineProofs-RL-test is a subset of FineProofs-RL, so the train set leaves those 128 problems out.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from string import Template

from . import data

TEMPLATES = Path(__file__).parent / "templates"
PROMPTS = Path(__file__).parent / "prompts"
DATASET = "qed-nano"
VERSION = "1.0"
AGENT_TIMEOUT = 3600
VERIFIER_TIMEOUT = 1800
SET_NAMES = {"train": "train", "test": "test", "imo": "imo-proofbench", "proofbench": "proofbench"}


def toml_str(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def task_toml(r: dict) -> str:
    meta = {k: r[k] for k in ("set", "source", "category", "competition", "level", "split") if r.get(k)}
    for k in ("reward_mean", "reward_std", "num_rewards", "reference_grade"):
        if r.get(k) is not None:
            meta[k] = r[k]
    meta_lines = "\n".join(f"{k} = {json.dumps(v, ensure_ascii=False)}" for k, v in meta.items())
    return f"""schema_version = "1.4"

[task]
name = "{DATASET}/{r['id']}"
version = "1.0.0"
description = {toml_str("Write a rigorous natural-language proof of an olympiad problem; graded 0-7 by an LLM judge against the problem's marking scheme.")}
keywords = ["olympiad", "proof", "rubric", "llm-judge", "qed-nano"]

[metadata]
source_dataset = "https://huggingface.co/lm-provers"
judge = "gpt-oss-20b, QED-Nano's {'IMO-ProofBench' if r['prompt_style'] == 'imo' else 'training'} prompt"
{meta_lines}

[agent]
timeout_sec = {AGENT_TIMEOUT}

[environment]
build_timeout_sec = 900
cpus = 1
memory_mb = 2048
storage_mb = 4096
workdir = "/workspace"

[verifier]
timeout_sec = {VERIFIER_TIMEOUT}
environment_mode = "separate"
env = {{ HF_TOKEN = "${{HF_TOKEN}}", JUDGE_MODEL = "${{JUDGE_MODEL:-openai/gpt-oss-20b:deepinfra}}", JUDGE_SAMPLES = "${{JUDGE_SAMPLES:-1}}" }}

[[verifier.collect]]
command = "python3 /opt/qed-nano/collect_proof.py"
timeout_sec = 60
"""


def write_task(r: dict, out: Path) -> dict:
    d = out / "tasks" / r["id"]
    if d.exists():
        shutil.rmtree(d)
    (d / "tests").mkdir(parents=True)
    (d / "environment").mkdir()
    shutil.copytree(TEMPLATES / "environment", d / "environment", dirs_exist_ok=True)
    instruction = Template((TEMPLATES / "instruction.md").read_text()).substitute(problem=r["problem"])
    (d / "instruction.md").write_text(instruction)
    (d / "task.toml").write_text(task_toml(r))
    task = {k: r[k] for k in ("id", "set", "problem", "marking_scheme", "prompt_style")}
    if r.get("reference_solution"):
        task["reference_solution"] = r["reference_solution"]
    (d / "tests" / "task.json").write_text(json.dumps(task, indent=2, ensure_ascii=False))
    shutil.copy(PROMPTS / f"{r['prompt_style']}.md", d / "tests" / "judge_prompt.md")
    for name in ("grade.py", "test.sh"):
        shutil.copy(TEMPLATES / "tests" / name, d / "tests" / name)
    has_oracle = bool(r.get("reference_proof")) and r.get("reference_grade") == 7
    proof = r.get("reference_proof") if has_oracle else r.get("reference_solution")
    if proof:
        (d / "solution").mkdir()
        (d / "solution" / "proof.md").write_text(proof.strip() + "\n")
        shutil.copy(TEMPLATES / "solution" / "solve.sh", d / "solution" / "solve.sh")
    for sh in d.rglob("*.sh"):
        sh.chmod(0o755)
    return {"task": r["id"], "set": r["set"], "oracle": "deepseek-math-v2" if has_oracle else
            ("reference-solution" if proof else None),
            **{k: r.get(k) for k in ("source", "category", "reward_mean", "reward_std", "level")}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path(".cache/qed-nano"))
    ap.add_argument("--sets", default="train,test,imo,proofbench")
    ap.add_argument("--limit", type=int, default=0, help="first N tasks of each set (for smoke tests)")
    args = ap.parse_args()

    sets = data.load_all(data.fetch(args.cache))
    test_ids = {r["id"] for r in sets["test"]}
    sets["train"] = [r for r in sets["train"] if r["id"] not in test_ids]

    manifest, registry = [], []
    for name in args.sets.split(","):
        records = sets[name][: args.limit or None]
        written = [write_task(r, args.out / SET_NAMES[name]) for r in records]
        manifest += written
        registry.append({"name": f"{DATASET}-{SET_NAMES[name]}", "version": VERSION,
                         "description": f"{len(written)} olympiad proof tasks ({SET_NAMES[name]}) from QED-Nano's release, rubric-graded by an LLM judge.",
                         "tasks": [{"name": w["task"], "path": f"{SET_NAMES[name]}/tasks/{w['task']}"} for w in written]})
        print(f"{name}: {len(written)} tasks, {sum(1 for w in written if w['oracle'])} with an oracle proof")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    (args.out / "registry.json").write_text(json.dumps(registry, indent=2))


if __name__ == "__main__":
    main()
