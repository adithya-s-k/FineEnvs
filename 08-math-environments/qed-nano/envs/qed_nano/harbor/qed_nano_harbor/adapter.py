"""Write QED-Nano's problems as Harbor tasks.

    python -m qed_nano_harbor.adapter --out out/qed-nano            # every set
    python -m qed_nano_harbor.adapter --out out/qed-nano --sets test,imo --limit 20

Each task:
    instruction.md          the problem and what to submit
    task.toml               limits, network policy, verifier settings, metadata (difficulty, source)
    environment/Dockerfile  a small Python image; the agent writes /workspace/proof.md (declared as an artifact)
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
DATASET = "qed-nano-envs"
VALIDATED_SCORE = 6  # a task is "validated" when its reference proof scores at least this (0-7) in the oracle sweep
VERSION = "1.0"
AGENT_TIMEOUT = 3600
VERIFIER_TIMEOUT = 3600  # room for judge retries and JUDGE_SAMPLES > 1
MIN_ORACLE_CHARS = 400
# The training reward is QED-Nano's (gpt-oss-20b). The benchmarks get a stronger default judge, as QED-Nano graded its
# benchmarks with a stronger model: gpt-oss-20b gave an IMO-ProofBench official solution 1/7 where gpt-oss-120b gave 7.
JUDGES = {"train": "openai/gpt-oss-20b:deepinfra", "test": "openai/gpt-oss-20b:deepinfra",
          "imo": "openai/gpt-oss-120b:novita", "proofbench": "openai/gpt-oss-120b:novita"}
SET_NAMES = {"train": "train", "test": "test", "imo": "imo-proofbench", "proofbench": "proofbench"}


def toml_str(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def task_toml(r: dict) -> str:
    meta = {k: r[k] for k in ("set", "source", "category", "competition", "level", "split") if r.get(k)}
    for k in ("reward_mean", "reward_std", "num_rewards", "reference_grade", "oracle_score", "oracle_judge"):
        if r.get(k) is not None:
            meta[k] = r[k]
    meta_lines = "\n".join(f"{k} = {json.dumps(v, ensure_ascii=False)}" for k, v in meta.items())
    return f"""schema_version = "1.4"
# The one file the verifier reads. Harbor copies it from the agent's sandbox to the same path in the verifier's.
artifacts = ["/workspace/proof.md"]

[task]
name = "{DATASET}/{r['id']}"
version = "1.0.0"
description = {toml_str("Write a rigorous natural-language proof of an olympiad problem; graded 0-7 by an LLM judge against the problem's marking scheme.")}
keywords = ["olympiad", "proof", "rubric", "llm-judge", "qed-nano"]

[metadata]
source_dataset = "https://huggingface.co/lm-provers"
judge = "{JUDGES[r['set']]}, QED-Nano's {'IMO-ProofBench' if r['prompt_style'] == 'imo' else 'training'} prompt"
{meta_lines}

[agent]
timeout_sec = {AGENT_TIMEOUT}
# Most problems are public forum threads. Harbor >= 0.25 still lets installed agents reach their model's host and
# turns off their built-in web search; a host-side agent (like qed-proof) calls its model from outside the sandbox.
network_mode = "no-network"

[environment]
build_timeout_sec = 900
cpus = 1
memory_mb = 2048
storage_mb = 4096
workdir = "/workspace"

[verifier]
timeout_sec = {VERIFIER_TIMEOUT}
environment_mode = "separate"
# The judge's key and settings come from the environment `harbor run` is started in. JUDGE_API_KEY is required.
env = {{ JUDGE_API_KEY = "${{JUDGE_API_KEY}}", JUDGE_PROVIDER = "${{JUDGE_PROVIDER:-hf}}", JUDGE_MODEL = "${{JUDGE_MODEL:-{JUDGES[r['set']]}}}", JUDGE_BASE_URL = "${{JUDGE_BASE_URL:-}}", JUDGE_REASONING_EFFORT = "${{JUDGE_REASONING_EFFORT:-medium}}", JUDGE_MAX_TOKENS = "${{JUDGE_MAX_TOKENS:-32768}}", JUDGE_EXTRA_BODY = "${{JUDGE_EXTRA_BODY:-}}", JUDGE_STREAM = "${{JUDGE_STREAM:-}}", JUDGE_SAMPLES = "${{JUDGE_SAMPLES:-1}}" }}
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
    if proof and len(proof.strip()) < MIN_ORACLE_CHARS:
        proof = None  # a few benchmark "solutions" are only the final answer: no use as an oracle proof
    if proof:
        (d / "solution").mkdir()
        (d / "solution" / "proof.md").write_text(proof.strip() + "\n")
        shutil.copy(TEMPLATES / "solution" / "solve.sh", d / "solution" / "solve.sh")
    for sh in d.rglob("*.sh"):
        sh.chmod(0o755)
    oracle_kind = None if not proof else ("deepseek-math-v2" if has_oracle else "reference-solution")
    return {"task": r["id"], "set": r["set"], "oracle": oracle_kind,
            **{k: r.get(k) for k in ("source", "category", "competition", "reward_mean", "reward_std", "level",
                                     "oracle_score", "oracle_judge")}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path(".cache/qed-nano"))
    ap.add_argument("--sets", default="train,test,imo,proofbench")
    ap.add_argument("--limit", type=int, default=0, help="first N tasks of each set (for smoke tests)")
    ap.add_argument("--oracle-results", type=Path,
                    help="JSONL from oracle_sweep: stamps oracle_score/oracle_judge and adds *-validated registries")
    args = ap.parse_args()
    oracle = {}
    if args.oracle_results:
        for line in args.oracle_results.read_text().splitlines():
            row = json.loads(line)
            if row.get("score") is not None:
                oracle[row["task"]] = {"oracle_score": row["score"], "oracle_judge": f"{row['judge']} ({row['effort']})"}

    sets = data.load_all(data.fetch(args.cache))
    test_ids = {r["id"] for r in sets["test"]}
    sets["train"] = [r for r in sets["train"] if r["id"] not in test_ids]

    manifest, registry = [], []
    for name in args.sets.split(","):
        records = [{**r, **oracle.get(r["id"], {})} for r in sets[name][: args.limit or None]]
        written = [write_task(r, args.out / SET_NAMES[name]) for r in records]
        manifest += written
        label = SET_NAMES[name]
        entry = lambda suffix, rows, what: {  # noqa: E731
            "name": f"{DATASET}-{label}{suffix}", "version": VERSION,
            "description": f"{len(rows)} olympiad proof tasks ({label}{what}) from QED-Nano's release, graded 0-7 by "
                           "an LLM judge against each problem's rubric; reward = score / 7.",
            "metrics": [{"type": "mean"}],
            "tasks": [{"name": w["task"], "path": f"{label}/tasks/{w['task']}"} for w in rows]}
        registry.append(entry("", written, ""))
        validated = [w for w in written if (w.get("oracle_score") or 0) >= VALIDATED_SCORE]
        if oracle:
            registry.append(entry("-validated", validated, f", reference proof scored >= {VALIDATED_SCORE}/7"))
        print(f"{name}: {len(written)} tasks, {sum(1 for w in written if w['oracle'])} with an oracle proof, "
              f"{len(validated)} validated")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    (args.out / "registry.json").write_text(json.dumps(registry, indent=2))
    write_parquet(args.out / "manifest.parquet", manifest)


def write_parquet(path: Path, rows: list[dict]) -> None:
    """manifest.parquet: one row per task, for filtering without walking the task folders."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    columns = ["task", "set", "source", "category", "competition", "reward_mean", "reward_std", "oracle",
               "oracle_score", "oracle_judge"]
    pq.write_table(pa.Table.from_pylist([{c: r.get(c) for c in columns} for r in rows]), path)


if __name__ == "__main__":
    main()
