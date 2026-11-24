"""Publish the trained weights to the Hub: merged models, and every LoRA adapter in one repo.

Runs as an HF Job (`train/jobs/hf_job.py publish`), because both halves move tens of gigabytes
between the runs bucket, the Hub and the job's disk, and all three sit in the same datacentre.

    publish_hub.py model    --checkpoint think-mixed-r4/checkpoint-100 --repo FineEnvs/<name>
    publish_hub.py adapters --repo FineEnvs/<name>

`model` merges one adapter into the base checkpoint (train/jobs/merge_lora.py, which merges in
the safetensors files themselves) and uploads the result. `adapters` uploads every adapter named
in train/checkpoints.json, one folder per checkpoint, with no merge.

The scores in the cards come from train/checkpoints.json, which was recomputed from every
evaluation shard on the runs bucket rather than copied from the running tables.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "train" / "checkpoints.json"
BASE_MODEL = "Qwen/Qwen3.8-27B"
RUNS_BUCKET = "AdithyaSK/retroenv-rl-runs"
ENV_SPACE = "FineEnvs/retroenv"
DATASET = "LiteFold/RetroEnv-Curated"
# Files that make an adapter loadable on its own; the optimizer and RNG state stay on the bucket.
ADAPTER_FILES = ("adapter_model.safetensors", "adapter_config.json", "chat_template.jinja")


def manifest():
    data = json.loads(MANIFEST.read_text())
    for row in data["checkpoints"]:
        row["id"] = f"{row['run']}/{row['checkpoint']}"
    return data


def find(data, checkpoint):
    for row in data["checkpoints"]:
        if row["id"] == checkpoint:
            return row
    raise SystemExit(f"{checkpoint} is not in {MANIFEST.name}; ids look like think-mixed-r4/checkpoint-100")


def best(data, thinking):
    rows = [r for r in data["checkpoints"] if r["complete"] and r["thinking"] is thinking]
    return max(rows, key=lambda r: r["pass_at_1"])


def table(rows, highlight=None):
    out = ["| Run | Step | Thinking | pass@1 | pass@4 | Training pool |", "|---|---:|---|---:|---:|---|"]
    for r in rows:
        mark = "**" if r["id"] == highlight else ""
        out.append(
            f"| `{r['run']}` | {mark}{r['step']}{mark} | {'yes' if r['thinking'] else 'no'} "
            f"| {mark}{r['pass_at_1']:.3f}{mark} | {r['pass_at_4']:.3f} | `{r['pool']}` |"
        )
    return "\n".join(out)


def model_card(data, row, repo):
    mode = "thinking" if row["thinking"] else "non-thinking"
    ev, tiers = data["eval"], row["tiers_first_attempt"]
    peers = [r for r in data["checkpoints"] if r["complete"] and r["thinking"] is row["thinking"]]
    return f"""---
base_model: {BASE_MODEL}
library_name: transformers
license: apache-2.0
pipeline_tag: text-generation
tags: [retrosynthesis, chemistry, reinforcement-learning, grpo, openenv, rl-environment]
datasets: [{DATASET}]
---

# RetroEnv · Qwen3.8-27B · GRPO ({mode})

The base model with the LoRA adapter from **step {row["step"]}** of `{row["run"]}` merged into its
weights, so it serves like any ordinary checkpoint. Trained with GRPO against
[RetroEnv]({f"https://huggingface.co/spaces/{ENV_SPACE}"}), a retrosynthesis environment: the model is
given a target molecule, searches for stock and precedent through tools, and submits one or more
synthesis trees, which a deterministic verifier grades against a frozen reaction library.

## What it scores

On `{ev["set"]}` in the {ev["harness"]} harness, {ev["attempts"]} attempts per task at temperature
{ev["temperature"]}:

| | pass@1 | pass@4 |
|---|---:|---:|
| `{BASE_MODEL}` ({mode}) | {data["baseline"][mode]["pass_at_1"]:.3f} | {data["baseline"][mode]["pass_at_4"]:.3f} |
| **this checkpoint** | **{row["pass_at_1"]:.3f}** | **{row["pass_at_4"]:.3f}** |

First-attempt pass rate by difficulty tier: easy {tiers.get("easy", float("nan")):.2f}, medium
{tiers.get("medium", float("nan")):.2f}, hard {tiers.get("hard", float("nan")):.2f}. Those three average to the
first-attempt rate ({row["pass_at_1_first_attempt"]:.3f}), not to the pass@1 above, which is the
unbiased estimate over all four attempts.

Every {mode} checkpoint that was evaluated completely:

{table(peers, highlight=row["id"])}

## How it was trained

LoRA r16, alpha 32, on the language-model layers only (attention q/k/v/o, linear attention, and the
MLP including the shared expert), merged here into the base weights. TRL AsyncGRPO, 32 rollouts per
task, 18 gradient-accumulation steps, constant learning rate, epsilon 0.2 / 0.28, and **no reference
KL penalty** — `AsyncGRPOConfig` has no `beta`, so nothing pulls the policy back toward the base
model. Rollouts ran against the hosted environment; the reward is the v8 shaped reward computed by
the server, which pays a solve 0.70–1.0 and a failure 0–0.20 by how much of the tree reaches
purchasable material.

Full training code, the launcher and the evaluation harnesses are in
[FineEnvs](https://github.com/adithya-s-k/FineEnvs) under `09-retroenv/`. Every run's checkpoints,
episodes and logs are public in the [`{RUNS_BUCKET}`](https://huggingface.co/buckets/{RUNS_BUCKET}) bucket.

## Using it

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{repo}", dtype="bfloat16", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("{repo}")
```

It expects the environment's tool schema and system prompt; on its own, outside that loop, it is
just the base model with a nudge toward route trees. Run it the way it was trained with
`09-retroenv/eval/run_eval.py --harness training`, or the way the public board is scored with the
default board harness.

{"It was trained with thinking on at low reasoning effort; serve it with the thinking template or it will behave unlike the numbers above." if row["thinking"] else "It was trained with thinking off. The thinking checkpoint scores considerably higher — see the collection."}

## What the numbers do not say

The verifier checks that each step matches a real reaction on record and that every leaf is
genuinely purchasable. It says nothing about yield, selectivity or safety, so a route it accepts is
not a route a chemist should run unexamined.
"""


def adapters_card(data, repo):
    comp = [r for r in data["checkpoints"] if r["complete"]]
    think = [r for r in comp if r["thinking"]]
    plain = [r for r in comp if not r["thinking"]]
    return f"""---
base_model: {BASE_MODEL}
library_name: peft
license: apache-2.0
tags: [retrosynthesis, chemistry, reinforcement-learning, grpo, lora, openenv]
datasets: [{DATASET}]
---

# RetroEnv · every GRPO LoRA checkpoint

{len(data["checkpoints"])} adapters from the {len({r["run"] for r in data["checkpoints"]})} GRPO runs
trained against [RetroEnv](https://huggingface.co/spaces/{ENV_SPACE}), one folder each, so the whole
training curve can be replayed rather than just its peak. The two best are also published merged, as
ordinary models — see the collection.

Folders are named `<run>/step-<absolute step>`. Steps are absolute: each run after the first resumes
from a checkpoint of the one before, so step numbers continue across runs rather than restarting.

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("{BASE_MODEL}", dtype="bfloat16", device_map="auto")
model = PeftModel.from_pretrained(base, "{repo}", subfolder="think-mixed-r4/step-525")
```

## Thinking runs

{table(think)}

## Non-thinking runs

{table(plain)}

{(lambda inc: "## Not comparable" + chr(10) + chr(10) + "These two lost evaluation shards to job timeouts, so their folders are here but their scores are not on the curve: " + ", ".join(f"`{r['id']}` (step {r['step']}, {r['tasks']} of 150 tasks)" for r in inc) + "." if inc else "")([r for r in data["checkpoints"] if not r["complete"]])}

Scored on `{data["eval"]["set"]}` in the {data["eval"]["harness"]} harness at {data["eval"]["attempts"]}
attempts, temperature {data["eval"]["temperature"]}. `checkpoints.json` in this repo carries every
number, including per-tier rates and how many episodes each evaluation graded.

{data["note"]}
"""


def upload(api, repo, folder, message, allow=None):
    api.create_repo(repo, repo_type="model", exist_ok=True)
    print(f"uploading {folder} -> {repo}", flush=True)
    api.upload_large_folder(
        repo_id=repo, repo_type="model", folder_path=str(folder), allow_patterns=allow, print_report=True
    )
    print(f"done: https://huggingface.co/{repo} ({message})", flush=True)


def publish_model(args, data):
    row = find(data, args.checkpoint) if args.checkpoint else best(data, thinking=not args.non_thinking)
    print(f"publishing {row['id']} (step {row['step']}, pass@1 {row['pass_at_1']})", flush=True)
    adapter = Path(args.outputs) / row["run"] / row["checkpoint"]
    if not (adapter / "adapter_model.safetensors").is_file():
        raise SystemExit(f"no adapter at {adapter}")
    merged = Path(args.workdir) / "merged" / row["run"] / row["checkpoint"]
    if merged.is_dir() and any(merged.glob("*.safetensors")):
        print(f"reusing the merge already at {merged}", flush=True)
    else:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "train" / "jobs" / "merge_lora.py"),
                "--model",
                BASE_MODEL,
                "--revision",
                data["base_model_revision"],
                "--adapter",
                str(adapter),
                "--output",
                str(merged),
            ],
            check=True,
        )
    (merged / "README.md").write_text(model_card(data, row, args.repo))
    (merged / "retroenv-checkpoint.json").write_text(json.dumps(row, indent=2))

    from huggingface_hub import HfApi

    upload(HfApi(), args.repo, merged, f"{row['id']} merged into {BASE_MODEL}")


def publish_adapters(args, data):
    staging = Path(args.workdir) / "adapters"
    staging.mkdir(parents=True, exist_ok=True)
    kept = []
    for row in data["checkpoints"]:
        source = Path(args.outputs) / row["run"] / row["checkpoint"]
        target = staging / row["run"] / f"step-{row['step']}"
        target.mkdir(parents=True, exist_ok=True)
        missing = [name for name in ADAPTER_FILES if not (source / name).is_file()]
        if "adapter_model.safetensors" in missing:
            print(f"skipping {row['id']}: no adapter file", flush=True)
            continue
        for name in ADAPTER_FILES:
            if (source / name).is_file():
                target.joinpath(name).write_bytes((source / name).read_bytes())
        target.joinpath("retroenv-checkpoint.json").write_text(json.dumps(row, indent=2))
        kept.append(row)
        print(f"staged {row['id']} -> {row['run']}/step-{row['step']}", flush=True)
    if not kept:
        raise SystemExit("no adapters were staged")
    (staging / "README.md").write_text(adapters_card(data, args.repo))
    (staging / "checkpoints.json").write_text(MANIFEST.read_text())

    from huggingface_hub import HfApi

    upload(HfApi(), args.repo, staging, f"{len(kept)} adapters")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("model", "adapters"))
    parser.add_argument("--repo", required=True, help="Target model repo, e.g. FineEnvs/retroenv-...")
    parser.add_argument("--checkpoint", help="model: <run>/checkpoint-N; omit for the best complete one")
    parser.add_argument("--non-thinking", action="store_true", help="model: pick the best non-thinking checkpoint")
    parser.add_argument("--outputs", default="/outputs", help="The runs bucket, mounted")
    parser.add_argument("--workdir", default=os.environ.get("RETROENV_PUBLISH_WORKDIR", "/workspace/publish"))
    args = parser.parse_args(argv)
    data = manifest()
    Path(args.workdir).mkdir(parents=True, exist_ok=True)
    (publish_model if args.mode == "model" else publish_adapters)(args, data)


if __name__ == "__main__":
    main()
