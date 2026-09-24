# /// script
# dependencies = [
#   "transformers>=4.56.0",
#   "torch>=2.4.0",
#   "torchvision",
#   "accelerate",
#   "peft",   # so a LoRA adapter repo can be evaluated directly
#   "datasets>=3.0.0",
#   "huggingface_hub>=1.22.0",
#   "math-verify",
#   "trackio",
# ]
# ///
"""Score any model on the SmolDataEnvs held-out split. Same environment as training.

    uv run eval_pass1.py                       # 144 eval tasks, Qwen3.5-2B
    MODEL=you/smoldataenvs-grpo-2b uv run eval_pass1.py
    SPLIT=test NUM_TASKS=250 uv run eval_pass1.py      # the benchmark split
    uv run eval_pass1.py --dry-run             # no GPU: prompts + grading only

Run it once before training and once after; the difference is the result. Nothing
here knows about training, so a number from this script means the same thing for
a base model, a checkpoint, or something you downloaded.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from collections import defaultdict

ROLLOUT_URL = os.environ.get(
    "ROLLOUT_URL",
    "https://raw.githubusercontent.com/adithya-s-k/FineEnvs/main/04-smoldataenvs/scripts/rollout.py",
)
if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rollout.py")):
    with open("rollout.py", "wb") as fh:
        fh.write(urllib.request.urlopen(ROLLOUT_URL).read())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

from rollout import SandboxRunner, build_prompt, rollout  # noqa: E402

MODEL = os.environ.get("MODEL", "Qwen/Qwen3.5-2B")
DATASET = os.environ.get("DATASET", "FineEnvs/SmolDataEnvs")
SPLIT = os.environ.get("SPLIT", "eval")
NUM_TASKS = int(os.environ.get("NUM_TASKS", 0))  # 0 = the whole split
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", 1024))
BATCH = int(os.environ.get("BATCH", 8))
OUT = os.environ.get("OUT", "eval_results.json")


def generate(rows) -> list[str]:
    """Greedy, one sample per task: that is what pass@1 means."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype="auto", device_map="auto")
    model.eval()

    out: list[str] = []
    for i in range(0, len(rows), BATCH):
        chunk = rows[i : i + BATCH]
        texts = [
            tok.apply_chat_template(
                build_prompt(r),
                tokenize=False,
                add_generation_prompt=True,
                # non-thinking, to match how the model is trained
                enable_thinking=False,
            )
            for r in chunk
        ]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            ids = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        out += tok.batch_decode(ids[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        print(f"  generated {min(i + BATCH, len(rows))}/{len(rows)}", flush=True)
    return out


def main() -> None:
    from datasets import load_dataset

    rows = list(load_dataset(DATASET, split=SPLIT))
    if NUM_TASKS:
        rows = rows[:NUM_TASKS]
    dry = "--dry-run" in sys.argv
    print(f"{MODEL} on {DATASET}:{SPLIT} — {len(rows)} tasks{' (dry run)' if dry else ''}")

    if dry:
        rows = rows[: int(os.environ.get("DRY_RUN_TASKS", 2))]
        completions = [f"```python\nprint({r['answer']!r})\n```" for r in rows]
    else:
        completions = generate(rows)

    runner = SandboxRunner()
    by_tier: dict[str, list[float]] = defaultdict(list)
    records = []
    try:
        for row, completion in zip(rows, completions):
            res = rollout(runner, row, completion)
            by_tier[row["difficulty_tier"]].append(res["reward"])
            records.append(
                {
                    "task_id": row["task_id"],
                    "tier": row["difficulty_tier"],
                    "gold": row["answer"],
                    "prediction": res["prediction"],
                    "reward": res["reward"],
                }
            )
            print(
                f"  {row['task_id'][:24]:26} {row['difficulty_tier']:7} "
                f"gold={row['answer'][:18]!r:22} got={res['prediction'][:18]!r:22} {res['reward']}"
            )
    finally:
        runner.close()

    scored = [r["reward"] for r in records]
    overall = sum(scored) / max(1, len(scored))
    summary = {
        "model": MODEL,
        "split": SPLIT,
        "n": len(scored),
        "pass@1": round(overall, 4),
        "by_tier": {k: round(sum(v) / len(v), 4) for k, v in sorted(by_tier.items())},
    }
    print("\n" + json.dumps(summary, indent=2))
    with open(OUT, "w") as fh:
        json.dump({"summary": summary, "records": records}, fh, indent=2)
    print(f"wrote {OUT}")

    if os.environ.get("TRACKIO_SPACE"):
        import trackio

        trackio.init(project="smoldataenvs", name=f"eval-{MODEL.split('/')[-1]}-{SPLIT}",
                     space_id=os.environ["TRACKIO_SPACE"])
        trackio.log({"pass@1": overall, **{f"pass@1/{k}": v for k, v in summary["by_tier"].items()}})
        trackio.finish()


if __name__ == "__main__":
    main()
