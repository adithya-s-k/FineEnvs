"""Preserve eval evidence across Job and Space restarts."""
import hashlib
import json
import os
from pathlib import Path

from common import MODEL, REVISION, write_json


def persist_trial(args, result):
    """Save harness-version evidence alongside each graded capture, before its trace."""
    import httpx
    trial = getattr(result, "trial_name", "")
    if not trial or Path(trial).name != trial:
        return
    output = Path(args.capture_dir).parent
    try:
        response = httpx.get(args.server.rstrip("/") + "/trial/" + trial + "/result", timeout=30)
        response.raise_for_status()
        write_json(output / "trials" / trial / "result.json", response.json())
    except Exception as exc:
        # Preserve the graded result even if metadata retrieval fails. The final
        # audit retries retrieval and refuses to publish an unverified baseline.
        write_json(output / "trial-evidence-errors" / (trial + ".json"),
                   {"trial": trial, "error_type": type(exc).__name__})


def restore_whitebox(output, prefix):
    from huggingface_hub import HfApi
    origin = output / "resume-origin"
    origin.mkdir()
    HfApi().sync_bucket(prefix, str(origin), include=["attempts.jsonl", "eval_config.json", "captures/**"], quiet=True)
    config = json.loads((origin / "eval_config.json").read_text())
    expected = {"model": MODEL, "revision": REVISION, "pass_k": 1,
                "temperature": 0.8, "top_p": 1.0, "max_output_tokens_per_call": 4096,
                "max_episode_completion_tokens": 16384, "max_model_calls": 17,
                "toolsets": ["bash", "seta"]}
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError("Saved baseline protocol differs from the requested evaluation")
    rows = []
    graded = set()
    for line in (origin / "attempts.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("capture_file"):
            relative = Path("captures") / Path(row["capture_file"]).name
            saved = origin / relative
            if not saved.is_file():
                raise ValueError("Restored attempt is missing its captured tokens")
            target = output / relative
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(saved.read_bytes())
            row["capture_file"] = str(target)
        if row.get("reward") in (0, 1) and row.get("tito_pass"):
            graded.add((row["harness"], row["index"]))
        rows.append(row)
    (output / "attempts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    write_json(output / "eval_config.json", config)
    write_json(output / "eval_resume.json", {"origin": prefix, "restored_graded": len(graded),
        "original_attempts_sha256": hashlib.sha256((origin / "attempts.jsonl").read_bytes()).hexdigest(),
        "selection": "First graded result, including zeros; only capture paths rebound"})
    from score_comparison import summarize
    summarize("whitebox", output)
