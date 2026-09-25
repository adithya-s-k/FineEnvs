"""Score candidate models on a frozen evaluation split, without training anything.

This exists to answer "which model is worth fine-tuning", which the GRPO runner cannot
answer cheaply: that one builds a TRL trainer, a reference model and a LoRA adapter before
it can generate a single token. Here a checkpoint is loaded, asked for one answer per task,
and graded by the environment.
"""

import argparse
import json
import os
import time
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

from multilingual_asr.client import connect
from multilingual_asr.models import AsrAction
from multilingual_asr.runtime import local_server
from multilingual_asr.training import AssetCache, task_rows

# Families a model must be prompted for. Whisper and friends transcribe but cannot be asked
# to name a language, so they are scored on what they can actually be asked to do.
TRANSCRIBING = ("transcription", "verbatim_transcription")


def chat_generate(model, processor, audio, prompt, max_tokens):
    """The path every instruction-tuned omni checkpoint in transformers shares."""
    import torch

    inputs = processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    # Qwen2.5-Omni can speak its answer as well as write it, and then returns a pair.
    if isinstance(output, tuple):
        output = output[0]
    return processor.batch_decode(
        output[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )[0].strip()


def whisper_generate(model, processor, audio, prompt, max_tokens):
    """Whisper takes no prompt: it is a reference line, not a candidate to instruct."""
    import torch
    from multilingual_asr.data.tasks import SAMPLING_RATE

    inputs = processor(audio, sampling_rate=SAMPLING_RATE, return_tensors="pt").to(
        model.device, dtype=model.dtype
    )
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_tokens)
    return processor.batch_decode(output, skip_special_tokens=True)[0].strip()


def load(model_id, revision, dtype, device):
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    kwargs = {"revision": revision, "dtype": getattr(torch, dtype)}
    processor = AutoProcessor.from_pretrained(model_id, revision=revision)
    if "whisper" in model_id.lower():
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, **kwargs)
        return model.to(device).eval(), processor, whisper_generate, TRANSCRIBING

    from transformers import AutoModelForCausalLM

    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except (ValueError, KeyError):
        # Omni checkpoints map through the multimodal auto class instead.
        from transformers import AutoModel

        model = AutoModel.from_pretrained(model_id, **kwargs)
    # Only the text branch is generated from; a speech decoder would waste the whole budget.
    if hasattr(model, "disable_talker"):
        model.disable_talker()
    return model.to(device).eval(), processor, chat_generate, None


def score(url, rows, model, processor, generate, cache, max_tokens, progress):
    samples, groups = [], defaultdict(list)
    started = time.monotonic()
    with connect(url) as client:
        for index, row in enumerate(rows, 1):
            observation = client.reset(task_id=row["task_id"]).observation
            prediction = generate(
                model,
                processor,
                cache.audio(observation),
                observation.prompt,
                max_tokens,
            )
            result = client.step(AsrAction(transcript=prediction))
            sample = {
                **row,
                "prediction": prediction,
                "reward": float(result.reward),
                **result.observation.metrics,
            }
            samples.append(sample)
            groups[f"{row['language']}/{row['family']}"].append(sample)
            if progress and index % progress == 0:
                rate = index / (time.monotonic() - started)
                print(
                    f"  {index}/{len(rows)} at {rate:.1f} task/s, "
                    f"running reward {sum(s['reward'] for s in samples) / index:.4f}",
                    flush=True,
                )
    return summarize(samples, groups, time.monotonic() - started)


def summarize(samples, groups, elapsed):
    by_group, by_family, by_language = {}, defaultdict(list), defaultdict(list)
    for key, values in sorted(groups.items()):
        language, family = key.split("/")
        entry = {
            "samples": len(values),
            "reward": sum(v["reward"] for v in values) / len(values),
        }
        for metric in ("wer", "cer", "exact_match"):
            observed = [v[metric] for v in values if metric in v]
            if observed:
                entry[metric] = sum(observed) / len(observed)
        by_group[key] = entry
        by_family[family].extend(values)
        by_language[language].extend(values)

    def mean(values):
        return sum(v["reward"] for v in values) / len(values)

    return {
        # Macro over language/family groups, so no language's share of the split decides
        # the headline. A micro average would let the better-covered languages dominate.
        "macro_reward": sum(e["reward"] for e in by_group.values()) / len(by_group),
        "micro_reward": sum(s["reward"] for s in samples) / len(samples),
        "by_family": {
            k: {"samples": len(v), "reward": mean(v)}
            for k, v in sorted(by_family.items())
        },
        "by_language": {
            k: {"samples": len(v), "reward": mean(v)}
            for k, v in sorted(by_language.items())
        },
        "by_language_task": by_group,
        "elapsed_seconds": round(elapsed, 1),
        "samples": samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus", default="", help="Corpus manifest JSON, or a snapshot"
    )
    parser.add_argument("--env-url", default="")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--eval-split", default="eval_21_validation")
    parser.add_argument(
        "--limit", type=int, default=0, help="Subsample the split evenly"
    )
    parser.add_argument("--families", nargs="+", default=[])
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--progress", type=int, default=50)
    parser.add_argument(
        "--output-dir", default=os.environ.get("OUTPUT_DIR", "artifacts/eval")
    )
    args = parser.parse_args()
    if bool(args.corpus) == bool(args.env_url):
        parser.error("Provide exactly one of --corpus or --env-url")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import model_info

    with ExitStack() as stack:
        url = args.env_url or stack.enter_context(local_server(args.corpus, 4))
        with connect(url) as client:
            manifest = client.manifest()
        served = manifest.get("eval_splits") or {}
        if args.eval_split not in served:
            raise SystemExit(
                f"{args.eval_split!r} is not a frozen evaluation split; this deployment "
                f"serves {sorted(served) or 'none'}"
            )
        # A FLEURS file is one row group of 310 MB to 1.5 GB, so a cold read is
        # expensive and a warm one is free. Grouping by language keeps each shard hot
        # while it is being used instead of paying for it once per task. Every task is
        # scored on its own, so order changes throughput and nothing else.
        rows = sorted(task_rows(url, args.eval_split), key=lambda r: r["language"])
        cache = AssetCache(url)
        print(
            f"{args.eval_split}: {len(rows)} tasks, evalset "
            f"{served[args.eval_split]['evalset_id'][:12]}",
            flush=True,
        )

        leaderboard = []
        for model_id in args.models:
            revision = model_info(model_id).sha
            print(f"\n=== {model_id} @ {revision[:12]} ===", flush=True)
            model, processor, generate, only = load(
                model_id, revision, args.dtype, args.device
            )
            families = args.families or (list(only) if only else [])
            chosen = [r for r in rows if not families or r["family"] in families]
            if args.limit:
                chosen = chosen[:: max(1, len(chosen) // args.limit)][: args.limit]
            result = score(
                url,
                chosen,
                model,
                processor,
                generate,
                cache,
                args.max_new_tokens,
                args.progress,
            )
            body = {
                "model": model_id,
                "model_revision": revision,
                "eval_split": args.eval_split,
                "evalset_id": served[args.eval_split]["evalset_id"],
                "snapshot_id": manifest["snapshot_id"],
                "grading": manifest["grading"],
                "families": families or "all",
                "tasks": len(chosen),
                **result,
            }
            name = model_id.replace("/", "__")
            (output / f"{name}.json").write_text(
                json.dumps(body, ensure_ascii=False, indent=2) + "\n"
            )
            leaderboard.append(
                {k: body[k] for k in ("model", "tasks", "macro_reward", "micro_reward")}
                | {
                    "by_family": body["by_family"],
                    "elapsed_seconds": body["elapsed_seconds"],
                }
            )
            print(
                f"  macro {body['macro_reward']:.4f}  micro {body['micro_reward']:.4f}  "
                f"in {body['elapsed_seconds']}s",
                flush=True,
            )
            del model, processor
            if args.device == "cuda":
                import torch

                torch.cuda.empty_cache()

        leaderboard.sort(key=lambda e: e["macro_reward"], reverse=True)
        (output / "leaderboard.json").write_text(
            json.dumps(
                {
                    "eval_split": args.eval_split,
                    "evalset_id": served[args.eval_split]["evalset_id"],
                    "snapshot_id": manifest["snapshot_id"],
                    "results": leaderboard,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
        print("\n" + json.dumps(leaderboard, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
