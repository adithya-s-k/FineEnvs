"""Score candidate models on a frozen evaluation split, with vLLM doing the generating.

`eval_asr.py` generates one answer at a time through transformers, which is fine for eight
tasks and far too slow for five hundred. vLLM batches continuously, so the GPU stays busy
while the environment is fetching the next task's audio.

vLLM and OpenEnv cannot share an environment — they disagree on fastmcp — so vLLM runs as
its own server in an isolated uv environment and is driven over its OpenAI-compatible API.
Audio travels as base64 WAV, which is what the environment already serves, so nothing is
decoded or resampled on the way.
"""

import argparse
import base64
import json
import os
import socket
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from pathlib import Path

import requests
from multilingual_asr.client import AsrClient, connect
from multilingual_asr.models import AsrAction
from multilingual_asr.runtime import local_server
from multilingual_asr.training import task_rows

VLLM_SPEC = "vllm[audio]==0.30.0"


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextmanager
def vllm_server(
    model, revision, *, max_model_len, gpu_fraction, extra_args, boot_seconds
):
    """Serve one checkpoint, in an environment of its own."""
    port = free_port()
    command = [
        "uv",
        "run",
        "--with",
        VLLM_SPEC,
        "--python",
        "3.12",
        "vllm",
        "serve",
        model,
        "--port",
        str(port),
        "--revision",
        revision,
        # One clip per prompt is all this task ever sends; a larger budget would
        # reserve multimodal cache for nothing. vLLM 0.30 takes JSON here, not the
        # key=value form older recipes show.
        "--limit-mm-per-prompt",
        json.dumps({"audio": 1}),
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        str(gpu_fraction),
        *extra_args,
    ]
    print("starting vLLM: " + " ".join(command), flush=True)
    # FlashInfer's sampler is JIT-compiled and wants a CUDA toolkit this image does not
    # carry, so the engine dies on the first token. Evaluation samples greedily and has
    # no use for it; vLLM's own sampler is what gets used either way.
    process = subprocess.Popen(
        command,
        stdout=sys.stdout,
        stderr=sys.stderr,
        env={**os.environ, "VLLM_USE_FLASHINFER_SAMPLER": "0"},
    )
    url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + boot_seconds
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"vLLM exited with code {process.returncode}")
            try:
                if requests.get(f"{url}/health", timeout=5).ok:
                    break
            except requests.RequestException:
                pass
            time.sleep(5)
        else:
            raise RuntimeError(f"vLLM did not become healthy within {boot_seconds}s")
        print(f"vLLM ready at {url}", flush=True)
        yield url
    finally:
        process.terminate()
        try:
            process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            process.kill()


def answer(vllm_url, model, prompt, wav, max_tokens, timeout):
    """One chat completion carrying the clip as base64 WAV, as OpenAI defines it."""
    response = requests.post(
        f"{vllm_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": base64.b64encode(wav).decode(),
                                "format": "wav",
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_completion_tokens": max_tokens,
            # Greedy, so a re-run of the same checkpoint gives the same score.
            "temperature": 0.0,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def open_client(env_url, timeout):
    """A session of our own, with a timeout that survives a cold shard read."""
    return AsrClient(
        base_url=env_url, connect_timeout_s=60, message_timeout_s=timeout
    ).sync()


def prefetch(env_url, rows, timeout, progress):
    """Walk the split once, in shard order, collecting each task's prompt and audio.

    This is deliberately sequential. A FLEURS file is a single row group of 310 MB to
    1.5 GB and the cache holds only a few, so concurrent workers landing on different
    languages each stall on their own cold read — which is what made reset time out.
    One reader in language order pays for each shard exactly once.
    """
    started = time.monotonic()
    prepared = []
    with open_client(env_url, timeout) as client:
        for index, row in enumerate(rows, 1):
            observation = client.reset(task_id=row["task_id"]).observation
            wav = requests.get(
                f"{env_url}/assets/{observation.asset_sha256}",
                params={"task_id": observation.task_id},
                timeout=timeout,
            )
            wav.raise_for_status()
            prepared.append((row, observation.prompt, wav.content))
            if progress and index % progress == 0:
                print(
                    f"  audio {index}/{len(rows)} "
                    f"({index / (time.monotonic() - started):.1f} task/s)",
                    flush=True,
                )
    print(
        f"  audio ready: {len(prepared)} clips, "
        f"{sum(len(w) for _, _, w in prepared) / 1e6:.0f} MB, "
        f"in {time.monotonic() - started:.0f}s",
        flush=True,
    )
    return prepared


def predict(vllm_url, model, prepared, max_tokens, timeout, workers, progress):
    """Ask vLLM for every answer at once. Nothing here touches the environment."""
    started = time.monotonic()
    done = [0]
    lock = threading.Lock()

    def one(item):
        _, prompt, wav = item
        text = answer(vllm_url, model, prompt, wav, max_tokens, timeout)
        with lock:
            done[0] += 1
            if progress and done[0] % progress == 0:
                print(
                    f"  generated {done[0]}/{len(prepared)} "
                    f"({done[0] / (time.monotonic() - started):.1f} task/s)",
                    flush=True,
                )
        return text

    with ThreadPoolExecutor(max_workers=workers) as pool:
        predictions = list(pool.map(one, prepared))
    print(
        f"  generation done in {time.monotonic() - started:.0f}s "
        f"({len(prepared) / (time.monotonic() - started):.1f} task/s)",
        flush=True,
    )
    return predictions


def grade(env_url, prepared, predictions, timeout, workers):
    """Let the environment score each answer. The audio cache is warm by now."""
    started = time.monotonic()
    samples, groups = [], defaultdict(list)
    lock = threading.Lock()
    local = threading.local()
    # Sessions are a bounded server resource, so they are tracked and released here.
    # Leaving one model's workers holding theirs starved the next model's grading, which
    # failed with the server closing the connection rather than anything about the answer.
    opened = []

    def one(pair):
        (row, _, _), prediction = pair
        client = getattr(local, "client", None)
        if client is None:
            client = local.client = open_client(env_url, timeout)
            with lock:
                opened.append(client)
        client.reset(task_id=row["task_id"])
        result = client.step(AsrAction(transcript=prediction))
        sample = {
            **row,
            "prediction": prediction,
            "reward": float(result.reward),
            **result.observation.metrics,
        }
        with lock:
            samples.append(sample)
            groups[f"{row['language']}/{row['family']}"].append(sample)

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for outcome in pool.map(one, zip(prepared, predictions, strict=True)):
                del outcome
    finally:
        for client in opened:
            try:
                client.close()
            except Exception as error:  # noqa: BLE001 - a closed session is the goal
                print(f"  session close failed: {error}", flush=True)
    print(f"  grading done in {time.monotonic() - started:.0f}s", flush=True)
    return samples, groups


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
        "samples": sorted(samples, key=lambda s: s["task_id"]),
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
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-fraction", type=float, default=0.85)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--request-timeout", type=int, default=600)
    parser.add_argument("--boot-seconds", type=int, default=2400)
    parser.add_argument("--vllm-arg", action="append", default=[])
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
        # Sessions are what bounds concurrency: a worker holds one for the whole run.
        env_url = args.env_url or stack.enter_context(
            local_server(args.corpus, args.workers + 4)
        )
        with connect(env_url) as client:
            manifest = client.manifest()
        served = manifest.get("eval_splits") or {}
        if args.eval_split not in served:
            raise SystemExit(
                f"{args.eval_split!r} is not a frozen evaluation split; this deployment "
                f"serves {sorted(served) or 'none'}"
            )
        # A FLEURS file is one row group of 310 MB to 1.5 GB and only a few fit the
        # cache at once, so workers march through the split in language order and stay
        # within a shard or two instead of thrashing it. Tasks are scored one by one, so
        # order changes throughput and nothing else.
        rows = sorted(task_rows(env_url, args.eval_split), key=lambda r: r["language"])
        if args.families:
            rows = [r for r in rows if r["family"] in args.families]
        if args.limit:
            rows = rows[:: max(1, len(rows) // args.limit)][: args.limit]
        print(
            f"{args.eval_split}: {len(rows)} tasks, evalset "
            f"{served[args.eval_split]['evalset_id'][:12]}",
            flush=True,
        )

        # Fetched once and reused by every candidate: the clips are the same, and a
        # second pass would re-read every shard for nothing.
        prepared = prefetch(env_url, rows, args.request_timeout, args.progress)

        leaderboard = []
        for model_id in args.models:
            revision = model_info(model_id).sha
            print(f"\n=== {model_id} @ {revision[:12]} ===", flush=True)
            with vllm_server(
                model_id,
                revision,
                max_model_len=args.max_model_len,
                gpu_fraction=args.gpu_fraction,
                extra_args=args.vllm_arg,
                boot_seconds=args.boot_seconds,
            ) as vllm_url:
                started = time.monotonic()
                predictions = predict(
                    vllm_url,
                    model_id,
                    prepared,
                    args.max_new_tokens,
                    args.request_timeout,
                    args.workers,
                    args.progress,
                )
            samples, groups = grade(
                env_url, prepared, predictions, args.request_timeout, args.workers
            )
            result = summarize(samples, groups, time.monotonic() - started)
            body = {
                "model": model_id,
                "model_revision": revision,
                "engine": VLLM_SPEC,
                "eval_split": args.eval_split,
                "evalset_id": served[args.eval_split]["evalset_id"],
                "snapshot_id": manifest["snapshot_id"],
                "grading": manifest["grading"],
                "families": args.families or "all",
                "tasks": len(rows),
                **result,
            }
            (output / f"{model_id.replace('/', '__')}.json").write_text(
                json.dumps(body, ensure_ascii=False, indent=2) + "\n"
            )
            leaderboard.append(
                {
                    "model": model_id,
                    "tasks": body["tasks"],
                    "macro_reward": body["macro_reward"],
                    "micro_reward": body["micro_reward"],
                    "by_family": body["by_family"],
                    "elapsed_seconds": body["elapsed_seconds"],
                }
            )
            print(
                f"  macro {body['macro_reward']:.4f}  micro {body['micro_reward']:.4f}  "
                f"in {body['elapsed_seconds']}s",
                flush=True,
            )
            # A job's filesystem does not outlive it, so the whole result goes to stdout
            # between markers rather than only to a file nobody will ever read.
            print(f"RESULT-BEGIN {model_id}", flush=True)
            print(json.dumps(body, ensure_ascii=False), flush=True)
            print(f"RESULT-END {model_id}", flush=True)

        leaderboard.sort(key=lambda e: e["macro_reward"], reverse=True)
        (output / "leaderboard.json").write_text(
            json.dumps(
                {
                    "eval_split": args.eval_split,
                    "evalset_id": served[args.eval_split]["evalset_id"],
                    "snapshot_id": manifest["snapshot_id"],
                    "engine": VLLM_SPEC,
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
