"""Score checkpoints on the frozen evaluation set, fast enough to run after every save.

Built for the hot loop, not for one-off scoring. Three costs dominate a naive eval and
each is paid once per session rather than once per checkpoint:

* **The pages.** Fetched once from the corpus and held, so a second checkpoint re-reads
  nothing. They do not depend on the model.
* **The engine.** vLLM boots once against the base model with LoRA updating enabled;
  each checkpoint is loaded as an adapter over the live server and unloaded after. A
  cold boot plus weight download is minutes; an adapter swap is seconds.
* **The GPU.** `--data-parallel-size` puts one engine on each card, so two GPUs answer
  one queue at roughly twice the rate.

Generating and grading run **concurrently**, because they bottleneck on different things:
generation is GPU-bound and wide, grading is judge-bound and narrow. Four of the five
families grade locally and instantly; `descriptive_vqa` goes out to a judge over the
network, and running it behind generation rather than after it hides that latency.

vLLM cannot share an environment with OpenEnv — they disagree on fastmcp — so it runs in
an isolated uv environment and is driven over its OpenAI-compatible API. Pages travel as
base64 data URIs, which is what the environment already serves, so nothing is re-encoded.
"""

import argparse
import base64
import json
import os
import queue
import socket
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path

import requests
from nayana_ocr.client import NayanaClient, connect
from nayana_ocr.data.evalset import load as load_evalset
from nayana_ocr.runtime import local_server
from nayana_ocr.training import step_with_judge_retry

VLLM_SPEC = "vllm==0.30.0"

# The 11 Indic-script languages of this corpus. Sanskrit is here and absent from FLEURS,
# which is why the ASR overlap set covers 21 languages and this one covers 22.
INDIC = ("bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "sa", "ta", "te")

# The one family graded off the machine. It sets the grading width, not the GPU.
JUDGED = "descriptive_vqa"

DONE = object()


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextmanager
def vllm_server(model, revision, *, args):
    """Boot one engine for the whole session, ready to take adapters."""
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
        # One page per prompt is all this task sends; a larger budget reserves
        # multimodal cache for nothing. vLLM 0.30 parses this with json.loads.
        "--limit-mm-per-prompt",
        json.dumps({"image": 1}),
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_fraction),
    ]
    if revision:
        command += ["--revision", revision]
    if args.gpus > 1:
        # Data parallel, not tensor parallel: this is a throughput problem, not a
        # latency one, so a whole engine per card beats splitting one across both.
        command += ["--data-parallel-size", str(args.gpus)]
    if args.adapters:
        command += [
            "--enable-lora",
            "--max-loras",
            str(args.max_loras),
            "--max-lora-rank",
            str(args.max_lora_rank),
        ]
    command += args.vllm_arg
    print("starting vLLM: " + " ".join(command), flush=True)
    process = subprocess.Popen(
        command,
        stdout=sys.stdout,
        stderr=sys.stderr,
        env={
            **os.environ,
            # FlashInfer's sampler is JIT-compiled and wants a CUDA toolkit this image
            # does not carry, so the engine dies on the first token. Greedy decoding has
            # no use for it.
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
            # What makes a checkpoint swap possible without a restart.
            "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "1",
        },
    )
    url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + args.boot_seconds
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
            raise RuntimeError(
                f"vLLM did not become healthy within {args.boot_seconds}s"
            )
        print(f"vLLM ready at {url}", flush=True)
        yield url
    finally:
        process.terminate()
        try:
            process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            process.kill()


@contextmanager
def adapter(vllm_url, name, path):
    """Serve one checkpoint over the live engine, and take it away afterwards."""
    started = time.monotonic()
    response = requests.post(
        f"{vllm_url}/v1/load_lora_adapter",
        json={"lora_name": name, "lora_path": str(path)},
        timeout=600,
    )
    response.raise_for_status()
    print(f"  adapter {name} loaded in {time.monotonic() - started:.1f}s", flush=True)
    try:
        yield name
    finally:
        try:
            requests.post(
                f"{vllm_url}/v1/unload_lora_adapter",
                json={"lora_name": name},
                timeout=120,
            ).raise_for_status()
        except requests.RequestException as error:
            print(f"  adapter {name} not unloaded: {error}", flush=True)


def answer(vllm_url, served, prompt, page, mime, max_tokens, timeout):
    """One chat completion carrying the page as a base64 data URI, as OpenAI defines it."""
    response = requests.post(
        f"{vllm_url}/v1/chat/completions",
        json={
            "model": served,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,"
                                + base64.b64encode(page).decode()
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_completion_tokens": max_tokens,
            # Greedy, so re-running a checkpoint reproduces its score exactly.
            "temperature": 0.0,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def open_client(env_url, timeout):
    """A session of our own, with a timeout that survives a cold row-group read."""
    return NayanaClient(
        base_url=env_url, connect_timeout_s=60, message_timeout_s=timeout
    ).sync()


def prefetch(env_url, rows, timeout, progress):
    """Walk the set once collecting each task's prompt and page image.

    Deliberately sequential. Pages live in row groups of which only a few fit the cache,
    so concurrent readers landing on different documents each stall on their own cold
    read. One reader pays for each group exactly once — and every later checkpoint pays
    nothing, because this is held for the whole session.
    """
    started = time.monotonic()
    prepared = []
    with open_client(env_url, timeout) as client:
        for index, row in enumerate(rows, 1):
            observation = client.reset(task_id=row["task_id"]).observation
            page = requests.get(
                f"{env_url}/assets/{observation.asset_sha256}",
                params={"task_id": observation.task_id},
                timeout=timeout,
            )
            page.raise_for_status()
            prepared.append(
                (row, observation.prompt, page.content, observation.mime or "image/png")
            )
            if progress and index % progress == 0:
                print(
                    f"  pages {index}/{len(rows)} "
                    f"({index / (time.monotonic() - started):.1f} task/s)",
                    flush=True,
                )
    print(
        f"  pages ready: {len(prepared)} images, "
        f"{sum(len(p) for _, _, p, _ in prepared) / 1e6:.0f} MB, "
        f"in {time.monotonic() - started:.0f}s (reused by every checkpoint)",
        flush=True,
    )
    return prepared


def run_one(env_url, vllm_url, served, prepared, args):
    """Generate and grade the whole set, both at once.

    Generation fills a queue as fast as the GPUs allow; grading drains it at whatever
    width the judge tolerates. Neither waits for the other to finish, so the judge's
    latency disappears under generation instead of being added to it.
    """
    started = time.monotonic()
    work = queue.Queue(maxsize=args.workers * 2)
    samples, groups = [], defaultdict(list)
    counts = {"generated": 0, "graded": 0}
    lock = threading.Lock()
    failures = []

    def generate_one(item):
        row, prompt, page, mime = item
        try:
            text = answer(
                vllm_url,
                served,
                prompt,
                page,
                mime,
                args.max_new_tokens,
                args.request_timeout,
            )
        except Exception as error:  # noqa: BLE001 - one bad task must not sink the run
            with lock:
                failures.append(
                    {
                        "task_id": row["task_id"],
                        "stage": "generate",
                        "error": str(error)[:200],
                    }
                )
            return
        work.put((row, text))
        with lock:
            counts["generated"] += 1
            if args.progress and counts["generated"] % args.progress == 0:
                print(
                    f"  generated {counts['generated']}/{len(prepared)} "
                    f"({counts['generated'] / (time.monotonic() - started):.1f} task/s)",
                    flush=True,
                )

    def grade_loop(local):
        # A worker that cannot open a session still has to drain the queue. If it just
        # died, generation would block on a full queue and the run would hang instead of
        # failing, so the session error is recorded and the loop continues.
        client = None
        try:
            client = open_client(env_url, args.request_timeout)
            local.append(client)
        except Exception as error:  # noqa: BLE001 - recorded, never fatal here
            with lock:
                failures.append(
                    {"task_id": None, "stage": "session", "error": str(error)[:200]}
                )
        while True:
            item = work.get()
            try:
                if item is DONE:
                    return
                row, prediction = item
                if client is None:
                    with lock:
                        failures.append(
                            {
                                "task_id": row["task_id"],
                                "stage": "grade",
                                "error": "no grading session",
                            }
                        )
                    continue
                try:
                    client.reset(task_id=row["task_id"])
                    result = step_with_judge_retry(client, prediction)
                except Exception as error:  # noqa: BLE001 - record, do not abort
                    with lock:
                        failures.append(
                            {
                                "task_id": row["task_id"],
                                "stage": "grade",
                                "error": str(error)[:200],
                            }
                        )
                    continue
                sample = {
                    **row,
                    "prediction": prediction,
                    "reward": float(result.reward),
                    **result.observation.metrics,
                }
                with lock:
                    samples.append(sample)
                    groups[f"{row['language']}/{row['family']}"].append(sample)
                    counts["graded"] += 1
                    if args.progress and counts["graded"] % args.progress == 0:
                        print(
                            f"  graded {counts['graded']}/{len(prepared)}",
                            flush=True,
                        )
            finally:
                work.task_done()

    opened = []
    graders = [
        threading.Thread(target=grade_loop, args=(opened,), daemon=True)
        for _ in range(args.grade_workers)
    ]
    for t in graders:
        t.start()
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(generate_one, prepared))
        for _ in graders:
            work.put(DONE)
        for t in graders:
            t.join()
    finally:
        for client in opened:
            try:
                client.close()
            except Exception as error:  # noqa: BLE001 - a closed session is the goal
                print(f"  session close failed: {error}", flush=True)
    if failures:
        print(
            f"  {len(failures)} task(s) failed; see 'failures' in the result",
            flush=True,
        )
    if not samples:
        raise RuntimeError("Every task failed; refusing to report a score")
    return summarize(samples, groups, failures, time.monotonic() - started)


def summarize(samples, groups, failures, elapsed):
    by_group = {}
    by_family, by_language, by_script = (
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
    )
    for key, values in sorted(groups.items()):
        language, family = key.split("/")
        entry = {
            "samples": len(values),
            "reward": sum(v["reward"] for v in values) / len(values),
        }
        for metric in ("exact_match", "cer", "wer", "iou", "accuracy"):
            observed = [v[metric] for v in values if metric in v]
            if observed:
                entry[metric] = sum(observed) / len(observed)
        by_group[key] = entry
        by_family[family].extend(values)
        by_language[language].extend(values)
        by_script["indic" if language in INDIC else "other"].extend(values)

    def mean(values):
        return sum(v["reward"] for v in values) / len(values)

    return {
        # Macro over language/family groups, so no language's share of the set decides
        # the headline. A micro average would let the better-covered ones dominate.
        "macro_reward": sum(e["reward"] for e in by_group.values()) / len(by_group),
        "micro_reward": sum(s["reward"] for s in samples) / len(samples),
        "by_script": {
            k: {
                "samples": len(v),
                "languages": len({s["language"] for s in v}),
                "reward": mean(v),
            }
            for k, v in sorted(by_script.items())
        },
        "by_family": {
            k: {"samples": len(v), "reward": mean(v)}
            for k, v in sorted(by_family.items())
        },
        "by_language": {
            k: {"samples": len(v), "reward": mean(v), "indic": k in INDIC}
            for k, v in sorted(by_language.items())
        },
        "by_language_task": by_group,
        "failures": failures,
        "elapsed_seconds": round(elapsed, 1),
        "samples": sorted(samples, key=lambda s: s["task_id"]),
    }


def parse_adapter(spec):
    name, _, path = spec.partition("=")
    if not path:
        raise argparse.ArgumentTypeError("Use name=/path/to/checkpoint")
    return name, path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus", default="", help="Corpus manifest JSON, or a snapshot"
    )
    parser.add_argument("--env-url", default="")
    parser.add_argument("--evalset", type=Path, required=True)
    parser.add_argument(
        "--base",
        default="",
        help="Model served once; checkpoints load over it as adapters",
    )
    parser.add_argument(
        "--adapters",
        nargs="*",
        type=parse_adapter,
        default=[],
        help="name=/path/to/checkpoint, scored one after another on the live engine",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=[],
        help="Whole models, each needing its own engine. Use --base/--adapters instead "
        "when scoring checkpoints of one run.",
    )
    parser.add_argument(
        "--vllm-url", default="", help="Reuse an engine that is already running"
    )
    parser.add_argument("--languages", nargs="*", default=[])
    parser.add_argument("--families", nargs="*", default=[])
    parser.add_argument("--limit", type=int, default=0, help="Subsample the set evenly")
    parser.add_argument("--gpus", type=int, default=1, help="Engines, one per card")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-fraction", type=float, default=0.85)
    parser.add_argument("--max-loras", type=int, default=2)
    parser.add_argument("--max-lora-rank", type=int, default=16)
    parser.add_argument("--workers", type=int, default=48, help="Generation width")
    parser.add_argument(
        "--grade-workers",
        type=int,
        default=8,
        help="Grading width; the judge sets this, not the GPU",
    )
    parser.add_argument("--request-timeout", type=int, default=900)
    parser.add_argument("--boot-seconds", type=int, default=2400)
    parser.add_argument("--vllm-arg", action="append", default=[])
    parser.add_argument("--progress", type=int, default=100)
    parser.add_argument(
        "--output-dir", default=os.environ.get("OUTPUT_DIR", "artifacts/eval")
    )
    args = parser.parse_args()
    if bool(args.corpus) == bool(args.env_url):
        parser.error("Provide exactly one of --corpus or --env-url")
    if not args.models and not (args.base and args.adapters):
        parser.error("Provide --models, or --base with --adapters")
    if args.models and args.adapters:
        parser.error("Score whole models or adapters of one base, not both at once")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    # The judge refuses work it cannot queue, so it must admit at least as many callers
    # as the grading pool has.
    os.environ.setdefault("NAYANA_JUDGE_CONCURRENCY", str(max(8, args.grade_workers)))

    from huggingface_hub import model_info

    with ExitStack() as stack:
        env_url = args.env_url or stack.enter_context(
            local_server(args.corpus, args.grade_workers + 4)
        )
        with connect(env_url) as client:
            manifest = client.manifest()
        frozen = load_evalset(args.evalset, manifest.get("snapshot_id"))
        rows = [
            {k: t[k] for k in ("task_id", "language", "family")}
            for t in frozen["tasks"]
        ]
        if args.languages:
            rows = [r for r in rows if r["language"] in args.languages]
        if args.families:
            rows = [r for r in rows if r["family"] in args.families]
        if args.limit:
            rows = rows[:: max(1, len(rows) // args.limit)][: args.limit]
        if not rows:
            raise SystemExit("No tasks left after filtering")
        langs = {r["language"] for r in rows}
        print(
            f"{args.evalset.name}: {len(rows)} tasks · {len(langs)} languages "
            f"({len(langs & set(INDIC))} Indic, "
            f"{sum(1 for r in rows if r['language'] in INDIC)} tasks) · "
            f"{len({r['family'] for r in rows})} families · "
            f"evalset {frozen['evalset_id'][:12]}",
            flush=True,
        )

        prepared = prefetch(env_url, rows, args.request_timeout, args.progress)

        common = {
            "engine": VLLM_SPEC,
            "evalset": args.evalset.name,
            "evalset_id": frozen["evalset_id"],
            "snapshot_id": manifest.get("snapshot_id"),
            "grading": manifest.get("grading"),
            "tasks": len(rows),
        }
        leaderboard = []

        def record(label, extra, result):
            body = {**common, **extra, **result}
            (output / f"{label.replace('/', '__')}.json").write_text(
                json.dumps(body, ensure_ascii=False, indent=2) + "\n"
            )
            leaderboard.append(
                {
                    "label": label,
                    "tasks": body["tasks"],
                    "macro_reward": body["macro_reward"],
                    "micro_reward": body["micro_reward"],
                    "by_script": body["by_script"],
                    "by_family": body["by_family"],
                    "elapsed_seconds": body["elapsed_seconds"],
                }
            )
            print(
                f"  macro {body['macro_reward']:.4f}  micro {body['micro_reward']:.4f}  "
                f"indic {body['by_script'].get('indic', {}).get('reward', float('nan')):.4f}  "
                f"in {body['elapsed_seconds']}s",
                flush=True,
            )
            # A job's filesystem does not outlive it, so the whole result goes to stdout
            # between markers rather than only to a file nobody will ever read.
            print(f"RESULT-BEGIN {label}", flush=True)
            print(json.dumps(body, ensure_ascii=False), flush=True)
            print(f"RESULT-END {label}", flush=True)

        if args.adapters:
            revision = model_info(args.base).sha
            server = (
                nullcontext(args.vllm_url)
                if args.vllm_url
                else vllm_server(args.base, revision, args=args)
            )
            with server as vllm_url:
                for name, path in args.adapters:
                    print(f"\n=== {name} (adapter over {args.base}) ===", flush=True)
                    with adapter(vllm_url, name, path):
                        result = run_one(env_url, vllm_url, name, prepared, args)
                    record(
                        name,
                        {
                            "base": args.base,
                            "base_revision": revision,
                            "adapter_path": str(path),
                        },
                        result,
                    )
        else:
            for model_id in args.models:
                revision = model_info(model_id).sha
                print(f"\n=== {model_id} @ {revision[:12]} ===", flush=True)
                server = (
                    nullcontext(args.vllm_url)
                    if args.vllm_url
                    else vllm_server(model_id, revision, args=args)
                )
                with server as vllm_url:
                    result = run_one(env_url, vllm_url, model_id, prepared, args)
                record(
                    model_id, {"model": model_id, "model_revision": revision}, result
                )

        leaderboard.sort(key=lambda e: e["macro_reward"], reverse=True)
        (output / "leaderboard.json").write_text(
            json.dumps({**common, "results": leaderboard}, ensure_ascii=False, indent=2)
            + "\n"
        )
        print("\n" + json.dumps(leaderboard, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
