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
import math
import os
import queue
import re
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
from nayana_ocr.training import READY, checkpoint_complete, step_with_judge_retry

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
    # A watched run's checkpoints arrive as adapters too, just later.
    if args.adapters or getattr(args, "watch", ""):
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


def page_cache_paths(cache_dir, evalset_id, task_id):
    """Where one task's page and prompt live. Keyed by set, so two sets cannot mix."""
    base = Path(cache_dir) / evalset_id
    return (
        base / f"{task_id.replace('/', '_')}.bin",
        base / f"{task_id.replace('/', '_')}.json",
    )


def prefetch(env_url, rows, timeout, progress, *, cache_dir=None, evalset_id=None):
    """Collect each task's prompt and page image, reusing a cache when there is one.

    The corpus read is deliberately sequential: pages live in row groups of which only a
    few fit the cache at once, so concurrent readers landing on different documents each
    stall on their own cold read. One reader pays for each group exactly once.

    That cost returns every time a fresh job starts, which is why the pages can also be
    written to a directory keyed by evalset_id. Mount a bucket there and the whole pass
    becomes local reads. Nothing here depends on the model, so one cache serves every
    checkpoint and every candidate.
    """
    started = time.monotonic()
    prepared, hits = [], 0
    with open_client(env_url, timeout) as client:
        for index, row in enumerate(rows, 1):
            if cache_dir:
                blob, meta = page_cache_paths(cache_dir, evalset_id, row["task_id"])
                if blob.is_file() and meta.is_file():
                    try:
                        info = json.loads(meta.read_text())
                        prepared.append(
                            (row, info["prompt"], blob.read_bytes(), info["mime"])
                        )
                        hits += 1
                        continue
                    except (OSError, ValueError, KeyError) as error:
                        # A half-written entry is refetched, never served.
                        print(
                            f"  cache unusable for {row['task_id']}: {error}",
                            flush=True,
                        )

            observation = client.reset(task_id=row["task_id"]).observation
            page = requests.get(
                f"{env_url}/assets/{observation.asset_sha256}",
                params={"task_id": observation.task_id},
                timeout=timeout,
            )
            page.raise_for_status()
            mime = observation.mime or "image/png"
            prepared.append((row, observation.prompt, page.content, mime))
            if cache_dir:
                try:
                    blob, meta = page_cache_paths(cache_dir, evalset_id, row["task_id"])
                    blob.parent.mkdir(parents=True, exist_ok=True)
                    # Bytes before sidecar: a crash then leaves an entry that reads as
                    # absent rather than as present but empty.
                    blob.write_bytes(page.content)
                    meta.write_text(
                        json.dumps({"prompt": observation.prompt, "mime": mime})
                    )
                except OSError as error:
                    print(f"  page cache disabled: {error}", flush=True)
                    cache_dir = None
            if progress and index % progress == 0:
                print(
                    f"  pages {index}/{len(rows)} "
                    f"({index / (time.monotonic() - started):.1f} task/s)",
                    flush=True,
                )
    print(
        f"  pages ready: {len(prepared)} images, "
        f"{sum(len(p) for _, _, p, _ in prepared) / 1e6:.0f} MB, "
        f"in {time.monotonic() - started:.0f}s "
        f"({hits} from cache, {len(prepared) - hits} fetched)",
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
        for metric in (
            "exact_match", "cer", "wer", "char_error_rate", "official_cer",
            "official_wer", "iou", "accuracy",
        ):
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
        **({"indic_ocr_bench": report} if (report := bench_report(samples)) else {}),
        "samples": sorted(samples, key=lambda s: s["task_id"]),
    }


def bench_report(samples):
    """Sarvam Indic OCR Bench numbers, computed the way its metrics.py reports them.

    Means are over scored samples (empty predictions dropped, as the benchmark does);
    the valid_* figures also drop runaway or catastrophic outputs. Returns None when
    the run held no benchmark tasks.
    """
    from nayana_ocr.data.indic_ocr_bench import is_bench_task
    from nayana_ocr.server.bench_rewards import POLICY

    # Benchmark crops are ordinary section-OCR tasks; their source is in the task ID.
    rows = [s for s in samples if is_bench_task(s.get("task_id"))]
    if not rows:
        return None
    scored = [s for s in rows if not s.get("missing_prediction")]
    valid = [s for s in scored if not s.get("loop_or_catastrophic")]

    def avg(values, key):
        return sum(v[key] for v in values) / len(values) if values else None

    by_language = defaultdict(list)
    for s in scored:
        by_language[s["language"]].append(s)
    cer, wer = avg(scored, "official_cer"), avg(scored, "official_wer")
    vcer, vwer = avg(valid, "official_cer"), avg(valid, "official_wer")
    return {
        "scorer": POLICY,
        "avg_metrics": {"cer": cer, "wer": wer},
        "word_accuracy": None if wer is None else 100.0 * (1.0 - wer),
        "valid_samples_cer": vcer,
        "valid_samples_wer": vwer,
        "valid_word_accuracy": None if vwer is None else 100.0 * (1.0 - vwer),
        "benchmark_sample_count": len(rows),
        "scored_sample_count": len(scored),
        "valid_sample_count": len(valid),
        "missing_prediction_count": len(rows) - len(scored),
        "loop_failure_count": sum(1 for s in scored if s.get("loop_or_catastrophic")),
        "lang_wise_scores": {
            lang: {
                "cer": avg(v, "official_cer"),
                "wer": avg(v, "official_wer"),
                "sample_count": len(v),
            }
            for lang, v in sorted(by_language.items())
        },
    }


# The benchmark's own CER and WER sit beside the environment's reward and CER, so a
# checkpoint can be read against Sarvam's published numbers as well as the corpus.
CURVE_METRICS = (
    "reward", "char_error_rate", "official_cer", "official_wer", "exact_match",
)


def paired(base, other, metric):
    """Mean of other minus base over the tasks both scored, with a 95% interval.

    Both sides are scored on the same crops, so the per-crop difference removes how
    hard each crop is, and its spread is far smaller than either score's.
    """
    before = {s["task_id"]: s[metric] for s in base if metric in s}
    pairs = [
        float(s[metric]) - float(before[s["task_id"]])
        for s in other
        if metric in s and s["task_id"] in before
    ]
    if len(pairs) < 2:
        return None
    mean = sum(pairs) / len(pairs)
    spread = math.sqrt(sum((d - mean) ** 2 for d in pairs) / (len(pairs) - 1))
    half = 1.96 * spread / math.sqrt(len(pairs))
    return {"delta": mean, "low": mean - half, "high": mean + half, "tasks": len(pairs)}


def curve_point(step, body, base):
    """One checkpoint's row of the run's curve: means, and paired change from base."""
    samples = body["samples"]
    point = {"step": step, "model": body.get("model") or body.get("label"),
             "tasks": len(samples)}
    for metric in CURVE_METRICS:
        observed = [float(s[metric]) for s in samples if metric in s]
        if observed:
            point[metric] = sum(observed) / len(observed)
        if base is not None:
            change = paired(base["samples"], samples, metric)
            if change:
                point[f"{metric}_change"] = change
    report = body.get("indic_ocr_bench") or {}
    for key, value in (report.get("avg_metrics") or {}).items():
        # Sarvam's report drops missing predictions from its means, so it is kept as
        # its own series rather than mixed with the per-sample ones.
        if isinstance(value, (int, float)):
            point[f"sarvam_{key}"] = float(value)
    return point


def checkpoint_steps(paths, prefix):
    """Steps whose checkpoint has been marked ready under `prefix`."""
    pattern = re.compile(rf"^{re.escape(prefix)}/checkpoint-(\d+)/{READY}$")
    return sorted({int(m.group(1)) for p in paths if (m := pattern.match(p))})


class RunWatcher:
    """Follow a training run's bucket and hand over each checkpoint once complete.

    The bucket is read through the Hub API, not a mount: a mount may cache a listing,
    and the run is being written by another machine. A checkpoint counts as complete
    once its ready.json exists and every adapter byte matches the hashes in it. A
    checkpoint that is listed but still uploading is retried on the next poll.
    """

    TERMINAL = {"COMPLETED", "ERROR", "CANCELED", "DELETED"}

    def __init__(self, location, work, job=None):
        from huggingface_hub import HfApi

        parts = location.strip("/").split("/")
        if len(parts) < 3:
            raise ValueError("Use --watch namespace/bucket/run-prefix")
        self.bucket, self.prefix = "/".join(parts[:2]), "/".join(parts[2:])
        self.work, self.job, self.api = Path(work), job, HfApi()

    def listing(self):
        return [
            getattr(item, "path", "")
            for item in self.api.list_bucket_tree(
                self.bucket, prefix=self.prefix + "/", recursive=True
            )
        ]

    def finished(self, paths):
        """The run is over: its job is terminal, or it wrote its final summary."""
        if f"{self.prefix}/summary.json" in paths:
            return True
        if self.job:
            stage = self.api.inspect_job(job_id=self.job).status.stage
            return str(getattr(stage, "value", stage)) in self.TERMINAL
        return False

    def fetch(self, step, names):
        """Download one checkpoint's adapter, or return None if it is not all there."""
        remote = f"{self.prefix}/checkpoint-{step}"
        local = self.work / f"checkpoint-{step}"
        local.mkdir(parents=True, exist_ok=True)
        self.api.download_bucket_files(
            self.bucket, [(f"{remote}/{n}", str(local / n)) for n in names]
        )
        ready = json.loads((local / READY).read_text())
        return local if checkpoint_complete(local, ready) else None

    def trainer_state(self, step):
        """The run's log history as of a checkpoint, for plotting the training side."""
        local = self.work / f"checkpoint-{step}" / "trainer_state.json"
        try:
            self.api.download_bucket_files(
                self.bucket,
                [(f"{self.prefix}/checkpoint-{step}/trainer_state.json", str(local))],
            )
        except Exception as error:  # noqa: BLE001 - a plot without it still has evals
            print(f"  no trainer state for step {step}: {error}", flush=True)
            return None
        return local


def plot(curve_path, trainer_state, out_dir, title):
    """Redraw the run's figures. A failed plot is reported and never stops scoring."""
    script = Path(__file__).with_name("plot_run.py")
    command = ["uv", "run", "--script", str(script), "--curve", str(curve_path),
               "--out", str(out_dir), "--title", title]
    if trainer_state:
        command += ["--trainer-state", str(trainer_state)]
    try:
        subprocess.run(command, check=True, timeout=600)
    except Exception as error:  # noqa: BLE001
        print(f"  plotting failed: {error}", flush=True)
        return []
    return sorted(Path(out_dir).glob("*.png"))


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
    parser.add_argument("--evalset", type=Path, default=None)
    parser.add_argument(
        "--split",
        default="",
        help="Score a split the environment serves, e.g. indic_ocr_bench_test, "
        "instead of a frozen evalset file",
    )
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
    parser.add_argument(
        "--page-cache",
        default=os.environ.get("NAYANA_PAGE_CACHE"),
        help="Directory holding fetched pages, keyed by evalset id. Mount a bucket here "
        "and the corpus read is paid once ever instead of once per job.",
    )
    parser.add_argument("--progress", type=int, default=100)
    parser.add_argument(
        "--watch",
        default="",
        help="namespace/bucket/run-prefix of a training run: score the base model, "
        "then every checkpoint as it lands, until the run finishes",
    )
    parser.add_argument(
        "--follow-job", default="", help="Training job to wait on while watching"
    )
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument(
        "--idle-hours",
        type=float,
        default=3.0,
        help="Give up watching after this long without a new checkpoint",
    )
    parser.add_argument("--trackio-space", default="")
    parser.add_argument("--trackio-project", default="huggingface")
    parser.add_argument("--run-name", default="")
    parser.add_argument(
        "--output-dir", default=os.environ.get("OUTPUT_DIR", "artifacts/eval")
    )
    args = parser.parse_args()
    if bool(args.evalset) == bool(args.split):
        parser.error("Score exactly one of --evalset FILE or --split NAME")
    if bool(args.corpus) == bool(args.env_url):
        parser.error("Provide exactly one of --corpus or --env-url")
    if args.watch and not args.base:
        parser.error("--watch scores checkpoints over --base")
    if not args.models and not (args.base and (args.adapters or args.watch)):
        parser.error("Provide --models, or --base with --adapters or --watch")
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
        if args.split:
            if args.split not in (manifest.get("splits") or []):
                raise SystemExit(
                    f"{args.split!r} is not served here; this deployment serves "
                    f"{sorted(manifest.get('splits') or [])}"
                )
            listed = []
            with connect(env_url) as client:
                total = client.num_tasks(args.split)
                for start in range(0, total, 1000):
                    listed.extend(
                        client.get_task_range(args.split, start, min(start + 1000, total))
                    )
            described = (manifest.get("eval_splits") or {}).get(args.split) or {}
            frozen = {
                "tasks": listed,
                "evalset_id": described.get("evalset_id") or f"split:{args.split}",
            }
            label = args.split
        else:
            frozen = load_evalset(args.evalset, manifest.get("snapshot_id"))
            label = args.evalset.name
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
            f"{label}: {len(rows)} tasks · {len(langs)} languages "
            f"({len(langs & set(INDIC))} Indic, "
            f"{sum(1 for r in rows if r['language'] in INDIC)} tasks) · "
            f"{len({r['family'] for r in rows})} families · "
            f"evalset {frozen['evalset_id'][:12]}",
            flush=True,
        )

        prepared = prefetch(
            env_url,
            rows,
            args.request_timeout,
            args.progress,
            cache_dir=args.page_cache,
            evalset_id=frozen["evalset_id"],
        )

        common = {
            "engine": VLLM_SPEC,
            "evalset": label,
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
            # A job's filesystem does not outlive it, so a copy goes to stdout — but
            # only the summary. Page transcripts run to thousands of characters and the
            # log transport wraps long lines mid-escape, which corrupted a whole run's
            # detail beyond recovery. Predictions travel by --artifact-repo, which is a
            # file transfer and does not rewrite what it carries.
            print(f"RESULT-BEGIN {label}", flush=True)
            print(
                json.dumps(
                    {k: v for k, v in body.items() if k != "samples"},
                    ensure_ascii=False,
                ),
                flush=True,
            )
            print(f"RESULT-END {label}", flush=True)
            return body

        if args.watch:
            watch(args, stack, env_url, prepared, record, output, len(rows))
            return

        if args.adapters:
            revision = model_info(args.base).sha
            server = (
                nullcontext(args.vllm_url)
                if args.vllm_url
                else vllm_server(args.base, revision, args=args)
            )
            with server as vllm_url:
                # The untuned model on the same engine, same tasks, same grading. A
                # curve of checkpoints says nothing without the point it started from,
                # and the trainer's own baseline uses a different harness and task
                # count, so it is not comparable with these numbers.
                print(f"\n=== {args.base} (no adapter) ===", flush=True)
                record(
                    args.base,
                    {"model": args.base, "model_revision": revision},
                    run_one(env_url, vllm_url, args.base, prepared, args),
                )
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


def watch(args, stack, env_url, prepared, record, output, tasks):
    """Score the base, then each checkpoint of a live run, on one engine.

    Every result is written as it is taken. The curve, the figures, and the Trackio run
    are redrawn after each checkpoint, so the run can be read while it trains, and a
    watcher that dies loses nothing it already scored.
    """
    import trackio
    from huggingface_hub import model_info

    watcher = RunWatcher(args.watch, output / "checkpoints", args.follow_job or None)
    revision = model_info(args.base).sha
    run_name = args.run_name or f"{watcher.prefix.split('/')[-1][:12]}-eval"
    label = args.split or args.evalset.name
    tracker = None
    if args.trackio_space:
        tracker = trackio.init(
            project=args.trackio_project,
            name=run_name,
            space_id=args.trackio_space,
            resume="allow",
            config={
                "watch": args.watch,
                "evalset": label,
                "languages": args.languages or "all",
                "tasks": tasks,
                "base": args.base,
            },
        )
    vllm_url = args.vllm_url or stack.enter_context(
        vllm_server(args.base, revision, args=args)
    )
    curve_path, plots = output / "curve.json", output / "plots"
    curve = json.loads(curve_path.read_text()) if curve_path.exists() else []
    done = {point["step"] for point in curve}
    base_file = output / f"{args.base.replace('/', '__')}.json"
    if base_file.exists():
        base = json.loads(base_file.read_text())
    else:
        print(f"\n=== {args.base} (no adapter) ===", flush=True)
        base = record(
            args.base,
            {"model": args.base, "model_revision": revision},
            run_one(env_url, vllm_url, args.base, prepared, args),
        )

    def publish(point, trainer_state):
        curve[:] = sorted([p for p in curve if p["step"] != point["step"]] + [point],
                          key=lambda p: p["step"])
        curve_path.write_text(json.dumps(curve, indent=2) + "\n")
        figures = plot(curve_path, trainer_state, plots, f"{run_name} - {label}")
        if tracker is None:
            return
        logged = {f"eval/{k}": v for k, v in point.items() if isinstance(v, float)}
        for metric in CURVE_METRICS:
            change = point.get(f"{metric}_change")
            if change:
                logged[f"eval/{metric}_change"] = change["delta"]
                logged[f"eval/{metric}_change_low"] = change["low"]
                logged[f"eval/{metric}_change_high"] = change["high"]
        logged.update({f"plots/{f.stem}": trackio.Image(str(f)) for f in figures})
        trackio.log(logged, step=point["step"])

    if 0 not in done:
        publish(curve_point(0, base, None), None)
        done.add(0)
    idle_since, attempts = time.monotonic(), {}
    names = (READY, "adapter_config.json", "adapter_model.safetensors")
    while True:
        paths = watcher.listing()
        over = watcher.finished(paths)  # read before the listing is acted on
        pending = [s for s in checkpoint_steps(paths, watcher.prefix) if s not in done]
        for step in pending:
            try:
                local = watcher.fetch(step, names)
            except Exception as error:  # noqa: BLE001 - a missing file is retried too
                print(f"  checkpoint-{step} not readable yet: {error}", flush=True)
                local = None
            if local is None:
                attempts[step] = attempts.get(step, 0) + 1
                # Once the run is over nothing more will arrive; a checkpoint that is
                # still incomplete then was cut off mid-write and cannot be scored.
                if over and attempts[step] >= 5:
                    print(f"  checkpoint-{step} never completed; skipped", flush=True)
                    done.add(step)
                else:
                    print(f"  checkpoint-{step} still uploading; retrying", flush=True)
                continue
            name = f"step-{step}"
            print(f"\n=== {name} (adapter over {args.base}) ===", flush=True)
            with adapter(vllm_url, name, local):
                result = run_one(env_url, vllm_url, name, prepared, args)
            body = record(
                name,
                {"base": args.base, "base_revision": revision, "adapter_path": str(local)},
                result,
            )
            point = curve_point(step, {**body, "model": name}, base)
            print("  curve " + json.dumps({k: v for k, v in point.items() if k != "model"}),
                  flush=True)
            publish(point, watcher.trainer_state(step))
            done.add(step)
            idle_since = time.monotonic()
        if over and not [s for s in checkpoint_steps(watcher.listing(), watcher.prefix)
                         if s not in done]:
            print(f"run finished; scored {len(done) - 1} checkpoints", flush=True)
            break
        if time.monotonic() - idle_since > args.idle_hours * 3600:
            print(f"no new checkpoint in {args.idle_hours}h; stopping", flush=True)
            break
        time.sleep(args.poll_seconds)
    if tracker is not None:
        trackio.finish()


if __name__ == "__main__":
    main()
