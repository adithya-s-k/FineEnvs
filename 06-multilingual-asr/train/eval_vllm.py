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
import math
import os
import re
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
from multilingual_asr.server.rewards import REWARD_UNITS
from multilingual_asr.training import READY, checkpoint_complete, task_rows

VLLM_SPEC = "vllm[audio]==0.30.0"


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextmanager
def vllm_server(
    model,
    revision,
    *,
    max_model_len,
    gpu_fraction,
    extra_args,
    boot_seconds,
    loras=0,
    max_lora_rank=16,
):
    """Serve one checkpoint, in an environment of its own.

    With loras set the engine also accepts adapters at runtime, so a whole run's
    checkpoints are scored on one boot. A cold boot plus a 16GB weight download is
    minutes; an adapter swap is seconds.
    """
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
    ]
    if loras:
        command += [
            "--enable-lora",
            "--max-loras",
            str(loras),
            "--max-lora-rank",
            str(max_lora_rank),
        ]
    command += extra_args
    print("starting vLLM: " + " ".join(command), flush=True)
    # FlashInfer's sampler is JIT-compiled and wants a CUDA toolkit this image does not
    # carry, so the engine dies on the first token. Evaluation samples greedily and has
    # no use for it; vLLM's own sampler is what gets used either way.
    process = subprocess.Popen(
        command,
        stdout=sys.stdout,
        stderr=sys.stderr,
        env={
            **os.environ,
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
            # Runtime load/unload is behind this flag; without it the endpoints 404
            # and every checkpoint would need its own boot.
            **({"VLLM_ALLOW_RUNTIME_LORA_UPDATING": "1"} if loras else {}),
        },
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


@contextmanager
def adapter(vllm_url, name, path):
    """Serve one checkpoint over the live engine, and take it away afterwards."""
    started = time.monotonic()
    requests.post(
        f"{vllm_url}/v1/load_lora_adapter",
        json={"lora_name": name, "lora_path": str(path)},
        timeout=600,
    ).raise_for_status()
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
            # An adapter left loaded wastes a slot; it does not invalidate the score
            # already taken, so this reports rather than discards the run.
            print(f"  adapter {name} not unloaded: {error}", flush=True)


def parse_adapter(spec):
    name, _, path = spec.partition("=")
    if not path:
        raise argparse.ArgumentTypeError("Use name=/path/to/checkpoint")
    return name, path


def answer(vllm_url, model, prompt, wav, max_tokens, timeout, temperature=0.0, n=1):
    """One chat completion carrying the clip as base64 WAV, as OpenAI defines it.

    With n > 1 this returns n sampled transcripts for the clip instead of one string.
    """
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
            # Greedy by default, so a re-run of the same checkpoint gives the same score.
            # Sampling exists to ask a different question: what the policy the trainer
            # optimised (it samples at T=0.9) actually does on held-out clips.
            "temperature": temperature,
            "n": n,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    texts = [c["message"]["content"].strip() for c in response.json()["choices"]]
    return texts if n > 1 else texts[0]


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


def predict(
    vllm_url, model, prepared, max_tokens, timeout, workers, progress,
    temperature=0.0, n=1,
):
    """Ask vLLM for every answer at once. Nothing here touches the environment."""
    started = time.monotonic()
    done = [0]
    lock = threading.Lock()

    def one(item):
        _, prompt, wav = item
        text = answer(vllm_url, model, prompt, wav, max_tokens, timeout, temperature, n)
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


def headroom(greedy, sampled):
    """Compare a model's greedy answer with its own samples, task by task.

    GRPO can only shift probability among answers the policy already produces. If no
    sample beats the greedy answer, no amount of reweighting moves greedy evaluation;
    if many do, the headroom is there and the question becomes why training missed it.

    greedy: {task_id: reward}; sampled: {task_id: [reward, ...]}.
    """
    keys = [k for k in greedy if sampled.get(k)]
    if not keys:
        return {}
    best = [max(sampled[k]) for k in keys]
    mean = [sum(sampled[k]) / len(sampled[k]) for k in keys]
    g = [greedy[k] for k in keys]
    better = [sum(r > greedy[k] + 1e-9 for r in sampled[k]) / len(sampled[k]) for k in keys]
    return {
        "headroom_tasks": len(keys),
        "samples_per_task": len(sampled[keys[0]]),
        "greedy_mean": sum(g) / len(g),
        "sampled_mean": sum(mean) / len(mean),
        "best_of_n_mean": sum(best) / len(best),
        "best_minus_greedy": sum(b - x for b, x in zip(best, g, strict=True)) / len(g),
        "tasks_with_a_better_sample": sum(
            b > x + 1e-9 for b, x in zip(best, g, strict=True)
        ) / len(g),
        "mean_share_of_samples_beating_greedy": sum(better) / len(better),
    }


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


CURVE_METRICS = ("reward", "cer", "wer", "exact_match")


def paired(base, other, metric):
    """Mean of other minus base over the tasks both scored, with a 95% interval.

    Both sides are scored on the same clips, so the per-clip difference removes how hard
    each clip is, and its spread is far smaller than either score's. This is what makes
    a half-point change on 838 clips readable at all.
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
    point = {"step": step, "model": body["model"], "tasks": len(samples)}
    for metric in CURVE_METRICS:
        observed = [float(s[metric]) for s in samples if metric in s]
        if observed:
            point[metric] = sum(observed) / len(observed)
        if base is not None:
            change = paired(base["samples"], samples, metric)
            if change:
                point[f"{metric}_change"] = change
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
    parser.add_argument(
        "--models",
        nargs="*",
        default=[],
        help="Whole models, each needing its own engine. Use --base with --adapters "
        "when scoring checkpoints of one run.",
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
        "--samples",
        type=int,
        default=1,
        help="Also draw this many sampled transcripts per clip and report sampled mean "
        "and best-of-n beside the greedy score. 1 scores greedy only.",
    )
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument(
        "--rows-from",
        type=Path,
        default=None,
        help="Score the train_tasks listed in a run-metadata.json instead of the eval "
        "split -- the clips a run trained on, to tell learning from memorising",
    )
    parser.add_argument(
        "--output-tag",
        default="",
        help="Prefix for result files, so two sweeps of one revision cannot overwrite "
        "each other's results",
    )
    parser.add_argument("--max-loras", type=int, default=2)
    parser.add_argument("--max-lora-rank", type=int, default=16)
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
        "--reward-unit",
        choices=REWARD_UNITS,
        default="script",
        help="Grade with the same policy the run trained on",
    )
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
    if args.watch and not args.base:
        parser.error("--watch scores checkpoints over --base")
    if not args.models and not (args.base and (args.adapters or args.watch)):
        parser.error("Provide --models, or --base with --adapters or --watch")
    if args.models and args.adapters:
        parser.error("Score whole models or adapters of one base, not both at once")
    if bool(args.corpus) == bool(args.env_url):
        parser.error("Provide exactly one of --corpus or --env-url")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import model_info

    with ExitStack() as stack:
        # Sessions are what bounds concurrency: a worker holds one for the whole run.
        env_url = args.env_url or stack.enter_context(
            local_server(args.corpus, args.workers + 4, args.reward_unit)
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
        if args.rows_from:
            listed = json.loads(Path(args.rows_from).read_text())["train_tasks"]
            rows = sorted(listed, key=lambda r: r["language"])
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

        def record(model_id, revision, result, rows_scored, adapter_path=None, extra=None):
            body = {
                "model": model_id,
                "model_revision": revision,
                "engine": VLLM_SPEC,
                "eval_split": args.eval_split,
                "evalset_id": served[args.eval_split]["evalset_id"],
                "snapshot_id": manifest["snapshot_id"],
                "grading": manifest["grading"],
                "families": args.families or "all",
                "tasks": rows_scored,
                **({"adapter_path": str(adapter_path)} if adapter_path else {}),
                **({"rows_from": str(args.rows_from)} if args.rows_from else {}),
                **result,
                **(extra or {}),
            }
            (output / f"{args.output_tag}{model_id.replace('/', '__')}.json").write_text(
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
            return body

        def score(vllm_url, served_name):
            """Greedy score, plus sampled and best-of-n when --samples asks for them."""
            started = time.monotonic()
            predictions = predict(
                vllm_url, served_name, prepared, args.max_new_tokens,
                args.request_timeout, args.workers, args.progress,
            )
            samples, groups = grade(
                env_url, prepared, predictions, args.request_timeout, args.workers
            )
            result = summarize(samples, groups, time.monotonic() - started)
            if args.samples <= 1:
                return result, None
            drawn = predict(
                vllm_url, served_name, prepared, args.max_new_tokens,
                args.request_timeout, args.workers, args.progress,
                temperature=args.temperature, n=args.samples,
            )
            flat_prepared = [
                item for item, texts in zip(prepared, drawn, strict=True) for _ in texts
            ]
            flat_texts = [text for texts in drawn for text in texts]
            sampled_rows_scored, _ = grade(
                env_url, flat_prepared, flat_texts, args.request_timeout, args.workers
            )
            greedy = {s["task_id"]: s["reward"] for s in samples}
            sampled = defaultdict(list)
            for s in sampled_rows_scored:
                sampled[s["task_id"]].append(s["reward"])
            stats = headroom(greedy, sampled)
            stats["temperature"] = args.temperature
            print("  headroom " + json.dumps(stats), flush=True)
            return result, stats

        if args.watch:
            watch(args, stack, score, record, rows, output)
            return

        if args.adapters:
            revision = model_info(args.base).sha
            print(f"\n=== base {args.base} @ {revision[:12]} ===", flush=True)
            with vllm_server(
                args.base,
                revision,
                max_model_len=args.max_model_len,
                gpu_fraction=args.gpu_fraction,
                extra_args=args.vllm_arg,
                boot_seconds=args.boot_seconds,
                loras=args.max_loras,
                max_lora_rank=args.max_lora_rank,
            ) as vllm_url:
                # The untuned model on the same engine, same tasks, same grading. A
                # curve of checkpoints says nothing without the point it started from,
                # and the trainer's own baseline uses a different harness and task
                # count, so it is not comparable with these numbers.
                print(f"\n=== {args.base} (no adapter) ===", flush=True)
                result, extra = score(vllm_url, args.base)
                record(args.base, revision, result, len(rows), extra=extra)

                for name, path in args.adapters:
                    print(f"\n=== {name} (adapter over {args.base}) ===", flush=True)
                    with adapter(vllm_url, name, path):
                        result, extra = score(vllm_url, name)
                    record(name, revision, result, len(rows), adapter_path=path, extra=extra)

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
                result, extra = score(vllm_url, model_id)
            record(model_id, revision, result, len(rows), extra=extra)

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


def watch(args, stack, score, record, rows, output):
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
    tracker = None
    if args.trackio_space:
        tracker = trackio.init(
            project=args.trackio_project,
            name=run_name,
            space_id=args.trackio_space,
            resume="allow",
            config={
                "watch": args.watch,
                "eval_split": args.eval_split,
                "tasks": len(rows),
                "base": args.base,
                "reward_unit": args.reward_unit,
            },
        )
    vllm_url = stack.enter_context(
        vllm_server(
            args.base,
            revision,
            max_model_len=args.max_model_len,
            gpu_fraction=args.gpu_fraction,
            extra_args=args.vllm_arg,
            boot_seconds=args.boot_seconds,
            loras=args.max_loras,
            max_lora_rank=args.max_lora_rank,
        )
    )
    curve_path, plots = output / "curve.json", output / "plots"
    curve = json.loads(curve_path.read_text()) if curve_path.exists() else []
    done = {point["step"] for point in curve}
    base_file = output / f"{args.base.replace('/', '__')}.json"
    if base_file.exists():
        base = json.loads(base_file.read_text())
    else:
        print(f"\n=== {args.base} (no adapter) ===", flush=True)
        result, extra = score(vllm_url, args.base)
        base = record(args.base, revision, result, len(rows), extra=extra)

    def publish(point, trainer_state):
        curve[:] = sorted([p for p in curve if p["step"] != point["step"]] + [point],
                          key=lambda p: p["step"])
        curve_path.write_text(json.dumps(curve, indent=2) + "\n")
        figures = plot(curve_path, trainer_state, plots, f"{run_name} - {args.eval_split}")
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
                result, extra = score(vllm_url, name)
            body = record(name, revision, result, len(rows), adapter_path=local, extra=extra)
            point = curve_point(step, body, base)
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
