"""Choose and audit the verifier models against synthetic renders with known truth.

Each image is a real target from the served test split, drawn in one of seven variants
whose literal transcription is known exactly (fixtures.render). Every candidate model
reads every image blind, through the same prompt and client the environment uses.
Reported per model:

  literal_exact     transcription equals the literal truth
  repaired_typo     a deliberate typo/missing/doubled letter came back "corrected"
  malformed_hit     a broken glyph was marked MALFORMED in the target span
  malformed_repair  a broken glyph was read as the intended letter (the dangerous miss)
  false_malformed   MALFORMED appeared in a clean render
  reward_mae        |reward from this reading - reward from the literal truth|

then every pair is scored with the environment's own combine() rule.

  ../../launch image-text-gen-rl --exec python image-text-gen-rl/train/calibrate_verifier.py \
      --targets 24 --output image-text-gen-rl/artifacts/calibration.json
"""

import argparse
import itertools
import json
import random
import statistics
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from image_text_gen.client import encode_image
from image_text_gen.data.catalog import configured_catalog
from image_text_gen.fixtures import fonts, render
from image_text_gen.server.images import prepare
from image_text_gen.server.scoring import MALFORMED, combine, normalise, score_reading
from image_text_gen.server.verifier import Verifier, VerifierModel, VerifierUnavailable

NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}
CANDIDATES = (
    VerifierModel("Qwen/Qwen3-VL-235B-A22B-Instruct", "deepinfra"),
    VerifierModel("Qwen/Qwen3-VL-30B-A3B-Instruct", "deepinfra"),
    VerifierModel("Qwen/Qwen3.8-27B", "deepinfra", NO_THINK),
    VerifierModel("Qwen/Qwen3.6-27B", "deepinfra", NO_THINK),
    VerifierModel("Qwen/Qwen3.6-35B-A3B", "deepinfra", NO_THINK),
    VerifierModel("Qwen/Qwen3.5-397B-A17B", "deepinfra", NO_THINK),
    VerifierModel("Qwen/Qwen3.5-27B", "deepinfra", NO_THINK),
    VerifierModel("google/gemma-4-31B-it", "deepinfra"),
    VerifierModel("zai-org/GLM-4.6V-Flash", "novita", {"thinking": {"type": "disabled"}}),
    VerifierModel("deepseek-ai/DeepSeek-V4-Flash-Vision-Exp", "deepinfra"),
    VerifierModel("meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", "novita"),
)
# Dropped after the 4-target pilot: Kimi-K2.5 (empty readings on 36% of images, likely
# reasoning consuming the token budget; highest cost) and MiniMax-M3 (8.8 s median latency).
VARIANTS = (
    "clean", "typo_swap", "missing_char", "doubled_char", "wrong_case",
    "malformed_glyph", "extra_text",
)
TYPO_VARIANTS = {"typo_swap", "missing_char", "doubled_char"}
PRICES = {}  # route -> (input $/M, output $/M), filled from the router


def load_prices(token):
    import requests

    data = requests.get(
        "https://router.huggingface.co/v1/models",
        headers={"Authorization": f"Bearer {token}"}, timeout=30,
    ).json()["data"]
    for model in data:
        for provider in model.get("providers", []):
            pricing = provider.get("pricing") or {}
            PRICES[f"{model['id']}:{provider['provider']}"] = (
                pricing.get("input") or 0.0, pricing.get("output") or 0.0
            )


def pick_targets(catalog, count, seed):
    """Stratify by target length so long texts are represented, not just the median."""
    rows = [catalog.at("test", i) for i in range(catalog.count("test"))]
    buckets = defaultdict(list)
    for row in rows:
        length = row["text_len"]
        key = 0 if length <= 10 else 1 if length <= 20 else 2 if length <= 40 else 3
        buckets[key].append(row)
    rng, shares = random.Random(seed), (0.3, 0.3, 0.25, 0.15)
    chosen = []
    for key, share in enumerate(shares):
        chosen += rng.sample(buckets[key], min(len(buckets[key]), max(1, round(count * share))))
    return chosen[:count]


def build_images(targets, seed):
    items, font_list = [], fonts() or [None]
    for n, task in enumerate(targets):
        for variant in VARIANTS:
            font = font_list[(n + VARIANTS.index(variant)) % len(font_list)]
            image, expected = render(task["target_text"], variant, seed=seed + n, size=(1024, 640),
                                     font_path=font)
            png, info = prepare(encode_image(image))
            items.append({"task_id": task["task_id"], "target": task["target_text"],
                          "variant": variant, "expected": expected, "font": font,
                          "png": png, "info": info})
    return items


def judge_reading(item, text):
    target, expected, variant = item["target"], item["expected"], item["variant"]
    literal = normalise(text) == normalise(expected)
    reading = score_reading(target, text)
    row = {
        "literal_exact": literal,
        "reward": combine(target, [text])[0],
        "expected_reward": combine(target, [expected])[0],
    }
    if variant in TYPO_VARIANTS:
        row["repaired_typo"] = normalise(target).lower() in normalise(text).lower() and not literal
    if variant == "malformed_glyph":
        row["malformed_hit"] = reading["malformed_glyphs"] > 0
        row["malformed_repair"] = reading["exact_match"]
    if variant == "clean":
        row["false_malformed"] = MALFORMED in normalise(text)
    return row


def run_model(spec, items, token, workers):
    verifier = Verifier([spec], token=token, concurrency=workers, timeout=90)
    results = [None] * len(items)

    def one(index):
        item = items[index]
        try:
            reading = verifier._read(spec, item["png"])
        except VerifierUnavailable as error:
            return index, {"error": str(error)[:160]}
        return index, {**reading, **judge_reading(item, reading["text"])}

    with ThreadPoolExecutor(workers) as pool:
        for future in as_completed([pool.submit(one, i) for i in range(len(items))]):
            index, row = future.result()
            results[index] = row
    verifier.pool.shutdown(wait=False)
    return results


def rate(rows, key):
    values = [r[key] for r in rows if key in r]
    return round(sum(values) / len(values), 4) if values else None


def summarise(spec, rows):
    ok = [r for r in rows if "error" not in r]
    input_price, output_price = PRICES.get(spec.route, (0.0, 0.0))
    cost = sum(r["prompt_tokens"] * input_price + r["completion_tokens"] * output_price for r in ok) / 1e6
    return {
        "route": spec.route,
        "answered": len(ok),
        "errors": len(rows) - len(ok),
        "literal_exact": rate(ok, "literal_exact"),
        "repaired_typo": rate(ok, "repaired_typo"),
        "malformed_hit": rate(ok, "malformed_hit"),
        "malformed_repair": rate(ok, "malformed_repair"),
        "false_malformed": rate(ok, "false_malformed"),
        "reward_mae": round(statistics.fmean(abs(r["reward"] - r["expected_reward"]) for r in ok), 4)
        if ok else None,
        "p50_latency_s": round(statistics.median(r["latency_s"] for r in ok), 2) if ok else None,
        "usd_per_1k_images": round(1000 * cost / len(ok), 3) if ok else None,
        "sample_errors": sorted({r["error"] for r in rows if "error" in r})[:2],
    }


def pair_scores(items, per_model, routes):
    out = []
    for a, b in itertools.combinations(routes, 2):
        errors, orderings = [], []
        by_task = defaultdict(dict)
        for i, item in enumerate(items):
            ra, rb = per_model[a][i], per_model[b][i]
            if "error" in ra or "error" in rb:
                continue
            value = combine(item["target"], [ra["text"], rb["text"]])[0]
            errors.append(abs(value - ra["expected_reward"]))
            by_task[item["task_id"]][item["variant"]] = value
        for variants in by_task.values():
            if "clean" in variants:
                orderings += [variants["clean"] > v for k, v in variants.items() if k != "clean"]
        if errors:
            out.append({
                "pair": [a, b],
                "reward_mae": round(statistics.fmean(errors), 4),
                "clean_ranked_above_defects": round(sum(orderings) / max(1, len(orderings)), 4),
                "images": len(errors),
            })
    return sorted(out, key=lambda p: (p["reward_mae"], -p["clean_ranked_above_defects"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--targets", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8, help="concurrent requests per model")
    parser.add_argument("--models", help="comma-separated routes to restrict the sweep")
    parser.add_argument("--output", type=Path, default=Path("image-text-gen-rl/artifacts/calibration.json"))
    args = parser.parse_args()

    from huggingface_hub import get_token

    token = get_token()
    load_prices(token)
    candidates = [c for c in CANDIDATES if not args.models or c.route in args.models.split(",")]
    items = build_images(pick_targets(configured_catalog(), args.targets, args.seed), args.seed)
    print(f"{len(items)} images x {len(candidates)} models", flush=True)
    per_model, summaries = {}, []
    with ThreadPoolExecutor(len(candidates)) as pool:
        futures = {pool.submit(run_model, spec, items, token, args.workers): spec for spec in candidates}
        for future in as_completed(futures):
            spec = futures[future]
            per_model[spec.route] = future.result()
            summaries.append(summarise(spec, per_model[spec.route]))
            print(json.dumps(summaries[-1]), flush=True)
    usable = [s["route"] for s in summaries if s["answered"] >= 0.9 * len(items)]
    pairs = pair_scores(items, per_model, usable)
    summaries.sort(key=lambda s: (s["reward_mae"] is None, s["reward_mae"] or 0))
    report = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "images": len(items),
        "targets": args.targets,
        "variants": list(VARIANTS),
        "models": summaries,
        "pairs": pairs[:15],
        "readings": [
            {"task_id": it["task_id"], "variant": it["variant"], "target": it["target"],
             "expected": it["expected"], "font": it["font"],
             "by_model": {route: {k: v for k, v in rows[i].items()
                                  if k in ("text", "error", "reward", "latency_s")}
                          for route, rows in per_model.items()}}
            for i, it in enumerate(items)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print("\nTOP PAIRS")
    for pair in pairs[:8]:
        print(json.dumps(pair))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
