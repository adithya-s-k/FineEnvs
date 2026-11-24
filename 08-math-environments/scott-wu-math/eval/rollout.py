"""Run language models on Scott Wu Math through the OpenEnv server, all in parallel.

Every episode opens its own environment session: reset deals the task, the model reads the
observation's prompt, and step submits the extracted answer, so the server's grader scores it.

    # the two hands from the video, three samples each, every model, reasoning on
    python eval/rollout.py --server http://localhost:8012 --split video --samples 3 --mode think --out runs/video-think

    # a mini board: the first five test tasks of every level
    python eval/rollout.py --split test --per-level 5 --mode nothink --out runs/test20-nothink

Keys come from the nearest .env: OPENAI_API_KEY, ANTHROPIC_API_KEY, HF_TOKEN (only those the chosen models use).
Results: <out>/episodes.jsonl (one line per episode, written as they finish) and <out>/summary.md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "envs"))
from scott_wu_math import CardAction, ScottWuMathEnv  # noqa: E402
from scott_wu_math.game.reward import extract_answer  # noqa: E402

MAX_TOKENS = 32_000
DIRECT_TOKENS = 300
DIRECT_RULE = (
    "Answer at once, the way a person does it in their head: reply with only the <answer></answer> tag, "
    "no working, no explanation."
)
TIMEOUT_S = 600
LIMITS = {"openai": 12, "anthropic": 12, "hf": 6}


def load_env_file() -> None:
    """Load the nearest .env above this file (the workspace keeps its keys there)."""
    from dotenv import load_dotenv

    for parent in HERE.parents:
        if (parent / ".env").is_file():
            load_dotenv(parent / ".env", override=False)
            return


# ---- model calls: each returns text, reasoning (summary or raw, if the API gives it), usage ----


async def call_openai(client, model: dict, setting: dict, system: str, user: str) -> dict:
    reasoning = {"effort": setting["effort"]}
    if setting["effort"] != "none":
        reasoning["summary"] = "auto"
    resp = await client.responses.create(
        model=model["id"],
        instructions=system,
        input=user,
        reasoning=reasoning,
        max_output_tokens=setting.get("max_tokens", MAX_TOKENS),
    )
    summary = []
    for item in resp.output:
        if item.type == "reasoning":
            summary += [s.text for s in (item.summary or [])]
    u = resp.usage
    details = getattr(u, "output_tokens_details", None)
    return {
        "text": resp.output_text or "",
        "reasoning": "\n\n".join(summary),
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
        "reasoning_tokens": getattr(details, "reasoning_tokens", None),
        "stop": resp.status,
    }


async def call_anthropic(client, model: dict, setting: dict, system: str, user: str) -> dict:
    kind = setting.get("thinking", "adaptive")
    thinking = {"type": "adaptive", "display": "summarized"} if kind == "adaptive" else {"type": kind}
    async with client.messages.stream(
        model=model["id"],
        max_tokens=setting.get("max_tokens", MAX_TOKENS),
        system=system,
        messages=[{"role": "user", "content": user}],
        thinking=thinking,
        output_config={"effort": setting["effort"]},
    ) as stream:
        msg = await stream.get_final_message()
    text = "".join(b.text for b in msg.content if b.type == "text")
    reasoning = "\n\n".join(b.thinking for b in msg.content if b.type == "thinking" and b.thinking)
    out = {
        "text": text,
        "reasoning": reasoning,
        "input_tokens": msg.usage.input_tokens,
        "output_tokens": msg.usage.output_tokens,
        "reasoning_tokens": None,
        "stop": msg.stop_reason,
    }
    if msg.stop_reason == "refusal" and getattr(msg, "stop_details", None):
        out["refusal_category"] = getattr(msg.stop_details, "category", None)
    return out


async def call_hf(client, model: dict, setting: dict, system: str, user: str) -> dict:
    extra: dict = {}
    kwargs: dict = {}
    if "enable_thinking" in setting:
        extra["chat_template_kwargs"] = {"enable_thinking": setting["enable_thinking"]}
    if "reasoning_effort" in setting:
        kwargs["reasoning_effort"] = setting["reasoning_effort"]
    resp = await client.chat.completions.create(
        model=model["id"],
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=setting.get("max_tokens", MAX_TOKENS),
        extra_body=extra or None,
        **kwargs,
    )
    msg = resp.choices[0].message
    extra_fields = msg.model_extra or {}
    reasoning = extra_fields.get("reasoning_content") or extra_fields.get("reasoning") or ""
    u = resp.usage
    details = getattr(u, "completion_tokens_details", None)
    return {
        "text": msg.content or "",
        "reasoning": reasoning if isinstance(reasoning, str) else json.dumps(reasoning),
        "input_tokens": getattr(u, "prompt_tokens", None),
        "output_tokens": getattr(u, "completion_tokens", None),
        "reasoning_tokens": getattr(details, "reasoning_tokens", None) if details else None,
        "stop": resp.choices[0].finish_reason,
    }


CALLS = {"openai": call_openai, "anthropic": call_anthropic, "hf": call_hf}


def make_clients(apis: set[str]) -> dict:
    """One client per API the chosen models use, so a run of OpenAI models needs no HF_TOKEN or `anthropic`."""
    clients: dict = {}
    if apis & {"openai", "hf"}:
        from openai import AsyncOpenAI
    if "openai" in apis:
        clients["openai"] = AsyncOpenAI(timeout=TIMEOUT_S, max_retries=2)
    if "anthropic" in apis:
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            sys.exit("the Anthropic models need the anthropic package: pip install -e 'envs/scott_wu_math[eval]'")
        clients["anthropic"] = AsyncAnthropic(timeout=TIMEOUT_S, max_retries=2)
    if "hf" in apis:
        token = os.environ.get("HF_TOKEN")
        if not token:
            sys.exit("the open models run on Hugging Face Inference Providers and need HF_TOKEN")
        clients["hf"] = AsyncOpenAI(
            base_url="https://router.huggingface.co/v1", api_key=token, timeout=TIMEOUT_S, max_retries=2
        )
    return clients


# ---- one episode -------------------------------------------------------------------------------


async def episode(server: str, model: dict, mode: str, split: str, index: int, sample: int, clients, sems) -> dict:
    # "direct" reuses the no-thinking settings, adds DIRECT_RULE and caps the reply at DIRECT_TOKENS.
    setting = dict(model["nothink" if mode == "direct" else mode])
    if mode == "direct":
        setting["max_tokens"] = DIRECT_TOKENS
    record = {
        "model": model["id"],
        "label": model["label"],
        "org": model["org"],
        "open": model["open"],
        "mode": mode,
        "floor": bool(setting.get("floor")),
        "split": split,
        "index": index,
        "sample": sample,
    }
    async with sems[model["api"]], ScottWuMathEnv(base_url=server) as env:
        reset = await env.reset(split=split, index=index)
        obs = reset.observation
        record.update(task_id=obs.task_id, level=obs.level, target=obs.target, cards=[c["label"] for c in obs.cards])
        started = time.monotonic()
        try:
            system = obs.system_prompt + ("\n\n" + DIRECT_RULE if mode == "direct" else "")
            out = await CALLS[model["api"]](clients[model["api"]], model, setting, system, obs.prompt)
            record.update(out)
        except Exception as exc:  # recorded, not raised: one provider failing must not stop the board
            record.update(text="", reasoning="", api_error=f"{type(exc).__name__}: {exc}"[:500])
        record["latency_s"] = round(time.monotonic() - started, 2)
        answer = extract_answer(record.get("text", ""))
        record["answer"] = answer
        result = await env.step(CardAction(expression=answer or ""))
        record["reward"] = result.reward
        record["result"] = result.observation.result
        record["solution"] = result.observation.solution
    return record


def task_indices(server: str, split: str, per_level: int | None, limit: int | None) -> list[int]:
    import httpx

    tasks = httpx.post(f"{server}/scott_wu_math/tasks", json={"split": split}, timeout=60).json()["tasks"]
    if per_level:
        seen: dict[str, int] = defaultdict(int)
        picked = []
        for i, t in enumerate(tasks):
            if seen[t["level"]] < per_level:
                seen[t["level"]] += 1
                picked.append(i)
        return picked
    return list(range(len(tasks)))[: limit or None]


def summarize(records: list[dict], out: Path) -> None:
    rows = defaultdict(list)
    for r in records:
        rows[(r["label"], r["mode"], r["floor"])].append(r)
    lines = [
        "| Model | Mode | Solved | Mean reward | Valid | Median time (s) | Mean output tokens | Errors |",
        "|---|---|---|---|---|---|---|---|",
    ]
    order = sorted(rows.items(), key=lambda kv: (-sum(1 for r in kv[1] if r["result"] and r["result"].get("exact")), kv[0]))
    for (label, mode, floor), rs in order:
        solved = sum(1 for r in rs if r["result"] and r["result"].get("exact"))
        valid = sum(1 for r in rs if r["result"] and r["result"].get("valid"))
        reward = sum(r["reward"] or 0 for r in rs) / len(rs)
        lat = sorted(r["latency_s"] for r in rs)[len(rs) // 2]
        toks = [r.get("output_tokens") for r in rs if r.get("output_tokens")]
        errors = sum(1 for r in rs if r.get("api_error"))
        mode_text = mode + (" (lowest effort)" if floor else "")
        lines.append(
            f"| {label} | {mode_text} | {solved}/{len(rs)} | {reward:.3f} | {valid}/{len(rs)} | {lat:.1f} | "
            f"{(sum(toks) / len(toks)) if toks else 0:,.0f} | {errors} |"
        )
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="http://localhost:8012")
    ap.add_argument("--models", default=str(HERE / "models.json"))
    ap.add_argument("--only", nargs="*", help="model ids or labels to run (default: all)")
    ap.add_argument("--mode", choices=["think", "nothink", "direct"], default="think")
    ap.add_argument("--split", default="video")
    ap.add_argument("--per-level", type=int, help="first N tasks of each level")
    ap.add_argument("--limit", type=int, help="first N tasks of the split")
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    load_env_file()
    models = json.loads(Path(args.models).read_text())["models"]
    if args.only:
        models = [m for m in models if m["id"] in args.only or m["label"] in args.only]
    indices = task_indices(args.server, args.split, args.per_level, args.limit)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    clients = make_clients({m["api"] for m in models})
    sems = {k: asyncio.Semaphore(v) for k, v in LIMITS.items()}
    # HF models each get their own limit, so one slow provider does not starve the others.
    hf_sems = {m["id"]: asyncio.Semaphore(LIMITS["hf"]) for m in models if m["api"] == "hf"}

    def limits(m: dict) -> dict:
        return sems | {"hf": hf_sems[m["id"]]} if m["api"] == "hf" else sems

    # Resume: skip episodes already in episodes.jsonl (failed ones were never written). The key has the
    # mode and split too, so reusing an --out directory for another mode or split runs every episode.
    done_path = out / "episodes.jsonl"
    done = set()
    if done_path.is_file():
        for line in done_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["model"], r["mode"], r["split"], r["index"], r["sample"]))
    jobs = [
        episode(args.server, m, args.mode, args.split, i, s, clients, limits(m))
        for m in models
        for i in indices
        for s in range(args.samples)
        if (m["id"], args.mode, args.split, i, s) not in done
    ]
    print(f"{len(jobs)} episodes: {len(models)} models x {len(indices)} tasks x {args.samples} samples, mode {args.mode}")
    meta = {"server": args.server, "split": args.split, "indices": indices, "samples": args.samples, "mode": args.mode, "started": time.time()}
    (out / "run.json").write_text(json.dumps(meta, indent=2))
    records = [json.loads(line) for line in done_path.read_text().splitlines()] if done_path.is_file() else []
    records = [r for r in records if r["mode"] == args.mode and r["split"] == args.split]
    with done_path.open("a") as f:
        for coro in asyncio.as_completed(jobs):
            try:
                r = await coro
            except Exception as exc:
                print(f"episode failed: {type(exc).__name__}: {exc}")
                continue
            records.append(r)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            mark = "✓" if r["result"] and r["result"].get("exact") else ("·" if not r.get("api_error") else "!")
            print(f"{mark} {r['label']:<18} {r['task_id']:<22} {r['latency_s']:>6.1f}s  {r.get('answer') or r.get('api_error', '')[:80]}")
    summarize(records, out)


if __name__ == "__main__":
    asyncio.run(main())
