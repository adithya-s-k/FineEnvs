"""Pass@1 with durable per-pair results and retries only for ungraded infrastructure failures."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
from pathlib import Path
import time
import uuid

from recipe import digest, summary, task_rows, write_json
from train.adapters import Factory


def capture_metrics(result):
    turns = [t for t in result.turns if t.trainable]
    if not turns:
        raise ValueError("No trainable capture from the token-enabled engine")
    for turn in turns:
        prompt, tokens, logps = turn.prompt_token_ids, turn.completion_token_ids, turn.per_token_logps
        mask = turn.loss_mask
        if not prompt or not tokens or len(tokens) != len(logps) or not all(math.isfinite(x) for x in logps):
            raise ValueError("Invalid captured token/logprob alignment")
        if mask is None or len(mask) != len(prompt) + len(tokens) or any(mask[:len(prompt)]):
            raise ValueError("Invalid full-sequence loss mask")
        if set(mask) - {0, 1} or not any(mask[len(prompt):]):
            raise ValueError("No valid supervised completion span")
    return {"tito_pass": True, "model_calls": len(turns),
            "generated_tokens": sum(len(t.completion_token_ids) for t in turns),
            "prompt_tokens": sum(len(t.prompt_token_ids) for t in turns)}


def blackbox_episode(factory, group_id):
    group = factory.groups[group_id]
    session = factory.create(factory.rows()[group_id]["prompt"], seed=group_id, episode_id=uuid.uuid4().hex)
    try:
        session.wait_for_completion(timeout_s=factory.cfg["agent_timeout_sec"] + 600)
        session.verify([])
        record = json.loads((Path(factory.cfg["output"]) / "rollouts" / f'{session.evidence["episode_id"]}.json').read_text())
        return {**record, **capture_metrics(session.result)}
    finally:
        session.close()


def whitebox_episode(cfg, server, vllm, index):
    from openai import OpenAI
    from transformers.utils import get_json_schema
    from whitebox_bash import white_box_bash_env
    import inspect
    from train.whitebox import SYSTEM

    env = white_box_bash_env(server, split="test", toolsets="bash,seta", step_limit=cfg["agent_step_limit"])()
    client = OpenAI(base_url=vllm.rstrip("/") + "/v1", api_key="unused", timeout=cfg["agent_timeout_sec"])
    calls, generated, inputs = 0, 0, 0
    try:
        text = env.reset(split="test", index=index)
        tools = [get_json_schema(m) for name, m in inspect.getmembers(env, inspect.ismethod)
                 if not name.startswith("_") and name not in {"reset", "get_reward"}]
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]
        for _ in range(cfg["agent_step_limit"]):
            response = client.chat.completions.create(model=cfg["profile"]["id"], messages=messages, tools=tools,
                temperature=cfg["temperature"], top_p=1.0, max_tokens=cfg["max_output_tokens"],
                extra_body={"chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": True}})
            generated += response.usage.completion_tokens
            inputs += response.usage.prompt_tokens
            msg = response.choices[0].message
            messages.append(msg.model_dump(exclude_none=True))
            if not msg.tool_calls:
                break
            submitted = False
            for call in msg.tool_calls:
                allowed = {t["function"]["name"] for t in tools}
                if call.function.name not in allowed:
                    raise ValueError("Model called an unexposed tool")
                value = str(getattr(env, call.function.name)(**json.loads(call.function.arguments)))
                calls += call.function.name != "submit_solution"
                submitted |= call.function.name == "submit_solution"
                messages.append({"role": "tool", "tool_call_id": call.id, "content": value})
            if submitted:
                break
        correctness = env.get_reward()
        if not math.isfinite(correctness):
            raise ValueError("Ungraded whitebox episode")
        return {"correctness": float(correctness >= 1), "tool_calls": calls,
                "generated_tokens": generated, "prompt_tokens": inputs}
    finally:
        try:
            if env._session is not None:
                env.get_reward()
        finally:
            env._mcp.close()
            client.close()


def evaluate(cfg, data, server, vllm, *, limit=None, checkpoint="baseline", max_attempts=3):
    out = Path(cfg["output"]) / "eval"
    out.mkdir(parents=True, exist_ok=True)
    tasks = sorted(task_rows("test"), key=lambda row: row["name"])
    selected = tasks[:limit] if limit else tasks
    harnesses = ["whitebox"] if cfg["mode"] == "whitebox" else cfg["harnesses"]
    groups = [{"task_name": row["name"], "task_index": i, "harness": h}
              for i, row in enumerate(selected) for h in harnesses]
    identity = {"model": cfg["profile"], "checkpoint": str(checkpoint), "tasks": selected,
                "harnesses": harnesses, "template": cfg["profile"]["template"], "config": cfg}
    signature = digest(identity)
    identity_path = out / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text())["sha256"] != signature:
        raise ValueError("Evaluation output belongs to a different checkpoint or protocol")
    write_json(identity_path, {"sha256": signature, **identity})
    # Both blackbox policies use the same four Harbor evaluation harnesses.
    factory = Factory({**cfg, "mode": "multi-harness"}, data, server, vllm, groups,
                      Path(cfg["output"]) / "trials", split="test") if cfg["mode"] != "whitebox" else None

    def one(item):
        group_id, group = item
        row = tasks[group["task_index"]]
        path = out / "pairs" / f'{row["name"]}--{group["harness"]}.json'
        if path.exists():
            prior = json.loads(path.read_text())
            if prior.get("correctness") in (0, 1) and not prior.get("error"):
                return prior
        for attempt in range(max_attempts):
            started = time.time()
            try:
                result = (blackbox_episode(factory, group_id) if factory else
                          whitebox_episode(cfg, server, vllm, group["task_index"]))
                if result.get("correctness") not in (0, 1):
                    raise ValueError("Verifier did not produce a binary grade")
                record = {**group, "difficulty": row["difficulty"], **result, "attempt": attempt + 1,
                          "elapsed_sec": time.time() - started, "error": None}
            except Exception as exc:
                record = {**group, "difficulty": row["difficulty"], "correctness": None,
                          "error": type(exc).__name__, "attempt": attempt + 1}
            write_json(path, record)
            if record["correctness"] is not None:
                return record
        return record

    records = []
    with ThreadPoolExecutor(max_workers=cfg["eval_concurrency"]) as pool:
        for future in as_completed([pool.submit(one, pair) for pair in enumerate(groups)]):
            records.append(future.result())
            write_json(out / "progress.json", summary(records, len(groups)))
    report = summary(records, len(groups))
    report["harnesses"] = {h: summary([r for r in records if r["harness"] == h], len(selected)) for h in harnesses}
    report["difficulty"] = {d: summary([r for r in records if r["difficulty"] == d],
                          sum(t["difficulty"] == d for t in selected) * len(harnesses)) for d in ("easy", "medium", "hard")}
    report["missing"] = [{"task_name": r["task_name"], "harness": r["harness"], "error": r["error"]}
                         for r in records if r["correctness"] is None]
    write_json(out / "summary.json", report)
    import trackio
    run = trackio.init(project=cfg["project"], name=cfg["run_name"] + "-eval",
                       space_id=cfg["trackio_space_id"], config={"protocol_sha256": signature, "checkpoint": str(checkpoint)})
    step = 0
    if checkpoint != "baseline":
        step = json.loads((Path(checkpoint) / "trainer_state.json").read_text())["global_step"]
    metrics = {"eval/coverage": report["coverage"], "eval/graded": report["graded"]}
    if report["pass_at_1"] is not None:
        metrics["eval/pass_at_1_observed"] = report["pass_at_1"]
    for harness, values in report["harnesses"].items():
        for key in ("pass_at_1", "coverage", "mean_tool_calls", "mean_generated_tokens"):
            if values[key] is not None:
                metrics[f"eval/{harness}/{key}"] = values[key]
    run.log(metrics, step=step)
    run.finish()
    print(json.dumps(report, indent=2))
    return report
