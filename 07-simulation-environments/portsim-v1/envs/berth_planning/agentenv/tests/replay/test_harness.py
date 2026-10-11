"""portsim-llm plays five published episodes, one per provider route, back from the fake LiteLLM against this env and
records each as upstream's harness did: the same messages, steps, turns, end and reward, the same token counts, and
the same requests as upstream's harness sends for those turns (golden/harness.json, recorded by the agentenv-portsim
plugin's scripts/record_harness.py)."""

import json
from pathlib import Path

import portsim_llm
import pytest
from agentenv_protocol.a2a_agent import TaskRequest, TextPart
from fake_litellm import FakeLiteLLM, digest, hf_turns

pytestmark = [pytest.mark.anyio, pytest.mark.network]

GOLDEN = json.loads((Path(__file__).resolve().parents[1] / "golden" / "harness.json").read_text())
KEY = "sk-fake-replay-key"
REPLAYS = [
    ("anthropic:claude-sonnet-5-5", "dock-36A-w15x2-storm-0", "anthropic/claude-sonnet-5-5"),
    ("anthropic:claude-sonnet-5-5", "dock-36A-w05x2-storm-0", "anthropic/claude-sonnet-5-5"),
    ("openai:gpt-6.1-sol", "dock-24B-w35x3-extreme-1", "openai/gpt-6.1-sol"),
    ("hf:zai-org/GLM-5.3-Flash:baseten", "dock-24B-w16x2-extreme-0", "fireworks_ai/glm-5p3-flash"),
    ("hf:zai-org/GLM-5.3-Flash:baseten", "dock-24B-w35x2-extreme-0", "fireworks_ai/glm-5p3-flash"),
]


def test_the_golden_covers_the_replays():
    assert GOLDEN.keys() == {f"{spec}|{task_id}" for spec, task_id, _ in REPLAYS}


@pytest.mark.parametrize(("spec", "task_id", "model"), REPLAYS)
async def test_portsim_llm_replays_the_published_episode(harness, episodes, monkeypatch, spec, task_id, model):
    episode = next(e for e in episodes if (e["model"], e["task_id"]) == (spec, task_id))
    messages = json.loads(episode["messages"])
    await harness.load_task(task_id)
    with FakeLiteLLM(hf_turns(episode)) as fake:
        monkeypatch.setenv("LITELLM_BASE_URL", fake.url)
        monkeypatch.setenv("LITELLM_API_KEY", KEY)
        monkeypatch.delenv("PORTSIM_MAX_COST_USD", raising=False)
        result = await portsim_llm.PortSimLLM().run(TaskRequest(
            task_id="replay", context_id="replay", parts=(TextPart(text=messages[1]["content"]),),
            config=portsim_llm.PortSimConfig(model=model, system_prompt=messages[0]["content"], max_turns=12,
                                             model_params={"max_tokens": 32000}, timeout_seconds=7200),
            mcp_servers={"portsim": {"url": f"{harness.base_url}/mcp"}}))
    record, summary = result.native_trajectory.payload, result.parts[1].data["structured_output"]

    assert record["messages"] == messages
    assert record["steps"] == json.loads(episode["steps"])
    assert (record["turns"], record["end_reason"]) == (episode["turns"], episode["end_reason"])
    assert record["final"]["plan"] == (episode["final_plan"] and json.loads(episode["final_plan"]))
    assert ((await harness.data())["reward"] or 0.0) == record["reward"] == episode["reward"]
    assert (summary["input_tokens"], summary["output_tokens"]) == (episode["input_tokens"], episode["output_tokens"])
    assert [{"path": path, "sha256": digest(body)} for path, _, body in fake.requests] == \
        GOLDEN[f"{spec}|{task_id}"]["requests"]
    assert {headers["authorization"] for _, headers, _ in fake.requests} == {f"Bearer {KEY}"}
    assert not any("x-api-key" in headers for _, headers, _ in fake.requests)
    assert KEY not in repr(result)
