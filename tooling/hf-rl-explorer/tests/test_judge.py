"""A Harbor task's variables and model-graded tasks (app/judge.py), on the patterns real datasets use (MiMo's
conversions, skilltrainbench, data-agent, WildClawBench): what each variable gets in a rollout here, in which phase,
and that nothing the agent sees is a key of ours."""

from __future__ import annotations

from app import judge

MIMO_GENERAL = '''
[verifier.env]
GA_JUDGE_KEY = "${HF_TOKEN}"
GA_JUDGE_URL = "${MIMO_JUDGE_URL:-https://router.huggingface.co/v1}"
GA_JUDGE_MODEL = "${MIMO_TEXT_JUDGE:-thinkingmachines/Inkling}"
VERIFY_DETERMINISTIC = "1"
'''
MIMO_WEBDEV = '''
[verifier.env]
WEBDEV_JUDGE_KEY = "${HF_TOKEN}"
WEBDEV_JUDGE_URL = "${MIMO_JUDGE_URL:-https://router.huggingface.co/v1}"
WEBDEV_JUDGE_MODEL = "${MIMO_VISION_JUDGE:-meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8}"
'''
HEALTHBENCH = '''
[verifier.env]
STBENCH_JUDGE_BASE_URL = "${STBENCH_JUDGE_BASE_URL}"
STBENCH_JUDGE_API_KEY = "${STBENCH_JUDGE_API_KEY}"
STBENCH_JUDGE_MODEL = "${STBENCH_JUDGE_MODEL}"
STBENCH_JUDGE_TEMPERATURE = "${STBENCH_JUDGE_TEMPERATURE}"
'''
HLE = '''
[verifier.env]
OPENAI_API_KEY = "${OPENAI_API_KEY}"
OPENAI_BASE_URL = "${OPENAI_BASE_URL:-}"
JUDGE_MODEL = "o3-mini-2025-01-31"
'''
DATA_AGENT = '''
[environment.env]
HF_TOKEN = "${HF_TOKEN}"
'''
NEEDS_SERVICE = '''
[environment.env]
DATABASE_URL = "${DATABASE_URL}"
'''


def test_a_model_graded_task_gets_the_relay_in_its_grading_phase_only():
    p = judge.plan(MIMO_GENERAL)
    assert p["judge"] == {"requested": "thinkingmachines/Inkling", "vision": False, "anthropic": False, "keys": ["GA_JUDGE_KEY", "GA_JUDGE_URL"]}
    assert p["missing"] == [] and p["agent_keys"] == []
    b = judge.bindings(p, relay="https://x.hf.space/capture", capability="jcap", judge="thinkingmachines/Inkling")
    assert b["verifier"]["GA_JUDGE_KEY"] == "jcap" and b["verifier"]["HF_TOKEN"] == "jcap"
    assert b["verifier"]["GA_JUDGE_URL"] == "https://x.hf.space/capture/v1"   # never the real router: the relay holds the token
    assert b["verifier"]["GA_JUDGE_MODEL"] == "thinkingmachines/Inkling"
    assert "jcap" not in b["environment"].values()


def test_a_vision_grader_asks_for_a_vision_judge():
    assert judge.plan(MIMO_WEBDEV)["judge"]["vision"] is True


def test_a_judge_without_defaults_gets_the_picked_model_and_deterministic_settings():
    p = judge.plan(HEALTHBENCH)
    assert p["judge"] and p["missing"] == []
    b = judge.bindings(p, relay="https://r", capability="c", judge="openai/gpt-oss-120b")["verifier"]
    assert b["STBENCH_JUDGE_MODEL"] == "openai/gpt-oss-120b" and b["STBENCH_JUDGE_TEMPERATURE"] == "0"
    assert b["STBENCH_JUDGE_BASE_URL"] == "https://r/v1" and b["STBENCH_JUDGE_API_KEY"] == "c"


def test_an_openai_client_with_an_empty_default_url_is_pointed_at_the_relay():
    p = judge.plan(HLE)
    assert p["judge"]["requested"] == "o3-mini-2025-01-31"
    b = judge.bindings(p, relay="https://r", capability="c", judge="deepseek-ai/DeepSeek-V4-Pro")["verifier"]
    assert b["OPENAI_BASE_URL"] == "https://r/v1" and b["OPENAI_API_KEY"] == "c"   # not api.openai.com with our key


def test_a_key_the_agent_would_see_is_left_empty():
    p = judge.plan(DATA_AGENT)
    assert p["judge"] is None and p["agent_keys"] == ["HF_TOKEN"] and p["missing"] == []
    assert judge.bindings(p)["environment"]["HF_TOKEN"] == ""


def test_a_variable_nothing_here_provides_stops_the_task():
    p = judge.plan(NEEDS_SERVICE)
    assert p["missing"] == ["DATABASE_URL"]


def test_without_a_judge_nothing_is_bound_to_a_relay():
    b = judge.bindings(judge.plan(MIMO_GENERAL))
    assert b["verifier"]["GA_JUDGE_KEY"] == "" and "GA_JUDGE_URL" not in b["verifier"]   # no capability: an empty key, the task's defaults


def test_the_table_says_how_each_variable_is_filled():
    rows = dict(judge.table(judge.plan(MIMO_GENERAL)))
    assert any("judge relay" in v for v in rows.values())
    assert not any("hf_" in v for v in rows.values())


def test_a_broken_toml_is_not_a_crash():
    p = judge.plan("this is = = not toml")
    assert p["invalid"] and p["vars"] == []
