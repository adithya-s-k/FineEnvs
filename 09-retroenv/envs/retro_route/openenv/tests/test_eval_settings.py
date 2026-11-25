"""Non-thinking and no-repair settings the evaluation harness sends to providers."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

from retroenv_openenv import agent_responses
from retroenv_openenv.agent import normalize_arguments
from retroenv_openenv.agent_anthropic import thinking_off

EVAL = Path(__file__).resolve().parents[4] / "eval"


def _script(name: str):
    """Import an eval/ script the way it runs: as a top-level module beside run_eval."""
    if str(EVAL) not in sys.path:
        sys.path.insert(0, str(EVAL))
    return importlib.import_module(name)


def test_claude_models_turn_thinking_off_the_way_each_accepts():
    assert thinking_off("claude-haiku-5-5") == ({"type": "disabled"}, None, True)
    assert thinking_off("claude-sonnet-5-5") == ({"type": "between_tools"}, None, True)
    # Opus 5.5 and Fable 5.1 reject disabled thinking; low effort is their floor and is labelled so.
    assert thinking_off("claude-opus-5-5") == (None, "low", False)
    assert thinking_off("claude-fable-5-1") == (None, "low", False)


def test_without_repair_a_broken_submission_reaches_the_verifier_unchanged():
    broken = '{"routes": [{"type": "mol", "smiles": "CCO", "children": []}]]'  # one stray bracket
    repaired, coerced = normalize_arguments("emit_routes", {"submission": broken})
    assert coerced and isinstance(repaired["submission"], dict)
    untouched, coerced = normalize_arguments("emit_routes", {"submission": broken}, repair=False)
    assert not coerced and untouched["submission"] == broken
    valid = json.dumps({"routes": []})
    assert normalize_arguments("emit_routes", {"submission": valid}, repair=False)[0]["submission"] == valid


def test_openai_models_without_a_none_effort_run_at_their_floor():
    assert agent_responses.thinking_off("gpt-6-sol") == "none"
    assert agent_responses.thinking_off("gpt-6-luna") == "none"
    assert agent_responses.thinking_off("gpt-6.1-sol") == "low"
    assert agent_responses.thinking_off("gpt-6-astra") == "low"


def test_a_model_id_names_its_provider():
    run_eval = _script("run_eval")
    assert run_eval.infer_provider("claude-opus-5-5") == "anthropic"
    assert run_eval.infer_provider("gpt-6.1-sol") == "openai"
    assert run_eval.infer_provider("Qwen/Qwen3.8-27B:novita") == "hf"
    assert run_eval.infer_provider("my-local-model") is None


def test_the_core30_board_resolves_to_thirty_final_eval_tasks_and_known_providers():
    evaluate = _script("evaluate")
    split, task_ids = evaluate.resolve_set("core30")
    assert split == "final_eval" and len(task_ids) == len(set(task_ids)) == 30
    assert evaluate.resolve_set("dev") == ("dev", None)
    board = json.loads((EVAL / "boards" / "core30-nothink.json").read_text())
    models = [entry if isinstance(entry, str) else entry["model"] for entry in board["models"]]
    assert len(models) == 13 and all(evaluate.infer_provider(model) for model in models)
