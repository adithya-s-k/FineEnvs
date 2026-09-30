import json
from collections import Counter
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from recipe import config, digest, reward, resume_state, schedule, summary, task_rows, write_json
from train.adapters import atif_count, opencode_count
from train.generation import group_size, protect_generation
from eval.evaluate import capture_metrics
from runtime.checkpoints import make_ready, saved, verify
from runtime.launch import stage
from runtime.models import check_visible_response


def test_ampere_attention_keeps_packed_flash_attention(tmp_path):
    from runtime.models import training_attention_backend
    from runtime.patches import apply_trl
    assert training_attention_backend((8, 0)) == "kernels-community/flash-attn2"
    assert training_attention_backend((9, 0)) == "kernels-community/flash-attn3"
    with pytest.raises(ValueError, match="Unqualified"):
        training_attention_backend((7, 0))
    path = tmp_path / "trl/experimental/async_grpo/async_grpo_trainer.py"
    path.parent.mkdir(parents=True)
    path.write_text('model = create_model_from_path(model, attn_implementation="kernels-community/flash-attn3", **model_init_kwargs)')
    apply_trl(tmp_path)
    apply_trl(tmp_path)
    calls = []
    kwargs = {"attn_implementation": training_attention_backend((8, 0)), "revision": "fixed"}
    exec(path.read_text(), {"create_model_from_path": lambda *a, **k: calls.append(k),
                            "model": "test", "model_init_kwargs": kwargs})
    assert calls == [{"attn_implementation": "kernels-community/flash-attn2", "revision": "fixed"}]


def test_frozen_data_and_rotation():
    train, test = task_rows("train"), task_rows("test")
    assert len(train) == 1000 and len(test) == 250
    assert Counter(r["difficulty"] for r in train) == {"medium": 400, "hard": 600}
    assert Counter(r["difficulty"] for r in test) == {"easy": 33, "medium": 118, "hard": 99}
    for key in ("source", "notebook", "instruction_sha256"):
        assert not {r[key] for r in train} & {r[key] for r in test}
    groups = schedule(config(), train)
    assert len(groups) == 2000
    assert Counter(g["harness"] for g in groups[:1000]) == {h: 250 for h in config()["harnesses"]}
    for a, b in zip(groups[:1000], groups[1000:], strict=True):
        assert a["task_name"] == b["task_name"] and a["harness"] != b["harness"]


def test_native_means_native():
    assert {g["harness"] for g in schedule(config(mode="opencode"), task_rows("train"))} == {"opencode"}


def test_nonthinking_preflight_checks_streaming_content(monkeypatch):
    hidden = False
    def create(*, stream, **kwargs):
        bad = stream and hidden
        message = SimpleNamespace(content="" if bad else "OK", model_extra={"reasoning": "OK" if bad else None})
        choice = SimpleNamespace(delta=message, message=message)
        response = SimpleNamespace(choices=[choice])
        return [response] if stream else response
    class Client:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))
        def __enter__(self): return self
        def __exit__(self, *args): pass
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=Client))
    assert len(check_visible_response(config(), "http://localhost")) == 2
    hidden = True
    with pytest.raises(ValueError, match="stream=True"):
        check_visible_response(config(), "http://localhost")


@pytest.mark.parametrize("kwargs", [{"eval_steps": 75}, {"num_generations": 1}, {"top_p": .95},
                                  {"max_outstanding_rollouts": 8}, {"eval_concurrency": 0}])
def test_reject_invalid_config(kwargs):
    with pytest.raises(ValueError):
        config(**kwargs)


def test_resume_skips_committed_groups_and_retries_uncommitted():
    groups = schedule(config(), task_rows("train"))
    state = {"execution_schedule_sha256": digest(groups), "settled_groups": [0, 1, 2],
             "admitted_rollouts": {"a": 0, "b": 2}}
    assert resume_state(groups, state)["settled_groups"] == [0, 2]
    with pytest.raises(ValueError):
        resume_state(groups[::-1], state)


def test_reward_does_not_pay_for_failure_or_unknown_counts():
    assert reward(None, 2, verified=True) is None
    assert reward(0, 1, verified=True) == 0
    assert reward(1, None, verified=False) == 1
    assert reward(1, 0, verified=True) == 1
    assert reward(1, 15, verified=True) == 1.05
    assert reward(1, 2, verified=True) > reward(1, 20, verified=True)


def test_native_counts_require_unambiguous_actions(tmp_path):
    path = tmp_path / "trajectory.json"
    write_json(path, {"schema_version": "ATIF-v1.6", "steps": [
        {"source": "agent", "tool_calls": [{"tool_call_id": "a"}, {"tool_call_id": "b"}]},
        {"source": "environment", "tool_calls": [{"tool_call_id": "a"}]}]})
    assert atif_count(path) == 2
    event = {"type": "tool_use", "part": {"callID": "a", "state": {"status": "completed"}}}
    stream = json.dumps(event) + '\n{"type":"step_finish"}'
    assert opencode_count(stream) == 1
    with pytest.raises(ValueError):
        opencode_count(stream + "\n" + json.dumps(event))
    with pytest.raises(ValueError):
        opencode_count(json.dumps(event))


def test_diverged_tool_histories_are_never_collapsed():
    assert group_size([[1], [1], [2], [2]], 2) == 2
    assert group_size([[1], [3], [2], [2]], 2) == 1
    assert group_size([[1], [1], [2]], 2) == 1


def test_whitebox_cache_reset_accepts_empty_success_and_rejects_failure():
    import requests
    response = requests.Response()
    response.status_code = 200
    response._content = b""
    client = SimpleNamespace(base_url="http://localhost", session=SimpleNamespace(post=lambda url: response))
    trainer = SimpleNamespace(vllm_generation=SimpleNamespace(vllm_client=client, generate=lambda **kw: None))
    protect_generation(trainer)
    client.reset_prefix_cache()
    response.status_code = 500
    with pytest.raises(requests.HTTPError):
        client.reset_prefix_cache()


def test_capture_uses_engine_tokens_and_masks():
    turn = SimpleNamespace(trainable=True, prompt_token_ids=[10, 20], completion_token_ids=[30],
                           per_token_logps=[-.5], loss_mask=[0, 0, 1])
    result = SimpleNamespace(turns=[turn])
    assert capture_metrics(result)["generated_tokens"] == 1
    turn.loss_mask = [1, 0, 1]
    with pytest.raises(ValueError):
        capture_metrics(result)
    turn.loss_mask = [0, 0, 1]
    turn.per_token_logps = [float("nan")]
    with pytest.raises(ValueError):
        capture_metrics(result)


def test_ungraded_is_not_incorrect():
    result = summary([{"correctness": 1}, {"correctness": 0}, {"correctness": None, "error": "transport"}], 3)
    assert result["pass_at_1"] == .5 and result["graded"] == 2 and not result["complete"]
    assert summary([], 1)["pass_at_1"] is None


def test_checkpoint_transfer_is_portable_and_detects_tampering(tmp_path):
    checkpoint = tmp_path / "original"
    checkpoint.mkdir()
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json", "training_args.bin",
                 "optimizer.pt", "scheduler.pt", "rng_state.pth", "model.safetensors"):
        (checkpoint / name).write_bytes(b"fixture")
    write_json(checkpoint / "trainer_state.json", {"global_step": 50})
    saved(checkpoint, config(), 50)
    make_ready(checkpoint)
    moved = tmp_path / "different-root" / "checkpoint-50"
    shutil.copytree(checkpoint, moved)
    assert verify(moved, resume=True)["step"] == 50
    (moved / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError):
        verify(moved)


def test_hf_source_bundle_omits_local_state(tmp_path):
    stage(tmp_path)
    paths = [str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*")]
    assert "configs/runtime-lock.json" in paths
    assert not any(p.startswith((".env", ".runtime", "runs", "prepared")) for p in paths)
