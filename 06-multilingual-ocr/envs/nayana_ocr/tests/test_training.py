import importlib.util
from collections import Counter
from pathlib import Path

import pytest
from nayana_ocr.training import balanced_rows, env_reward


def test_balance_is_independent_of_discovery_order_and_rejects_missing_groups():
    rows = [
        {"task_id": f"{lang}-{family}-{i}", "language": lang, "family": family}
        for lang in ("en", "kn")
        for family in ("section_ocr", "mcq_vqa")
        for i in range(7 if lang == "en" else 3)
    ]
    selected = balanced_rows(rows, ["en", "kn"], ["section_ocr", "mcq_vqa"])
    assert len(selected) == 12
    assert selected == balanced_rows(
        reversed(rows), ["en", "kn"], ["section_ocr", "mcq_vqa"]
    )
    assert set(Counter((r["language"], r["family"]) for r in selected).values()) == {3}
    with pytest.raises(ValueError, match="Missing language/task groups"):
        balanced_rows(rows, ["ar"], ["section_ocr"])


def test_reward_rejects_misrouted_group_before_scoring():
    class WrongTask:
        task_id = "different"
        requested_task_id = "different"

    with pytest.raises(RuntimeError, match="wrong rollout task"):
        env_reward(["B"], [WrongTask()], ["requested"])


def test_reward_routes_on_the_requested_task_not_the_substituted_one():
    # A spare stood in for an unrenderable page: the rollout ran a different task than
    # the sampler named, and the reward still belongs to the row that was asked for.
    class Substituted:
        task_id = "spare"
        requested_task_id = "requested"

    with pytest.raises(RuntimeError, match="wrong rollout task"):
        env_reward(["B"], [Substituted()], ["somebody-else"])


class FakeClient:
    """Reset fails for pages the pixel guard rejects, as the real server does."""

    def __init__(self, unrenderable, snapshot_id="snap"):
        self.unrenderable = set(unrenderable)
        self.snapshot_id = snapshot_id
        self.attempts = []

    def reset(self, task_id):
        self.attempts.append(task_id)
        if task_id in self.unrenderable:
            raise RuntimeError(
                f"Server error: {task_id}: 7016x9934 exceeds max_pixels=50000000"
            )
        observation = type(
            "Observation",
            (),
            {
                "task_id": task_id,
                "snapshot_id": self.snapshot_id,
                "prompt": f"prompt for {task_id}",
                "asset_sha256": "0" * 64,
            },
        )()
        return type("Result", (), {"observation": observation})()

    def close(self):
        pass


def environment_with(client):
    from nayana_ocr.training import TrainingEnvironment

    environment = TrainingEnvironment.__new__(TrainingEnvironment)
    environment.client = client
    environment.cache = type("Cache", (), {"image": staticmethod(lambda o: "image")})()
    environment.snapshot_id = client.snapshot_id
    environment.task_id = None
    environment.requested_task_id = None
    environment.substitutions = 0
    return environment


def test_reset_substitutes_a_spare_for_an_unrenderable_page():
    client = FakeClient(unrenderable={"bad"})
    environment = environment_with(client)
    content = environment.reset("bad", spare_task_ids=["also-bad-later", "good"])
    assert client.attempts == ["bad", "also-bad-later"]
    assert environment.task_id == "also-bad-later"
    assert environment.requested_task_id == "bad"
    assert environment.substitutions == 1
    assert content[1]["text"] == "prompt for also-bad-later"


def test_reset_prefers_the_requested_task_and_records_no_substitution():
    client = FakeClient(unrenderable=set())
    environment = environment_with(client)
    environment.reset("good", spare_task_ids=["spare"])
    assert client.attempts == ["good"]
    assert environment.task_id == environment.requested_task_id == "good"
    assert environment.substitutions == 0


def test_reset_does_not_substitute_for_an_unrelated_failure():
    class Broken(FakeClient):
        def reset(self, task_id):
            self.attempts.append(task_id)
            raise RuntimeError("Server error: judge unreachable")

    client = Broken(unrenderable=set())
    environment = environment_with(client)
    with pytest.raises(RuntimeError, match="judge unreachable"):
        environment.reset("task", spare_task_ids=["spare"])
    assert client.attempts == ["task"]


def test_reset_raises_when_every_candidate_is_unrenderable():
    client = FakeClient(unrenderable={"a", "b"})
    environment = environment_with(client)
    with pytest.raises(RuntimeError, match="exceeds max_pixels"):
        environment.reset("a", spare_task_ids=["b"])


@pytest.mark.parametrize("mode", ["map", "iterable"])
def test_installed_trl_repeats_same_task_within_each_grpo_group(tmp_path, mode):
    path = Path(__file__).resolve().parents[3] / "train" / "grpo_nayana.py"
    spec = importlib.util.spec_from_file_location("grpo_nayana_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [{"prompt": "hello", "task_id": str(i)} for i in range(8)]
    assert_trl_groups(tmp_path, module.build_dataset(rows, mode))


def assert_trl_groups(tmp_path, dataset):
    trl = pytest.importorskip(
        "trl", reason="Install --extra train to verify TRL's CPU sampler"
    )
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    vocab = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3, "hello": 4}
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="[UNK]")),
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
    )
    model = GPT2LMHeadModel(
        GPT2Config(
            n_layer=1,
            n_head=2,
            n_embd=16,
            vocab_size=5,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
    )
    trainer = trl.GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        reward_funcs=lambda completions, **kwargs: [0.0] * len(completions),
        args=trl.GRPOConfig(
            output_dir=str(tmp_path),
            max_steps=2,
            num_generations=2,
            per_device_train_batch_size=4,
            use_cpu=True,
            bf16=False,
            gradient_checkpointing=False,
            report_to="none",
            dataloader_num_workers=0,
            accelerator_config={"dispatch_batches": False},
        ),
    )
    batches = iter(trainer.get_train_dataloader())
    for _ in range(2):
        batch = next(batches)
        assert len(batch) == 4
        assert batch[0]["task_id"] == batch[1]["task_id"]
        assert batch[2]["task_id"] == batch[3]["task_id"]
        assert batch[0]["task_id"] != batch[2]["task_id"]
        # reset() can only substitute if the spares survive TRL's batching alongside
        # the task they stand in for.
        for row in batch:
            assert row["spare_task_ids"]
            assert row["task_id"] not in row["spare_task_ids"]


def test_accumulation_gives_an_optimizer_step_several_distinct_prompts(tmp_path):
    """The whole point of the knob: one step must see more than one task.

    Without accumulation TRL fills the batch with a single prompt's generations, so
    every gradient came from one task. This asserts the config actually widens it.
    """
    trl = pytest.importorskip(
        "trl", reason="Install --extra train to verify TRL's CPU sampler"
    )
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[PAD]": 0, "[BOS]": 1, "[EOS]": 2,
                                              "[UNK]": 3, "hello": 4}, unk_token="[UNK]")),
        pad_token="[PAD]", bos_token="[BOS]", eos_token="[EOS]", unk_token="[UNK]",
    )
    model = GPT2LMHeadModel(GPT2Config(n_layer=1, n_head=2, n_embd=16, vocab_size=5,
                                       bos_token_id=1, eos_token_id=2, pad_token_id=0))
    path = Path(__file__).resolve().parents[3] / "train" / "grpo_nayana.py"
    spec = importlib.util.spec_from_file_location("grpo_nayana_accum", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dataset = module.build_dataset(
        [{"prompt": "hello", "task_id": str(i)} for i in range(32)], "map"
    )
    accumulation = 4
    trainer = trl.GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        reward_funcs=lambda completions, **kwargs: [0.0] * len(completions),
        args=trl.GRPOConfig(
            output_dir=str(tmp_path),
            max_steps=2,
            num_generations=2,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=accumulation,
            use_cpu=True, bf16=False, gradient_checkpointing=False,
            report_to="none", dataloader_num_workers=0,
            accelerator_config={"dispatch_batches": False},
        ),
    )
    assert trainer.args.gradient_accumulation_steps == accumulation
    batches = iter(trainer.get_train_dataloader())
    seen = set()
    for _ in range(accumulation):
        seen.update(row["task_id"] for row in next(batches))
    assert len(seen) == accumulation, (
        f"one optimizer step covered {len(seen)} distinct tasks, expected {accumulation}"
    )


def _eval_module():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[3] / "train" / "eval_vllm.py"
    spec = importlib.util.spec_from_file_location("eval_vllm_watch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_checkpoint_is_complete_only_when_every_adapter_byte_has_arrived(tmp_path):
    """A watcher reads checkpoints while they upload; a partial adapter must not score."""
    import json

    from nayana_ocr.training import READY, checkpoint_complete, mark_checkpoint_ready

    saved = tmp_path / "checkpoint-25"
    saved.mkdir()
    (saved / "adapter_config.json").write_text('{"r": 16}')
    (saved / "adapter_model.safetensors").write_bytes(b"w" * 4096)
    mark_checkpoint_ready(saved, 25)
    ready = json.loads((saved / READY).read_text())
    assert ready["step"] == 25
    copy = tmp_path / "copy"
    copy.mkdir()
    (copy / "adapter_config.json").write_text('{"r": 16}')
    (copy / "adapter_model.safetensors").write_bytes(b"w" * 1024)
    assert not checkpoint_complete(copy, ready)
    (copy / "adapter_model.safetensors").write_bytes(b"x" * 4096)
    assert not checkpoint_complete(copy, ready)
    (copy / "adapter_model.safetensors").write_bytes(b"w" * 4096)
    assert checkpoint_complete(copy, ready)


def test_curve_points_pair_with_base_and_keep_the_benchmark_report():
    """Benchmark crops carry the reward, CER and Sarvam's own CER/WER; all are tracked."""
    module = _eval_module()
    base = {"model": "base", "samples": [
        {"task_id": "a", "reward": 0.4, "char_error_rate": 0.5, "official_cer": 0.4},
        {"task_id": "b", "reward": 0.6, "char_error_rate": 0.3, "official_cer": 0.2},
    ]}
    step = {"model": "step-25", "indic_ocr_bench": {"avg_metrics": {"cer": 0.25, "wer": 0.5}},
            "samples": [
        {"task_id": "a", "reward": 0.5, "char_error_rate": 0.4, "official_cer": 0.3},
        {"task_id": "b", "reward": 0.7, "char_error_rate": 0.2, "official_cer": 0.2},
    ]}
    point = module.curve_point(25, step, base)
    assert abs(point["reward_change"]["delta"] - 0.1) < 1e-9
    assert abs(point["official_cer_change"]["delta"] + 0.05) < 1e-9
    assert point["sarvam_cer"] == 0.25 and point["sarvam_wer"] == 0.5
    assert module.paired(base["samples"], step["samples"][:1], "reward") is None


def test_only_checkpoints_marked_ready_are_picked_up_in_step_order():
    module = _eval_module()
    paths = [
        "rev/checkpoint-50/ready.json",
        "rev/checkpoint-25/ready.json",
        "rev/checkpoint-75/adapter_model.safetensors",
        "rev/evals/checkpoints/checkpoint-25/ready.json",
    ]
    assert module.checkpoint_steps(paths, "rev") == [25, 50]
