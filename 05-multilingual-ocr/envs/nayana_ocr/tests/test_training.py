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

    with pytest.raises(RuntimeError, match="wrong rollout task"):
        env_reward(["B"], [WrongTask()], ["requested"])


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
