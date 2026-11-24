"""Optional real TRL checks: tiny random CPU models, no downloaded weights or sandboxes."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("trl", minversion="1.12.0")

import torch
from datasets import Dataset
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer

from test_smoldataenvs_scripts import load_script


def tiny_policy():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    vocabulary = {
        "<pad>": 0,
        "<eos>": 1,
        "<unk>": 2,
        "q": 3,
        "a": 4,
        "b": 5,
        "User": 6,
        "Assistant": 7,
        ":": 8,
    }
    raw = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>"
    )
    tokenizer.chat_template = (
        '{% for message in messages %}{{message["role"] + ": " + message["content"] + "\\n"}}'
        '{% endfor %}{% if add_generation_prompt %}{{"assistant: "}}{% endif %}'
    )
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(vocabulary),
            n_positions=128,
            n_embd=16,
            n_layer=1,
            n_head=2,
            bos_token_id=1,
            eos_token_id=1,
            pad_token_id=0,
        )
    )
    model.config._name_or_path = "offline-tiny"
    return model, tokenizer


def test_real_grpo_excludes_ungraded_sandbox_and_updates_weights(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    training = load_script("train_grpo")
    monkeypatch.setattr(
        training,
        "_grade_batch",
        lambda completions, **columns: [
            {"reward": None, "ran": 0.0},
            {"reward": 1.0, "ran": 1.0},
            {"reward": 0.0, "ran": 1.0},
        ],
    )
    model, tokenizer = tiny_policy()
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=Dataset.from_list(
            [{"prompt": [{"role": "user", "content": "q"}]}] * 3
        ),
        reward_funcs=[training.reward_correct, training.reward_ran],
        args=GRPOConfig(
            output_dir=str(tmp_path / "grpo"),
            use_cpu=True,
            bf16=False,
            fp16=False,
            report_to="none",
            max_steps=1,
            per_device_train_batch_size=3,
            gradient_accumulation_steps=1,
            num_generations=3,
            max_completion_length=8,
            # This check tests the baseline and update; separate backend tests check the cap mask.
            mask_truncated_completions=False,
            scale_rewards="none",
            reward_weights=[1.0, 0.0],
            save_strategy="no",
            learning_rate=1e-3,
            disable_tqdm=True,
        ),
    )
    original_score = trainer._generate_and_score_completions
    advantages = []

    def score(inputs):
        result = original_score(inputs)
        advantages.append(result["advantages"].tolist())
        return result

    monkeypatch.setattr(trainer, "_generate_and_score_completions", score)
    before = model.transformer.wte.weight.detach().clone()
    result = trainer.train()
    assert advantages == [[0.0, 0.5, -0.5]]
    assert result.global_step == 1
    assert not torch.equal(before, model.transformer.wte.weight.detach())


def test_real_sft_script_keeps_tools_disables_thinking_and_updates_weights(
    tmp_path, monkeypatch
):
    import datasets
    import trl

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("MODEL", "offline-tiny")
    monkeypatch.setenv("RUN_NAME", str(tmp_path / "sft"))
    monkeypatch.setenv("HUB_MODEL_ID", "")
    monkeypatch.setenv("TRACKIO_SPACE", "")
    monkeypatch.setenv("USE_LORA", "0")
    monkeypatch.setenv("MAX_SAMPLES", "0")
    monkeypatch.setenv("MAX_STEPS", "1")
    monkeypatch.setenv("MAX_LENGTH", "32")
    monkeypatch.setenv("BATCH_SIZE", "2")
    monkeypatch.setenv("GRAD_ACCUM", "1")
    model, tokenizer = tiny_policy()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Execute code",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "a"},
                ],
                "tools": tools,
            }
        ]
        * 20
    )
    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: dataset)
    seen = []
    original_template = tokenizer.apply_chat_template

    def template(messages, **kwargs):
        seen.append((kwargs.get("tools"), kwargs.get("enable_thinking")))
        return original_template(messages, **kwargs)

    monkeypatch.setattr(tokenizer, "apply_chat_template", template)

    def cpu_config(**kwargs):
        # Keep the actual script's SFTConfig fields; adapt only its hardware/logging
        # settings for this offline test. An unsupported field still raises.
        kwargs.update(
            use_cpu=True,
            bf16=False,
            fp16=False,
            report_to="none",
            eval_strategy="no",
            save_strategy="no",
            gradient_checkpointing=False,
            disable_tqdm=True,
        )
        return SFTConfig(**kwargs)

    def cpu_trainer(**kwargs):
        assert kwargs.pop("model") == "offline-tiny"
        return SFTTrainer(model=model, processing_class=tokenizer, **kwargs)

    monkeypatch.setattr(trl, "SFTConfig", cpu_config)
    monkeypatch.setattr(trl, "SFTTrainer", cpu_trainer)
    before = model.transformer.wte.weight.detach().clone()
    training = load_script("train_sft")
    assert {"messages", "tools", "chat_template_kwargs"} <= set(
        training.ds.column_names
    )
    assert training.ds[0]["tools"] == tools
    assert seen and all(item == (tools, False) for item in seen)
    assert training.trainer.state.global_step == 1
    assert not torch.equal(before, model.transformer.wte.weight.detach())
