

def test_env_sessions_cover_every_rollout_an_optimizer_step_holds_open():
    """A smoke run died at 20/20 sessions because sizing ignored accumulation."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[3] / "train" / "grpo_asr.py"
    spec = importlib.util.spec_from_file_location("grpo_asr_sessions", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.Config(num_generations=16, gradient_accumulation_steps=4)
    assert config.rollouts_in_flight == 64
    assert module.Config(num_generations=8).rollouts_in_flight == 8


def test_padding_gives_every_clip_one_feature_shape():
    """Batches spanning several tasks need one audio shape, or TRL raises."""
    import numpy
    from multilingual_asr.training import SAMPLING_RATE, AssetCache

    cache = AssetCache("http://localhost", pad_seconds=30.0)
    short = cache._fit(numpy.ones(4 * SAMPLING_RATE, dtype="float32"))
    long = cache._fit(numpy.ones(28 * SAMPLING_RATE, dtype="float32"))
    assert short.shape == long.shape == (30 * SAMPLING_RATE,)
    # Real samples survive; only the tail is padding.
    assert short[: 4 * SAMPLING_RATE].all() and not short[4 * SAMPLING_RATE :].any()

    over = cache._fit(numpy.ones(35 * SAMPLING_RATE, dtype="float32"))
    assert over.shape == (30 * SAMPLING_RATE,)

    unpadded = AssetCache("http://localhost")
    kept = numpy.ones(4 * SAMPLING_RATE, dtype="float32")
    assert unpadded._fit(kept).shape == kept.shape


def test_adapter_spec_parsing_and_mutually_exclusive_candidates():
    """Checkpoints are named name=path; models and adapters cannot mix."""
    import argparse
    import importlib.util
    from pathlib import Path

    import pytest as pt

    path = Path(__file__).resolve().parents[3] / "train" / "eval_vllm.py"
    spec = importlib.util.spec_from_file_location("eval_vllm_adapters", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.parse_adapter("step-50=/outputs/rev/checkpoint-50") == (
        "step-50",
        "/outputs/rev/checkpoint-50",
    )
    with pt.raises(argparse.ArgumentTypeError):
        module.parse_adapter("checkpoint-50")


def test_lora_targets_skip_towers_that_cannot_receive_gradient():
    """A 250-step run ended with 56 of 122 adapter modules still exactly zero.

    TRL's GRPO loss forward has no input_features parameter and its multimodal branch is
    gated on images, so the audio tower never enters the backward graph. Targeting it
    spends parameters and hides that nothing there is learning.
    """
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[3] / "train" / "grpo_asr.py"
    spec = importlib.util.spec_from_file_location("grpo_asr_targets", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Fake:
        """Stands in for the meta-device skeleton, with Gemma 4's wrapper nesting."""

        def named_modules(self):
            import torch

            names = [
                "language_model.layers.0.self_attn.q_proj",
                "language_model.layers.0.self_attn.v_proj",
                "audio_tower.layers.0.attention.q_proj.linear",
                "audio_tower.layers.0.attention.v_proj.linear",
                "vision_tower.encoder.layers.0.self_attn.q_proj",
            ]
            return [(n, torch.nn.Linear(2, 2)) for n in names]

    import torch

    adaptable = (torch.nn.Linear, torch.nn.Embedding, torch.nn.Conv1d, torch.nn.Conv2d)
    wanted = ("q_proj", "v_proj")
    chosen = sorted(
        name
        for name, mod in Fake().named_modules()
        if isinstance(mod, adaptable)
        and any(p in wanted for p in name.split("."))
        and not any(t in name for t in module.SKIPPED_TOWERS)
    )
    assert chosen == [
        "language_model.layers.0.self_attn.q_proj",
        "language_model.layers.0.self_attn.v_proj",
    ]
    assert not [c for c in chosen if "audio_tower" in c or "vision_tower" in c]
