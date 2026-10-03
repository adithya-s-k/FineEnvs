

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


def test_features_are_padded_with_a_mask_so_padding_is_not_heard():
    """Waveform padding was heard as ~18s of silence; feature padding is masked out."""
    import types

    import numpy
    import pytest as pt

    module = pt.importorskip("transformers.models.gemma4.feature_extraction_gemma4")
    from multilingual_asr.training import SAMPLING_RATE, pad_audio_features

    extractor = module.Gemma4AudioFeatureExtractor()
    rng = numpy.random.default_rng(0)
    short = rng.standard_normal(4 * SAMPLING_RATE).astype("float32") * 0.1
    long = rng.standard_normal(28 * SAMPLING_RATE).astype("float32") * 0.1
    bare = extractor([short], sampling_rate=SAMPLING_RATE)
    pad_audio_features(types.SimpleNamespace(feature_extractor=extractor), 30)
    pad_audio_features(types.SimpleNamespace(feature_extractor=extractor), 30)  # idempotent
    padded = [extractor([clip], sampling_rate=SAMPLING_RATE) for clip in (short, long)]
    shapes = {numpy.asarray(p["input_features"]).shape for p in padded}
    assert len(shapes) == 1  # one shape, so TRL can stack a batch of different clips
    frames = numpy.asarray(bare["input_features"]).shape[1]
    mask = numpy.asarray(padded[0]["input_features_mask"])[0]
    # Only the real frames are marked valid, and they are the unpadded features exactly.
    assert mask.sum() == numpy.asarray(bare["input_features_mask"]).sum() and not mask[frames:].any()
    assert numpy.allclose(
        numpy.asarray(padded[0]["input_features"])[0][:frames],
        numpy.asarray(bare["input_features"])[0],
    )
    # A saved processor still names the real class, so it reloads anywhere.
    assert extractor.to_dict()["feature_extractor_type"] == "Gemma4AudioFeatureExtractor"


def test_audio_rows_stack_per_prompt_fields_into_tensors():
    import numpy
    import pytest as pt

    torch = pt.importorskip("torch")
    from multilingual_asr.training import audio_rows

    fields = {
        "input_features": [numpy.zeros((5, 2)), numpy.ones((5, 2))],
        "input_features_mask": [numpy.ones(5, bool), numpy.zeros(5, bool)],
        "token_type_ids": [[0, 0]],
    }
    rows = audio_rows(fields)
    assert set(rows) == {"input_features", "input_features_mask"}
    assert rows["input_features"].shape == (2, 5, 2) and rows["input_features"][1].eq(1).all()
    assert audio_rows({"token_type_ids": [[0]]}) == {}
    with pt.raises(RuntimeError, match="Expected both"):
        audio_rows({"input_features": [numpy.zeros((5, 2))]})
    assert isinstance(rows["input_features_mask"], torch.Tensor)


def _recorder():
    import torch

    class Recorder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = []

        def forward(self, input_ids=None, input_features=None, input_features_mask=None):
            self.seen.append(None if input_features is None else input_features[:, 0, 0].tolist())
            return input_ids

    return Recorder()


def test_audio_forward_gives_each_chunk_its_own_clips_in_order():
    """TRL scores a batch in chunks; every chunk must carry the audio of its own rows."""
    import pytest as pt

    torch = pt.importorskip("torch")
    from multilingual_asr.training import audio_forward

    model = _recorder()
    rows = {
        "input_features": torch.arange(5.0).view(5, 1, 1).expand(5, 3, 2).clone(),
        "input_features_mask": torch.ones(5, 3, dtype=torch.bool),
    }
    with audio_forward(model, rows):
        model(input_ids=torch.zeros(2, 4, dtype=torch.long))
        model(input_ids=torch.zeros(3, 4, dtype=torch.long))
    assert model.seen == [[0.0, 1.0], [2.0, 3.0, 4.0]]
    # Outside the block nothing is attached.
    model(input_ids=torch.zeros(1, 4, dtype=torch.long))
    assert model.seen[-1] is None


def test_audio_forward_leaves_generation_inputs_alone_and_catches_misalignment():
    import pytest as pt

    torch = pt.importorskip("torch")
    from multilingual_asr.training import audio_forward

    model = _recorder()
    rows = {
        "input_features": torch.zeros(2, 3, 2),
        "input_features_mask": torch.ones(2, 3, dtype=torch.bool),
    }
    own = torch.full((1, 3, 2), 7.0)
    with pt.raises(RuntimeError, match="misaligned"):
        with audio_forward(model, rows):
            # A forward that brings its own features (generation) is not touched...
            model(input_ids=torch.zeros(1, 4, dtype=torch.long), input_features=own)
            # ...and a batch that does not use every row is an error, not a silent skip.
            model(input_ids=torch.zeros(1, 4, dtype=torch.long))
    assert model.seen == [[7.0], [0.0]]
    with pt.raises(RuntimeError, match="wants rows"):
        with audio_forward(model, rows):
            model(input_ids=torch.zeros(3, 4, dtype=torch.long))


def test_the_audio_trainer_carries_clips_into_every_log_prob_forward():
    """TRL 1.13 drops audio before the loss; the subclass must route it back in."""
    import pytest as pt

    pt.importorskip("trl")
    from multilingual_asr.training import audio_grpo_trainer

    trainer = audio_grpo_trainer()
    for name in (
        "_tokenize_prompts",
        "_generate_and_score_completions",
        "_compute_loss",
        "_get_per_token_logps_and_entropies",
    ):
        # Each override must still exist upstream, or it silently stops being called.
        assert name in vars(trainer) and hasattr(trainer.__mro__[1], name)


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


def test_headroom_compares_greedy_with_the_policys_own_samples():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[3] / "train" / "eval_vllm.py"
    spec = importlib.util.spec_from_file_location("eval_vllm_headroom", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    stats = module.headroom(
        {"a": 0.5, "b": 0.8},
        {"a": [0.4, 0.6, 0.5], "b": [0.8, 0.8]},
    )
    assert stats["headroom_tasks"] == 2
    assert abs(stats["greedy_mean"] - 0.65) < 1e-9
    assert abs(stats["best_of_n_mean"] - 0.7) < 1e-9
    assert abs(stats["best_minus_greedy"] - 0.05) < 1e-9
    # Only task a has any sample above its greedy answer, and one of its three does.
    assert abs(stats["tasks_with_a_better_sample"] - 0.5) < 1e-9
    assert abs(stats["mean_share_of_samples_beating_greedy"] - (1 / 3) / 2) < 1e-9
    assert module.headroom({"a": 0.5}, {}) == {}


def test_large_draws_are_fetched_within_the_servers_position_limit(monkeypatch):
    """2,282 positions in one request is a 400; the draw must arrive in pieces."""
    from multilingual_asr import training

    sizes = []

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def num_group_tasks(self, split, lang, fam):
            return 2282

        def get_group_tasks(self, split, lang, fam, positions):
            assert len(positions) <= training.POSITIONS_PER_REQUEST
            sizes.append(len(positions))
            return [{"task_id": f"{lang}-{p}", "language": lang, "family": fam}
                    for p in positions]

    monkeypatch.setattr(training, "connect", lambda url: Client())
    rows = training.sampled_rows("http://x", "train", ["kn_in"], ["transcription"], 42, 2282)
    assert len(rows) == 2282
    assert sizes == [1000, 1000, 282]


def test_a_checkpoint_is_complete_only_when_every_adapter_byte_has_arrived(tmp_path):
    """A watcher reads checkpoints while they upload; a partial adapter must not score."""
    import json

    from multilingual_asr.training import (
        READY,
        checkpoint_complete,
        mark_checkpoint_ready,
    )

    saved = tmp_path / "checkpoint-25"
    saved.mkdir()
    (saved / "adapter_config.json").write_text('{"r": 16}')
    (saved / "adapter_model.safetensors").write_bytes(b"w" * 4096)
    mark_checkpoint_ready(saved, 25)
    ready = json.loads((saved / READY).read_text())
    assert ready["step"] == 25 and set(ready["files"]) == {
        "adapter_config.json",
        "adapter_model.safetensors",
    }
    copy = tmp_path / "copy"
    copy.mkdir()
    (copy / "adapter_config.json").write_text('{"r": 16}')
    (copy / "adapter_model.safetensors").write_bytes(b"w" * 1024)  # still arriving
    assert not checkpoint_complete(copy, ready)
    (copy / "adapter_model.safetensors").write_bytes(b"x" * 4096)  # right size, wrong bytes
    assert not checkpoint_complete(copy, ready)
    (copy / "adapter_model.safetensors").write_bytes(b"w" * 4096)
    assert checkpoint_complete(copy, ready)


def _eval_module():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[3] / "train" / "eval_vllm.py"
    spec = importlib.util.spec_from_file_location("eval_vllm_watch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_paired_change_is_per_clip_and_carries_an_interval():
    """Pairing removes clip difficulty; the interval says whether a change is real."""
    import pytest as pt

    module = _eval_module()
    base = [{"task_id": f"t{i}", "reward": 0.2 + i / 100} for i in range(50)]
    better = [{"task_id": s["task_id"], "reward": s["reward"] + 0.05} for s in base]
    change = module.paired(base, better, "reward")
    assert change["delta"] == pt.approx(0.05) and change["tasks"] == 50
    assert change["low"] == pt.approx(0.05) == change["high"]  # identical shift
    noisy = [{"task_id": s["task_id"], "reward": s["reward"] + (0.1 if i % 2 else -0.1)}
             for i, s in enumerate(base)]
    spread = module.paired(base, noisy, "reward")
    assert spread["low"] < 0 < spread["high"]
    # A task only one side scored is left out rather than paired with nothing.
    assert module.paired(base, better[:1], "reward") is None


def test_curve_points_average_every_metric_and_compare_to_base():
    module = _eval_module()
    base = {"model": "base", "samples": [
        {"task_id": "a", "reward": 0.4, "cer": 0.3, "exact_match": False},
        {"task_id": "b", "reward": 0.6, "cer": 0.1, "exact_match": True},
    ]}
    step = {"model": "step-25", "samples": [
        {"task_id": "a", "reward": 0.5, "cer": 0.2, "exact_match": True},
        {"task_id": "b", "reward": 0.6, "cer": 0.1, "exact_match": True},
    ]}
    point = module.curve_point(25, step, base)
    assert point["step"] == 25 and point["reward"] == 0.55 and point["exact_match"] == 1.0
    assert abs(point["cer_change"]["delta"] + 0.05) < 1e-9
    assert "wer" not in point and "reward_change" in point
    assert "reward_change" not in module.curve_point(0, base, None)


def test_only_checkpoints_marked_ready_are_picked_up_in_step_order():
    module = _eval_module()
    paths = [
        "rev/checkpoint-50/ready.json",
        "rev/checkpoint-25/ready.json",
        "rev/checkpoint-75/adapter_model.safetensors",  # saved, not yet marked
        "rev/evals/checkpoints/checkpoint-25/ready.json",  # the watcher's own copy
        "other/checkpoint-100/ready.json",
    ]
    assert module.checkpoint_steps(paths, "rev") == [25, 50]
