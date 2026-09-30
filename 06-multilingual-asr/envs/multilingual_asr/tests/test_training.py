

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
