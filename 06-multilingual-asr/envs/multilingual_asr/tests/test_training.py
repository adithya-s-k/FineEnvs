

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
