"""SkyRL entrypoint for DesignGym: SkyRL's PPO/GRPO experiment with our step-wise VLM generator.

    uv run --isolated --extra megatron --with websockets python -m rl.skyrl.entrypoint \
        trainer.policy.model.path=Qwen/Qwen3.8-27B data.train_data="['/work/data/train.parquet']" ...

`environment.env_class=designgym` names a no-op skyrl-gym env: the generator talks to DesignGym itself (the shared
rollout core), so nothing calls it, but SkyRL's dataset and config checks expect a registered env class.
"""

from __future__ import annotations

import multiprocessing as mp
import sys

import ray

from skyrl.backends.skyrl_train.inference_servers.utils import resolve_policy_model_name
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.utils import validate_cfg
from skyrl.train.utils.utils import initialize_ray
from skyrl_gym.envs import register
from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput

mp.set_start_method("spawn", force=True)


class NoopEnv(BaseTextEnv):
    """Registered so `environment.env_class=designgym` resolves; DesignGymVLMGenerator never steps it."""

    def __init__(self, env_config=None, extras=None):
        super().__init__()

    def step(self, action: str) -> BaseTextEnvStepOutput:
        return BaseTextEnvStepOutput(observations=[], reward=0.0, done=True, metadata={})


class DesignGymExp(BasePPOExp):
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        from rl.skyrl.designgym_generator import DesignGymVLMGenerator

        return DesignGymVLMGenerator(cfg.generator, inference_engine_client, tokenizer,
                                     policy_model_name=resolve_policy_model_name(cfg),
                                     model_path=cfg.trainer.policy.model.path)


@ray.remote(num_cpus=1)
def designgym_entrypoint(cfg: SkyRLTrainConfig):
    register(id="designgym", entry_point="rl.skyrl.entrypoint:NoopEnv")
    DesignGymExp(cfg).run()


def main() -> None:
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(designgym_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
