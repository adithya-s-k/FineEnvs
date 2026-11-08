"""CLI entry point: RL on any MCP OpenEnv environment with Tinker (white-box path).

Example (DesignGym, tools surface, reference mode)::

    python -m openenv_rl.train env_url=http://127.0.0.1:8007 tasks_path=tasks/train.jsonl \
        initial_messages=designgym_rl:initial_messages exclude_tools=start_task \
        finalize_tool=submit model_name=Qwen/Qwen3.6-35B-A3B renderer_name=qwen3_5_disable_thinking

`tasks_path` is JSONL, one `reset()` kwargs dict per line. `initial_messages` names a
`module:function` mapping `(reset StepResult, task)` to the opening messages.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import chz
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.rl.train import AsyncConfig, Config, main

from .openenv_env import ImageWindow, OpenEnvRLDatasetBuilder
from .sessions import MCPSessionFactory, default_initial_messages

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    """Command-line configuration for OpenEnv RL training."""

    # Environment
    env_url: str = "http://127.0.0.1:8000"
    tasks_path: str = "tasks/train.jsonl"
    test_tasks_path: str | None = None
    initial_messages: str | None = None
    exclude_tools: str = ""  # comma-separated tool names hidden from the agent
    finalize_tool: str | None = None
    reward_source: str = "verify"
    max_images: int | None = None
    keep_images: int = 3
    tool_images_as_user: bool = False  # tool-result images in a following user turn (the eval harness's layout)
    message_timeout_s: float = 300.0

    # Model
    model_name: str = "Qwen/Qwen3.6-35B-A3B"
    renderer_name: str | None = None
    lora_rank: int = 32
    load_checkpoint_path: str | None = None
    max_tokens: int = 1024
    temperature: float = 1.0

    # Episodes
    max_turns: int = 40
    max_trajectory_tokens: int = 48 * 1024

    # Training
    group_size: int = 8
    groups_per_batch: int = 16
    learning_rate: float = 1e-5
    kl_penalty_coef: float = 0.0
    loss_fn: str = "importance_sampling"  # importance_sampling | ppo | cispo | dro (cispo is the safe off-policy default)
    loss_fn_config: str | None = None  # JSON, e.g. {"clip_low_threshold": 0.0, "clip_high_threshold": 4.0}
    remove_constant_reward_groups: bool = False  # drop groups whose rollouts all score the same (zero advantage)
    parse_error_retries: int = 0  # malformed tool calls retried with a corrective message (0: the first one ends the episode)
    parse_error_penalty: float = 0.0  # reward on each retried malformed turn is minus this
    num_substeps: int = 1
    max_steps: int | None = None
    max_steps_off_policy: int | None = None

    # Logging / checkpoints
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    eval_every: int = 5
    save_every: int = 5
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


def _load_tasks(path: str | None) -> list[dict] | None:
    if not path:
        return None
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _resolve(spec: str | None):
    if not spec:
        return default_initial_messages
    module, _, attr = spec.partition(":")
    return getattr(importlib.import_module(module), attr)


def _rollout_config(cfg: CLIConfig):
    """Parse-error retries for long agent episodes: one malformed call should not throw away a 100-turn rollout."""
    if not cfg.parse_error_retries:
        return None
    from tinker_cookbook.rl.rollout_limits import ParseErrorPolicy
    from tinker_cookbook.rl.rollout_presets import RolloutConfig

    return RolloutConfig(parse_errors=ParseErrorPolicy(max_consecutive=cfg.parse_error_retries,
                                                       penalty_per_error=cfg.parse_error_penalty,
                                                       mask_error_turns=True))


async def cli_main(cfg: CLIConfig) -> None:
    renderer_name = cfg.renderer_name or model_info.get_recommended_renderer_name(cfg.model_name)
    run_name = (
        f"openenv-{cfg.model_name.replace('/', '-')}-{cfg.lora_rank}rank-{cfg.learning_rate}lr-"
        f"{cfg.group_size}group-{cfg.groups_per_batch}batch-{datetime.now(UTC).strftime('%Y-%m-%d-%H-%M')}"
    )
    log_path = cfg.log_path or f"/tmp/tinker-examples/openenv_rl/{run_name}"
    factory = MCPSessionFactory(
        cfg.env_url,
        initial_messages_builder=_resolve(cfg.initial_messages),
        exclude_tools=tuple(t.strip() for t in cfg.exclude_tools.split(",") if t.strip()),
        message_timeout_s=cfg.message_timeout_s,
    )
    dataset_builder = OpenEnvRLDatasetBuilder(
        session_factory=factory,
        tasks=_load_tasks(cfg.tasks_path),
        test_tasks=_load_tasks(cfg.test_tasks_path),
        model_name=cfg.model_name,
        renderer_name=renderer_name,
        batch_size=cfg.groups_per_batch,
        group_size=cfg.group_size,
        max_turns=cfg.max_turns,
        max_trajectory_tokens=cfg.max_trajectory_tokens,
        reward_source=cfg.reward_source,
        finalize_tool=cfg.finalize_tool,
        image_window=ImageWindow(cfg.max_images, cfg.keep_images) if cfg.max_images else None,
        rollout_config=_rollout_config(cfg),
        tool_images_as_user=cfg.tool_images_as_user,
    )
    config = Config(
        learning_rate=cfg.learning_rate,
        dataset_builder=dataset_builder,
        model_name=cfg.model_name,
        recipe_name="recipe_openenv_rl",
        lora_rank=cfg.lora_rank,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        wandb_project=cfg.wandb_project,
        wandb_name=cfg.wandb_name or run_name,
        log_path=log_path,
        load_checkpoint_path=cfg.load_checkpoint_path,
        kl_penalty_coef=cfg.kl_penalty_coef,
        loss_fn=cfg.loss_fn,
        loss_fn_config=json.loads(cfg.loss_fn_config) if cfg.loss_fn_config else None,
        remove_constant_reward_groups=cfg.remove_constant_reward_groups,
        num_substeps=cfg.num_substeps,
        eval_every=cfg.eval_every,
        save_every=cfg.save_every,
        async_config=AsyncConfig(max_steps_off_policy=cfg.max_steps_off_policy, groups_per_batch=cfg.groups_per_batch)
        if cfg.max_steps_off_policy is not None
        else None,
        max_steps=cfg.max_steps,
    )
    cli_utils.check_log_dir(log_path, behavior_if_exists=cfg.behavior_if_log_dir_exists)
    await main(config)


if __name__ == "__main__":
    asyncio.run(cli_main(chz.entrypoint(CLIConfig)))
