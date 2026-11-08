"""Train and evaluate on any OpenEnv environment with Tinker.

Laid out as a tinker-cookbook recipe (`tinker_cookbook/recipes/openenv_rl/`); it uses only OpenEnv's public
client APIs (`openenv>=0.7`), so OpenEnv itself needs no changes. Two paths, matching OpenEnv's harness modes:

- **White-box** (`OpenEnvEnvGroupBuilder`, `OpenEnvRLDatasetBuilder`): Tinker samples, the cookbook renderer
  parses tool calls, and each call runs through an OpenEnv `ResourceSession`. Works with any
  `ResourceSessionFactory` (every `envs/<env>/harness.py`) and, through `MCPSessionFactory`, with any MCP
  environment. Tool results may carry images.
- **Black-box** (`TinkerChatServer`, `run_black_box_group`, `train_step`): an agent harness owns its loop
  (Harbor tasks with OpenCode, Claude Code, ...) and calls a model endpoint through OpenEnv's capture proxy.
  `TinkerChatServer` is that endpoint; the captured `TrainingTrace`s become Tinker datums.
"""

from .blackbox import (
    BlackBoxGroup,
    BlackBoxRollout,
    group_advantages,
    groups_to_datums,
    run_black_box_group,
    trace_to_datums,
    train_step,
)
from .chat_server import TinkerChatServer
from .content import to_content_parts
from .openenv_env import (
    ImageWindow,
    MissingRewardError,
    OpenEnvEnvGroupBuilder,
    OpenEnvRLDataset,
    OpenEnvRLDatasetBuilder,
    make_renderer,
)
from .openenv_tools import EpisodeLedger, SessionTool
from .sessions import MCPSession, MCPSessionFactory, default_initial_messages

__all__ = [
    "BlackBoxGroup",
    "BlackBoxRollout",
    "EpisodeLedger",
    "ImageWindow",
    "MCPSession",
    "MCPSessionFactory",
    "MissingRewardError",
    "OpenEnvEnvGroupBuilder",
    "OpenEnvRLDataset",
    "OpenEnvRLDatasetBuilder",
    "SessionTool",
    "TinkerChatServer",
    "default_initial_messages",
    "group_advantages",
    "groups_to_datums",
    "make_renderer",
    "run_black_box_group",
    "to_content_parts",
    "trace_to_datums",
    "train_step",
]
