# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data-analysis agent tasks, with token-level capture.

An agent gets a question and a directory of real tables, works in a sandbox with its own tools, and
files its answer to `/workdir/answer.txt`. The environment grades it and returns every model call the
agent made -- with the ENGINE'S OWN token ids, so a trainer never re-renders a prompt.

Nothing from `server/` is exported here. A client talks to a running deployment over HTTP; importing
the server would drag a dataset, a sandbox SDK and a web framework into a trainer that needs none of
them.

Examples:

```python
from data_agent_env import DataAgentEnv, DataAgentSessionFactory

# Direct use: run one rollout against a deployment.
env = DataAgentEnv("http://127.0.0.1:8200")
result = env.run_rollout(split="train:medium", index=0, llm_url=VLLM_URL, model=MODEL)
print(result.reward, len(result.turns))

# Training use: hand the factory to TRL's HarnessRolloutWorker.
factory = DataAgentSessionFactory(
    "http://127.0.0.1:8200", split="train:medium", llm_url=VLLM_URL, model=MODEL,
    sampling={"temperature": 0.8, "top_p": 1.0, "top_k": 0},
)
```
"""

from .client import DataAgentEnv
from .config import DataAgentConfig
from .harness import (
    DataAgentSession,
    DataAgentSessionFactory,
    opencode_agent_turns,
    to_trace_entries,
)
from .models import DataAgentRolloutResult, DataAgentState, DataAgentTurn
from .task import DataAgentTask


__all__ = [
    "DataAgentConfig",
    "DataAgentEnv",
    "DataAgentRolloutResult",
    "DataAgentSession",
    "DataAgentSessionFactory",
    "DataAgentState",
    "DataAgentTask",
    "DataAgentTurn",
    "opencode_agent_turns",
    "to_trace_entries",
]
