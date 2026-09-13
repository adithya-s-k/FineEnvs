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

"""White-box bash/Jupyter/file agent environment.

Install this package on the TRAINER side and hand `white_box_bash_env(...)` to TRL's
`GRPOTrainer(environment_factory=...)`. The trainer owns the loop; the sandbox lives behind a hosted
OpenEnv server.
"""

from .client import exposed_tool_names, white_box_bash_env
from .tools import DEFAULT_TOOLSETS, TOOLSETS, resolve, specs_for, tool_names


__all__ = [
    "DEFAULT_TOOLSETS",
    "TOOLSETS",
    "exposed_tool_names",
    "resolve",
    "specs_for",
    "tool_names",
    "white_box_bash_env",
]
