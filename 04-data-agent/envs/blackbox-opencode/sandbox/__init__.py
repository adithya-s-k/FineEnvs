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

"""Sandbox selection for this environment: `openenv.core.sandbox`, plus our default image.

THE BACKENDS ARE NOT IMPLEMENTED HERE, AND THAT IS THE POINT. They live in `openenv.core.sandbox`,
shared with `opencode_env` and `pi_env`. An earlier version of this file vendored copies of
`base.py`, `e2b.py` and `hf.py` so the package would stand alone -- which traded one problem for a
worse one, because a copied sandbox layer drifts precisely in the per-backend details that are
expensive to get wrong (`sandbox_home` being the standing example). Depending on `openenv-core`,
which is already this package's dependency, keeps it self-contained with respect to other
ENVIRONMENTS without duplicating anything.

All this module adds is the image that carries the data-science stack the instruction promises.
"""

from __future__ import annotations

from typing import Any

from openenv.core.sandbox import (
    available,
    BACKENDS,
    build_backend as _build_backend,
    describe,
    sandbox_home,
    SandboxBackend,
)


# pandas / numpy / scipy and friends. A task's instruction names them, so a bare image turns every
# rollout into a package-install exercise the reward does not pay for.
DEFAULT_IMAGE = "docker.io/savatar101/env-data-agent-train:base"


def build_backend(
    backend: str, *, image: str = DEFAULT_IMAGE, **kwargs: Any
) -> SandboxBackend:
    """`openenv.core.sandbox.build_backend`, defaulted to this environment's image."""
    return _build_backend(backend, image=image, **kwargs)


__all__ = [
    "BACKENDS",
    "DEFAULT_IMAGE",
    "SandboxBackend",
    "available",
    "build_backend",
    "describe",
    "sandbox_home",
]
