# SPDX-License-Identifier: BSD-3-Clause

"""Client for the watercolour environment."""


from __future__ import annotations

import sys
from pathlib import Path

# Two deliberate departures from the layout in CONTRIBUTING.md, both forced by the
# same thing: this folder is called `openenv` and so is the installed package.
#
# 1. There is no `__init__.py` here. With one, `import openenv` resolves to this
#    folder whenever the interpreter's working directory is the environment root,
#    which is what the Dockerfile sets, and `openenv.core` then does not exist.
# 2. The path entry is appended rather than inserted, and points at the folder
#    holding `core/`, so an installed package always wins over a local directory.
_ENV_ROOT = str(Path(__file__).resolve().parents[1])
if not __package__ and _ENV_ROOT not in sys.path:
    sys.path.append(_ENV_ROOT)


from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

if __package__:
    from .models import (  # noqa: E402
        WatercolourAction,
        WatercolourObservation,
        WatercolourState,
    )
else:
    from models import (  # noqa: E402
        WatercolourAction,
        WatercolourObservation,
        WatercolourState,
    )


class WatercolourEnv(
    EnvClient[WatercolourAction, WatercolourObservation, WatercolourState]
):
    """Connects to a running watercolour environment server.

    Examples:

    ```python
    with WatercolourEnv(base_url="http://localhost:8000") as env:
        observation = env.reset().observation
        reply = my_model(observation.system_prompt, observation.prompt)
        result = env.step(WatercolourAction(response=reply))
        print(result.reward, result.observation.feedback)
    ```
    """

    def _step_payload(self, action: WatercolourAction) -> Dict[str, Any]:
        """Convert an action into the JSON body of a step request."""
        return {"response": action.response}

    def _parse_result(
        self, payload: Dict[str, Any]
    ) -> StepResult[WatercolourObservation]:
        """Parse a server response into a typed step result.

        The server hoists `reward` and `done` onto the response envelope and
        drops them from the serialised observation, so they are read from the
        envelope first and only then from the observation body.
        """
        data = payload.get("observation", {})
        reward = payload.get("reward", data.get("reward"))
        done = payload.get("done", data.get("done", False))
        observation = WatercolourObservation(
            prompt=data.get("prompt", ""),
            system_prompt=data.get("system_prompt", ""),
            task_id=data.get("task_id", ""),
            subject=data.get("subject", ""),
            feedback=data.get("feedback", ""),
            gate_passed=data.get("gate_passed", False),
            length_score=data.get("length_score", 0.0),
            judge_score=data.get("judge_score", 0.0),
            judged=data.get("judged", False),
            paint_fraction=data.get("paint_fraction", 0.0),
            finished=data.get("finished", False),
            violations=data.get("violations", []),
            js_errors=data.get("js_errors", []),
            breakdown=data.get("breakdown", {}),
            image_png_base64=data.get("image_png_base64"),
            done=bool(done),
            reward=reward,
            metadata=payload.get("metadata", data.get("metadata", {})),
        )
        return StepResult(
            observation=observation,
            reward=observation.reward,
            done=observation.done,
        )

    def _parse_state(self, payload: Dict[str, Any]) -> WatercolourState:
        """Parse a response from the state endpoint into a typed state."""
        return WatercolourState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", ""),
            submitted=payload.get("submitted", False),
        )
