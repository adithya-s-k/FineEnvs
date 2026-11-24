"""Client for a running Scott Wu Math server (local, Docker or a Hugging Face Space).

    from scott_wu_math import CardAction, ScottWuMathEnv

    with ScottWuMathEnv(base_url="http://localhost:8012").sync() as env:
        obs = env.reset(level="four-digit", seed=7).observation
        print(obs.prompt)
        result = env.step(CardAction(expression="..."))
        print(result.reward, result.observation.result)
"""

from __future__ import annotations

from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import CardAction, CardObservation, CardState


class ScottWuMathEnv(EnvClient[CardAction, CardObservation, CardState]):
    def _step_payload(self, action: CardAction) -> Dict[str, Any]:
        return action.model_dump(exclude={"metadata"})

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[CardObservation]:
        obs = CardObservation(**payload.get("observation", {}), done=payload.get("done", False), reward=payload.get("reward"))
        return StepResult(observation=obs, reward=payload.get("reward"), done=payload.get("done", False))

    def _parse_state(self, payload: Dict[str, Any]) -> CardState:
        return CardState(**payload)
