"""Action, observation and state of the Scott Wu Math environment."""

from __future__ import annotations

from typing import Any, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class CardAction(Action):
    """The one move of an episode: the final expression. An empty expression gives up."""

    expression: str = Field(
        default="",
        description="Arithmetic over the card values using each card once, e.g. '(13 - 12 + 11 - 10 + 7) * 9 * 5 * 5'",
    )


class CardObservation(Observation):
    task_id: str = Field(default="", description="Task identifier")
    level: str = Field(default="", description="24, three-digit, year, four-digit or custom")
    target: int = Field(default=0, description="The number to make")
    cards: list[dict[str, Any]] = Field(default_factory=list, description="Dealt cards: rank (1-13), suit (S H D C), label")
    prompt: str = Field(default="", description="The user prompt a language model reads")
    system_prompt: str = Field(default="", description="The system prompt a language model reads")
    result: Optional[dict[str, Any]] = Field(default=None, description="Grader output after the submission")
    solution: Optional[str] = Field(default=None, description="A reference solution, shown only once the episode is over")
    solution_steps: list[str] = Field(default_factory=list, description="The reference solution one combination at a time")
    elapsed_s: Optional[float] = Field(default=None, description="Seconds from reset to submission")


class CardState(State):
    task_id: str = ""
    level: str = ""
    target: int = 0
    cards: list[dict[str, Any]] = Field(default_factory=list)
    submitted: bool = False
    reward_mode: str = "shaped"
