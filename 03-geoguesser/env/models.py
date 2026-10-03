# SPDX-License-Identifier: BSD-3-Clause

"""Data models for the GeoGuesser environment.

An episode places the agent at an unknown street-level location. It may look
around, walk along the road, and pin candidate coordinates on a map before
committing to a final guess. Only the final guess is scored.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from openenv.core.env_server import Action, Observation, State
from pydantic import BaseModel, Field


class EpisodeMode(str, Enum):
    """How much of the tool surface an episode exposes.

    Attributes:
        SINGLE_SHOT:
            One view, one guess. No pin loop, no navigation.
        AGENTIC:
            The full tool surface, subject to what the backend supports.
        NMPZ:
            No move, pan or zoom — the competitive "NMPZ" mode. Only pinning
            and guessing.
    """

    SINGLE_SHOT = "single_shot"
    AGENTIC = "agentic"
    NMPZ = "nmpz"


class RewardMode(str, Enum):
    """Which ground-truth granularity the reward is computed against."""

    COORDS = "coords"
    COUNTRY_ONLY = "country_only"


# =============================================================================
# Actions
# =============================================================================


class TypedAction(Action):
    """Base class for the ergonomic, per-operation action types.

    These are what callers construct in Python. They are converted to the flat
    [`GeoGuesserAction`] wire type by the client, because an HTTP environment
    declares exactly one action schema and `Action` forbids unknown fields.
    """


class LookAction(TypedAction):
    """Render a perspective view out of the current panorama.

    Args:
        heading_deg (`float`):
            Absolute compass heading in degrees, `0` being true north.
        pitch_deg (`float`, *optional*, defaults to `0.0`):
            Vertical angle in degrees; positive looks up.
        fov_deg (`float`, *optional*, defaults to `90.0`):
            Horizontal field of view. Smaller values zoom in.
    """

    heading_deg: float = Field(default=0.0, ge=-3600.0, le=3600.0)
    pitch_deg: float = Field(default=0.0, ge=-90.0, le=90.0)
    fov_deg: float = Field(default=90.0, ge=10.0, le=120.0)


class PanAction(TypedAction):
    """Turn relative to the current heading.

    Args:
        delta_deg (`float`):
            Degrees to turn; positive turns right.
    """

    delta_deg: float = Field(ge=-3600.0, le=3600.0)


class ZoomAction(TypedAction):
    """Change the field of view without turning.

    Args:
        fov_deg (`float`):
            New horizontal field of view in degrees.
    """

    fov_deg: float = Field(ge=10.0, le=120.0)


class MoveAction(TypedAction):
    """Walk along the captured sequence.

    Args:
        direction (`str`):
            Either `"forward"` or `"backward"` along the sequence.
        meters (`float`, *optional*, defaults to `10.0`):
            Requested distance. Frame spacing is irregular, so the observation
            reports how far the move actually travelled.
    """

    direction: str = Field(pattern="^(forward|backward)$")
    meters: float = Field(default=10.0, gt=0.0, le=500.0)


class PinAction(TypedAction):
    """Place a candidate pin and receive a map of where it landed.

    The response describes the pinned location only. It carries no information
    about the true location.

    Args:
        lat (`float`):
            Latitude of the candidate.
        lon (`float`):
            Longitude of the candidate.
        label (`str`, *optional*):
            Free-text note carried back in the pin list.
        span_deg (`float`, *optional*, defaults to `7.0`):
            Half-width in degrees of the map window returned with the pin.
            Choosing the zoom matters: below roughly 4 degrees the map adds
            roads, urban areas and town names, which is what makes aiming
            within a city possible rather than guessing at its centre.
    """

    lat: float = Field(ge=-90.0, le=90.0)
    lon: float = Field(ge=-180.0, le=180.0)
    label: str | None = None
    span_deg: float = Field(default=7.0, gt=0.02, le=180.0)


class ViewMapAction(TypedAction):
    """Pan and zoom the map without committing a pin.

    Args:
        lat (`float`):
            Latitude at the centre of the view.
        lon (`float`):
            Longitude at the centre of the view.
        span_deg (`float`, *optional*, defaults to `7.0`):
            Half-width of the window in degrees.
    """

    lat: float = Field(ge=-90.0, le=90.0)
    lon: float = Field(ge=-180.0, le=180.0)
    span_deg: float = Field(default=7.0, gt=0.05, le=180.0)


class MeasureAction(TypedAction):
    """Great-circle distance between two of the agent's own coordinates."""

    lat_a: float = Field(ge=-90.0, le=90.0)
    lon_a: float = Field(ge=-180.0, le=180.0)
    lat_b: float = Field(ge=-90.0, le=90.0)
    lon_b: float = Field(ge=-180.0, le=180.0)


class GuessAction(TypedAction):
    """Commit a final guess. Terminal.

    Either supply `response` and let the environment parse it, or supply
    `lat`/`lon` directly. Passing the raw reply keeps extraction failures
    visible in the score rather than hidden in the harness.

    Args:
        response (`str`, *optional*):
            The model's unedited reply. Coordinates are extracted from it.
        lat (`float`, *optional*):
            Latitude, when the caller has already parsed the reply.
        lon (`float`, *optional*):
            Longitude, when the caller has already parsed the reply.
        country (`str`, *optional*):
            ISO-3166 alpha-2 code or country name, scored for partial credit.
        confidence (`float`, *optional*):
            Self-reported confidence in [0, 1], recorded for calibration.
        reasoning (`str`, *optional*):
            Free-text rationale, recorded but not scored.
    """

    response: str | None = None
    lat: float | None = Field(default=None, ge=-90.0, le=90.0)
    lon: float | None = Field(default=None, ge=-180.0, le=180.0)
    country: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    reasoning: str | None = None


# =============================================================================
# Wire action
# =============================================================================


class GeoGuesserAction(Action):
    """The single action schema the server accepts.

    An HTTP environment declares one action class, so every operation travels
    as this flat record with an `op` discriminator. Callers normally build a
    [`TypedAction`] subclass instead and let the client convert.

    Attributes:
        op (`str`):
            Which operation to perform: `"look"`, `"pan"`, `"zoom"`, `"move"`,
            `"pin"`, `"view_map"`, `"measure"` or `"guess"`.
    """

    op: Literal["look", "pan", "zoom", "move", "pin", "view_map", "measure", "guess"]

    heading_deg: float | None = None
    pitch_deg: float | None = None
    fov_deg: float | None = None
    delta_deg: float | None = None
    direction: str | None = None
    meters: float | None = None
    lat: float | None = None
    lon: float | None = None
    label: str | None = None
    span_deg: float | None = None
    lat_a: float | None = None
    lon_a: float | None = None
    lat_b: float | None = None
    lon_b: float | None = None
    response: str | None = None
    country: str | None = None
    confidence: float | None = None
    reasoning: str | None = None


_OP_BY_TYPE: dict[type, str] = {}
_TYPE_BY_OP: dict[str, type] = {}


def _register_ops() -> None:
    pairs = [
        (LookAction, "look"),
        (PanAction, "pan"),
        (ZoomAction, "zoom"),
        (MoveAction, "move"),
        (PinAction, "pin"),
        (ViewMapAction, "view_map"),
        (MeasureAction, "measure"),
        (GuessAction, "guess"),
    ]
    for cls, op in pairs:
        _OP_BY_TYPE[cls] = op
        _TYPE_BY_OP[op] = cls


_register_ops()


def to_wire(action: TypedAction) -> GeoGuesserAction:
    """
    Convert a typed action into the flat wire action.

    Args:
        action ([`TypedAction`]):
            The action to convert.

    Returns:
        [`GeoGuesserAction`]: The same action, flattened, with `op` set.
    """
    op = _OP_BY_TYPE.get(type(action))
    if op is None:
        raise TypeError(f"No wire op registered for {type(action).__name__}")
    payload = action.model_dump(exclude_none=True, exclude={"metadata"})
    return GeoGuesserAction(op=op, **payload)


def from_wire(action: GeoGuesserAction) -> TypedAction:
    """
    Rebuild the typed action a wire action stands for.

    Args:
        action ([`GeoGuesserAction`]):
            The received wire action.

    Returns:
        [`TypedAction`]: The corresponding typed action, validated.
    """
    cls = _TYPE_BY_OP.get(action.op)
    if cls is None:
        raise ValueError(f"Unknown op: {action.op!r}")
    fields = set(cls.model_fields) - {"metadata"}
    payload = {
        k: v for k, v in action.model_dump(exclude_none=True).items() if k in fields
    }
    return cls(**payload)


# =============================================================================
# Observation
# =============================================================================


class Pin(BaseModel):
    """One candidate pin and what the environment could say about it.

    Attributes:
        index (`int`):
            1-based position in the pin list.
        lat (`float`):
            Latitude of the candidate.
        lon (`float`):
            Longitude of the candidate.
        label (`str` or `None`):
            The note the agent attached, if any.
        description (`str`):
            What the environment could say about the pinned coordinate. Never
            anything about the target.
    """

    index: int
    lat: float
    lon: float
    label: str | None = None
    description: str = ""


class GeoGuesserObservation(Observation):
    """What the agent sees after a reset or a step.

    The schema is identical across backends. A capability the backend lacks
    shows up as an unregistered tool and an empty field, never as a different
    shape, so one policy runs against every backend.

    Attributes:
        prompt (`str`):
            Instructions, populated on reset.
        image_base64 (`str` or `None`):
            The most recent rendered image as base64 PNG or JPEG — a
            perspective view, or a map after a pin.
        image_kind (`str`):
            Either `"view"`, `"map"` or `"none"`, saying what the image shows.
        heading_deg (`float`):
            Current compass heading in degrees.
        pitch_deg (`float`):
            Current vertical angle in degrees.
        fov_deg (`float`):
            Current field of view in degrees.
        moved_meters (`float`):
            Distance actually travelled by the last move.
        total_moved_meters (`float`):
            Cumulative distance travelled this episode.
        can_move_forward (`bool`):
            Whether a forward frame exists on the sequence.
        can_move_backward (`bool`):
            Whether a backward frame exists on the sequence.
        available_tools (`list[str]`):
            Tool names this backend actually registered.
        steps_remaining (`int`):
            Actions left before the episode is cut off.
        pins (`list[Pin]`):
            Candidates placed so far, in order.
        feedback (`str`):
            Text describing the result of the last action. For a pin, this
            describes the pinned location and nothing about the target.
        captured_at (`str`):
            Capture date of the current panorama, `YYYY-MM` — a legitimate
            meta clue, as in the real game.
        distance_km (`float` or `None`):
            Distance from guess to truth. Populated only after a guess.
        score (`float` or `None`):
            Distance score in [0, 1], before action costs. After a guess only.
        action_cost (`float`):
            Reward already spent on information gathering this episode. Visible
            throughout, not only after the guess, so a policy can see what it
            has committed.
        true_lat (`float` or `None`):
            Ground truth latitude, revealed only after a guess.
        true_lon (`float` or `None`):
            Ground truth longitude, revealed only after a guess.
        parsed_ok (`bool`):
            Whether coordinates could be extracted from the guess.
    """

    prompt: str = ""
    image_base64: str | None = None
    image_kind: str = "none"

    heading_deg: float = 0.0
    pitch_deg: float = 0.0
    fov_deg: float = 90.0

    moved_meters: float = 0.0
    total_moved_meters: float = 0.0
    can_move_forward: bool = False
    can_move_backward: bool = False

    available_tools: list[str] = Field(default_factory=list)
    steps_remaining: int = 0
    pins: list[Pin] = Field(default_factory=list)
    feedback: str = ""
    captured_at: str = ""

    distance_km: float | None = None
    score: float | None = None
    action_cost: float | None = None
    true_lat: float | None = None
    true_lon: float | None = None
    parsed_ok: bool = True


# =============================================================================
# State
# =============================================================================


class GeoGuesserState(State):
    """Internal episode state. Never sent to the agent verbatim.

    Attributes:
        task_index (`int`):
            Index into the frozen task list, `-1` before the first reset.
        task_id (`str`):
            Stable identifier of the sampled task.
        frame_index (`int`):
            Position within the task's sequence.
        heading_deg (`float`):
            Current heading in degrees.
        pitch_deg (`float`):
            Current pitch in degrees.
        fov_deg (`float`):
            Current field of view in degrees.
        n_looks (`int`):
            Count of view renders, for action cost.
        n_maps (`int`):
            Count of map renders that were not pins.
        n_pins (`int`):
            Count of pins placed.
        n_moves (`int`):
            Count of moves taken.
        n_free (`int`):
            Count of free-tool calls, which cost nothing but are capped so they
            cannot be issued forever.
        total_moved_meters (`float`):
            Cumulative distance travelled.
        submitted (`bool`):
            Whether the single allowed guess has been made.
        ended_without_guess (`bool`):
            Whether the episode ended because the action budget was exhausted.
        pins (`list[dict]`):
            Pins placed so far.
    """

    task_index: int = -1
    task_id: str = ""
    frame_index: int = 0
    heading_deg: float = 0.0
    pitch_deg: float = 0.0
    fov_deg: float = 90.0
    n_looks: int = 0
    n_maps: int = 0
    n_pins: int = 0
    n_moves: int = 0
    n_free: int = 0
    total_moved_meters: float = 0.0
    submitted: bool = False
    ended_without_guess: bool = False
    pins: list[dict[str, Any]] = Field(default_factory=list)
