# SPDX-License-Identifier: BSD-3-Clause

"""The GeoGuesser environment.

Exposes its tools over MCP so an agent can look around, walk, and check
candidate coordinates on a map before committing to a guess. Non-MCP
structured actions route through `_step_impl`, which is what makes single-shot
GRPO and the agentic loop the same environment rather than two.

Two rules are load-bearing:

- Pin feedback describes only where the agent pointed. Any signal about the
  target would make binary search optimal, and the benchmark would measure
  bisection instead of geography.
- Tools the backend cannot serve are never registered, rather than registered
  and failing, so a policy does not learn to spend steps on dead ends.
"""

from __future__ import annotations

import logging
import random
import uuid
from typing import Any

from fastmcp import FastMCP
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
from openenv.core.env_server.types import Action, Observation

from ..models import (
    EpisodeMode,
    from_wire,
    GeoGuesserAction,
    GeoGuesserObservation,
    GeoGuesserState,
    GuessAction,
    LookAction,
    MeasureAction,
    MoveAction,
    PanAction,
    Pin,
    PinAction,
    RewardMode,
    TypedAction,
    ViewMapAction,
    ZoomAction,
)
from .backends.base import Task
from .backends.panorama import PanoramaBackend
from .parser import parse_guess
from .render.minimap import (
    describe_pin,
    locate,
    render_map,
    street_detail_enabled,
    street_fetch_failed,
)
from .render.pano import to_base64
from .scoring import (
    action_cost,
    compute_reward,
    haversine_km,
    MAX_COST_FRACTION,
    verdict,
)


logger = logging.getLogger(__name__)

PROMPT = (
    "You are dropped at an unknown street-level location somewhere in the "
    "world. Work out where you are.\n\n"
    "Available tools: {tools}\n\n"
    "Looking around and checking the map cost a little reward each, so gather "
    "the evidence you need and then commit. You have {steps} actions. "
    "Placing a pin shows you where on the map that coordinate falls - it "
    "tells you nothing about whether you are right. Finish with "
    "submit_guess."
)


class UnknownSplitError(KeyError, IndexError):
    """A split name that is not configured.

    Inherits both exception types deliberately. `KeyError` is what a Python
    caller expects from a bad name, while the core Task API dispatcher maps
    only `NotImplementedError` and `IndexError` onto HTTP status codes -- so
    without `IndexError` an unknown split surfaces as a 500 instead of a 400.
    """


class GeoGuesserEnvironment(MCPEnvironment):
    """A GeoGuessr-style geolocation episode.

    Args:
        index_path (`str`):
            JSONL task index built by `dataset/build_pano_tasks.py`.
        cache_dir (`str`):
            Directory of cached panorama JPEGs.
        episode_mode (`str`, *optional*, defaults to `"agentic"`):
            One of `"agentic"`, `"single_shot"` or `"nmpz"`.
        max_steps (`int`, *optional*, defaults to `24`):
            Actions allowed before the episode is cut off. Twelve is tight
            for an agentic episode: looking in four directions and walking
            a block spends most of it before any reasoning about the map.
        reward_mode (`str`, *optional*, defaults to `"coords"`):
            `"coords"` scores distance; `"country_only"` scores the country.
        max_free_calls (`int`, *optional*, defaults to `24`):
            How many free-tool calls (`measure`) an episode may make before they
            begin consuming the step budget. Free tools return arithmetic on
            coordinates the agent supplied and so reveal nothing, but without a
            cap a policy can issue them indefinitely and never terminate.
        reward_shape (`str`, *optional*, defaults to `"geoguessr"`):
            Distance curve. `"geoguessr"` is the game's own; `"mixture"` adds a
            5000 km scale so a wrong-continent guess still has a gradient. Use
            `"mixture"` for training and `"geoguessr"` for anything you report.
        cost_mode (`str`, *optional*, defaults to `"subtract"`):
            `"subtract"` takes the action cost off the score and floors at zero,
            as the game does. `"multiply"` scales the score by `1 - cost`, which
            preserves the ordering of bad guesses -- required for training, see
            [`~scoring.compute_reward`].
        hide_task_identity (`bool`, *optional*, defaults to `False`):
            Drop `task_index`, `task_id`, `sequence_id` and `attribution` from
            the per-turn observation metadata. **Set this for RL training**: the
            contributor username alone determines the country for 74% of
            training tasks, so leaving it in lets a policy score without looking
            at the image. The terminal observation carries them either way.
        hierarchical_reward (`bool`, *optional*, defaults to `False`):
            Add country and region partial credit to the distance score.
        view_size (`int`, *optional*, defaults to `640`):
            Edge length in pixels of rendered views.
        allow_fetch (`bool`, *optional*, defaults to `True`):
            Whether a cache miss may reach the Mapillary API.
        hires_zoom (`bool`, *optional*, defaults to `True`):
            Render zoomed views from the full-resolution original, so a narrow
            field of view actually resolves detail such as distant signage.
        reveal_map (`bool`, *optional*, defaults to `True`):
            Draw a map of the guess against the truth on the terminal
            observation. Costs about 280 ms, which is the single largest cost in
            an episode, and a training run does not read it — the reward and the
            distance are in the observation either way. Leave it on for evals,
            demos and traces; turn it off for throughput.

    Examples:

    ```python
    env = GeoGuesserEnvironment("tasks/pano_v1.jsonl", "data/panos")
    observation = env.reset(task_index=0)
    result = env.step(LookAction(heading_deg=90))
    ```
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        index_path: str | None = None,
        cache_dir: str = "",
        episode_mode: str = EpisodeMode.AGENTIC.value,
        max_steps: int = 24,
        max_free_calls: int = 24,
        reward_mode: str = RewardMode.COORDS.value,
        hierarchical_reward: bool = False,
        reward_shape: str = "geoguessr",
        cost_mode: str = "subtract",
        hide_task_identity: bool = False,
        view_size: int = 640,
        allow_fetch: bool = True,
        hires_zoom: bool = True,
        reveal_map: bool = True,
        splits: dict[str, str] | None = None,
        default_split: str = "train",
    ):
        if splits:
            self._split_paths = {name: str(path) for name, path in splits.items()}
        elif index_path:
            # Single-index callers keep working: one nameless index becomes the
            # default split, so existing tests, harness runs and the play UI
            # need no change.
            self._split_paths = {default_split: str(index_path)}
        else:
            raise ValueError("Provide either splits= or index_path=.")
        if default_split not in self._split_paths:
            raise ValueError(
                f"default_split {default_split!r} is not one of "
                f"{sorted(self._split_paths)}."
            )
        self._default_split = default_split
        self._cache_dir = cache_dir
        self._allow_fetch = allow_fetch
        self._hires_zoom = hires_zoom
        self._backends: dict[str, PanoramaBackend] = {}
        self._split = default_split
        self._backend = self._backend_for(default_split)
        self._reveal_map = reveal_map
        self._mode = EpisodeMode(episode_mode)
        self._max_steps = max_steps
        # Free tools reveal nothing, so they stay free -- but not unlimited.
        self._max_free_calls = max_free_calls
        self._reward_mode = RewardMode(reward_mode)
        self._hierarchical = hierarchical_reward
        self._reward_shape = reward_shape
        self._cost_mode = cost_mode
        self._hide_task_identity = hide_task_identity
        # Validate now rather than at the end of the first episode.
        compute_reward(1.0, shape=reward_shape, cost_mode=cost_mode)
        self._view_size = (view_size, view_size)
        self._state = GeoGuesserState()
        self._task = None
        self._rng = random.Random()
        self._last_mcp_observation: GeoGuesserObservation | None = None

        mcp = FastMCP("geoguesser_env")
        self._register_tools(mcp)
        super().__init__(mcp)

    # -- splits ------------------------------------------------------------

    def _backend_for(self, split: str) -> PanoramaBackend:
        """
        Return the backend serving one split, building it on first use.

        Splits are built lazily so a deployment that only mounts the eval index
        is not forced to carry a training one, and because the parsed index is
        cached process-wide anyway.

        Args:
            split (`str`):
                Split name.

        Returns:
            [`PanoramaBackend`]: Backend for that split.

        Raises:
            UnknownSplitError: When the split is not configured.
        """
        if split not in self._split_paths:
            raise UnknownSplitError(
                f"Unknown split {split!r}. Available: {sorted(self._split_paths)}."
            )
        backend = self._backends.get(split)
        if backend is None:
            backend = PanoramaBackend(
                self._split_paths[split],
                self._cache_dir,
                allow_fetch=self._allow_fetch,
                hires_zoom=self._hires_zoom,
            )
            self._backends[split] = backend
        return backend

    @staticmethod
    def _split_type(split: str) -> str:
        """
        Map a split name onto the type vocabulary the core Task API knows.

        Core normalises anything outside `{train, validation, test}` to
        `validation`, so `eval` is declared as `test` explicitly rather than
        being silently downgraded.
        """
        if split == "train":
            return "train"
        if split in {"eval", "test"}:
            return "test"
        return "validation"

    def _task_spec(self, split: str, task: Task) -> dict[str, Any]:
        """
        Describe one task for the Task API.

        Deliberately truth-free: no coordinates and no country. Task specs
        travel to whatever orchestrates training, and the moment a label sits
        in a spec someone can build a prompt from it. The true location is
        revealed in the observation metadata after the guess, which is the one
        place it belongs.
        """
        return {
            "task_index": task.task_index,
            "task_id": task.task_id,
            "split": split,
            "n_frames": len(task.frames),
            "provider": task.provider,
            "sequence_id": task.sequence_id,
            "offline_ready": bool(task.meta.get("offline_ready", False)),
        }

    def list_splits(self) -> list[dict[str, Any]]:
        """
        Task API: describe every configured split.

        Returns:
            `list[dict]` with keys:
                - `name` (`str`):
                    Split name, as accepted by `reset(split=)`.
                - `type` (`str`):
                    One of `train`, `test` or `validation`.
                - `num_tasks` (`int`):
                    Task count in the split.
                - `default` (`bool`):
                    Whether `reset()` uses this split when none is given.
        """
        return [
            {
                "name": name,
                "type": self._split_type(name),
                "num_tasks": self._backend_for(name).n_tasks,
                "default": name == self._default_split,
            }
            for name in self._split_paths
        ]

    def num_tasks(self, split: str) -> int:
        """Task API: how many tasks a split holds."""
        return self._backend_for(split).n_tasks

    def get_task(self, split: str, index: int) -> dict[str, Any]:
        """Task API: describe one task by split and index."""
        return self._task_spec(split, self._backend_for(split).task(index))

    def list_tasks(self, split: str) -> list[dict[str, Any]]:
        """Task API: describe every task in a split."""
        backend = self._backend_for(split)
        return [
            self._task_spec(split, backend.task(index))
            for index in range(backend.n_tasks)
        ]

    def get_task_range(
        self, split: str, start: int | None = None, stop: int | None = None
    ) -> list[dict[str, Any]]:
        """Task API: describe a slice-style range of tasks in a split."""
        backend = self._backend_for(split)
        indices = range(*slice(start, stop).indices(backend.n_tasks))
        return [self._task_spec(split, backend.task(index)) for index in indices]

    # -- capability-aware tool registration --------------------------------

    def _navigational(self) -> bool:
        return self._mode is EpisodeMode.AGENTIC and self._backend.supports_move

    def _can_look(self) -> bool:
        return self._mode is EpisodeMode.AGENTIC and self._backend.supports_look

    def _can_gather(self) -> bool:
        """Whether the episode has an investigation phase at all.

        `single_shot` deliberately has none: one view, one guess, which is the
        shape a VLM GRPO run wants. `nmpz` keeps the map but takes the camera
        away, mirroring the game's own hardest mode.
        """
        return self._mode is not EpisodeMode.SINGLE_SHOT

    def _register_tools(self, mcp: FastMCP) -> None:
        """Register only the tools this configuration can actually serve."""
        if self._can_look():

            @mcp.tool
            def look(
                heading_deg: float, pitch_deg: float = 0.0, fov_deg: float = 90.0
            ) -> str:
                """Look in a direction. heading_deg is absolute, 0 = true north.

                Args:
                    heading_deg: Compass heading in degrees.
                    pitch_deg: Vertical angle; positive looks up.
                    fov_deg: Field of view; smaller values zoom in.
                """
                return self._mcp_feedback(
                    self._apply(
                        LookAction(
                            heading_deg=heading_deg,
                            pitch_deg=pitch_deg,
                            fov_deg=fov_deg,
                        )
                    )
                )

            @mcp.tool
            def pan(delta_deg: float) -> str:
                """Turn relative to the current heading; positive turns right.

                Args:
                    delta_deg: Degrees to turn.
                """
                return self._mcp_feedback(
                    self._apply(PanAction(delta_deg=delta_deg))
                )

            @mcp.tool
            def zoom(fov_deg: float) -> str:
                """Change field of view without turning. 30 reads distant signs.

                Args:
                    fov_deg: New field of view in degrees.
                """
                return self._mcp_feedback(self._apply(ZoomAction(fov_deg=fov_deg)))

        if self._navigational():

            @mcp.tool
            def move(direction: str, meters: float = 10.0) -> str:
                """Walk along the road. Reports how far you actually travelled.

                Args:
                    direction: Either "forward" or "backward".
                    meters: Requested distance in metres.
                """
                return self._mcp_feedback(
                    self._apply(MoveAction(direction=direction, meters=meters))
                )

        if self._can_gather():
            self._register_map_tools(mcp)
        self._register_guess_tool(mcp)

    def _register_map_tools(self, mcp: FastMCP) -> None:
        """Register the map and pin tools, which every gathering mode has."""

        @mcp.tool
        def place_pin(
            lat: float, lon: float, label: str = "", span_deg: float = 7.0
        ) -> str:
            """Pin a candidate and see where it falls on the map.

            Tells you what is at that coordinate. Says nothing about whether
            you are right.

            Args:
                lat: Latitude of the candidate.
                lon: Longitude of the candidate.
                label: Optional note.
                span_deg: Half-width of the map window in degrees. Below about
                    4 the map adds roads, urban areas and town names, which is
                    how you aim within a city rather than at its centre.
            """
            return self._mcp_feedback(
                self._apply(
                    PinAction(lat=lat, lon=lon, label=label or None, span_deg=span_deg)
                )
            )

        @mcp.tool
        def view_map(lat: float, lon: float, span_deg: float = 7.0) -> str:
            """Pan and zoom the map without placing a pin.

            Args:
                lat: Latitude at the centre of the view.
                lon: Longitude at the centre of the view.
                span_deg: Half-width of the window in degrees.
            """
            return self._mcp_feedback(
                self._apply(ViewMapAction(lat=lat, lon=lon, span_deg=span_deg))
            )

        @mcp.tool
        def list_pins() -> str:
            """List the candidates pinned so far. Free."""
            if not self._state.pins:
                return "No pins placed yet."
            return "\n".join(
                f"{i}. {p['lat']:.4f}, {p['lon']:.4f} - {p['description']}"
                for i, p in enumerate(self._state.pins, 1)
            )

        @mcp.tool
        def clear_pins() -> str:
            """Remove all pins. Free."""
            self._state.pins = []
            return "Pins cleared."

        @mcp.tool
        def measure(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> str:
            """Distance in km between two coordinates of your own choosing. Free.

            Args:
                lat_a: Latitude of the first point.
                lon_a: Longitude of the first point.
                lat_b: Latitude of the second point.
                lon_b: Longitude of the second point.
            """
            km = haversine_km(lat_a, lon_a, lat_b, lon_b)
            return f"{km:.0f} km between those two points."

        @mcp.tool
        def reverse_geocode(lat: float, lon: float) -> str:
            """Name the country and nearest city at a coordinate. Free.

            Args:
                lat: Latitude in degrees.
                lon: Longitude in degrees.
            """
            place = locate(lat, lon)
            where = place.country or "open water"
            return (
                f"{lat:.4f}, {lon:.4f} is in {where}. Nearest major city: "
                f"{place.nearest_city}, ~{place.city_distance_km:.0f} km "
                f"{place.city_bearing}."
            )

    def _register_guess_tool(self, mcp: FastMCP) -> None:
        """Register the terminal action, which every mode has."""

        @mcp.tool
        def submit_guess(
            lat: float,
            lon: float,
            country: str = "",
            confidence: float = -1.0,
            reasoning: str = "",
        ) -> str:
            """Commit your final answer. Ends the episode.

            Args:
                lat: Latitude of your guess.
                lon: Longitude of your guess.
                country: Optional ISO-3166 alpha-2 code or country name.
                confidence: Optional self-reported confidence in [0, 1].
                reasoning: Optional rationale, recorded but not scored.
            """
            observation = self._apply(
                GuessAction(
                    lat=lat,
                    lon=lon,
                    country=country or None,
                    confidence=None if confidence < 0 else confidence,
                    reasoning=reasoning or None,
                )
            )
            return self._mcp_feedback(observation)

    def _mcp_feedback(self, observation: GeoGuesserObservation) -> str:
        self._last_mcp_observation = observation
        return observation.feedback

    @staticmethod
    def _terminal_mcp_metadata(
        observation: GeoGuesserObservation,
    ) -> dict[str, Any]:
        return {
            **(observation.metadata or {}),
            "feedback": observation.feedback,
            "distance_km": observation.distance_km,
            "score": observation.score,
            "action_cost": observation.action_cost,
            "true_lat": observation.true_lat,
            "true_lon": observation.true_lon,
            "parsed_ok": observation.parsed_ok,
        }

    def _propagate_mcp_terminal(
        self, call: CallToolObservation
    ) -> CallToolObservation:
        observation = self._last_mcp_observation
        self._last_mcp_observation = None
        if observation is not None and observation.done and call.error is None:
            call.done = True
            call.reward = observation.reward
            call.metadata = self._terminal_mcp_metadata(observation)
        return call

    def _handle_call_tool(
        self,
        action: CallToolAction,
        timeout_s: float | None = None,
    ) -> CallToolObservation:
        self._last_mcp_observation = None
        return self._propagate_mcp_terminal(
            super()._handle_call_tool(action, timeout_s=timeout_s)
        )

    async def _async_handle_call_tool(
        self,
        action: CallToolAction,
        timeout_s: float | None = None,
    ) -> CallToolObservation:
        self._last_mcp_observation = None
        return self._propagate_mcp_terminal(
            await super()._async_handle_call_tool(action, timeout_s=timeout_s)
        )

    # -- lifecycle ---------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        split: str | None = None,
        index: int | None = None,
        task_index: int | None = None,
        **kwargs: Any,
    ) -> GeoGuesserObservation:
        """
        Start an episode.

        Selection is explicit, because training and demoing want opposite
        things. `index` picks one exact task and is byte-identical on repeat,
        which is what a GRPO group needs. `seed` picks `tasks[seed % n_tasks]`.
        Neither means a random task; omitting both does, and the chosen split
        and index are always reported in the observation metadata so a random
        episode stays replayable.

        Args:
            seed (`int`, *optional*):
                Deterministic selector, `tasks[seed % n_tasks]`.
            episode_id (`str`, *optional*):
                Caller-supplied episode identifier.
            split (`str`, *optional*):
                Which split to draw from. Defaults to the environment's default
                split.
            index (`int`, *optional*):
                Exact task to play, within `split`. Takes precedence over
                `seed`.
            task_index (`int`, *optional*):
                Deprecated alias for `index`, kept so existing callers and
                saved trajectories keep working.

        Returns:
            [`GeoGuesserObservation`]: The opening view and prompt.

        Raises:
            UnknownSplitError: When `split` is not configured.
        """
        self._split = split or self._default_split
        self._backend = self._backend_for(self._split)

        if index is None:
            index = task_index
        n = self._backend.n_tasks
        if index is not None:
            # An explicit index is an address, so a bad one is an error rather
            # than something to wrap: silently scoring task 100 when the harness
            # asked for task 500 is a measurement bug you cannot see.
            chosen = int(index)
            if not 0 <= chosen < n:
                raise IndexError(
                    f"task index {chosen} out of range for split "
                    f"{self._split!r} with {n} tasks"
                )
        elif seed is not None:
            # A seed is not an address, so wrapping is the point.
            chosen = int(seed) % n
        else:
            chosen = self._rng.randrange(n)

        self._task = self._backend.task(chosen)
        start = self._task.frames[self._task.start_frame]
        self._state = GeoGuesserState(
            episode_id=episode_id or str(uuid.uuid4()),
            task_index=chosen,
            task_id=self._task.task_id,
            frame_index=self._task.start_frame,
            heading_deg=0.0,
            pitch_deg=0.0,
            fov_deg=90.0,
        )

        observation = self._render_view_observation()
        observation.prompt = PROMPT.format(
            tools=", ".join(self._tool_names()), steps=self._max_steps
        )
        observation.feedback = "Episode started."
        observation.captured_at = start.captured_at
        observation.metadata = self._metadata()
        return observation

    @property
    def state(self) -> GeoGuesserState:
        """Current internal state."""
        return self._state

    def _step_impl(self, action: Action, **kwargs: Any) -> Observation:
        """Handle structured, non-MCP actions.

        Accepts either the flat wire action the HTTP layer delivers or a typed
        action constructed in process, so tests and the harness can bypass
        serialisation without a second code path.
        """
        if isinstance(action, GeoGuesserAction):
            return self._apply(from_wire(action))
        if isinstance(action, TypedAction):
            return self._apply(action)
        raise TypeError(f"Unsupported action type: {type(action).__name__}")

    # -- the actual mechanics ---------------------------------------------

    def _tool_names(self) -> list[str]:
        """Names of the tools this configuration actually registered."""
        names: list[str] = []
        if self._can_look():
            names += ["look", "pan", "zoom"]
        if self._navigational():
            names.append("move")
        if self._can_gather():
            names += [
                "place_pin",
                "view_map",
                "list_pins",
                "clear_pins",
                "measure",
                "reverse_geocode",
            ]
        names.append("submit_guess")
        return names

    def _steps_used(self) -> int:
        s = self._state
        return s.n_looks + s.n_maps + s.n_pins + s.n_moves

    def _steps_remaining(self) -> int:
        return max(0, self._max_steps - self._steps_used())

    def _cost(self) -> float:
        s = self._state
        return action_cost(
            n_looks=s.n_looks, n_maps=s.n_maps, n_pins=s.n_pins, n_moves=s.n_moves
        )

    def _metadata(self) -> dict[str, Any]:
        # Identity fields are a reward-hacking channel, not just clutter. The
        # Mapillary contributor determines the country outright for 74% of
        # training tasks ("amsterdam" only maps the Netherlands), and
        # task_index/task_id/sequence_id are a few thousand memorisable keys
        # straight to a coordinate -- either lets a policy score without ever
        # reading the image. Kept by default so the play UI can show the licence
        # credit and recordings keep their provenance; RL training must set
        # hide_task_identity=True. The terminal observation carries them
        # regardless, since by then the truth is already revealed.
        if self._hide_task_identity:
            return {
                "split": self._split,
                "street_detail": (
                    "unavailable"
                    if street_fetch_failed()
                    else ("on" if street_detail_enabled() else "off")
                ),
                "backend": "mapillary",
                "episode_mode": self._mode.value,
                "frame_index": self._state.frame_index,
            }
        return {
            "split": self._split,
            # A map quietly missing its streets looks like a styling choice, so
            # say so. Overpass 504s from datacenter egress, which is how a Space
            # ends up with poorer maps than a laptop for the same task.
            "street_detail": (
                "unavailable"
                if street_fetch_failed()
                else ("on" if street_detail_enabled() else "off")
            ),
            "task_index": self._state.task_index,
            "task_id": self._state.task_id,
            "backend": "mapillary",
            # Opaque upstream identifier, not ground truth: it makes a recorded
            # episode traceable back to its source sequence.
            "sequence_id": self._task.sequence_id if self._task else None,
            "episode_mode": self._mode.value,
            "frame_index": self._state.frame_index,
            "captured_at": self._task.frames[self._state.frame_index].captured_at,
            "attribution": self._task.attribution,
        }

    def _base_observation(self) -> GeoGuesserObservation:
        s = self._state
        return GeoGuesserObservation(
            heading_deg=s.heading_deg % 360,
            pitch_deg=s.pitch_deg,
            fov_deg=s.fov_deg,
            total_moved_meters=s.total_moved_meters,
            can_move_forward=self._navigational()
            and self._backend.can_move(self._task, s.frame_index, "forward"),
            can_move_backward=self._navigational()
            and self._backend.can_move(self._task, s.frame_index, "backward"),
            available_tools=self._tool_names(),
            steps_remaining=self._steps_remaining(),
            action_cost=self._cost(),
            pins=[Pin(**p) for p in s.pins],
            captured_at=self._task.frames[s.frame_index].captured_at,
            metadata=self._metadata(),
        )

    def _render_view_observation(self) -> GeoGuesserObservation:
        s = self._state
        view = self._backend.render_view(
            self._task, s.frame_index, s.heading_deg, s.pitch_deg, s.fov_deg
        )
        if view.size != self._view_size:
            view = view.resize(self._view_size)
        observation = self._base_observation()
        observation.image_base64 = to_base64(view, "JPEG")
        observation.image_kind = "view"
        return observation

    def _render_map_observation(
        self,
        pins: list[tuple[float, float]],
        focus,
        span: float,
        truth: tuple[float, float] | None = None,
    ) -> GeoGuesserObservation:
        image = render_map(pins, focus=focus, span_deg=span, truth=truth)
        observation = self._base_observation()
        observation.image_base64 = to_base64(image, "PNG")
        observation.image_kind = "map"
        return observation

    def _apply(self, action: Action) -> GeoGuesserObservation:
        """Execute one action and produce the resulting observation."""
        if self._task is None:
            raise RuntimeError("reset() must be called before step().")

        s = self._state
        s.step_count += 1

        if s.submitted:
            observation = self._base_observation()
            observation.done = True
            observation.feedback = (
                "Out of actions. The episode is over; it ended with no guess."
                if s.ended_without_guess
                else "The episode is over; the guess was already made."
            )
            return observation

        if isinstance(action, GuessAction):
            return self._finish(action)

        if self._steps_remaining() <= 0:
            return self._finish_without_guess()

        if isinstance(action, LookAction):
            s.heading_deg = action.heading_deg
            s.pitch_deg = action.pitch_deg
            s.fov_deg = action.fov_deg
            s.n_looks += 1
            observation = self._render_view_observation()
            observation.feedback = (
                f"Facing {s.heading_deg % 360:.0f} deg, {s.fov_deg:.0f} deg field of "
                f"view. {self._steps_remaining()} actions left."
            )
            return self._maybe_finish_without_guess(observation)

        if isinstance(action, PanAction):
            s.heading_deg = (s.heading_deg + action.delta_deg) % 360
            s.n_looks += 1
            observation = self._render_view_observation()
            observation.feedback = (
                f"Turned to {s.heading_deg:.0f} deg. "
                f"{self._steps_remaining()} actions left."
            )
            return self._maybe_finish_without_guess(observation)

        if isinstance(action, ZoomAction):
            s.fov_deg = action.fov_deg
            s.n_looks += 1
            observation = self._render_view_observation()
            observation.feedback = (
                f"Field of view now {s.fov_deg:.0f} deg. "
                f"{self._steps_remaining()} actions left."
            )
            return self._maybe_finish_without_guess(observation)

        if isinstance(action, MoveAction):
            new_index, travelled = self._backend.step_along(
                self._task, s.frame_index, action.direction, action.meters
            )
            s.n_moves += 1
            if new_index == s.frame_index:
                observation = self._render_view_observation()
                observation.feedback = (
                    f"Cannot go {action.direction} - the captured road ends here. "
                    f"{self._steps_remaining()} actions left."
                )
                return self._maybe_finish_without_guess(observation)
            s.frame_index = new_index
            s.total_moved_meters += travelled
            observation = self._render_view_observation()
            observation.moved_meters = travelled
            observation.feedback = (
                f"Moved {travelled:.0f} m {action.direction} "
                f"({s.total_moved_meters:.0f} m total). "
                f"{self._steps_remaining()} actions left."
            )
            return self._maybe_finish_without_guess(observation)

        if isinstance(action, PinAction):
            previous = (s.pins[-1]["lat"], s.pins[-1]["lon"]) if s.pins else None
            description = describe_pin(
                len(s.pins) + 1, action.lat, action.lon, previous=previous
            )
            s.pins.append(
                {
                    "index": len(s.pins) + 1,
                    "lat": action.lat,
                    "lon": action.lon,
                    "label": action.label,
                    "description": description,
                }
            )
            s.n_pins += 1
            pins = [(p["lat"], p["lon"]) for p in s.pins]
            observation = self._render_map_observation(
                pins, (action.lat, action.lon), action.span_deg
            )
            observation.feedback = (
                f"{description} {self._steps_remaining()} actions left."
            )
            return self._maybe_finish_without_guess(observation)

        if isinstance(action, ViewMapAction):
            s.n_maps += 1
            pins = [(p["lat"], p["lon"]) for p in s.pins]
            observation = self._render_map_observation(
                pins, (action.lat, action.lon), action.span_deg
            )
            place = locate(action.lat, action.lon)
            observation.feedback = (
                f"Map centred on {action.lat:.3f}, {action.lon:.3f} "
                f"({place.country or 'open water'}), "
                f"{action.span_deg * 2:.0f} deg across. "
                f"{self._steps_remaining()} actions left."
            )
            return self._maybe_finish_without_guess(observation)

        if isinstance(action, MeasureAction):
            # Free, because it is arithmetic on two coordinates the agent
            # supplied: it reveals nothing about where the agent is. But free
            # used to mean unbounded -- it incremented no counter, so it never
            # advanced the step budget and a policy could issue it forever. Past
            # the cap it starts costing a map action, so the episode terminates.
            if s.n_free >= self._max_free_calls:
                s.n_maps += 1
            else:
                s.n_free += 1
            km = haversine_km(action.lat_a, action.lon_a, action.lat_b, action.lon_b)
            observation = self._base_observation()
            observation.feedback = f"{km:.0f} km between those two points."
            if s.n_free >= self._max_free_calls:
                observation.feedback += " Free-tool budget spent; this now costs."
            return self._maybe_finish_without_guess(observation)

        raise TypeError(f"Unsupported action type: {type(action).__name__}")

    def _maybe_finish_without_guess(
        self, observation: GeoGuesserObservation
    ) -> GeoGuesserObservation:
        if self._steps_remaining() <= 0:
            return self._finish_without_guess(observation)
        return observation

    def _finish_without_guess(
        self, observation: GeoGuesserObservation | None = None
    ) -> GeoGuesserObservation:
        """End an exhausted episode with the documented no-guess reward."""
        s = self._state
        s.submitted = True
        s.ended_without_guess = True
        true_lat, true_lon = self._task.truth
        observation = observation or self._base_observation()
        observation.done = True
        observation.reward = 0.0
        observation.score = 0.0
        observation.parsed_ok = False
        observation.true_lat = true_lat
        observation.true_lon = true_lon
        observation.action_cost = self._cost()
        observation.feedback = (
            f"{observation.feedback.rstrip()} Out of actions. "
            "Episode ended with no guess. Scored 0."
        ).strip()
        observation.metadata = {
            **self._metadata(),
            "no_guess": True,
            "country": self._task.country,
            "task_index": self._state.task_index,
            "task_id": self._state.task_id,
            "sequence_id": self._task.sequence_id,
            "attribution": self._task.attribution,
            "n_looks": s.n_looks,
            "n_maps": s.n_maps,
            "n_pins": s.n_pins,
            "n_moves": s.n_moves,
            "total_moved_meters": s.total_moved_meters,
        }
        return observation

    def _finish(self, action: GuessAction) -> GeoGuesserObservation:
        """Score the final guess and end the episode."""
        s = self._state
        s.submitted = True
        true_lat, true_lon = self._task.truth

        if action.lat is not None and action.lon is not None:
            lat, lon, parsed_ok, note = action.lat, action.lon, True, ""
        else:
            parsed = parse_guess(action.response or "")
            lat, lon, parsed_ok, note = parsed.lat, parsed.lon, parsed.ok, parsed.note

        cost = self._cost()
        observation = self._base_observation()
        observation.done = True
        observation.parsed_ok = parsed_ok
        observation.true_lat = true_lat
        observation.true_lon = true_lon
        observation.action_cost = cost

        if not parsed_ok:
            observation.reward = 0.0
            observation.score = 0.0
            observation.feedback = f"No usable guess. {note} Scored 0."
            observation.metadata = {
                **self._metadata(),
                "parse_failure": True,
                "country": self._task.country,
                "task_index": self._state.task_index,
                "task_id": self._state.task_id,
                "sequence_id": self._task.sequence_id,
                "attribution": self._task.attribution,
            }
            return observation

        distance = haversine_km(lat, lon, true_lat, true_lon)

        # A guess previously returned no image at all, which left the outcome
        # invisible in a trace and gave a policy nothing to learn the shape of
        # its error from. Truth is only ever drawn here, after scoring.
        if not self._reveal_map:
            observation.image_kind = "none"
        separation = max(abs(lat - true_lat), abs(lon - true_lon))
        if self._reveal_map and separation > 25.0:
            # Framing both points would squash a hemisphere into the panel and
            # tell you nothing. The useful second view is where it actually was.
            reveal_focus = (true_lat, true_lon)
            reveal_span = 12.0
        elif self._reveal_map:
            reveal_focus = ((lat + true_lat) / 2, (lon + true_lon) / 2)
            reveal_span = max(0.05, separation * 0.75 + 0.4)
        if self._reveal_map:
            reveal = self._render_map_observation(
                [(lat, lon)], reveal_focus, reveal_span, truth=(true_lat, true_lon)
            )
            observation.image_base64 = reveal.image_base64
            observation.image_kind = "map"

        truth_place = locate(true_lat, true_lon)
        guess_place = locate(lat, lon)
        # The task index carries the country the imagery was harvested under, so
        # prefer it over reverse-geocoding the truth: Natural Earth 110m
        # resolves to None for a point a few metres offshore, and 1 of the 200
        # eval truths does exactly that, which made that task unwinnable in
        # country_only mode and had its reveal say "open water".
        truth_country = self._task.country or truth_place.country
        country_hit = bool(truth_country and truth_country == guess_place.country)
        region_hit = bool(
            truth_place.subregion and truth_place.subregion == guess_place.subregion
        )

        if self._reward_mode is RewardMode.COUNTRY_ONLY:
            reward = max(0.0, float(country_hit) - cost)
            score = float(country_hit)
        else:
            score = compute_reward(
                distance,
                cost=0.0,
                country_hit=country_hit,
                region_hit=region_hit,
                hierarchical=self._hierarchical,
                shape=self._reward_shape,
            )
            reward = compute_reward(
                distance,
                cost=cost,
                country_hit=country_hit,
                region_hit=region_hit,
                hierarchical=self._hierarchical,
                shape=self._reward_shape,
                cost_mode=self._cost_mode,
            )

        observation.distance_km = distance
        observation.score = score
        observation.reward = reward
        observation.feedback = (
            f"{verdict(distance)} - {distance:.0f} km away. True location "
            f"{true_lat:.4f}, {true_lon:.4f} "
            f"({truth_country or 'open water'}). "
            + (
                f"Score {score:.3f} scaled by "
                f"{1 - min(cost, MAX_COST_FRACTION):.2f} "
                f"action cost = {reward:.3f}."
                if self._cost_mode == "multiply"
                else f"Score {score:.3f} minus {cost:.2f} action cost = {reward:.3f}."
            )
        )
        observation.metadata = {
            **self._metadata(),
            # Revealed here and only here, next to true_lat/true_lon: the
            # country is the label a per-region breakdown needs, and a task spec
            # deliberately does not carry it.
            "country": self._task.country,
            # Restored here even under hide_task_identity: the episode is over,
            # so provenance can no longer be used to shortcut it.
            "task_index": self._state.task_index,
            "task_id": self._state.task_id,
            "sequence_id": self._task.sequence_id,
            "attribution": self._task.attribution,
            "guess": [lat, lon],
            "guess_country": guess_place.country,
            "verdict": verdict(distance),
            "distance_km": distance,
            "country_hit": country_hit,
            "region_hit": region_hit,
            "confidence": action.confidence,
            "reasoning": action.reasoning,
            "n_looks": s.n_looks,
            "n_maps": s.n_maps,
            "n_pins": s.n_pins,
            "n_moves": s.n_moves,
            "total_moved_meters": s.total_moved_meters,
        }
        return observation
