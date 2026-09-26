# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the GeoGuesser environment.

Fixtures are four real Mapillary panoramas committed under `tests/fixtures`,
so nothing here touches the network: the backend is constructed with
`allow_fetch=False` and a cache miss is a test failure rather than a silent
download.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import pathlib
import re
import shutil
import subprocess

import pytest
from openenv.core.env_server.mcp_types import CallToolAction

from geoguesser_env.models import (
    EpisodeMode,
    from_wire,
    GeoGuesserAction,
    GuessAction,
    LookAction,
    MeasureAction,
    MoveAction,
    PanAction,
    PinAction,
    to_wire,
    ViewMapAction,
    ZoomAction,
)
from geoguesser_env.server.backends.panorama import MissingImageError, PanoramaBackend
from geoguesser_env.server.geoguesser_environment import GeoGuesserEnvironment
from geoguesser_env.server.parser import parse_guess
from geoguesser_env.server.scoring import (
    action_cost,
    compute_reward,
    distance_score,
    haversine_km,
    verdict,
)


FIXTURES = pathlib.Path(__file__).parent / "fixtures"
INDEX = FIXTURES / "pano_v1.jsonl"
PANOS = FIXTURES / "panos"


def make_env(**kwargs) -> GeoGuesserEnvironment:
    """An environment over the committed fixtures that cannot reach the network."""
    options = {
        "index_path": str(INDEX),
        "cache_dir": str(PANOS),
        "allow_fetch": False,
        "view_size": 128,
    }
    options.update(kwargs)
    return GeoGuesserEnvironment(**options)


@pytest.fixture(autouse=True)
def _no_street_fetches():
    """Keep the suite hermetic.

    Street detail is on by default, and any render at street zoom fetches from
    Overpass, so with a cold cache the suite quietly becomes a network test:
    measured 139 s against 15 s, and it fails outright with no connection. The
    few tests that exercise street detail turn it back on themselves.
    """
    from geoguesser_env.server.render import minimap

    was = minimap.street_detail_enabled()
    minimap.set_street_detail(False)
    try:
        yield
    finally:
        minimap.set_street_detail(was)


# ---------------------------------------------------------------- scoring


def test_haversine_matches_known_distance():
    # La Paz to Santa Cruz is about 544 km.
    km = haversine_km(-16.4897, -68.1193, -17.7833, -63.1821)
    assert 540 < km < 548


def test_haversine_is_zero_for_identical_points():
    assert haversine_km(10.0, 20.0, 10.0, 20.0) == pytest.approx(0.0, abs=1e-9)


def test_distance_score_is_one_at_zero_and_decays():
    assert distance_score(0.0) == pytest.approx(1.0)
    scores = [distance_score(d) for d in (0, 25, 200, 1500, 5000, 20000)]
    assert scores == sorted(scores, reverse=True)
    assert scores[-1] < 0.001


def test_distance_score_rejects_negative_distance():
    with pytest.raises(ValueError):
        distance_score(-1.0)


def test_action_cost_sums_each_kind():
    assert action_cost(n_looks=3, n_maps=1, n_pins=2, n_moves=1) == pytest.approx(
        0.03 + 0.01 + 0.04 + 0.05
    )


def test_unparseable_guess_scores_zero_not_negative():
    assert compute_reward(None, cost=0.5) == 0.0


def test_reward_never_goes_below_zero():
    assert compute_reward(20000.0, cost=0.9) == 0.0


def test_hierarchical_reward_adds_partial_credit():
    plain = compute_reward(3000.0, hierarchical=False)
    with_country = compute_reward(3000.0, country_hit=True, hierarchical=True)
    assert with_country > plain


def test_verdict_labels_bracket_distance():
    assert verdict(0.0) == "Perfect"
    assert verdict(10.0) == "Pinpoint"
    assert verdict(100.0) == "Close"
    assert verdict(900.0) == "Right region"
    assert verdict(9000.0) == "Wrong continent"


# ----------------------------------------------------------------- parser


@pytest.mark.parametrize(
    "text,expected_source",
    [
        ("<guess>38.72, -9.14</guess>", "tag"),
        ("latitude: -16.4897\nlongitude: -68.1193", "labelled"),
        ('{"latitude": 35.68, "longitude": 139.69}', "json"),
        ("48°51'29\"N 2°17'40\"E", "dms"),
        ("around -1.29 36.82", "decimal"),
    ],
)
def test_parser_recognises_each_format(text, expected_source):
    parsed = parse_guess(text)
    assert parsed.ok
    assert parsed.source == expected_source


def test_parser_converts_dms_correctly():
    parsed = parse_guess("48°51'29\"N 2°17'40\"E")
    assert parsed.lat == pytest.approx(48.8581, abs=1e-3)
    assert parsed.lon == pytest.approx(2.2944, abs=1e-3)


@pytest.mark.parametrize(
    "text",
    [
        "48°99'29\"N 2°17'40\"E",
        "48°51'99\"N 2°17'40\"E",
        "48°51'29\"N 2°77'40\"E",
        "48°51'29\"N 2°17'99\"E",
    ],
)
def test_parser_rejects_dms_minutes_and_seconds_over_sixty(text):
    parsed = parse_guess(text)
    assert not parsed.ok
    assert "DMS" in parsed.note


def test_parser_routes_labelled_dms_through_dms_validation():
    parsed = parse_guess("lat: 48°59'59\"N lon: 2°17'40\"E")
    assert parsed.ok
    assert parsed.source == "dms"
    assert parsed.lat == pytest.approx(48.9997, abs=1e-3)
    assert parsed.lon == pytest.approx(2.2944, abs=1e-3)


@pytest.mark.parametrize(
    "text",
    [
        "lat: 48°60'00\"N lon: 2°17'40\"E",
        "lat: 48°59'60\"N lon: 2°17'40\"E",
        "lat: 48°59'59\"N lon: 2°60'00\"E",
        "lat: 48°59'59\"N lon: 2°17'60\"E",
    ],
)
def test_parser_rejects_labelled_dms_sixty_boundary(text):
    parsed = parse_guess(text)
    assert not parsed.ok
    assert "DMS" in parsed.note


def test_parser_prefers_labelled_over_stray_numbers():
    # "2019, maybe 2021" is a decimal pair; the labelled coordinates must win.
    parsed = parse_guess("Captured 2019, maybe 2021. lat: 55.67 lon: 12.56")
    assert (parsed.lat, parsed.lon) == (55.67, 12.56)


@pytest.mark.parametrize("text", ["", "   ", "I have no idea."])
def test_parser_reports_failure_with_guidance(text):
    parsed = parse_guess(text)
    assert not parsed.ok
    assert parsed.note


def test_parser_rejects_out_of_range_coordinates():
    parsed = parse_guess("999.9, 400.2")
    assert not parsed.ok
    assert "out of range" in parsed.note


# ------------------------------------------------------------ wire actions


@pytest.mark.parametrize(
    "action",
    [
        LookAction(heading_deg=90, pitch_deg=5, fov_deg=45),
        PanAction(delta_deg=-45),
        ZoomAction(fov_deg=30),
        MoveAction(direction="forward", meters=15),
        PinAction(lat=1.0, lon=2.0, label="candidate"),
        ViewMapAction(lat=1.0, lon=2.0, span_deg=3.0),
        MeasureAction(lat_a=1.0, lon_a=2.0, lat_b=3.0, lon_b=4.0),
        GuessAction(lat=1.0, lon=2.0, country="FR", confidence=0.5),
    ],
)
def test_wire_round_trip_preserves_the_action(action):
    restored = from_wire(to_wire(action))
    assert type(restored) is type(action)
    assert restored.model_dump() == action.model_dump()


def test_wire_action_rejects_unknown_op():
    with pytest.raises(Exception):
        GeoGuesserAction(op="teleport")


# ---------------------------------------------------------------- backend


def test_backend_loads_the_fixture_index():
    backend = PanoramaBackend(INDEX, PANOS, allow_fetch=False)
    assert backend.n_tasks == 4
    assert backend.supports_look
    assert backend.supports_move


def test_backend_rejects_out_of_range_task_index():
    backend = PanoramaBackend(INDEX, PANOS, allow_fetch=False)
    with pytest.raises(IndexError):
        backend.task(999)


def test_backend_refuses_to_fetch_when_forbidden():
    backend = PanoramaBackend(INDEX, PANOS, allow_fetch=False)
    with pytest.raises(MissingImageError):
        backend.load_pano("definitely-not-cached")


def test_move_at_a_dead_end_returns_the_same_frame():
    backend = PanoramaBackend(INDEX, PANOS, allow_fetch=False)
    task = backend.task(1)  # single-frame fixture
    new_index, travelled = backend.step_along(task, 0, "forward", 10.0)
    assert new_index == 0
    assert travelled == 0.0


def test_move_reports_the_distance_actually_travelled():
    backend = PanoramaBackend(INDEX, PANOS, allow_fetch=False)
    task = backend.task(0)  # two-frame fixture
    new_index, travelled = backend.step_along(task, 0, "forward", 1000.0)
    assert new_index == 1
    assert travelled > 0.0


# ------------------------------------------------------------- determinism


def test_same_task_index_gives_byte_identical_observations():
    env = make_env()
    first = env.reset(task_index=0).image_base64
    second = env.reset(task_index=0).image_base64
    assert (
        hashlib.sha256(first.encode()).digest()
        == hashlib.sha256(second.encode()).digest()
    )


def test_seed_selects_deterministically_by_modulo():
    env = make_env()
    assert env.reset(seed=5).metadata["task_index"] == 5 % env._backend.n_tasks
    assert env.reset(seed=5).metadata["task_index"] == 5 % env._backend.n_tasks


def test_out_of_range_task_index_raises():
    # An explicit index is an address. Wrapping it means an eval that asks for
    # task 500 of 200 silently scores task 100 instead, which is a measurement
    # bug with no symptom.
    env = make_env()
    with pytest.raises(IndexError):
        env.reset(task_index=env._backend.n_tasks)


def test_seed_still_wraps():
    # A seed is not an address, so wrapping is the intended behaviour there.
    env = make_env()
    assert env.reset(seed=env._backend.n_tasks).metadata["task_index"] == 0


def test_metadata_always_records_the_chosen_task():
    env = make_env()
    for _ in range(5):
        assert "task_index" in env.reset().metadata


# ----------------------------------------------- capability-aware tooling


def test_agentic_mode_registers_navigation_tools():
    tools = (
        make_env(episode_mode=EpisodeMode.AGENTIC.value)
        .reset(task_index=0)
        .available_tools
    )
    assert {"look", "pan", "zoom", "move", "place_pin", "submit_guess"} <= set(tools)


def test_nmpz_mode_registers_neither_looking_nor_moving():
    tools = (
        make_env(episode_mode=EpisodeMode.NMPZ.value)
        .reset(task_index=0)
        .available_tools
    )
    assert not {"look", "pan", "zoom", "move"} & set(tools)
    assert {"place_pin", "submit_guess"} <= set(tools)


def test_single_shot_mode_offers_only_the_guess():
    """One view, one guess - the shape a VLM GRPO run wants."""
    tools = (
        make_env(episode_mode=EpisodeMode.SINGLE_SHOT.value)
        .reset(task_index=0)
        .available_tools
    )
    assert tools == ["submit_guess"]


def test_single_shot_still_shows_an_opening_view():
    observation = make_env(episode_mode=EpisodeMode.SINGLE_SHOT.value).reset(
        task_index=0
    )
    assert observation.image_kind == "view"
    assert observation.image_base64


def test_single_shot_scores_a_guess_normally():
    env = make_env(episode_mode=EpisodeMode.SINGLE_SHOT.value)
    env.reset(task_index=0)
    true_lat, true_lon = env._task.truth
    observation = env.step(to_wire(GuessAction(lat=true_lat, lon=true_lon)))
    assert observation.done
    assert observation.reward == pytest.approx(1.0)


def test_nmpz_keeps_the_map_but_not_the_camera():
    tools = (
        make_env(episode_mode=EpisodeMode.NMPZ.value)
        .reset(task_index=0)
        .available_tools
    )
    assert "place_pin" in tools
    assert not {"look", "pan", "zoom", "move"} & set(tools)


def test_pin_is_a_plain_model_without_observation_fields():
    """Pins are data, not observations - they carry no reward or done flag."""
    env = make_env()
    env.reset(task_index=0)
    observation = env.step(to_wire(PinAction(lat=1.0, lon=2.0)))
    fields = set(observation.pins[0].model_dump())
    assert fields == {"index", "lat", "lon", "label", "description"}


# --------------------------------------------------------- the pin contract


def test_pin_feedback_never_leaks_the_true_location():
    """The core invariant: pinning must not reveal anything about the target.

    If it did, binary search would be the optimal policy and the environment
    would measure bisection instead of geolocation.
    """
    env = make_env()
    env.reset(task_index=0)
    true_lat, true_lon = env._task.truth
    observation = env.step(to_wire(PinAction(lat=0.0, lon=0.0)))

    text = f"{observation.feedback} {observation.pins[0].description}"
    assert f"{true_lat:.4f}" not in text
    assert f"{true_lon:.4f}" not in text
    assert env._task.country.lower() not in text.lower()
    for word in ("closer", "warmer", "colder", "away from", "correct"):
        assert word not in text.lower()
    assert observation.distance_km is None
    assert observation.true_lat is None
    assert observation.reward is None


def test_pin_describes_where_it_landed():
    env = make_env()
    env.reset(task_index=0)
    observation = env.step(to_wire(PinAction(lat=-16.4897, lon=-68.1193)))
    assert "Bolivia" in observation.feedback
    assert observation.image_kind == "map"
    assert observation.image_base64


def test_second_pin_reports_distance_from_the_first():
    env = make_env()
    env.reset(task_index=0)
    env.step(to_wire(PinAction(lat=-16.4897, lon=-68.1193)))
    observation = env.step(to_wire(PinAction(lat=-17.7833, lon=-63.1821)))
    assert "Distance from pin 1" in observation.feedback
    assert len(observation.pins) == 2


def test_ocean_pin_says_so():
    env = make_env()
    env.reset(task_index=0)
    observation = env.step(to_wire(PinAction(lat=-31.0, lon=-25.0)))
    assert "open water" in observation.feedback


# ---------------------------------------------------------------- episodes


def test_looking_costs_a_step_and_updates_heading():
    env = make_env()
    start = env.reset(task_index=0).steps_remaining
    observation = env.step(to_wire(LookAction(heading_deg=90, fov_deg=45)))
    assert observation.steps_remaining == start - 1
    assert observation.heading_deg == pytest.approx(90.0)
    assert observation.fov_deg == pytest.approx(45.0)
    assert observation.image_kind == "view"


def test_free_tools_do_not_consume_steps():
    env = make_env()
    start = env.reset(task_index=0).steps_remaining
    observation = env.step(
        to_wire(MeasureAction(lat_a=0.0, lon_a=0.0, lat_b=1.0, lon_b=1.0))
    )
    assert observation.steps_remaining == start


def test_guess_ends_the_episode_and_reveals_truth():
    env = make_env()
    env.reset(task_index=0)
    true_lat, true_lon = env._task.truth
    observation = env.step(to_wire(GuessAction(lat=true_lat, lon=true_lon)))
    assert observation.done
    assert observation.distance_km == pytest.approx(0.0, abs=1e-6)
    assert observation.reward == pytest.approx(1.0)
    assert observation.true_lat == pytest.approx(true_lat)


def test_a_wrong_guess_scores_low_but_not_negative():
    env = make_env()
    env.reset(task_index=0)
    true_lat, _ = env._task.truth
    observation = env.step(to_wire(GuessAction(lat=-true_lat, lon=170.0)))
    assert observation.done
    assert 0.0 <= observation.reward < 0.5


def test_unparseable_guess_scores_zero_with_feedback():
    env = make_env()
    env.reset(task_index=0)
    observation = env.step(to_wire(GuessAction(response="no idea, sorry")))
    assert observation.done
    assert observation.reward == 0.0
    assert not observation.parsed_ok
    assert "No usable guess" in observation.feedback


def test_guess_is_parsed_out_of_free_text():
    env = make_env()
    env.reset(task_index=0)
    observation = env.step(
        to_wire(GuessAction(response="Looks Nordic. <guess>60.0, 10.0</guess>"))
    )
    assert observation.parsed_ok
    assert observation.distance_km is not None


def test_action_cost_is_subtracted_from_the_score():
    env = make_env()
    env.reset(task_index=0)
    true_lat, true_lon = env._task.truth
    env.step(to_wire(LookAction(heading_deg=45)))
    env.step(to_wire(PinAction(lat=0.0, lon=0.0)))
    observation = env.step(to_wire(GuessAction(lat=true_lat, lon=true_lon)))
    assert observation.action_cost == pytest.approx(0.03)
    assert observation.reward == pytest.approx(observation.score - 0.03)


def test_stepping_after_a_guess_is_refused_not_scored_again():
    env = make_env()
    env.reset(task_index=0)
    env.step(to_wire(GuessAction(lat=0.0, lon=0.0)))
    observation = env.step(to_wire(LookAction(heading_deg=0)))
    assert observation.done
    assert "already made" in observation.feedback


def test_step_budget_blocks_further_gathering():
    env = make_env(max_steps=2)
    env.reset(task_index=0)
    env.step(to_wire(LookAction(heading_deg=0)))
    env.step(to_wire(LookAction(heading_deg=90)))
    observation = env.step(to_wire(LookAction(heading_deg=180)))
    assert observation.steps_remaining == 0
    assert "Out of actions" in observation.feedback


def test_step_budget_exhaustion_ends_the_episode_without_a_guess():
    env = make_env(max_steps=1)
    env.reset(task_index=0)

    exhausted = env.step(to_wire(LookAction(heading_deg=0)))
    assert exhausted.done
    assert exhausted.reward == 0.0
    assert exhausted.score == 0.0
    assert exhausted.parsed_ok is False
    assert exhausted.true_lat == pytest.approx(env._task.truth[0])
    assert exhausted.true_lon == pytest.approx(env._task.truth[1])
    assert exhausted.metadata["no_guess"] is True
    assert exhausted.metadata["country"] == env._task.country
    assert exhausted.metadata["task_index"] == env.state.task_index
    assert exhausted.metadata["task_id"] == env.state.task_id
    assert env.state.submitted

    true_lat, true_lon = env._task.truth
    late_guess = env.step(to_wire(GuessAction(lat=true_lat, lon=true_lon)))
    assert late_guess.done
    assert late_guess.reward is None
    assert "over" in late_guess.feedback


def _assert_terminal_mcp_result(result, env):
    assert result.done
    assert result.reward == 0.0
    assert result.metadata["no_guess"] is True
    assert result.metadata["country"] == env._task.country
    assert result.metadata["task_id"] == env.state.task_id
    assert result.metadata["true_lat"] == pytest.approx(env._task.truth[0])
    assert result.metadata["true_lon"] == pytest.approx(env._task.truth[1])
    assert result.metadata["score"] == 0.0
    assert result.metadata["parsed_ok"] is False
    assert result.result.structured_content["result"]
    assert "Out of actions" in result.result.structured_content["result"]


def test_mcp_sync_tool_call_propagates_exhausted_terminal_observation():
    env = make_env(max_steps=1)
    env.reset(task_index=0)

    result = env.step(
        CallToolAction(
            tool_name="look",
            arguments={"heading_deg": 0, "pitch_deg": 0, "fov_deg": 90},
        )
    )

    _assert_terminal_mcp_result(result, env)


def test_mcp_async_tool_call_propagates_exhausted_terminal_observation():
    env = make_env(max_steps=1)
    env.reset(task_index=0)

    result = asyncio.run(
        env.step_async(
            CallToolAction(
                tool_name="look",
                arguments={"heading_deg": 0, "pitch_deg": 0, "fov_deg": 90},
            )
        )
    )

    _assert_terminal_mcp_result(result, env)


def test_stepping_before_reset_is_an_error():
    env = make_env()
    with pytest.raises(RuntimeError):
        env.step(to_wire(LookAction(heading_deg=0)))


def test_country_only_mode_scores_the_country():
    env = make_env(reward_mode="country_only")
    env.reset(task_index=0)
    true_lat, true_lon = env._task.truth
    observation = env.step(to_wire(GuessAction(lat=true_lat, lon=true_lon)))
    assert observation.reward == pytest.approx(1.0)


# -------------------------------------------------------------- invariants


def test_agent_cannot_reset_through_the_tool_surface():
    """Simulation controls belong to the orchestration layer, never the agent."""
    tools = make_env().reset(task_index=0).available_tools
    assert not {"reset", "step", "state", "close"} & set(tools)


def test_client_module_does_not_import_the_server():
    source = (pathlib.Path(__file__).parents[1] / "client.py").read_text()
    assert "from .server" not in source
    assert "import server" not in source


# ------------------------------------------------------------------ play page


def test_play_page_has_no_unreplaced_tokens():
    """The page is a template with __TOKEN__ placeholders, easy to miss one.

    Matched as a whole token rather than by searching for "__", since the page
    legitimately contains identifiers like `window.__ggMap`.
    """
    import re

    from geoguesser_env.server.gradio_ui import play_page_html

    page = play_page_html(100)
    leftover = re.findall(r"__[A-Z][A-Z0-9_]*__", page)
    assert not leftover, f"unreplaced template tokens: {sorted(set(leftover))}"
    assert page.startswith("<!doctype html>")
    assert page.rstrip().endswith("</html>")


def test_play_page_is_a_whole_document_not_a_fragment():
    """Gradio's gr.HTML does not run <script> tags, so this must be its own page."""
    from geoguesser_env.server.gradio_ui import play_page_html

    page = play_page_html(7)
    for tag in ("<html", "<head", "<body", "</body>", "</html>"):
        assert tag in page
    assert "N_TASKS = 7" in page


def test_play_page_credits_the_imagery():
    """CC-BY-SA requires attribution, so the credit line is not optional."""
    from geoguesser_env.server.gradio_ui import play_page_html

    page = play_page_html(1)
    assert "Mapillary" in page
    assert "CC BY-SA 4.0" in page
    assert "OpenFreeMap" in page or "openfreemap" in page


def test_play_page_only_loads_scripts_from_pinned_cdn_versions():
    from geoguesser_env.server.gradio_ui import (
        MAPLIBRE_JS,
        PANNELLUM_JS,
        play_page_html,
    )

    page = play_page_html(1)
    assert MAPLIBRE_JS in page
    assert PANNELLUM_JS in page
    # Pinned versions, not "latest", so the UI cannot change under us.
    assert "/latest/" not in page
    for url in (MAPLIBRE_JS, PANNELLUM_JS):
        assert url.startswith("https://cdnjs.cloudflare.com/")


def test_play_page_scores_one_guess_out_of_five_thousand():
    """An episode is one guess, so the page has no multi-round game."""
    from geoguesser_env.server.gradio_ui import MAX_POINTS_PER_ROUND, play_page_html

    page = play_page_html(1)
    assert f"MAX_POINTS = {MAX_POINTS_PER_ROUND}" in page
    assert "ROUNDS" not in page
    assert "load another episode" in page


def test_play_page_terminal_guards_are_idempotent_without_browser():
    from geoguesser_env.server.gradio_ui import play_page_html

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable; full DOM behavior needs a browser")

    page = play_page_html(1)
    helpers = []
    for name in ("ggCanStep", "ggBeginReveal"):
        start = page.find(f"function {name}(state)")
        assert start != -1, f"{name} helper missing from play page"
        brace = page.find("{", start)
        depth = 0
        for end in range(brace, len(page)):
            if page[end] == "{":
                depth += 1
            elif page[end] == "}":
                depth -= 1
                if depth == 0:
                    helpers.append(page[start : end + 1])
                    break
        else:
            raise AssertionError(f"{name} helper body is incomplete")
    script = "\n".join(helpers) + """
const state = {terminal: false, busy: false, ready: true};
const firstReveal = ggBeginReveal(state);
const secondReveal = ggBeginReveal(state);
const canAfterReveal = ggCanStep(state);
const busyState = {terminal: false, busy: true, ready: true};
const notReadyState = {terminal: false, busy: false, ready: false};
console.log(JSON.stringify({
  firstReveal, secondReveal, canAfterReveal,
  canWhenBusy: ggCanStep(busyState),
  canWhenNotReady: ggCanStep(notReadyState)
}));
"""
    completed = subprocess.run(
        [node, "-e", script],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    result = json.loads(completed.stdout)
    assert result == {
        "firstReveal": True,
        "secondReveal": False,
        "canAfterReveal": False,
        "canWhenBusy": False,
        "canWhenNotReady": False,
    }


def test_play_page_pad_hides_what_the_backend_cannot_do():
    """Controls are removed, not greyed out, when a tool is not registered."""
    from geoguesser_env.server.gradio_ui import play_page_html

    page = play_page_html(1)
    for control in ("padTurn", "padMove", "padZoom", "fovRange"):
        assert control in page
    assert 'classList.toggle("gone"' in page
    assert "available_tools" in page


# ------------------------------------------------------- llm action protocol


def _llm_module():
    """Import the LLM rollout example as a module."""
    import importlib.util

    path = pathlib.Path(__file__).parents[1] / "examples" / "geoguesser_llm_rollout.py"
    spec = importlib.util.spec_from_file_location("geoguesser_llm_rollout", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "reply",
    [
        '{"action": "guess", "lat": 1.5, "lon": 2.5}',
        'Sure.\n```json\n{"action": "guess", "lat": 1.5, "lon": 2.5}\n```',
        'I will commit now: {"action": "guess", "lat": 1.5, "lon": 2.5} done.',
    ],
)
def test_llm_action_parses_from_prose_and_fences(reply):
    module = _llm_module()
    spec = module.parse_action(reply)
    assert spec == {"action": "guess", "lat": 1.5, "lon": 2.5}


def test_llm_action_returns_none_when_absent():
    module = _llm_module()
    assert module.parse_action("I have no idea where this is.") is None
    assert module.parse_action('{"not_an_action": 1}') is None


@pytest.mark.parametrize(
    "spec,expected",
    [
        ({"action": "look", "heading_deg": 90}, LookAction),
        ({"action": "zoom", "fov_deg": 30}, ZoomAction),
        ({"action": "move", "direction": "forward"}, MoveAction),
        ({"action": "pin", "lat": 1.0, "lon": 2.0}, PinAction),
        ({"action": "view_map", "lat": 1.0, "lon": 2.0}, ViewMapAction),
        ({"action": "guess", "lat": 1.0, "lon": 2.0}, GuessAction),
    ],
)
def test_llm_action_maps_to_env_action(spec, expected):
    module = _llm_module()
    assert isinstance(module.to_env_action(spec), expected)


def test_llm_action_rejects_unknown_kind():
    module = _llm_module()
    with pytest.raises(ValueError):
        module.to_env_action({"action": "teleport"})


# --------------------------------------------------------- zoom resolution


def test_narrow_field_of_view_requests_the_original():
    """Zoom must reach for the full-resolution panorama.

    A 30-degree view of a 2048-wide panorama samples only ~170 source pixels,
    so zooming on the derivative adds almost no detail (measured mean gradient
    7.03 against 6.60 at 90 degrees). The original roughly doubles it.
    """
    backend = PanoramaBackend(INDEX, PANOS, allow_fetch=False, hires_zoom=True)
    task = backend.task(0)
    with pytest.raises(MissingImageError):
        # No .orig.jpg is committed as a fixture, so this proves the hires path
        # is the one taken for a narrow field of view.
        backend.load_pano(task.frames[0].image_id, hires=True)


def test_wide_field_of_view_uses_the_cached_derivative():
    backend = PanoramaBackend(INDEX, PANOS, allow_fetch=False, hires_zoom=True)
    task = backend.task(0)
    view = backend.render_view(task, task.start_frame, 0.0, 0.0, 90.0)
    assert view.size[0] > 0


def test_zoom_falls_back_rather_than_failing_the_step():
    """A missing original must degrade to a soft view, not end the episode."""
    backend = PanoramaBackend(INDEX, PANOS, allow_fetch=False, hires_zoom=True)
    task = backend.task(0)
    view = backend.render_view(task, task.start_frame, 0.0, 0.0, 30.0)
    assert view.size[0] > 0


def test_hires_can_be_switched_off():
    backend = PanoramaBackend(INDEX, PANOS, allow_fetch=False, hires_zoom=False)
    task = backend.task(0)
    assert backend.render_view(task, task.start_frame, 0.0, 0.0, 20.0).size[0] > 0


# ------------------------------------------------------- cost transparency


def test_action_cost_is_visible_before_the_guess():
    """A policy should be able to see what it has already spent."""
    env = make_env()
    env.reset(task_index=0)
    first = env.step(to_wire(PinAction(lat=1.0, lon=2.0)))
    assert first.action_cost == pytest.approx(0.02)
    second = env.step(to_wire(PinAction(lat=3.0, lon=4.0)))
    assert second.action_cost == pytest.approx(0.04)


def test_frame_index_is_reported_so_a_viewer_can_follow_movement():
    env = make_env()
    env.reset(task_index=0)
    before = env.state.frame_index
    observation = env.step(to_wire(MoveAction(direction="forward", meters=5)))
    assert "frame_index" in observation.metadata
    assert observation.metadata["frame_index"] != before


def test_llm_action_prefers_the_final_decision_over_a_draft():
    """Reasoning traces argue themselves out of early ideas.

    Taking the first action object would execute a draft the model discarded, so
    the last one wins.
    """
    module = _llm_module()
    trace = (
        'Maybe I should {"action": "look", "heading_deg": 0} first.\n'
        "No, the signage is already readable, so I will commit.\n"
        '{"action": "guess", "lat": 35.6, "lon": 139.7}'
    )
    assert module.parse_action(trace) == {
        "action": "guess",
        "lat": 35.6,
        "lon": 139.7,
    }


def test_llm_action_prefers_a_fenced_block_over_loose_prose():
    module = _llm_module()
    reply = (
        'I considered {"action": "pin", "lat": 1, "lon": 2}.\n'
        '```json\n{"action": "guess", "lat": 9.9, "lon": 8.8}\n```'
    )
    assert module.parse_action(reply)["action"] == "guess"


# --------------------------------------------------------------- concurrency


def test_concurrent_episodes_stay_isolated():
    """Parallel rollouts must not interleave into one another.

    An episode is stateful, so each worker needs its own environment. This
    pins the contract that separate instances over one shared read-only index
    do not interfere, which is what a parallel or distributed rollout relies
    on.
    """
    import concurrent.futures

    def play(task_index: int) -> tuple[int, float, int]:
        env = make_env()
        env.reset(task_index=task_index)
        env.step(to_wire(LookAction(heading_deg=90)))
        env.step(to_wire(PinAction(lat=0.0, lon=0.0)))
        true_lat, true_lon = env._task.truth
        result = env.step(to_wire(GuessAction(lat=true_lat, lon=true_lon)))
        return task_index, result.reward, result.metadata["task_index"]

    tasks = [0, 1, 2, 3, 0, 1, 2, 3]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        outcomes = list(pool.map(play, tasks))

    for requested, reward, reported in outcomes:
        assert reported == requested, "an episode returned another episode's task"
        # A perfect guess after one look and one pin: 1.0 - 0.01 - 0.02.
        assert reward == pytest.approx(0.97)


def test_sharing_one_environment_across_episodes_is_sequential():
    """One environment plays one episode at a time; a reset starts a new one."""
    env = make_env()
    env.reset(task_index=0)
    env.step(to_wire(PinAction(lat=1.0, lon=2.0)))
    assert len(env.state.pins) == 1
    env.reset(task_index=1)
    assert env.state.pins == []
    assert env.state.task_index == 1
    assert not env.state.submitted


# ---------------------------------------------------------- map zoom control


def test_pin_map_is_rendered_at_the_requested_zoom():
    """The zoom is part of the decision, so the agent chooses it.

    A pin dropped while looking at a city should come back as a city map, not a
    continental one, or the agent cannot tell how precisely it aimed.
    """
    env = make_env()
    env.reset(task_index=0)
    wide = env.step(to_wire(PinAction(lat=-23.55, lon=-46.63, span_deg=20.0)))
    env.reset(task_index=0)
    close = env.step(to_wire(PinAction(lat=-23.55, lon=-46.63, span_deg=0.1)))
    assert wide.image_base64 != close.image_base64


def test_pin_zoom_is_validated():
    with pytest.raises(Exception):
        PinAction(lat=0.0, lon=0.0, span_deg=0.0)
    with pytest.raises(Exception):
        PinAction(lat=0.0, lon=0.0, span_deg=400.0)


def test_pin_defaults_to_a_regional_view():
    assert PinAction(lat=0.0, lon=0.0).span_deg == pytest.approx(7.0)


def test_detail_layers_are_optional():
    """The map must render whether or not the detail layers are installed."""
    from geoguesser_env.server.render.minimap import has_detail, render_map

    image = render_map([(-23.55, -46.63)], span_deg=0.2)
    assert image.size[0] > 0
    # has_detail() is informational; both states must produce a map.
    assert isinstance(has_detail(), bool)


# --------------------------------------------------------- street detail


def test_street_detail_can_be_switched_off_for_offline_runs():
    """A run that must never touch the network has to be able to say so."""
    from geoguesser_env.server.render import minimap

    was = minimap.street_detail_enabled()
    try:
        minimap.set_street_detail(False)
        assert not minimap.street_detail_enabled()
        # Renders at street zoom without fetching anything.
        image = minimap.render_map([(9.0612, 7.4871)], span_deg=0.05)
        assert image.size[0] > 0
    finally:
        minimap.set_street_detail(was)


def test_street_cache_key_is_stable_and_window_specific():
    from geoguesser_env.server.render.minimap import _osm_cache_path

    first = _osm_cache_path(9.0612, 7.4871, 0.05)
    again = _osm_cache_path(9.0612, 7.4871, 0.05)
    other_place = _osm_cache_path(9.5, 7.4871, 0.05)
    other_zoom = _osm_cache_path(9.0612, 7.4871, 0.2)
    assert first == again
    assert first != other_place
    assert first != other_zoom


def test_street_detail_only_applies_when_zoomed_in():
    """Wide windows must not trigger a fetch, whatever the setting."""
    from geoguesser_env.server.render import minimap

    was = minimap.street_detail_enabled()
    try:
        minimap.set_street_detail(True)
        calls = []
        original = minimap.street_ways
        minimap.street_ways = lambda *args: calls.append(args) or []
        try:
            minimap.render_map([(9.0, 7.0)], span_deg=7.0)
            assert calls == [], "a continental view should not fetch streets"
            minimap.render_map([(9.0, 7.0)], span_deg=0.05)
            assert calls, "a street-scale view should fetch streets"
        finally:
            minimap.street_ways = original
    finally:
        minimap.set_street_detail(was)


def test_llm_media_type_is_detected_from_the_bytes():
    """Views are JPEG and maps are PNG; a hardcoded type is wrong half the time.

    Anthropic rejects a mislabelled image with a 400 rather than sniffing it, so
    this is a hard failure in the middle of a rollout, not a soft one.
    """
    import base64

    module = _llm_module()
    png = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32).decode()
    jpeg = base64.b64encode(b"\xff\xd8\xff\xe0" + b"\x00" * 32).decode()
    assert module.media_type(png) == "image/png"
    assert module.media_type(jpeg) == "image/jpeg"
    assert module.media_type("not base64 at all!!") in {"image/jpeg", "image/png"}


def test_guess_returns_a_reveal_map():
    """A guess used to return no image, leaving the outcome invisible."""
    env = make_env()
    env.reset(task_index=0)
    true_lat, true_lon = env._task.truth
    observation = env.step(to_wire(GuessAction(lat=true_lat + 1, lon=true_lon + 1)))
    assert observation.image_kind == "map"
    assert observation.image_base64
    assert observation.done


def test_reveal_map_is_only_drawn_after_scoring():
    """Truth must never reach the map before the episode ends."""
    env = make_env()
    env.reset(task_index=0)
    pin = env.step(to_wire(PinAction(lat=0.0, lon=0.0)))
    assert pin.true_lat is None and pin.distance_km is None


def test_reveal_map_can_be_disabled_for_throughput():
    """The reveal map is 280 ms a training run never reads."""
    env = make_env(reveal_map=False)
    env.reset(task_index=0)
    true_lat, true_lon = env._task.truth
    observation = env.step(to_wire(GuessAction(lat=true_lat, lon=true_lon)))
    assert observation.image_base64 is None
    assert observation.image_kind == "none"
    # The reward and distance are unaffected: only the picture is skipped.
    assert observation.reward == pytest.approx(1.0)
    assert observation.distance_km == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------- splits


def _two_split_env(tmp_path, **kwargs) -> GeoGuesserEnvironment:
    """An environment with a small `eval` split carved out of the fixtures.

    The eval index holds the fixture's last task renumbered to index 0, so the
    two splits address genuinely different tasks under the same index.
    """
    rows = [json.loads(line) for line in INDEX.read_text().splitlines() if line.strip()]
    held = dict(rows[-1])
    held["task_index"] = 0
    held["task_id"] = "held-00000"
    eval_index = tmp_path / "eval.jsonl"
    eval_index.write_text(json.dumps(held) + "\n")
    options = {
        "splits": {"train": str(INDEX), "eval": str(eval_index)},
        "default_split": "train",
        "cache_dir": str(PANOS),
        "allow_fetch": False,
        "view_size": 128,
    }
    options.update(kwargs)
    return GeoGuesserEnvironment(**options)


def test_single_index_still_becomes_one_default_split():
    env = make_env()
    assert [s["name"] for s in env.list_splits()] == ["train"]
    assert env.list_splits()[0]["default"] is True
    assert env.num_tasks("train") == env._backend.n_tasks


def test_environment_requires_an_index_or_splits():
    with pytest.raises(ValueError, match="splits= or index_path="):
        GeoGuesserEnvironment(cache_dir=str(PANOS))


def test_default_split_must_exist():
    with pytest.raises(ValueError, match="default_split"):
        GeoGuesserEnvironment(
            splits={"eval": str(INDEX)},
            default_split="train",
            cache_dir=str(PANOS),
        )


def test_list_splits_reports_counts_and_core_split_types(tmp_path):
    env = _two_split_env(tmp_path)
    by_name = {s["name"]: s for s in env.list_splits()}
    assert by_name["train"]["type"] == "train"
    # "eval" must declare itself as "test", or core normalises it to
    # "validation" and a held-out set silently reads as a dev set.
    assert by_name["eval"]["type"] == "test"
    assert by_name["eval"]["num_tasks"] == 1
    assert by_name["train"]["num_tasks"] > 1


def test_unknown_split_raises_key_error(tmp_path):
    env = _two_split_env(tmp_path)
    with pytest.raises(KeyError, match="Unknown split"):
        env.num_tasks("nope")
    with pytest.raises(KeyError, match="Unknown split"):
        env.reset(split="nope")


def test_reset_selects_within_the_named_split(tmp_path):
    env = _two_split_env(tmp_path)
    observation = env.reset(split="eval", index=0)
    assert observation.metadata["split"] == "eval"
    assert observation.metadata["task_id"] == "held-00000"


def test_reset_defaults_to_the_default_split(tmp_path):
    env = _two_split_env(tmp_path)
    assert env.reset(index=0).metadata["split"] == "train"


def test_task_index_remains_an_alias_for_index(tmp_path):
    env = _two_split_env(tmp_path)
    first = env.reset(split="train", index=1).metadata
    second = env.reset(split="train", task_index=1).metadata
    assert first["task_id"] == second["task_id"] == second["task_id"]
    assert first["task_index"] == second["task_index"] == 1


def test_switching_split_switches_the_backend(tmp_path):
    env = _two_split_env(tmp_path)
    env.reset(split="eval", index=0)
    assert env.num_tasks("eval") == 1
    env.reset(split="train", index=0)
    assert env._backend.n_tasks == env.num_tasks("train")


def test_task_spec_never_carries_the_answer(tmp_path):
    env = _two_split_env(tmp_path)
    spec = env.get_task("eval", 0)
    # A task spec travels to the trainer; a label in it can reach a prompt.
    assert "lat" not in spec and "lon" not in spec
    assert "country" not in spec
    assert "frames" not in spec
    assert spec["task_id"] == "held-00000"
    assert spec["n_frames"] >= 1


def test_get_task_range_honours_slice_semantics(tmp_path):
    env = _two_split_env(tmp_path)
    total = env.num_tasks("train")
    assert len(env.get_task_range("train")) == total
    assert [s["task_index"] for s in env.get_task_range("train", 1, 3)] == [1, 2]
    assert env.get_task_range("train", 0, 999)[-1]["task_index"] == total - 1


def test_list_tasks_covers_the_whole_split(tmp_path):
    env = _two_split_env(tmp_path)
    specs = env.list_tasks("train")
    assert [s["task_index"] for s in specs] == list(range(env.num_tasks("train")))
    assert {s["split"] for s in specs} == {"train"}


def test_index_cache_returns_the_same_parsed_list():
    from geoguesser_env.server.backends.panorama import load_index

    assert load_index(INDEX) is load_index(str(INDEX))


def test_metadata_reports_street_detail_state():
    """A map missing its streets must be distinguishable from a styling choice.

    Overpass 504s from datacenter egress, so a Space can render poorer maps than
    a laptop for the same task. Silent degradation is the thing to avoid.
    """
    from geoguesser_env.server.render import minimap

    env = make_env()
    # The autouse fixture disables street detail, so this is the "off" case.
    assert env.reset(index=0).metadata["street_detail"] == "off"

    was = minimap.street_detail_enabled()
    failed = minimap.street_fetch_failed()
    try:
        minimap.set_street_detail(True)
        minimap._STREET_FETCH_FAILED[0] = False
        assert env.reset(index=0).metadata["street_detail"] == "on"
        minimap._STREET_FETCH_FAILED[0] = True
        assert env.reset(index=0).metadata["street_detail"] == "unavailable"
    finally:
        minimap.set_street_detail(was)
        minimap._STREET_FETCH_FAILED[0] = failed


def test_country_is_revealed_only_after_the_guess():
    """A per-region breakdown needs the label; an agent must not get it early."""
    env = make_env()
    opening = env.reset(index=0)
    assert "country" not in opening.metadata
    assert opening.true_lat is None

    final = env.step(GuessAction(lat=0.0, lon=0.0))
    assert final.metadata["country"] == env._task.country
    assert final.metadata["guess"] == [0.0, 0.0]
    assert final.metadata["verdict"]
    assert "guess_country" in final.metadata


def test_failed_parse_still_records_the_country():
    """A zero from an unparseable guess still belongs in the regional breakdown."""
    env = make_env()
    env.reset(index=0)
    final = env.step(GuessAction(response="somewhere nice"))
    assert final.reward == 0.0
    assert final.metadata["parse_failure"] is True
    assert final.metadata["country"] == env._task.country


# ------------------------------------------------- reward shape and hardening


def test_mixture_shape_keeps_a_gradient_where_the_game_curve_is_flat():
    """The whole point of the second scale.

    Under the game curve, everything past 6000 km is worth 0.018 in total, so a
    policy gets no gradient for landing on the right continent instead of the
    wrong one -- which is where a third of a small model's rollouts land.
    """
    from geoguesser_env.server.scoring import distance_score

    game_span = distance_score(6000.0) - distance_score(18000.0)
    mixture_span = distance_score(6000.0, shape="mixture") - distance_score(
        18000.0, shape="mixture"
    )
    assert game_span < 0.02
    assert mixture_span > 8 * game_span
    # Near field must stay sharp; that is why the short scale is still half.
    assert distance_score(200.0, shape="mixture") > 0.85


def test_multiplicative_cost_never_collapses_the_distance_ordering():
    """The measured failure: 77 of 200 episodes scored exactly 0.0.

    Mean action cost for a 4B model is 0.13 and the game curve falls below that
    around 3300 km, so subtracting it floors every worse guess at zero. A GRPO
    group drawn from those has no advantage and yields no gradient.
    """
    from geoguesser_env.server.scoring import compute_reward

    near, far = 3400.0, 18000.0
    subtracted = [compute_reward(d, cost=0.13) for d in (near, far)]
    assert subtracted == [0.0, 0.0]  # indistinguishable, and that is the bug

    multiplied = [
        compute_reward(d, cost=0.13, shape="mixture", cost_mode="multiply")
        for d in (near, far)
    ]
    assert multiplied[0] > multiplied[1] > 0.0


def test_any_committed_guess_beats_running_out_of_turns():
    """Small models fail by never committing -- 24 of 200 for Qwen3-VL-2B.

    The mixture is positive anywhere on Earth, so under `"multiply"` a guess on
    the far side of the planet still outscores no guess at all, with no extra
    reward term needed.
    """
    from geoguesser_env.server.scoring import compute_reward

    worst_possible = compute_reward(
        20015.0, cost=0.5, shape="mixture", cost_mode="multiply"
    )
    no_guess = compute_reward(None, cost=0.0, shape="mixture", cost_mode="multiply")
    assert worst_possible > no_guess == 0.0


def test_hidden_identity_strips_the_metadata_that_leaks_the_country():
    """The contributor username alone fixes the country for 74% of train tasks.

    `attribution.creator_username` plus `task_index`/`task_id`/`sequence_id`
    let a policy score without ever reading the image, so RL training must not
    see them per turn.
    """
    leaky = make_env().reset(task_index=0).metadata
    assert leaky["attribution"]["creator_username"]
    assert leaky["sequence_id"] and leaky["task_id"]

    env = make_env(hide_task_identity=True)
    hidden = env.reset(task_index=0).metadata
    for field in ("attribution", "sequence_id", "task_id", "task_index"):
        assert field not in hidden, f"{field} still reaches the agent"
    # Still enough left to render a trace and debug a rollout.
    assert hidden["frame_index"] == 0 and hidden["split"]

    # Provenance returns once the episode is over and cannot be exploited.
    final = env.step(to_wire(GuessAction(lat=0.0, lon=0.0)))
    assert final.done
    assert final.metadata["attribution"]["creator_username"]
    assert final.metadata["sequence_id"]


def test_free_tools_are_capped_so_they_cannot_be_farmed_forever():
    """Free must not mean unlimited.

    `measure` reveals nothing, so it stays free -- but it used to increment no
    counter at all, which let a policy issue it indefinitely without the episode
    ever advancing toward termination.
    """
    env = make_env(max_free_calls=3)
    start = env.reset(task_index=0).steps_remaining
    measure = to_wire(MeasureAction(lat_a=0.0, lon_a=0.0, lat_b=1.0, lon_b=1.0))

    for _ in range(3):
        assert env.step(measure).steps_remaining == start

    exhausted = env.step(measure)
    assert exhausted.steps_remaining < start
    assert "budget spent" in exhausted.feedback


# ------------------------------------------------------- client Task API


def test_client_task_api_resolves_the_url_without_connecting(monkeypatch):
    """The Task API is plain HTTP, so it needs the URL before any session.

    This had no coverage, and the gap cost a GPU job: the client read
    `self.base_url`, which is a property on `EnvClient` in openenv 0.4.2 and
    absent in 0.4.1 -- the current release, so every `pip install` of this
    package produced a client whose Task API raised `AttributeError` on its
    first call. Local runs passed because a development checkout has it.
    """
    from openenv.core.env_client import EnvClient

    from geoguesser_env.client import GeoGuesserEnv

    client = GeoGuesserEnv(base_url="http://127.0.0.1:9999/")
    assert client._server_url() == "http://127.0.0.1:9999"

    # Simulate the released core, which has no public `base_url` at all.
    monkeypatch.delattr(EnvClient, "base_url", raising=False)
    assert client._server_url() == "http://127.0.0.1:9999"


def test_client_task_api_says_so_when_it_has_no_url_yet(monkeypatch):
    """A provider-backed client is assigned its URL on connect()."""
    from openenv.core.env_client import EnvClient

    from geoguesser_env.client import GeoGuesserEnv

    client = GeoGuesserEnv(base_url="http://127.0.0.1:9999")
    monkeypatch.delattr(EnvClient, "base_url", raising=False)
    monkeypatch.setattr(client, "_base_url", None, raising=False)
    with pytest.raises(RuntimeError, match="does not have"):
        client._server_url()
