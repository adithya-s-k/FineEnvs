import random
from dataclasses import replace

import berth_core
import pytest
from berth_core import load_pack, plan_from_list

from port_week_core import marine, world
from port_week_core.marine import CONTAINER

PACK = marine.pack()
EXAMPLE = PACK.get("dock-24B-w07x1-busy-0")
V2_OPTIMUM = plan_from_list(load_pack().get("dock-24B-w07x1-busy-0").reference["optimal_plan"])


def test_tugs_by_length_type_and_wind():
    assert [marine.tugs_needed(CONTAINER, m, False) for m in (119.9, 120, 199.9, 200, 299.9, 300, 400)] == [
        0, 1, 1, 2, 2, 3, 3]
    assert [marine.tugs_needed(CONTAINER, m, True) for m in (100, 150, 250, 366)] == [1, 2, 3, 4]
    assert [marine.tugs_needed(kind, 250, False) for kind in ("Transbordadors", "Passatge", "Iot", "Ro-Ro")] == [
        0, 0, 0, 2]
    assert [marine.tugs_needed(kind, 180, True) for kind in ("Transbordadors", "Ro-Ro", "Iot", "Passatge")] == [
        0, 1, 0, 1]
    assert [marine.tugs_needed(kind, 210, True) for kind in ("Transbordadors", "Ro-Ro", "Iot", "Passatge")] == [
        1, 3, 0, 1]


def test_the_wind_is_the_announced_25_kn_windows():
    assert marine.wind(EXAMPLE) == [(115, 144)]
    assert marine.wind(world.known(EXAMPLE, [])) == []


def unlimited(task):
    pools = task.rules.get("marine", {"hours": 264, "movements": []})
    return replace(task, rules=task.rules | {"marine": pools | {"pilots": 10**6, "tugs": 10**6}})


def test_with_unlimited_pools_the_marine_checker_is_berth_cores_on_every_stored_optimum():
    tasks = load_pack().tasks + PACK.tasks
    assert len(tasks) == 1100 + 18
    for task in map(unlimited, tasks):
        plan = plan_from_list(task.reference["optimal_plan"])
        assert marine.evaluate(task, plan) == berth_core.evaluate(task, plan), task.task_id


def recount(task, plan) -> list[tuple[int, str]]:
    """The pilot and tug lines, counted hour by hour apart from port_week_core.marine."""
    rules = task.rules["marine"]
    gales = [(w["start"], w["end"]) for w in task.rules["no_moves"] if w["min_length"] >= 300]

    def tugs(kind, length_m, hour):
        if kind in ("Transbordadors", "Passatge", "Iot"):
            base = 0
        else:
            base = sum(length_m >= edge for edge in (120, 200, 300))
        windy = any(a <= hour < b for a, b in gales)
        return base + (windy and kind != "Iot" and not (kind in ("Transbordadors", "Ro-Ro") and length_m < 200))

    def room(pool, hour):
        if not 0 <= hour < rules["hours"]:
            return rules[pool]
        cut = sum(e.get(pool, 0) for e in task.disruptions
                  if e["type"] in ("tug_outage", "pilot_shortage") and e["start"] <= hour < e["end"])
        other = [m for m in rules["movements"] if m[0] == hour]
        used = len(other) if pool == "pilots" else sum(tugs(kind, length_m, hour) for _, kind, length_m in other)
        return max(rules[pool] - cut - used, 0)

    at: dict[int, list[int]] = {}
    for sid, (h, _, cranes) in plan.items():
        s = task.ships[sid]
        for hour in (h, task.departure(s, h + s.handling_for(cranes))):
            at.setdefault(hour, []).append(sid)
    lines = []
    for hour, ships in at.items():
        need = {"pilots": len(ships), "tugs": sum(tugs(CONTAINER, task.ships[s].length_m, hour) for s in ships)}
        for pool in ("pilots", "tugs"):
            if need[pool] > room(pool, hour):
                lines += [(s, f"{pool} short at hour {hour}: your ships need {need[pool]}, {room(pool, hour)} free")
                          for s in ships]
    return sorted(lines)


def perturbed(task, view, rng: random.Random) -> dict:
    plan = {sid: e for sid, e in plan_from_list(task.reference["optimal_plan"]).items() if sid < len(view.ships)}
    for sid in rng.sample(sorted(plan), k=rng.randint(1, len(plan))):
        s = view.ships[sid]
        h, sec, cranes = plan[sid]
        plan[sid] = (max(s.arrival, h + rng.choice([-3, -2, -1, 1, 2, 3, 6, 12, 140])), sec,
                     rng.randint(s.min_cranes, s.max_cranes))
    return plan


@pytest.mark.parametrize("known", [False, True], ids=["true-week", "hour-0-view"])
def test_an_independent_hourly_recount_agrees_on_perturbed_plans(known):
    rng = random.Random(7)
    lines_seen = 0
    for task in PACK.tasks:
        view = world.known(task, []) if known else task
        for _ in range(40):
            plan = perturbed(task, view, rng)
            res = marine.evaluate(view, plan)
            marine_lines = sorted((r.ship, p) for r in res.ships for p in r.problems
                                  if world.rule(p) in ("pilots", "tugs"))
            expected = recount(view, plan)
            assert marine_lines == expected
            assert res.feasible == (berth_core.evaluate(view, plan).feasible and not expected)
            lines_seen += len(expected)
    assert lines_seen > 300


def test_a_ship_gets_one_line_per_pool_and_hour_in_hour_then_pilots_then_tugs_order():
    view = replace(EXAMPLE, disruptions=EXAMPLE.disruptions + [
        {"type": "pilot_shortage", "start": 54, "end": 55, "pilots": 7}])
    res = marine.evaluate(view, V2_OPTIMUM)
    assert not res.feasible and res.cost is None
    assert res.ships[4].problems == [
        "pilots short at hour 54: your ships need 2, 0 free", "tugs short at hour 54: your ships need 3, 2 free"]
    assert res.ships[5].problems == res.ships[4].problems


def test_free_takes_the_announced_cuts_the_other_traffic_and_the_announced_wind():
    rules = EXAMPLE.rules["marine"]
    room = marine.free(EXAMPLE)
    assert {pool: len(hours) for pool, hours in room.items()} == {"pilots": 264, "tugs": 264}
    assert [m for m in rules["movements"] if m[0] == 54] == [
        [54, "Portacontenidors", 139.0], [54, "Portacontenidors", 148.0], [54, "Ro-Ro", 238.0]]
    assert (room["pilots"][54], room["tugs"][54]) == (7 - 2 - 3, 8 - 2 - 4)
    at_0 = marine.free(world.known(EXAMPLE, []))
    assert (at_0["pilots"][54], at_0["tugs"][54]) == (4, 4)
    windy = [h for h in range(264) if room["tugs"][h] != marine.free(replace(
        EXAMPLE, rules=EXAMPLE.rules | {"no_moves": []}))["tugs"][h]]
    assert windy and all(115 <= h < 144 for h in windy)


def test_own_moves_are_each_berthing_and_departure_with_the_winds_extra_tug():
    res = berth_core.evaluate(EXAMPLE, V2_OPTIMUM)
    moves = marine.own_moves(EXAMPLE, res)
    assert len(moves) == 2 * len(EXAMPLE.ships)
    assert [m for m in moves if m[0] in (4, 5, 7)] == [(4, 47, 1), (4, 54, 1), (5, 54, 2), (5, 64, 2), (7, 93, 1),
                                                      (7, 115, 2)]
    assert all(t == marine.tugs_needed(CONTAINER, EXAMPLE.ships[s].length_m, 115 <= h < 144) for s, h, t in moves)


def test_the_section_at_hour_0_of_the_example_week():
    view = world.known(EXAMPLE, [])
    lines = marine.section(view).split("\n")
    assert lines[:3] == [
        "## Pilots and tugs",
        "- Each berthing and each departure takes a pilot and tugs in its hour, from the port's 7 pilots and 8 tugs on "
        "duty, shared with the port's other traffic. In any hour your ships may need no more pilots or tugs than are "
        "free for them; each ship moving in an hour short of either breaks this rule.",
        "- Tugs per movement: 0 under 120 m, 1 under 200 m, 2 under 300 m, 3 from 300 m."]
    assert lines[3] == ("- Your ships take 3 tugs: ships 3, 8; 2 tugs: ships 0, 2, 5, 9, 10, 15; 1 tug: ships 1, 4, 6, "
                        "7, 11, 12, 13, 14.")
    assert lines[4:6] == [
        "- Free for your ships after the other traffic, the cuts and the wind, pilots | tugs, each hour from 00:00 in "
        "blocks of 6 hours:",
        "- Mon: 745565 266666 764766 354136 | 867675 286888 867888 854765"]
    assert [line.split(":")[0] for line in lines[5:16]] == [
        "- Mon", "- Tue", "- Wed", "- Thu", "- Fri", "- Sat", "- Sun", "- +7d", "- +8d", "- +9d", "- +10d"]
    assert lines[16:] == ["- From hour 264: 7 pilots and 8 tugs free."]
    assert len(marine.section(view).encode()) == 1393


def test_the_section_once_all_the_news_is_in():
    text = marine.section(EXAMPLE)
    assert "- From hour 115 to hour 144 (wind above 25 kn), each movement takes one more tug." in text
    assert "2 tugs: ships 0, 2, 5, 9, 10, 15, 16;" in text
    assert "- Tug company: 2 of the 8 tugs out from hour 54 to hour 78." in text
    assert "- Pilot station: 2 of the 7 pilots out from hour 54 to hour 66." in text
    assert len(text.encode()) == 1603


def test_step_marine_is_the_free_series_our_use_by_hour_and_the_announced_cuts():
    step = marine.step_marine(EXAMPLE, V2_OPTIMUM)
    room = marine.free(EXAMPLE)
    assert step["hours"] == 264
    assert step["pilots"] == {"pool": 7, "free": room["pilots"]} and step["tugs"] == {"pool": 8, "free": room["tugs"]}
    assert [54, 2, 3] in step["ours"] and step["ours"] == sorted(step["ours"])
    assert sum(p for _, p, _ in step["ours"]) == 2 * len(V2_OPTIMUM)
    assert step["cuts"] == [{"from": "Tug company", "pool": "tugs", "count": 2, "start": 54, "end": 78},
                            {"from": "Pilot station", "pool": "pilots", "count": 2, "start": 54, "end": 66}]
    assert marine.step_marine(world.known(EXAMPLE, []), {})["ours"] == []
