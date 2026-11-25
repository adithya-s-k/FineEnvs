"""airops-occ3 as its image runs it, on a fixture date: the card, a day loaded with no desk, the rest of the day run by
the crew and passenger systems alone at end-day, and data/get holding the day graded under P as the same day played in
process grades it, with reward_L against the pack's bound. In process, or in a container of the image with
AIROPS_OCC3_IMAGE set."""

import httpx
from agentenv_protocol import client
from airops_core.occ.day3 import OccDay3
from airops_core.occ.pack import Pack3
from paths import OCC3_PACK
from serving import served

from airops_agentenv.occ.live import END_DAY_URI, LOAD_URI

CONTROL = "b6-20250115-0400"


def test_a_day_with_no_desk_grades_under_p_as_in_process():
    with served("airops_agentenv.occ3_server", {"AIROPS_OCC3_PACK": str(OCC3_PACK)}) as base:
        card = httpx.get(f"{base}/.well-known/agent-env.json").raise_for_status().json()
        assert card["name"] == "airops-occ3"
        loaded = httpx.post(f"{base}{client.extension_params(card, LOAD_URI)['endpoint']}",
                            json={"task_id": CONTROL, "mode": "idle"}, timeout=120).json()
        assert loaded["task_id"] == CONTROL and loaded["mode"] == "idle"
        httpx.post(f"{base}{client.extension_params(card, END_DAY_URI)['endpoint']}", json={},
                   timeout=600).raise_for_status()
        day = httpx.post(f"{base}/agentenv", json={"jsonrpc": "2.0", "id": 1, "method": "data/get"},
                         timeout=60).json()["result"]["parts"][0]["data"]
    world = Pack3(OCC3_PACK).world(CONTROL)
    again = OccDay3(world, "idle")
    again.end()
    assert day["done"] and day["audit"]["ok"] and day["digest"] == again.digest()
    assert day["grade"]["cost"] == again.grade["cost"] and day["grade"]["passengers"] == again.grade["passengers"]
    assert day["grade"]["bound"] == world.refs["bound"] and day["grade"]["reward_L"] is not None
