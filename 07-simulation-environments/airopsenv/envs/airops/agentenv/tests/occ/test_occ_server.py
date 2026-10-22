"""airops-occ as its image runs it, on a fixture date: the card, a day loaded with no desk, the rest of the day run by
the crew system alone at end-day, and data/get holding the day graded at the pack's idle reference. In process, or in a
container of the image with AIROPS_OCC_IMAGE set."""

import httpx
from agentenv_protocol import client
from airops_core.occ.pack import Pack
from paths import OCC_PACK
from serving import served

from airops_agentenv.occ.live import END_DAY_URI, LOAD_URI

CONTROL = "b6-20250115-0400"


def test_a_day_with_no_desk_grades_at_the_idle_reference():
    with served("airops_agentenv.occ_server", {"AIROPS_OCC_PACK": str(OCC_PACK)}) as base:
        card = httpx.get(f"{base}/.well-known/agent-env.json").raise_for_status().json()
        assert card["name"] == "airops-occ"
        loaded = httpx.post(f"{base}{client.extension_params(card, LOAD_URI)['endpoint']}",
                            json={"task_id": CONTROL, "mode": "idle"}, timeout=120).json()
        assert loaded["task_id"] == CONTROL and loaded["mode"] == "idle"
        httpx.post(f"{base}{client.extension_params(card, END_DAY_URI)['endpoint']}", json={},
                   timeout=600).raise_for_status()
        day = httpx.post(f"{base}/agentenv", json={"jsonrpc": "2.0", "id": 1, "method": "data/get"},
                         timeout=60).json()["result"]["parts"][0]["data"]
    assert day["done"] and day["audit"]["ok"]
    assert day["grade"]["cost"] == Pack(OCC_PACK).world(CONTROL).refs["idle"]
