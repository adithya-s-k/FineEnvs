"""Each env's card, byte for byte as the agentenv-portsim plugin's v0.4.5 server serves it (fixtures/cards). The wind
card holds nothing from the wind pack, so the wind env serves it on the synthetic wind weeks."""

from pathlib import Path

import httpx
import pytest
from serving import served
from wind_fixture import tasks_dir

CARDS = Path(__file__).resolve().parent / "fixtures" / "cards"


@pytest.mark.parametrize("environment_name", ["portsim-live", "portsim-marine", "portsim-wind"])
def test_the_card_is_the_plugins(environment_name, tmp_path):
    with served(environment_name, tasks_dir(tmp_path) if environment_name == "portsim-wind" else None) as base:
        card = httpx.get(f"{base}/.well-known/agent-env.json").content
    assert card == (CARDS / f"{environment_name}.json").read_bytes()
