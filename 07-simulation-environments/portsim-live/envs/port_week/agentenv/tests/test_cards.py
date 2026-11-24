"""Each env's card, byte for byte as the agentenv-portsim plugin's server serves it (fixtures/cards; live, marine and
wind at v0.4.5, which its server at 83eca5c still serves, lines at 83eca5c). The wind and lines cards hold nothing from
their packs, so those envs serve them on the synthetic weeks."""

from pathlib import Path

import httpx
import pytest
from serving import served
from wind_fixture import tasks_dir

CARDS = Path(__file__).resolve().parent / "fixtures" / "cards"


@pytest.mark.parametrize("environment_name", ["portsim-live", "portsim-marine", "portsim-wind", "portsim-lines"])
def test_the_card_is_the_plugins(environment_name, tmp_path):
    synthetic = environment_name in ("portsim-wind", "portsim-lines")
    with served(environment_name, tasks_dir(tmp_path) if synthetic else None) as base:
        card = httpx.get(f"{base}/.well-known/agent-env.json").content
    assert card == (CARDS / f"{environment_name}.json").read_bytes()
