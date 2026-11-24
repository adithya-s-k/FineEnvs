"""The env's card, byte for byte as the agentenv-portsim plugin's v0.4.5 server serves it (card.json)."""

from pathlib import Path

import httpx


def test_the_card_is_the_plugins(server):
    golden = Path(__file__).with_name("card.json").read_bytes()
    assert httpx.get(f"{server}/.well-known/agent-env.json").content == golden
