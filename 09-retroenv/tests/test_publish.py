from __future__ import annotations

from dataset.publish_release import MANIFEST_ROW, RUNS_ROW, release_card

CARD = f"| Path | Contents |\n|---|---|\n{MANIFEST_ROW}\n"


def test_the_card_lists_runs_only_when_they_are_published():
    assert release_card(CARD, runs=False) == CARD
    with_runs = release_card(CARD, runs=True)
    assert with_runs.index(RUNS_ROW) > with_runs.index(MANIFEST_ROW)
    assert release_card(with_runs, runs=True) == with_runs
