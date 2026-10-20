"""The pack's winter control (its fixture is the published date) played in process: both desks idle cost the pack's
idle reference, and the scripted desks (crew forwarding its system's words, ops accepting its cancellations) play
programs/occ.py's talk-idle day."""

from pathlib import Path

from occ_helpers import case, play, scripted

from airops_core.occ.pack import Pack
from airops_core.programs import crewplan, occ

PACK = Path(__file__).resolve().parents[1] / "fixtures" / "b6-v2"
CONTROL = "b6-20250115-0400"


def test_the_winter_control_idle_and_scripted_match_the_pack_and_the_talk_idle_day():
    w = Pack(PACK).world(CONTROL)
    idle = play(w)
    assert idle.grade["cost"] == w.refs["idle"] and idle.audit()["ok"]
    talk = play(w, policies=scripted("crew", "ops"))
    assert talk.grade["cost"] == round(crewplan.worth(occ.run(case(w), "talk-idle").day, False), 1)
    assert talk.audit()["ok"] and talk.acausal == []
