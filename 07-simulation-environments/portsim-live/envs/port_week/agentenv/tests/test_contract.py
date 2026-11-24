"""The gateway triggers a live week registers: one per watch after the first, delivering its notices in order."""

from port_week_core.schedule import schedule

from port_week_agentenv.contract import triggers


def test_each_watch_after_the_first_is_one_trigger_on_advance(example):
    out = triggers(schedule(example))
    assert [t["id"] for t in out] == [f"watch-{k}" for k in range(1, 7)]
    assert out[3] == {
        "id": "watch-4",
        "when": {"type": "action", "tool": "advance", "where": {"result.watch": {"equals": 4}}},
        "barrier": {"at": "provoking_call", "timeout_seconds": 60},
        "actions": [
            {"type": "tool", "tool": "port_notice", "args": {
                "event_id": "late-4", "name": "MAERSK NARMADA",
                "text": "Ship 10 MAERSK NARMADA was delayed leaving Valencia: it now arrives at hour 129 (planned slot "
                        "hour 112). Its planned departure stays hour 128."}},
            {"type": "tool", "tool": "port_notice", "args": {
                "event_id": "crane_outage-6", "name": "Terminal ops",
                "text": "2 of the terminal's 9 quay cranes are out of service from hour 108 to hour 155 (gantry "
                        "repair): 7 cranes in that window."}}]}
