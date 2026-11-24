"""What the AgentEnv envs and their tasks agree on: the env names, the extensions, the agent's tools and the gateway
trigger per watch that delivers the watch's notices through the hidden port_notice tool."""

from port_week_core.schedule import Watch

LIVE_ENV = "portsim-live"
MARINE_ENV = "portsim-marine"
WIND_ENV = "portsim-wind"
LINES_ENV = "portsim-lines"
LIVE_LOAD_URI = "urn:portsim:live-load/v1"
END_WEEK_URI = "urn:portsim:end-week/v1"
CLOCK_URI = "urn:agentenv:clock/v1"
TOOLS = ("get_situation", "check_plan", "confirm_berths", "advance")
NOTICE_TOOL = "port_notice"
LINE_TOOLS = ("ship_status", "agree_arrival")
LINE_AGENT = "portsim-line"
LINE_ROLE = "line"
BARRIER_SECONDS = 60


def triggers(watches: list[Watch]) -> list[dict]:
    return [{"id": f"watch-{w.index}",
             "when": {"type": "action", "tool": "advance", "where": {"result.watch": {"equals": w.index}}},
             "barrier": {"at": "provoking_call", "timeout_seconds": BARRIER_SECONDS},
             "actions": [{"type": "tool", "tool": NOTICE_TOOL,
                          "args": {"event_id": n.event_id, "name": n.name, "text": n.text}} for n in w.notices]}
            for w in watches[1:]]
