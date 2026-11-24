"""The watch timetable of a live day, the news each watch delivers, and the gateway triggers that deliver it.

Watches are core.known.watch_times; a news item reaches the planner at the first watch at or after its ``at`` (those
before the first watch come with the opening briefing, those after the last never arrive)."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from .core.known import watch_times
from .core.model import Task

ENV = "airops-day"
AGENT = "airops-llm"
VERIFIER = "airops-verifier"
LIVE_LOAD_URI = "urn:airops:live-load/v1"
END_DAY_URI = "urn:airops:end-day/v1"
CLOCK_URI = "urn:agentenv:clock/v1"
TOOLS = ("get_situation", "check_plan", "confirm_changes", "advance")
NOTICE_TOOL = "airops_notice"
PLANNING_CALLS = 3
TURNS_PER_WATCH = 5
EXTRA_TURNS = 4
BARRIER_SECONDS = 60
TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


@dataclass(frozen=True)
class Watch:
    index: int
    t: int
    news: tuple[str, ...]


def timetable(task: Task) -> list[Watch]:
    times = watch_times(task)
    due: dict[int, list[str]] = {k: [] for k in range(len(times))}
    for n in sorted(task.news, key=lambda n: (n.at, n.id)):
        k = next((k for k, t in enumerate(times) if t >= n.at), None)
        if k is not None:
            due[k].append(n.id)
    return [Watch(k, t, tuple(due[k])) for k, t in enumerate(times)]


def triggers(watches: list[Watch]) -> list[dict]:
    return [{"id": f"watch-{w.index}",
             "when": {"type": "action", "tool": "advance", "where": {"result.watch": {"equals": w.index}}},
             "barrier": {"at": "provoking_call", "timeout_seconds": BARRIER_SECONDS},
             "actions": [{"type": "tool", "tool": NOTICE_TOOL, "args": {"news_id": n}} for n in w.news]}
            for w in watches[1:] if w.news]


def max_turns(watches: list[Watch]) -> int:
    return TURNS_PER_WATCH * len(watches) + EXTRA_TURNS


def virtual_time(task: Task, minute: int) -> str:
    """The minute of the task's day, hub-local, as the UTC instant the gateway clock reads."""
    midnight = datetime.fromisoformat(task.date).replace(tzinfo=ZoneInfo(task.tz))
    return (midnight + timedelta(minutes=minute)).astimezone(UTC).strftime(TIME_FORMAT)
