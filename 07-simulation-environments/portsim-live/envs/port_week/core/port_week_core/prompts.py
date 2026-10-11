"""What the planner is told about a live week: the opening and the rules (live and marine; wind adds the forecast)."""

from berth_core import Task
from berth_core import rules as berth_rules

from .schedule import PLANNING_CALLS

LIVE_OPENING = ("{situation}\n\nIt is watch 0, Monday 00:00. Confirm berth windows with confirm_berths, then call "
                "advance.")
LIVE_RULES = """How the week runs:
- The week is played in watches, starting at watch 0 on Monday 00:00 (hour 0). The table shows the week as the port \
knows it now. News reaches you as messages in tool results: ships report delays and unscheduled calls, the harbour \
master issues gale warnings and emergencies, terminal ops announce closures and crane outages, and the line desk flags \
priority cargo.
- Time only passes when you call advance: the port moves to the next bulletin, ships berth and sail on their confirmed \
windows, and the bulletin's messages arrive with your next tool result. Call advance and get_situation together in one \
turn.
- From watch 1 on, a window that starts before the freeze line (6 hours after the current hour) is frozen: it can't be \
changed, and new or changed windows must start at or after the freeze line. At watch 0 every window is open.
- You have 3 planning calls (check_plan or confirm_berths) per watch.

Tools:
- get_situation(): the hour, the week as known now, your confirmed windows (departed, berthed, frozen or open), the \
ships still without a window, and new messages.
- check_plan(plan): your entries over your confirmed windows, checked on the week as known now: the ships with a \
problem or a cost, the plan's cost, and the entries confirm_berths would refuse. Problems and cost that news brought \
to a frozen window are listed as excused and not counted. It confirms nothing.
- confirm_berths(plan): confirms windows with the ships; ships you leave out keep their windows.
- advance(): ends the watch. After the last watch it returns "done": the rest of the week runs on your confirmed \
windows.

The week is graded once, on the windows you confirmed and the week as it really happened: an infeasible week scores \
low, a feasible one higher the closer its cost is to the best possible in hindsight. Cost or rule breaks that news \
brings to a window that was already frozen are not charged to you. Every ship needs a confirmed window before the week \
ends; if you stop before the last watch, the rest of the week runs on the windows you confirmed."""
WIND_RULES = LIVE_RULES.replace(
    "the harbour master issues gale warnings and emergencies, terminal ops announce closures and crane outages, and "
    "the line desk flags priority cargo.",
    "the harbour master issues emergencies, terminal ops announce closures and crane outages, the line desk flags "
    "priority cargo, and Barcelona Port Control sends the wind forecast at every watch.").replace(
    "the week as it really happened: an infeasible week scores low, a feasible one higher the closer its cost is to "
    "the best possible in hindsight. Cost or rule breaks that news brings to a window that was already frozen are not "
    "charged to you.",
    "the week as it really happened, in the wind that really blew: an infeasible week scores low, a feasible one "
    "higher the closer its cost is to the best possible. Cost or rule breaks that news, or wind the forecast didn't "
    "show, brings to a window after your last chance to change it are not charged to you; wind the forecast showed "
    "then is.")


def live_rules(task: Task, rules: str = LIVE_RULES) -> str:
    return berth_rules(task, PLANNING_CALLS).split("\n\nTools:\n")[0] + "\n\n" + rules
