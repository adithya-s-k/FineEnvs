"""What the planner is told about a live week: the opening and the rules (live and marine; wind adds the forecast,
lines the ships' agents)."""

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


LINES_RULES = WIND_RULES.replace(
    "ships report delays and unscheduled calls, the harbour master",
    "each ship's agent reports its ETA and calls in before it arrives, the line desk reports unscheduled calls, the "
    "harbour master").replace(
    "Cost or rule breaks that news, or wind the forecast didn't show,",
    "An ETA that proves wrong is not news: a ship that arrives later than its agent said is berthed by the port at "
    "its next free slot, one that arrives earlier waits at anchor for its window, and the lateness and hours at anchor "
    "that follow are charged; a ship whose arrival you agreed arrives at that hour. Cost or "
    "rule breaks that news, or wind the forecast didn't show,").replace(
    "your confirmed windows (departed, berthed, frozen or open),",
    "your confirmed windows (departed, berthed, frozen or open, and where the port berths a ship that gets in after "
    "its window, on its latest ETA until it is in),").replace(
    "- confirm_berths(plan): confirms windows with the ships; ships you leave out keep their windows.",
    "- confirm_berths(plan): confirms windows with the ships; ships you leave out keep their windows. It returns the "
    "confirmed plan's problems and cost, as check_plan does."
) + """

The ships' agents:
- Each ship's agent, the firm that deals with the port for its line, reports the ship's ETA 48 and 24 hours out and \
calls in an hour before the ship arrives. The published schedule is the lines' estimate from about three days out. \
Both are optimistic: most ships arrive later than they say, by less as they near.
- message_agent(agent, requests) sends one agent requests for its ships: {"ship": 7, "ask": "eta", \
"request_arrival": null} for the latest ETA, {"ship": 9, "ask": null, "request_arrival": 130} to ask the ship to \
arrive at that hour. Once the ship has sailed from its previous port or its agent's estimate is within 48 hours, the \
agent agrees \
when it can make the hour at cruising speed or by slowing once it has sailed, and the ship then arrives at that hour; \
otherwise it answers with the earliest or the latest hour the ship can make. Before that, the agent can't commit to \
an hour and gives its ETA.
- Until the port's news about a delayed ship, its agent works from the plan as it stood. The news keeps an earlier \
agreement only if the ship can still make that hour; otherwise the agreement lapses.
- A ship can't berth before it arrives. If it gets in after its window starts, the port berths it at the first hour \
a free stretch of quay fits it, with the same cranes, around every other ship's window and the rules, never ahead of \
an earlier ship whose arrival you agreed still waiting; nothing else moves. A ship that gets in before its window \
waits at anchor for it.
- Ships berth first come, first served: no ship may start berthing while a ship earlier in the queue whose arrival \
you agreed waits at anchor, unless the wind holds that ship or the closures leave no stretch of quay long enough for \
it. A ship's turn is when it would arrive at cruising speed, \
so a ship slowed at your request keeps its turn.
- Each hour a ship waits at anchor between its arrival and its berthing costs its sections, as an hour late does."""


def live_rules(task: Task, rules: str = LIVE_RULES) -> str:
    return berth_rules(task, PLANNING_CALLS).split("\n\nTools:\n")[0] + "\n\n" + rules
