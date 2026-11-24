"""The lines tasks: a lines week's steps against its wind week's (portsim-lines, the line role that sees only the
agents' two tools, one portsim-line agent per firm peered with the planner, a trigger per watch carrying the agents'
reports, the lines rules and turns)."""

import pytest
from agent_env.task_step.registry import get_task_step_registry
from agent_env.task_step.task_step import TaskStep
from berth_core import rules
from lines_fixture import require_lines
from port_week_core import lines, prompts
from port_week_core.lines import LinesWeek
from port_week_core.schedule import LINES_MAX_TURNS, schedule
from port_week_core.wind import WindWeek

from port_week_agentenv import tasks
from port_week_agentenv.contract import triggers

require_lines()

PACK = lines.pack()
TASK = PACK.get("dock-24B-w06x1-busy-0-e15")
STEPS = tasks.lines_steps(TASK, 5.0)
BY_ID = {s["id"]: s for s in STEPS}
FIRMS = sorted(TASK.rules["lines"]["firms"])
OPENING = "\n\nIt is watch 0, Monday 00:00. Confirm berth windows with confirm_berths, then call advance."


def wind_steps(task, episode_cap_usd: float = 5.0) -> list[dict]:
    return tasks.live_steps(task, episode_cap_usd, env="portsim-lines", week=LinesWeek, rules=prompts.LINES_RULES,
                            max_turns=LINES_MAX_TURNS)


def test_a_lines_week_is_its_wind_week_on_portsim_lines_with_the_agents_deployed_and_peered():
    assert [s["id"] for s in STEPS] == [
        "deploy", "load-task", "hide-port-notice", "agents-only", "agents-tools", "watches", "deploy-agent",
        *(f"deploy-{f}" for f in FIRMS), "peers", "clock", "play", "end-week", "grade"]
    assert FIRMS == ["aperez", "boluda", "cmacgm", "cosco", "globalliner", "hapag", "noatum", "one"]
    plain = {s["id"]: s for s in wind_steps(TASK)}
    for step in STEPS:
        if step["id"] in plain and step["id"] not in ("hide-port-notice", "watches"):
            assert step == plain[step["id"]], step["id"]
    assert {s.get("env_id") for s in STEPS} == {"portsim-lines", None}
    assert BY_ID["load-task"]["directives"] == [{"service": "portsim-lines", "uri": "urn:portsim:live-load/v1",
                                                 "args": {"task_id": TASK.task_id}}]
    assert BY_ID["deploy-agent"] == {"id": "deploy-agent", "type": "deploy_agent", "a2a_agent_id": "portsim-llm",
                                     "env_ids": ["portsim-lines"], "env_vars": {"PORTSIM_MAX_COST_USD": "5"}}


def test_the_line_role_sees_only_the_agents_two_tools_and_the_planner_neither_nor_port_notice():
    assert BY_ID["hide-port-notice"] == {"id": "hide-port-notice", "type": "modify_env_tool_access",
                                         "env_id": "portsim-lines", "action": "disable", "role": "default",
                                         "tools": ["port_notice", "ship_status", "agree_arrival"]}
    assert [BY_ID["agents-only"], BY_ID["agents-tools"]] == [
        {"id": "agents-only", "type": "modify_env_tool_access", "env_id": "portsim-lines", "action": "disable",
         "role": "line", "tools": ["*"]},
        {"id": "agents-tools", "type": "modify_env_tool_access", "env_id": "portsim-lines", "action": "enable",
         "role": "line", "tools": ["ship_status", "agree_arrival"]}]
    ids = [s["id"] for s in STEPS]
    assert ids.index("hide-port-notice") < ids.index("agents-only") < ids.index("agents-tools") < ids.index(
        "deploy-agent")


def test_one_portsim_line_agent_per_firm_named_by_its_id_and_peered_with_the_planner_after_the_deploys():
    deploys = [s for s in STEPS if s["id"].startswith("deploy-") and s["id"] != "deploy-agent"]
    assert deploys == [{"id": f"deploy-{f}", "type": "deploy_agent", "a2a_agent_id": "portsim-line", "agent_name": f,
                        "role": "line", "env_ids": ["portsim-lines"]} for f in FIRMS]
    assert BY_ID["peers"] == {"id": "peers", "type": "peer_agents", "peerings": [
        {"source_agent_name": "default-agent", "peer_agent_names": FIRMS}]}
    assert TaskStep.DEFAULT_AGENT_NAME == "default-agent" and "agent_name" not in BY_ID["deploy-agent"]
    ids = [s["id"] for s in STEPS]
    assert max(ids.index(s["id"]) for s in deploys) < ids.index("peers") < ids.index("clock") < ids.index("play")


@pytest.mark.parametrize("task", [pytest.param(t, id=t.task_id) for t in PACK.tasks])
def test_every_week_deploys_the_firms_of_its_ships(task):
    steps = tasks.lines_steps(task, 5.0)
    firms = sorted({lines.firm(task, s.id) for s in task.ships})
    assert [s["agent_name"] for s in steps if s.get("a2a_agent_id") == "portsim-line"] == firms
    assert next(s for s in steps if s["id"] == "peers")["peerings"][0]["peer_agent_names"] == firms


def test_every_step_parses_with_agent_envs_step_types_keeping_what_it_sets():
    registry = get_task_step_registry()
    for step in STEPS:
        parsed = registry[step["type"]].from_dict(step).to_dict()
        assert {k: parsed[k] for k in step if k != "directives"} == {k: v for k, v in step.items() if k != "directives"}


def test_the_watches_carry_the_agents_reports_after_each_bulletins_news():
    watches = BY_ID["watches"]
    assert (watches["watch_roles"], watches["triggers"]) == (["default"], triggers(lines.timetable(TASK)))
    plain = triggers(schedule(lines.as_v4(TASK)))
    assert [t["id"] for t in watches["triggers"]] == [t["id"] for t in plain] == [f"watch-{k}" for k in range(1, 6)]
    for ours, theirs in zip(watches["triggers"], plain, strict=True):
        assert ours["actions"][:len(theirs["actions"])] == theirs["actions"]
        assert ours["when"] == theirs["when"] and ours["barrier"] == theirs["barrier"]
    assert [a["args"] for a in watches["triggers"][0]["actions"][len(plain[0]["actions"]):]] == [
        {"event_id": "eta-2-24", "name": "CMA CGM Ibérica",
         "text": "CMA CGM DALILA (ship 2): ETA Tue 01:00 (hour 25), as of Mon 01:00."},
        {"event_id": "eta-4-48", "name": "COSCO Shipping Lines Spain",
         "text": "SEASPAN OCEANIA (ship 4): ETA Wed 01:00 (hour 49), as of Mon 01:00."},
        {"event_id": "call-1-1", "name": "A. Pérez y Cía",
         "text": "NAVIOS INDIGO (ship 1) calls in: at buoy S, arriving hour 5 (Mon 05:00)."},
        {"event_id": "eta-5-48-plan", "name": "Noatum Maritime",
         "text": "NANTO (ship 5): ETA Wed 09:00 (hour 57), as of Mon 15:00."},
        {"event_id": "eta-3-24", "name": "Global Liner Agencias",
         "text": "STEEN (ship 3): ETA Tue 19:00 (hour 43), as of Mon 23:00."},
        {"event_id": "call-2-1", "name": "CMA CGM Ibérica",
         "text": "CMA CGM DALILA (ship 2) calls in: at buoy S, arriving hour 25 (Tue 01:00)."},
        {"event_id": "eta-6-24-plan", "name": "Ocean Network Express Spain",
         "text": "ONE FREEDOM (ship 6): ETA Wed 06:00 (hour 54), as of Tue 00:00."}]
    assert {a["tool"] for t in watches["triggers"] for a in t["actions"]} == {"port_notice"}


def test_the_opening_ends_with_the_agents_and_the_rules_are_the_lines_rules():
    play = BY_ID["play"]
    week = LinesWeek(TASK)
    assert play["prompt"] == week.situation()["situation"] + OPENING
    assert play["prompt"].endswith(lines.section(week) + OPENING)
    assert "## Arrivals and the ships' agents" in play["prompt"] and "FOS EXPRESS" not in play["prompt"]
    assert play["system_prompt"] == rules(TASK, 3).split("\n\nTools:\n")[0] + "\n\n" + prompts.LINES_RULES
    assert play["max_turns"] == LINES_MAX_TURNS == 7 * 10 + 2
    assert play["prompt"] != tasks.live_steps(TASK, 5.0, week=WindWeek)[6]["prompt"]


def test_the_lines_rules_are_the_wind_rules_with_four_sentences_changed_and_the_agents_added():
    wind_rules, ours = prompts.WIND_RULES.split("\n"), prompts.LINES_RULES.split("\n")
    assert ours[len(wind_rules):] == [
        "",
        "The ships' agents:",
        "- Each ship's agent, the firm that deals with the port for its line, reports the ship's ETA 48 and 24 hours "
        "out and calls in an hour before the ship arrives. The published schedule is the lines' estimate from about "
        "three days out. Both are optimistic: most ships arrive later than they say, by less as they near.",
        "- message_agent(agent, requests) sends one agent requests for its ships: {\"ship\": 7, \"ask\": \"eta\", "
        "\"request_arrival\": null} for the latest ETA, {\"ship\": 9, \"ask\": null, \"request_arrival\": 130} to ask "
        "the ship to arrive at that hour. Once the ship has sailed from its previous port or its agent's estimate is "
        "within 48 hours, the agent agrees when it can make the hour at cruising speed or by slowing once it has "
        "sailed, and the ship then arrives at that hour; otherwise it answers with the earliest or the latest hour the "
        "ship can make. Before that, the agent can't commit to an hour and gives its ETA.",
        "- Until the port's news about a delayed ship, its agent works from the plan as it stood. The news keeps an "
        "earlier agreement only if the ship can still make that hour; otherwise the agreement lapses.",
        "- A ship can't berth before it arrives. If it gets in after its window starts, the port berths it at the "
        "first hour a free stretch of quay fits it, with the same cranes, around every other ship's window and the "
        "rules, never ahead of an earlier ship whose arrival you agreed still waiting; nothing else moves. A ship that "
        "gets in before its window waits at anchor for it.",
        "- Ships berth first come, first served: no ship may start berthing while a ship earlier in the queue whose "
        "arrival you agreed waits at anchor, unless the wind holds that ship or the closures leave no stretch of quay "
        "long enough for it. A ship's turn is when it would arrive at cruising speed, so a ship slowed at your request "
        "keeps its turn.",
        "- Each hour a ship waits at anchor between its arrival and its berthing costs its sections, as an hour late "
        "does."]
    changed = [i for i, (a, b) in enumerate(zip(wind_rules, ours, strict=False)) if a != b]
    situation = next(i for i, line in enumerate(wind_rules) if line.startswith("- get_situation()"))
    confirm = next(i for i, line in enumerate(wind_rules) if line.startswith("- confirm_berths("))
    assert changed == [1, situation, confirm, len(wind_rules) - 1]
    assert ours[confirm] == wind_rules[confirm] + (" It returns the confirmed plan's problems and cost, as check_plan "
                                                   "does.")
    assert ours[situation] == wind_rules[situation].replace(
        "(departed, berthed, frozen or open)",
        "(departed, berthed, frozen or open, and where the port berths a ship that gets in after its window, on its "
        "latest ETA until it is in)")
    assert ours[1] == wind_rules[1].replace(
        "ships report delays and unscheduled calls, the harbour master",
        "each ship's agent reports its ETA and calls in before it arrives, the line desk reports unscheduled calls, "
        "the harbour master")
    assert ours[len(wind_rules) - 1] == wind_rules[-1].replace(
        "Cost or rule breaks that news,",
        "An ETA that proves wrong is not news: a ship that arrives later than its agent said is berthed by the port at "
        "its next free slot, one that arrives earlier waits at anchor for its window, and the lateness and hours at "
        "anchor that follow are charged; a ship whose arrival you agreed arrives at that hour. Cost or rule breaks "
        "that news,")


def test_an_unscheduled_calls_firm_is_on_the_roster_from_the_start():
    """By design every firm of the week is deployed and peered at the start, so where an unscheduled call is its firm's
    only ship the firm's name is visible from the start; the ship and its timing are not, until its notice."""
    task = PACK.get("dock-36A-w35x1-standard-0-e03")
    [extra] = [e["ship"] for e in task.disruptions if e["type"] == "extra"]
    firm = lines.firm(task, extra)
    week = LinesWeek(task)
    deployed = [s["agent_name"] for s in tasks.lines_steps(task, 5.0) if s.get("a2a_agent_id") == "portsim-line"]
    assert firm in deployed and firm not in lines.firms_in(week.view)
    assert [s.id for s in task.ships if lines.firm(task, s.id) == firm] == [extra]
    assert extra not in [s.id for s in week.view.ships] and f"({firm})" not in lines.section(week)
    with pytest.raises(ValueError, match=f"^no ship {extra} in the week$"):
        week.status(firm, extra)
    lone = [(t.task_id, f) for t in PACK.tasks for f in set(t.rules["lines"]["firms"]) - set(
        lines.firms_in(LinesWeek(t).view))]
    assert lone == [(task.task_id, firm)]

