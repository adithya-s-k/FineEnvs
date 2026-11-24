# PortSimEnv v2–v5: design notes

The details behind the [README](README.md): how a week is played in watches, how AgentEnv's virtual clock and triggers
deliver it, the rules each env adds, how a week is graded and which weeks are played. The quay, the dock-v1 weeks, the
checker and reward v3 are PortSimEnv v1's, unchanged; [its design notes](../portsim-v1/DESIGN.md) explain them.

| env | weeks | adds to the week before it | graded against |
|---|---|---|---|
| `portsim-live` | 15 one-week dock-v1-eval weeks | v1's week, played as it unfolds: news arrives watch by watch and windows freeze | the hindsight optimum |
| `portsim-marine` | the same 15, from `tasks/dock-v1-marine` | pilots and tugs, shared hour by hour with the port's other 2024 traffic, and cuts to both | the marine hindsight optimum |
| `portsim-wind` | 15, each a marine week moved into a real weather week; built locally | the wind that blew instead of the simulated gale, and an ECMWF forecast at every watch | the anchor |
| `portsim-lines` | the 15 wind weeks; built locally | each ship's agent as its own A2A agent: ETAs that firm up, requested arrival times, late ships the port berths, a first-come queue and the cost of waiting at anchor | the lines anchor |

## A week in watches

A live week is a one-week dock-v1 task (`dock-<quay>-wNNx1-…`). Hour 0 is Monday 00:00 of the week; at hour 0 the agent
knows the published arrivals and the closures, which are planned works, and no unscheduled call. Every other
disruption of the task becomes a notice from the party that would send it (the ship, the harbour master, terminal ops,
the line desk, and in the marine port the tug company and the pilot station), revealed a fixed lead before the hour it
is about (`core/port_week_core/schedule.py`):

| news | revealed |
|---|---|
| closure | at hour 0 |
| late ship, bunched ships | 24 hours before the original arrival (bunching: the earliest planned hour) |
| unscheduled call, priority cargo | 24 hours before the arrival |
| emergency | 12 hours before the arrival |
| crane outage, gale, tug outage, pilot shortage | 24 hours before it starts |

Reveal hours are floored at 0 and rounded down to a 6-hour bulletin, and a ship's priority or emergency notice never
comes before its own delay or call. Each bulletin is a **watch**: watch 0 at hour 0, then one per bulletin with news,
2 to 9 watches a week. The week holds still between watches.

- **Tools.** `get_situation()` gives the watch, its time, the week as known now in v1's Markdown, the confirmed
  windows and the ships without one. `check_plan(plan)` checks windows merged over the confirmed ones and confirms
  nothing; `confirm_berths(plan)` sets or changes the windows it lists, and refuses each one it can't with the harbour
  master's reason. `advance()` ends the watch. A plan is a list of windows, `{"ship", "berth_hour", "section",
  "cranes"}`, as in v1; ships left out keep their windows.
- **The freeze.** From watch 1 on, a window that starts less than 6 hours after the current hour is frozen: it can't
  change, and a new or changed window must start at or after that line.
- **Planning calls.** `check_plan` and `confirm_berths` share 3 calls a watch.
- **Advance.** Time passes only when the agent calls `advance`: the port moves to the next bulletin, ships berth and
  sail on their confirmed windows, and the bulletin's messages arrive with the next tool result. After the last
  watch, `advance` returns `done` with the week's reward. An agent that stops early is graded on what it confirmed:
  the task's next step runs the remaining watches with no changes, then grades the week.
- **Turns.** Every live and marine week gets 47 turns, 5 a watch for the longest week's 9 watches plus 2; every wind
  week gets 52, those of 10 watches. One number for all weeks keeps the turn count from revealing how many bulletins
  are coming.

## On AgentEnv: the clock and the triggers

The envs need two things only AgentEnv's gateway provides, a virtual clock and triggers, so they run on AgentEnv
alone; there is no OpenEnv port of them. Each week is one task of these steps (`port_week_agentenv/tasks.py`; the lines
weeks add the ships' agents, [below](#a-lines-week-on-agentenv)):

```
deploy_env              portsim-live, portsim-marine or portsim-wind, behind the gateway
apply_server_config     urn:portsim:live-load/v1 {task_id}: the week as known at hour 0
modify_env_tool_access  hide port_notice from the agent's role
register_env_triggers   watch-1 ... watch-N
deploy_agent            portsim-llm
sync_env_clock          the week's start, rate 0 (stopped)
prompt_agent            the rules and the watch-0 briefing
apply_server_config     urn:portsim:end-week/v1: run the watches the agent didn't reach, grade and audit
env_outcome_verifier    portsim-live-verifier
```

**Triggers.** The trigger `watch-k` fires when `advance` returns `"watch": k`. It calls the hidden tool
`port_notice(event_id, name, text)` once for each of the watch's notices, in order, and its barrier holds the agent's
next call (up to 60 seconds) until they are applied. That is why the news arrives with the next result, and why the
rules ask the agent to call `advance` and `get_situation` in one turn. The env, not the task, decides what each notice
does to the week: `port_notice` applies the news and queues the message.

**The clock.** `sync_env_clock` arms the gateway's clock at the week's start, stopped, and the gateway hands the env
its clock URL through `urn:agentenv:clock/v1`. At each `advance` the env sets the clock to the next bulletin's time,
again stopped, through the gateway's `PUT /clock/set-time`. So the gateway's own `get_time` tool, the trajectory's
timestamps and the trigger log all show the watch's time, and no time passes while the model thinks. `get_time` calls
never reach the env, so the env's call count leaves them out.

**The audit.** A week is valid when every watch reached got exactly its scheduled notices, in order, before the
agent's next call; every notice after hour 0 came at least 6 hours ahead; and a finished week revealed each notice
once. `data/get` reports the plan, the notice log, the frozen windows of each watch, refusals, excuses, the audit, the
grade and the reward, never the task's reference plans. `portsim-live-verifier` scores a valid week with its reward
and adds the audit as a row of weight 0; for a week that never finished or failed the audit it raises, so the run is
not scored.

## The grade

The confirmed windows are checked against the week as it really happened and scored with berth_core's reward v3, as in
v1: a week that breaks a rule scores 0.2 × its share of clean ships, below 0.2; a valid week scores
`0.2 + 0.8·e^(−g/0.5)` with `g = max(0, cost − best) / (max(0, best − floor) + 100)`, where `best` is the hindsight
optimum (on the wind weeks, the anchor) and the floor is the cost no plan can avoid. Rewards are rounded to six
decimals, and the same windows always get the same reward.

**Excuses.** News can land on a window that is already frozen. At each notice the env checks the frozen windows on
their own, before and after the news; a rule break or a cost that only the news added is not charged, and
`check_plan` lists it as excused. A ship the agent moves into trouble after the news is charged. The wind and lines
ports judge excuses per ship instead ([below](#excuses-on-the-wind-weeks)).

## Which weeks are played

`tasks/live/references.jsonl` plays each of the 18 one-week dock-v1-eval weeks two ways, through the same week model
and grader as the env:

- **Rolling:** CP-SAT re-plans the week as known at every watch, with the frozen windows pinned and every other ship at
  or after the freeze line.
- **Naive:** keep each window that still fits; move the rest to the earliest legal hour at or after the freeze line,
  on the nearest section, with the same cranes.

A week is played when the rolling re-planner reaches the hindsight optimum under at least 3 of 4 solver
configurations (1 or 8 workers, each with or without an earliest-finish tie-break), so 1.0 is reachable on what the
agent could know. 15 of the 18 weeks qualify; on them the rolling re-planner scores 1.0 and the naive policy 0.202 on
average. `dock-36A-w15x1-busy-0` (every configuration stops above the optimum), `dock-36A-w16x1-standard-0` and
`dock-36A-w17x1-standard-0` (too few configurations reach it) are left out. CP-SAT runs on a deterministic time limit,
with the 8-worker search interleaved so that it reproduces; the tests replay the stored plans without ortools.

```bash
cd envs/port_week
uv run -p 3.12 --with ortools==9.15.6755 --with-editable core python tools/live_references.py   # about 4 minutes
```

## marine: pilots and tugs

Every berthing and every departure takes a pilot and tugs in its hour, from port-wide pools the quay shares with the
rest of the port's real 2024 traffic: ferries, cruise ships, tankers, car carriers and the container ships at other
quays. In any hour the week's ships may need no more pilots or tugs than are free for them; each ship moving in an hour
short of either breaks the rule, as a crane-pool or movement-limit break does. `get_situation` and the opening end
with a "Pilots and tugs" section: the pools, the tugs each ship takes, the cuts announced so far, and the pilots and
tugs free each hour up to hour 264. Each number comes from a public source or is labelled an assumption; the pack's
manifest (`tasks/dock-v1-marine/manifest.json`, under `marine`) carries the same list.

| rule | value | grounded in |
|---|---|---|
| who takes a pilot | every berthing and departure of a ship of 45 m or more | the port's [maritime operations FAQ](https://contentv5.portdebarcelona.cat/cntmng/gd/d/workspace/SpacesStore/0f925a59-a5c8-4c7f-9b44-28c80f50f88a/FAQ_ATRACS_es.pdf), item 15 (over 45 m and 500 GT); going by length alone is an **assumption** |
| how long a job takes | the hour the ship berths or leaves, for the pilot and the tugs alike | the [pilotage specification](https://www.boe.es/diario_boe/txt.php?id=BOE-A-2020-15568), annex III, sizes the service on a mean of 1 hour a job; which hour is an **assumption** |
| pilots | 7 on duty every hour | **assumption**, calibrated: the smallest pool the port's 2024 traffic exceeds in under 1% of hours (0.25%) |
| tugs | 8 on duty every hour | the port has ["8 or more"](https://infopuertos.com/la-fuerza-que-mueve-los-puertos-espanoles/); all 8 on duty at once is an **assumption** (2024 exceeds it in 0.01% of hours) |
| tugs per movement | 0 under 120 m, 1 under 200 m, 2 under 300 m, 3 from 300 m; ferries, cruise ships and yachts 0 | **assumption**: the port publishes no table; the master decides with the pilot ([towage specification](https://www.boe.es/diario_boe/txt.php?id=BOE-A-2015-10895), prescription 3) |
| wind | one more tug per movement inside an announced window above 25 kn, except Ro-Ro ships and ferries under 200 m, and yachts | the port's [traffic ordinance](https://www.boe.es/diario_boe/txt.php?id=BOE-A-2023-6719), 4.1.2.2 |
| other traffic | every 2024 call of a ship of 45 m or more that doesn't stop at the task's quay, from hour 0 to 264: a movement each time it berths, and one when it leaves a berth for the anchorage or the sea | the port's 2024 calls of every ship type ([data/barcelona/SOURCE.md](data/barcelona/SOURCE.md)) |
| cuts | a tug outage (2 tugs for 24 hours) and a pilot shortage (2 pilots for 12 hours), each announced 24 hours ahead | towage specification, prescription 2 and annex 1, for tug outages; sizes and timing are **assumptions** |

Response times (30 minutes for a pilot, 25 for a tug) fall inside the hour and aren't modelled, and neither is waiting
for a tug: a short hour is a rule break. The gales are PortSimEnv's simulated ones, so the other traffic keeps its 2024
hours through them, which is an **assumption** too.

The cuts arrive on bulletins the week already has, after its own disruptions, so the live week's event ids and watches
hold. The marine hindsight optimum is CP-SAT's with the whole week known, proven optimal, with one cumulative
constraint per pool and the wind's extra tug; it is never below the live week's. The rolling re-planner plans with the
pools; in an hour the pools can't cover for the frozen windows' own movements, it lets those have what they need,
since the grade excuses them. The naive policy is the live port's and knows nothing of pilots and tugs. The 15 marine
weeks are the 15 live weeks; on them the rolling re-planner scores 1.0 and the naive policy 0.187, infeasible in 9.
The pack shares dock-v1-eval's task ids, so the two are never loaded together (`PORT_WEEK_TASKS_DIR` holds it, never
`BERTH_TASKS_DIR`).

```bash
cd envs/port_week
uv run -p 3.12 --with ortools==9.15.6755 --with-editable core python tools/marine_references.py   # about 7 minutes
```

The script downloads the port's 2024 calls once into `~/.cache/portsim-live/raw/8e726a4/`, refuses any file but the
pinned one, and rewrites the pack and the references byte for byte.

## wind: real wind and its forecasts

A wind week is a marine week moved into a real Barcelona weather week of 2023 to 2025: its hour 0 is the weather
week's Monday 00:00 UTC, hour for hour, and the task keeps its 2024 week, so no real date reaches the agent. The
simulated gale is gone; every other bulletin stays, word for word and at its hour. The hours the wind blew above the
port's thresholds become the week's no-movement windows (`rules["no_moves"]`), which the grade reads and the agent
never sees. At every watch Barcelona Port Control sends the latest ECMWF forecast published by then, adjusted to the
port's anemometer; the agent plans on the forecasts as they were issued, and the week is graded on the wind that
blew. Task ids end in the weather week's id, `-e00` to `-e15`.

| rule | value | grounded in |
|---|---|---|
| thresholds | above 25 kn no ship of 300 m or more berths or leaves, and each movement takes one more tug; above 30 kn no ship moves | the port's [traffic ordinance](https://www.boe.es/diario_boe/txt.php?id=BOE-A-2023-6719), 4.1, 4.1.2.1 and 4.1.2.2, simplified to no-movement windows as in v1 |
| anemometer | Meteocat's XEMA station Y7, Port de Barcelona – Bocana Sud | the ordinance's annex III measures at the Dique Sur's red light; taking Y7 for it is an **assumption** |
| reading | an hour is above T kn when either half-hour has 1.045 × the 30-minute mean or 0.664 × the 3-second gust above T | annex III takes the 10-minute mean, or the largest 1-minute mean above 1.25 T; the factors (10- over 30-minute mean; 1-minute mean over 3-second gust, 0.83 / 1.25) are **assumptions** |
| windows | the hours above each threshold over the 264 hours from Monday 00:00 UTC, gaps of 2 hours or less merged | **assumption** |
| validation | readings count as published, validated or not; each window records its share of unvalidated readings | **assumption** |
| weather weeks | 16, fixed: storm weeks of 2023-W04 to 2025-W52 with a window above 25 kn of 4 hours or more, or an hour above 30 kn (not resting on one unvalidated gust), and bust weeks, below that rule, whose forecasts showed wind above 25 kn that didn't blow | PortSimEnv's choice |
| forecast | ECMWF HRES, 4 runs a day: 10 m wind every 3 hours to 72 hours, at the nearest sea grid point (0.4° before 2024-02-28 06 UTC, 0.25° from then) | [ECMWF open data](https://www.ecmwf.int/en/forecasts/datasets/open-data), CC BY 4.0 ([data/ecmwf/SOURCE.md](data/ecmwf/SOURCE.md)); that it stands in for the port's own forecasts is an **assumption** |
| publication | a run is in force from 9 hours after its start, with all its steps; a run that is missing, late or incomplete leaves the one before in force | **assumption**, after the times the runs appeared in ECMWF's public bucket |
| calibration | each run's speeds mapped to Y7's reading by quantiles (99 percentiles), one map per weather year and lead block (0–23, 24–47, 48–72 hours), rounded to whole knots; each year's maps are fitted on another year's 00 and 12 UTC runs on the same grid | **assumption** |
| forecast watches | the marine watches, plus one at 00:00 or 12:00 (hours 12 to 156) when the run in force shows an hour above 25 or 30 kn, 6 to 42 hours ahead, that the run at the watch before didn't | **assumption** |

**The week as known** keeps only the latest forecast. Its no-movement windows are the wind observed in the hours
before the watch and the forecast's windows from the watch on, and the wind tug follows them too. `get_situation` and
the opening end with a wind section after the pilots and tugs: the thresholds, the forecast with its issue time and
3-hourly knots, and what has blown so far. Nothing in it comes from a run published after the watch or from an hour
after it.

### Excuses on the wind weeks

Excuses are judged per ship, at its freeze, on the whole plan. A ship first frozen at watch k is compared with the week
as known at watch k−1, its news and its forecast; a ship never frozen, with the last watch reached. A rule break or a
cost the week adds to that, as known now or, once it ends, as it happened, is excused, once. So wind no forecast had
shown by then, and later news, are waived; wind the forecast showed and that blew is charged, as an ignored warning;
and a false alarm withdrawn after the freeze leaves no waiver. A problem ships share (an overlap, or pilots, tugs or the
movement limit short at an hour) is judged for each ship as that ship's last decision saw it. The per-notice excuse of
the live and marine weeks is off, so nothing is waived twice. With no wind at all, a feasible wind week grades exactly
as its marine week.

### The anchor

Hindsight pays for all the wind, while the agent's cost is net of its excuses, so following the forecast can cost
less than the hindsight optimum; anchored on hindsight alone, a plan clearly worse than a forecast-follower's would
still score 1.0. A wind week therefore scores 1.0 at its **anchor**: the lower of the hindsight optimum and the best
cost of the forecast-following re-planner over its 4 solver configurations. The anchor is the task's `optimal_cost`,
with the hindsight plan as its reference plan, and the floor is the cost no plan can avoid in that wind.

### References and the deal

`tasks/wind/references.jsonl` (built) plays every wind week five ways, through the same week model and grader as the
env: **hindsight**, CP-SAT with the whole week known in the wind that blew; **rolling**, the marine re-planner on the
week as known at each watch, forecast included (a frozen ship may sit in a window forecast after its freeze, since
nothing can move it); **blind**, the same with the wind taken out of the week as known; **hold**, on bust weeks, holding
every window any forecast so far has shown; and **naive**, the live port's policy.

A marine week and a weather week make a wind week when (a) the rolling re-planner costs no more than the hindsight
optimum in at least 3 of its 4 configurations, so 1.0 is reachable on what the agent could know; (b) the blind
re-planner, or on a bust week the hold one, scores 0.9 or less in at least 3 of 4, so the forecast matters; (c) the week
has 10 watches or fewer, which the 52 turns cover; and (d) no ship alongside at hour 0 is due to leave inside a window
above 30 kn. The plugin screened all 240 pairs of the 15 marine weeks and the 16 weather weeks and dealt 15 tasks from
those that qualify: in round 1 the i-th marine week takes the first qualifying storm week no other has, looking from the
i-th storm week on in date order; each bust week then goes to the first marine week it qualifies on; below a gate of
12, round 2 lets each marine week in turn take another qualifying storm week, which another marine week may already
have, up to 15. `tools/build_wind.py` fixes the 16 weather weeks and the 15 pairs as identifiers, so the task ids don't
move when Meteocat validates a reading.

### Building the wind weeks

Nothing derived from the Y7 readings is committed ([data/meteocat/SOURCE.md](data/meteocat/SOURCE.md)). The build
writes `tasks/dock-v1-wind/` and `tasks/wind/` on your machine, each with a `NOTICE`, in six steps:

| step | what it does |
|---|---|
| `fetch` | the 37 months of Y7 readings into `~/.cache/portsim-live/wind/y7/` (a rerun fetches only the missing ones), each checked against its pin in `data/meteocat/pins.json`: exact, values-match (validated since, same readings), values-changed or missing |
| `weather` | the 16 weather weeks: the windows that blew, and the forecasts as calibrated knots, from the readings and the committed ECMWF points ([data/ecmwf/points.jsonl.gz](data/ecmwf/points.jsonl.gz)) |
| `solve` | the hindsight, rolling, blind, hold and naive plays of the 15 pairs, cached by a digest of their inputs, the solver version and the solving code |
| `pack`, `references` | the wind pack and its references |
| `verify` | compares each built file, line by line, with its pin and lists any task that differs |

`python tools/build_wind.py` runs them all and stops at the first failure; two builds give the same tasks, references
and weather, and the manifest and NOTICE record the dates. The weather file is checked for information only, since
validation changes only its unvalidated shares; the pack's tasks and the references must match their pins. Outputs
that differ stay in place, but `rollout.py` won't play them until `--accept-drift` records the drift in the pack's
manifest; their scores are then not comparable with the published runs.

## lines: the shipping lines answer back

A lines week is a wind week with the same freeze, wind, pilots and tugs, and with the shipping lines in it
(`core/port_week_core/lines.py`). Each ship has its agent, the firm that deals with the port for its line, and each
firm with a ship in the week is deployed as its own A2A agent, `portsim-line`, peered with the planner. The published
schedule is the line's estimate from 72 hours out, so it doesn't give the arrivals away; the agents report ETAs that
firm up as the ships near, answer the planner's questions and agree arrival times; and the week is graded on when the
ships really arrived.

### The rules and their sources

Each number comes from a public source or is labelled an assumption; the pack's manifest
(`tasks/dock-v1-lines/manifest.json`, under `lines.grounding`) carries the sources in brief.

| rule | value | grounded in |
|---|---|---|
| agents | the shipping agent the port's 2024 record names for the call (CONSIGNATARI); a week has 7 to 9, for its 14 to 28 ships | the port's 2024 calls of every ship type ([data/barcelona/SOURCE.md](data/barcelona/SOURCE.md)), pinned by sha256 as for the marine pack; an unscheduled call takes its ship's nearest 2024 call |
| schedule | the line's estimate from 72 hours out: each ship arrives at its scheduled hour plus round(20% × its bias × 72) hours, mostly later, a delayed or bunched ship with its delay on top; the scheduled hour is never later than the ship could still make by slowing from hour 0. An unscheduled call keeps the wind week's arrival, which its notice announces | **assumption**: the ETA error below, from 72 hours out |
| reports | ETAs 48 and 24 hours out, and an arrival call 1 hour out; a report made between watches arrives with the next one, and the latest made before hour 0 opens the week | the EU and Spain require a notice at least 24 hours before arrival ([Directive 2002/59/EC](https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX%3A32002L0059) art. 4; [Orden FOM/1194/2011](https://www.boe.es/buscar/act.php?id=BOE-A-2011-8339) art. 9); the [IMO Just In Time Arrival Guide](https://wwwcdn.imo.org/localresources/en/OurWork/PartnershipsProjects/Documents/GIA-just-in-time-lores.pdf) puts berth planning at about 48 hours; the 2025 berthing ordinance ([BOE-A-2025-5157](https://www.boe.es/diario_boe/txt.php?id=BOE-A-2025-5157)) defines arrival by the call made 1 hour before buoy S. The 48-hour report and delivery at the next watch are **assumptions** |
| ETA error | 20% of the hours left, counted to 72 at most, times the ship's bias: positive (the ship later than its ETA) in 80% of ships, its size half-normal with mean 1, capped at 4, drawn once per ship. The mean absolute error is about 10 hours at 48 hours out and 5 at 24; no ETA is later than the slowest the ship can arrive, and the call is exact | **assumption**, inside the published range: Busan New Port 2022–23 (agent ETAs 24–48 hours out, mean error 5 h 15 min, [JMSE](https://doi.org/10.3390/jmse11061234)), Antwerp 2019 (10–15 hours, [Veenstra and Harmelink](https://pure.eur.nl/ws/files/58012351/Veenstra_Harmelink2021_Article_OnTheQualityOfShipArrivalPredi.pdf)), MSC's Barcelona–Valencia loop 2017–18 (80% later than the leg's ETA, 60% of those within 10 hours, [STM](https://stm-stmvalidation.s3.eu-west-1.amazonaws.com/uploads/20200225092917/STM_ID_5.3.4_STM-VALIDATION-USE-CASES-ANALYISIS-1.pdf)) |
| commitment | an agent commits to an arrival hour once its ship has sailed from its previous port or its own estimate is within 48 hours; before that it refuses and gives its ETA | the IMO Just In Time Arrival Guide's berth-planning horizon, about 48 hours |
| requests | once it can commit, an agent agrees an hour the ship can make at cruising speed, or by slowing once it has sailed, and the ship arrives at that hour; otherwise it offers the earliest or the latest hour the ship can make, and an offered hour can be agreed at once | the requested time of arrival of the IMO Just In Time Arrival Guide: the port requests an arrival time and the ship adjusts its speed to make it, or answers with what it can make |
| speeds | cruise 17 kn, slowest 10 kn: a ship h hours out can arrive up to 0.7 h hours later, and only once it has left its previous port | **assumption**: two-stroke engines run continuously down to 10% load (Wärtsilä); cruising speeds of 12–20 kn on MSC's loop; the IMO JIT container study's economical speed is 10 kn |
| previous port | its distance from Barcelona, so a ship from Valencia (165 nm, more than a quarter of the ships in these weeks) can absorb at most about 7 hours | [NGA Pub. 151](https://maritimesafetyinnovationlab.org/wp-content/uploads/2015/01/pub151-distances-btw-ports.pdf), Distances Between Ports (2001; no copyright claimed); a port it doesn't list takes its nearest listed neighbour |
| news | until the port's news about a delayed or bunched ship, its agent works from the plan as it stood; at the news an earlier agreement stands if the ship can still make that hour, and lapses otherwise | **assumption** |
| late ships | the port berths a ship that gets in after its window starts at the first hour a free stretch of quay fits it, breaking no rule and moving nobody else; a ship that gets in early waits at anchor for its window | **assumption** |
| queue | first come, first served for the ships whose arrival was agreed, a slowed ship keeping its turn; the exceptions are an earlier ship the wind holds, or one for whose stay the closures leave no stretch of quay long enough | Barcelona's container terminals berthed first come, first served in 2024 ([El Mercantil](https://elmercantil.com/2024/11/05/barcelona-ultima-su-ordenanza-de-prioridad-de-escalas-para-pertrecharse-contra-la-congestion/), [Transporte XXI](https://www.transportexxi.com/preferencia-para-atracar-en-barcelona/)), and the 2025 ordinance made it the general rule (art. 3), exempting a ship the quay can't take. Keeping a slowed ship's turn is the virtual-arrival arrangement. Holding the queue only for agreed ships is an **assumption**: the planner can't foresee an unagreed ship arriving off its ETA. The ordinance's other exceptions (occupancy thresholds, BEST's local-cargo priority) are left out |
| anchor | an hour at anchor costs the ship's sections, as an hour late does | the main-engine fuel a ship saves by slowing instead of waiting is worth 1.3–1.8 hours of its charter (Third IMO GHG Study table 14; VLSFO $626/t, Ship & Bunker's 2024 average; Alphaliner charter rates, June 2024), and an hour late costs more than charter alone, so 1 is the round value inside the evidence |

The pack is the wind pack with each ship's agent, its previous port's distance and its ETA bias; each arrival moved off
its schedule by its agent's error from 72 hours out, a delayed or bunched ship's whole story with it; and the wind
week's delay, bunching and unscheduled-call notices rewritten to give the agents' ETAs at the watch they arrive. The
watches are the wind week's, unchanged: the agents' routine reports add none.

### A lines week on AgentEnv

The task is the wind week's, with the agents added (`port_week_agentenv/tasks.py`, `lines_steps`):

```
deploy_env              portsim-lines, behind the gateway
apply_server_config     live-load: the week as known at hour 0
modify_env_tool_access  role default: hide port_notice, ship_status and agree_arrival
modify_env_tool_access  role line: every tool off, then ship_status and agree_arrival on
register_env_triggers   one trigger per watch; each delivers the watch's notices, the agents' reports among them
deploy_agent            portsim-llm, the planner
deploy_agent × firms    portsim-line, one per firm with a ship in the week, named by the firm's id, role line
peer_agents             the planner peered with every firm, so its model gets message_agent
sync_env_clock          the week's start, stopped
prompt_agent            portsim-llm: the rules, the watch-0 briefing, 72 turns
apply_server_config     end-week: the rest of the week, the grade and the audit
env_outcome_verifier    portsim-live-verifier
```

The planner keeps the wind env's tools, with `message_agent` added and these changes:

- `get_situation` ends with a section on arrivals and the ships' agents: their rules, which agent handles which ship,
  and each ship's arrival as known now (agreed, called in, its latest ETA or the published schedule). A window row adds
  `port_berth_hour` and `port_section` when the port would berth its ship elsewhere, on the ship's latest ETA until it
  is in; `check_plan` rows add each ship's hours at anchor (`anchor_h`).
- `confirm_berths` also returns the confirmed plan's problems and cost, as `check_plan` does, in the same planning
  call. The other envs' `confirm_berths` is unchanged.
- `advance` returns only the watch, the hour and the freeze line: which ships berthed or left is news the bulletin
  brings.
- `message_agent(agent, requests)` comes from the peering: up to 20 requests a message, each
  `{"ship": 7, "ask": "eta", "request_arrival": null}` for the latest ETA or
  `{"ship": 9, "ask": null, "request_arrival": 130}` for an arrival hour. The gateway's `get_time` stays visible.

How the agents and the port answer:

- **The agents.** A firm agent has no model. It answers each request from the env, through the gateway's `/step`
  route under its role: `ship_status` for an ETA, `agree_arrival` for an hour. The same requests therefore always get
  the same answers, and a week replays the same way. The agents' calls don't count as the planner's.
- **Reports.** The agents' routine reports reach the planner the way all news does, through the watch triggers: a
  report made between watches arrives with the next bulletin, and an agreed ship's reports restate its agreement. An
  agent's later ETAs for a ship never fall below the earliest hour it has offered for it. The audit holds the reports
  to no lead, since they report rather than announce.
- **News.** The wind week's notices of a delayed ship and of ships bunched by the Red Sea diversions keep their
  watches, and give the agents' ETAs. At the news, an earlier agreement stands if the ship, on its true track at the
  paces its agreements have set since the first, can still make that hour; otherwise it lapses, the ship cruises in
  from where it is, and the notice gives its agent's estimate of that arrival. Without an agreement, the notice's ETA
  is what the agent last said moved later by the delay.
- **The port's dispatch.** A ship that gets in after its window starts is berthed at the first hour from its arrival
  when a free stretch of quay fits it, nearest its planned sections first, with the same cranes, around every other
  ship's window without breaking a rule, and never ahead of an earlier ship whose arrival was agreed and which still
  waits; nobody else moves. Late ships are placed hour by hour, in queue order; after 168 hours without a slot, a
  window just starts when its ship gets in, or once the wind lets it move.
- The firm agents call `/step` rather than opening an MCP session: AgentEnv's gateway runs its MCP lifespan per
  session, so a second client's short session would replace, then close, the env sessions and trigger driver the
  planner relies on.

### The lines grade and the references

The grade is the wind env's, on the windows as the port runs them: a late ship where the port berths it, and each hour
a ship waits at anchor charged its sections. The cost is lateness, plus hours at anchor, plus move penalties; a ship
that starts berthing while an earlier agreed ship waits breaks the queue rule, and any rule break makes the week
infeasible, under 0.2. The excuse is the wind env's, per ship: cost or a rule break that news, or wind the forecast
didn't show, brings to a window after its last chance to change is not charged. An ETA that proves wrong is not news:
the excuse weighs every window against the ships' real arrivals, so what a ship arriving off its ETA brings is
charged. The score is reward v3's against each week's **anchor**, the lower of hindsight's cost and the follower's
best.

`tasks/lines/references.jsonl` (built) plays every lines week four ways, through the same week model and grader as the
env:

- **Hindsight:** CP-SAT knowing every arrival. It asks each ship at the first watch its agent could commit, and not
  before its news, and protects in the queue only the ships it slows, as the port protects only agreed ones. It is
  the anchor in 13 of the 15 weeks.
- **Follower:** re-plans at every watch with CP-SAT on the arrivals as known, the queue and the cost of waiting, and
  requests arrival times through the same agents: for each ship whose window would freeze before the next watch, each
  ship it plans to slow, and each ship due within 12 hours of the next watch. It takes each counter-offer, and a
  refusal is a fact it keeps. Its best cost over its four solver configurations is 0.78 to 2.25 times hindsight's,
  below it in 2 weeks, where it is the anchor, and it scores 0.548 in its first configuration (0.200 to 1.0 by week).
- **Silent:** the follower that never asks: 0.242. It keeps every week feasible, since the port berths a ship that
  misses its window and only agreed ships hold the queue, but its ships wait 2,386 hours at anchor over the 15 weeks,
  against the follower's 550.
- **Naive:** the live port's policy: 0.184, feasible in 7 of the 15 weeks.

The references keep the queue stricter than the grade, with no wind or closure exception, so hindsight isn't proven
optimal: a planner that uses an exception can beat the anchor, and scores 1.0. A week qualifies when (a) at least 3 of
the follower's configurations finish feasible, (b) the silent follower scores at most 0.9 in at least 3 of 4, so asking
the agents matters, and (c) it has at most 10 watches. All 15 wind weeks qualify, each with all 4 of the follower's
configurations feasible and the silent follower at most 0.9 in all 4. A lines week gets 72 turns, 7 for each of 10
watches and 2 more, so the turns don't give the watch count away.

### Building the lines weeks

The lines weeks derive from the wind weeks, so nothing of them is committed either. `tools/build_lines.py` builds
them after `tools/build_wind.py` and writes `tasks/dock-v1-lines/` and `tasks/lines/` on your machine, each with a
`NOTICE`, in four steps:

| step | what it does |
|---|---|
| `pack` | the built wind weeks, which must match their pins or have their drift accepted, with the agents from the port's pinned 2024 calls, staged in `~/.cache/portsim-live/lines/stage/` |
| `screen` | the hindsight, follower, silent and naive plays of the 15 weeks, cached by a digest of each week, the solver version and the solving code (27 minutes on 14 processes here, set by the slowest week) |
| `references` | the pack with each week's anchor, and the references, replaying the follower's stored plans and requests |
| `verify` | compares the pack's tasks, its manifest and the references with their pins in `data/meteocat/pins.json` and lists any task that differs |

`tools/lines_references.py` holds the players and the pack, as the plugin's script does, and runs its steps on their
own. The references leave out each week's solve time, which the plugin's file also records, so two builds write the
same bytes. Outputs that differ stay in place, but `rollout.py` won't play them until `--accept-drift` records the
drift in the pack's manifest.
