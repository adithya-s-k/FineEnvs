# AirOpsEnv design (v1 and v2)

How the two envs work: what a day holds, what is known when, how the plan is flown, what it costs, the references the
grade reads, and the contract the env servers keep. The README says how to run them; each module's docstring has the
rule in full. Module paths are under `envs/airops/core/airops_core/` unless they name `airops_agentenv`.

**Contents:** [Units and time](#units-and-time) · [v1: one desk at one hub](#v1-one-desk-at-one-hub) ·
[v2: the ops control center](#v2-the-ops-control-center) · [The env servers](#the-env-servers) ·
[What no tool reveals](#what-no-tool-reveals) · [Determinism and replay](#determinism-and-replay)

## Units and time

- Every time is an **integer number of minutes from 00:00 hub-local on the task's date**; times pass 1440 for flights
  after midnight. They are shown as `HH:MM` local, with `+1` past midnight. The gateway's clock reads the same minute as
  a UTC instant (`schedule.virtual_time`).
- Costs are in **passenger-minutes**, at the FAA's $1.07 per passenger-minute (2025 dollars) where a dollar cost is
  converted.

## v1: one desk at one hub

### The task

One line of `tasks/b6-v1/tasks.jsonl` (`core/model.py`):

```
task_id    "b6-jfk-20230714-1400"            # carrier-hub-date-windowstart (local HHMM)
split      "eval" | "control" | "train"
carrier, hub, date, tz
window     [840, 1200]                       # the disruption window that defines the scope
start      780                               # first watch = window[0] - 60; flights flown before it are fixed
turn       {"A32X": 49, "E190": 37, "A220": 47}   # minimum turn by aircraft family
dmax       360                               # a flight leaves at most dmax after its schedule, else auto-cancels
airports, tails, flights                     # each flight with its schedule, block, pax, family, lb and actual
caps       {"JFK:arr": {"13": 7, ...}, ...}  # the truth, by local hour
nominal    {"JFK:arr": {"13": 9, ...}, ...}  # JetBlue's scheduled movements per hour: the view with no program
news       [{"id", "at", "kind": "advisory" | "taf" | "metar" | "outlook", "speaker", "text", "data"}]
frozen     [...]                             # legs flown before start
reference  {"floor", "hindsight", "anchor", "follower", "blind", "naive", "airline", "gap_pct", "qualifies",
            "clauses", "hindsight_plan", "follower_plan"}
sources    {...}                             # sha256 of each pinned input
```

- `lb`, a flight's exogenous delay: `max(0, DepDelay − LateAircraftDelay)` for a flight that flew; for a cancelled one
  the 75th percentile of that among JetBlue departures from its origin within 90 minutes.
- `caps[a:kind][h]`: JetBlue's ration-by-schedule share of the movements all carriers actually made at airport `a` in
  local hour `h`, never below what JetBlue's in-scope flights moved there. Hours not listed are unconstrained.
- The scope: the tails that fly a leg touching the hub inside the window, and their legs from `start` to the end of the
  day.

### Decisions and execution

The planner controls, per flight, **cancel** (or restore) and **not_before** (hold the departure until a minute).
Aircraft of one family at one station are interchangeable, so the env dispatches them (`core/sim.py`):

1. Flights are taken in order of their earliest candidate departure, `max(sched_dep + lb, not_before)`.
2. A flight takes the first-ready aircraft of its family at its origin; it leaves at the later of its candidate time
   and the aircraft's ready time (previous arrival plus the family's turn).
3. If the departure hour at the origin or the arrival hour at the destination is full, the flight moves to the next
   hour with room at both and is queued again.
4. A flight that can't leave by `sched_dep + dmax` is cancelled automatically.
5. At the end of the day each aircraft not where tomorrow needs it is ferried there.

### Information and the freeze

At a watch at minute `t` (`core/known.py`):

- a flight's `lb` is known once `sched_dep ≤ t + 120`; before that it is 0;
- an hour's cap is known once `60·h ≤ t + 240`; before that the nominal schedule stands in;
- news with `at ≤ t` has arrived, delivered at the first watch at or after it was issued (`schedule.timetable`).

Watches run every 60 minutes from `start` until every flight is frozen or flown. Nothing is frozen at the first watch.
From the second on, flights the plan has leaving before `t + 60` are frozen and stay frozen, and the env holds any other
flight the plan would have leaving before that line to it (`held_to_freeze_line`), so no decision reaches back past it.
Each watch allows 3 planning calls (`check_plan` or `confirm_changes`); a day gets `5 × watches + 4` turns.

### Cost and reward

- Cost = Σ pax × max(0, arr − sched_arr) + 523 × Σ pax of cancelled flights + Σ ferries (`core/cost.py`).
  - 523 = NEXTOR's 457-minute average re-accommodation delay for disrupted passengers plus refund exposure (0.3 take-up
    × the $235 DB1B median fare ÷ the FAA's $1.07 per passenger-minute).
  - A ferry costs 92 × (30 + miles / 8): A4A's $98.41 airline cost per block minute ÷ $1.07, over an assumed block.
- Reward = 0.2 + 0.8 · exp(−g / 0.5), g = max(0, cost − anchor) / (anchor − floor + 0.02 · anchor) (`core/reward.py`).
  There are no rule breaks: the env never strands a flight.

### References (`refs/`, `tools/references.py`)

- `hindsight`, the anchor: the time-line MIP on the truth (HiGHS, 1-minute arcs, proven to 0.01%).
- `floor`: the same without aircraft routing, each flight alone at its earliest time under the caps.
- `follower`: the MIP re-solved at each watch on the day as known (persistence forecast for unknown delays and caps),
  freezing as the env does, its final plan flown on the truth. Its watch solves prove their optimum and then pick the
  plan with the earliest weighted departures, so it comes out the same on every machine.
- `blind`: the follower with no forecast. `naive`: the empty plan. `airline`: JetBlue's own cancellations and
  departure times as the plan (shown, never the anchor).

## v2: the ops control center

### The pack

`tasks/b6-v2/` holds `manifest.json` (each date's task id, split and the sha256 of every file) and one
`dates/<date>.json.gz` per date (`occ/pack.py`): the whole-day task (`programs/model.py`), its crew world frozen at
04:00 (every unit, every airport a unit may reach with its time zone, and the round trips of the planned rotations) and
the references the grade reads. The env never rebuilds units, so a day starts the same on any machine.

The task is JetBlue's whole network from 04:00 Eastern: every flight and tail from BTS on-time data, with its schedule,
its own delay (its departure delay less the late-aircraft and program parts) and its block time (capped at scheduled
plus 2 h); where each aircraft really was at 04:00; the FAA Command Center's ground stops and ground delay programs at
JFK and Boston as issued and as they ran (`programs/advisories.py`); and other carriers' arrivals at the two hubs, which
share the programs' slots (`programs/rbs.py`, `programs/unseen.py`).

### Watches, turns and the desks

Each hourly watch from 04:00 until the last flight's deadline runs a crew turn, an ops turn and, when ops sent crew a
request or a text, a crew reply turn; in mode `one`, one turn (`occ/day.py`). A turn opens on the desk's page, the day
as projected now on what is known (`occ/pages.py`, about 6,000 characters), and ends with `end_turn`, whose result is
the next page. A desk acts only in its turn and through its own tools: ops never names crews, crew never cancels or
holds, and an accepted request is the receiver's decision, carried out by the env. A request lapses when its flight
leaves, is cancelled or its time passes; a newer one on the same flight replaces it.

- **What the desks know.** A flight's own delay is known 2 hours before its scheduled departure (`?` until then); FAA
  advisories arrive as issued, delivered by the trigger of the watch that opens after them. The FAA's CDM refuses a
  slot claim outside the program's pool.
- **The freeze.** A hold can start no earlier than the freeze line, an hour ahead (at the first watch, from now), and
  flights due out before it are committed: they can't be cancelled, held or released, except that accepting the crew
  desk's CANCEL cancels one, since no crew can come.
- **Decided now.** Every decision of a turn is decided at the watch minute and reaches the truth from then on, so
  nothing before it changes; each commit checks that (`occ/run.py`).

### Execution

`programs/sim.py` dispatches the aircraft inside the FAA's programs: a ground stop at a flight's destination hub holds
it; a flight a ground delay program governs needs one of JetBlue's slots in that segment's pool (its claimed slot if it
can still make it, else the earliest free one), leaves at its EDCT and lands after its own block; any other hub arrival
needs room under the hour's nominal cap. A cancelled or late flight leaves its slot to the flights behind
(compression). A flight not gone 15 hours after its schedule, or by 03:00 from JFK or Boston, auto-cancels there.

`programs/crewsim.py` adds the crews: a flight ready to leave also needs a legal pilot unit and a legal cabin unit of
its type group (A32X, A220, E190) at its origin. Pilots fly under FAR 117 (Table A flight time, Table B duty by report
time and legs, the 2-hour extension, 10 hours' rest); flight attendants under 14 CFR 121.467 (14 hours' scheduled duty,
10 hours' rest) and 121.391's cabin minimums (`programs/crews.py`, every parameter with its source in
`programs/crew_params.json`). A unit (a pilot pair or a cabin team) stays together for the day; crews start the day
rested where they overnighted. The crew system crews every flight the crew desk leaves alone by standard practice (its
planned crew, else the cheapest of a reserve, a swap, a deadhead or the extension, each priced by the passengers' wait,
the deadhead, the hotel and its effect on tomorrow); it can't cancel, so where it is stuck it asks ops to (CANCEL), and
the flight waits for a crew or its deadline.

The crews are generated: JetBlue's pairings are not public, so `programs/pairings.py` builds trips from the published
schedule (never the flown day) by set covering, per crew kind and type group, preferring crews to stay with their
aircraft, and reserves are set at the middle of a sourced industry band (pilots 16%, flight attendants 19%).

### Cost and reward

`programs/cost.py`: each passenger's arrival delay; each cancelled passenger 523, plus one passenger-minute for every
minute the cancellation was decided later than 2 hours before the scheduled departure (an auto-cancel counts as decided
at its deadline); the airline's own aircraft cost at the FAA's Table 4-7 rates (each minute a flight leaves late costs
crew and fixed costs, each minute its block runs long the total; a cancellation saves fuel and maintenance over its
block); tonight's ferries; and from the crew sim, deadheads, hotels away from base and the units left out of place for
tomorrow's trips.

Reward = 0.2 + 0.8 · exp(−g / 0.5), g = max(0, cost − best) / max(B0 − best, 0.1 · best) (`occ/day.py`, `reward`):
1.0 at the best plan or under it, about 0.31 at B0, never below 0.2. `reward_L` is the same with the bound in place of
best.

### References (`programs/references.py`, `programs/crewplan.py`, `programs/crewbound.py`)

- `best`: the cheapest day of aircraft and crews together that the joint search (`crewplan.joint`) found knowing the
  whole day, replayed exactly in the crew sim with a legal crew on every flight it flies.
- `bound`: a proven lower bound on any day's cost (`crewbound.py`); `gap_pct` = (best − bound) / best.
- `B0`: the standard crew desk with the aircraft doing nothing, notice-priced (and `flat`, every cancellation at 523).
- `idle`: the crew system alone, nobody at the desks; `scripted`: the env's scripted desks in each mode
  (`tools/scripted_desks.py`); `airline`: JetBlue's own decisions replayed with the generated crews; `calm`: the calm day.
  The bound is never above any of them.

### The audit

A day is scored only if its audit passes (`OccDay.audit`): A1 every watch's advisories arrived once, through its
trigger, before the watch's first turn; A2 every call was made in its desk's turn and the turns ran in order; A3 no
decision reached back before its minute; A4 each desk kept to its role; A5 every flight flew with a legal crew, and the
named crews replay to the same cost; A6 the day's cost is not under the bound; A7 both desks were present to the end or
stood down.

## The env servers

Both are AgentEnv environments (`airops_agentenv`), MCP servers on port 18765 behind the gateway, with these
extensions:

| env | load | end | clock | hidden tool |
|---|---|---|---|---|
| `airops-day` (`live.py`) | `urn:airops:live-load/v1 {task_id}` | `urn:airops:end-day/v1` | `urn:agentenv:clock/v1` | `airops_notice(news_id)` |
| `airops-occ` (`occ/live.py`) | `urn:airops:occ-load/v1 {task_id, mode, turn_timeout_s}` | `urn:airops:end-day/v1` | `urn:agentenv:clock/v1` | `airops_notice(news_id)` |

- **One day per deploy.** Loading again is refused: it would replay the day with its news already seen.
- **The clock and the triggers.** The task arms the gateway's clock frozen at the first watch (`sync_env_clock`, rate
  0). v1's `advance` re-arms it at the next watch; v2 moves it when a watch opens. One trigger per watch with news calls
  the hidden `airops_notice` on the call that opens the watch (`advance` on v1, `end_turn` on v2), behind a barrier on
  that call, so the news comes with the agent's next tool result. v2's env delivers a watch's advisories itself if the
  trigger hasn't in 60 seconds, logged as late, which fails the audit.
- **v2's turns over MCP.** `wait_turn` long-polls (240 seconds, a re-poll costs no model call) until the desk's turn is
  granted; `end_turn` returns at once, so the trigger can fire. A desk holding its turn past the turn timeout (1,800
  seconds) is taken off. The caller's role on the gateway (`ops`, `crew` or `both`) picks its desk.
- **Strict schemas.** Every field of every tool is required and nullable where optional, so the schemas pass OpenAI's
  strict mode; a call may still leave a nullable field out.
- **end-day** runs whatever the agents left of the day, flies it on the truth and grades it; **data/get** serves the
  day's record, which the verifier scores (`agentenv/verifiers/`).

## What no tool reveals

No tool reply carries a reference, the bound, the reward or the truth ahead of the watch while the day is open. v1's
replies are the day as known (caps as known, delays once known, the plan's cost on the day as known); its `advance`
returns the reward only after the last watch, and `end-day` after the play. v2's pages and lookups are the projection on
what is known at the watch; `wait_turn` returns the reward only once the day is over. data/get serves the record, with
the grade only once the day is done. `tools/scripted_desks.py` checks it on each date it plays: with the delays and
blocks past a watch's look-ahead changed, every call and result up to that watch is the same.

## Determinism and replay

The same calls in the same order on the same pack give the same replies, grade and audit. v1's
`airops_agentenv.replay` plays an agent's transcript back through a `Day` and compares each output with the recording;
v2's `occ/replay.py` plays the env's own call log back through an `OccDay`, comparing each result's digest, and the
day's digest at the end. `tools/replay_runs.py` runs both over every published run: all 5 v1 runs and all 55 recorded
v2 days replay that way. One part of the v2 record can depend on the platform: the end-of-day crew problem
(`programs/crews.py`, `transport`, solved by HiGHS) can have more than one cheapest way to send the units home, and
which one HiGHS returns can differ between machines. On 2026-06-14 (one agent), replayed on macOS, two cabin units'
end-of-day shares come out 67.68 passenger-minutes apart from the record's, one higher and one lower, the day's cost
the same; replayed in the `airops-occ` image on Linux, as the board ran, the record matches in full.
