<div align="center">

<a href="https://huggingface.co/spaces/earakely-scale/AirOpsEnv-AgentEnv"><img src="./assets/banner.jpg" alt="AirOpsEnv v2: JetBlue's operations control center in 3D on the Boston storm day of 13 July 2024, replayed from GPT-6.1 Sol's two desks: the ops controller and the crew scheduler at their pods under the video wall of the network, as the crew desk's hold of B6 515 is accepted" width="100%"></a>

<h1>AirOpsEnv v1–v2: the storm day and the ops control center</h1>

<h3>Run JetBlue through a real storm day: one hub's desk, then the whole network from the two desks of its control center</h3>

<p>JetBlue's storm days at JFK and Boston rebuilt from public BTS, FAA and NWS records and played watch by watch on
AgentEnv's virtual clock, with the FAA's ground stops and ground delay programs arriving as they were issued, and the
pilots and flight attendants under their federal duty rules.</p>

<a href="https://huggingface.co/spaces/earakely-scale/AirOpsEnv-AgentEnv"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Space-Replays%20in%203D-FFD21E?style=for-the-badge&labelColor=1a1a1a" alt="Replays in 3D" height="32"></a>
<a href="https://huggingface.co/datasets/earakely-scale/AirOpsEnv-AgentEnv"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-Days%20%26%20runs-4F46E5?style=for-the-badge&labelColor=1a1a1a" alt="Dataset" height="32"></a>
<a href="https://github.com/scaleapi/agentenv-framework"><img src="https://img.shields.io/badge/framework-AgentEnv-3B82F6?style=for-the-badge&labelColor=1a1a1a" alt="AgentEnv" height="32"></a>
<a href="https://github.com/earakely-scale/agentenv-airops-plugin"><img src="https://img.shields.io/badge/GitHub-agentenv--airops%20plugin-181717?style=for-the-badge&logo=github&labelColor=1a1a1a" alt="The agentenv-airops plugin on GitHub" height="32"></a>

<sub>Two GPT-6.1 Sol desks run JetBlue's network through the Boston storm of 13 July 2024; every run replays in the
Space. The control-center room is drawn for display only; its wall map is us-atlas (U.S. Census Bureau boundaries) and
world-atlas (Natural Earth) under NEXRAD radar via the Iowa Environmental Mesonet.</sub>

</div>

---

## The task

An airline's operations control center keeps the schedule flying when the weather doesn't: it cancels and holds
flights, and it finds crews for the flights that keep flying. Both envs play one of JetBlue's real storm days at JFK or
Boston. The day starts with only what the airline knew then; the rest arrives in hourly watches: each flight's own
delay 2 hours before it leaves, and the FAA Command Center's ground stops and ground delay programs as they were
issued. Time passes only when the watch ends: when v1's planner advances, or v2's desks have ended their turns. After
the last watch the day is flown on what really happened and graded once, with no judge.

| env | the day | graded against |
|---|---|---|
| v1 `airops-day` | one hub's 3 to 6 storm hours, about 60 to 190 flights on 30 to 60 aircraft, in 10 to 22 watches, with the METARs, TAFs and forecast discussions as issued; aircraft only | the hindsight optimum a MIP proves |
| v2 `airops-occ` | JetBlue's whole network from 04:00, about 700 flights, 220 aircraft and 900 crews and reserves, run by two desks that talk: the ops controller and the crew scheduler | the best plan of aircraft and crews a solver found in hindsight, next to its proven bound |

**v1, one desk.** The planner cancels, restores and holds flights. Aircraft of one type at an airport are
interchangeable, so the env dispatches them, keeps each hour's slot cap at JFK, Boston, Newark and LaGuardia, and
ferries aircraft at night to where tomorrow needs them.

| tool | what it does |
|---|---|
| `get_situation()` | the time, the freeze line, the slot caps as known for the next 8 hours, the FAA programs in effect, the latest METAR and TAF, every flight and where the aircraft are |
| `check_plan(changes)` | runs changes over the plan on the day as known, without committing them |
| `confirm_changes(changes)` | commits them; with `check_plan`, 3 planning calls a watch |
| `advance()` | ends the watch: the clock moves an hour and flights leave as planned |

**v2, two desks.** The ops controller runs the flights and aircraft with the FAA: it cancels flights and round trips,
holds them, and claims and swaps arrival slots in the FAA's programs. The crew scheduler runs the pilots (FAR 117) and
flight attendants (14 CFR 121.467): it calls reserves, swaps crews between trips, deadheads them and rests them in
hotels. Neither can do the other's job, so they talk, through typed requests the other desk accepts or declines (HOLD a
flight until its crew is ready, CANCEL, CANCEL_PAIR, COVER a flight) and free text. Each watch the crew desk takes its
turn, then ops, then the crew desk again if ops asked it something. A crew system crews every flight the crew desk
leaves alone by standard practice and can't cancel. In mode `one`, one agent runs both desks from one page, with no
channel.

| desk | tools |
|---|---|
| ops | `ops_flights`, `ops_program`, `ops_aircraft`, `ops_check`, `ops_confirm`, `ops_request` (COVER, TEXT), `ops_reply` |
| crew | `crew_flights`, `crew_crews`, `crew_reserves`, `crew_program`, `crew_options`, `crew_check`, `crew_confirm`, `crew_request` (HOLD, CANCEL, CANCEL_PAIR, TEXT), `crew_reply` |
| both | `end_turn` (its result is the next page), `wait_turn`, `resign` |

Each desk is its own A2A agent, `airops-desk`, in role `ops`, `crew` or `both`, and sees only its role's tools on the
AgentEnv gateway; v1's planner is `airops-llm`. These envs run on AgentEnv only: they need its gateway's virtual clock,
which the env moves at each watch, its triggers, which deliver each watch's news through a tool hidden from the agents,
and, for v2, its per-role tool access. [DESIGN.md](DESIGN.md) explains the watches, what is known when, the sim, the
crews, the references and the grade.

## Grading

**v1.** The final plan is flown on the day as it happened. Its cost, in passenger-minutes, is every passenger's
arrival delay, plus 523 per passenger on a cancelled flight, plus the night's ferries. The reward is
`0.2 + 0.8·e^(−g/0.5)`, with `g = max(0, cost − hindsight) / (hindsight − floor + 0.02·hindsight)`: 1.0 at the
hindsight optimum (a time-line MIP on the truth, proven to 0.01%), and never below 0.2, since the env never strands a
flight. The floor is the same MIP without aircraft routing.

**v2.** The day's cost adds, to the passengers' delay and cancellations (more when a cancellation comes under 2 hours
ahead), the FAA's rates for aircraft time on the ground and in the air, the night's ferries, crew deadheads, hotels
away from base and crews left out of place for tomorrow. The reward is `0.2 + 0.8·e^(−g/0.5)`, with
`g = max(0, cost − best) / max(B0 − best, 0.1·best)`:

- **best** scores 1.0: the cheapest day of aircraft and crews together a solver found knowing the whole day, replayed
  exactly with a legal crew on every flight. The joint problem can't be proven optimal at this size;
- **bound** is a proven lower bound on any day's cost. The gap between them is published per date (eval dates 2.4% to
  13.9%, median 5.1%), and `reward_L`, the same reward against the bound, is published beside the reward;
- **B0** scores 0.308: standard crew-desk practice with the aircraft doing nothing (each flight's planned crew, else
  the cheapest of a reserve, a swap, a deadhead or the 2-hour pilot extension, cancelling only when no legal crew can
  arrive in time).

**What is excused.** Nothing. Every cost of the day is charged, including what news after a flight's last chance to
change brings to it; the anchor knows every delay and program in advance, so 1.0 is out of reach online. On v1 the
re-planning follower, the MIP re-solved at every watch on the day as known, is the online reference. A v2 day that
comes in under the best plan, which is not a proven optimum, scores 1.0 and is flagged.

**When a run is scored.** v1: only if the day finished and every watch's news arrived before the agent's next call.
v2: only if its audit passes: every watch's news on time, every call in its desk's turn, no decision reaching back
before its minute, each desk in its role, a legal crew on every flight flown, the cost not under the proven bound, and
both desks present to the end or stood down.

**How the days were picked.**

| env | eval days | kept when |
|---|---|---|
| v1 `airops-day` | 20 hub windows of pack `b6-v1`, 12 at JFK and 8 at Boston, drawn from a pool covering 13 months between December 2022 and July 2025; the 20 fall in 7 months, 2023-07-09 to 2025-07-14. The board below played 5 of them, a first look picked by hand (3 JFK and 2 Boston; 3 summer and 2 winter), not a random sample; the plugin didn't record how they were chosen | among the 60 (day, hub) pairs of most disruption, the window of 3 to 6 whole hours where doing nothing costs most over the hindsight optimum, among windows where every hub movement has a typed tail, weather and air traffic control (BTS causes B and C) caused at least 80% of the cancellations in scope and 30 to 80 aircraft are in play; then doing nothing costs at least 10% over the optimum, the follower lands within 15% of it and the optimum is proven within 60 s; at most one per date and hub |
| v2 `airops-occ` | 24 storm dates of pack `b6-v2`, 12 drawn per storm hub (8 summer, 4 winter), each from its own storm event, from a pool covering 19 months between December 2022 and July 2026; the 24 fall in 11 of them | a storm date is one where the FAA's programs at JFK or Boston controlled or held at least 5 JetBlue flights; a date stays only if the crew model passes a filter fixed before the run (JetBlue's real day, crewed by the generated trips, leaves at most 1% of its flights without a legal crew, and crews add at most 2% to its calm day), which kept 100 of 180 storm dates; the eval dates were then drawn with seed 0, and the 44 training dates are every kept date of an event with no eval date, so no storm is in both |

Each eval card in [results/](results/) (`eval-v1.json`, `eval-v2.json`) states the same, with the baselines and where
the runs are.

## Results

GPT-6.1 Sol played both envs, one run per day, through the agentenv-airops plugin, whose env servers, task steps,
verifiers and agents at [v0.3.3](https://github.com/earakely-scale/agentenv-airops-plugin/tree/v0.3.3) now live here;
every run replays through them byte for byte, to the same grade:

| Env | Runs | Mean reward (95% CI) | Valid days | Baselines on the same days | Cost per episode |
|---|---:|---|---:|---|---:|
| v1, one desk | 5 of the 20 eval days, picked by hand | **0.419** (0.319 to 0.547) | 5 of 5 | follower 0.730, doing nothing 0.247, JetBlue's own decisions 0.206 (on all 20 days: 0.729, 0.297, 0.210) | $0.60 |
| v2, two desks talking | 24 eval dates | **0.598** (0.545 to 0.658) | 24 of 24 | B0 0.308, the crew system alone 0.305, scripted desks 0.308 | $2.01 |
| v2, one agent at both desks | 24 eval dates | **0.550** (0.497 to 0.609) | 24 of 24 | as above; one scripted desk 0.295 | $1.02 |

- **v1.** Sol beats doing nothing and JetBlue's own decisions on every day it played and stays well below the
  re-planning follower: its days cost 18.9% over the hindsight optimum on average. Five runs on five days picked by
  hand are a first look, not a board, and not an estimate over the 20: doing nothing scores 0.247 on them against
  0.297 over all 20, and the blind follower 0.594 against 0.626.
- **v2.** Fixed before the first run, as the plugin's FIDELITY.md records: the code, the pack's hash, the dates, their
  order, the caps and the budget. Two Sol desks beat one Sol agent at both desks by +0.049 reward, paired by date (95%
  CI 0.024 to 0.075), higher on 18 of 24 dates, and both recover most of what standard crew-desk practice loses: at the
  median, about 60%.
- **It doesn't show that talking helps.** The two desks used about twice the compute (2.0× the spend, 2.0× the input
  and 2.5× the output tokens), and before any model ran, scripted desks failed the gate for talking: the channel was
  worth at least 5 points of the avoidable cost on 3 of 24 dates, against 15 needed.
- **Flagged.** On 2023-07-13 the two desks came in 1.5% under the best plan (reward 1.0), whose bound is 13.8% lower;
  `reward_L`, 0.578, is the fairer reading there. On 2026-07-09 the first two-desk attempt lost the crew desk's
  connection mid-day; the registered retry (0.464) replaces it, and the dataset keeps both.

These scores are not comparable with each other or with PortSimEnv's: each env plays its own days with its own harness.
[results/board.md](results/board.md) has every date, `reward_L`, the spend and how the runs were made.

## Known limits

[FIDELITY.md](https://github.com/earakely-scale/agentenv-airops-plugin/blob/v0.3.3/FIDELITY.md) in the plugin says
what is real data, what is modelled and how each layer was checked against the record, with every miss. In short:

- **The crews are generated.** JetBlue's pairings, rosters and reserve lists are not public: a cost-optimizing builder
  makes trips from the published schedule, and reserves sit at the middle of a sourced industry band. Union contract
  rules and multi-day limits are not modelled (crews start the day rested).
- **The crew filter leans the eval dates toward moderate storms.** The kept dates have fewer JetBlue cancellations
  than the dropped ones (median 20.5 against 28) and less aircraft-only leverage (6.1% against 10.4%), and the five
  dates of most aircraft-only leverage in the world's first build were all dropped. The room the crew desk leaves is
  similar (32.2% against 33.5% of the best plan).
- **The best plan is not a proven optimum, and the bound is loose.** The bound closes on no eval date (gaps of 2.4% to
  13.9%, median 5.1%), and, as FIDELITY.md puts it, "most of the gap is the bound's weakness rather than a measured
  distance to the optimum". So `reward_L` understates, and 1.0 against the best plan is not 1.0 against the
  optimum.
- **Missed checks, shipped as reported:** the calm-day check passed on 85.6% of dates against a 90% bar; on one winter
  control doing nothing ran 8.9% over the optimum because of one flight's real 733-minute delay; the standard crew
  desk missed the controls' 3% margin (+4.0% and +17.4%).
- **The FAA's slots are modelled** in v2: shared out by schedule over every carrier's arrivals plus an unseen stream,
  whose one scale per hub was fitted on the training dates of the world's first, aircraft-only build. The eval dates
  were drawn again over the dates the crew filter kept, and 14 of the 24 had been among those fit dates: in
  FIDELITY.md's words, they "were [training dates of the first build], where the unseen-traffic scale was fitted; the
  FAA-delay check on this eval set is not fully out of sample". v1 uses hourly caps instead: JetBlue's
  ration-by-schedule share of what all carriers actually moved, which a different plan would not have changed.
- **Left out:** passengers' connections and rebooking (a cancelled passenger costs a flat 523 passenger-minutes),
  maintenance, dispatch, gates, and other airlines as actors. Delays are exogenous: holding or cancelling a flight
  never changes another airline's.
- **v1 is aircraft only**, at one hub, with interchangeable aircraft of a type and no crews.

## Start here

| You want to… | Open |
|---|---|
| Play a day on AgentEnv | [envs/airops/agentenv](envs/airops/agentenv/) |
| Read how the watches, the sim, the crews, the references and the grade work | [DESIGN.md](DESIGN.md) |
| See every run's score | [results/board.md](results/board.md) |
| Watch any run in 3D | [earakely-scale/AirOpsEnv-AgentEnv](https://huggingface.co/spaces/earakely-scale/AirOpsEnv-AgentEnv) |
| Get the days and every recorded run | [datasets/earakely-scale/AirOpsEnv-AgentEnv](https://huggingface.co/datasets/earakely-scale/AirOpsEnv-AgentEnv) |
| Watch the films | the plugin's [v0.2.0](https://github.com/earakely-scale/agentenv-airops-plugin/releases/tag/v0.2.0) (v1) and [v0.3.0](https://github.com/earakely-scale/agentenv-airops-plugin/releases/tag/v0.3.0) (v2) releases |
| Run pilots, film runs or publish to the Hub | [agentenv-airops plugin](https://github.com/earakely-scale/agentenv-airops-plugin) |

## Run it locally

You need [uv](https://docs.astral.sh/uv/) and Python 3.12, as CI and the images use, and Docker for the runs through
AgentEnv: `rollout.py` stages a bundle, agent-env builds the env and agent images on first use, starts a registry on
`127.0.0.1:5000` and runs the day on local Docker. The days the envs play are fetched, not committed (11 MB): one
command rebuilds them byte for byte from the dataset's tables at v0.3.3 and checks every file against the committed
manifests, which pin them.

```bash
cd envs/airops
uv run -p 3.12 --with pyarrow python tools/fetch_packs.py           # tasks/b6-v1/tasks.jsonl and tasks/b6-v2/dates/
cd agentenv
uv run -p 3.12 --frozen --extra dev pytest ../core/tests -q         # the day, the world, the crews, the references
uv run -p 3.12 --frozen --extra dev pytest tests -q                 # the envs, the task steps, the recorded v1 run
uv run -p 3.12 --frozen --extra dev pytest tests -q -m slow         # the minute-long replays: followers, a v2 day
uv run -p 3.12 python rollout.py --smoke                            # no model: v1's wiring-noplay, the naive day
uv run -p 3.12 python rollout.py --env occ --script                 # no model: v2's scripted desks, both modes
uv run -p 3.12 python rollout.py --model anthropic/claude-haiku-5-5 # through LITELLM_* or .agentenv/config.toml
HF_TOKEN=hf_... uv run -p 3.12 python rollout.py --model zai-org/GLM-5.3-Flash:baseten   # v1 on the HF router
uv run -p 3.12 python rollout.py --env occ --model openai/gpt-6.1-sol --mode joint --episode-cap-usd 5
```

The tests run on committed fixtures (four v1 days, three v2 dates, a recorded run of each and the task files the
published runs played), so they need neither the fetched packs, the network nor Docker. The board ran v2 at $5 a desk
for two desks and $8 for one agent (`--episode-cap-usd`, 2 by default). `tools/replay_runs.py` replays every published
run through this code; `tools/build_pack.py`, `references.py`, `world_check.py`, `crew_check.py`,
`build_occ_pack.py` and `scripted_desks.py` rebuild the packs and their references from the public records
([envs/airops/agentenv/README.md](envs/airops/agentenv/README.md) has the commands and their costs).

## Layout

```
07-simulation-environments/airopsenv/
├── README.md, DESIGN.md      this page; how the watches, the sim, the crews, the references and the grade work
├── assets/banner.jpg         the banner: the control-center room in the Space's replay of 13 July 2024
├── data/                     a SOURCE.md per public source: BTS, FAA advisories, the FAA registry, FAA airport data,
│                             the IEM, airportsdata, and the crew rules and costs
├── envs/airops/
│   ├── core/                 airops_core: v1's day (core/, refs/, day.py, schedule.py), v2's world (programs/), the
│   │                         ops control center's day (occ/), and the pack builder (build/)
│   ├── agentenv/             airops_agentenv: the two envs' servers, task steps, verifiers, the airops-llm and
│   │                         airops-desk agents, rollout.py, Dockerfiles
│   ├── tasks/                b6-v1/ and b6-v2/: the packs' manifests (committed) and their days (fetched);
│   │                         b6-v2-split.json, the v2 split
│   └── tools/                fetch_packs.py, replay_runs.py, and the pack and reference builders
└── results/                  board.md; eval-v1.json and eval-v2.json, each env's eval card
```

## Data and licence

The flights, tails and delays (BTS on-time data), passengers (T-100), the fare in v1's cancellation cost (DB1B),
aircraft (Form 41 Schedule B-43 and the FAA registry), the FAA Command Center's advisories and the FAA's airport data
are U.S. Government works in the public domain; so are the METARs, TAFs and forecast discussions, NWS products archived
by the Iowa Environmental Mesonet, which asks for attribution, given here. Airport names, coordinates and time zones
come from airportsdata (MIT licence). v2's crew rules (14 CFR parts 117 and 121) and costs (the FAA's economic values,
BLS wages and CPI-U, GSA and DoD per diem rates) are U.S. Government works; its crew bases, fleet configurations and
deadhead fare are facts cited from JetBlue's 10-K filings and press releases, ALPA and Airline Pilot Central, each with
its URL in [crew_params.json](envs/airops/core/airops_core/programs/crew_params.json). Each source's `data/*/SOURCE.md`
says what it is, where it comes from, how it is pinned and its terms. The packs, the test fixtures and the recorded runs
and transcripts derived from them carry the same terms: public domain, with the IEM's attribution and the airportsdata
MIT notice. The fixtures are four v1 days and three v2 dates cut from the packs, the plugin's fixture v1 days, and
slices of the raw inputs: BTS on-time, T-100 and B-43 rows, ATCSCC advisory pages, the FAA registry, IEM METARs, TAFs
and forecast discussions (`core/tests/pack/fixtures/`), and NASR airport rows and BTS taxi-out times
(`core/tests/programs/fixtures/`). The Space's 3D airports are © OpenStreetMap contributors (ODbL 1.0) over USGS The
National Map imagery (public domain); they are the plugin's and the dataset's and are not in this repository. The
plugin's dataset is published under the terms on its card and isn't part of this repository. Code is Apache-2.0 like the
rest of the repository. Neither JetBlue, the FAA, the BTS, the NWS, the USGS, the IEM nor any union is affiliated with
this project or endorses it.

Contributed from the agentenv-airops plugin (https://github.com/earakely-scale/agentenv-airops-plugin) by Edgar
Arakelyan (Scale AI), built on AgentEnv (https://github.com/scaleapi/agentenv-framework).
