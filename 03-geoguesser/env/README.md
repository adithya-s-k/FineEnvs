# The environment

A model is dropped at a random street corner on Earth. It can turn its head, zoom in to read a sign, walk a few metres, and drop pins on a world map to check what is actually at a coordinate. Then it commits to a latitude and longitude and is scored on how far off it was.

Built on [OpenEnv](https://github.com/openenv/OpenEnv). It speaks HTTP for trainers and MCP for agents.

## Two ways to run it

### Online: the hosted Space

Nothing to install.

**[huggingface.co/spaces/HuggingEnvs/geoguesser-env](https://huggingface.co/spaces/HuggingEnvs/geoguesser-env)**

```bash
python ../eval/geoeval.py run --env space --provider anthropic \
    --model claude-sonnet-5 --split eval --limit 5
```

Good for playing a round, scoring an API model, or checking something quickly.

What it cannot do:

- Four concurrent sessions. `MAX_CONCURRENT_ENVS` defaults to 4. Training needs one per rank plus generation, so 8. You must deploy your own Space and raise it, and duplicating a Space copies files but not variables.
- Latency per turn. Every `look` is a network round trip. A 12-turn episode over the wire is far slower than in-process.
- No config changes. Step budget, reward shape and view size are baked into the Space's variables. Changing them means redeploying.
- It sleeps. A cold Space takes a minute to wake.
- No street detail unless the bucket behind it carries the detail vectors.

### Locally

Full control, and fast enough to debug a rollout for pennies.

```bash
cd env
uv sync

# 86,366 panoramas, 22 GB. Public bucket, no token needed to read.
hf sync hf://buckets/HuggingEnvs/geoguesser-panos/panos ./data/panos

# Confirm the split is complete before trusting any number it produces
python ../dataset/verify_offline.py tasks/eval_pano_v3.jsonl --check
#   200 tasks · 4673 frames · 0 tasks incomplete · 0 frames absent

bash serve.sh          # play at http://localhost:8000/web/
```

Then point anything at it with `--env local`, or `--env local:8161` on another port.

This costs about 25 GB of disk and one slow first sync.

Local and hosted are verified equivalent. The same task returns identical image checksums, reward and distance from either.

## About the imagery bucket

The 22 GB of panoramas live in a Hugging Face Storage Bucket, not in git, and that trade has real edges worth knowing:

- A bucket is mutable object storage, not version control. No history, no diff, no review. If someone re-harvests, the old bytes are simply gone, and a number you measured last week may not be reproducible.
- `hf sync` is one-way and propagates deletions. It is not a backup.
- First sync is slow and bandwidth-bound. Budget for it.
- On a Space it mounts read-only at `/data`, and the environment is configured with `GEOGUESSER_ALLOW_FETCH=0` there. A cache miss becomes an error rather than a silent network call. That is deliberate: the store is complete, so a miss means the mount is wrong.
- Space disk is ephemeral and capped well below 22 GB. That is the whole reason the bucket exists rather than baking imagery into the image.

The task indexes are metadata only, holding coordinates, headings and capture dates, so walking down a road needs no network. Image bytes are separate because Mapillary serves them through expiring signed URLs that cannot be stored.

Imagery is Mapillary, CC BY-SA 4.0. Every task carries its contributor, and the environment displays the credit.

## What the agent can do

Eleven tools. Five matter:

| tool | does |
|---|---|
| `look` | turn the camera to a heading. This is how it reads signage, road markings, vegetation, architecture |
| `zoom` | narrow the field of view to read something distant, such as a shop name or a road number |
| `move` | walk forward or back along the road |
| `pin` | drop a candidate coordinate on a map. Returns **what is actually there**: country, nearest city, distance from the last pin |
| `guess` | commit. Ends the episode, scored on distance |

The rest, `pan`, `view_map`, `list_pins`, `clear_pins`, `measure` and `reverse_geocode`, are conveniences over those.

`pin` is the one worth understanding. It says what is at a coordinate but never whether it is close to the truth. Any signal about the target would make binary search optimal, and the benchmark would measure bisection instead of geography.

Every action costs a small amount of reward, so the policy has to decide when it has seen enough. That one term drove most of what we observed. See [`../LEARNINGS.md`](../LEARNINGS.md).

## Configuration

Everything is an environment variable, so a Space and a local server are configured the same way.

| variable | default | does |
|---|---|---|
| `PORT` | `8000` | where to listen |
| `MAX_CONCURRENT_ENVS` | `4` | concurrent sessions. raise to 8 or more for training |
| `GEOGUESSER_MAX_STEPS` | `24` | turn budget. Match it to your trainer's limit |
| `GEOGUESSER_DEFAULT_SPLIT` | first available | `train` or `eval`. Unset picks `train` if present, else whatever resolved |
| `GEOGUESSER_VIEW_SIZE` | `640` | rendered frame, px |
| `GEOGUESSER_ALLOW_FETCH` | `1` | `0` makes a cache miss an error instead of a fetch |
| `GEOGUESSER_INDEX` | eval split | fallback task index. `tasks/pano_v1.jsonl` is a 100-task demo index that predates the contamination rule and holds a task 603 m from an eval start, so never train on it |
| `GEOGUESSER_STREET_DETAIL` | off | roads and place names on the minimap. Fetched from Overpass per pin, so it puts a network call inside a rollout |
| `GEOGUESSER_PLAY_ROUTES` | on | the browser game's routes. They serve raw panoramas and a task's coordinates over plain HTTP, so set `0` on a Space a trainer points at |
| `GEOGUESSER_HIDE_IDENTITY` | off | strips `attribution`, `sequence_id` and `task_id` from observation metadata. The training harness never forwards metadata, so this matters only for a harness that does |
| `GEOGUESSER_REWARD_SHAPE` | `geoguessr` | reward curve |
| `GEOGUESSER_COST_MODE` | `subtract` | how the action cost is applied |

Show the model exactly one turn budget. If the environment counts down 24 while your trainer stops at 12, the model paces itself for 24 and never commits. That produced 0 answers in 6 episodes here, every reward exactly 0.0.

## Layout

```
env/
├── client.py      the typed client a trainer talks to
├── models.py      action and observation schemas
├── harness.py     the in-process driver
├── serve.sh       bring up a local server
├── Dockerfile     the Space image, self-contained
├── server/        environment, reward, panorama backend, renderers
├── tasks/         the frozen 200-task eval split
├── data/geo/      1.6 MB of geometry; imagery syncs from the bucket
├── examples/      two rollouts, one needing no API key
└── tests/
```

Design decisions and their rationale are in [`DESIGN.md`](./DESIGN.md).

## Tests

```bash
uv run pytest tests/ -q
```

They use five committed fixture panoramas, so they need no imagery sync.
