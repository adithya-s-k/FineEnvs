# RL Explorer: design

A Space for exploring the RL environments on the Hugging Face Hub, whatever their framework:

- **Harbor** task datasets (folders with `task.toml`), indexed and run on HF Sandboxes;
- **OpenEnv** Spaces, live: woken or restarted, their own app, a playground with rewards, their Task API, and an
  MCP server for coding agents; OpenEnv × Harbor Spaces with what they can run;
- **custom environments**: datasets whose tasks are rows (Verifiers, NeMo Gym, verl, MiMo's raw release, Harbor
  tasks packed into parquet, traces, any table of prompts), each read by a *processor* (CUSTOM_ENVS.md).

Browse an environment's tasks, read exactly what an agent gets and how it is graded, run an agent on a task in a
sandbox, watch it, and compare rollouts. The MiMo RL Environment Explorer did this for one release; MiMo is one
environment among the others here (its raw rows through the `mimo` processor, linked to its Harbor twins).

## What a Harbor dataset is

A task is a folder with `task.toml` (settings, metadata), `instruction.md` (the prompt), `environment/`
(a `Dockerfile`, or a prebuilt `docker_image` in `task.toml`, sometimes a compose file), `tests/` (the
verifier: `test.sh` writes `/logs/verifier/reward.txt`, one number, or `reward.json`, named scores) and
often `solution/`. A survey of the 30 most-downloaded `harbor` datasets (2026-10-03) found:

| Layout | Examples | Share |
|---|---|---|
| `tasks/<task>/` | terminal-bench-2.1/3.0, MiMo, data-agent, repo2rlenv | ~60% |
| `tasks/<group>/.../<task>/` | terminal-bench-science | few |
| `<task>/` at the repository root | terminal-bench-2.0, harbor-mix, WildClawBench, dabstep | ~25% |
| `<benchmark>/tasks/<task>/` | skilltrainbench-public, Lego-RL | few |
| packed (`tasks.parquet`, `.tar`, `.zip`) | TaskTrove, TerminalWorld, FACET | ~10%: read as rows (below) |

So a task is found by where its `task.toml` is, not by a fixed path. Most tasks ship only a `Dockerfile`;
a prebuilt `docker_image` is the minority (terminal-bench-2.1, LHTB, MiMo, apex). Metadata keys differ per
dataset, and some are answers (`gold_answer`, `expected_output`, `solution`): those are never shown.

## Architecture

Two Spaces, one image, one private bucket (`FineEnvs/rl-explorer-data`, mounted at `/data` by both):

```
HF RL Explorer (public Space, app.main)              Admin (private Space, app.admin_app, FineEnvs only)
  catalog: Harbor datasets + OpenEnv Spaces            overview, rollouts (with who ran them), environments
  indexes and packs (/data/indexes, /data/packs)       (pin, hide, feature, index), collections, indexes,
  runner: OpenEnv's run_rollout + capture proxy        settings (rollout switch, limits, agents, banner), audit
  (/capture), HF Sandbox on the visitor's account               │
  store: /data/runs                                             │ writes /data/admin/settings.json (+ audit.jsonl)
        ▲  re-reads settings.json (mtime) ◄─────────────────────┘ reads /data/runs, builds indexes
```

The admin shares the bucket, not the process: pins, hidden environments, collections, rollout switches,
rollouts taken off Community and requests to stop a running rollout all go through the settings file, which the
explorer re-reads within seconds. The admin Space is private (only org members reach it) and every admin route
checks membership again (public member list, or the memberships the Hub reported at sign-in).

- **Catalog.** Datasets tagged `harbor` and Spaces tagged `openenv` or `rl-environment` (Docker, not articles), plus
  the collections admins curate (a dataset id, or `space:<id>`), and the signed-in visitor's own datasets, private
  ones included. Private and gated datasets are read with the visitor's token; every request re-checks access.
- **Index and pack.** On first open (or ahead of time with `python -m app.precache`), one listing finds every
  `task.toml`; the files a task page shows are fetched in parallel (one GET each, 48 at a time). A repository too big
  to list in 25 s (tasks that each ship a whole app, like microsoft/ProgramDistill's 4,375) is *walked* instead: a
  level is listed, the Hub's batched `paths-info` says which of its folders hold a `task.toml` (500 a call), and only
  the files the index reads are looked up; the task list is published as soon as it's known (the page opens, and fills
  in as files arrive), and a task's other folders are listed when someone opens them in the file viewer. The index is one row per task (title, facets,
  grading, environment, network policy, whether it can run here); the pack holds every task's text, de-duplicated,
  and its file list, so task pages never wait on the Hub. The first open shows a step timeline with live counts.
- **Answers stay out.** Answer-like metadata keys, `solution/`, data files beside the grader (rubrics, expected
  outputs), and answers written into grader scripts (answer-named heredocs and literals) are withheld or masked.
- **Rollouts.** `openenv.harbor.rollout.run_rollout`, the same code as OpenEnv's Harbor UI. The agent (OpenCode,
  Terminus 2, mini-SWE-agent, Pi) runs in an HF Sandbox from the task's image, or, for a Dockerfile-only task, from
  the Dockerfile's base image with its RUN/COPY/ENV/WORKDIR replayed first (app/dockerfile.py; about 99% of the
  indexed Dockerfiles qualify). `Sandbox.create` is wrapped so the sandbox runs on the visitor's account. Tasks
  that ask for no internet are refused with the reason: HF Sandboxes can't enforce it. Models: Inference Providers
  through the router (`:fastest`) with the visitor's token, or an endpoint they bring (re-checked before use).
- **OpenEnv Spaces, live** (`app/spaces_live.py`). What a server offers comes from itself: its `/openapi.json`
  (routes: reset/step, the Task API, custom ones), `/metadata`, `/schema`, `/list_environments` and `tools/list` on
  `/mcp`; its web page from the README's `base_path`, then `/web`, then `/`. A sleeping Space is woken the way its Hub
  page lets any visitor (`POST /spaces/<id>/start`, at most once a minute); a broken one is restarted with the
  visitor's own token, which works only if they may write to it. The **playground** holds one WebSocket to the
  server's `/ws` per session (reset, step, state, and MCP calls as `{"type": "mcp"}` on the same episode: the HTTP
  routes are stateless), shows each observation (images, audio, chats) and reward, and plays a Task API task by its
  split and index. **OpenEnv × Harbor** servers show their engine, sandboxes, validated agents and served datasets,
  mapped back to the Hub (`/data/org__name` is `org/name`) so their tasks run on the visitor's account here.
  **Other servers** (NeMo Gym, ORS, any FastAPI app) are used through their own routes, read from their OpenAPI:
  a form per route from its request schema, called with the session's own cookie jar (NeMo Gym keeps an episode in a
  cookie; it stays on the explorer). Only plain paths the server published, GET or POST, path parameters checked;
  GET answers drop answer-like fields. OpenEnv servers' own extra routes aren't offered (geoguesser's return the
  answer's coordinates): they're used through reset/step and tools.
- **MCP for coding agents** (`app/mcp_bridge.py`, `/mcp/<org>/<name>`). OpenEnv's `/mcp` answers `tools/list` and
  `tools/call` but not MCP's handshake, so Claude Code, Codex, Cursor, VS Code, Gemini CLI and the rest can't use it
  directly. The bridge speaks Streamable HTTP: the Space's own tools in a session of their own, plus `reset`/`step`/
  `state` for servers without tools (step's input is the action schema), plus `list_splits`/`list_tasks`/`get_task`
  for the Task API, plus one tool per route for servers that aren't OpenEnv (NeMo Gym's `seed_session`, `guess`,
  `verify`); images come back as image content. Sessions are capped per address, rate-limited, and end idle.
- **One contract for every format** (`app/envs/contract.py`, CUSTOM_ENVS.md). An environment is read by an adapter
  that answers five questions: its tasks (a summary, cards, each task as sections of blocks, its files), what they run
  in, how they can run (run options, each by a runner, with its own inputs), which other tasks are the same task
  (aliases: their rollouts are shared), and nothing more for MCP. So one environment page (`/d/<org>/<name>`, `env.js`) and one
  task page (`/t/<org>/<name>/<ref>`, `task.js`) serve Harbor folders, the MiMo release and any rows dataset; a raw
  MiMo row and its Harbor conversion link to each other and list each other's rollouts. Adapters: `harbor.py` (the
  index above), `mimo.py`, `rows.py` (with processors: rows from the Hub's dataset viewer, or straight from JSON
  Lines / JSON / parquet / CSV; a processor turns a row into a task; answers left out by name at any depth). The
  registry picks the surest adapter that confirms; linkers and aliasers relate tasks across adapters.
- **The MiMo release** (`app/mimo`, `app/envs/mimo.py`). The MiMo RL Environment Explorer's backend, kept close to its
  source: all 7,780 tasks with the brief, the agent's exact prompt, systems and their databases, workspace previews,
  Code tasks' repositories (snapshots in `/data/repo-snapshots`), the grader with its judge prompts, bands and music
  features, and the reward design across the release. Two runners: its Harbor conversion (the default) or its own
  harness (OpenCode in the task's image, Xiaomi's graders, a judge model the visitor picks), whose sandbox reaches
  models through `/api/llm/<capability>` and never holds a token. Its run page (events, live), compare view and
  community (leaderboard by domain, the tasks nobody has tried) work for every environment.
- **MCP for environments** (`/mcp/d/<org>/<name>`): describe, list and search tasks, read one in full as text, read
  its files, a random task; answers withheld as on the page. Any adapter, no code.
- **Visibility.** Public rollouts appear in Community without who ran them, private ones only to their owner; only
  finished, graded rollouts are shared; a private dataset's rollouts stay private; admins can take one off.

## Look

The MiMo explorer's design (its `web/app.css`, unchanged; `web/rlx.css`, `fv.css` and `sp.css` add the rest), with
the Hugging Face logo and the name HF RL Explorer. Explore opens with FineEnvs' banner (one link, closable for a
week), the numbers, two rows of trending environments with a pager, then one search, filters (kind: Harbor, OpenEnv,
Verifiers, NeMo Gym, other) and cards. Elsewhere the header's search lists trending environments before you type.
Files open in an editor-like viewer (folders, highlighting, line links, full view) that fits whatever column it's in.

## Security

`uv run pytest tests -q` (no network). `test_security.py`: sessions can't be forged or outlive their expiry;
admin routes are org-only and absent from the public app; cross-site writes are refused; private rollouts and
datasets stay private, per token; public rollouts carry no user or endpoint; answers are withheld or masked;
task files stay inside their task; visitor endpoints must be public https; tokens never reach the store; headers.
`test_live.py`: only a Space's own hf.space host is ever called (redirects elsewhere aren't followed, `base_path`
can't point off it), private Spaces are refused, waking is limited, restarting needs write access, playground
sessions belong to who started them and are capped, the Task API browser drops answers, and the MCP bridge speaks
the protocol (handshake, notifications, errors as tool results, sessions, rate limit, no cross-origin pages).
`test_envs.py`: processors claim their formats, answers stay out (nested, packed solutions, MiMo's patches), archives
can't escape or blow up, filters can't inject, the direct reader pages right, and the walker never lists a task's
own folders. `test_contract.py`: a toy adapter registered from outside works end to end (pages, private keys stripped,
runs refused unless the chosen option can run, its fields checked, rollouts merged across aliases, MCP).
`test_mimo.py`: the MiMo adapter's tiles, facets and map, row refs, the default runner, judges from the pool only, no
token in a MiMo run record, links back from the Harbor twins, scrubbed public events, sandboxed raw files.
`tests/ui-audit.mjs` clicks through every page at four widths in both themes (needs network).

Never sent to a Space: the visitor's token. A Space's run_rollout is offered only when it has its own engine.

## Next

A Space adapter (an OpenEnv server's Task API tasks as task pages, its playground as a run option); a folder-of-tasks
processor for repositories without `task.toml`; model token costs for Harbor rollouts; MiMo's repository snapshots
synced into this bucket.

## Data layer

The catalog is built offline and read as an immutable SQLite snapshot, so no app process fetches the Hub's listing
or builds indexes to answer a page, and the home page asks the server for one page of cards instead of downloading
every environment (about 4 MB of JSON) to filter it in the browser.

```
HF scheduled Job, hourly (scripts/schedule_indexer.py)          Explorer / admin Spaces (app/snapshot.py)
  python -m app.indexer --store /data                              every 60 s: read snapshots/LATEST.v1.json
   1 listing   catalog._listing_fetch (every Space, ~2.5 min)       new? copy the .db to local disk, check sha256,
   2 indexes   rebuild what changed (catalog's builders), budget        open read-only + immutable, check schema,
   3 build     SQLite on local disk: envs, tasks, FTS, indexes          counts, quick_check; swap (queries in flight
   4 publish   snapshots/catalog-<UTC>.db, then the pointer             finish on the old one)
   5 prune     keep 5 (and whatever a pointer names)                  none / corrupt: build the same DB from the
        │                                                              live catalog (local dev, first boot)
        └──── bucket FineEnvs/rl-explorer-data, mounted at /data ──────────▲
```

- **Indexer** (`app/indexer.py`). One run fetches the listing and refuses it when it is much smaller than the last
  snapshot's (overall, datasets or Spaces under 80%: catalog skips a Hub tag listing that fails, and a half listing
  must not empty the site; `--force` overrides). It writes `listing.json.gz` for the app's own code paths, then
  rebuilds dataset indexes incrementally: a dataset whose index is current (same `INDEX_VERSION`, Hub
  `lastModified` older than the index's build) costs no Hub call; one touched around or after its build has its
  revision checked; changed revisions, older formats, indexes left partial and never-indexed Harbor datasets are
  rebuilt in that order (featured first, then by trending) with catalog's own `build_index`, within a time and count
  budget. A build that fails isn't retried at the same revision for a day (`indexer-state.json`). Then it builds the
  snapshot on local disk (`journal_mode=OFF`, `synchronous=OFF`, rows streamed one dataset's index at a time, FTS
  rebuilt, indexes, `ANALYZE`, `VACUUM`, `quick_check`), copies it to `snapshots/catalog-<UTC ts>.db`, checks it
  landed whole (size, and sha256 when the store can be re-read), and only then writes the pointer
  `snapshots/LATEST.v<SCHEMA>.json` (`{"db", "sha256", "size", "built_at", "counts", "schema"}`) atomically. A run
  that fails anywhere before that publishes nothing. Old snapshots are pruned (the newest 5, and any a pointer of any
  schema names). Reruns are safe: current indexes are skipped and every run publishes a complete snapshot.
- **Snapshot** (`app/snapshot.py`). `envs` (one row per public environment: id, key, kind, framework, heading,
  brief, counters, dates, stage, MCP, OpenEnv version, badges and tags as JSON, indexed task count and revision,
  the facet values that don't depend on admin settings, and a lowercased search blob), `envs_fts` (FTS5, trigram:
  substring search, as the page always did), `tasks` (dataset, ref, title, brief, category, difficulty, group, how it
  runs, how it's graded, tags, a few extras), `tasks_fts` (FTS5, unicode61, token prefixes), `meta`. Only what the
  explorer already shows publicly: no private or gated dataset, no index a visitor built with their own token, no
  metadata values, no file texts. 7,225 environments and 51,498 tasks (the Harbor indexes and MiMo's release) made a
  43 MB file, built in about a second; a run with the full listing took under five minutes.
- **Reading it.** Each process checks the pointer on startup and every 60 s; SQLite never opens a file on the
  bucket mount (the database is copied to `RLX_CACHE_DIR/snapshots` first), connections are read-only
  (`mode=ro&immutable=1`, a small pool per snapshot), and a swap retires the old snapshot when its last query is done.
  A pointer that names a missing, corrupt, truncated or mismatched database (sha256, schema, row counts, quick_check)
  is rejected and logged (`snapshot.rejected`), the current one kept, and the same file isn't fetched again for ten
  minutes. With no snapshot at all, the same database is built in the process from `catalog.environments()` and the
  indexes on disk, so local development and a first boot work without the Job. Bumping `SCHEMA` publishes under a
  new pointer name: an app of the old version keeps reading its own.
- **What admins change is applied per query.** Hidden environments, pins and collections come from the admin
  settings on every request (an admin's change shows within seconds, between indexer runs), as do public rollout
  counts.
- **Search API** (`app/search_api.py`, a router). `GET /api/search?q=&kind=&collection=&f=&sort=&page=&size=`
  returns a page of cards, the total and every facet's counts, with the page's exact semantics: the Kind taxonomy
  (OpenEnv Spaces including FineEnvs' curated servers, other Spaces including ORS, Harbor whenever an index found task
  folders, Verifiers, NeMo Gym, OpenEnv datasets, verl, other rows), pinned first in trending, FineEnvs' curated
  servers first among OpenEnv Spaces, each facet counted over what the other filters leave, ties in listing order.
  `trending=24` adds the trending row and header numbers to the same response (one request for the first paint);
  `mine=1` merges the signed-in visitor's own datasets, private ones included (a private response). Search words are
  data: each is a quoted FTS string (and checked again as a plain substring), so operators, `NEAR`, `*`, column
  filters and stray quotes can't change the query. `GET /api/search/tasks?q=&env=` searches every indexed task
  (each word a quoted prefix, best match first). Responses carry `Cache-Control` (public 30 s, private for `mine`),
  a weak ETag (304 on a repeat) and `Server-Timing`. A parity test compares every answer, order and facet with a
  plain-Python port of the page's old filtering.
- **Home page** (`web/js/home.js`): the first paint is one request of 12 KB on the wire (55 KB of JSON; it was
  458 KB on the wire, 4 MB of JSON), and the first card shows in 283 ms locally instead of 467 ms (7.1 s to 5.2 s on
  DevTools' Fast 3G, where the data request itself went from 3.4 s to 0.65 s); search (debounced), filters, sort and paging ask the server, an older request is cancelled when a newer
  one starts, a slow answer shows placeholder cards of the real size, and the URL keeps the page's state as before.
- **Scheduling** (`scripts/schedule_indexer.py`, prints the Job; `--apply` creates it): the explorer Space's own image,
  `@hourly` with concurrency off, `cpu-upgrade`, a 55-minute timeout, the bucket mounted read-write at `/data`
  (`STORAGE_DIR=/data`; written files upload on close, and a rename is one bucket batch, so temp-then-rename and the
  pointer-last order hold for readers on other mounts, which see changes within ~10–30 s), `HF_TOKEN` as a Job secret
  (higher rate limits for public downloads only). `--store hf://buckets/<org>/<name>` publishes through the bucket
  API instead, with the same database-then-pointer order.
- **On the Spaces**, once the Job runs: set `RLX_LISTING_FROM_STORE=1` on both Spaces, so they read the Job's
  `listing.json.gz` (a stat every 30 s, a read when it changes) instead of each fetching the Hub's whole listing every
  15 minutes; a listing older than 3 h (the Job stopped) is fetched again as before. `/readyz` reports the snapshot in
  use; `GET /api/search/status` says when it last looked. `GET /api/search/rank?key=` gives one environment's trending
  rank among its kind (the Space page's "#599 of 6,818"), so no page loads the whole listing any more.

## Production checklist (before the first deploy)

1. **Indexes and packs at INDEX_VERSION 13**: rebuild (`uv run python -m app.precache --featured --top 40 --use-token`
   or one indexer run) and sync to `FineEnvs/rl-explorer-data`; old-version files are rebuilt on demand, slowly.
2. **MiMo repo snapshots** (2,651) synced from `FineEnvs/mimo-explorer-runs` into the explorer's bucket.
3. **Spaces**: `FineEnvs/RL-Explorer` (public; the sandbox reaches its `/capture`) and the private admin Space, both
   with the bucket mounted at `/data`, `SESSION_SECRET` and the OAuth app's secrets set.
4. **Indexer Job**: `uv run python scripts/schedule_indexer.py --apply` once the explorer Space has built; then
   `RLX_LISTING_FROM_STORE=1` on both Spaces.
5. **Smoke tests on the Space**: `/readyz`; `scripts/check_judge_relay.py`'s path through the Space's `/capture`
   (a model-graded rollout, e.g. a MiMo General twin with Inkling as judge: reward recorded, `judge_calls > 0`);
   `node tests/ui-audit.mjs https://<space>`.

## Admin source boundary

This public tooling directory contains only the Explorer. The dashboard application,
its assets, membership checks, tests and deployment tooling are maintained separately
in the private `FineEnvs/RL-Explorer-admin` Space. The Explorer reads shared settings
and moderation requests from the data bucket without importing the admin app.

## Search indexing and social previews

The Space card uses the committed `web/social/rl-explorer.png` thumbnail, so a
sharing crawler does not have to start the application. HTML pages also publish
Open Graph and Twitter metadata with 1200×630 previews. Page-specific images fit
long names into the card; public task links retain their own canonical address
both in the initial HTML and after browser navigation.

`/sitemap.xml` links to shards of up to 10,000 public task URLs. Dataset tasks come
from local indexes or the immutable catalog snapshot, with MiMo entries deduplicated.
OpenEnv task coordinates come from advertised split counts on checked public
Spaces. Their sitemaps expand ranges one shard at a time, without downloading tasks
or creating millions of URLs in memory. These counts describe published entries,
not individually tested episodes or Google-indexed pages.

Space-check inventory refreshes outside request threads and never holds the reader
lock during bucket I/O. Writes update the cached record immediately; concurrent
refreshes preserve newer evidence. A cold process temporarily has no verified
Spaces, and cached evidence still expires according to its original check time.

Task HTML includes public task text or structured input facts. Public row datasets
and OpenEnv Task API records can be read anonymously on demand, with four concurrent
reads and bounded caches. These reads do not build an index, wake a Space, run an
episode, or forward a visitor's token. The existing answer-withholding rules apply.
Invalid tasks return 404; temporary upstream failures return 503 with Retry-After.
Private, gated and hidden content is excluded from sitemap generation.

Public read APIs may be fetched to render the interactive pages, but API responses
carry X-Robots-Tag: noindex. Rollouts, account pages and comparisons remain excluded.
Submit `https://fineenvs-rl-explorer.hf.space/sitemap.xml` for the matching URL-prefix
property in Google Search Console to monitor discovery and indexing. Robots.txt
also advertises it. Google chooses which pages to index; no application setting
can guarantee indexing or rankings. See Google's sitemap guidance:
https://developers.google.com/search/docs/crawling-indexing/sitemaps/build-sitemap
