# Adding an environment format

Every environment the explorer shows is read by an **adapter** (`app/envs/contract.py`): Harbor task folders
(`harbor.py`), Xiaomi's MiMo release (`mimo.py`), any dataset of rows (`rows.py`, with its row readers). An adapter
answers the same five questions for its format, and the pages, the runner, rollouts and MCP then work for it with
nothing else changed:

| part | what the adapter gives | where it shows |
|---|---|---|
| tasks | `summary`, `tasks` (cards), `task` (one in full, as sections of blocks), `file`, `folder`, `random` | `/d/<org>/<name>`, `/t/<org>/<name>/<ref>` |
| environment | a section of the task: what it runs in | the task page |
| harness | `run_options`: how it can run, by which runner, with which inputs; `materialize` / `run_task` for the runner | the run panel, `POST /api/runs` |
| rollouts | `aliases`: the same task elsewhere (a raw row and its Harbor conversion); their rollouts are this task's | the task page, compare, community |
| mcp | nothing to do: describe, list, search, read and files come from the above; `mcp_tools` / `mcp_call` add more | `/mcp/d/<org>/<name>` |

A task is named by a **ref**, a string the adapter defines and parses: a Harbor task's folder (`tasks/fix-bug`), a
row (`code/train/6`, or `train/6` in the default config), a MiMo task's id (`format-code-task-000720`). An adapter
may accept old refs and answer with the canonical one (`view["ref"]`): the page moves to it.

Most formats don't need a new adapter: a dataset of rows needs a **row reader** (below). Write an adapter when tasks
aren't rows, when they need an index of their own, or when the format has its own harness.

## An adapter

```python
from app.envs import contract as c

class MyAdapter(c.Adapter):
    id, name, framework = "mine", "My format", "Mine"     # id: logs, URLs, run records; framework: the badge
    about = "One sentence: what a task is here, and how it's graded."

    def detect(self, kind, meta):                          # 0 to 1, cheap: the Hub's record (tags, card)
        return 0.9 if "my-format" in (meta.get("tags") or []) else 0.0

    def confirm(self, env):                                # optional: a closer look when unsure (< 1)
        return True

    def summary(self, env, subset=None):
        return {"state": "ready", "total": 120, "inline": True, "search": True, "subsets": [],
                "how": {"framework": self.framework, "name": self.name, "about": self.about},
                "facets": [c.facet("level", "Difficulty", "level")],
                "overview": [c.section("graded", "How it's graded", [c.kv([("Reward", "1 if the tests pass")])], icon="target")]}

    def tasks(self, env, *, subset=None, q="", filters=None, offset=0, everything=False):
        cards = [c.card(t.id, t.title, brief=t.brief, lead=t.group, facets={"level": [t.level]}) for t in load(env)]
        return {"cards": cards, "total": len(cards), "offset": 0, "page": len(cards)}

    def task(self, env, ref):
        t = find(env, ref)                                 # LookupError -> 404
        return {"ref": ref, "title": t.title, "chips": [t.level], "withheld": ["solution"],
                "sections": [c.section("task", "The task", [c.markdown(t.prompt)], icon="message"),
                             c.section("files", "Files", [c.files(t.tree)], icon="folder")],
                "glance": [["Reward", "one score"]], "links": []}

    def run_options(self, env, ref, view):                 # [] and `run_note` when it can't run here
        return [c.run_option("harbor", "Harbor, on an HF Sandbox", ok=True, default=True)]

    def materialize(self, env, ref):                       # the task as a Harbor task folder, for the Harbor runner
        ...

    def aliases(self, env, ref):                           # [(environment key, ref)]: the same task elsewhere
        return []
```

Put an instance in `ADAPTERS` (`app/envs/registry.py`); order doesn't matter, `detect` decides. While an adapter
is still finding out (an index being built) it returns `False` from `settled` and its summary `{"state":
"indexing", "progress": …}`; one that finds the source isn't its after all returns `{"state": "handoff"}` and the
next adapter reads it.

**Inline or paged.** `inline: True` means `tasks(everything=True)` returns every task, and the page searches and
filters as you type (fuzzy, ranked, highlighted), with tiles and a map. Otherwise the page asks `tasks` for a page at
a time, passing `subset`, `q`, `filters` ({facet: [values]}) and `offset`, and draws `facets` with the `values` you
counted.

**An inline list can say more:** `tiles` (`c.tiles(key, items)`: a big button per value of a facet, each with a note,
an icon and a colour), facets with a `scope` (shown only with one of these tiles picked, `"*"` with none), a `map`
(`c.treemap(by_tile={tile: facet})`: with no tile picked, every tile's top values side by side), `order: "shuffle"`
(a fresh order each visit) and `noun` (`["environment", "environments"]`). An overview section can be `collapsed`.

## Blocks

A section is a list of blocks, built with the helpers in `contract.py`. The page draws each kind (`web/js/blocks.js`)
and MCP turns each into text (`app/envs/tools.py`).

| block | for | drawn as |
|---|---|---|
| `kv(rows)` | label/value facts (values: inline Markdown or numbers; `shares(...)` too) | a definition list |
| `shares(rows, of)` | parts of a whole | rows with bars |
| `stats(rows)` | a few figures | a row of figure boxes |
| `markdown(text)` | prose, the task as the agent gets it | rendered Markdown |
| `code(text, path, label)` | a file shown inline | highlighted; "open in Files" when the files hold it |
| `note(text, icon)` | a line of fine print | an icon and a sentence |
| `value(v)` / `messages(v)` | any JSON / a chat | folding rows, images, audio, tables / a conversation |
| `steps(items)` | a multi-step task | a numbered list |
| `links(items)` | other pages (https) | buttons |
| `disclose(label, blocks)` | something long | folded under a line |
| `files(tree, hub=…)` | a task's files | the file viewer; serve contents from `file`, unlisted folders from `folder` |
| `custom(renderer, kind, data, text)` | a view no block covers | `web/js/renderers/<renderer>.js` |

**Custom views.** Prefer blocks: they need no front-end code and agents read them as text. When a format needs a view
of its own (MiMo's systems and their databases, its workspace previews, its grader), write a renderer module:
`web/js/renderers/<name>.js` exporting `render(kind, data, ctx)` (HTML) and optionally `wire(root, ctx)` (clicks).
`ctx` has `spec` and `ref`; fetch anything more through your adapter's `data(env, ref, name, params)`
(`/api/env/<org>/<name>/data?ref&part=<name>&…`) and serve a task's files as they are with `raw(env, ref, path)`
(`/raw?ref&f`, always with a sandboxing CSP). Give every custom block a `text`: it is what MCP and any page that
can't load the renderer show.

## Running

A run option is one way to run a task, by a **runner**:

- `harbor`: OpenEnv's Harbor runner on an HF Sandbox (`app/runner.py`). Implement `materialize` (the task as a Harbor
  folder) and `run_task` (title, sha, image, bytes, runnable). The agent is the visitor's pick (OpenCode, Terminus 2,
  mini-SWE-agent, Pi).
- `mimo`: the MiMo release's own harness (`app/mimo/runner`): OpenCode in the task's image, Xiaomi's graders.

`c.run_option(runner, label, ok=, why=, default=, about=, notes=, warnings=, fields=, harnesses=, endpoint=,
sandbox=, estimate=)`. `fields` are the option's own inputs, each a `c.field(key, label, type, …)`: `text`, `number`
(with `min`/`max`), `bool`, `select` (`options`), or `model` (a judge from `pool`, `"text_judges"` or
`"vision_judges"`), `advanced=True` to fold it under Advanced settings, `required=True` when a run needs it. The
server checks every field against the option the run picked. `endpoint` lets the visitor bring an OpenAI-compatible
endpoint; `estimate` ({tokens_in, tokens_out, minutes, flavor}) prices a typical run on the panel. A new runner is a
branch in `runner.submit` that writes a run record with `dataset`, `path` (the ref), `env`, `adapter` and `runner`,
so its rollouts list with every other; if it records events instead of a trajectory, `run.js` hands its page over
(see `run-mimo.js`).

## The same task in two places

`aliases` says which other (environment, ref) pairs are this very task; their rollouts are listed with it, and the
compare view takes rollouts from all of them. A relation neither adapter knows (a converted dataset and its source)
goes in the registry instead: a **linker** `(env, ref, view) -> [c.link(label, href, note, rel="same")]` puts a link
on the page, an **aliaser** `(env, ref) -> [(key, ref)]` merges the rollouts. `mimo.py`'s `harbor_to_mimo` and
`harbor_aliases` link FineEnvs' Harbor conversions back to the MiMo release.

## Rules

- **Never return an answer.** Leave out reference solutions, gold labels, a rubric's anchors, the code that checks
  them, anything that would let an agent read off the answer; name what you left out in `withheld`. Files that hold
  answers are listed with `withheld: True` and never served. Keys starting with `_` in a task view are yours: the
  registry strips them before anything leaves (use them to hand `run_options` what it needs).
- **Use the visitor's token only for private sources** (`env.read_token`); public data is read anonymously and
  cached for everyone.
- **Be quick, or say you're working.** A slow first read belongs behind an index (with `progress`) or a background
  load; pages poll.

## Row readers

A dataset of rows (Verifiers, NeMo Gym, verl, agent traces, Harbor tasks packed into rows, any table of prompts)
needs no adapter: `RowsAdapter` reads it through the Hub's dataset viewer (pages, filters, statistics for facets,
search), or, when the viewer can't, straight from its JSON Lines / JSON / parquet / CSV files. A **processor**
(`app/envs/processors.py`) says how one row becomes a task; the generic one infers each column's role from its name
and values. Write one when a format deserves more: a nested request to unpack, an archive to open, a grader to
explain, a twin to run.

```python
from app.envs.base import Dataset, Processor, section, withhold, parse_json, text_of, first_line

class MyRows(Processor):
    id, name, framework = "my-rows", "My rows", "My rows"
    about = "One sentence: what a row is and how it's graded."

    def match(self, ds: Dataset) -> float:                   # the generic reader scores 0.1
        return 0.9 if ds.has("my_request", "my_checks") else 0

    def view(self, row, i, ds, roles):
        clean, gone = withhold({k: parse_json(v) for k, v in row.items()})   # answers out, by name, at any depth
        return {"title": first_line(text_of(row["my_request"])) or f"Row {i}", "id": str(row.get("id")),
                "sections": [section("task", "The task", row["my_request"]),
                             section("grading", "How it's graded", clean["my_checks"], kind="value")],
                "withheld": gone, "glance": [["Row", f"{i:,}"]],
                "run": None}   # or {"dataset": "org/harbor-twin", "path": "tasks/x"}: a twin that runs here
```

Add it to `PROCESSORS`. Section kinds: `markdown`, `messages`, `value`, `code` ({path, text}), `files` ({tree},
serve each with `file(row, ds, rel)`) and `blocks` (a list of contract blocks, for a reader that builds its own:
`custom` views, `kv`, `links`, ...); the rows adapter turns them into blocks. A row's ref is `<split>/<i>`, or
`<config>/<split>/<i>` outside the default config.

A processor can say more than a row:

- `overview(ds, roles, stats, config, split)`: sections for the dataset's own page (what a row asks, how it's graded,
  how to run it), from the first rows (`ds.sample`) and the dataset viewer's column statistics (`stats`, `[]` when the
  rows come from files). Don't read more than that: it runs on every dataset page.
- `framework_for(ds)`: the badge, when one reader covers two formats (verl and SkyRL).
- In `view`: `framework_run`, a `c.run_option(...)` with `ok=False` that says in one sentence why the task can't run
  here and points at the page's run section (the run panel shows it; the runner refuses it); `summary`, a line under
  the title (what scores it); `framework`, this row's badge.
- `withhold(value, hide=frozenset({...}))`: answers whose names don't say so ("expected_action", "verifier.patterns"),
  as dotted names, through lists. Put a format's set in `processors.HIDDEN` too: search never looks at them.

**The framework readers.** NeMo Gym, Verifiers and verl / SkyRL rows are read by modules of their own, which show a
row as its framework sees it and say how to run it there (they never run here: the sandbox runner runs Harbor tasks,
and these need their framework's servers, sandboxes or trainers):

| module | knows | how it's kept current |
|---|---|---|
| `nemogym.py` | every agent and resources server NeMo Gym's configs define, what each verifies, which row fields its verifier reads and which hold the answer (`nemo_gym_catalog.json`) | rebuild from a checkout: `uv run python -m app.envs.nemo_gym_build /path/to/Gym` (reviewed overrides in `EXTRA_HIDE` / `NOT_SECRET`) |
| `verl_rows.py` | verl's `default_compute_score` routing (`data_source` → scorer), SkyRL's `env_class` registry, the trainers' commands | `SCORERS`, `SKYRL_ENVS`, `run_blocks` |
| `verifiers_rows.py` | Prime's datasets and the Hub environment that loads each, their API (v0/v1, asked of the Environments Hub), SWE grading material to withhold | `KNOWN`, `DEFAULTS`, `SWE_HIDE`, `DATASET_HIDE` |

They draw with blocks and one renderer, `web/js/renderers/rl.js`: `transcript` (a conversation as the policy gets it:
messages, folded reasoning, tool calls with their outputs paired by call id; a long middle folds), `tools` (each tool
as a signature: parameters, types, required, allowed values) and `fields` (a row's other fields, label over value).
`convo.py` reads Responses API items and chat messages into `turns`, tool definitions into signatures, and makes
titles (`task_title`: an issue's own title, past a template's preamble).

| processor | reads | example |
|---|---|---|
| `harbor-packed` | a Harbor task packed into each row (tar/tar.gz/zip, base64 or bytes); runs | open-thoughts/TaskTrove |
| `mimo` | MiMo-V2.6 RL rows (verl-shaped, with an instance packed in `extra_info`) | copies of the release |
| `nemo-gym` | `responses_create_params` (the conversation, the tools), `agent_ref` (→ its resources server), the verifier's task data; Pivot rows' expert step | nvidia/Nemotron-RL-* |
| `verl` | prompt, data_source (→ verl's scorer), ability, reward_model, extra_info; SkyRL's env_class and reward_spec | sungyub/*-verl, NovaSky-AI/SkyRL-SQL-* |
| `traces` | recorded rollouts: the agent's turns, its result | open-thoughts/AgentTrove |
| `verifiers` | data for a Verifiers environment: v0 question/prompt/answer/info, Prime's SWE tasksets | PrimeIntellect/* |
| `generic` | anything else | FineEnvs/SmolDataEnvs |

## Trying one

```bash
uv run python -c "
from app.envs import registry
s = registry.summary('org/my-dataset'); print(s['env'], s['total'], [f['key'] for f in s['facets']])
r = registry.tasks('org/my-dataset', everything=True)['cards'][0]['ref']
print(registry.task('org/my-dataset', r)['sections'])"
```

Then open `http://localhost:8060/d/org/my-dataset`, connect an agent to `/mcp/d/org/my-dataset`, and add tests:
`tests/test_contract.py` registers a toy adapter and checks the whole contract end to end (pages, runs, aliases,
MCP); `tests/test_mimo.py` and `tests/test_envs.py` show an adapter's and a processor's own checks, answers first.
