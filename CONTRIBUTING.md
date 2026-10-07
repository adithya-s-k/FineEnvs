# Contributing

**We are actively looking for new end-to-end recipes.** That's the single most valuable thing you can
contribute here, and it's what this repo is for.

## What an end-to-end recipe is

A complete, reproducible path from *"here is a task"* to *"here is a trained model"*:

1. **An environment** — the task, the tools, the observations, the reward.
2. **A rollout** that demonstrably works against it, with a trajectory a human can read.
3. **A training run** — config, launch command, hardware, and roughly what it cost.
4. **Results** — the curve, and the honest version of it. What worked, what didn't, where the model
   found a shortcut you didn't intend.
5. **Artifacts on the Hub** — the environment Space, the task dataset, the trained model.

If you have all five, open a PR adding a new numbered project folder. If you have two or three of
them, open an issue anyway — a half-finished recipe with real numbers beats a polished one that
nobody ran. We'd rather help you land it than never see it.

Recipes on domains we don't cover yet are especially welcome: web browsing, SQL and data analysis,
games, robotics simulators, tool-use over real APIs, multi-agent setups, long-horizon software
engineering. Different base models and different trainers are welcome too — the point is coverage of
the *space*, not one blessed stack.

## Everything else that helps

1. **A new environment**, even without the training half.
2. **A new framework port** of an environment that already exists.
3. **A reproduction that disagrees with ours.** If you ran a recipe and got a different curve, that's
   a bug report we want — please include your config and hardware.
4. **Fixes to rollouts that broke.** Frameworks move fast; things rot.
5. **Corrections to the guide or the slides.** If an explanation is wrong or unclear, say so.

## Repository shape

```
FineEnvs/
├── NN-<project>/       a self-contained project
│   ├── README.md       the project page
│   ├── project.yaml    manifest — drives the generated README tables
│   ├── envs/<env>/     core/ + one folder per framework
│   ├── train/          configs + launch scripts
│   ├── notebooks/
│   └── results/
├── content/            articles/ and slides/
├── tools/              deploy.py, build_index.py, jupyter_launch.py
└── .claude/skills/     the five env-authoring agent skills
```

Those folder names mean the same thing in every project. Keep it that way.

## Adding a project

A project is a top-level numbered folder. Copy the shape of an existing one:

```
NN-<name>/
├── README.md       what it is, how to run it, what the results were
├── project.yaml    manifest — Hub links and per-framework status
├── envs/<env>/     core/ + one folder per framework
├── train/          config + launch script
├── notebooks/
└── results/        curves, and links to the Hub artifacts
```

Pick the next free number. Register the Hub repos you own in `project.yaml`, then run
`python3 tools/build_index.py` so the root README picks it up. A project that ships in parts can keep
one manifest per part, one level down (`NN-<name>/<part>/project.yaml`). CI fails if a numbered
folder has no manifest at all.

The table is generated, the rest of the root README is not. In the same PR, add a short section for
the project under **Projects** (what it trains, the headline result, links to the Space, article and
collection), a line in **Repository layout**, and a card under **Articles & talks** if it ships an
article.

## Adding an environment

The paved path is the agent skills — they do the scaffolding and the rollout smoke test:

```bash
npx skills add adithya-s-k/FineEnvs
# then, in your agent: "make me an env where the agent plays connect-four"
```

By hand, inside the project you're adding to:

1. Put the domain logic in `envs/<env>/core/` — the game, the controller, the task list. Anything a
   framework port would otherwise copy belongs here. **If two framework folders contain the same
   file, it belongs in `core/`.**
2. Add one folder per framework beside it. Each ships a `README.md`, a `pyproject.toml`, and a
   runnable `rollout.py`.
3. Framework code imports from `core/`:
   ```python
   _ENV_ROOT = str(Path(__file__).resolve().parents[1])
   if _ENV_ROOT not in sys.path:
       sys.path.insert(0, _ENV_ROOT)
   from core.game import WordleGame  # noqa: E402
   ```
4. Record it in `project.yaml`, then run `python3 tools/build_index.py` to refresh the tables.

## Before you open a PR

```bash
python3 tools/build_index.py --check    # generated tables are current
uv run python rollout.py                # your env actually runs
```

**Read one trajectory by hand.** Did the model see what you expected? Did the tool returns make
sense? Did the reward fire? If a human can't read the trajectory and tell whether the model did
well, neither can a reward function — and no amount of training will surface that.

## Articles and slides

Sources live in `content/`, never as a nested git clone. Deployment goes over the Hub HTTP endpoint:

```bash
python3 tools/deploy.py content/slides/<deck> FineEnvs/<space>
```

Each item's `space.md` is its Space card — the frontmatter there decides how it deploys.

## What doesn't belong in git

Build output (`node_modules/`, `dist/`, `.astro/`), slide exports (`export/` — PPTX and PDF go to the
Hub), virtualenvs, and anything over a few MB. Large artifacts belong on
[FineEnvs](https://huggingface.co/FineEnvs), linked from the project README.

## License

Contributions are under [Apache 2.0](./LICENSE).
