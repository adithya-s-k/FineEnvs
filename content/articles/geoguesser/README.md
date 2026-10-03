# How to turn a game into an RL environment

Source for the research article of the same name. Built with
[research-article-template](https://huggingface.co/spaces/tfrere/research-article-template) and
deployed as a Docker Space: **[FineEnvs/geoguesser-article](https://huggingface.co/spaces/FineEnvs/geoguesser-article)**.

The project it documents lives in [`03-geoguesser/`](../../../03-geoguesser/): the environment, the
dataset pipeline, the eval harness, the training script and the results.

## Quick start

```bash
cd app
npm install
npm run dev           # http://localhost:4321
npm run build         # static build, into app/dist
```

## Where the content lives

| Path | What |
| --- | --- |
| `app/src/content/article.mdx` | Frontmatter and the chapter registry. The import list *is* the running order |
| `app/src/content/chapters/*.mdx` | One file per section |
| `app/src/content/embeds/*.html` | Self-contained figures, each one theme-aware |
| `app/src/content/assets/data/*.json` | The numbers behind the figures, symlinked to `app/public/data` |
| `app/public/thumbs/`, `app/public/episode/` | Rendered imagery for the map hover cards and the episode player |
| `space.md` | The Space card. `README.md` stays a repo README |

## Regenerating the figure data

Every JSON file under `assets/data/` is derived from the project's own published records rather than
typed by hand. The scripts that build them:

```bash
# the hero: 200 paired guesses, untrained against trained
python3 scripts/build_hero_guesses.py ../../../03-geoguesser/results/raw/passk-run1-full

# the run 1 curves, from the Trackio database
hf download --repo-type bucket FineEnvs/geoguesser-trackio-bucket trackio/geoguesser.db --local-dir .
python3 scripts/build_run1_curve.py trackio/geoguesser.db --run run1-4b
python3 scripts/build_runs_std.py trackio/geoguesser.db

# the rollout comparison, and the episode player's frames
python3 scripts/build_rollout_compare.py ../../../03-geoguesser/results/raw/passk-run1-full \
  --example "18:base=13535,ckpt1000=505" --example "100:base=16995,ckpt1000=22"
python3 scripts/build_episode_web.py <a render_rollout.py output dir>

# the map hover previews, from the synced panorama bucket
python3 scripts/build_task_thumbs.py --env ../../../03-geoguesser/env
```

## Deploying

```bash
python3 tools/deploy.py content/articles/geoguesser FineEnvs/geoguesser-article
```

The Space builds the Astro app itself, so `node_modules/` and `dist/` are excluded from the upload.
