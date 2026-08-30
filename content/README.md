# Content

Everything written or presented: long-form articles, talk decks, and the sources that build them.
Each item ships to the Hub as a Space; the source of truth is here.

## Articles

| Article | Source | Live |
|---|---|---|
| **The ultimate guide to RL environments** | [`articles/rl-environments-guide/`](./articles/rl-environments-guide/) | [▶️ Space](https://huggingface.co/spaces/AdithyaSK/rl-environments-guide) |
| **The ultimate guide to multi-harness RL** | [`articles/multi-harness-rl/`](./articles/multi-harness-rl/) | 🚧 drafting |

Built with [research-article-template](https://huggingface.co/spaces/tfrere/research-article-template)
(Astro), served as a Docker Space.

```bash
cd content/articles/rl-environments-guide/app
npm install && npm run dev          # http://localhost:4321
```

Prose lives in `app/src/content/` — `article.mdx`, `chapters/`, `embeds/`, `bibliography.bib`.

## Slides

| Deck | Source | Live |
|---|---|---|
| **RL Environments 101** | [`slides/rl-environments-101/`](./slides/rl-environments-101/) | [▶️ Space](https://huggingface.co/spaces/AdithyaSK/rl-environments-101-slides) |
| **Scaling RL for LLMs** (AMD AI Dev Day) | [`slides/scaling-rl-amd/`](./slides/scaling-rl-amd/) | [▶️ Space](https://huggingface.co/spaces/AdithyaSK/scaling-rl-for-llms-amd-ai-dev-day) |
| **Multi-Harness Training** (OpenEnv × Harbor) | [`slides/multi-harness-training/`](./slides/multi-harness-training/) | [▶️ Space](https://huggingface.co/spaces/AdithyaSK/multi-harness-training-slides) |

React components on a fixed 1280×720 canvas that scales to any projector, dark/light themes,
arrow-key navigation. All three bind port 5173, so run one at a time.

```bash
cd content/slides/<deck>
npm install && npm run dev          # http://localhost:5173
```

| Key | Action |
|---|---|
| `→` / `Space` / `PgDn` | Next slide |
| `←` / `PgUp` | Previous slide |
| `t` | Toggle dark / light |
| `f` | Fullscreen |

`npm run export` produces PPTX + PDF into `export/`. **Exports are not tracked in git** — they're
tens of megabytes each and belong on the Hub.

## Which project does what belong to?

| Content | Project |
|---|---|
| The RL environments guide | [00 · RL Environments 101](../00-environments-101/) |
| RL Environments 101 deck | [00 · RL Environments 101](../00-environments-101/) |
| Scaling RL for LLMs deck | [00 · RL Environments 101](../00-environments-101/) |
| Multi-Harness Training deck | cross-cutting |

## Deploying

Articles and decks ship over the Hub HTTP endpoint — **not** `git push` to a Space remote, so no
nested git repos live in this tree:

```bash
# article (Docker SDK) — upload the directory as-is
hf upload HuggingEnvs/<space> content/articles/<name> . --repo-type space

# deck (static SDK) — build first, upload dist/
cd content/slides/<name> && npm run build
hf upload HuggingEnvs/<space> content/slides/<name>/dist . --repo-type space
```

Each item's own `README.md` carries the HF frontmatter (`sdk:`, `app_file:`/`app_port:`) — that file
*is* the Space card, so keep it accurate. `tools/deploy.py` reads it and picks the right upload.
