---
title: "The ultimate guide to multi-harness RL"
short_description: "Training and evals together across many agent harnesses"
emoji: 🔀
colorFrom: green
colorTo: purple
sdk: docker
app_port: 8080
header: mini
pinned: false
tags:
  - research-article-template
  - rl-environments
  - llm-training
  - reinforcement-learning
  - agents
thumbnail: >-
  https://raw.githubusercontent.com/adithya-s-k/HuggingEnvs/refs/heads/main/assets/content/multi-harness-training.png
---

# Multi-Harness RL

[Read the article](https://huggingface.co/spaces/HuggingEnvs/multi-harness-rl).

A research article built with [research-article-template](https://huggingface.co/spaces/tfrere/research-article-template).

Source lives in [HuggingEnvs](https://github.com/adithya-s-k/HuggingEnvs) under
`content/articles/multi-harness-rl/`.

## Quick start

```bash
cd app
npm install
npm run dev           # http://localhost:4321
```

## Where the content lives

| Path | What |
| --- | --- |
| `app/src/content/article.mdx` | Frontmatter and the chapter registry — the explicit import list *is* the running order |
| `app/src/content/chapters/` | One `.mdx` per section |
| `app/src/content/embeds/` | Standalone HTML/D3 visualizations, one file each |
| `app/src/content/assets/image/` | Images |
| `app/src/content/assets/data/` | Data files, served at `/data/<name>` |
| `app/src/content/bibliography.bib` | References, cited as `[@key]` |

## Deploy

From the repo root, over the Hub HTTP endpoint (no git remote, no nested repo):

```bash
python3 tools/deploy.py content/articles/multi-harness-rl HuggingEnvs/multi-harness-rl
```

The Dockerfile and nginx config are included; this README is the Space card.
