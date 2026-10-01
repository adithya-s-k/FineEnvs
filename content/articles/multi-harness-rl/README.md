---
title: "The ultimate guide to multi-harness RL"
short_description: "Train open models with RL inside real agent harnesses"
emoji: 🔀
colorFrom: green
colorTo: purple
sdk: docker
app_port: 8080
header: mini
pinned: false
hf_oauth: true
hf_oauth_expiration_minutes: 480
tags:
  - research-article-template
  - rl-environments
  - llm-training
  - reinforcement-learning
  - agents
  - openenv
  - harbor
  - grpo
  - trl
thumbnail: >-
  https://huggingface.co/spaces/FineEnvs/multi-harness-rl/resolve/main/app/public/og/og-image.png
---

# Multi-Harness RL

[Read the article](https://huggingface.co/spaces/FineEnvs/multi-harness-rl).

A research article built with [research-article-template](https://huggingface.co/spaces/tfrere/research-article-template).

Source lives in [FineEnvs](https://github.com/adithya-s-k/FineEnvs) under
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
python3 tools/deploy.py content/articles/multi-harness-rl FineEnvs/multi-harness-rl
```

The Dockerfile and nginx config are included; this README is the Space card.

## Review mode

The Space can take comments and suggested edits on the article, Google Docs style, without any of it
showing on the published page.

- **Published (default).** With `REVIEW_MODE` unset or `off`, the page never loads the review
  layer and every `/api/review/` endpoint answers 404.
- **Review.** With the Space variable `REVIEW_MODE=on`, open the article with `?review`
  (`https://fineenvs-multi-harness-rl.hf.space/?review`) and sign in with Hugging Face. Select text to
  comment, suggest a deletion or suggest new wording, or use a figure's Comment button. Cards sit in
  the margin next to their text; the Comments panel lists open, resolved and detached threads. Only the
  Space owner and the usernames in `REVIEWERS` can read or write, and only the owner can accept or
  reject a suggestion. **Exit** (or `?review=off`) goes back to the published view.

For an organization-owned Space, set `REVIEW_OWNER` to the maintainer's HF username. This keeps
review moderation tied to a person rather than the organization name.

Threads are stored one JSON file each under `/data/review-comments/threads/`, on the private bucket
`AdithyaSK/multi-harness-rl-review`, mounted at `/data` (Space settings → Storage Buckets). To read them all at once:

```bash
uv run --with huggingface_hub python review/pull_comments.py   # writes review/.pulled/comments.md (git-ignored)
```

The script calls the owner-only `/api/review/export` endpoint with your local Hugging Face login;
`/api/review/import` takes threads back in the same format.

The server is `review/server.py` (FastAPI, run next to nginx by `entrypoint.sh`); the page side is
`app/public/review/`. For local testing, run the server with `COMMENTS_DIR` pointing at a folder and
`REVIEW_MODE=on`; the dev server proxies `/api/review` and `/oauth` to it, and sign-in is mocked
with your local Hugging Face login.
