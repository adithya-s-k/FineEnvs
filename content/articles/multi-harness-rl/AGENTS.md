# Agent Instructions

This file provides context for AI coding agents (Claude Code, GitHub Copilot, Cursor, Windsurf, etc.) working on this project.

## Project overview

This is a **research article template** built with Astro + MDX. It produces interactive, web-native scientific papers with D3.js data visualizations, dark mode support, and PDF export.

### Key paths

| What | Where |
|------|-------|
| Astro app | `app/` |
| MDX content | `app/src/content/` (article.mdx, chapters/) |
| Components | `app/src/components/` |
| HTML embeds (D3 charts) | `app/src/content/embeds/` |
| Data files (CSV/JSON) | `app/src/content/assets/data/` |
| Styles & CSS variables | `app/src/styles/` |
| Scripts (export, import) | `app/scripts/` |

### Tech stack

- **Astro 4** with MDX
- **D3.js v7** for data visualization (loaded via CDN in embeds)
- **Svelte** for interactive components (e.g., Trackio)
- **CSS custom properties** for theming (light/dark mode)

## Agent skills

Detailed guides for common tasks live in `.ai/skills/`. Read the relevant skill file before starting work:

| Skill | File | When to use |
|-------|------|-------------|
| Create HTML embed | [`.ai/skills/create-html-embed/SKILL.md`](.ai/skills/create-html-embed/SKILL.md) | Creating D3 charts, visualizations, interactive embeds |
| Article frontmatter | [`.ai/skills/article-frontmatter/SKILL.md`](.ai/skills/article-frontmatter/SKILL.md) | Configuring title, authors, template variant, DOI, PDF |
| Deploy to HF Spaces | [`.ai/skills/deploy-to-hf-spaces/SKILL.md`](.ai/skills/deploy-to-hf-spaces/SKILL.md) | Deploying, pushing, or updating a Hugging Face Space |

> **Cursor users**: these skills are auto-detected via a symlink at `.cursor/skills/`.

## Creating HTML embed charts

When asked to create a chart, visualization, or D3 embed, read the skill files:

1. **Workflow & quick reference**: `.ai/skills/create-html-embed/SKILL.md`
2. **Full conventions & directives**: `.ai/skills/create-html-embed/directives.md`

### TL;DR

- Embeds are self-contained `.html` files in `app/src/content/embeds/`
- Structure: single root `<div>` + scoped `<style>` + IIFE `<script>`
- Colors from `window.ColorPalettes`, theming via CSS variables
- Dark mode aware, responsive (ResizeObserver), mount-guarded (`data-mounted`)
- Integrated in MDX via `<HtmlEmbed src="d3-name.html" title="..." />`

## Article frontmatter

When asked to configure article metadata (title, authors, template variant, etc.), read `.ai/skills/article-frontmatter/SKILL.md`.

Key points:
- `template: "article"` (default) = full layout; `template: "paper"` = lighter single-column
- Titles support `\n` for line breaks; long titles are auto-sized
- Authors use 1-based affiliation indices

## Deploy to Hugging Face Spaces

When asked to deploy or update a Space, read `.ai/skills/deploy-to-hf-spaces/SKILL.md`.

Quick reference:
- SDK is Docker (`app_port: 8080`), build uses the included `Dockerfile`
- Push with `git push space main` (add remote first if needed)
- Cache issues: push an empty commit to force rebuild
- **Never remove the `research-article-template` tag from `README.md`** - it powers the [Research Article Gallery](https://huggingface.co/spaces/tfrere/research-article-gallery)

## Coding conventions

- **Commits**: follow [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, etc.)
- **Styles**: scoped CSS, use existing CSS variables from `app/src/styles/_variables.css`
- **Components**: Astro components (`.astro`), Svelte for interactivity (`.svelte`)
- **No globals**: embeds use IIFEs, no `window` pollution
- **Accessibility**: `aria-label` on interactive elements, `alt` on images, focus-visible styles
