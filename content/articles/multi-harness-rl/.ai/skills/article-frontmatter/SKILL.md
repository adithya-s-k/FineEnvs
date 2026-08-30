---
name: article-frontmatter
description: Configure article metadata via MDX frontmatter. Use when the user asks about titles, authors, affiliations, template variants, banner, DOI, PDF export, or any article.mdx frontmatter field.
---

# Article Frontmatter Reference

All metadata lives in `app/src/content/article.mdx` frontmatter (YAML block).

## Frontmatter fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `title` | string | required | Article title. Supports `\n` for line breaks (rendered as `<br/>`). |
| `description` | string | `""` | Short description / subtitle shown below the title. |
| `authors` | array | `[]` | List of authors (see below). |
| `affiliations` | array | `[]` | List of affiliations (see below). |
| `published` | string | — | Publication date, e.g. `"Apr. 04, 2026"`. |
| `template` | `"article"` or `"paper"` | `"article"` | Layout variant (see below). |
| `banner` | string | `"banner.html"` | Banner embed filename in `embeds/`. |
| `doi` | string | — | DOI identifier, shown in footer. |
| `showPdf` | boolean | `true` | Show PDF download button in metadata bar. |
| `tableOfContentsAutoCollapse` | boolean | `false` | Auto-collapse TOC sections on scroll. |
| `licence` | string | — | Licence text (HTML allowed), shown in footer. |
| `pdfProOnly` | boolean | `false` | Gate PDF download behind HF Pro badge. |
| `seoThumbImage` | string | — | Custom OG image URL for social sharing. |
| `links` | array | `[]` | External links shown in paper template hero (see below). |

## Title line breaks

Long titles are automatically balanced across lines (`text-wrap: balance`). Titles longer than 60 characters are automatically downsized for readability (>100 chars: even smaller).

To force a manual line break, use `\n` inside the title string:

```yaml
title: "Why Open-Source LLMs\nAre Reshaping the AI Landscape"
```

This renders as two lines in the Hero section. The plain-text version (for SEO / PDF) strips the break automatically.

## Template variants

| Value | Layout | Features |
|-------|--------|----------|
| `article` (default) | Full layout | Banner, sidebar TOC, figure numbering, citation block, DOI, PDF export |
| `paper` | Single centered column | No TOC sidebar, no figure numbering, no citation/DOI block, lighter footer |

## Authors and affiliations

```yaml
authors:
  - name: "Alice Martin"
    url: "https://example.com/alice"
    affiliations: [1]
  - name: "Bob Chen"
    affiliations: [1, 2]
affiliations:
  - name: "Hugging Face"
    url: "https://huggingface.co"
  - name: "MIT"
    url: "https://mit.edu"
```

Affiliation indices are 1-based and rendered as superscript numbers next to author names.

## External links (paper template)

The `links` field adds buttons below the author/affiliation line in the `paper` template hero. Each link has a `label` and a `url`:

```yaml
links:
  - label: "Paper"
    url: "https://arxiv.org/abs/..."
  - label: "Code"
    url: "https://github.com/..."
  - label: "Demo"
    url: "https://huggingface.co/spaces/..."
  - label: "Data"
    url: "https://huggingface.co/datasets/..."
```

Links are rendered as pill-shaped buttons and only visible in the `paper` template. They are hidden in the `article` template.

## README tag (critical)

The project `README.md` contains a YAML frontmatter block with a `tags` field:

```yaml
tags:
  - research-article-template
```

**NEVER remove the `research-article-template` tag from the README.** This tag is used by the [Research Article Gallery](https://huggingface.co/spaces/tfrere/research-article-gallery) to discover and list all articles built with this template. Removing it will make the Space invisible in the gallery.

## Complete example

```yaml
---
title: "Scaling Laws for\nNeural Language Models"
description: "An empirical study of scaling behavior across model size, data, and compute"
authors:
  - name: "Alice Martin"
    url: "https://example.com/alice"
    affiliations: [1]
affiliations:
  - name: "Hugging Face"
    url: "https://huggingface.co"
published: "Apr. 04, 2026"
template: "article"
banner: "banner.html"
doi: "10.1234/example.2026"
showPdf: true
tableOfContentsAutoCollapse: true
licence: "This work is licensed under <a href='https://creativecommons.org/licenses/by/4.0/'>CC BY 4.0</a>."
---
```
