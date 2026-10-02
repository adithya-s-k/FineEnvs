---
name: create-html-embed
description: Create self-contained D3 HTML embed charts for the research article template. Use when the user asks to create a chart, visualization, embed, D3 chart, line chart, bar chart, scatter plot, sankey diagram, or any data visualization as an HTML embed file.
---

# Create HTML Embed

Create self-contained D3.js chart embeds for the research article template.

## Before you start

**Read the full directives file** for all conventions, patterns, and checklists:

- [directives.md](directives.md) — single source of truth for embed authoring rules

This covers: colors & palettes, layout, SVG scope, mounting, theming, controls, tooltips, data loading, responsiveness, legends, accessibility, performance, error handling, printing, and the full agent checklist.

## Workflow

### Step 1: Understand the request

Clarify with the user:
- What type of chart? (line, bar, scatter, sankey, waffle, heatmap, custom)
- What data source? (CSV path, JSON, inline data)
- Interactive controls needed? (metric selector, filters)
- Any specific design requirements?

### Step 2: Create the HTML file

- Location: `app/src/content/embeds/`
- Naming: `d3-<descriptive-name>.html` (e.g., `d3-training-loss.html`)
- Root class: `.d3-<descriptive-name>` (must match filename)

### Step 3: Follow the mandatory structure

Every embed must have this structure:

```html
<div class="d3-yourname"></div>
<style>
  .d3-yourname { /* scoped styles */ }
</style>
<script>
  (() => {
    const ensureD3 = (cb) => { /* D3 CDN loader */ };
    const bootstrap = () => {
      /* mount guard + container selection */
      /* tooltip setup */
      /* SVG scaffolding */
      /* data loading */
      /* render function with ResizeObserver */
    };
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', () => ensureD3(bootstrap), { once: true });
    } else { ensureD3(bootstrap); }
  })();
</script>
```

### Step 4: Integrate in MDX

Import and use the `HtmlEmbed` component:

```mdx
import HtmlEmbed from '../../components/HtmlEmbed.astro';

<HtmlEmbed src="d3-yourname.html" title="Chart Title" desc="Description text" />
```

#### HtmlEmbed props

| Prop | Type | Description |
|------|------|-------------|
| `src` | string | Path to HTML file in `embeds/` (required) |
| `title` | string | Title above the card |
| `desc` | string | Description below (supports HTML) |
| `frameless` | boolean | Removes card background/border |
| `wide` | boolean | Wide layout (~1100px) |
| `data` | string or string[] | Path(s) to data files |
| `config` | object | JSON config passed via `data-config` attribute |

#### Usage examples

```mdx
<!-- Simple embed -->
<HtmlEmbed src="d3-training-loss.html" title="Training Loss" />

<!-- With external data -->
<HtmlEmbed src="d3-line-simple.html" title="Attention" data="attention_loss.csv" />

<!-- With config -->
<HtmlEmbed
  src="d3-line-simple.html"
  title="Learning Rate"
  data="lr_loss.csv"
  config={{ defaultMetric: 'loss', xDomain: [0, 45e9] }}
/>

<!-- Multiple data files -->
<HtmlEmbed
  src="d3-comparison.html"
  title="A vs B"
  data={['formatting_filters.csv', 'relevance_filters.csv']}
/>

<!-- Frameless -->
<HtmlEmbed frameless src="d3-banner.html" />
```

## Key conventions (quick reference)

Full details in the directives file. The critical ones:

1. **Colors**: Use `window.ColorPalettes.getColors('categorical', n)` — never hardcode palettes
2. **CSS variables**: `--text-color`, `--surface-bg`, `--border-color`, `--axis-color`, `--tick-color`, `--grid-color`
3. **Dark mode**: Check `document.documentElement.getAttribute('data-theme') === 'dark'`
4. **Mount guard**: Always set `container.dataset.mounted = 'true'`
5. **Data loading**: Try `/data/<file>` first, then `./assets/data/<file>` — use `fetchFirstAvailable()`
6. **Responsiveness**: `ResizeObserver` on container, recompute on resize
7. **Legend**: HTML-based, title "Legend", swatch 14x14px
8. **Controls**: HTML only (no SVG UI), selects labeled "Metric" when applicable
9. **Tooltip**: Single `.d3-tooltip` absolutely positioned inside container
10. **No globals**: Everything in IIFE, nothing on `window`

## Data files

- Store data in: `app/src/content/assets/data/`
- Served from: `/data/` (public) at build time
- Formats: CSV (preferred for tabular), JSON (for nested/hierarchical)

## Post-creation checklist

After creating the embed, verify against the **Agent Checklist** (section 14.1) and **Definition of Done** (section 14.2) in [directives.md](directives.md).
