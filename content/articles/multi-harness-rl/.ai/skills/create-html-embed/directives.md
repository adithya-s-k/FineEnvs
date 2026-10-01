## Embed Chart Authoring Guidelines

### Quickstart (TL;DR)
- Create a single self-contained HTML fragment: root div + scoped style + IIFE script.
- Draw marks/axes in SVG; render UI (legend and controls) in HTML.
- Place legend and controls BELOW the chart (header appended after the chart). Include a legend title "Legend" and a select labeled "Metric" when relevant.
- Load data from public `/data` first, then fall back to `assets/data`.
- Use `window.ColorPalettes` for colors; stick to CSS variables for theming.

Minimal header markup:
```html
<div class="legend">
  <div class="legend-title">Legend</div>
  <div class="items"></div>
  <!-- items populated by JS: <span class="item"><span class="swatch"></span><span>Name</span></span> -->
</div>
<div class="controls">
  <div class="control-group">
    <label for="metric-select-<id>">Metric</label>
    <select id="metric-select-<id>"></select>
  </div>
  <!-- optional: other controls -->
</div>
```

See also: `d3-line-simple.html`, `d3-line-quad.html`, `d3-benchmark.html`.

Authoring rules for creating a new interactive chart as a single self-contained `.html` file under `src/content/embeds/`. These conventions are derived from `d3-bar.html`, `d3-comparison.html`, `d3-neural.html`, `d3-line.html`, and `d3-pie.html`.

### A) Colors & palettes (MANDATORY)
- Always obtain color arrays from `window.ColorPalettes`; do not hardcode palettes.
- Use the categorical/sequential/diverging helpers and the current primary color.
- If you change `--primary-color` dynamically, call `window.ColorPalettes.refresh()` so listeners update.

Usage:
```js
// Usage (with explicit counts)
const cat = window.ColorPalettes.getColors('categorical', 8);
const seq = window.ColorPalettes.getColors('sequential', 8);
const div = window.ColorPalettes.getColors('diverging', 7);

// For current primary color string
const primaryHex = window.ColorPalettes.getPrimary();

// If you change --primary-color dynamically, call refresh to notify listeners
document.documentElement.style.setProperty('--primary-color', '#6D4AFF');
window.ColorPalettes.refresh();
```

Notes:
- Keep chart accents (lines, markers, selection) aligned with `--primary-color`.
- Prefer CSS variables for fills/strokes when possible; derive series colors via `ColorPalettes`.
- Provide a graceful fallback to CSS variables if `window.ColorPalettes` is unavailable.

### B) Layout & form elements (HTML-only)
- All UI controls (labels, selects, sliders, buttons, toggles) must be plain HTML inside the root container.
- Do not draw controls with SVG; style them consistently (rounded 8px, custom caret, focus ring).
- Use `<label>` wrapping inputs for accessibility and concise text (e.g., "Metric", "Model Size").
- Manage layout with CSS inside the scoped `<style>` for the root class; avoid global rules.

### C) SVG scope: charts only; UI in HTML
- SVG is for chart primitives (marks, axes, gridlines) only.
- Put legends and controls in HTML (adjacent DOM is preferred; `foreignObject` only if necessary).
- Tooltips are HTML positioned absolutely inside the root container.
  - Details: see sections 4 (controls), 5 (tooltips), 8 (legends).

### 1) File, naming, and structure
- Name files with a clear prefix and purpose: `d3-<type>.html` (e.g., `d3-scatter.html`).
- Wrap everything in a single `<div class="<root-class>">`, a `<style>` block scoped to that root class, and a `<script>` IIFE.
- Do not leak globals; do not attach anything to `window`.
- Use a unique, descriptive root class (e.g., `.d3-scatter`).

Minimal skeleton:
```html
<div class="d3-yourchart"></div>
<style>
  .d3-yourchart {/* all styles scoped to the root */}
</style>
<script>
  (() => {
    // Optional dependency loader (e.g., D3)
    const ensureD3 = (cb) => {
      if (window.d3 && typeof window.d3.select === 'function') return cb();
      let s = document.getElementById('d3-cdn-script');
      if (!s) { s = document.createElement('script'); s.id = 'd3-cdn-script'; s.src = 'https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js'; document.head.appendChild(s); }
      const onReady = () => { if (window.d3 && typeof window.d3.select === 'function') cb(); };
      s.addEventListener('load', onReady, { once: true });
      if (window.d3) onReady();
    };

    const bootstrap = () => {
      const scriptEl = document.currentScript;
      // Prefer the closest previous sibling with the root class
      let container = scriptEl ? scriptEl.previousElementSibling : null;
      if (!(container && container.classList && container.classList.contains('d3-yourchart'))) {
        // Fallback: pick the last unmounted instance in the page
        const candidates = Array.from(document.querySelectorAll('.d3-yourchart'))
          .filter((el) => !(el.dataset && el.dataset.mounted === 'true'));
        container = candidates[candidates.length - 1] || null;
      }
      if (!container) return;
      if (container.dataset) {
        if (container.dataset.mounted === 'true') return;
        container.dataset.mounted = 'true';
      }

      // Tooltip (optional)
      container.style.position = container.style.position || 'relative';
      let tip = container.querySelector('.d3-tooltip'); let tipInner;
      if (!tip) {
        tip = document.createElement('div'); tip.className = 'd3-tooltip';
        Object.assign(tip.style, { position:'absolute', top:'0px', left:'0px', transform:'translate(-9999px, -9999px)', pointerEvents:'none', padding:'8px 10px', borderRadius:'8px', fontSize:'12px', lineHeight:'1.35', border:'1px solid var(--border-color)', background:'var(--surface-bg)', color:'var(--text-color)', boxShadow:'0 4px 24px rgba(0,0,0,.18)', opacity:'0', transition:'opacity .12s ease' });
        tipInner = document.createElement('div'); tipInner.className = 'd3-tooltip__inner'; tipInner.style.textAlign='left'; tip.appendChild(tipInner); container.appendChild(tip);
      } else { tipInner = tip.querySelector('.d3-tooltip__inner') || tip; }

      // SVG scaffolding (if using D3)
      const svg = d3.select(container).append('svg').attr('width','100%').style('display','block');
      const gRoot = svg.append('g');

      // State & layout
      let width = 800, height = 360; const margin = { top: 16, right: 28, bottom: 56, left: 64 };
      function updateSize(){
        const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
        width = container.clientWidth || 800;
        height = Math.max(260, Math.round(width / 3));
        svg.attr('width', width).attr('height', height);
        gRoot.attr('transform', `translate(${margin.left},${margin.top})`);
        return { innerWidth: width - margin.left - margin.right, innerHeight: height - margin.top - margin.bottom, isDark };
      }

      function render(){
        const { innerWidth, innerHeight } = updateSize();
        // ... draw/update your chart here using data joins
      }

      // Initial render + resize handling
      render();
      const rerender = () => render();
      if (window.ResizeObserver) { const ro = new ResizeObserver(() => rerender()); ro.observe(container); }
      else { window.addEventListener('resize', rerender); }
    };

    if (document.readyState === 'loading') { document.addEventListener('DOMContentLoaded', () => ensureD3(bootstrap), { once: true }); }
    else { ensureD3(bootstrap); }
  })();
</script>
```

### 2) Mounting and re-entrancy
- Select the closest previous sibling with the root class; fallback to the last unmounted matching element in the document.
- Gate with `data-mounted` to avoid double-initialization when the fragment re-runs.
- Assume the chart can appear multiple times on the same page.

### 3) Styling and theming
- Scope all rules under the root class; do not style `body`, `svg` globally.
- Use CSS variables for theme alignment: `--primary-color`, `--text-color`, `--muted-color`, `--surface-bg`, `--border-color`.
- Derive palette colors from `window.ColorPalettes` (categorical, sequential, diverging); do not hardcode arrays.
- For dark mode–aware strokes/ticks, either:
  - Read `document.documentElement.getAttribute('data-theme') === 'dark'`, or
  - Prefer CSS-only where possible.
- Keep backgrounds light and borders subtle; the outer card frame is handled by `HtmlEmbed.astro`.

Standard axis/tick/grid colors (global variables from `_variables.css`):

```css
/* Provided globally */
:root {
  --axis-color: var(--text-color);
  --tick-color: var(--muted-color);
  --grid-color: rgba(0,0,0,.08);
}
[data-theme="dark"] {
  --axis-color: var(--text-color);
  --tick-color: var(--muted-color);
  --grid-color: rgba(255,255,255,.10);
}
/* Apply inside charts */
.your-root-class .axes path,
.your-root-class .axes line { stroke: var(--axis-color); }
.your-root-class .axes text { fill: var(--tick-color); }
.your-root-class .grid line { stroke: var(--grid-color); }
```

#### 3.1) Text on fixed-colored backgrounds

- When rendering text over cells/areas with fixed background colors (that do not change with theme), compute a readable text style once from the actual background color.
- Use `window.ColorPalettes.getTextStyleForBackground(bgCss, { blend: 0.6 })` when available; avoid tying text color to dark mode toggles since the background is constant.
- Do not re-evaluate on theme toggle unless the background color itself changes.

Example:
```js
const bg = getComputedStyle(cellRect).fill; // e.g., 'rgb(12, 34, 56)'
const style = window.ColorPalettes?.getTextStyleForBackground
  ? window.ColorPalettes.getTextStyleForBackground(bg, { blend: 0.6 })
  : { fill: 'var(--text-color)' };
textSel.style('fill', style.fill);
```

### 4) Controls (labels, selects, sliders)
- Compose controls as plain HTML elements appended inside the root container (no SVG UI).
- Style selects like in `d3-line.html`/`d3-bar.html` for consistency (rounded 8px, custom caret via data-URI, focus ring).
- Use `<label>` wrapping the input for accessibility; set concise text (e.g., "Metric", "Model Size").

#### 4.1) Required select label: "Metric"
- When a select is used to switch metrics, include a visible label above the select with the exact text "Metric".
- Preferred markup (grouped for easy vertical stacking):

```html
<div class="controls">
  <div class="control-group">
    <label for="metric-select-<unique>">Metric</label>
    <select id="metric-select-<unique>"></select>
  </div>
</div>
```

Minimal CSS (match project styles):

```css
.controls { display:flex; gap:16px; align-items:center; justify-content:flex-end; flex-wrap:wrap; }
.controls .control-group { display:flex; flex-direction:column; align-items:flex-start; gap:6px; }
.controls label { font-size:12px; font-weight:700; color: var(--text-color); }
.controls select { font-size:12px; padding:8px 28px 8px 10px; border:1px solid var(--border-color); border-radius:8px; background: var(--surface-bg); color: var(--text-color); }
```

### 5) Tooltip pattern
- Create a single `.d3-tooltip` absolutely positioned inside the container.
- Show on hover, hide on leave; position using `d3.pointer(event, container)` plus a small offset.
- Keep content in a `.d3-tooltip__inner` node; avoid large inner HTML.

### 6) Data loading
- Prefer public assets first, then fall back to content assets:
  - Example CSV paths: `/data/<file>.csv`, then `./assets/data/<file>.csv`, `../assets/data/<file>.csv`, etc.
- Implement `fetchFirstAvailable(paths)`; try in order with `cache:'no-cache'`; handle errors gracefully with a red `<pre>` message.
- For images or JSON models, mirror the same approach (see `d3-comparison.html`, `d3-neural.html`).

#### 6.1) Data props (HtmlEmbed → embed)

- HtmlEmbed accepts an optional `data` prop that can be a string (single file) or an array of strings (multiple files).
- This prop is passed to the fragment via the `data-datafiles` HTML attribute.
- In the embed script, read this attribute from the closest ancestor that carries it (the `HtmlEmbed` wrapper), not necessarily the chart’s direct container.
- Recommended normalization: if a value contains no slash, automatically prefix it with `/data/` to target the public data folder.
- If `data` is not provided, keep the usual fallback (public, then `assets/data`).

Optional configuration (e.g., default metric):

```js
// In HtmlEmbed usage (MDX):
<HtmlEmbed src="d3-line-simple.html" data="internal_deduplication.csv" config={{ defaultMetric: 'average_rank' }} />

// In the embed script (read from closest ancestor):
let mountEl = container;
while (mountEl && !mountEl.getAttribute?.('data-datafiles') && !mountEl.getAttribute?.('data-config')) {
  mountEl = mountEl.parentElement;
}
let providedConfig = null;
try {
  const cfg = mountEl && mountEl.getAttribute ? mountEl.getAttribute('data-config') : null;
  if (cfg && cfg.trim()) providedConfig = cfg.trim().startsWith('{') ? JSON.parse(cfg) : cfg;
} catch(_) {}
// Example: selecting initial metric if present
const desired = providedConfig && providedConfig.defaultMetric ? String(providedConfig.defaultMetric) : null;
```

Examples (MDX):

```mdx
<HtmlEmbed src="d3-line-simple.html" title="Run A" data="formatting_filters.csv" />
<HtmlEmbed src="d3-line-simple.html" title="Run B" data="relevance_filters.csv" />

<HtmlEmbed
  src="d3-line-simple.html"
  title="Comparison A vs B"
  data={[ 'formatting_filters.csv', 'relevance_filters.csv' ]}
/>
```

Reading on the embed side (JS):

```js
// Find the closest ancestor that carries the attribute
let mountEl = container;
while (mountEl && !mountEl.getAttribute?.('data-datafiles')) {
  mountEl = mountEl.parentElement;
}
let providedData = null;
try {
  const attr = mountEl && mountEl.getAttribute ? mountEl.getAttribute('data-datafiles') : null;
  if (attr && attr.trim()) {
    providedData = attr.trim().startsWith('[') ? JSON.parse(attr) : attr.trim();
  }
} catch(_) {}

const DEFAULT_CSV = '/data/formatting_filters.csv';
const ensureDataPrefix = (p) => (typeof p === 'string' && p && !p.includes('/')) ? `/data/${p}` : p;
const normalizeInput = (inp) => Array.isArray(inp)
  ? inp.map(ensureDataPrefix)
  : (typeof inp === 'string' ? [ ensureDataPrefix(inp) ] : null);

const CSV_PATHS = Array.isArray(providedData)
  ? normalizeInput(providedData)
  : (typeof providedData === 'string' ? normalizeInput(providedData) || [DEFAULT_CSV] : [
      DEFAULT_CSV,
      './assets/data/formatting_filters.csv',
      '../assets/data/formatting_filters.csv',
      '../../assets/data/formatting_filters.csv'
    ]);

const fetchFirstAvailable = async (paths) => {
  for (const p of paths) {
    try {
      const r = await fetch(p, { cache: 'no-cache' });
      if (r.ok) return await r.text();
    } catch(_){}
  }
  throw new Error('CSV not found');
};
```

### 7) Responsiveness and layout
- Compute `width = container.clientWidth`, and a height derived from width (e.g., `width / 3`), with a sensible minimum height.
- Maintain a `margin` object and derive `innerWidth/innerHeight` for plots.
- Use a `ResizeObserver` on the container; fallback to `window.resize`.
- Recompute scales/axes/grid on every render.

### 8) Legends and labels
- Prefer HTML for legends for wrapping and accessibility; avoid SVG-based legends.
- Always add axis labels when applicable (e.g., `Step`, `Value`).
- Standardize legend swatch size: 14×14px, border-radius 3px, 1px border `var(--border-color)`.

#### 8.1) Required legend title: "Legend"
- Always render a visible title above legend items with the exact text "Legend".
- Canonical markup:

```html
<div class="legend">
  <div class="legend-title">Legend</div>
  <div class="items">
    <!-- <span class="item"><span class="swatch"></span><span>Series A</span></span> ... -->
  </div>
</div>
```

Minimal CSS (match project styles):

```css
.legend { display:flex; flex-direction:column; align-items:flex-start; gap:6px; }
.legend-title { font-size:12px; font-weight:700; color: var(--text-color); }
.legend .items { display:flex; flex-wrap:wrap; gap:8px 14px; }
.legend .item { display:inline-flex; align-items:center; gap:6px; white-space:nowrap; font-size:12px; color: var(--text-color); }
.legend .swatch { width:14px; height:14px; border-radius:3px; border:1px solid var(--border-color); }
```

Recommended JS pattern to (re)build the legend:

```js
function makeLegend(seriesNames, colorFor) {
  let legend = container.querySelector('.legend');
  if (!legend) { legend = document.createElement('div'); legend.className = 'legend'; container.appendChild(legend); }
  let title = legend.querySelector('.legend-title'); if (!title) { title = document.createElement('div'); title.className = 'legend-title'; title.textContent = 'Legend'; legend.appendChild(title); }
  let items = legend.querySelector('.items'); if (!items) { items = document.createElement('div'); items.className = 'items'; legend.appendChild(items); }
  items.innerHTML = '';
  seriesNames.forEach(name => {
    const el = document.createElement('span'); el.className = 'item';
    const sw = document.createElement('span'); sw.className = 'swatch'; sw.style.background = colorFor(name);
    const txt = document.createElement('span'); txt.textContent = name;
    el.appendChild(sw); el.appendChild(txt); items.appendChild(el);
  });
}
```

### 9) Accessibility
- Provide `alt` attributes on `<img>` (see `d3-comparison.html`).
- Provide `aria-label` on interactive buttons (e.g., the erase button in `d3-neural.html`).
- Ensure focus-visible styles for interactive controls; avoid relying on color alone to encode meaning.

### 10) Performance and updates
- Use D3 data joins (`.data().join()` or explicit enter/merge/exit) and keep transitions short (≤200ms).
- Recompute only what is necessary on each render; avoid repeated DOM clears if not needed.
- Debounce or gate expensive computations, especially on `mousemove`.

### 11) External dependencies
- Load D3 (and optional TFJS) via CDN only once using an element id (e.g., `d3-cdn-script`, `tfjs-cdn-script`).
- After `.load`, verify the expected API (e.g., `window.d3.select`).
- Prefer pure D3 and built-ins; do not introduce new runtime dependencies unless necessary.

### 12) Error handling and fallbacks
- Fail gracefully: append a small `<pre>` with a readable message inside the container.
- For optional models (e.g., TFJS), attempt multiple URLs and fall back to a heuristic if load fails.

### 13) Printing
- Favor vector (`svg`) or simple shapes; avoid large bitmap backgrounds.
- Let `HtmlEmbed.astro` handle most print constraints; ensure the chart scales with width 100% and auto height.

### 14) Conventions checklist (before committing)
- Root class is unique and matches file name (`d3-<type>`).
- No globals added; script wrapped in an IIFE.
- `data-mounted` guard is present to avoid double-mount.
- Colors come from `window.ColorPalettes` (no hardcoded arrays); `--primary-color` respected.
- Uses CSS variables for colors; dark-mode friendly.
- Responsive: recomputes layout on resize; uses `ResizeObserver`.
- Controls are HTML-only, accessible, and consistently styled.
- Legends and tooltips are HTML, not SVG.
- Data loading includes public-path-first strategy and graceful error.
- Axes/labels are legible at small widths.
- Code is easy to skim: clear naming, early returns, short functions.

### 14.1) Agent Checklist (operational)
- Ensure root: one `<div .d3-xyz>` + scoped `<style>` + IIFE `<script>`
- Gate mount with `data-mounted` and select closest previous sibling instance
- Load D3 once via `#d3-cdn-script`; verify `window.d3.select`
- Colors from `window.ColorPalettes` with CSS variable fallbacks
- Legend present with visible title “Legend”; HTML-based, not SVG
- Controls in HTML only; if metric select exists, label text must be “Metric”
- Tooltip is a single absolute `.d3-tooltip` within the container
- Data load public-first; implement `fetchFirstAvailable([...])` with `cache:'no-cache'`
- Read optional HtmlEmbed `data-datafiles` and `data-config` per section 6.1
- Responsiveness: width from container; `ResizeObserver` fallback to `window.resize`
- Axis/tick/grid use CSS variables (`--axis-color`, `--tick-color`, `--grid-color`)
- SVG for marks only; UI/legend/controls in HTML
- No globals leaked; no external runtime deps besides D3/TFJS when necessary
- Error path: append small red `<pre>` with a readable message inside container
- Print-friendly: `svg` width 100%, height responsive, avoid heavy bitmaps

### 14.2) Definition of Done (DoD)
- Implements root structure and mounting guard
- Uses `ColorPalettes` (with safe fallback) and CSS variables for theming
- Legend with title “Legend” and consistent swatch style (14×14, r=3, 1px border)
- Metric select labelled “Metric” when present; accessible markup (`<label for>`)
- Tooltip works (show on hover, hide on leave, positioned via `d3.pointer`)
- Public-first data loading + HtmlEmbed prop support when applicable
- Responsive: resizes smoothly; axes and grid legible at small widths
- No console errors; graceful error message on load failures
- File is self‑contained; no globals; lints pass

### 14.3) Prompt modèle (for the agent)
```markdown
You are implementing a self-contained D3 embed fragment.
Name: d3-<type>.html (root class .d3-<type>)
Requirements:
- One root div + scoped style + IIFE script; no globals
- UI in HTML (legend+controls), chart primitives in SVG
- Legend title text exactly “Legend”; swatch 14×14, r=3, 1px border
- If a select toggles metrics, visible label text exactly “Metric”
- Colors via window.ColorPalettes (categorical/sequential/diverging), fallback to CSS variables or Tableau10
- Tooltip: single .d3-tooltip inside container, HTML, positioned via d3.pointer
- Data loading: try `/data/<file>` first, then `./assets/data/<file>`, `../assets/data/<file>`; implement fetchFirstAvailable(paths)
- Read optional HtmlEmbed attributes `data-datafiles` and `data-config` if present (see section 6.1)
- Responsiveness: compute width from container, use ResizeObserver; axis/tick/grid via CSS vars
- Error handling: append small red <pre> inside container on failure
Deliver one .html file with only the required elements.
```

### 15) Example: small bar chart (structure only)
```html
<div class="d3-mini-bar"></div>
<style>
  .d3-mini-bar .bar { stroke: none; }
</style>
<script>
  (() => {
    const ensureD3 = (cb) => {
      if (window.d3 && d3.select) return cb();
      let s = document.getElementById('d3-cdn-script');
      if (!s) { s = document.createElement('script'); s.id='d3-cdn-script'; s.src='https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js'; document.head.appendChild(s); }
      const onReady = () => { if (window.d3 && d3.select) cb(); };
      s.addEventListener('load', onReady, { once:true }); if (window.d3) onReady();
    };
    const bootstrap = () => {
      const scriptEl = document.currentScript;
      let container = scriptEl ? scriptEl.previousElementSibling : null;
      if (!(container && container.classList && container.classList.contains('d3-mini-bar'))){
        const cs = Array.from(document.querySelectorAll('.d3-mini-bar')).filter(el => !(el.dataset && el.dataset.mounted==='true'));
        container = cs[cs.length-1] || null;
      }
      if (!container) return;
      if (container.dataset){ if (container.dataset.mounted==='true') return; container.dataset.mounted='true'; }

      const svg = d3.select(container).append('svg').attr('width','100%').style('display','block');
      const g = svg.append('g');
      let width=800,height=280; const margin={top:16,right:16,bottom:40,left:40};
      const x=d3.scaleBand().padding(0.2), y=d3.scaleLinear();
      const data=[{k:'A',v:3},{k:'B',v:7},{k:'C',v:5}];

      function render(){
        width = container.clientWidth || 800; height = Math.max(220, Math.round(width/3.2));
        svg.attr('width', width).attr('height', height);
        g.attr('transform',`translate(${margin.left},${margin.top})`);
        const iw=width-margin.left-margin.right, ih=height-margin.top-margin.bottom;
        x.domain(data.map(d=>d.k)).range([0,iw]); y.domain([0, d3.max(data,d=>d.v)||1]).range([ih,0]).nice();
        const bars=g.selectAll('rect.bar').data(data);
        bars.join('rect').attr('class','bar').attr('x',d=>x(d.k)).attr('y',d=>y(d.v)).attr('width',x.bandwidth()).attr('height',d=>Math.max(0.5, ih - y(d.v))).attr('fill','var(--primary-color)');
        g.selectAll('.x').data([0]).join('g').attr('class','x').attr('transform',`translate(0,${ih})`).call(d3.axisBottom(x));
        g.selectAll('.y').data([0]).join('g').attr('class','y').call(d3.axisLeft(y).ticks(5));
      }
      render();
      const ro = window.ResizeObserver ? new ResizeObserver(() => render()) : null; if (ro) ro.observe(container); else window.addEventListener('resize', render);
    };
    if (document.readyState==='loading'){ document.addEventListener('DOMContentLoaded', () => ensureD3(bootstrap), { once:true }); } else { ensureD3(bootstrap); }
  })();
</script>
```


