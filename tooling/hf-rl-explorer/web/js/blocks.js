// The blocks an environment's sections are made of (app/envs/contract.py): one renderer for every format, so a Harbor
// task, a MiMo row and a NeMo Gym row are drawn by the same code. Text is escaped; inline Markdown is `code`, **bold**
// and [links](https://...). A `custom` block is drawn by its renderer module (web/js/renderers/<name>.js), loaded
// first with loadRenderers(); it exports render(kind, data, ctx) -> HTML and, optionally, wire(root, ctx).
import { esc, fmt, md } from "./util.js";
import { icon } from "./icons.js";
import { value, object } from "./render.js";
import { codeBlock } from "./files.js";

export function inline(text) {
  return esc(String(text ?? ""))
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\[([^\]]+)\]\((https:\/\/[^)\s]+)\)/g, '<a class="u" href="$2" target="_blank" rel="noopener">$1</a>');
}

const pct = (n, d) => (d ? `${Math.round((100 * n) / d)}%` : "–");
function shares(b) {
  return `${b.rows.map(([label, n]) => `<div>${inline(label)} <span class="faint">${fmt.format(n)} · ${pct(n, b.of)}</span>
    <span class="share" aria-hidden="true"><i style="width:${b.of ? (100 * n) / b.of : 0}%"></i></span></div>`).join("")}
    ${b.note ? `<div class="faint xs">${inline(b.note)}</div>` : ""}`;
}

const cell = (v) => (v && typeof v === "object" && v.type === "shares" ? shares(v) : typeof v === "number" ? esc(fmt.format(v)) : inline(v));

// a section's blocks as HTML; `ctx.files` collects the file-viewer slots for the page to mount, `ctx.paths` (the files
// they list) lets a code block offer to open its file there
export function blocks(list, ctx = { files: [] }) {
  return (list || []).map((b) => {
    switch (b.type) {
      case "kv": return b.rows.length ? `<dl class="kv">${b.rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${cell(v)}</dd>`).join("")}</dl>` : "";
      case "markdown": return b.text?.trim() ? `<div class="prose">${md(b.text)}</div>` : `<p class="muted sm">Empty.</p>`;
      case "code": return `${b.label ? `<div class="sec-label bl-label">${esc(b.label)}${ctx.paths?.has(b.label) ? ` <button class="link sm" type="button" data-open-file="${esc(b.label)}">open in Files</button>` : ""}</div>` : ""}${codeBlock(b.text, b.path || b.label || "", { tall: true })}`;
      // the text is one flex item (the note is a row: icon, text), so its inline code wraps with the words
      case "note": return `<p class="fine bl-note">${icon(b.icon || "info", 13)}<span>${inline(b.text)}</span></p>`;
      case "value": return b.value && typeof b.value === "object" && !Array.isArray(b.value) ? object(b.value, 0) : value(b.value);
      case "messages": return value(b.value, "messages");
      case "steps": return `<ol class="bl-steps">${b.items.map(([n, t]) => `<li><b>${esc(n)}</b> ${inline(t)}</li>`).join("")}</ol>`;
      case "shares": return shares(b);
      case "disclose": return `<details class="bl-disclose"><summary class="disclose">${icon("chevronRight", 14, "chev")}${esc(b.label)}</summary>${blocks(b.blocks, ctx)}</details>`;
      case "files": { ctx.files.push(b); return `<div class="bl-files" data-files="${ctx.files.length - 1}"></div>`; }
      case "stats": return b.rows.length ? `<div class="cm-stats bl-stats">${b.rows.map(([k, v]) => `<div class="stat"><span>${esc(k)}</span><b>${inline(v)}</b></div>`).join("")}</div>` : "";
      case "links": return `<div class="brief-links">${b.items.map(([label, href]) => /^https:\/\//.test(href)
        ? `<a class="btn sm" href="${esc(href)}" target="_blank" rel="noopener">${esc(label)}${icon("external", 13)}</a>` : "").join("")}</div>`;
      case "custom": {
        const r = ctx.renderers?.[b.renderer];
        return r ? `<div class="bl-custom">${r.render(b.kind, b.data, ctx)}</div>` : (b.text ? `<div class="prose">${md(b.text)}</div>` : "");
      }
      default: return "";
    }
  }).join("");
}

// the renderer modules a page's blocks need, by name (web/js/renderers/<name>.js)
export async function loadRenderers(sections) {
  const names = new Set();
  const walk = (list) => (list || []).forEach((b) => { if (b.type === "custom" && /^[a-z0-9-]{1,40}$/.test(b.renderer)) names.add(b.renderer); if (b.type === "disclose") walk(b.blocks); });
  (sections || []).forEach((s) => walk(s.blocks));
  const out = {};
  await Promise.all([...names].map(async (n) => { try { out[n] = await import(`./renderers/${n}.js`); } catch { /* drawn from its text instead */ } }));
  return out;
}
