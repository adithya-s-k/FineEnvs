// Shared helpers: DOM, formatting, API, markdown, modal, loading states.
import { icon, FILE_ICON } from "./icons.js";

export const $ = (s, el = document) => el.querySelector(s);
export const $$ = (s, el = document) => [...el.querySelectorAll(s)];
export const fmt = new Intl.NumberFormat("en-US");
export const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
export const vals = (v) => (Array.isArray(v) ? v : v == null ? [] : [v]);
export const DOMAIN_NAME = { code: "Code", webdev: "Webdev", cyber: "Cyber", music: "Music", general: "General" };

export const storage = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode */ } },
};

export async function api(path, opts = {}) {
  const res = await fetch(path, {
    credentials: "same-origin",
    ...opts,
    headers: { ...(opts.body ? { "Content-Type": "application/json" } : {}), ...(opts.headers || {}) },
    body: opts.body && typeof opts.body !== "string" ? JSON.stringify(opts.body) : opts.body,
  });
  if (!res.ok) {
    let msg = `${res.status}`;
    try { msg = (await res.json()).detail || msg; } catch { /* not json */ }
    const err = new Error(msg);
    err.status = res.status;
    throw err;
  }
  return res.json();
}

// Data ships gzipped (Spaces serve static files uncompressed); inflate in the browser.
export async function getJSONgz(url) {
  const res = await fetch(url, { cache: "no-cache" });   // revalidate (a cheap 304), so a redeploy never serves stale data
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const buf = new Uint8Array(await res.arrayBuffer());
  if (buf[0] !== 0x1f || buf[1] !== 0x8b) return JSON.parse(new TextDecoder().decode(buf));
  return new Response(new Blob([buf]).stream().pipeThrough(new DecompressionStream("gzip"))).json();
}

export function money(x) {
  if (x == null || isNaN(x)) return "–";
  if (x === 0) return "$0";
  if (x < 0.01) return "<$0.01";
  return "$" + (x < 10 ? x.toFixed(2) : x.toFixed(0));
}
export function ago(ts) {
  if (!ts) return "";
  const s = Date.now() / 1000 - ts;
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  return new Date(ts * 1000).toLocaleDateString();
}
export function dur(s) {
  if (s == null) return "";
  s = Math.max(0, Math.round(s));
  return s < 60 ? `${s}s` : s < 3600 ? `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s` : `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
}
export function bytes(n) {
  if (n == null) return "";
  return n < 1024 ? `${n} B` : n < 1048576 ? `${(n / 1024).toFixed(0)} KB` : `${(n / 1048576).toFixed(1)} MB`;
}
export function tokensShort(n) {
  if (!n) return "0";
  return n < 1000 ? String(n) : n < 1e6 ? `${(n / 1000).toFixed(n < 1e4 ? 1 : 0)}k` : `${(n / 1e6).toFixed(2)}M`;
}

// Escape first, then headings, lists, fences, inline code, bold, links.
export function md(src) {
  const out = [];
  const lines = String(src ?? "").replace(/\r/g, "").split("\n");
  let i = 0, para = [], list = null;
  const inline = (s) => esc(s)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[\s(])\*([^*\s][^*]*?)\*(?=[\s.,;:!?)]|$)/g, "$1<em>$2</em>")
    .replace(/\[([^\]]+)\]\((https?:[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  const flush = () => {
    if (para.length) { out.push(`<p>${inline(para.join(" "))}</p>`); para = []; }
    if (list) { out.push(`<${list.tag}>${list.items.map((x) => `<li>${inline(x)}</li>`).join("")}</${list.tag}>`); list = null; }
  };
  while (i < lines.length) {
    const line = lines[i];
    if (/^\s*```/.test(line)) {
      flush();
      const buf = [];
      i++;
      while (i < lines.length && !/^\s*```/.test(lines[i])) buf.push(lines[i++]);
      out.push(`<pre><code>${esc(buf.join("\n"))}</code></pre>`);
      i++;
      continue;
    }
    const h = line.match(/^\s*(#{1,4})\s+(.*)$/);
    const ul = line.match(/^\s*[-*•]\s+(.*)$/);
    const ol = line.match(/^\s*\d+[.)]\s+(.*)$/);
    const tr = /^\s*\|.*\|\s*$/.test(line);
    if (h) { flush(); out.push(`<h4>${inline(h[2])}</h4>`); }
    else if (tr) {
      flush();
      const rows = [];
      while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) rows.push(lines[i++]);
      const cells = rows.filter((r) => !/^\s*\|[\s:|-]+\|\s*$/.test(r)).map((r) => r.trim().slice(1, -1).split("|").map((c) => c.trim()));
      out.push(`<div class="tbl"><table>${cells.map((r, n) => `<tr>${r.map((c) => (n ? `<td>${inline(c)}</td>` : `<th>${inline(c)}</th>`)).join("")}</tr>`).join("")}</table></div>`);
      continue;
    } else if (ul || ol) {
      if (para.length) { out.push(`<p>${inline(para.join(" "))}</p>`); para = []; }
      const tag = ul ? "ul" : "ol";
      if (!list || list.tag !== tag) { if (list) flush(); list = { tag, items: [] }; }
      list.items.push((ul || ol)[1]);
    } else if (!line.trim()) flush();
    else { if (list) flush(); para.push(line.trim()); }
    i++;
  }
  flush();
  return out.join("");
}

export function toast(text, ms = 2200) {
  const t = $("#toast");
  t.textContent = text;
  t.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => (t.hidden = true), ms);
}

// A thin bar across the top while something loads. Nested calls are counted, so it finishes once.
let busy = 0, crawl = null;
export const progress = {
  start() {
    const bar = $("#progress i");
    if (!bar) return;
    if (busy++ === 0) {
      clearInterval(crawl);
      bar.style.transition = "none"; bar.style.opacity = "1"; bar.style.width = "0";
      requestAnimationFrame(() => { bar.style.transition = ""; bar.style.width = "30%"; });
      crawl = setInterval(() => { const w = parseFloat(bar.style.width) || 0; if (w < 90) bar.style.width = w + (90 - w) * 0.08 + "%"; }, 250);
    }
  },
  done() {
    const bar = $("#progress i");
    if (!bar || busy === 0 || --busy > 0) return;
    clearInterval(crawl);
    bar.style.width = "100%";
    setTimeout(() => { if (!busy) { bar.style.opacity = "0"; setTimeout(() => { if (!busy) bar.style.width = "0"; }, 300); } }, 180);
  },
  async wrap(p) { this.start(); try { return await p; } finally { this.done(); } },
};

// Skeleton building blocks: widths are percentages, so they read as text rather than as boxes.
export const sk = {
  line: (w = 100, h = 12) => `<span class="sk line" style="width:${w}%;height:${h}px"></span>`,
  lines: (...ws) => ws.map((w) => sk.line(w)).join(""),
  box: (h, extra = "") => `<span class="sk" style="height:${h}px;${extra}"></span>`,
  card: (inner) => `<div class="sk-card">${inner}</div>`,
};

let modalReturn = null;
export function openModal(title, html, { raw, kind } = {}) {
  modalReturn = document.activeElement;
  $("#modal-title").textContent = title;
  $("#modal-icon").innerHTML = kind ? icon(FILE_ICON[kind] || kind, 16, "kind") : "";
  $("#modal-body").innerHTML = html;
  const open = $("#modal-open");
  open.hidden = !raw;
  if (raw) { open.href = raw; open.innerHTML = `${icon("external", 14)}Open original`; }
  $("#modal").hidden = false;
  $("#scrim").hidden = false;
  document.body.classList.add("noscroll");
  $("#modal-close").focus();
  return $("#modal-body");
}
export function closeModal() {
  if ($("#modal").hidden && !$("#dialog-root").innerHTML) return;
  $("#modal").hidden = true;
  $("#scrim").hidden = true;
  $("#modal-body").innerHTML = "";
  $("#dialog-root").innerHTML = "";
  document.body.classList.remove("noscroll");
  if (modalReturn?.focus) modalReturn.focus();
  modalReturn = null;
}
export function openDialog(html) {
  modalReturn = document.activeElement;
  const root = $("#dialog-root");
  root.innerHTML = `<div class="dialog" role="dialog" aria-modal="true">${html}</div>`;
  $("#scrim").hidden = false;
  document.body.classList.add("noscroll");
  return root.firstElementChild;
}

export const spinner = (cls = "") => `<span class="spinner ${cls}" aria-hidden="true"></span>`;
export function emptyState(ic, title, text = "", action = "") {
  return `<div class="empty-state">${icon(ic, 28)}<h3>${esc(title)}</h3>${text ? `<p>${text}</p>` : ""}${action}</div>`;
}

const NUM = /^-?[$€£]?\(?-?[\d,]*\.?\d+\)?%?$/;
export function table(rows, { header = true, max = 200 } = {}) {
  if (!rows || !rows.length) return `<p class="muted">Empty.</p>`;
  const [head, ...body] = header ? rows : [null, ...rows];
  return `<div class="tbl"><table>${head ? `<thead><tr>${head.map((c) => `<th>${esc(c)}</th>`).join("")}</tr></thead>` : ""}
    <tbody>${body.slice(0, max).map((r) => `<tr>${r.map((c) => `<td${NUM.test(String(c ?? "").trim()) && String(c ?? "").trim() ? ' class="num"' : ""}>${esc(c)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
}

export const LIVE_STATUSES = ["queued", "starting", "setup", "running", "verifying"];
const STATUS = { queued: ["Queued", "clock"], starting: ["Starting sandbox", "box"], setup: ["Setting up", "box"], running: ["Agent running", "terminal"],
                 verifying: ["Grading", "scale"], done: ["Done", "check"], failed: ["Failed", "alert"], cancelled: ["Stopped", "stop"], interrupted: ["Interrupted", "alert"] };
export function statusPill(status) {
  const live = LIVE_STATUSES.includes(status);
  const [label, ic] = STATUS[status] || [status, "info"];
  return `<span class="status s-${esc(status)}${live ? " live" : ""}">${live ? '<i class="pulse"></i>' : icon(ic, 12)}${esc(label)}</span>`;
}

export const rewardClass = (r) => (r == null ? "none" : r >= 0.999 ? "full" : r > 0 ? "part" : "zero");
export const rewardText = (r) => (r == null ? "–" : Number(r) % 1 === 0 ? Number(r).toFixed(0) : Number(r).toFixed(2));
export function rewardBadge(reward, status) {
  if (reward == null) return status === "done" ? `<span class="reward none">not scored</span>` : `<span class="reward none">–</span>`;
  return `<span class="reward ${rewardClass(reward)}" title="Reward ${Number(reward).toFixed(3)}">${rewardText(reward)}</span>`;
}

// A spreadsheet-like grid: column letters, row numbers, numbers right-aligned, formulas shown as formulas,
// and runs of empty rows collapsed to one thin row.
const colName = (i) => { let s = ""; i += 1; while (i) { const m = (i - 1) % 26; s = String.fromCharCode(65 + m) + s; i = Math.floor((i - 1) / 26); } return s; };
export function sheetGrid(rows) {
  if (!rows || !rows.length) return `<p class="muted">Empty sheet.</p>`;
  const width = Math.max(...rows.map((r) => r.length));
  let used = 0;
  rows.forEach((r) => r.forEach((c, i) => { if (c !== "" && c != null) used = Math.max(used, i + 1); }));
  const cols = Math.max(1, Math.min(width, used));
  const out = [];
  let gap = 0;
  rows.forEach((r, n) => {
    const empty = !r.slice(0, cols).some((c) => c !== "" && c != null);
    if (empty) { gap++; return; }
    if (gap) { out.push(`<tr class="gap"><th></th><td colspan="${cols}"></td></tr>`); gap = 0; }
    // a lone text cell (a title or a section heading) spills across the empty cells to its right, as in Excel
    const filled = r.slice(0, cols).filter((c) => c !== "" && c != null).length;
    if (filled === 1 && r[0] !== "" && r[0] != null && !String(r[0]).startsWith("=") && cols > 1) {
      out.push(`<tr><th>${n + 1}</th><td class="span" colspan="${cols}">${esc(r[0])}</td></tr>`);
      return;
    }
    out.push(`<tr><th>${n + 1}</th>${Array.from({ length: cols }, (_, i) => {
      const v = r[i] ?? "";
      const s = String(v);
      const cls = s.startsWith("=") ? "fx" : NUM.test(s.trim()) && s.trim() ? "num" : "";
      return `<td class="${cls}"${s.length > 40 ? ` title="${esc(s)}"` : ""}>${cls === "fx" ? `<span>ƒ</span>${esc(s)}` : esc(s)}</td>`;
    }).join("")}</tr>`);
  });
  return `<div class="tbl sheet"><table><thead><tr><th></th>${Array.from({ length: cols }, (_, i) => `<th>${colName(i)}</th>`).join("")}</tr></thead>
    <tbody>${out.join("")}</tbody></table></div>`;
}
