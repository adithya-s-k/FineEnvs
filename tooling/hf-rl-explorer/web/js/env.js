// An environment, whatever its format: /d/<org>/<name>. Its adapter (app/envs) says how it's read, gives an overview
// (how its tasks are graded, what they run in), its facets and its tasks as cards; this page draws them one way:
//   inline   every task at once (a Harbor index): search and filters run here as you type, with tiles and a map
//   paged    a page at a time (dataset rows): subsets, search and filters run on the server
// A Harbor dataset is indexed the first time it's opened; the page shows the indexing's progress meanwhile.
import { $, api, esc, fmt, ago, sk, emptyState, toast, spinner, nav, setMeta } from "./util.js";
import { icon } from "./icons.js";
import { blocks, loadRenderers } from "./blocks.js";
import { taskHref } from "./task.js";
import { typeset } from "./math.js";
import { agentName, connectPanel, wireConnect } from "./connect.js";

const PAGE = 30;
const compact = new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 });
const enc = (s) => s.split("/").map(encodeURIComponent).join("/");
const envApi = (spec) => `/api/env/${enc(spec)}`;
// indexing, step by step: what each step does, and how much of the whole it is
const STEPS = [["listing", "List the repository", 0.1], ["reading", "Fetch every task's files", 0.7], ["parsing", "Read every task", 0.15],
               ["saving", "Save the index", 0.05]];
// difficulty reads easy to hard, not by count
const LEVEL = ["trivial", "very easy", "easy", "medium", "moderate", "intermediate", "hard", "difficult", "very hard", "expert", "extreme"];
const level = (v) => { const i = LEVEL.indexOf(String(v).toLowerCase()); return i < 0 ? (Number.isFinite(+v) ? +v : 99) : i; };

let poll = null, ticker = null, root = null;
let st = {};
let matches = [], shown = 0;
const expanded = new Set(), facetOpen = new Map();

export function unmount() { clearTimeout(poll); clearInterval(ticker); scrollWatch?.disconnect(); root = null; document.body.classList.remove("noscroll"); }

// the URL's state: ?s=<subset>&q=…&f={"facet":["value",…]}&t=<tile>&o=<offset>; older links (?c=&s= of the rows
// page, f=k:v|v;k2:v of the Harbor page) still open where they pointed
function readURL(qs) {
  const sel = {};
  const f = qs.get("f") || "";
  if (f.startsWith("{")) {
    try { Object.entries(JSON.parse(f)).forEach(([k, v]) => { sel[k] = new Set((Array.isArray(v) ? v : [v]).map(String)); }); } catch { /* a broken link: no filters */ }
  } else {
    f.split(";").filter(Boolean).forEach((pair) => {
      const i = pair.indexOf(":");
      if (i > 0) sel[pair.slice(0, i)] = new Set(pair.slice(i + 1).split("|").map(decodeURIComponent));
    });
  }
  const c = qs.get("c"), s = qs.get("s");
  return { q: qs.get("q") || "", sel, tile: qs.get("t") ?? qs.get("d") ?? null, offset: Math.max(0, +qs.get("o") || 0),
           subset: c ? (c === "default" ? s : `${c}/${s}`) : s, open: qs.get("open") || (qs.get("rewards") ? "rewards" : null) };
}
function writeURL() {
  const p = new URLSearchParams();
  if (st.subset && st.sum?.subsets?.length > 1) p.set("s", st.subset);
  if (st.q) p.set("q", st.q);
  if (st.tile != null) p.set("t", st.tile);
  const sel = Object.fromEntries(Object.entries(st.sel).filter(([, s]) => s.size).map(([k, s]) => [k, [...s]]));
  if (Object.keys(sel).length) p.set("f", JSON.stringify(sel));
  if (st.offset) p.set("o", st.offset);
  history.replaceState(null, "", `/d/${enc(st.spec)}${p.toString() ? `?${p}` : ""}`);
}

// ── page ─────────────────────────────────────────────────────────────────────
export async function mount(el, { spec, qs }) {
  root = el;
  wireConnect(el);
  st = { spec, sum: null, cards: null, partial: false, ...readURL(qs) };
  el.innerHTML = `<div class="wrap page fade-in" aria-busy="true"><div id="head">${sk.head()}</div>
    <div id="body"><section class="panel rw-how"><div class="panel-b sk-in">${sk.line(16, 14)}${sk.lines(92, 60)}</div></section>${sk.list()}</div></div>`;
  await load(el);
}

async function load(el) {
  let r;
  try { r = await api(`${envApi(st.spec)}${st.subset ? `?subset=${encodeURIComponent(st.subset)}` : ""}`, { retry: 2 }); } catch (e) {
    if (root !== el) return;
    if (e.status === 429) {   // others are being indexed: wait for a slot
      $("#body", el).innerHTML = `<section class="panel ix"><div class="panel-b"><div class="ix-top"><div><h2>Waiting for a slot</h2>
        <p class="muted sm">${esc(e.message)}. This page starts indexing by itself as soon as one frees up.</p></div></div><div class="ix-bar busy"><i style="width:3%"></i></div></div></section>`;
      poll = setTimeout(() => load(el), 5000);
      return;
    }
    if (st.subset && e.status !== 404 && !st.retried) { st.retried = true; st.subset = null; return load(el); }   // a subset that's gone
    $("#head", el).innerHTML = crumbs();
    $("#body", el).innerHTML = emptyState("alert", "Couldn't open this environment", esc(e.message),
      `<a class="btn" href="https://huggingface.co/datasets/${enc(st.spec)}" target="_blank" rel="noopener">${icon("external", 14)}See it on the Hub</a>`);
    return;
  }
  if (root !== el) return;
  st.sum = r; st.subset = r.subset || null;
  st.renderers = await loadRenderers(r.overview || []);
  if (root !== el) return;
  renderHead(el);
  if (r.state === "indexing") { renderProgress(el, r.progress || {}); poll = setTimeout(() => load(el), 1000); return; }
  clearInterval(ticker);
  if (r.state === "error") {
    $("#body", el).innerHTML = `<div class="note-box err">${icon("alert")}<span>Couldn't read this environment: ${esc(r.progress?.error || r.error || "")}</span></div>`;
    return;
  }
  if (r.inline) {
    // a big Harbor dataset: its tasks are known, their files still being read; titles and filters fill in when done
    if (r.progress && r.progress.total) {
      if (!st.partial) { await loadCards(); if (root !== el) return; renderBody(el); st.partial = true; }
      partialNote(el, r.progress);
      poll = setTimeout(() => load(el), 4000);
      return;
    }
    const was = st.partial;
    st.partial = false;
    await loadCards();
    if (root !== el) return;
    renderBody(el);
    if (was) toast("Every task's details are in");
    return;
  }
  renderBody(el);
}

async function loadCards() {
  try { st.cards = (await api(`${envApi(st.spec)}/tasks?all=1`)).cards; } catch (e) { st.cards = []; toast(e.message); }
  const n = st.cards.length;
  st.cards.forEach((c) => { c._blob = `${c.title} ${c.ref} ${c.id || ""} ${c.brief || ""} ${c.text || ""}`.toLowerCase(); });
  // what every task shares says nothing on a card
  const count = new Map();
  st.cards.forEach((c) => c.chips.forEach((x) => count.set(x, (count.get(x) || 0) + 1)));
  st.shared = new Set(n > 1 ? [...count].filter(([, k]) => k === n).map(([x]) => x) : []);
}

// ── head ─────────────────────────────────────────────────────────────────────
function crumbs() {
  const name = st.spec.split("/")[1], c = st.sum?.collection;
  return `<nav class="crumbs" aria-label="Breadcrumb"><a href="/">Environments</a>${icon("chevronRight", 13)}
    ${c ? `<a href="/?c=${esc(c.id)}">${esc(c.group)}</a>${icon("chevronRight", 13)}` : ""}<span>${esc(name)}</span></nav>`;
}

function renderHead(el) {
  const s = st.sum, i = s.info, c = s.collection;
  const [org, name] = st.spec.split("/");
  const norm = (x) => String(x || "").toLowerCase().replace(/[^a-z0-9]/g, "");
  const lede = i.brief || (i.heading && !norm(i.heading).includes(norm(name)) ? i.heading : "");   // not the name again
  const n = s.subsets?.length > 1 ? null : s.total;
  setMeta({ title: `${name} · ${s.env.framework}`, description: `${st.spec}: ${s.env.framework} environment on Hugging Face${s.total ? ` with ${fmt.format(s.total)} tasks` : ""}. ${lede || s.how?.about || ""}` });
  $("#head", el).innerHTML = `${crumbs()}
    <header class="tp-head ds-head">
      <div class="kick">${c ? `<span class="badge" style="--dc:var(--c-${c.color || c.id})">${icon(c.icon, 13)}${esc(c.group)}</span>` : ""}
        <span class="chip fw">${esc(s.env.framework)}</span>${(i.badges || []).map((b) => `<span class="chip">${esc(b)}</span>`).join("")}</div>
      <h1><span class="org">${esc(org)}/</span>${esc(name)}</h1>
      ${lede ? `<p class="lede">${esc(lede)}</p>` : ""}
      <div class="facts">
        ${n != null ? `<span>${icon("list", 14)}<b>${fmt.format(n)}</b> task${n === 1 ? "" : "s"}</span>` : s.subsets?.length > 1 ? `<span>${icon("columns", 14)}<b>${fmt.format(s.subsets.length)}</b> subsets</span>` : ""}
        <span title="downloads in the last month">${icon("download", 14)}<b>${compact.format(i.downloads || 0)}</b> downloads</span>
        <span>${icon("heart", 14)}<b>${fmt.format(i.likes || 0)}</b> likes</span>
        ${i.license ? `<span>${icon("scale", 14)}${esc(i.license)}</span>` : ""}
        ${i.updated ? `<span>${icon("clock", 14)}Updated ${esc(ago(Date.parse(i.updated) / 1000))}</span>` : ""}
        <span title="the revision this page reads">${icon("flag", 14)}<code>${esc((i.sha || "").slice(0, 8))}</code></span>
        <a class="u" href="https://huggingface.co/datasets/${enc(st.spec)}" target="_blank" rel="noopener">${icon("external", 14)}On the Hub</a>
        <button class="link" type="button" id="copy">${icon("link", 14)}Copy link</button>
      </div></header>`;
  $("#copy", el).addEventListener("click", () => navigator.clipboard?.writeText(location.href).then(() => toast("Link copied")));
}

// ── indexing progress: a timeline of the steps, live counts, elapsed time and what's left ─
const secsText = (s) => (s < 1 ? "<1 s" : s < 60 ? `${Math.round(s)} s` : `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`);
function renderProgress(el, r) {
  const body = $("#body", el);
  if (!$(".ix", body)) {
    body.innerHTML = `<section class="panel ix"><div class="panel-b">
      <div class="ix-top"><div><h2>Indexing this dataset</h2><p class="muted sm">The first visit reads every task once; after that it opens instantly, for everyone.</p></div>
        <div class="ix-time"><b id="ix-pct">0%</b><span id="ix-eta"></span></div></div>
      <div class="ix-bar"><i id="ix-fill"></i></div>
      <ol class="ix-steps">${STEPS.map(([k, label]) => `<li data-step="${k}"><span class="mk"></span><div><b>${label}</b><span class="d"></span></div><span class="t"></span></li>`).join("")}</ol>
      </div></section>`;
  }
  const skew = Date.now() / 1000 - (r.now || Date.now() / 1000);   // server clock to ours
  const t = r.t || {}, at = Math.max(0, STEPS.findIndex(([k]) => k === r.state));
  const frac = r.total ? Math.min(1, (r.done || 0) / r.total) : 0;
  const pct = STEPS.slice(0, at).reduce((n, [, , w]) => n + w, 0) + (STEPS[at]?.[2] || 0) * (r.state === "listing" ? 0.5 : frac);
  $("#ix-pct", body).textContent = `${Math.round(pct * 100)}%`;
  $("#ix-fill", body).style.width = `${Math.max(3, pct * 100)}%`;
  $(".ix-bar", body).classList.toggle("busy", !r.state || r.state === "listing" || r.state === "saving");
  const detail = {
    listing: () => (r.files ? `${fmt.format(r.files)} files listed${r.tasks ? `, ${fmt.format(r.tasks)} tasks found` : ""}` : "Asking the Hub for every file"),
    reading: () => (r.total ? `${fmt.format(r.done || 0)} of ${fmt.format(r.total)} files: task.toml, instruction, tests, Dockerfile` : ""),
    parsing: () => (r.total ? `${fmt.format(r.done || 0)} of ${fmt.format(r.total)} tasks` : ""),
    saving: () => "Kept for the next visit",
  };
  const done = { listing: () => `${fmt.format(r.files || 0)} files, ${fmt.format(r.tasks || 0)} tasks`, reading: () => "Fetched", parsing: () => `${fmt.format(r.tasks || 0)} tasks read`, saving: () => "Saved" };
  STEPS.forEach(([k], i) => {
    const li = $(`[data-step="${k}"]`, body);
    li.className = i < at ? "ok" : i === at ? "on" : "off";
    li.querySelector(".mk").innerHTML = i < at ? icon("check", 12) : "";
    li.querySelector(".d").textContent = i < at ? done[k]() : i === at ? detail[k]() : "";
    const next = STEPS[i + 1] && t[STEPS[i + 1][0]];
    li.querySelector(".t").dataset.start = t[k] ? t[k] + skew : "";
    li.querySelector(".t").dataset.end = i < at && next ? next + skew : "";
  });
  const start = t[r.state] ? t[r.state] + skew : null;
  const rate = start && r.done ? r.done / (Date.now() / 1000 - start) : 0;
  $("#ix-eta", body).textContent = rate && r.total > r.done ? `about ${secsText((r.total - r.done) / rate)} left in this step` : "";
  clearInterval(ticker);
  const tick = () => body.querySelectorAll(".ix-steps .t").forEach((x) => {
    const a = +x.dataset.start, b = +x.dataset.end;
    x.textContent = a ? secsText((b || Date.now() / 1000) - a) : "";
  });
  tick();
  ticker = setInterval(tick, 250);
}

// a slim strip above the tasks while a walked dataset's files are read
function partialNote(el, p) {
  let box = $("#ix-partial", el);
  if (!box) { $("#body", el).insertAdjacentHTML("afterbegin", `<div class="ix-partial" id="ix-partial"></div>`); box = $("#ix-partial", el); }
  const pct = p.total ? Math.min(100, Math.round((100 * (p.done || 0)) / p.total)) : 0;
  box.innerHTML = `${spinner("xs")}<span><b>Reading every task's files</b> ${p.total ? `${fmt.format(p.done || 0)} of ${fmt.format(p.total)}` : "…"}. A big dataset: its tasks are listed already, and titles, grading and filters fill in when this is done. Any task opens now.</span>
    <span class="ix-partial-bar"><i style="width:${pct}%"></i></span>`;
}

// ── body ─────────────────────────────────────────────────────────────────────
const ROLE = [["task", "The task"], ["id", "Its id"], ["title", "Its title"], ["grading", "Graded by"], ["environment", "Runs in"], ["answer", "Withheld"]];
function how(s) {
  const h = s.how || {}, roles = h.roles || {};
  const cols = (v) => (Array.isArray(v) ? v : [v]).map((x) => `<code>${esc(x)}</code>`).join(" ");
  const rows = ROLE.filter(([k]) => roles[k] && (!Array.isArray(roles[k]) || roles[k].length)).map(([k, label]) => `<dt>${label}</dt><dd>${cols(roles[k])}</dd>`).join("");
  const via = s.inline ? "indexed from its files" : s.source === "files" ? "straight from its files" : "through the Hub's dataset viewer";
  return `<section class="panel rw-how"><div class="panel-h"><h2>${icon("table", 15)}How it's read</h2><span class="aside">${via}</span></div>
    <div class="panel-b"><p class="rw-about"><b>${esc(h.name || s.env.name)}.</b> ${esc(h.about || s.env.about || "")}</p>
      ${rows ? `<dl class="kv rw-roles">${rows}</dl>` : ""}
      <details class="bl-disclose cn-wrap"><summary class="disclose">${icon("chevronRight", 14, "chev")}Browse it from a coding agent, over MCP</summary>
        ${connectPanel({ url: `${location.origin}/mcp/d/${st.spec}`, name: agentName(st.spec),
          note: "Tools: <code>describe</code>, <code>list_tasks</code> (search and filters), <code>get_task</code> (a task in full, as text), <code>read_file</code> and <code>random_task</code>. Answers are withheld, as they are here. No sign-in needed; public datasets only." })}
      </details></div></section>`;
}

function renderBody(el) {
  const s = st.sum;
  if (s.inline && !st.cards.length) {
    $("#body", el).innerHTML = `<section class="panel"><div class="panel-b">${emptyState("archive", "No tasks to show", esc(s.note || "Nothing in this dataset reads as a task."))}</div></section>`;
    return;
  }
  st.facets = (s.facets || []).filter((f) => f.key);
  st.noun = s.noun || ["task", "tasks"];
  const nvals = (key) => (s.inline ? counts(st.cards, key).length : (st.facets.find((f) => f.key === key)?.values || []).length);
  // tiles: the adapter's (with notes and colours), or, for an inline list, a facet with a handful of values
  const tkey = s.tiles?.key || s.tile;
  st.tileKey = s.inline && tkey && nvals(tkey) >= 2 ? tkey : null;
  st.tileMeta = new Map((s.tiles?.items || []).map((t) => [t.value, t]));
  if (st.tile != null && !(st.tileKey && counts(st.cards, st.tileKey).some(([v]) => v === st.tile))) st.tile = null;
  // the map: by each tile's own facet (with tiles), the adapter's facet, or the first facet with enough values
  st.mapBy = s.map?.by_tile && Object.keys(s.map.by_tile).length ? s.map : null;
  const mapOrder = [typeof s.map === "string" ? s.map : s.map?.key, "category", "group", ...st.facets.map((f) => f.key).filter((k) => k.startsWith("meta:")), "tags", "difficulty"].filter(Boolean);
  st.mapKey = s.inline && !st.mapBy ? mapOrder.find((k) => k !== st.tileKey && st.facets.some((f) => f.key === k) && nvals(k) >= 5) || null : null;
  if (s.inline && s.order === "shuffle") { const seed = Math.random(); st.cards.forEach((x) => (x._rand = hash(x.ref + seed))); st.cards.sort((a, b) => a._rand - b._rand); }
  buildSearch();
  const color = s.collection ? `var(--c-${s.collection.color || s.collection.id})` : "var(--text)";
  const subsets = !s.inline && s.subsets?.length > 1;
  const panel = (sec) => sec.collapsed
    ? `<details class="panel ov-fold" id="ov-${esc(sec.id)}" ${st.open === sec.id ? "open" : ""}><summary class="panel-h">${icon("chevronRight", 15, "chev")}<h2>${icon(sec.icon || "list", 15)}${esc(sec.title)}</h2>${sec.note ? `<span class="aside">${esc(sec.note)}</span>` : ""}</summary>
        <div class="panel-b">${blocks(sec.blocks, { files: [], renderers: st.renderers, spec: st.spec })}</div></details>`
    : `<section class="panel"><div class="panel-h"><h2>${icon(sec.icon || "list", 15)}${esc(sec.title)}</h2>${sec.note ? `<span class="aside">${esc(sec.note)}</span>` : ""}</div>
        <div class="panel-b">${blocks(sec.blocks, { files: [], renderers: st.renderers, spec: st.spec })}</div></section>`;
  const open = (s.overview || []).filter((x) => !x.collapsed), folded = (s.overview || []).filter((x) => x.collapsed);
  const what = st.noun[1];
  $("#body", el).innerHTML = `${how(s)}
    ${open.length ? `<div class="ds-facts" style="--dc:${color}">${open.map(panel).join("")}</div>` : ""}
    ${folded.map(panel).join("")}
    ${st.tileKey ? `<div class="domains" id="tiles" role="tablist" aria-label="${esc(label(st.tileKey))}"></div>` : ""}
    ${st.mapKey || st.mapBy ? `<details class="panel map-card" id="map-card" ${matchMedia("(max-width: 760px)").matches ? "" : "open"}>
      <summary>${icon("chevronRight", 15, "chev")}<h2 id="map-title"></h2><span class="hint">Area is the number of ${esc(what)}. Click a block to filter.</span></summary>
      <div id="map" class="map"></div></details>` : ""}
    <div class="toolbar" id="toolbar">
      ${subsets ? `<label class="tb-sub"><span class="sr-only">Subset</span><select class="input" id="subset" aria-label="Subset">${s.subsets.map((x) =>
        `<option value="${esc(x.id)}" ${x.id === st.subset ? "selected" : ""}>${esc(x.label)}</option>`).join("")}</select></label>` : ""}
      <div class="search">${icon("search", 16)}<input id="q" data-search type="search" placeholder="${s.search === false ? "Search isn't available for this dataset" : s.inline ? `Search titles, briefs, categories…` : `Search these ${esc(what)}…`}"
        ${s.search === false ? "disabled" : ""} autocomplete="off" spellcheck="false" aria-label="Search ${esc(what)}"><kbd>/</kbd></div>
      <button class="btn" id="random" type="button" title="Open a random one${s.inline ? " from the current results" : ""}">${icon("shuffle", 15)}<span class="long">Random</span></button>
      ${st.facets.length ? `<button class="btn only-mobile" id="open-filters" type="button">${icon("filter", 15)}Filters<span id="nfilt"></span></button>` : `<span id="nfilt" hidden></span>`}
    </div>
    <div class="layout ${st.facets.length ? "" : "bare"}">
      <div class="sheet-scrim" id="sheet-scrim" hidden></div>
      <aside class="filters" id="filters" aria-label="Filters">
        <div class="sheet-head"><b>Filters</b><button class="icon-btn" id="close-filters" type="button" aria-label="Close filters">${icon("x", 18)}</button></div>
        <div class="filters-head"><span>Filters</span><button class="link" id="clear" type="button">Clear all</button></div>
        <div id="facets"></div>
        <div class="sheet-foot"><button class="btn" id="clear3" type="button">Clear</button><button class="btn primary grow" id="show-results" type="button">Show results</button></div>
      </aside>
      <section class="results" aria-live="polite">
        <div class="results-head"><span class="count" id="count"></span><span class="order" id="order-note"></span><div class="active" id="active"></div>
          ${s.inline ? "" : `<span class="grow"></span><span class="tk-pager"><button class="icon-btn sm" type="button" id="prev" aria-label="Previous page">${icon("chevronRight", 15, "flip")}</button><span id="range"></span><button class="icon-btn sm" type="button" id="next" aria-label="Next page">${icon("chevronRight", 15)}</button></span>`}</div>
        <div id="note"></div>
        <ol class="list" id="list"></ol>
        <div class="sentinel" id="sentinel"></div>
        <div class="more-row" id="more-row" hidden><span class="muted sm" id="more-count"></span><button class="btn" id="more" type="button">Show more</button></div>
        <div id="empty" hidden>${emptyState("search", `No ${esc(what)} match`, "Try fewer filters or a different word.", `<button class="btn" id="clear2" type="button">Clear search and filters</button>`)}</div>
      </section>
    </div>`;
  Object.values(st.renderers || {}).forEach((r) => r.wire?.(el, { spec: st.spec }));
  if (st.open) requestAnimationFrame(() => $(`#ov-${CSS.escape(st.open)}`, el)?.scrollIntoView({ block: "start" }));
  $("#q", el).value = st.q;
  wire(el);
  run();
}

function hash(s) { let h = 2166136261; for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619); return h >>> 0; }

// ── search: fuzzy and ranked when MiniSearch is there (prefix, a typo or two, title first), plain words otherwise ─
const CJK = /[㐀-鿿豈-﫿]/;
let mini = null, hits = null;
function buildSearch() {
  mini = null;
  if (!st.sum.inline || !window.MiniSearch) return;
  mini = new window.MiniSearch({ idField: "ref", fields: ["title", "brief", "text"], storeFields: [],
    searchOptions: { boost: { title: 3, text: 2 }, prefix: true, fuzzy: 0.15, combineWith: "AND" } });
  mini.addAll(st.cards);
}
function search() {
  const q = st.q.trim();
  if (!q) { hits = null; return null; }
  if (!mini || CJK.test(q)) {
    const words = q.toLowerCase().split(/\s+/);
    const found = st.cards.filter((x) => words.every((w) => x._blob.includes(w)));
    hits = new Set(found.map((x) => x.ref));
    return null;
  }
  const res = mini.search(q);
  hits = new Set(res.map((r) => r.id));
  return new Map(res.map((r) => [r.id, r.score]));
}
const terms = () => (st.q.trim() ? st.q.trim().toLowerCase().split(/\s+/).filter((t) => t.length > 1) : []);
function highlight(text) {
  const t = terms();
  if (!t.length) return esc(text);
  const re = new RegExp(`(${t.map((x) => x.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})`, "gi");
  return esc(text).replace(re, "<mark>$1</mark>");
}

const label = (key) => st.facets.find((f) => f.key === key)?.label || key;
const order = (key) => st.facets.find((f) => f.key === key)?.order || "count";
// a facet shows when its scope allows: everywhere, with no tile picked ("*"), or with one of its tiles picked
const inScope = (f) => !f.scope || (st.tile == null ? f.scope.includes("*") : f.scope.includes(st.tile));
function counts(cards, key) {
  const c = new Map();
  cards.forEach((x) => (x.facets[key] || []).forEach((v) => c.set(v, (c.get(v) || 0) + 1)));
  return [...c.entries()].sort(order(key) === "level" ? (a, b) => level(a[0]) - level(b[0]) || String(a[0]).localeCompare(String(b[0]))
    : (a, b) => b[1] - a[1] || String(a[0]).localeCompare(String(b[0])));
}
function passes(x, skipKey, ignoreTile = false) {
  if (!ignoreTile && st.tile != null && !(x.facets[st.tileKey] || []).includes(st.tile)) return false;
  if (hits && !hits.has(x.ref)) return false;
  for (const [k, set] of Object.entries(st.sel)) {
    if (k === skipKey || !set.size) continue;
    if (!(x.facets[k] || []).some((v) => set.has(v))) return false;
  }
  return true;
}

// one pass: the matches (inline) or the page (paged), then everything that shows them
let loading = 0, autoLoaded = 0;
async function run() {
  const nf = Object.values(st.sel).reduce((n, s) => n + s.size, 0);
  $("#nfilt", root).textContent = nf ? ` (${nf})` : "";
  writeURL();
  if (st.sum.inline) {
    const visible = new Set(st.facets.filter(inScope).map((f) => f.key));
    Object.keys(st.sel).forEach((k) => { if (!visible.has(k)) delete st.sel[k]; });   // a facet of a tile no longer picked
    const scores = search();
    matches = st.cards.filter((x) => passes(x));
    if (scores) matches.sort((a, b) => (scores.get(b.ref) ?? 0) - (scores.get(a.ref) ?? 0));
    $("#order-note", root).innerHTML = scores ? "best match first" : st.sum.order === "shuffle" ? `${icon("shuffle", 13)}shuffled each visit` : "";
    renderTiles(); renderFacets(); renderActive(matches.length); renderMap(); renderList(true);
    return;
  }
  renderFacets();
  const token = ++loading, box = $("#list", root);
  box.hidden = false; $("#empty", root).hidden = true;
  box.innerHTML = sk.cards(6);
  const p = new URLSearchParams({ offset: st.offset });
  if (st.subset) p.set("subset", st.subset);
  if (st.q) p.set("q", st.q);
  const sel = Object.fromEntries(Object.entries(st.sel).filter(([, s]) => s.size).map(([k, s]) => [k, [...s]]));
  if (Object.keys(sel).length) p.set("f", JSON.stringify(sel));
  let r;
  const slow = setTimeout(() => { if (token === loading && root) $("#note", root).innerHTML = `<div class="note-box">${spinner("xs")}<span>Still working: the first search or filter of a dataset can take up to a minute while its rows are read. After that it's quick.</span></div>`; }, 4000);
  try { r = await api(`${envApi(st.spec)}/tasks?${p}`, { retry: 1 }); } catch (e) {
    clearTimeout(slow);
    if (token !== loading || !root) return;
    $("#note", root).innerHTML = "";
    box.innerHTML = `<li class="rw-wide">${e.status === 409 ? `<div class="note-box">${spinner("xs")}<span>${esc(e.message)}</span></div>` : `<div class="note-box err">${icon("alert")}<span>${esc(e.message)}</span></div>`}</li>`;
    if (e.status === 409) setTimeout(() => token === loading && root && run(), 8000);
    return;
  }
  clearTimeout(slow);
  if (token !== loading || !root) return;
  st.page = r;
  $("#note", root).innerHTML = r.note ? `<div class="note-box">${icon("info")}<span>${esc(r.note)}</span></div>` : "";
  renderActive(r.total, r.total == null ? r.cards.length : null);
  $("#range", root).textContent = r.cards.length ? `${fmt.format(r.offset + 1)}–${fmt.format(r.offset + r.cards.length)}` : "";
  $("#prev", root).disabled = r.offset === 0;
  $("#next", root).disabled = r.cards.length < r.page || (r.total != null && r.offset + r.page >= r.total);
  box.innerHTML = r.cards.map(card).join("");
  box.hidden = !r.cards.length;
  $("#empty", root).hidden = r.cards.length > 0;
}

function renderTiles() {
  const el = $("#tiles", root);
  if (!el) return;
  const all = st.cards, key = st.tileKey, t = st.sum.tiles || {};
  const filtered = hits || Object.keys(st.sel).length;
  const values = st.tileMeta.size ? [...st.tileMeta.keys()] : counts(all, key).map(([v]) => v);
  const tiles = [null, ...values];
  const color = (v) => (v != null && st.tileMeta.get(v)?.color ? `var(--c-${st.tileMeta.get(v).color})` : st.sum.collection ? `var(--c-${st.sum.collection.color || st.sum.collection.id})` : "var(--text)");
  el.style.gridTemplateColumns = `repeat(${Math.min(6, tiles.length)}, minmax(0, 1fr))`;
  el.innerHTML = tiles.map((v) => {
    const meta = v == null ? null : st.tileMeta.get(v);
    const mine = v == null ? all : all.filter((x) => (x.facets[key] || []).includes(v));
    const hit = filtered ? mine.filter((x) => passes(x, null, true)).length : null;
    const graded = [...new Set(mine.flatMap((x) => x.facets.graded || []))].map((g) => g.toLowerCase());
    const note = v == null ? (t.all_note || `${label(key)} below`) : meta?.note || (graded.length ? `Graded by ${graded.join(", ")}` : "");
    return `<button class="dom" role="tab" data-tile="${v == null ? "" : esc(v)}" aria-selected="${st.tile === v}" style="--dc:${color(v)}" ${meta?.title ? `title="${esc(meta.title)}"` : ""}>
      <span class="name">${v == null ? icon("grid", 14) : meta?.icon ? icon(meta.icon, 14) : ""}${esc(v == null ? t.all_label || "All tasks" : v)}</span>
      <span class="n">${fmt.format(hit ?? mine.length)}${hit != null ? `<span class="of"> / ${fmt.format(mine.length)}</span>` : ""}</span>
      <span class="ver">${esc(note)}</span></button>`;
  }).join("");
}

function renderFacets() {
  const inline = st.sum.inline;
  const list = st.facets.filter(inScope);
  $("#facets", root).innerHTML = list.map((f) => {
    const key = f.key;
    let items;
    if (inline) {
      const pool = st.cards.filter((x) => passes(x, key));
      items = counts(pool, key);
      // a value every task carries (the dataset's own tag, say) tells nothing apart
      if (key === "tags") items = items.filter(([v, c]) => c < pool.length || st.sel[key]?.has(v));
    } else {
      items = (f.values || []).map(([v, c]) => [String(v), c]);
      if (f.order === "level") items.sort((a, b) => level(a[0]) - level(b[0]) || a[0].localeCompare(b[0]));
    }
    const sel = st.sel[key] || new Set();
    sel.forEach((v) => { if (!items.some(([x]) => x === v)) items.push([v, 0]); });
    if (items.length <= 1 && !sel.size) return "";
    const limit = expanded.has(key) ? Infinity : matchMedia("(max-width: 760px)").matches ? 5 : 8;
    const hidden = Math.max(0, items.length - limit);
    const open = facetOpen.has(key) ? facetOpen.get(key) : (!matchMedia("(max-width: 760px)").matches || sel.size > 0);
    return `<details class="facet" data-facet="${esc(key)}" ${open ? "open" : ""}><summary><h3>${esc(f.label)}${f.sampled ? ' <em class="faint" title="counted from the first rows">≈</em>' : ""}</h3>${sel.size ? `<span class="fsel">${sel.size} selected</span>` : ""}${icon("chevronDown", 14, "chev")}</summary>
      ${items.slice(0, limit).map(([v, c]) => `<button class="opt ${c || !inline ? "" : "zero"}" data-k="${esc(key)}" data-v="${esc(v)}" aria-pressed="${sel.has(v)}">
        <span class="box">${sel.has(v) ? icon("check", 11) : ""}</span><span class="lab" title="${esc(v)}">${esc(v)}</span><span class="c">${c ? fmt.format(c) : ""}</span></button>`).join("")}
      ${hidden || expanded.has(key) ? `<button class="link more" data-more="${esc(key)}">${icon(expanded.has(key) ? "chevronDown" : "chevronRight", 13)}${expanded.has(key) ? "Show fewer" : `Show ${hidden} more`}</button>` : ""}</details>`;
  }).join("") || `<p class="fine">${st.facets.length ? "Every one has the same values here: nothing to filter by." : "This dataset has nothing to filter by."}</p>`;
}

function toggle(key, v) {
  if (key === st.tileKey) { st.tile = st.tile === v ? null : v; st.sel = {}; run(); return; }   // a tile's facet is the tile
  const set = (st.sel[key] ||= new Set());
  set.has(v) ? set.delete(v) : set.add(v);
  if (!set.size) delete st.sel[key];
  st.offset = 0;
  run();
}

function renderActive(n, atLeast = null) {
  const pills = [];
  if (st.tile != null) pills.push(`<button class="pill" data-tile=""><span>${esc(label(st.tileKey))}</span>${esc(st.tile)}${icon("x", 13)}</button>`);
  Object.entries(st.sel).forEach(([k, set]) => set.forEach((v) =>
    pills.push(`<button class="pill" data-k="${esc(k)}" data-v="${esc(v)}"><span>${esc(label(k))}</span>${esc(v)}${icon("x", 13)}</button>`)));
  $("#active", root).innerHTML = pills.join("");
  const filtered = st.q || pills.length;
  const [one, many] = st.noun;
  $("#count", root).innerHTML = n == null ? `${fmt.format(atLeast || 0)}+ ${esc(many)} <span class="faint">(still counting)</span>`
    : `<b>${fmt.format(n)}</b> ${filtered && !st.sum.inline ? "matching " : ""}${esc(n === 1 ? one : many)}`;
  const sr = $("#show-results", root);
  if (sr) sr.textContent = n == null ? "Show results" : `Show ${fmt.format(n)} ${n === 1 ? one : many}`;
}

// ── map ──────────────────────────────────────────────────────────────────────
// By one facet; or, with per-tile facets and no tile picked, each tile's top values side by side in its colour.
const LEFTOVER = /^(Other|Unknown|unknown|None named|Unspecified|Unrated)$/;
const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
function mapData() {
  const top = (sorted, n) => {
    const head = sorted.slice(0, n), rest = sorted.slice(n).reduce((a, [, c]) => a + c, 0);
    const kids = head.map(([name, value], rank) => ({ name, value, rank, other: LEFTOVER.test(name) }));
    if (rest) kids.push({ name: `${sorted.length - head.length} more`, value: rest, rank: head.length, other: true, rest: true });
    return kids;
  };
  if (st.mapBy) {
    if (st.tile != null) {
      const key = st.mapBy.by_tile[st.tile];
      return { key, title: `${st.tile}, by ${label(key).toLowerCase()}`, children: [{ name: st.tile, tile: st.tile, key, children: top(counts(st.cards.filter((x) => passes(x, key)), key), 40) }] };
    }
    const children = Object.entries(st.mapBy.by_tile).map(([tile, key]) => {
      const pool = st.cards.filter((x) => (x.facets[st.tileKey] || []).includes(tile) && passes(x));
      return { name: tile, tile, key, children: top(counts(pool, key), st.mapBy.top?.[tile] || 9).map((k) => ({ ...k, tile, key })) };
    }).filter((d) => d.children.length);
    return { title: `${st.noun[1][0].toUpperCase()}${st.noun[1].slice(1)}, by ${label(st.tileKey).toLowerCase()}`, children };
  }
  const key = st.mapKey;
  return { key, title: `${st.noun[1][0].toUpperCase()}${st.noun[1].slice(1)}, by ${label(key).toLowerCase()}`, children: [{ name: "all", key, children: top(counts(st.cards.filter((x) => passes(x, key)), key), 14) }] };
}
function renderMap() {
  const el = root && $("#map", root);
  if (!el || !window.d3) return;
  const W = el.clientWidth, H = el.clientHeight;
  if (!W || !H) return;
  const data = mapData();
  $("#map-title", root).textContent = data.title;
  el.innerHTML = "";
  const leaves = data.children.reduce((n, g) => n + g.children.length, 0);
  if (!leaves) { el.innerHTML = `<p class="empty-state" style="padding:24px">Nothing to map.</p>`; return; }
  const h = d3.hierarchy({ name: "all", children: data.children }).sum((d) => (d.children ? 0 : d.value || 0))
    .sort((a, b) => (!!a.data.rest - !!b.data.rest) || b.value - a.value);
  d3.treemap().size([W, H]).paddingInner(2).paddingOuter(data.children.length > 1 ? 1 : 0).round(true).tile(d3.treemapSquarify.ratio(1.35))(h);
  const surface = cssVar("--surface");
  const base = (tile) => cssVar(tile && st.tileMeta.get(tile)?.color ? `--c-${st.tileMeta.get(tile).color}` : st.sum.collection ? `--c-${st.sum.collection.color || st.sum.collection.id}` : "--c-hub");
  const svg = d3.select(el).append("svg").attr("viewBox", `0 0 ${W} ${H}`).attr("role", "img").attr("aria-label", data.title);
  const g = svg.selectAll("g.leaf").data(h.leaves()).join("g")
    .attr("class", (d) => { const sel = st.sel[d.data.key || d.parent.data.key]; return "cell leaf" + (sel?.size && !sel.has(d.data.name) ? " dim" : ""); })
    .attr("transform", (d) => `translate(${d.x0},${d.y0})`);
  const n = (d) => d.parent.children.length;
  g.append("rect").attr("width", (d) => Math.max(0, d.x1 - d.x0)).attr("height", (d) => Math.max(0, d.y1 - d.y0)).attr("rx", 4)
    .attr("fill", (d) => d3.interpolateRgb(surface, base(d.data.tile || d.parent.data.tile))(d.data.other ? 0.42 : 0.95 - 0.38 * (d.data.rank / Math.max(n(d) - 1, 6))))
    .on("click", (_, d) => {
      tip();
      if (d.data.rest) return;
      const tile = d.data.tile || d.parent.data.tile;
      if (st.mapBy && st.tile == null && tile) st.tile = tile;   // a block of one tile opens that tile, filtered to it
      toggle(d.data.key || d.parent.data.key, d.data.name);
    })
    .on("mousemove", (ev, d) => tip(ev, `<b>${esc(d.data.name)}</b>${d.data.tile && st.tile == null ? ` <span class="faint">${esc(d.data.tile)}</span>` : ""}<br>${fmt.format(d.value)} ${esc(st.noun[1])}`))
    .on("mouseleave", () => tip());
  g.each(function (d) {
    const w = d.x1 - d.x0, hh = d.y1 - d.y0;
    if (w < 44 || hh < 26) return;
    const color = d3.lab(this.querySelector("rect").getAttribute("fill")).l > 64 ? "#17181c" : "#ffffff";
    const max = Math.floor((w - 14) / 6.6);
    const text = d.data.name.length > max ? d.data.name.slice(0, Math.max(1, max - 1)) + "…" : d.data.name;
    const t = d3.select(this);
    t.append("text").attr("class", "lbl").attr("x", 8).attr("y", 18).attr("fill", color).text(text);
    if (hh > 42) t.append("text").attr("class", "cnt").attr("x", 8).attr("y", 34).attr("fill", color).text(fmt.format(d.value));
  });
}
function tip(ev, html) {
  const el = $("#tip");
  if (!ev) { el.hidden = true; return; }
  el.innerHTML = html; el.hidden = false;
  el.style.left = Math.min(ev.clientX + 14, innerWidth - el.offsetWidth - 8) + "px";
  el.style.top = ev.clientY + 16 + "px";
}

// ── list ─────────────────────────────────────────────────────────────────────
function card(x) {
  const color = x.color ? `var(--c-${x.color})` : st.sum.collection ? `var(--c-${st.sum.collection.color || st.sum.collection.id})` : "var(--c-hub)";
  const chips = x.chips.filter((c) => !st.shared?.has(c) && c !== x.lead);
  return `<li><a class="card" href="${taskHref(st.spec, x.ref)}" style="--dc:${color}">
    <div class="row1">${x.lead ? `<span class="dn">${x.icon ? icon(x.icon, 13) : ""}${esc(x.lead)}</span>` : ""}${x.sub.map((a) => `<span class="faint">/</span><span class="cat">${esc(a)}</span>`).join("")}
      ${x.id || !x.lead ? `<span class="when org">${esc(x.id || `#${x.ref.split("/").pop()}`)}</span>` : ""}</div>
    <p class="t">${highlight(x.title)}</p>${x.brief ? `<p class="s">${highlight(x.brief)}</p>` : ""}
    ${chips.length || x.stats?.length ? `<div class="chips">${chips.slice(0, 6).map((c) => `<span class="chip">${esc(c)}</span>`).join("")}${(x.stats || []).map(([n, l]) => `<span class="chip outline"><b>${esc(n)}</b> ${esc(l)}</span>`).join("")}</div>` : ""}</a></li>`;
}
const AUTO_PAGES = 3;   // pages that load by themselves as you scroll; after that a button, so the footer stays reachable
function renderList(reset) {
  const list = $("#list", root);
  if (reset) { list.innerHTML = ""; shown = 0; autoLoaded = 0; }
  const next = matches.slice(shown, shown + PAGE);
  list.insertAdjacentHTML("beforeend", next.map(card).join(""));
  typeset(list);
  shown += next.length;
  $("#empty", root).hidden = matches.length > 0;
  list.hidden = !matches.length;
  const left = matches.length - shown;
  $("#more-row", root).hidden = left <= 0 || autoLoaded < AUTO_PAGES;
  $("#more-count", root).textContent = `Showing ${fmt.format(shown)} of ${fmt.format(matches.length)}`;
  $("#more", root).textContent = `Show ${fmt.format(Math.min(PAGE, left))} more`;
}

let scrollWatch = null;
function wire(el) {
  let t;
  $("#q", el).addEventListener("input", (ev) => {
    clearTimeout(t);
    t = setTimeout(() => { st.q = ev.target.value.trim(); st.offset = 0; run(); }, st.sum.inline ? 120 : 350);
  });
  $("#subset", el)?.addEventListener("change", (ev) => {
    st.subset = ev.target.value; st.offset = 0; st.sel = {}; st.q = "";
    writeURL();
    load(el);   // its own facets and total
  });
  $("#tiles", el)?.addEventListener("click", (ev) => { const b = ev.target.closest(".dom"); if (b) { st.tile = b.dataset.tile || null; st.sel = {}; run(); } });
  $("#facets", el).addEventListener("click", (ev) => {
    const o = ev.target.closest(".opt");
    if (o) return toggle(o.dataset.k, o.dataset.v);
    const m = ev.target.closest("[data-more]");
    if (m) { expanded.has(m.dataset.more) ? expanded.delete(m.dataset.more) : expanded.add(m.dataset.more); renderFacets(); }
  });
  $("#facets", el).addEventListener("toggle", (e) => { const d = e.target.closest?.("[data-facet]"); if (d) facetOpen.set(d.dataset.facet, d.open); }, true);
  $("#active", el).addEventListener("click", (ev) => {
    const p = ev.target.closest(".pill");
    if (!p) return;
    if (p.dataset.tile != null) { st.tile = null; st.sel = {}; run(); } else toggle(p.dataset.k, p.dataset.v);
  });
  const clearAll = () => { st.sel = {}; st.q = ""; st.tile = null; st.offset = 0; $("#q", el).value = ""; run(); };
  $("#clear", el).addEventListener("click", clearAll);
  $("#clear2", el).addEventListener("click", clearAll);
  const sheet = (on) => { $("#filters", el).classList.toggle("open", on); $("#sheet-scrim", el).hidden = !on; document.body.classList.toggle("noscroll", on); };
  $("#open-filters", el)?.addEventListener("click", () => sheet(true));
  $("#close-filters", el).addEventListener("click", () => sheet(false));
  $("#sheet-scrim", el).addEventListener("click", () => sheet(false));
  $("#show-results", el).addEventListener("click", () => { sheet(false); $("#toolbar", el).scrollIntoView({ block: "start" }); });
  $("#clear3", el).addEventListener("click", () => { st.sel = {}; st.tile = null; st.offset = 0; run(); });
  $("#map-card", el)?.addEventListener("toggle", () => renderMap());
  $("#more", el).addEventListener("click", () => renderList(false));
  const page = (d) => { st.offset = Math.max(0, st.offset + d * (st.page?.page || 24)); run(); $("#toolbar", el).scrollIntoView({ block: "start" }); };
  $("#prev", el)?.addEventListener("click", () => page(-1));
  $("#next", el)?.addEventListener("click", () => page(1));
  $("#random", el).addEventListener("click", async () => {
    if (st.sum.inline) {
      const pool = matches.length ? matches : st.cards;
      nav(taskHref(st.spec, pool[Math.floor(Math.random() * pool.length)].ref));
      return;
    }
    try { const r = await api(`${envApi(st.spec)}/random${st.subset ? `?subset=${encodeURIComponent(st.subset)}` : ""}`); nav(taskHref(st.spec, r.ref)); }
    catch (e) { toast(e.message); }
  });
  // the next page loads by itself as the end of the list comes into view, a few times
  scrollWatch?.disconnect();
  if (st.sum.inline) {
    scrollWatch = new IntersectionObserver((es) => {
      if (!es.some((e) => e.isIntersecting) || !root || shown >= matches.length || autoLoaded >= AUTO_PAGES) return;
      autoLoaded++;
      renderList(false);
    }, { rootMargin: "600px 0px" });
    scrollWatch.observe($("#sentinel", el));
  }
}

let rt;   // one listener for the page's lifetime: the map redraws at its new size
addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => root && st.cards && (st.mapKey || st.mapBy) && renderMap(), 150); });
