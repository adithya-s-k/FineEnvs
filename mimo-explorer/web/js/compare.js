// Compare rollouts of one task side by side: the settings and what they cost, the grade check by check, what each
// agent produced, how it worked, and the traces. It reads the same endpoints as the rollout page, so it shows exactly
// what this viewer may see: a private rollout in a shared link stays hidden from everyone but its owner.
import { $, $$, api, esc, checkMsg, md, money, dur, ago, tokensShort, statusPill, rewardBadge, DOMAIN_NAME, LIVE_STATUSES, sk, emptyState, toast, fmt,
  progress, visibilityBadge } from "./util.js";
import { icon, DOMAIN_ICON } from "./icons.js";
import { getSession } from "./session.js";
import { traceHtml, diffHtml, prettyTool, isMcpTool } from "./run.js";
import * as explore from "./explore.js";

export const MAX = 4;
const LETTER = ["A", "B", "C", "D"];
const THINK = { none: "off", low: "low", medium: "medium", high: "high", default: "model default" };
const METRICS = [["calls", "Model calls"], ["tokens", "Tokens"], ["cost", "Model cost"]];
let alive = false, ro = null, offDoc = null;

export function unmount() {
  alive = false;
  ro?.disconnect(); ro = null;
  io?.disconnect(); io = null;
  if (offDoc) { document.removeEventListener("click", offDoc); document.removeEventListener("keydown", offDoc); }
  offDoc = null;
}

const enc = encodeURIComponent;
const short = (m) => (m || "").split("/")[1] || m || "";
const modelName = (r) => short(r.model);   // the full id is in each title and the summary
const servedBy = (r) => (r.endpoint ? "own endpoint" : r.provider || "auto");
const isLive = (r) => LIVE_STATUSES.includes(r.status);
const shared = (r) => r.visibility === "public" && r.status === "done" && r.reward != null;   // what others can open
const humanize = (id) => { const t = String(id || "").replace(/[_-]+/g, " ").trim(); return t.charAt(0).toUpperCase() + t.slice(1); };
const listOf = (xs) => (xs.length > 1 ? `${xs.slice(0, -1).join(", ")} and ${xs[xs.length - 1]}` : xs[0] || "");
const tag = (slot) => `<span class="cmp-tag"><i style="background:var(--s${slot + 1})"></i>${LETTER[slot]}</span>`;

// Slots, not a plain list: a rollout keeps its letter and colour when another is removed, and so does a shared link
// ("?r=a,,c" leaves B empty).
export function compareHash(task, slots) {
  const s = [...slots];
  while (s.length && !s[s.length - 1]) s.pop();
  return `#/compare/${enc(task)}${s.length ? "?r=" + s.map((x) => (x ? enc(x) : "")).join(",") : ""}`;
}
function parseSlots(qs) {
  const seen = new Set();
  return (new URLSearchParams(qs || "").get("r") || "").split(",").slice(0, MAX).map((x) => {
    x = x.trim();
    if (!x || seen.has(x)) return null;
    seen.add(x);
    return x;
  });
}

function skeleton() {
  return `<div class="wrap page"><div class="crumbs">${sk.line(24)}</div>
    <div class="tp-head">${sk.line(22, 22)}${sk.line(55, 28)}${sk.line(30)}</div>
    <div class="cmp-cols" style="--n:2">${[0, 1].map(() => sk.card(sk.line(45, 14) + sk.line(70))).join("")}</div>
    ${[6, 4].map((n) => `<div class="panel" style="margin-top:16px"><div class="panel-h">${sk.line(18, 14)}</div>
      <div class="panel-b" style="display:grid;gap:10px">${sk.lines(...[100, 92, 96, 88, 94, 90].slice(0, n))}</div></div>`).join("")}</div>`;
}

export async function mount(el, { task, qs }) {
  alive = true;
  const st = { task, slots: parseSlots(qs), data: new Map(), cands: [], metric: "calls", tab: 0, info: null };
  el.innerHTML = skeleton();
  await Promise.all([fetchRuns(st), fetchCandidates(st)]);
  if (!alive) return;
  if (!visible(st).length && !st.cands.length) st.info = await explore.load().then((i) => i.envs.find((e) => e.id === task)).catch(() => null);
  if (!alive) return;
  render(el, st);
  el.addEventListener("click", (e) => onClick(e, el, st));
  el.addEventListener("change", (e) => {
    if (e.target.id === "cmp-same") $("#cmp-summary table", el)?.classList.toggle("hide-same", e.target.checked);
  });
  // the "Add a rollout" list closes on a click outside it and on Escape
  offDoc = (e) => {
    const d = $(".cmp-add[open]", el);
    if (!d) return;
    if (e.type === "keydown" ? e.key === "Escape" : !e.composedPath().includes(d)) { d.open = false; if (e.type === "keydown") $("summary", d).focus(); }
  };
  document.addEventListener("click", offDoc);
  document.addEventListener("keydown", offDoc);
}

// ── data ─────────────────────────────────────────────────────────────────────
async function fetchRuns(st) {
  await Promise.all(st.slots.filter((id) => id && !st.data.has(id)).map(async (id) => {
    try {
      const d = await api(`/api/runs/${enc(id)}`);
      st.data.set(id, d.run.task_id === st.task ? { run: d.run, events: d.events, dg: digest(d.run, d.events) } : { other: true });
    } catch { st.data.set(id, { missing: true }); }
  }));
}

// What can be added: your own rollouts of this task (any state), then everyone's public ones.
async function fetchCandidates(st) {
  const mine = getSession().user ? api(`/api/runs?task_id=${enc(st.task)}`).then((d) => d.runs).catch(() => []) : [];
  const pub = api(`/api/tasks/${enc(st.task)}/rollouts?limit=200`).then((d) => d.runs).catch(() => []);
  const [a, b] = await Promise.all([mine, pub]);
  const own = new Set(a.map((r) => r.id));
  st.cands = [...a.map((r) => ({ ...r, mine: true })), ...b.filter((r) => !own.has(r.id))];
}

const visible = (st) => st.slots.map((id, slot) => (id && st.data.get(id)?.run ? { id, slot, ...st.data.get(id) } : null)).filter(Boolean);

// Everything the page compares, read once from a trace.
function digest(run, events) {
  const d = { tools: {}, toolCalls: 0, toolErrors: 0, messages: 0, thinkChars: 0, calls: 0, points: [], phases: {} };
  let tokens = 0;
  for (const e of events) {
    if (e.kind === "tool") { d.toolCalls++; if (e.status === "error") d.toolErrors++; d.tools[e.tool || "?"] = (d.tools[e.tool || "?"] || 0) + 1; }
    else if (e.kind === "text") { d.messages++; d.final = e; }
    else if (e.kind === "thinking") d.thinkChars += (e.text || "").length;
    else if (e.kind === "step") {
      const t = e.tokens || {};
      tokens += (t.input || 0) + (t.cache_read || 0) + (t.cache_write || 0) + (t.output || 0) + (t.reasoning || 0);
      d.points.push({ t: e.t, calls: ++d.calls, tokens, cost: e.cost || 0 });   // a step records the model cost so far
    } else if (e.kind === "phase") (d.phases[e.name] ||= {})[e.status] = e;
    else if (["checks", "prompt", "diff", "files", "image"].includes(e.kind)) d[e.kind] = e;
  }
  const a = d.phases.agent || {}, end = a.done || a.error;
  d.agentTime = a.start && end ? end.t - a.start.t : null;
  d.start = a.start ? a.start.t : 0;
  d.end = events.length ? events[events.length - 1].t : 0;
  const t0 = run.started_at || run.created_at;
  d.duration = run.finished_at && t0 ? run.finished_at - t0 : isLive(run) && t0 ? Date.now() / 1000 - t0 : d.end || null;
  if (d.diff) d.changed = diffStat(d.diff);
  return d;
}

function diffStat(e) {
  const out = Object.fromEntries((e.files || []).map((f) => [f, null]));
  let cur = null;
  for (const l of (e.text || "").split("\n")) {
    const m = l.match(/^diff --git a\/(.+?) b\/(.+)$/);
    if (m) { cur = out[m[2]] = { add: 0, del: 0 }; continue; }
    if (!cur || l.startsWith("+++") || l.startsWith("---")) continue;
    if (l.startsWith("+")) cur.add++;
    else if (l.startsWith("-")) cur.del++;
  }
  return out;
}

// ── page ─────────────────────────────────────────────────────────────────────
const JUMPS = [["summary", "Summary"], ["grade", "Grade"], ["outcome", "Output"], ["work", "How they worked"], ["prompt", "Instructions"], ["traces", "Traces"]];

function render(el, st) {
  const runs = visible(st);
  // the newest record's title: early rollouts stored a first-sentence title that could be a greeting ("Hi,")
  const newest = [...runs.map((x) => x.run), ...st.cands].sort((a, b) => (b.created_at || 0) - (a.created_at || 0))[0];
  const title = newest?.title || st.info?.t || st.task;
  const domain = newest?.domain || st.info?.d;
  const mine = runs.length && runs.every((x) => x.run.is_owner);   // only your own: it belongs under My rollouts
  document.querySelectorAll("[data-nav]").forEach((a) => a.classList.toggle("on", a.dataset.nav === (mine ? "runs" : "community")));
  history.replaceState(null, "", compareHash(st.task, st.slots));
  st.tab = Math.min(st.tab, Math.max(0, runs.length - 1));
  const live = runs.filter((x) => isLive(x.run));
  const parts = runs.length >= 2 ? [summary(runs), grade(runs), outcome(runs), working(st, runs), instructions(runs), traces(st, runs)].filter(Boolean) : [];
  const jumps = JUMPS.filter(([id]) => parts.some((p) => p.includes(`id="cmp-${id}"`)));
  el.innerHTML = `<div class="wrap page cmp fade-in" style="--n:${Math.max(runs.length, 1)}">
    <nav class="crumbs cmp-crumbs" aria-label="Breadcrumb"><a class="home" href="#/">Environments</a><span class="sep">${icon("chevronRight", 13)}</span>
      ${domain ? `<a href="#/?d=${esc(domain)}">${esc(DOMAIN_NAME[domain])}</a>${icon("chevronRight", 13)}` : ""}
      <a class="tid" href="#/task/${enc(st.task)}">${esc(st.task)}</a>${icon("chevronRight", 13)}<b>Compare</b></nav>
    <header class="tp-head cmp-head">
      <div class="kick">${domain ? `<span class="badge">${icon(DOMAIN_ICON[domain], 13)}${esc(DOMAIN_NAME[domain])}</span>` : ""}
        <span class="chip">${runs.length ? `Comparing ${runs.length} rollout${runs.length === 1 ? "" : "s"}` : "Compare rollouts"}</span></div>
      <h1><a href="#/task/${enc(st.task)}">${esc(title)}</a></h1>
      <div class="facts">${addButton(st)}<button class="link" type="button" data-copy>${icon("link", 14)}Copy link</button>
        <a class="link" href="#/task/${enc(st.task)}">${icon("file", 14)}View task</a>${shareNote(runs)}</div>
    </header>
    ${live.length ? `<div class="note-box" style="margin-bottom:14px">${icon("clock")}<span>${listOf(live.map((x) => LETTER[x.slot]))} ${live.length === 1 ? "is" : "are"} still running.
      This page shows ${live.length === 1 ? "it" : "them"} as of when you opened it. <button class="link u" type="button" data-refresh>Refresh</button></span></div>` : ""}
    ${columns(st)}
    ${parts.length ? `<nav class="cmp-jumps cmp-toc" aria-label="Sections">${jumpLinks(jumps)}</nav>${parts.join("")}` : pickPanel(st, runs)}
  </div>
  ${parts.length ? strip(runs, jumps) : ""}`;
  if (parts.length) { drawChart(el, st, runs); watchStrip(el); }
}

const jumpLinks = (jumps) => jumps.map(([id, l]) => `<a href="#" data-jump="${id}">${l}</a>`).join("");

// Once the rollout cards scroll away, a slim bar under the site header keeps who is who in view, with the sections.
function strip(runs, jumps) {
  return `<div class="cmp-strip" id="cmp-strip" inert><div class="wrap cmp-strip-in">
    <div class="cmp-strip-runs">${runs.map((x) => `<a class="cmp-chip" href="#/run/${enc(x.id)}" title="${esc(x.run.model)}">${tag(x.slot)}<span class="m">${esc(modelName(x.run))}</span>${rewardBadge(x.run.reward, x.run.status)}</a>`).join("")}</div>
    <nav class="cmp-jumps" aria-label="Sections">${jumpLinks(jumps)}</nav>
  </div></div>`;
}

let io = null;
function watchStrip(el) {
  io?.disconnect();
  const bar = $("#cmp-strip", el), cards = $(".cmp-cols", el);
  if (!bar || !cards) return;
  const header = parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--header-h")) || 56;
  io = new IntersectionObserver(([e]) => {
    const on = !e.isIntersecting && e.boundingClientRect.top < header;
    bar.classList.toggle("on", on);
    bar.inert = !on;
  }, { rootMargin: `-${header}px 0px 0px 0px` });
  io.observe(cards);
}

function shareNote(runs) {
  if (!runs.length) return "";
  const open = runs.filter((x) => shared(x.run));
  if (open.length === runs.length) return "";
  const text = open.length ? `Anyone else opening this link sees ${listOf(open.map((x) => LETTER[x.slot]))} only: the others are private or not scored.`
    : "Only you can see these rollouts: private and unscored ones stay hidden in a shared link.";
  return `<span class="cmp-note">${icon("shield", 14)}${text}</span>`;
}

function columns(st) {
  const cards = st.slots.map((id, slot) => {
    if (!id) return "";
    const d = st.data.get(id) || {};
    const rm = `<button class="icon-btn" type="button" data-remove="${slot}" aria-label="Remove ${LETTER[slot]} from the comparison" title="Remove">${icon("x", 14)}</button>`;
    if (!d.run) {
      return `<div class="cmp-col gone"><div class="cmp-col-h">${tag(slot)}<span class="m">Unavailable</span>${rm}</div>
        <p class="faint xs">${d.other ? "This rollout is of another task." : "It may be private, or the link is wrong."} <code>${esc(id)}</code></p></div>`;
    }
    const r = d.run;
    return `<div class="cmp-col"><div class="cmp-col-h">${tag(slot)}<a class="m" href="#/run/${enc(id)}" title="Open ${esc(r.model)}'s rollout">${esc(modelName(r))}</a>${rm}</div>
      <div class="cmp-col-b">${r.status === "done" ? rewardBadge(r.reward, r.status) : statusPill(r.status)}${r.is_owner ? visibilityBadge(r.visibility) : ""}
        <span class="faint xs">${esc(servedBy(r))} · ${money((r.cost || {}).total)} · ${ago(r.created_at)}</span></div></div>`;
  }).join("");
  return cards ? `<div class="cmp-cols">${cards}</div>` : "";
}

function addButton(st) {
  return `<details class="cm-more cmp-add"><summary class="btn sm">${icon("plus", 13)}Add a rollout</summary>
    <div class="cm-pop cmp-pop">${candList(st)}</div></details>`;
}

function candList(st) {
  const inView = new Set(st.slots.filter(Boolean));
  const full = inView.size >= MAX;
  if (!st.cands.length) return `<p class="muted sm" style="padding:6px">No rollouts of this task you can see yet.</p>`;
  const row = (r) => {
    const on = inView.has(r.id);
    return `<button type="button" class="cmp-cand" data-add="${esc(r.id)}" ${on || full ? "disabled" : ""}>${r.status === "done" ? rewardBadge(r.reward, r.status) : statusPill(r.status)}
      <span class="m">${esc(modelName(r))}<span class="faint"> · ${esc(servedBy(r))}${r.params?.thinking ? ` · thinking ${esc(THINK[r.params.thinking] || r.params.thinking)}` : ""}</span></span>
      <span class="faint xs">${on ? "in view" : `${money((r.cost || {}).total)} · ${ago(r.created_at)}`}</span></button>`;
  };
  const mine = st.cands.filter((r) => r.mine), pub = st.cands.filter((r) => !r.mine);
  return `${full ? `<p class="faint xs" style="padding:4px 6px">Up to ${MAX} at once. Remove one to add another.</p>` : ""}
    ${mine.length ? `<div class="sec-label">Your rollouts</div>${mine.map(row).join("")}` : ""}
    ${pub.length ? `<div class="sec-label">Community, anonymous</div>${pub.map(row).join("")}` : ""}`;
}

function pickPanel(st, runs) {
  const inView = new Set(st.slots.filter(Boolean));
  const others = st.cands.filter((r) => !inView.has(r.id)).length;
  const body = others ? `<div class="cmp-pick">${candList(st)}</div>`
    : emptyState("columns", "Nothing to compare with yet", "No other rollout of this task is visible to you. Run it with another model or setting, then compare the two.",
      `<a class="btn primary" href="#/task/${enc(st.task)}">${icon("play", 14)}Run this task</a>`);
  return sec("pick", "plus", runs.length ? "Pick a rollout to compare with" : "Pick rollouts to compare", `up to ${MAX}, all of this task`, body);
}

const sec = (id, ic, title, aside, body, tools = "") => `<section class="panel cmp-sec" id="cmp-${id}"><div class="panel-h"><h2>${icon(ic, 15)}${esc(title)}</h2>
  ${aside ? `<span class="aside">${aside}</span>` : ""}${tools}</div><div class="panel-b">${body}</div></section>`;

// A table with one column per rollout. Rows: {label, cells, same, cls} or {group}.
function ctable(runs, rows, cls = "") {
  const head = `<tr><th></th>${runs.map((x) => `<th>${tag(x.slot)}<span class="m">${esc(modelName(x.run))}</span></th>`).join("")}</tr>`;
  const body = rows.map((r) => r.group ? `<tr class="grp${r.same ? " same" : ""}"><th colspan="${runs.length + 1}">${esc(r.group)}</th></tr>`
    : `<tr class="${[r.same && "same", r.cls].filter(Boolean).join(" ")}"><th scope="row">${r.label}</th>${r.cells.map((c) => `<td>${c}</td>`).join("")}</tr>`).join("");
  return `<div class="cmp-tw"><table class="cmp-t ${cls}"><thead>${head}</thead><tbody>${body}</tbody></table></div>`;
}

// ── summary ──────────────────────────────────────────────────────────────────
function summary(runs) {
  const music = runs[0].run.domain === "music";
  const p = (x) => x.run.params || {};
  const tk = (x) => x.run.tokens || {};
  const c = (x) => x.run.cost || {};
  const unmetered = (x) => x.run.endpoint && x.run.endpoint.price_in == null && x.run.endpoint.price_out == null;
  const specs = [
    { group: "Result" },
    { label: "Reward", v: (x) => x.run.reward, html: (x) => rewardBadge(x.run.reward, x.run.status), best: "max" },
    { label: "Status", v: (x) => x.run.status, html: (x) => statusPill(x.run.status) },
    { group: "Model and settings" },
    { label: "Model", v: (x) => x.run.model, html: (x) => `<code>${esc(x.run.model)}</code>` },
    { label: "Served by", v: (x) => servedBy(x.run) },
    runs.some((x) => x.run.judge) && { label: "Judge", v: (x) => x.run.judge || "none", html: (x) => (x.run.judge ? `<code>${esc(x.run.judge)}</code>` : "none") },
    { label: "Thinking", v: (x) => THINK[p(x).thinking] || "model default" },
    { label: "Temperature", v: (x) => p(x).temperature ?? "model default" },
    music ? { label: "Max tokens", v: (x) => p(x).max_tokens || p(x).max_tokens_used || 100000, html: (x) => fmt.format(p(x).max_tokens || p(x).max_tokens_used || 100000) }
      : { label: "Step cap", v: (x) => p(x).steps || "task default" },
    !music && { label: "Time limit", v: (x) => (p(x).timeout_min ? `${p(x).timeout_min} min` : "task default") },
    !music && { label: "Max tokens per reply", v: (x) => p(x).max_tokens || "default", html: (x) => (p(x).max_tokens ? fmt.format(p(x).max_tokens) : "default") },
    { group: "How it went" },
    { label: "Duration", v: (x) => x.dg.duration, html: (x) => dur(x.dg.duration) || "–", best: "min" },
    !music && { label: "Agent time", v: (x) => x.dg.agentTime, html: (x) => dur(x.dg.agentTime) || "–" },
    !music && { label: "Model calls", v: (x) => x.dg.calls, num: true },
    !music && { label: "Tool calls", v: (x) => x.dg.toolCalls, num: true },
    !music && { label: "Failed tool calls", v: (x) => x.dg.toolErrors, num: true },
    { label: "Messages", v: (x) => x.dg.messages, num: true },
    { label: "Thinking written", v: (x) => x.dg.thinkChars, html: (x) => (x.dg.thinkChars ? `${fmt.format(x.dg.thinkChars)} characters` : "none") },
    runs.some((x) => x.dg.diff) && { label: "Files changed", v: (x) => (x.dg.diff ? x.dg.diff.files.length : 0), num: true },
    runs.some((x) => x.dg.files) && { label: "Files delivered", v: (x) => (x.dg.files ? x.dg.files.files.length : 0), num: true },
    { group: "Tokens and cost" },
    { label: "Tokens in", v: (x) => (tk(x).input || 0) + (tk(x).cache_read || 0) + (tk(x).cache_write || 0),
      html: (x) => `${tokensShort((tk(x).input || 0) + (tk(x).cache_read || 0) + (tk(x).cache_write || 0))}${tk(x).cache_read ? ` <span class="faint">(${tokensShort(tk(x).cache_read)} cached)</span>` : ""}` },
    { label: "Tokens out", v: (x) => tk(x).output || 0, html: (x) => tokensShort(tk(x).output) },
    { label: "Thinking tokens", v: (x) => tk(x).reasoning || 0, html: (x) => tokensShort(tk(x).reasoning) },
    { label: "Model", v: (x) => (unmetered(x) ? null : c(x).model), html: (x) => (unmetered(x) ? '<span class="faint">not metered</span>' : money(c(x).model)), best: "min" },
    !music && { label: "Sandbox", v: (x) => c(x).sandbox, html: (x) => money(c(x).sandbox) },
    runs.some((x) => c(x).judge) && { label: "Judge calls", v: (x) => c(x).judge || 0, html: (x) => money(c(x).judge || 0) },
    { label: "<b>Total</b>", v: (x) => c(x).total, html: (x) => `<b>${money(c(x).total)}</b>`, best: "min" },
    { group: "Versions" },
    { label: "Started", v: (x) => x.run.created_at, html: (x) => esc(new Date(x.run.created_at * 1000).toLocaleString([], { dateStyle: "medium", timeStyle: "short" })) },
    { label: "Explorer", v: (x) => x.run.provenance?.app?.source || null, html: (x) => (x.run.provenance?.app ? `v${esc(x.run.provenance.app.version)} · <code>${esc(x.run.provenance.app.source)}</code>` : "–") },
    !music && { label: "OpenCode", v: (x) => x.run.provenance?.harness?.installed || x.run.provenance?.harness?.version || null },
    { label: "Dataset", v: (x) => (x.run.provenance?.dataset?.revision || "").slice(0, 10) || null, html: (x) => (x.run.provenance?.dataset?.revision ? `<code>${esc(x.run.provenance.dataset.revision.slice(0, 10))}</code>` : "–") },
    { label: "Rollout", v: (x) => x.id, html: (x) => `<a class="u" href="#/run/${enc(x.id)}"><code>${esc(x.id)}</code></a>` },
  ].filter(Boolean);
  const rows = [];
  let group = null;
  for (const s of specs) {
    if (s.group) { group = { group: s.group, same: true }; rows.push(group); continue; }
    const vals = runs.map(s.v);
    if (vals.every((v) => v == null || v === "")) continue;
    const same = vals.every((v) => JSON.stringify(v) === JSON.stringify(vals[0]));
    // bold the best only where it means something: a real difference, between rollouts that finished (a run that
    // stopped early is neither fast nor cheap)
    const nums = runs.map((x, i) => (typeof vals[i] === "number" && (s.best === "max" || x.run.status === "done") ? vals[i] : null));
    const pool = nums.filter((v) => v != null);
    const top = s.best && !same && pool.length > 1 ? (s.best === "max" ? Math.max(...pool) : Math.min(...pool)) : null;
    const cells = runs.map((x, i) => {
      const h = s.html ? s.html(x) : vals[i] == null || vals[i] === "" ? "–" : s.num ? fmt.format(vals[i]) : esc(vals[i]);
      return top != null && nums[i] === top ? `<span class="best">${h}</span>` : h;
    });
    rows.push({ label: s.label, cells, same });
    if (!same) group.same = false;
  }
  return sec("summary", "list", "Summary", "", `${ctable(runs, rows)}<p class="fine" style="margin-top:8px">Faint rows are the same in every rollout. Bold marks the best reward, the fastest and the cheapest.</p>`,
    `<label class="switch" style="margin-left:auto"><input type="checkbox" id="cmp-same"> Hide rows that match</label>`);
}

// ── grade ────────────────────────────────────────────────────────────────────
function grade(runs) {
  if (!runs.some((x) => x.dg.checks)) return sec("grade", "scale", "Grade", "", `<p class="muted sm">None of these rollouts was graded.</p>`);
  const shots = runs.some((x) => x.dg.image) ? `<div class="cmp-grid cmp-shots">${runs.map((x) => {
    const e = x.dg.image, src = e && `/api/runs/${enc(x.id)}/artifacts/${enc(e.name)}`;
    return `<figure>${e ? `<a class="shot" href="${src}" target="_blank" rel="noopener" title="Open the full-page render"><img loading="lazy" src="${src}" alt="What the judge saw for ${LETTER[x.slot]}"></a>`
      : `<div class="shot none">No render</div>`}<figcaption>${tag(x.slot)}${esc(modelName(x.run))}</figcaption></figure>`;
  }).join("")}</div>` : "";
  const order = [], by = new Map();
  for (const x of runs) for (const c of x.dg.checks?.checks || []) if (!by.has(c.id)) { by.set(c.id, c); order.push(c.id); }
  const outcome = (c) => (c ? `${c.passed}|${c.score ?? ""}` : "none");
  const cell = (c) => !c ? `<span class="faint">–</span>`
    : `<span class="cmp-ck ${c.passed === true ? "ok" : c.passed === false ? "bad" : "na"}">${icon(c.passed === true ? "check" : c.passed === false ? "x" : "more", 13)}${
      c.score != null && c.passed == null ? Number(c.score).toFixed(2) : c.score != null ? `${c.passed ? "Pass" : "Fail"} · ${Number(c.score).toFixed(2)}` : c.passed === true ? "Pass" : c.passed === false ? "Fail" : "–"}</span>
      ${c.message ? `<span class="cmp-msg" title="${esc(checkMsg(c.message))}">${esc(checkMsg(c.message).slice(0, 160))}</span>` : ""}`;
  const rows = [
    { label: "Reward", cells: runs.map((x) => rewardBadge(x.run.reward, x.run.status)) },
    // the summary is worth a row only where it adds to the checks (a lone test check already says "6 passed")
    runs.some((x) => x.dg.checks?.summary && !(x.dg.checks.checks || []).some((c) => String(c.message || "").includes(x.dg.checks.summary))) && { label: "Grader's summary",
      cells: runs.map((x) => { const s = x.dg.checks?.summary || ""; return s ? `<span class="cmp-msg" title="${esc(s)}">${esc(s.slice(0, 220))}${s.length > 220 ? "…" : ""}</span>` : "–"; }) },
    ...order.map((id) => {
      const cs = runs.map((x) => (x.dg.checks?.checks || []).find((c) => c.id === id));
      const q = by.get(id).question;
      // a rollout that was never graded has no say in whether the others disagree
      const graded = cs.filter((c, i) => runs[i].dg.checks);
      return { label: q ? `<span title="${esc(id)}">${esc(q)}</span>` : esc(humanize(id)), cells: cs.map(cell), cls: graded.length > 1 && new Set(graded.map(outcome)).size > 1 ? "split" : "" };
    }),
  ].filter(Boolean);
  const split = rows.filter((r) => r.cls === "split").length;
  const gradedRuns = runs.filter((x) => x.dg.checks);
  const note = gradedRuns.length < 2 ? ` · only ${LETTER[gradedRuns[0].slot]} was graded`
    : split ? ` · shaded: the ${split} where they differ` : " · they agree on every check";
  return sec("grade", "scale", "Grade", order.length ? `${order.length} check${order.length === 1 ? "" : "s"}${note}` : "",
    shots + ctable(runs, rows, "cmp-grade"));
}

// ── what they produced ───────────────────────────────────────────────────────
function outcome(runs) {
  const parts = [];
  if (runs.some((x) => x.dg.final)) {
    parts.push(`<div class="sec-label">Last message from the model</div><div class="cmp-grid">${runs.map((x) => `<div class="cmp-cell"><div class="cmp-cell-h">${tag(x.slot)}${esc(modelName(x.run))}</div>
      ${x.dg.final ? `<div class="cmp-final prose">${md(x.dg.final.text)}</div>` : `<p class="muted sm">No message.</p>`}</div>`).join("")}</div>`);
  }
  if (runs.some((x) => x.dg.diff)) {
    const files = [...new Set(runs.flatMap((x) => Object.keys(x.dg.changed || {})))].sort();
    const stat = (s) => (s === undefined ? `<span class="faint">–</span>` : s === null ? "changed" : `<span class="da">+${s.add}</span> <span class="dd">−${s.del}</span>`);
    const rows = files.map((f) => {
      const cs = runs.map((x) => (x.dg.changed || {})[f]);
      return { label: `<code>${esc(f)}</code>`, cells: cs.map(stat), cls: cs.some((s) => s === undefined) ? "split" : "" };
    });
    parts.push(`<div class="sec-label">Files changed</div>${ctable(runs, rows, "cmp-files")}
      <div class="cmp-grid" style="margin-top:12px">${runs.map((x) => `<div class="cmp-cell">${x.dg.diff
        ? `<details class="cmp-diff"><summary class="disclose">${icon("chevronRight", 14, "chev")}${tag(x.slot)}The change · ${x.dg.diff.files.length} file${x.dg.diff.files.length === 1 ? "" : "s"}</summary>
          <pre class="code-block diff">${diffHtml(x.dg.diff.text)}</pre></details>`
        : `<p class="muted sm">${tag(x.slot)} No change recorded.</p>`}</div>`).join("")}</div>`);
  }
  return parts.length ? sec("outcome", "doc", "What each one produced", "", parts.join("")) : "";
}

// ── how they worked ──────────────────────────────────────────────────────────
function working(st, runs) {
  const parts = [];
  if (runs.filter((x) => x.dg.points.length).length && runs.some((x) => x.dg.points.length > 1)) {
    parts.push(`<div class="cmp-chart-bar"><div class="sec-label" style="margin:0">Over time, cumulative</div>
      <div class="seg" role="group" aria-label="What to chart">${METRICS.map(([k, l]) => `<button type="button" data-metric="${k}" aria-pressed="${st.metric === k}">${l}</button>`).join("")}</div></div>
      <div class="cmp-chart" id="cmp-chart"></div>
      <div class="cmp-legend">${runs.map((x) => `<span><i style="background:var(--s${x.slot + 1})"></i><b>${LETTER[x.slot]}</b>${esc(modelName(x.run))}</span>`).join("")}</div>`);
  }
  const names = [...new Set(runs.flatMap((x) => Object.keys(x.dg.tools)))];
  if (names.length) {
    const total = (n) => runs.reduce((s, x) => s + (x.dg.tools[n] || 0), 0);
    names.sort((a, b) => total(b) - total(a) || a.localeCompare(b));
    const rows = names.map((n) => ({ label: isMcpTool(n) ? `<span title="${esc(n)}">${esc(prettyTool(n))}</span> <span class="chip">MCP</span>` : `<code>${esc(n)}</code>`,
      cells: runs.map((x) => (x.dg.tools[n] ? fmt.format(x.dg.tools[n]) : `<span class="faint">–</span>`)) }));
    rows.push({ label: "<b>All tool calls</b>", cells: runs.map((x) => `<b>${fmt.format(x.dg.toolCalls)}</b>`), cls: "total" });
    parts.push(`<div class="sec-label" style="margin-top:18px">Tools used</div>${ctable(runs, rows, "cmp-tools")}`);
  }
  return parts.length ? sec("work", "gauge", "How they worked", "", parts.join("")) : "";
}

const samePrompt = (runs) => runs.every((x) => x.dg.prompt) && runs.every((x) => x.dg.prompt.text === runs[0].dg.prompt.text);

function instructions(runs) {
  if (!runs.some((x) => x.dg.prompt)) return "";
  const block = (e) => `<details class="cmp-prompt"><summary class="disclose">${icon("chevronRight", 14, "chev")}Show the instructions · ${(e.text || "").length.toLocaleString()} characters</summary>
    <div class="prompt-text">${esc(e.text)}</div></details>`;
  const same = samePrompt(runs);
  const body = same ? block(runs[0].dg.prompt)
    : `<div class="cmp-grid">${runs.map((x) => `<div class="cmp-cell"><div class="cmp-cell-h">${tag(x.slot)}${esc(modelName(x.run))}</div>${x.dg.prompt ? block(x.dg.prompt)
      : `<p class="muted sm">Not recorded: rollouts from before 26 Sep don't store it.</p>`}</div>`).join("")}</div>`;
  return sec("prompt", "doc", "Instructions to the model", same ? `the same for all ${runs.length}` : "not the same for all", body);
}

function traces(st, runs) {
  const skip = samePrompt(runs) ? ["prompt"] : [];
  const cols = runs.map((x, i) => {
    const t = traceHtml(x.id, x.events, { skip });
    return `<div class="cmp-tcol${i === st.tab ? " on" : ""}" data-col="${i}"><div class="cmp-th">${tag(x.slot)}<a class="m" href="#/run/${enc(x.id)}">${esc(modelName(x.run))}</a>
      ${rewardBadge(x.run.reward, x.run.status)}<span class="faint xs">${t.steps} step${t.steps === 1 ? "" : "s"}</span></div>
      ${t.html ? `<ol class="timeline">${t.html}</ol>` : `<p class="muted sm">No agent steps were recorded.</p>`}</div>`;
  }).join("");
  const tabs = `<div class="seg cmp-tabs" role="group" aria-label="Which trace to show">${runs.map((x, i) =>
    `<button type="button" data-tab="${i}" aria-pressed="${i === st.tab}">${tag(x.slot)}${esc(modelName(x.run))}</button>`).join("")}</div>`;
  return sec("traces", "list", "Traces", "", `${tabs}<div class="cmp-grid cmp-tl">${cols}</div>`,
    `<button class="btn sm ghost" id="cmp-expand" type="button">Expand all</button>`);
}

// ── chart ────────────────────────────────────────────────────────────────────
// One line per rollout, cumulative, against minutes since it started. Step-shaped: the value changes only when a
// model call finishes. A crosshair reads every rollout at the pointer's time; the letters at the line ends and the
// legend below carry identity, so colour is never the only cue.
function drawChart(el, st, runs) {
  const box = $("#cmp-chart", el);
  if (!box || !window.d3) return;
  const d3 = window.d3;
  const draw = () => {
    if (!box.isConnected) return;
    const W = Math.max(280, box.clientWidth), H = 240, m = { t: 12, r: 30, b: 26, l: 54 };
    const k = st.metric;
    const series = runs.filter((x) => x.dg.points.length).map((x) => {
      const pts = x.dg.points, last = pts[pts.length - 1];
      const end = Math.max(x.dg.end, last.t);
      return { x, end, data: [{ t: x.dg.start || 0, v: 0 }, ...pts.map((p) => ({ t: p.t, v: p[k] })), { t: end, v: last[k] }] };
    });
    const tmax = d3.max(series, (s) => s.end) || 1, vmax = d3.max(series, (s) => d3.max(s.data, (p) => p.v)) || 1;
    const xs = d3.scaleLinear([0, tmax / 60], [m.l, W - m.r]);
    const ys = d3.scaleLinear([0, vmax], [H - m.b, m.t]).nice(4);
    const fv = k === "cost" ? (v) => money(v) : k === "tokens" ? (v) => tokensShort(v) : (v) => fmt.format(Math.round(v));
    const ft = (v) => (tmax < 120 ? `${Math.round(v * 60)}s` : `${+v.toFixed(1)}m`);
    box.innerHTML = "";
    const svg = d3.select(box).append("svg").attr("viewBox", `0 0 ${W} ${H}`).attr("width", W).attr("height", H).attr("role", "img")
      .attr("aria-label", `${METRICS.find(([x]) => x === k)[1]} over time for rollouts ${series.map((s) => LETTER[s.x.slot]).join(", ")}. The summary table has the final values.`);
    const g = svg.append("g");
    g.selectAll("line.grid").data(ys.ticks(4)).join("line").attr("class", "grid").attr("x1", m.l).attr("x2", W - m.r).attr("y1", (v) => ys(v)).attr("y2", (v) => ys(v));
    g.selectAll("text.yt").data(ys.ticks(4)).join("text").attr("class", "tick yt").attr("x", m.l - 8).attr("y", (v) => ys(v) + 4).attr("text-anchor", "end").text((v) => fv(v));
    g.append("line").attr("class", "base").attr("x1", m.l).attr("x2", W - m.r).attr("y1", H - m.b).attr("y2", H - m.b);
    g.selectAll("text.xt").data(xs.ticks(Math.max(2, Math.floor(W / 170)))).join("text").attr("class", "tick xt").attr("x", (v) => xs(v)).attr("y", H - 8).attr("text-anchor", "middle").text(ft);
    const line = d3.line().x((p) => xs(p.t / 60)).y((p) => ys(p.v)).curve(d3.curveStepAfter);
    for (const s of series) g.append("path").attr("class", "ln").attr("d", line(s.data)).style("stroke", `var(--s${s.x.slot + 1})`);
    const placed = [];
    for (const s of series) {
      const p = s.data[s.data.length - 1], cx = xs(p.t / 60), cy = ys(p.v);
      g.append("circle").attr("class", "end").attr("cx", cx).attr("cy", cy).attr("r", 4).style("fill", `var(--s${s.x.slot + 1})`);
      const lx = Math.min(cx + 8, W - 12);
      if (!placed.some((q) => Math.abs(q.y - cy) < 13 && Math.abs(q.x - lx) < 18)) {   // a label that would collide is left to the legend
        g.append("text").attr("class", "lbl").attr("x", lx).attr("y", cy + 4).text(LETTER[s.x.slot]);
        placed.push({ x: lx, y: cy });
      }
    }
    // crosshair: snaps to the nearest recorded step of any rollout, reads them all
    const times = [...new Set(series.flatMap((s) => s.data.map((p) => p.t)))].sort((a, b) => a - b);
    const hair = g.append("line").attr("class", "hair").attr("y1", m.t).attr("y2", H - m.b).attr("visibility", "hidden");
    const at = (s, t) => { let v = 0; for (const p of s.data) { if (p.t <= t) v = p.v; else break; } return v; };
    const tip = $("#tip");
    svg.append("rect").attr("x", m.l).attr("y", m.t).attr("width", W - m.l - m.r).attr("height", H - m.t - m.b).attr("fill", "transparent")
      .on("pointermove", (ev) => {
        const [px] = d3.pointer(ev), t0 = xs.invert(px) * 60;
        const t = times.reduce((a, b) => (Math.abs(b - t0) < Math.abs(a - t0) ? b : a), times[0]);
        hair.attr("x1", xs(t / 60)).attr("x2", xs(t / 60)).attr("visibility", "visible");
        tip.innerHTML = `<div class="cmp-tip-h">${esc(dur(t))} in</div>${series.map((s) => `<div class="cmp-tip-r"><b>${esc(fv(at(s, t)))}</b>
          <span><i style="background:var(--s${s.x.slot + 1})"></i>${LETTER[s.x.slot]} ${esc(modelName(s.x.run))}${t > s.end ? " · finished" : ""}</span></div>`).join("")}`;
        tip.hidden = false;
        tip.style.left = Math.min(ev.clientX + 14, innerWidth - tip.offsetWidth - 8) + "px";
        tip.style.top = ev.clientY + 16 + "px";
      })
      .on("pointerleave", () => { hair.attr("visibility", "hidden"); tip.hidden = true; });
  };
  draw();
  ro?.disconnect();
  let t;
  ro = new ResizeObserver(() => { clearTimeout(t); t = setTimeout(draw, 80); });
  ro.observe(box);
  st.redraw = draw;
}

// ── interactions ─────────────────────────────────────────────────────────────
async function onClick(e, el, st) {
  const j = e.target.closest("[data-jump]");
  if (j) { e.preventDefault(); $(`#cmp-${j.dataset.jump}`, el)?.scrollIntoView({ behavior: "smooth", block: "start" }); return; }
  const add = e.target.closest("[data-add]");
  if (add && !add.disabled) {
    const free = st.slots.findIndex((x) => !x);
    const slot = free === -1 ? st.slots.length : free;
    if (slot >= MAX) return toast(`Up to ${MAX} rollouts at once`);
    st.slots[slot] = add.dataset.add;
    add.closest("details")?.removeAttribute("open");
    await progress.wrap(fetchRuns(st));
    if (alive) render(el, st);
    return;
  }
  const rm = e.target.closest("[data-remove]");
  if (rm) { st.slots[Number(rm.dataset.remove)] = null; render(el, st); return; }
  if (e.target.closest("[data-refresh]")) {
    visible(st).filter((x) => isLive(x.run)).forEach((x) => st.data.delete(x.id));
    await progress.wrap(fetchRuns(st));
    if (alive) render(el, st);
    return;
  }
  if (e.target.closest("[data-copy]")) {
    try { await navigator.clipboard.writeText(location.href); toast("Link copied"); } catch { toast("Copy the address bar"); }
    return;
  }
  const tb = e.target.closest("[data-tab]");
  if (tb) {
    st.tab = Number(tb.dataset.tab);
    $$(".cmp-tabs [data-tab]", el).forEach((b) => b.setAttribute("aria-pressed", String(b === tb)));
    $$(".cmp-tcol", el).forEach((c) => c.classList.toggle("on", Number(c.dataset.col) === st.tab));
    return;
  }
  const mt = e.target.closest("[data-metric]");
  if (mt) {
    st.metric = mt.dataset.metric;
    $$("[data-metric]", el).forEach((b) => b.setAttribute("aria-pressed", String(b === mt)));
    st.redraw?.();
    return;
  }
  const ex = e.target.closest("#cmp-expand");
  if (ex) {
    const open = ex.textContent === "Expand all";
    $$(".cmp-tl details", el).forEach((d) => (d.open = open));
    ex.textContent = open ? "Collapse all" : "Expand all";
  }
}
