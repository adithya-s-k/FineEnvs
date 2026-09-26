// Task page: one scrolling page built from sections, an outline to jump between them, and a run panel.
// Each section builder returns null when the task has nothing for it, so every domain gets only what applies.
import { $, $$, api, esc, md, money, ago, bytes, openModal, table, sheetGrid, statusPill, rewardBadge, toast, DOMAIN_NAME, vals,
  sk, spinner, emptyState, progress } from "./util.js";
import { icon, DOMAIN_ICON, FILE_ICON } from "./icons.js";
import { getSession, refreshActive } from "./session.js";
import { picker } from "./picker.js";

let modelsP = null;
const getModels = () => (modelsP ||= api("/api/models").catch((e) => { modelsP = null; throw e; }));
// Tokens (input, output) of the test rollouts per domain. Agents re-send their whole context every step, so
// input dominates: a General task read ~1.2M input tokens against ~20k written.
const TYPICAL = { code: [100e3, 3e3], cyber: [1.2e6, 30e3], general: [1.2e6, 20e3], webdev: [60e3, 20e3], music: [500, 6e3] };
const TYPICAL_MIN = { code: 8, cyber: 12, general: 15, webdev: 10, music: 0 };
const KIND_LABEL = { spreadsheet: "Excel", document: "Word", pdf: "PDF", slides: "PowerPoint", image: "image", web: "web page", text: "text", archive: "archive" };
const KIND_NOUN = { spreadsheet: "spreadsheet", document: "document", pdf: "PDF", slides: "slide deck", image: "image", web: "web page", text: "text file", archive: "archive", other: "file" };
const LEFTOVER = /^(None named|Unspecified|Unknown|Unrated|Other)$/;
let spy = null, alive = false;

export function unmount() { alive = false; if (spy) spy.disconnect(); spy = null; }

function skeleton() {
  return `<div class="wrap page">
    <div class="crumbs">${sk.line(22)}</div>
    <div class="tp-head">${sk.line(24, 22)}${sk.line(62, 28)}${sk.line(40)}</div>
    <div class="tp-grid">
      <nav class="toc">${sk.lines(70, 85, 60, 75, 50)}</nav>
      <div class="tp-body">${[5, 3, 4].map((n) => `<div class="panel"><div class="panel-h">${sk.line(24, 14)}</div>
        <div class="panel-b" style="display:grid;gap:10px">${sk.lines(...[100, 96, 88, 92, 70].slice(0, n))}</div></div>`).join("")}</div>
      <aside class="tp-side"><div class="panel"><div class="panel-h">${sk.line(40, 14)}</div>
        <div class="panel-b" style="display:grid;gap:12px">${sk.line(90)}${sk.box(52, "border-radius:9px")}${sk.box(96, "border-radius:10px")}${sk.box(42, "border-radius:9px")}</div></div></aside>
    </div></div>`;
}

export async function mount(el, id) {
  alive = true;
  el.innerHTML = skeleton();
  let v;
  try { v = await api(`/api/tasks/${encodeURIComponent(id)}`); }
  catch (e) {
    if (!alive) return;
    el.innerHTML = `<div class="wrap page">${emptyState(e.status === 404 ? "search" : "alert", e.status === 404 ? "No such environment" : "Couldn't open this environment",
      esc(e.status === 404 ? `There is no task with the id ${id}.` : e.message), `<a class="btn" href="#/">${icon("arrowLeft")}All environments</a>`)}</div>`;
    return;
  }
  if (!alive) return;
  getModels().catch(() => {});   // start fetching models while the page renders
  const sections = [brief, systems, workspace, setup, grading, rollouts].map((f) => f(v)).filter(Boolean);
  el.innerHTML = `
    <div class="wrap page tp fade-in" style="--dc:var(--c-${v.domain})">
      <nav class="crumbs" aria-label="Breadcrumb"><a href="#/">Environments</a>${icon("chevronRight", 13)}
        <a href="#/?d=${v.domain}">${esc(DOMAIN_NAME[v.domain])}</a>${icon("chevronRight", 13)}<span>${esc(v.id)}</span></nav>
      ${header(v)}
      <div class="tp-grid">
        <nav class="toc" aria-label="On this page"><p>On this page</p>${sections.map((s) => `<a href="#" data-jump="${s.id}">${esc(s.nav)}${s.count != null ? `<em>${s.count}</em>` : ""}</a>`).join("")}</nav>
        <div class="tp-body">${sections.map((s) => `<section class="panel tp-sec" id="sec-${s.id}">
          <div class="panel-h"><h2>${icon(s.icon, 15)}${esc(s.title)}</h2>${s.note ? `<span class="aside">${esc(s.note)}</span>` : ""}</div>
          <div class="panel-b">${s.html}</div></section>`).join("")}</div>
        <aside class="tp-side"><div class="panel runbox" id="runbox"><div class="panel-h"><h3>${icon("play", 14)}Run a rollout</h3></div>
          <div class="panel-b"><div class="loading-row">${spinner()}Loading models from Inference Providers…</div></div></div></aside>
      </div>
    </div>`;
  wire(el, v);
  runPanel($("#runbox", el), v);
  loadRollouts(el, v);
}

function header(v) {
  const vf = v.verify || {};
  const chips = Object.values(v.facets || {}).flatMap(vals).filter((x) => !LEFTOVER.test(x)).slice(0, 6);
  const facts = [];
  if (v.systems) facts.push(["plug", `<b>${v.systems.length}</b> systems`], ["terminal", `<b>${v.systems.reduce((n, s) => n + s.tools.length, 0)}</b> tools`], ["folder", `<b>${v.files.length}</b> files`]);
  if (vf.checks && vf.kind === "rubric") facts.push(["check", `<b>${vf.checks.length}</b> rubric checks`]);
  if (vf.kind === "tests" || vf.kind === "terminal") facts.push(["flask", `<b>${vf.files.length}</b> hidden test files`]);
  if (vf.kind === "crash" && vf.expected?.function) facts.push(["target", `crash in <code>${esc(vf.expected.function)}</code>`]);
  if (vf.kind === "visual") facts.push(["image", "graded from a screenshot"]);
  if (vf.kind === "music") facts.push(["gauge", "scored by code, no model"]);
  if (vf.needs_judge) facts.push(["scale", `needs a ${vf.needs_judge} judge`]);
  return `<header class="tp-head">
    <div class="kick"><span class="badge">${icon(DOMAIN_ICON[v.domain], 13)}${esc(DOMAIN_NAME[v.domain])}</span>${chips.map((c) => `<span class="chip">${esc(c)}</span>`).join("")}</div>
    <h1>${esc(v.title)}</h1>
    <div class="facts">${facts.map(([i, f]) => `<span>${icon(i, 14)}${f}</span>`).join("")}
      <button class="link" data-copy>${icon("link", 14)}Copy link</button></div>
  </header>`;
}

// ── sections ─────────────────────────────────────────────────────────────────
function brief(v) {
  const vf = v.verify || {};
  let spec = "";
  if (vf.kind === "music" && vf.spec) {
    const s = vf.spec;
    spec = `<div class="spec">${[["Style", s.style], ["Tempo", s.bpm && `${s.bpm} BPM`], ["Meter", s.meter], ["Length", s.bars && `${s.bars} bars`], ["Voices", s.voices]]
      .filter(([, x]) => x).map(([k, x]) => `<div class="stat"><span>${k}</span><b>${esc(x)}</b></div>`).join("")}</div>`;
  }
  const meta = (v.meta || []).filter(([, x]) => x && String(x).length < 200 && !/^(none named|unspecified|unknown|n\/a|none)$/i.test(String(x).trim()));
  return {
    id: "brief", nav: "The task", icon: "message", title: "The task", note: "exactly what the agent receives",
    html: `${spec}<div class="prose">${md(v.brief)}</div>
      ${meta.length ? `<div class="brief-meta"><dl class="kv">${meta.map(([k, x]) => `<dt>${esc(k)}</dt><dd>${esc(x)}</dd>`).join("")}</dl></div>` : ""}
      ${v.links?.length ? `<div class="brief-links">${v.links.map((l) => `<a class="btn sm" href="${esc(l.url)}" target="_blank" rel="noopener">${esc(l.label)}${icon("external", 13)}</a>`).join("")}</div>` : ""}`,
  };
}

function systems(v) {
  if (!v.systems?.length) return null;
  return {
    id: "systems", nav: "Systems", count: v.systems.length, icon: "plug", title: "Systems the agent works through",
    note: "MCP servers, each backed by its own database",
    html: `<p class="muted sm" style="margin-bottom:12px">The agent can only reach this data through the tools below. Open a table to see the rows it starts with.</p>
      <div class="systems">${v.systems.map((s, i) => `
      <details class="sys" ${i === 0 ? "open" : ""}>
        <summary>${icon("chevronRight", 14, "chev")}<b>${esc(s.label)}</b><span class="meta">${s.tools.length} tools · ${s.tables.length} tables · ${s.tables.reduce((n, t) => n + t.rows, 0).toLocaleString()} rows</span></summary>
        <div class="sys-in">
          <div><div class="sec-label">Tools</div>
            <ul class="tools">${s.tools.map((t) => `<li><code class="sig">${esc(t.name)}(<span>${t.params.map((p) => esc(p.name) + (p.default != null ? "?" : "")).join(", ")}</span>)</code>
              ${t.doc ? `<p>${esc(t.doc)}</p>` : ""}</li>`).join("")}</ul></div>
          <div><div class="sec-label">Data</div>
            <div class="tables">${s.tables.map((t) => `<button class="tbl-btn" data-sys="${esc(s.name)}" data-label="${esc(s.label)}" data-table="${esc(t.table)}">${icon("table", 15)}
              <b>${esc(t.table)}</b><span>${t.rows.toLocaleString()} × ${t.columns.length}</span></button>`).join("")}</div></div>
        </div>
      </details>`).join("")}</div>`,
  };
}

function workspace(v) {
  if (!v.files?.length) return null;
  const kinds = {};
  v.files.forEach((f) => (kinds[f.kind] = (kinds[f.kind] || 0) + 1));
  return {
    id: "files", nav: "Workspace", count: v.files.length, icon: "folder", title: "Workspace files",
    note: Object.entries(kinds).map(([k, n]) => `${n} ${KIND_NOUN[k] || k}${n > 1 ? "s" : ""}`).join(" · "),
    html: `<div class="files-grid">${v.files.map((f) => `<button class="file k-${esc(f.kind)}" data-path="${esc(f.path)}" data-kind="${esc(f.kind)}" title="${esc(f.path)}">
      <span class="fi">${icon(FILE_ICON[f.kind] || "file", 16)}</span><span class="fx"><span class="fn">${esc(f.path).replace(/([_/.-])/g, "$1<wbr>")}</span><span class="fs">${esc(KIND_LABEL[f.kind] || f.kind)} · ${bytes(f.size)}</span></span></button>`).join("")}</div>`,
  };
}

function setup(v) {
  const e = v.environment || {};
  const rows = [];
  if (e.sandbox === false) rows.push(["Where it runs", "No sandbox: one model call, then the scorer"]);
  if (e.image) rows.push(["Sandbox image", e.image.replace("docker.io/", "")]);
  if (e.cwd) rows.push(["Working directory", e.cwd]);
  if (e.deliver) rows.push(["Deliverable", `${e.deliver}/index.html (and its assets)`]);
  if (e.ports) rows.push(["MCP ports", e.ports.join(", ")]);
  if (e.cpus) rows.push(["Resources", `${e.cpus} CPU · ${e.memory_mb} MB · internet ${e.internet ? "on" : "off"}`]);
  if (e.tags) rows.push(["Tags", e.tags.join(", ")]);
  const how = {
    code: "Git history is hidden while the agent works, so the fix cannot be read out of later commits.",
    cyber: "The agent runs as an unprivileged user. It gets the project source, a prebuilt fuzz binary and submit.sh; the verifier runs as root.",
    general: "The agent is an unprivileged user in the workspace. The systems' databases and code are root-only, so MCP is the only way in.",
    webdev: "Node, pnpm, Playwright and Chromium are in the image; the agent builds however it likes and delivers to dist/.",
    music: "",
  }[v.domain];
  if (!rows.length && !how) return null;
  return {
    id: "setup", nav: "Sandbox", icon: "box", title: "How the sandbox is set up",
    html: `${rows.length ? `<dl class="kv">${rows.map(([k, x]) => `<dt>${esc(k)}</dt><dd><code>${esc(x)}</code></dd>`).join("")}</dl>` : ""}
      ${how ? `<div class="note-box" style="margin-top:14px">${icon("shield")}<span>${esc(how)} Web tools are off for the agent, and curl, wget and git fetch are blocked in its shell.</span></div>` : ""}`,
  };
}

function grading(v) {
  const g = v.verify;
  if (!g) return null;
  let body = `<p class="lead">${esc(g.summary)}</p>`;
  if (g.steps) body += `<ol class="steps">${g.steps.map((s) => `<li><span>${md(s).replace(/^<p>|<\/p>$/g, "")}</span></li>`).join("")}</ol>`;
  if (g.kind === "tests") {
    body += `<div class="sec-label" style="margin-top:18px">Hidden tests</div><div class="tbl"><table><thead><tr><th>File</th><th>Lines</th><th></th></tr></thead><tbody>${g.files.map((f) =>
      `<tr><td><code>${esc(f.path)}</code></td><td class="pm"><span class="a">+${f.added}</span> <span class="d">−${f.removed}</span></td><td>${f.new ? '<span class="chip">new</span>' : ""}</td></tr>`).join("")}</tbody></table></div>`;
    if (g.script) body += disclose("Test command script", `<pre class="code-block"><code>${esc(g.script)}</code></pre>`);
    if (g.patch) body += disclose(`Full hidden test patch (${g.patch.length.toLocaleString()} characters)`, `<pre class="code-block diff">${diffHtml(g.patch)}</pre>`);
  }
  if (g.kind === "crash" && g.expected) {
    const x = g.expected;
    body += `<div class="target">${[["Sanitizer", esc(x.sanitizer)], ["Bug class", esc(x.error_type)], ["Function", `<code>${esc(x.function)}</code>`], ["File", `<code>${esc(x.file)}</code>`]]
      .map(([k, val]) => `<div class="stat"><span>${k}</span><b>${val}</b></div>`).join("")}</div>`;
  }
  if (g.kind === "rubric") {
    // verify.py counts a check with no weight as weight 1 (weights are relative, not percentages)
    const wt = (c) => (c.weight == null ? 1 : Number(c.weight));
    const tot = g.checks.reduce((s, c) => s + wt(c), 0) || 1;
    const top = Math.max(...g.checks.map(wt)) || 1;
    body += `<div class="sec-label" style="margin-top:18px">${g.checks.length} checks, weighted</div><ol class="checks">${g.checks.map((c) => {
      const pct = Math.round((wt(c) / tot) * 100);
      return `<li><div class="ck-top"><span class="tier t-${esc(c.tier)}">${esc(c.tier || "")}</span>
      <span>${c.method === "llm" ? "judged by a model" : "checked by code"}</span>
      <span class="w" title="Share of the reward"><i><b style="width:${Math.round((wt(c) / top) * 100)}%"></b></i>${pct}%</span></div>
      <p>${esc(c.question || c.id)}</p></li>`; }).join("")}</ol><p class="muted xs" style="margin-top:10px">The answer each check expects is not shown.</p>`;
  }
  if (g.kind === "terminal") {
    body += `<div class="sec-label" style="margin-top:18px">Hidden test files</div>${table([["File", "Size"], ...g.files.map((f) => [f.path, bytes(f.size)])])}`;
    if (g.script) body += disclose("tests/test.sh", `<pre class="code-block"><code>${esc(g.script)}</code></pre>`);
    if (g.tests) body += disclose("tests/test_outputs.py", `<pre class="code-block"><code>${esc(g.tests)}</code></pre>`);
  }
  if (g.kind === "visual") {
    const parts = [["visual", "Visual quality", "mean of five criteria"], ["query_fulfillment", "Brief fulfilment", ""], ["premium_assets", "Asset quality", ""]];
    body += `<div class="parts">${parts.map(([key, name, how]) => {
      const ds = g.dims.filter((d) => d.group === key);
      return ds.length ? `<div class="rpart"><div class="rpart-h"><b>${name}</b><span>⅓ of the score${how ? ` · ${how}` : ""}</span></div>
        <dl>${ds.map((d) => `<div><dt>${esc(d.label)}</dt><dd>${esc(d.desc)}</dd></div>`).join("")}</dl></div>` : "";
    }).join("")}</div>
      <p class="muted xs" style="margin-top:12px">${esc(g.note)}</p>`;
  }
  if (g.kind === "music") {
    const groups = {};
    g.features.forEach((f) => (groups[f.group] ||= []).push(f));
    body += `<div class="note-box gated">${icon("alert")}<span><b>Validity gate first.</b> No notation errors, fewer than 10 bars of the wrong length, no blank lines in the tune,
      and one instrument per MIDI channel. A piece that fails any of these scores 0.</span></div>
      <div class="feat-groups">${Object.entries(groups).map(([gname, fs]) => `<div><div class="sec-label">${esc(gname)}</div>
      ${fs.map((f) => `<div class="feat"><b>${esc(f.label || f.name)}</b><span>${f.rule === "band" ? "within the human range" : f.rule === "high" ? "the higher the better" : "the lower the better"} · <code>${esc(f.name)}</code></span>
        ${f.band ? `<em>${fmtNum(f.band[0])} – ${fmtNum(f.band[1])}</em>` : ""}</div>`).join("")}</div>`).join("")}</div>`;
  }
  return { id: "grading", nav: "Grading", icon: "scale", title: "How it is graded", note: g.needs_judge ? `needs a ${g.needs_judge} judge model` : "no model in the loop", html: body };
}

function rollouts() {
  return { id: "runs", nav: "Your rollouts", icon: "list", title: "Your rollouts on this task",
    html: `<div id="task-runs"><div class="loading-row">${spinner()}Loading your rollouts…</div></div>` };
}

const disclose = (label, inner) => `<details style="margin-top:12px"><summary class="disclose">${icon("chevronRight", 14, "chev")}${esc(label)}</summary>${inner}</details>`;
const fmtNum = (x) => (x == null ? "" : Math.abs(x) >= 100 ? x.toFixed(0) : Math.abs(x) >= 1 ? x.toFixed(2) : x.toFixed(3));

function diffHtml(patch) {
  return patch.split("\n").slice(0, 4000).map((l) => {
    const cls = l.startsWith("+++") || l.startsWith("---") || l.startsWith("diff ") ? "dh" : l.startsWith("+") ? "da" : l.startsWith("-") ? "dd" : l.startsWith("@@") ? "dc" : "";
    return `<span class="${cls}">${esc(l)}</span>`;
  }).join("\n");
}

// ── interactions: file previews, tables, outline ─────────────────────────────
function wire(el, v) {
  el.addEventListener("click", async (ev) => {
    const j = ev.target.closest("[data-jump]");
    if (j) { ev.preventDefault(); $(`#sec-${j.dataset.jump}`, el).scrollIntoView({ behavior: "smooth", block: "start" }); return; }
    if (ev.target.closest("[data-copy]")) { try { await navigator.clipboard.writeText(location.href); toast("Link copied"); } catch { toast("Copy the address bar"); } return; }
    const f = ev.target.closest(".file");
    if (f) return previewFile(v, f.dataset.path, f.dataset.kind);
    const t = ev.target.closest(".tbl-btn");
    if (t) return previewTable(v, t.dataset.sys, t.dataset.table, t.dataset.label);
  });
  const links = new Map($$("[data-jump]", el).map((a) => [a.dataset.jump, a]));
  spy = new IntersectionObserver((es) => es.forEach((e) => {
    if (e.isIntersecting) { links.forEach((a) => a.classList.remove("on")); links.get(e.target.id.slice(4))?.classList.add("on"); }
  }), { rootMargin: "-20% 0px -70% 0px" });
  $$(".tp-sec", el).forEach((s) => spy.observe(s));
}

const previewSkeleton = (kind) => kind === "spreadsheet"
  ? `<div style="display:flex;gap:6px;margin-bottom:12px">${sk.box(28, "width:90px;border-radius:7px")}${sk.box(28, "width:70px;border-radius:7px")}</div>${sk.box(360, "border-radius:10px")}`
  : `<div style="max-width:860px;margin:0 auto;display:grid;gap:10px">${sk.line(40, 18)}${sk.lines(100, 96, 90, 98, 70)}<div style="height:8px"></div>${sk.lines(94, 100, 88, 60)}</div>`;

async function previewFile(v, path, kind) {
  const raw = `/api/tasks/${encodeURIComponent(v.id)}/file?path=${encodeURIComponent(path)}`;
  const body = openModal(path, previewSkeleton(kind), { raw, kind });
  if (kind === "image") { body.innerHTML = `<div class="pv-img"><img src="${raw}" alt="${esc(path)}"></div>`; return; }
  if (kind === "web") { body.innerHTML = `<iframe class="pv-frame" sandbox src="${raw}" title="${esc(path)}"></iframe>`; return; }
  let p;
  try { p = await progress.wrap(api(`/api/tasks/${encodeURIComponent(v.id)}/preview?path=${encodeURIComponent(path)}`)); }
  catch (e) { body.innerHTML = emptyState("alert", "Couldn't open this file", esc(e.message)); return; }
  if (!body.isConnected || $("#modal").hidden) return;
  if (p.type === "sheets") {
    body.innerHTML = `<div class="pv-tabs" role="tablist">${p.sheets.map((s, i) => `<button role="tab" data-i="${i}" aria-selected="${i === 0}">${icon("sheet", 13)}${esc(s.name)}${s.dims ? `<em>${esc(s.dims)}</em>` : ""}</button>`).join("")}</div><div id="pv-sheet"></div>`;
    const show = (i) => {
      $("#pv-sheet", body).innerHTML = sheetGrid(p.sheets[i].rows) + (p.sheets[i].rows.length >= 60 ? `<p class="pv-foot">Showing the first 60 rows and 24 columns. Open the original for the whole workbook.</p>` : "");
      $$(".pv-tabs button", body).forEach((b) => b.setAttribute("aria-selected", b.dataset.i == i));
    };
    body.querySelector(".pv-tabs").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) show(+b.dataset.i); });
    show(0);
  } else if (p.type === "document") {
    body.innerHTML = `<div class="pv-doc prose">${p.blocks.map((b) => b.table ? table(b.table, { header: true }) : b.h ? `<h${Math.min(b.h + 2, 5)}>${esc(b.text)}</h${Math.min(b.h + 2, 5)}>` : `<p>${esc(b.text)}</p>`).join("")}</div>`;
  } else if (p.type === "slides") {
    body.innerHTML = `${p.repeated?.length ? `<div class="note-box" style="margin-bottom:14px">${icon("info")}<span>On every slide, not repeated below: ${p.repeated.map((t) => `<b>${esc(t)}</b>`).join(" · ")}</span></div>` : ""}
      <div class="pv-slides">${p.slides.map((s) => `<article class="slide"><span class="n">${s.n}</span><div>
        <b>${esc(s.title || "Untitled slide")}</b>${s.text.map((t) => `<p>${esc(t).replace(/\n/g, "<br>")}</p>`).join("")}
        ${(s.tables || []).map((t) => table(t)).join("")}</div></article>`).join("")}</div>`;
  } else if (p.type === "pdf") {
    body.innerHTML = `<div class="pv-tabs"><button data-m="view" aria-selected="true">${icon("doc", 13)}Document</button><button data-m="text" aria-selected="false">${icon("list", 13)}Text <em>${p.page_count} pages</em></button></div>
      <div id="pv-pdf"><iframe class="pv-frame" src="${raw}" title="${esc(path)}"></iframe></div>`;
    body.querySelector(".pv-tabs").addEventListener("click", (e) => {
      const b = e.target.closest("button"); if (!b) return;
      $$(".pv-tabs button", body).forEach((x) => x.setAttribute("aria-selected", x === b));
      $("#pv-pdf", body).innerHTML = b.dataset.m === "view" ? `<iframe class="pv-frame" src="${raw}" title="${esc(path)}"></iframe>`
        : `<div class="pv-doc">${p.pages.map((pg) => `<div class="sec-label">Page ${pg.n}</div><pre class="code-block wrap">${esc(pg.text)}</pre>`).join("")}</div>`;
    });
  } else if (p.type === "text") body.innerHTML = `<pre class="code-block wrap" style="max-height:none">${esc(p.text)}</pre>`;
  else if (p.type === "archive") body.innerHTML = table([["Entry", "Size"], ...p.entries.map((x) => [x.name, bytes(x.size)])]);
  else if (p.type === "error") body.innerHTML = emptyState("alert", "Couldn't preview this file", esc(p.error), `<a class="btn" href="${raw}&download=1">Download</a>`);
  else body.innerHTML = emptyState("file", "No preview for this file type", "", `<a class="btn" href="${raw}&download=1">Download</a>`);
}

async function previewTable(v, sys, tbl, label) {
  const body = openModal(`${label || sys} · ${tbl}`, sk.box(420, "border-radius:10px"), { kind: "table" });
  try {
    const d = await progress.wrap(api(`/api/tasks/${encodeURIComponent(v.id)}/systems/${encodeURIComponent(sys)}/${encodeURIComponent(tbl)}?limit=100`));
    if ($("#modal").hidden) return;
    body.innerHTML = table([d.columns, ...d.rows.map((r) => r.map((c) => (c == null ? "" : String(c))))]) + `<p class="pv-foot">${d.rows.length >= 100 ? "First 100 rows" : `${d.rows.length} row${d.rows.length === 1 ? "" : "s"}`} · ${d.columns.length} columns · the state of this database when the agent starts</p>`;
  } catch (e) { body.innerHTML = emptyState("alert", "Couldn't load this table", esc(e.message)); }
}

// ── the run panel ────────────────────────────────────────────────────────────
// Settings that are worth keeping between visits live in this browser. An endpoint's API key stays in this tab only.
const LS = { get: (k, d) => { try { return JSON.parse(localStorage.getItem(k)) ?? d; } catch { return d; } },
             set: (k, v) => { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* private mode */ } } };
const SS = { get: (k) => { try { return sessionStorage.getItem(k) || ""; } catch { return ""; } },
             set: (k, v) => { try { sessionStorage.setItem(k, v); } catch { /* private mode */ } } };
const THINKING = [["default", "Model default"], ["none", "Off"], ["low", "Low"], ["medium", "Medium"], ["high", "High"]];

async function runPanel(box, v) {
  const s = getSession();
  const inner = $(".panel-b", box);
  if (!v.runnable) { inner.innerHTML = `<p class="muted sm">This task can't be run here yet.</p>`; return; }
  let cat;
  try { cat = await getModels(); }
  catch (e) {
    if (!alive) return;
    inner.innerHTML = `<div class="note-box err">${icon("alert")}<span>Couldn't reach HF Inference Providers: ${esc(e.message)}</span></div>
      <button class="btn block" id="rb-retry">${icon("refresh")}Try again</button>`;
    $("#rb-retry", inner).addEventListener("click", () => { inner.innerHTML = `<div class="loading-row">${spinner()}Loading models…</div>`; runPanel(box, v); });
    return;
  }
  if (!alive) return;
  const music = v.domain === "music";
  const need = v.verify?.needs_judge;
  const judges = need === "vision" ? cat.vision_judges : need === "text" ? cat.text_judges : [];
  const gaps = s.missing_scopes || [];
  const d = v.run_defaults || {};
  const ep = LS.get("byo", { base_url: "", model: "", price_in: "", price_out: "" });
  const adv = LS.get("adv", { thinking: "default", temperature: "" });
  const st = { source: LS.get("source", "hf"), probe: null };
  inner.innerHTML = `
    <p class="intro">${music ? "One model call, then Xiaomi's scorer. No sandbox." : "A fresh HF Sandbox from this task's image, OpenCode as the harness, then the task's own grader."}</p>
    <div class="seg src" role="tablist" aria-label="Where the model runs">
      <button type="button" data-src="hf">HF Inference Providers</button><button type="button" data-src="byo">Your endpoint</button></div>
    <div class="field" data-pane="hf"><span>Agent model</span><div id="rb-model"></div></div>
    <div class="byo" data-pane="byo">
      <label class="field"><span>Base URL <em>any OpenAI-compatible API</em></span>
        <input class="input mono" id="ep-url" placeholder="https://your-litellm.example.com/v1" value="${esc(ep.base_url)}" spellcheck="false" autocomplete="off"></label>
      <label class="field"><span>API key <em>kept in this tab only</em></span>
        <input class="input mono" id="ep-key" type="password" placeholder="sk-…" value="${esc(SS.get("byo-key"))}" spellcheck="false" autocomplete="off"></label>
      <label class="field"><span>Model <button class="link" type="button" id="ep-load">${icon("refresh", 12)}Load models</button></span>
        <input class="input mono" id="ep-model" list="ep-models" placeholder="gpt-5, claude-sonnet-5, openai/gpt-oss-120b…" value="${esc(ep.model)}" spellcheck="false" autocomplete="off">
        <datalist id="ep-models"></datalist></label>
      <div class="field"><span>Price per 1M tokens <em>optional, only for the cost shown</em></span>
        <div class="pair"><input class="input" id="ep-pin" type="number" min="0" step="0.01" placeholder="input $" value="${esc(ep.price_in)}">
          <input class="input" id="ep-pout" type="number" min="0" step="0.01" placeholder="output $" value="${esc(ep.price_out)}"></div></div>
      <button class="btn block" type="button" id="ep-test">${icon("plug", 14)}Test connection</button>
      <div id="ep-status"></div>
      <p class="fine">Works with a LiteLLM proxy, vLLM, OpenAI, Together, OpenRouter and the like. The sandbox still runs on your
        Hugging Face account, so you sign in with HF either way. The key is used for this rollout only and never stored on the server,
        but the agent's own processes in the sandbox can see it: use a key you can revoke.</p>
    </div>
    ${judges.length ? `<div class="field"><span>Judge model <em>${need === "vision" ? "scores a screenshot of the page" : "answers each rubric check"}</em></span><div id="rb-judge"></div></div>` : ""}
    <details class="adv" id="rb-adv"><summary class="disclose">${icon("chevronRight", 14, "chev")}Advanced settings<span class="adv-sum" id="adv-sum"></span></summary>
      <div class="adv-grid">
        <label class="field"><span>Thinking</span><select class="input" id="p-think">${THINKING.map(([k, l]) => `<option value="${k}" ${adv.thinking === k ? "selected" : ""}>${l}</option>`).join("")}</select></label>
        <label class="field"><span>Temperature</span><input class="input" id="p-temp" type="number" min="0" max="2" step="0.1" placeholder="default" value="${esc(adv.temperature)}"></label>
        ${music ? "" : `<label class="field"><span>Step cap</span><input class="input" id="p-steps" type="number" min="1" max="1000" placeholder="${d.steps}" value=""></label>
        <label class="field"><span>Time limit, min</span><input class="input" id="p-time" type="number" min="2" max="120" placeholder="${d.timeout_min}" value=""></label>`}
        <label class="field"><span>Max tokens${music ? "" : " / step"}</span><input class="input" id="p-max" type="number" min="256" max="128000" step="256" placeholder="${d.max_tokens || "default"}" value=""></label>
      </div>
      <p class="fine" style="text-align:left">Thinking is sent as <code>reasoning_effort</code>; "Off" disables it on models that support that.
        ${music ? "" : `The step cap and time limit default to the values Xiaomi's training harness uses for this domain.`}</p>
    </details>
    <div class="estimate" id="rb-est"></div>
    ${gaps.length ? `<div class="note-box warn">${icon("alert")}<span>Your sign-in may be missing <b>${esc(gaps.join(", "))}</b>. If the rollout fails to start, sign in again with a write token.</span></div>` : ""}
    ${s.user ? `<button class="btn primary lg block" id="rb-go">${icon("play", 15)}Run rollout</button>`
      : `<button class="btn primary lg block" type="button" data-signin>${icon("user", 15)}Sign in to run</button>`}
    <p class="fine">Keeps running if you close this page; find it under <a href="#/runs">Rollouts</a>. The sandbox is billed to ${s.user ? `<b>${esc(s.user.name)}</b>` : "your account"} on Hugging Face.</p>`;

  const def = cat.agents.some((m) => m.id === cat.default_agent) ? cat.default_agent : cat.agents[0]?.id;
  const sel = picker($("#rb-model", box), { models: cat.agents, value: def, onChange: () => update(),
    note: "Every model here supports tool calling. Prices are per million tokens, input / output, at the cheapest provider that can call tools." });
  const judgeSel = judges.length ? picker($("#rb-judge", box), { models: judges.map((j) => ({ ...j, featured: true })), value: judges[0].id, onChange: () => {}, groupLabel: "Tested judges",
    note: need === "vision" ? "Tested on a real render with Xiaomi's rubric. Judges disagree (0.34–0.87 on the same page), so compare webdev scores only between runs graded by the same judge." : "Tested on this dataset's real rubric prompt: each returned a usable verdict on every check." }) : null;
  const val = (id) => { const e = $(id, box); return e ? e.value.trim() : ""; };
  const num = (id) => { const x = val(id); return x === "" ? null : Number(x); };
  const endpoint = () => ({ base_url: val("#ep-url"), api_key: val("#ep-key") || null, model: val("#ep-model"),
                            price_in: num("#ep-pin"), price_out: num("#ep-pout") });
  const params = () => {
    const p = { thinking: val("#p-think") || "default" };
    if (num("#p-temp") != null) p.temperature = num("#p-temp");
    if (num("#p-max") != null) p.max_tokens = num("#p-max");
    if (num("#p-steps") != null) p.steps = num("#p-steps");
    if (num("#p-time") != null) p.timeout_min = num("#p-time");
    return p;
  };
  const save = () => {
    const e = endpoint();
    LS.set("byo", { base_url: e.base_url, model: e.model, price_in: val("#ep-pin"), price_out: val("#ep-pout") });
    SS.set("byo-key", val("#ep-key"));
    LS.set("adv", { thinking: val("#p-think"), temperature: val("#p-temp") });
  };
  const tok = (n) => (n >= 1e6 ? (n / 1e6).toFixed(1) + "M" : Math.round(n / 1e3) + "k");
  const [ti, to] = TYPICAL[v.domain];
  function update() {
    save();
    $$("[data-src]", box).forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.src === st.source)));
    $$("[data-pane]", box).forEach((p) => (p.hidden = p.dataset.pane !== st.source));
    const p = params();
    const changed = [p.thinking !== "default" && `thinking ${THINKING.find(([k]) => k === p.thinking)[1].toLowerCase()}`,
      p.temperature != null && `temp ${p.temperature}`, p.steps && `${p.steps} steps`, p.timeout_min && `${p.timeout_min} min`, p.max_tokens && `${p.max_tokens} max tokens`].filter(Boolean);
    $("#adv-sum", box).textContent = changed.length ? changed.join(" · ") : "";
    let price;
    if (st.source === "byo") {
      const e = endpoint();
      price = e.price_in != null || e.price_out != null ? [e.price_in || 0, e.price_out || 0] : null;
    } else price = [sel.model.input, sel.model.output];
    const minutes = p.timeout_min ? Math.min(p.timeout_min, TYPICAL_MIN[v.domain] || 0) : TYPICAL_MIN[v.domain];
    const sandbox = music ? 0 : (minutes / 60) * ((cat.sandbox_price_per_hour || {})[v.domain === "webdev" ? "cpu-upgrade" : "cpu-basic"] ?? 0.01);
    const model = price ? (ti * price[0] + to * price[1]) / 1e6 : null;
    $("#rb-est", box).innerHTML = `
      <div class="row"><span>Model · ${tok(ti)} in, ${tok(to)} out</span><b>${model == null ? "your provider's price" : money(model)}</b></div>
      ${music ? "" : `<div class="row"><span>Sandbox · ~${minutes} min</span><b>${money(sandbox)}</b></div>`}
      <div class="row total"><span>Typical ${esc(DOMAIN_NAME[v.domain])} rollout</span><b>${model == null ? `${money(sandbox)} + tokens` : `~${money(model + sandbox)}`}</b></div>
      ${(v.domain === "general" || v.domain === "cyber") && st.source === "hf" ? `<div class="why">Agents re-read their context every step, so input tokens dominate. A cheaper model makes a big difference here.</div>` : ""}`;
    const go = $("#rb-go", box);
    if (go && !go.dataset.busy) {
      const ready = st.source === "hf" || (st.probe?.ok && (music || st.probe.tools));
      go.disabled = !ready;
      go.title = ready ? "" : "Test the connection first";
    }
  }
  const status = (kind, html) => { $("#ep-status", box).innerHTML = html ? `<div class="note-box ${kind}">${icon(kind === "err" ? "alert" : kind === "warn" ? "alert" : "check")}<span>${html}</span></div>` : ""; };
  box.addEventListener("click", async (e) => {
    const b = e.target.closest("[data-src]");
    if (b) { st.source = b.dataset.src; LS.set("source", st.source); update(); return; }
    if (e.target.closest("#ep-load")) {
      const btn = $("#ep-load", box); btn.disabled = true;
      try {
        const r = await api("/api/endpoints/models", { method: "POST", body: { base_url: val("#ep-url"), api_key: val("#ep-key") || null } });
        $("#ep-models", box).innerHTML = r.models.map((m) => `<option value="${esc(m)}">`).join("");
        status("", ""); toast(`${r.models.length} models found. Start typing to pick one.`);
        $("#ep-model", box).focus();
      } catch (err) { status("err", esc(err.status === 401 ? "Sign in first: this server makes the call for you." : err.message)); }
      btn.disabled = false; return;
    }
    if (e.target.closest("#ep-test")) {
      const btn = $("#ep-test", box); btn.disabled = true; btn.innerHTML = `${spinner()}Testing…`;
      try {
        const r = await api("/api/endpoints/test", { method: "POST", body: { base_url: val("#ep-url"), api_key: val("#ep-key") || null, model: val("#ep-model") } });
        st.probe = r;
        status(r.tools || music ? "ok" : "warn", `${esc(r.detail)} <span class="faint">${r.ms} ms</span>${!r.tools && !music ? " Agents need tool calling, so this endpoint can't run this task." : ""}`);
      } catch (err) { st.probe = null; status("err", esc(err.status === 401 ? "Sign in first: this server makes the test call for you." : err.message)); }
      btn.disabled = false; btn.innerHTML = `${icon("plug", 14)}Test connection`; update(); return;
    }
  });
  box.addEventListener("input", (e) => { if (e.target.closest(".byo") && !e.target.matches("#ep-pin, #ep-pout")) st.probe = null; update(); });
  box.addEventListener("change", update);
  update();

  const go = $("#rb-go", box);
  if (go) go.addEventListener("click", async () => {
    go.disabled = true; go.dataset.busy = "1"; go.innerHTML = `${spinner()}Starting…`;
    const body = { task_id: v.id, judge: judgeSel?.value || null, params: params() };
    if (st.source === "byo") body.endpoint = endpoint(); else body.model = sel.value;
    try {
      const run = await api("/api/runs", { method: "POST", body });
      refreshActive();
      location.hash = `#/run/${run.id}`;
    } catch (e) {
      toast(e.message, 5000);
      delete go.dataset.busy; go.innerHTML = `${icon("play", 15)}Run rollout`; update();
      if (e.status === 401) document.querySelector("[data-signin]")?.click();
    }
  });
}

async function loadRollouts(el, v) {
  const box = $("#task-runs", el);
  if (!getSession().user) { box.innerHTML = `<p class="muted sm">Sign in to run this task and see your rollouts here.</p>`; return; }
  try {
    const { runs } = await api(`/api/runs?task_id=${encodeURIComponent(v.id)}`);
    if (!box.isConnected) return;
    box.innerHTML = runs.length ? `<div class="runlist">${runs.map((r) => `<a class="runrow" href="#/run/${esc(r.id)}">${statusPill(r.status)}
      <span class="m">${esc(r.model.split("/")[1] || r.model)}</span>${rewardBadge(r.reward, r.status)}<span class="c">${money((r.cost || {}).total)}</span><span class="w2">${ago(r.created_at)}</span></a>`).join("")}</div>`
      : `<p class="muted sm">None yet. Pick a model and run one.</p>`;
  } catch (e) { box.innerHTML = `<p class="err-text sm">${esc(e.message)}</p>`; }
}
