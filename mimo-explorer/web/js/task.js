// Task page: one scrolling page built from sections, an outline to jump between them, and a run panel.
// Each section builder returns null when the task has nothing for it, so every domain gets only what applies.
import { $, $$, api, esc, md, money, ago, bytes, openModal, table, sheetGrid, statusPill, rewardBadge, toast, DOMAIN_NAME, vals } from "./util.js";
import { getSession, refreshActive } from "./session.js";
import { picker } from "./picker.js";

let modelsP = null;
const getModels = () => (modelsP ||= api("/api/models"));
// Tokens (input, output) of the test rollouts per domain. Agents re-send their whole context every step, so
// input dominates: a General task read ~1.2M input tokens against ~20k written.
const TYPICAL = { code: [100e3, 3e3], cyber: [1.2e6, 30e3], general: [1.2e6, 20e3], webdev: [60e3, 20e3], music: [500, 6e3] };
const GLYPH = { spreadsheet: "▦", document: "¶", pdf: "▤", slides: "▭", image: "◩", web: "◎", text: "≡", archive: "◫", other: "•" };
let spy = null;

export function unmount() { if (spy) spy.disconnect(); spy = null; }

export async function mount(el, id) {
  el.innerHTML = `<div class="wrap page"><p class="loading">Opening ${esc(id)}…</p></div>`;
  let v;
  try { v = await api(`/api/tasks/${encodeURIComponent(id)}`); }
  catch (e) {
    el.innerHTML = `<div class="wrap page"><p class="empty">Could not open this task: ${esc(e.message)}</p><p><a href="#/">← All environments</a></p></div>`;
    return;
  }
  const sections = [brief, systems, workspace, setup, grading, rollouts].map((f) => f(v)).filter(Boolean);
  el.innerHTML = `
    <div class="wrap page tp" style="--dc:var(--c-${v.domain})">
      <a class="back" href="#/">← All environments</a>
      ${header(v)}
      <div class="tp-grid">
        <nav class="tp-outline" aria-label="On this page">${sections.map((s) => `<a href="#" data-jump="${s.id}">${esc(s.nav)}</a>`).join("")}</nav>
        <div class="tp-body">${sections.map((s) => `<section class="tp-sec" id="sec-${s.id}"><h2>${esc(s.title)}${s.note ? `<span class="note">${esc(s.note)}</span>` : ""}</h2>${s.html}</section>`).join("")}</div>
        <aside class="tp-side"><div class="runbox" id="runbox"><p class="loading">Loading models…</p></div></aside>
      </div>
    </div>`;
  wire(el, v);
  runPanel($("#runbox", el), v);
  loadRollouts(el, v);
}

function header(v) {
  const vf = v.verify || {};
  const chips = Object.values(v.facets || {}).flatMap(vals).filter((x) => !/^(None named|Unspecified|Unknown|Unrated|Other)$/.test(x)).slice(0, 6);
  const facts = [];
  if (v.systems) facts.push(`<b>${v.systems.length}</b> systems`, `<b>${v.systems.reduce((n, s) => n + s.tools.length, 0)}</b> tools`, `<b>${v.files.length}</b> files`);
  if (vf.checks && vf.kind === "rubric") facts.push(`<b>${vf.checks.length}</b> checks`);
  if (vf.kind === "tests" || vf.kind === "terminal") facts.push(`<b>${vf.files.length}</b> hidden test files`);
  if (vf.kind === "crash" && vf.expected?.function) facts.push(`crash in <code>${esc(vf.expected.function)}</code>`);
  if (vf.kind === "visual") facts.push("vision-graded");
  if (vf.kind === "music") facts.push("no model in the grader");
  return `<header class="tp-head">
    <div class="kick"><span class="badge"><i class="dot"></i>${esc(DOMAIN_NAME[v.domain])}</span>${chips.map((c) => `<span class="chip">${esc(c)}</span>`).join("")}</div>
    <h1>${esc(v.title)}</h1>
    <div class="facts">${facts.map((f) => `<span>${f}</span>`).join("")}<code class="eid" title="Task id">${esc(v.id)}</code>
      <button class="link" data-copy="${esc(location.href)}">Copy link</button></div>
  </header>`;
}

// ── sections ─────────────────────────────────────────────────────────────────
function brief(v) {
  const vf = v.verify || {};
  let spec = "";
  if (vf.kind === "music" && vf.spec) {
    const s = vf.spec;
    spec = `<div class="spec">${[["Style", s.style], ["Tempo", s.bpm && `${s.bpm} BPM`], ["Meter", s.meter], ["Length", s.bars && `${s.bars} bars`], ["Voices", s.voices]]
      .filter(([, x]) => x).map(([k, x]) => `<div><span>${k}</span><b>${esc(x)}</b></div>`).join("")}</div>`;
  }
  const meta = (v.meta || []).filter(([, x]) => x && String(x).length < 200);
  return {
    id: "brief", nav: "The task", title: "The task", note: "exactly what the agent receives",
    html: `${spec}<div class="prose">${md(v.brief)}</div>
      ${meta.length ? `<dl class="kv">${meta.map(([k, x]) => `<dt>${esc(k)}</dt><dd>${esc(x)}</dd>`).join("")}</dl>` : ""}
      ${(v.links || []).map((l) => `<a class="out" href="${esc(l.url)}" target="_blank" rel="noopener">${esc(l.label)} ↗</a>`).join(" ")}`,
  };
}

function systems(v) {
  if (!v.systems?.length) return null;
  return {
    id: "systems", nav: `Systems (${v.systems.length})`, title: "Systems the agent works through",
    note: "each is an MCP server backed by its own database; the agent can only reach the data through these tools",
    html: `<div class="sys-grid">${v.systems.map((s, i) => `
      <details class="sys" ${i === 0 ? "open" : ""}>
        <summary><b>${esc(s.label)}</b><span>${s.tools.length} tools · ${s.tables.length} tables · ${s.tables.reduce((n, t) => n + t.rows, 0).toLocaleString()} rows</span></summary>
        <div class="sys-in">
          <h4>Tools</h4>
          <ul class="tools">${s.tools.map((t) => `<li><code class="sig">${esc(t.name)}(<span>${t.params.map((p) => esc(p.name) + (p.default != null ? "?" : "")).join(", ")}</span>)</code>
            ${t.doc ? `<p>${esc(t.doc)}</p>` : ""}</li>`).join("")}</ul>
          <h4>Data</h4>
          <div class="tables">${s.tables.map((t) => `<button class="tbl-btn" data-sys="${esc(s.name)}" data-table="${esc(t.table)}">
            <b>${esc(t.table)}</b><span>${t.rows.toLocaleString()} rows · ${t.columns.length} cols</span></button>`).join("")}</div>
        </div>
      </details>`).join("")}</div>`,
  };
}

function workspace(v) {
  if (!v.files?.length) return null;
  const kinds = {};
  v.files.forEach((f) => (kinds[f.kind] = (kinds[f.kind] || 0) + 1));
  return {
    id: "files", nav: `Workspace (${v.files.length})`, title: "Workspace files",
    note: Object.entries(kinds).map(([k, n]) => `${n} ${k}${n > 1 ? "s" : ""}`).join(" · "),
    html: `<div class="files-grid">${v.files.map((f) => `<button class="file" data-path="${esc(f.path)}" data-kind="${esc(f.kind)}">
      <span class="g">${GLYPH[f.kind] || "•"}</span><span class="fn">${esc(f.path)}</span><span class="fs">${f.kind} · ${bytes(f.size)}</span></button>`).join("")}</div>`,
  };
}

function setup(v) {
  const e = v.environment || {};
  const rows = [];
  if (e.sandbox === false) rows.push(["Where it runs", "No sandbox: one model call, then the scorer"]);
  if (e.image) rows.push(["Sandbox image", e.image.replace("docker.io/", "")]);
  if (e.cwd) rows.push(["Agent's working directory", e.cwd]);
  if (e.deliver) rows.push(["Deliverable", `${e.deliver}/index.html (and its assets)`]);
  if (e.ports) rows.push(["MCP ports", e.ports.join(", ")]);
  if (e.cpus) rows.push(["Resources", `${e.cpus} CPU · ${e.memory_mb} MB · internet ${e.internet ? "on" : "off"}`]);
  if (e.tags) rows.push(["Tags", e.tags.join(", ")]);
  const how = {
    code: "Git history is hidden while the agent works, so the fix cannot be read out of later commits.",
    cyber: "The agent runs as an unprivileged user. It gets the project source, a prebuilt fuzz binary and submit.sh; the verifier runs as root.",
    general: "The agent is an unprivileged user in the workspace. The systems' databases and code are root-only, so MCP is the way in.",
    webdev: "Node, pnpm, Playwright and Chromium are in the image; the agent builds however it likes and delivers to dist/.",
    music: "",
  }[v.domain];
  if (!rows.length && !how) return null;
  return {
    id: "setup", nav: "Setup", title: "How the sandbox is set up",
    html: `${rows.length ? `<dl class="kv">${rows.map(([k, x]) => `<dt>${esc(k)}</dt><dd><code>${esc(x)}</code></dd>`).join("")}</dl>` : ""}
      ${how ? `<p class="muted">${esc(how)} Web tools are off for the agent, and curl/wget/git fetch are blocked in its shell.</p>` : ""}`,
  };
}

function grading(v) {
  const g = v.verify;
  if (!g) return null;
  let body = `<p class="lead">${esc(g.summary)}</p>`;
  if (g.steps) body += `<ol class="steps">${g.steps.map((s) => `<li>${md(s).replace(/^<p>|<\/p>$/g, "")}</li>`).join("")}</ol>`;
  if (g.kind === "tests") {
    body += `<h4>Hidden tests</h4><div class="tbl"><table><thead><tr><th>File</th><th>Lines</th><th></th></tr></thead><tbody>${g.files.map((f) =>
      `<tr><td><code>${esc(f.path)}</code></td><td class="pm"><span class="a">+${f.added}</span> <span class="d">−${f.removed}</span></td><td>${f.new ? '<span class="chip">new</span>' : ""}</td></tr>`).join("")}</tbody></table></div>`;
    if (g.script) body += `<details class="code"><summary>Test command script</summary><pre><code>${esc(g.script)}</code></pre></details>`;
    if (g.patch) body += `<details class="code"><summary>Full hidden test patch (${g.patch.length.toLocaleString()} chars)</summary><pre class="diff">${diffHtml(g.patch)}</pre></details>`;
  }
  if (g.kind === "crash" && g.expected) {
    const x = g.expected;
    body += `<div class="target"><div><span>Sanitizer</span><b>${esc(x.sanitizer)}</b></div><div><span>Bug class</span><b>${esc(x.error_type)}</b></div>
      <div><span>Function</span><b><code>${esc(x.function)}</code></b></div><div><span>File</span><b><code>${esc(x.file)}</code></b></div></div>`;
  }
  if (g.kind === "rubric") {
    const tot = g.checks.reduce((s, c) => s + (c.weight || 0), 0) || 1;
    body += `<ol class="checks">${g.checks.map((c) => `<li><div class="ck-top"><span class="tier t-${esc(c.tier)}">${esc(c.tier || "")}</span>
      <span class="how">${c.method === "llm" ? "judged by a model" : "checked by code"}</span>
      <span class="w"><i style="width:${Math.round(((c.weight || 0) / tot) * 100)}%"></i>${Math.round(((c.weight || 0) / tot) * 100)}%</span></div>
      <p>${esc(c.question || c.id)}</p></li>`).join("")}</ol><p class="muted sm">The answer each check expects is not shown.</p>`;
  }
  if (g.kind === "terminal") {
    body += `<h4>Hidden test files</h4>${table([["File", "Size"], ...g.files.map((f) => [f.path, bytes(f.size)])])}`;
    if (g.script) body += `<details class="code"><summary>tests/test.sh</summary><pre><code>${esc(g.script)}</code></pre></details>`;
    if (g.tests) body += `<details class="code"><summary>tests/test_outputs.py</summary><pre><code>${esc(g.tests)}</code></pre></details>`;
  }
  if (g.kind === "visual") {
    body += `<div class="dims">${g.dims.map((d) => `<div class="dim g-${esc(d.group)}"><b>${esc(d.label)}</b><span>${esc(d.desc)}</span></div>`).join("")}</div>
      <p class="muted sm">${esc(g.formula)}. ${esc(g.note)}</p>`;
  }
  if (g.kind === "music") {
    const groups = {};
    g.features.forEach((f) => (groups[f.group] ||= []).push(f));
    body += `<p class="muted sm">Before any of this counts, the piece must pass a validity gate: no notation errors, fewer than 10 bars of the wrong length, no blank lines in the tune, one instrument per MIDI channel.</p>
      <div class="feat-groups">${Object.entries(groups).map(([gname, fs]) => `<div class="fg"><h4>${esc(gname)}</h4>
      ${fs.map((f) => `<div class="feat"><b>${esc(f.label || f.name)}</b><span>${f.rule === "band" ? "within the human range" : f.rule === "high" ? "the higher the better" : "the lower the better"} · <code>${esc(f.name)}</code></span>
        ${f.band ? `<em>${fmtNum(f.band[0])} – ${fmtNum(f.band[1])}</em>` : ""}</div>`).join("")}</div>`).join("")}</div>`;
  }
  return { id: "grading", nav: "Grading", title: "How it is graded", note: g.needs_judge ? `needs a ${g.needs_judge} judge model` : "no model in the loop", html: body };
}

function rollouts() {
  return { id: "runs", nav: "Your rollouts", title: "Your rollouts on this task", html: `<div id="task-runs"><p class="loading">Loading…</p></div>` };
}

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
    const c = ev.target.closest("[data-copy]");
    if (c) { try { await navigator.clipboard.writeText(location.href); toast("Link copied"); } catch { toast("Copy the address bar"); } return; }
    const f = ev.target.closest(".file");
    if (f) return previewFile(v, f.dataset.path, f.dataset.kind);
    const t = ev.target.closest(".tbl-btn");
    if (t) return previewTable(v, t.dataset.sys, t.dataset.table);
  });
  const links = new Map($$("[data-jump]", el).map((a) => [a.dataset.jump, a]));
  spy = new IntersectionObserver((es) => es.forEach((e) => {
    if (e.isIntersecting) { links.forEach((a) => a.classList.remove("on")); links.get(e.target.id.slice(4))?.classList.add("on"); }
  }), { rootMargin: "-20% 0px -70% 0px" });
  $$(".tp-sec", el).forEach((s) => spy.observe(s));
}

async function previewFile(v, path, kind) {
  const raw = `/api/tasks/${encodeURIComponent(v.id)}/file?path=${encodeURIComponent(path)}`;
  const body = openModal(path, `<p class="loading">Opening…</p>`, raw);
  if (kind === "image") { body.innerHTML = `<div class="pv-img"><img src="${raw}" alt=""></div>`; return; }
  if (kind === "web") { body.innerHTML = `<iframe class="pv-frame" sandbox src="${raw}"></iframe>`; return; }
  let p;
  try { p = await api(`/api/tasks/${encodeURIComponent(v.id)}/preview?path=${encodeURIComponent(path)}`); }
  catch (e) { body.innerHTML = `<p class="empty">${esc(e.message)}</p>`; return; }
  if (p.type === "sheets") {
    body.innerHTML = `<div class="pv-tabs">${p.sheets.map((s, i) => `<button data-i="${i}" aria-selected="${i === 0}">${esc(s.name)}${s.dims ? `<em>${esc(s.dims)}</em>` : ""}</button>`).join("")}</div><div id="pv-sheet"></div>`;
    const show = (i) => { $("#pv-sheet", body).innerHTML = sheetGrid(p.sheets[i].rows) + (p.sheets[i].rows.length >= 60 ? `<p class="muted xs">Showing the first 60 rows and 24 columns · open raw for the whole workbook.</p>` : "");
      $$(".pv-tabs button", body).forEach((b) => b.setAttribute("aria-selected", b.dataset.i == i)); };
    body.querySelector(".pv-tabs").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) show(+b.dataset.i); });
    show(0);
  } else if (p.type === "document") {
    body.innerHTML = `<div class="pv-doc">${p.blocks.map((b) => b.table ? table(b.table, { header: true }) : b.h ? `<h${b.h + 2}>${esc(b.text)}</h${b.h + 2}>` : `<p>${esc(b.text)}</p>`).join("")}</div>`;
  } else if (p.type === "slides") {
    body.innerHTML = `<div class="pv-slides">${p.slides.map((s) => `<div class="slide"><span class="n">${s.n}</span><b>${esc(s.title)}</b>${s.text.map((t) => `<p>${esc(t)}</p>`).join("")}</div>`).join("")}</div>`;
  } else if (p.type === "pdf") {
    body.innerHTML = `<div class="pv-tabs"><button data-m="view" aria-selected="true">Document</button><button data-m="text" aria-selected="false">Text (${p.page_count} pages)</button></div>
      <div id="pv-pdf"><iframe class="pv-frame" src="${raw}"></iframe></div>`;
    body.querySelector(".pv-tabs").addEventListener("click", (e) => {
      const b = e.target.closest("button"); if (!b) return;
      $$(".pv-tabs button", body).forEach((x) => x.setAttribute("aria-selected", x === b));
      $("#pv-pdf", body).innerHTML = b.dataset.m === "view" ? `<iframe class="pv-frame" src="${raw}"></iframe>`
        : `<div class="pv-doc">${p.pages.map((pg) => `<h4>Page ${pg.n}</h4><pre class="wrap">${esc(pg.text)}</pre>`).join("")}</div>`;
    });
  } else if (p.type === "text") body.innerHTML = `<pre class="wrap">${esc(p.text)}</pre>`;
  else if (p.type === "archive") body.innerHTML = table([["Entry", "Size"], ...p.entries.map((x) => [x.name, bytes(x.size)])]);
  else if (p.type === "error") body.innerHTML = `<p class="empty">Could not preview: ${esc(p.error)}</p><p><a href="${raw}&download=1">Download</a></p>`;
  else body.innerHTML = `<p class="empty">No preview for this file type.</p><p><a class="btn" href="${raw}&download=1">Download</a></p>`;
}

async function previewTable(v, sys, tbl) {
  const body = openModal(`${sys} · ${tbl}`, `<p class="loading">Loading rows…</p>`);
  try {
    const d = await api(`/api/tasks/${encodeURIComponent(v.id)}/systems/${encodeURIComponent(sys)}/${encodeURIComponent(tbl)}?limit=100`);
    body.innerHTML = table([d.columns, ...d.rows.map((r) => r.map((c) => (c == null ? "" : String(c))))]) + (d.rows.length >= 100 ? `<p class="muted sm">First 100 rows.</p>` : "");
  } catch (e) { body.innerHTML = `<p class="empty">${esc(e.message)}</p>`; }
}

// ── the run panel ────────────────────────────────────────────────────────────
async function runPanel(box, v) {
  const s = getSession();
  if (!v.runnable) { box.innerHTML = `<h3>Run a rollout</h3><p class="muted">This task can't be run here yet.</p>`; return; }
  let cat;
  try { cat = await getModels(); } catch (e) { box.innerHTML = `<h3>Run a rollout</h3><p class="empty">Models unavailable: ${esc(e.message)}</p>`; return; }
  const need = v.verify?.needs_judge;
  const judges = need === "vision" ? cat.vision_judges : need === "text" ? cat.text_judges : [];
  box.innerHTML = `
    <h3>Run a rollout</h3>
    <p class="muted sm">${v.domain === "music" ? "One model call, then Xiaomi's scorer." : "A fresh HF Sandbox from this task's image, OpenCode as the harness, then this task's own grader."}</p>
    <div class="fld"><span>Agent model</span><div id="rb-model"></div></div>
    ${judges.length ? `<div class="fld"><span>Judge model <em>${need === "vision" ? "looks at a screenshot of the page" : "answers each rubric check"}</em></span><div id="rb-judge"></div></div>` : ""}
    <div class="estimate" id="rb-est"></div>
    ${s.user ? `<button class="btn primary block" id="rb-go">Run rollout</button>`
      : `<a class="btn primary block" href="/login" ${window.top !== window.self ? 'target="_blank" rel="noopener"' : ""}>Sign in with Hugging Face to run</a>`}
    <p class="muted xs">It keeps running if you close this page. Find it under <a href="#/runs">My rollouts</a>. Billed to ${s.user ? `<b>${esc(s.user.name)}</b>` : "your account"} on Hugging Face.</p>`;
  const def = cat.agents.some((m) => m.id === cat.default_agent) ? cat.default_agent : cat.agents[0]?.id;
  const update = (m) => {
    const [ti, to] = TYPICAL[v.domain];
    const est = (ti * m.input + to * m.output) / 1e6;
    $("#rb-est", box).innerHTML = `A typical ${DOMAIN_NAME[v.domain]} rollout with this model: <b>~${money(est)}</b>
      <span class="muted">(${(ti / 1e6 >= 1 ? (ti / 1e6).toFixed(1) + "M" : Math.round(ti / 1e3) + "k")} tokens in, ${Math.round(to / 1e3)}k out)</span>${v.domain !== "music" ? ", plus sandbox time at about $0.01 an hour" : ""}.
      ${v.domain === "general" || v.domain === "cyber" ? "Long agent loops re-read their context every step, so input tokens dominate: a cheaper model makes a big difference here." : ""}`;
  };
  const sel = picker($("#rb-model", box), { models: cat.agents, value: def, onChange: update,
    note: "Every model here supports tool calling. Prices are per million tokens, input / output, at the cheapest provider that can call tools." });
  update(sel.model);
  const judgeSel = judges.length ? picker($("#rb-judge", box), { models: judges.map((j) => ({ ...j, featured: true })), value: judges[0].id, onChange: () => {},
    note: need === "vision" ? "Tested on a real render with Xiaomi's rubric. Judges disagree (0.34–0.87 on the same page), so compare webdev scores only between runs graded by the same judge." : "Tested on this dataset's real rubric prompt: each returned a usable verdict on every check." }) : null;
  const go = $("#rb-go", box);
  if (go) go.addEventListener("click", async () => {
    go.disabled = true; go.textContent = "Starting…";
    try {
      const run = await api("/api/runs", { method: "POST", body: { task_id: v.id, model: sel.value, judge: judgeSel?.value || null } });
      refreshActive();
      location.hash = `#/run/${run.id}`;
    } catch (e) {
      toast(e.message, 5000);
      go.disabled = false; go.textContent = "Run rollout";
      if (e.status === 401) location.href = "/login";
    }
  });
}

async function loadRollouts(el, v) {
  const box = $("#task-runs", el);
  if (!getSession().user) { box.innerHTML = `<p class="muted">Sign in to see your rollouts.</p>`; return; }
  try {
    const { runs } = await api(`/api/runs?task_id=${encodeURIComponent(v.id)}`);
    box.innerHTML = runs.length ? `<div class="runlist">${runs.map((r) => `<a class="runrow" href="#/run/${esc(r.id)}">${statusPill(r.status)}
      <span class="m">${esc(r.model.split("/")[1])}</span>${rewardBadge(r.reward, r.status)}<span class="c">${money((r.cost || {}).total)}</span><span class="w">${ago(r.created_at)}</span></a>`).join("")}</div>`
      : `<p class="muted">None yet. Pick a model on the right and run one.</p>`;
  } catch (e) { box.innerHTML = `<p class="empty">${esc(e.message)}</p>`; }
}
