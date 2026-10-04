// A task, whatever its environment's format: /t/<org>/<name>/<ref>. The adapter that reads the environment
// (app/envs/contract.py) builds its sections out of blocks (blocks.js draws them, and a renderer module draws what no
// block covers), says how it can run (its run options) and which other tasks are this very one (a raw row and its
// Harbor conversion), so a Harbor folder, a MiMo task and a NeMo Gym row get the same page: contents on the left, the
// sections, then the run panel and the facts. Answers stay out: adapters withhold them, and the page names them only.
import { $, $$, api, esc, fmt, sk, spinner, toast, ago, dur, money, statusPill, rewardBadge, rewardText, visibilityBadge, emptyState, confetti,
  PUBLIC_NOTE, PRIVATE_NOTE, nav as navigate, setMeta } from "./util.js";
import { icon } from "./icons.js";
import { getSession, refreshActive } from "./session.js";
import { picker } from "./picker.js";
import { fileViewer } from "./files.js";
import { blocks, inline, loadRenderers } from "./blocks.js";
import { typeset } from "./math.js";

let models = null;
const getModels = () => (models ||= api("/api/models").catch((e) => { models = null; throw e; }));

const enc = (s) => s.split("/").map(encodeURIComponent).join("/");
const envApi = (spec) => `/api/env/${enc(spec)}`;
export const taskHref = (spec, ref) => `/t/${enc(spec)}/${enc(ref)}`;
export const compareHref = (spec, ref, ids) => `/compare/${enc(spec)}/${enc(ref)}?r=${ids.map(encodeURIComponent).join(",")}`;
const subsetLabel = (s) => String(s || "").split("/").join(" · ");
const COMPARE_MAX = 4;

let observer = null, viewers = [], keys = null;
export function unmount() {
  observer?.disconnect(); observer = null;
  viewers.forEach((v) => v.destroy()); viewers = [];
  if (keys) removeEventListener("keydown", keys);
  keys = null;
}

export async function mount(el, { spec, path: ref, qs }) {
  const panel = (lines) => `<section class="panel"><div class="panel-h">${sk.box(16, "width:140px")}</div><div class="panel-b sk-in">${sk.lines(...lines)}</div></section>`;
  el.innerHTML = `<div class="wrap page" aria-busy="true">${sk.head()}
    <div class="tp-grid"><nav class="toc sk-in sk-toc">${sk.lines(70, 55, 80, 60, 50)}</nav>
      <div class="tp-body">${panel([96, 100, 88, 94, 70, 92, 60])}${panel([90, 70, 84])}${panel([80, 60])}</div>
      <aside class="tp-side">${panel([100, 90, 100, 70])}${panel([60, 80, 50])}</aside></div></div>`;
  getModels().catch(() => {});   // start fetching models while the page loads
  let v, renderers;
  try {
    v = await api(`${envApi(spec)}/task?ref=${encodeURIComponent(ref)}`, { retry: 2 });
    renderers = await loadRenderers(v.sections);
  } catch (e) {
    el.innerHTML = `<div class="wrap page">${emptyState(e.status === 404 ? "search" : "alert", e.status === 404 ? "No such task" : "Couldn't open this task", esc(e.message),
      `<a class="btn" href="/d/${enc(spec)}">${icon("arrowLeft", 15)}Open the environment</a>`)}</div>`;
    return;
  }
  if (v.ref !== ref) {   // an older address of this task (a row number): the page lives at its own ref
    ref = v.ref;
    history.replaceState(null, "", `${taskHref(spec, ref)}${qs.toString() ? `?${qs}` : ""}`);
  }
  const [org, name] = spec.split("/");
  const c = v.collection, nav = v.nav;
  setMeta({ title: `${v.title} · ${name}`, description: `${v.title}: a task in ${spec} (${v.env.framework}). What the agent is asked, how it's graded, what it runs in, and its files.` });
  const fileBlocks = v.sections.flatMap((s) => s.blocks.filter((b) => b.type === "files"));
  const ctx = { files: [], paths: new Set(fileBlocks.flatMap((b) => b.tree.map((f) => f.path))), renderers, spec, ref };
  const sections = [...v.sections.map((s) => ({ ...s, html: blocks(s.blocks, ctx), count: s.blocks.find((b) => b.type === "files")?.tree.filter((f) => !f.dir).length })),
    { id: "rollouts", title: "Rollouts", icon: "play", note: "yours, and everyone's public ones", html: `<div id="t-runs">${sk.lines(70, 50)}</div>` }];
  const where = nav?.subset ? `${subsetLabel(nav.subset)} #${fmt.format(nav.index)}` : v.ref || "(root)";
  const same = (v.links || []).filter((l) => l.rel === "same"), related = (v.links || []).filter((l) => l.rel !== "same");
  const kind = v.color ? `<span class="badge" style="--dc:var(--c-${esc(v.color)})">${v.icon ? icon(v.icon, 13) : ""}${esc(v.framework || v.env.framework)}</span>`
    : `<span class="chip fw">${esc(v.env.framework)}</span>`;
  el.innerHTML = `<div class="wrap page fade-in" ${v.color ? `style="--dc:var(--c-${esc(v.color)})"` : ""}>
    <nav class="crumbs" aria-label="Breadcrumb"><a href="/">Environments</a>${icon("chevronRight", 13)}
      ${c ? `<a href="/?c=${esc(c.id)}">${esc(c.group)}</a>${icon("chevronRight", 13)}` : ""}
      <a href="/d/${enc(spec)}${nav?.subset ? `?s=${encodeURIComponent(nav.subset)}` : ""}">${esc(name)}</a>${icon("chevronRight", 13)}<span>${esc(where)}</span></nav>
    <header class="tp-head">
      <div class="kick">${c ? `<span class="badge" style="--dc:var(--c-${c.color || c.id})">${icon(c.icon, 13)}${esc(c.group)}</span>` : ""}
        ${kind}${(v.chips || []).map((x) => `<span class="chip">${esc(x)}</span>`).join("")}</div>
      <h1>${esc(v.title)}</h1>
      <div class="facts">
        <a class="u" href="/d/${enc(spec)}">${icon("database", 14)}<code>${esc(org)}/${esc(name)}</code></a>
        ${v.id && v.id !== v.title && v.id !== where ? `<span title="its id">${icon("flag", 14)}<code>${esc(v.id)}</code></span>` : ""}
        ${v.summary ? `<span>${icon("target", 14)}${esc(v.summary)}</span>` : ""}
        ${nav ? `<span class="tp-nav">${icon("list", 14)}<span>${nav.label ? `${esc(nav.label)} ` : ""}${fmt.format(nav.index)}${nav.total != null ? ` of ${fmt.format(nav.total)}` : ""}</span>
          <a class="btn sm ${nav.prev == null ? "disabled" : ""}" ${nav.prev == null ? 'aria-disabled="true"' : `href="${taskHref(spec, nav.prev)}"`} title="Previous task (←)" aria-label="Previous task">${icon("chevronRight", 14, "flip")}</a>
          <a class="btn sm ${nav.next == null ? "disabled" : ""}" ${nav.next == null ? 'aria-disabled="true"' : `href="${taskHref(spec, nav.next)}"`} title="Next task (→)" aria-label="Next task">${icon("chevronRight", 14)}</a></span>` : ""}
        <button class="link" type="button" id="random">${icon("shuffle", 14)}Another task</button>
        <button class="link" type="button" id="copy">${icon("link", 14)}Copy link</button>
        ${v.hub ? `<a class="u" href="${esc(v.hub)}" target="_blank" rel="noopener">${icon("external", 14)}${/\/viewer\//.test(v.hub) ? "In the Hub's viewer" : "On the Hub"}</a>` : ""}
      </div>
      ${same.length || related.length ? `<div class="tp-links">${[...same, ...related].map((l) => `<p>${icon(l.rel === "same" ? "link" : "external", 13)}
        <a class="u" href="${esc(l.href)}">${esc(l.label)}</a>${l.note ? ` <span class="faint">${esc(l.note)}</span>` : ""}</p>`).join("")}</div>` : ""}
    </header>
    <div class="tp-grid">
      <nav class="toc" aria-label="On this page"><p>On this page</p>${sections.map((s) => `<a href="#" data-jump="${esc(s.id)}">${esc(s.title)}${s.count ? `<em>${fmt.format(s.count)}</em>` : ""}</a>`).join("")}</nav>
      <div class="tp-body">${sections.map((s) => `<section class="panel tp-sec" id="sec-${esc(s.id)}">
        <div class="panel-h"><h2>${icon(s.icon || "list", 15)}${esc(s.title)}</h2>${s.note ? `<span class="aside">${esc(s.note)}</span>` : ""}</div>
        <div class="panel-b">${s.html}</div></section>`).join("")}</div>
      <aside class="tp-side split"><section class="panel first runbox" id="run-card"></section>${glance(v)}</aside>
    </div>
    <div class="cmp-bar" id="cmp-bar" role="region" aria-label="Compare the selected rollouts" hidden></div></div>`;
  Object.values(renderers).forEach((r) => r.wire?.(el, ctx));

  // files: one viewer per files block, the first keeps its open file and lines in the URL (?f=…&L=…)
  ctx.files.forEach((b, i) => {
    const raw = b.raw && b.hub ? b.hub.replace("/blob/", "/resolve/") : null;
    viewers.push(fileViewer($(`[data-files="${i}"]`, el), {
      files: b.tree, initial: i === 0 ? qs.get("f") : null, line: i === 0 ? qs.get("L") : null,
      load: (rel) => api(`${envApi(spec)}/file?ref=${encodeURIComponent(ref)}&f=${encodeURIComponent(rel)}`),
      listDir: async (dir) => (await api(`${envApi(spec)}/folder?ref=${encodeURIComponent(ref)}&f=${encodeURIComponent(dir)}`)).entries,
      hubUrl: (rel) => (b.hub ? b.hub + enc(rel) : null),
      rawUrl: (rel) => (raw ? raw + enc(rel) : null),
      onChange: i === 0 ? (rel, lines) => {
        const q = new URLSearchParams(location.search.slice(1) || "");
        q.set("f", rel); lines ? q.set("L", lines) : q.delete("L");
        history.replaceState(null, "", `${location.pathname}?${q}`);
      } : () => {},
    }));
  });
  if (qs.get("f") && viewers.length) requestAnimationFrame(() => $(".bl-files", el)?.closest(".tp-sec")?.scrollIntoView({ block: "start" }));

  const another = async () => {
    try { const r = await api(`${envApi(spec)}/random${nav?.subset ? `?subset=${encodeURIComponent(nav.subset)}` : ""}`); navigate(taskHref(spec, r.ref)); }
    catch (err) { toast(err.message); }
  };
  const picks = new Set();
  el.addEventListener("click", (e) => {
    const j = e.target.closest("[data-jump]");
    if (j) { e.preventDefault(); $(`#sec-${CSS.escape(j.dataset.jump)}`, el)?.scrollIntoView({ behavior: "smooth", block: "start" }); return; }
    const of = e.target.closest("[data-open-file]");
    if (of && viewers[0]) { viewers[0].open(of.dataset.openFile); $(".bl-files", el)?.closest(".tp-sec")?.scrollIntoView({ behavior: "smooth", block: "start" }); return; }
    if (e.target.closest("#copy")) navigator.clipboard?.writeText(location.href).then(() => toast("Link copied"));
    if (e.target.closest("#random")) another();
    if (e.target.closest("[data-cmp-clear]")) { picks.clear(); syncPicks(el, spec, ref, picks); }
  });
  el.addEventListener("change", (e) => {
    const cb = e.target.closest("[data-cmp]");
    if (!cb) return;
    cb.checked ? picks.add(cb.dataset.cmp) : picks.delete(cb.dataset.cmp);
    syncPicks(el, spec, ref, picks);
  });
  // ← and → step through the tasks of a list that has an order
  if (nav) {
    keys = (e) => {
      if (e.metaKey || e.ctrlKey || e.altKey || /INPUT|TEXTAREA|SELECT/.test(document.activeElement?.tagName) || document.activeElement?.isContentEditable || !$("#modal")?.hidden) return;
      const to = e.key === "ArrowLeft" ? nav.prev : e.key === "ArrowRight" ? nav.next : null;
      if (to != null) navigate(taskHref(spec, to));
    };
    addEventListener("keydown", keys);
  }
  runCard($("#run-card", el), v, spec, ref);
  taskRuns($("#t-runs", el), spec, ref, v, () => syncPicks(el, spec, ref, picks));
  typeset($(".tp-head", el), $(".tp-body", el));   // math in the title and the prompt (KaTeX loads only if there is some)
  // the contents list follows the section in view
  const links = new Map([...el.querySelectorAll("[data-jump]")].map((a) => [`sec-${a.dataset.jump}`, a]));
  const seen = new Map();
  observer = new IntersectionObserver((es) => {
    es.forEach((x) => seen.set(x.target.id, x.isIntersecting));
    const first = [...links.keys()].find((id) => seen.get(id));
    links.forEach((a, id) => a.classList.toggle("on", id === first));
  }, { rootMargin: "-80px 0px -55% 0px" });
  el.querySelectorAll(".tp-sec").forEach((s) => observer.observe(s));
}

function glance(v) {
  const rows = v.glance || [];
  const withheld = v.withheld || [], cut = v.truncated || [];
  if (!rows.length && !withheld.length && !cut.length) return "";
  return `<section class="panel last"><div class="panel-h"><h3>${icon("info", 14)}At a glance</h3></div><div class="panel-b">
    ${rows.length ? `<dl class="kv">${rows.map(([a, b]) => `<dt>${esc(a)}</dt><dd>${typeof b === "number" ? fmt.format(b) : inline(b)}</dd>`).join("")}</dl>` : ""}
    ${withheld.length ? `<p class="fine" style="margin-top:12px">${icon("shield", 12)} Withheld, as they hold the answer: ${withheld.slice(0, 12).map((k) => `<code>${esc(k)}</code>`).join(", ")}${withheld.length > 12 ? ` and ${withheld.length - 12} more` : ""}.</p>` : ""}
    ${cut.length ? `<p class="fine">Cut short by the dataset viewer: ${cut.map((k) => `<code>${esc(k)}</code>`).join(", ")}.</p>` : ""}
  </div></section>`;
}

// ── running it ───────────────────────────────────────────────────────────────
// The run panel, from the task's run options: one per runner that can run it, the default first. Each says whether
// it can run this task (and why not), what it does, and the inputs it takes beyond the agent and the model. The model
// comes from HF Inference Providers or, where the runner allows, your own OpenAI-compatible endpoint. Settings worth
// keeping live in this browser; an endpoint's key stays in this tab only.
const LS = { get: (k, d) => { try { return JSON.parse(localStorage.getItem(k)) ?? d; } catch { return d; } },
             set: (k, v) => { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* private window */ } } };
const SS = { get: (k) => { try { return sessionStorage.getItem(k) || ""; } catch { return ""; } },
             set: (k, v) => { try { sessionStorage.setItem(k, v); } catch { /* private window */ } } };
const BILLING = `<a href="https://huggingface.co/settings/billing" target="_blank" rel="noopener">huggingface.co/settings/billing</a>`;
const optValue = (x) => (x && typeof x === "object" ? x.value : x);
const optLabel = (x) => (x && typeof x === "object" ? x.label ?? x.value : x);
const tok = (n) => (n >= 1e6 ? (n / 1e6).toFixed(1) + "M" : n >= 1e3 ? Math.round(n / 1e3) + "k" : String(Math.round(n)));

export async function runCard(box, v, spec, ref) {
  const opts = v.run?.options || [];
  const head = `<div class="panel-h"><h3>${icon("play", 14)}${opts.some((o) => o.ok || o.href) ? "Run a rollout" : "Run it"}</h3></div>`;   // nothing runs here: no promise of one
  if (!opts.length) {
    box.innerHTML = `<div class="panel-h"><h3>${icon("play", 14)}Run it</h3></div><div class="panel-b"><p class="muted sm">${inline(v.run?.note || "This task can't be run here.")}</p></div>`;
    return;
  }
  const saved = LS.get("run-runner", {})[spec];
  let pick = opts.find((o) => o.runner === saved && o.ok) || opts.find((o) => o.default && o.ok) || opts.find((o) => o.ok) || opts[0];
  const st = { source: LS.get("source", "hf"), probe: null, visibility: v.restricted ? "private" : "public", pickers: {} };
  const s = getSession();
  const choose = () => (opts.length > 1 ? `<div class="field"><span>Runner</span><span class="seg" id="run-runner" role="tablist">${opts.map((o) =>
    `<button type="button" data-r="${esc(o.runner)}" aria-pressed="${o === pick}" title="${esc(o.ok ? o.label : `${o.label}: can't run this task`)}">${esc(o.label)}</button>`).join("")}</span></div>` : "");
  const about = (o) => `${o.about ? `<p class="intro">${inline(o.about)}</p>` : ""}${opts.length === 1 ? `<p class="fine" style="text-align:left">${icon("box", 12)} ${esc(o.label)}</p>` : ""}
    ${o.notes.map((n) => `<p class="fine" style="text-align:left">${icon("info", 12)} ${inline(n)}</p>`).join("")}
    ${o.warnings.map((w) => `<div class="note-box warn">${icon("alert")}<span>${inline(w)}</span></div>`).join("")}`;

  const render = async () => {
    const o = pick;
    if (o.href) {
      box.innerHTML = `${head}<div class="panel-b" style="display:grid;gap:10px">${choose()}${about(o)}<a class="btn primary lg block" href="${esc(o.href)}">${icon("play", 15)}Open it</a></div>`;
      return;
    }
    if (!o.ok) {
      box.innerHTML = `${head}<div class="panel-b" style="display:grid;gap:10px">${choose()}${about(o)}<div class="note-box warn">${icon("alert")}<span>This task can't run here: ${esc(o.why || "no runnable form")}.</span></div></div>`;
      return;
    }
    if (!s.user) {
      box.innerHTML = `${head}<div class="panel-b" style="display:grid;gap:10px">${choose()}${about(o)}<button class="btn primary lg block" type="button" data-signin>${icon("user", 15)}Sign in to run</button>
        <p class="fine">The ${o.sandbox ? "sandbox and the " : ""}model's tokens are billed to your Hugging Face account.</p></div>`;
      return;
    }
    box.innerHTML = `${head}<div class="panel-b"><div class="loading-row">${spinner()}Loading models from Inference Providers…</div></div>`;
    let cat;
    try { cat = await getModels(); } catch (e) {
      box.innerHTML = `${head}<div class="panel-b"><div class="note-box err">${icon("alert")}<span>Couldn't reach HF Inference Providers: ${esc(e.message)}</span></div>
        <button class="btn block" type="button" id="run-retry">${icon("refresh")}Try again</button></div>`;
      $("#run-retry", box).addEventListener("click", render);
      return;
    }
    if (pick !== o) return;   // another runner was picked meanwhile
    const prefs = LS.get("run-prefs", {});
    const agents = cat.agents.filter((a) => !o.harnesses || o.harnesses.includes(a.id));
    const agent = agents.some((a) => a.id === prefs.agent) ? prefs.agent : agents.some((a) => a.id === cat.default_agent) ? cat.default_agent : agents[0]?.id;
    const adv = LS.get(`run-adv:${o.runner}`, {});
    const ep = LS.get("byo", { base_url: "", model: "", price_in: "", price_out: "" });
    if (!o.endpoint) st.source = "hf";
    const main = o.fields.filter((f) => !f.advanced), more = o.fields.filter((f) => f.advanced);
    const bill = s.billing || {};
    const noCredit = o.sandbox && (bill.can_pay === false || bill.refused_at);
    const gaps = s.missing_scopes || [];
    box.innerHTML = `${head}<div class="panel-b">${choose()}${about(o)}
      ${o.endpoint ? `<div class="seg src" role="tablist" aria-label="Where the model runs"><button type="button" data-src="hf">HF Inference Providers</button><button type="button" data-src="byo">Your endpoint</button></div>` : ""}
      ${agents.length > 1 ? `<label class="field"><span>Agent <em>runs in the sandbox, except Terminus 2</em></span>
        <select class="hf-select-mimo" id="run-agent">${agents.map((a) => `<option value="${esc(a.id)}" ${a.id === agent ? "selected" : ""}>${esc(a.name)}</option>`).join("")}</select></label>`
        : `<input type="hidden" id="run-agent" value="${esc(agent || "")}">`}
      <div class="field" data-pane="hf"><span>${agents.length > 1 ? "Model" : "Agent model"} <em>HF Inference Providers, $ per 1M tokens in / out</em></span><div id="run-model"></div></div>
      ${o.endpoint ? `<div class="byo" data-pane="byo">
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
        <p class="fine">Works with a LiteLLM proxy, vLLM, OpenAI, Together, OpenRouter and the like. The ${o.sandbox ? "sandbox still runs" : "rollout still runs"} on your
          Hugging Face account, so you sign in with HF either way. The key is used for this rollout only and never stored on the server,
          but the agent's own processes in the sandbox can see it: use a key you can revoke.</p></div>` : ""}
      ${main.map((f) => fieldHtml(f, adv)).join("")}
      ${more.length ? `<details class="adv" id="run-adv"><summary class="disclose">${icon("chevronRight", 14, "chev")}Advanced settings<span class="adv-sum" id="adv-sum"></span></summary>
        <div class="adv-grid">${more.map((f) => fieldHtml(f, adv)).join("")}</div>
        ${more.some((f) => f.help) ? `<p class="fine" style="text-align:left">${more.filter((f) => f.help).map((f) => `${esc(f.label)}: ${esc(f.help)}.`).join(" ")}</p>` : ""}</details>` : ""}
      ${o.estimate ? `<div class="estimate" id="run-est"></div>` : ""}
      <div class="field vis"><span>Visibility</span><span class="seg" id="run-vis">${["public", "private"].map((x) => `<button type="button" data-vis="${x}" ${v.restricted && x === "public" ? "disabled" : ""}>${icon(x === "public" ? "globe" : "shield", 12)}${x === "public" ? "Public" : "Private"}</button>`).join("")}</span>
        <p class="fine" id="run-vis-note" style="text-align:left"></p></div>
      ${gaps.length ? `<div class="note-box warn">${icon("alert")}<span>Your sign-in may be missing <b>${esc(gaps.join(", "))}</b>. If the rollout fails to start, sign in again with a write token.</span></div>` : ""}
      ${noCredit ? `<div class="note-box warn">${icon("alert")}<span>${bill.refused_at
        ? `Your last rollout couldn't start because your Hugging Face account had no prepaid credit for sandboxes. Add credit at ${BILLING}, then run again.`
        : `Your Hugging Face account can't pay for sandboxes yet, so a rollout would stop before it starts. Add prepaid credit at ${BILLING} first.`}
        Nothing is charged for a rollout that can't start.</span></div>` : ""}
      <div id="run-err"></div>
      <button class="btn ${noCredit ? "" : "primary"} lg block" type="button" id="run-go">${icon("play", 15)}${noCredit ? "Run anyway" : "Run rollout"}</button>
      <p class="fine">Keeps running if you close this page; find it under <a href="/runs">My rollouts</a>. ${o.sandbox ? "The sandbox is" : "The model's tokens are"} billed to <b>${esc(s.user.name)}</b> on Hugging Face.</p></div>`;
    const pk = picker($("#run-model", box), { models: cat.models, value: prefs.model || cat.default_model, onChange: () => update(),
      note: "Every model here supports tool calling. Prices are per million tokens, input / output." });
    st.pickers = {};
    for (const f of o.fields.filter((x) => x.type === "model")) {
      const pool = (cat[f.pool] || []).map((m) => ({ ...m, featured: true }));
      if (!pool.length) continue;
      st.pickers[f.key] = picker($(`#run-f-${CSS.escape(f.key)}`, box), { models: pool, value: adv[f.key] || pool[0].id, onChange: () => update(), groupLabel: "Tested judges",
        note: f.pool === "vision_judges" ? "Tested on a real render with Xiaomi's rubric. Judges disagree, so compare scores only between runs graded by the same judge."
          : "Tested on this dataset's real rubric prompt: each returned a usable verdict on every check." });
    }
    const val = (id) => { const e = $(id, box); return e ? e.value.trim() : ""; };
    const num = (id) => { const x = val(id); return x === "" ? null : Number(x); };
    const endpoint = () => ({ base_url: val("#ep-url"), api_key: val("#ep-key") || null, model: val("#ep-model"), price_in: num("#ep-pin"), price_out: num("#ep-pout") });
    const fields = () => fieldValues(box, o.fields, st.pickers);
    const save = () => {
      if (o.endpoint) {
        const e = endpoint();
        LS.set("byo", { base_url: e.base_url, model: e.model, price_in: val("#ep-pin"), price_out: val("#ep-pout") });
        SS.set("byo-key", val("#ep-key"));
      }
      LS.set(`run-adv:${o.runner}`, fields());
      LS.set("run-prefs", { ...LS.get("run-prefs", {}), agent: val("#run-agent"), model: pk.value });
    };
    const status = (kind, html) => { $("#ep-status", box).innerHTML = html ? `<div class="note-box ${kind}">${icon(kind === "ok" ? "check" : "alert")}<span>${html}</span></div>` : ""; };
    function update() {
      save();
      $$("[data-src]", box).forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.src === st.source)));
      $$("[data-pane]", box).forEach((p) => (p.hidden = p.dataset.pane !== st.source));
      $$("[data-vis]", box).forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.vis === st.visibility)));
      $("#run-vis-note", box).textContent = v.restricted ? "This dataset is private or gated, so its rollouts stay private." : st.visibility === "public" ? PUBLIC_NOTE : PRIVATE_NOTE;
      const fv = fields();
      const changed = more.filter((f) => fv[f.key] != null && fv[f.key] !== f.default).map((f) => `${f.label.toLowerCase()} ${optLabel((f.options || []).find((x) => optValue(x) === fv[f.key])) ?? fv[f.key]}`);
      if ($("#adv-sum", box)) $("#adv-sum", box).textContent = changed.join(" · ");
      if (o.estimate) {
        const e = o.estimate;
        let price;
        if (st.source === "byo") { const x = endpoint(); price = x.price_in != null || x.price_out != null ? [x.price_in || 0, x.price_out || 0] : null; }
        else price = [pk.model.input, pk.model.output];
        const minutes = fv.timeout_min ? Math.min(fv.timeout_min, e.minutes || 0) : e.minutes;
        const sandbox = o.sandbox ? (minutes / 60) * ((cat.prices || {})[e.flavor] ?? cat.sandbox_price_per_hour ?? 0.01) : 0;
        const model = price ? (e.tokens_in * price[0] + e.tokens_out * price[1]) / 1e6 : null;
        $("#run-est", box).innerHTML = `
          <div class="row"><span>Model · ${tok(e.tokens_in)} in, ${tok(e.tokens_out)} out</span><b>${model == null ? "your provider's price" : money(model)}</b></div>
          ${o.sandbox ? `<div class="row"><span>Sandbox · ~${minutes} min</span><b>${money(sandbox)}</b></div>` : ""}
          <div class="row total"><span>A typical rollout</span><b>${model == null ? `${money(sandbox)} + tokens` : `~${money(model + sandbox)}`}</b></div>
          ${e.tokens_in > 5e5 && st.source === "hf" ? `<div class="why">Agents re-read their context every step, so input tokens dominate. A cheaper model makes a big difference here.</div>` : ""}`;
      }
      const go = $("#run-go", box);
      if (go && !go.dataset.busy) {
        const ready = st.source === "hf" || (st.probe?.ok && (!o.sandbox || st.probe.tools));
        go.disabled = !ready;
        go.title = ready ? "" : "Test the connection first";
      }
    }
    box.oninput = (e) => { if (e.target.closest(".byo") && !e.target.matches("#ep-pin, #ep-pout")) st.probe = null; update(); };
    box.onchange = update;
    box.onclick = async (e) => {
      const r = e.target.closest("#run-runner [data-r]");
      if (r) { pick = opts.find((x) => x.runner === r.dataset.r) || pick; LS.set("run-runner", { ...LS.get("run-runner", {}), [spec]: pick.runner }); render(); return; }
      const vb = e.target.closest("[data-vis]");
      if (vb && !vb.disabled) {
        const was = st.visibility; st.visibility = vb.dataset.vis; update();
        if (was === "private" && st.visibility === "public") { const rc = vb.getBoundingClientRect(); confetti(rc.left + rc.width / 2, rc.top); toast("Thank you for sharing with the community"); }
        return;
      }
      const b = e.target.closest("[data-src]");
      if (b) { st.source = b.dataset.src; LS.set("source", st.source); update(); return; }
      if (e.target.closest("#ep-load")) {
        const btn = $("#ep-load", box); btn.disabled = true;
        try {
          const res = await api("/api/endpoints/models", { method: "POST", body: { base_url: val("#ep-url"), api_key: val("#ep-key") || null } });
          $("#ep-models", box).innerHTML = res.models.map((m) => `<option value="${esc(m)}">`).join("");
          status("", ""); toast(`${res.models.length} models found. Start typing to pick one.`);
          $("#ep-model", box).focus();
        } catch (err) { status("err", esc(err.message)); }
        btn.disabled = false; return;
      }
      if (e.target.closest("#ep-test")) {
        const btn = $("#ep-test", box); btn.disabled = true; btn.innerHTML = `${spinner()}Testing…`;
        try {
          const res = await api("/api/endpoints/test", { method: "POST", body: { base_url: val("#ep-url"), api_key: val("#ep-key") || null, model: val("#ep-model") } });
          st.probe = res;
          status(res.tools || !o.sandbox ? "ok" : "warn", `${esc(res.detail)} <span class="faint">${res.ms} ms</span>${!res.tools && o.sandbox ? " Agents need tool calling, so this endpoint can't run this task." : ""}`);
        } catch (err) { st.probe = null; status("err", esc(err.message)); }
        btn.disabled = false; btn.innerHTML = `${icon("plug", 14)}Test connection`; update(); return;
      }
      if (e.target.closest("#run-go")) {
        const go = $("#run-go", box);
        go.disabled = true; go.dataset.busy = "1"; go.innerHTML = `${spinner()}Starting…`;
        $("#run-err", box).innerHTML = "";
        const body = { dataset: spec, path: ref, runner: o.runner, fields: fields(), harness: val("#run-agent"), visibility: st.visibility };
        if (st.source === "byo") body.endpoint = endpoint(); else body.model = pk.value;
        try {
          const res = await api("/api/runs", { method: "POST", body });
          refreshActive();
          navigate(`/run/${encodeURIComponent(res.run.id)}`);
        } catch (err) {
          $("#run-err", box).innerHTML = `<div class="note-box err">${icon("alert")}<span>${esc(err.message)}</span></div>`;
          delete go.dataset.busy; go.innerHTML = `${icon("play", 15)}Run rollout`; update();
          if (err.status === 401) document.querySelector("[data-signin]")?.click();
        }
      }
    };
    update();
  };
  render();
}

// a run option's own inputs (contract.field)
function fieldHtml(f, saved) {
  const id = `run-f-${esc(f.key)}`, help = f.help && !f.advanced ? ` <em>${esc(f.help)}</em>` : "";
  const v = saved[f.key] ?? f.default;
  if (f.type === "model") return `<div class="field"><span>${esc(f.label)}${help}</span><div id="${id}"></div></div>`;
  if (f.type === "bool") return `<label class="check"><input type="checkbox" id="${id}" ${v ? "checked" : ""}> ${esc(f.label)}${help}</label>`;
  if (f.type === "select") return `<label class="field"><span>${esc(f.label)}${help}</span><select class="input" id="${id}">${(f.options || []).map((x) =>
    `<option value="${esc(optValue(x))}" ${optValue(x) === v ? "selected" : ""}>${esc(optLabel(x))}</option>`).join("")}</select></label>`;
  const attrs = [f.min != null && `min="${f.min}"`, f.max != null && `max="${f.max}"`, f.step != null && `step="${f.step}"`].filter(Boolean).join(" ");
  return `<label class="field"><span>${esc(f.label)}${f.advanced && f.help === "minutes" ? ", min" : ""}${help}</span><input class="input" id="${id}" type="${f.type === "number" ? "number" : "text"}" ${attrs}
    placeholder="${esc(f.placeholder || "")}" value="${esc(v ?? "")}"></label>`;
}
function fieldValues(box, fields, pickers) {
  const out = {};
  for (const f of fields) {
    if (f.type === "model") { if (pickers[f.key]) out[f.key] = pickers[f.key].value; continue; }
    const el = box.querySelector(`#run-f-${CSS.escape(f.key)}`);
    if (!el) continue;
    if (f.type === "bool") out[f.key] = el.checked;
    else if (f.type === "number") { if (el.value !== "") out[f.key] = +el.value; }
    else if (f.type === "select") { const x = (f.options || []).find((o) => String(optValue(o)) === el.value); if (x != null) out[f.key] = optValue(x); }
    else if (el.value !== "") out[f.key] = el.value;
  }
  return out;
}

// ── rollouts: this task's, and those of the same task in its other forms ────
// Every public rollout of it (how rewards spread, how each model does) and yours; tick two or more to compare them.
function stats(runs) {
  const scored = runs.filter((r) => r.reward != null && r.status === "done");
  const hist = Array(10).fill(0);
  scored.forEach((r) => (hist[Math.min(9, Math.floor(r.reward * 10))] += 1));
  const by = new Map();
  scored.forEach((r) => { const k = r.endpoint ? `${r.model} (own endpoint)` : r.model || "?"; by.set(k, [...(by.get(k) || []), r.reward]); });
  const models = [...by].map(([m, xs]) => ({ model: m, runs: xs.length, mean: xs.reduce((a, b) => a + b, 0) / xs.length, best: Math.max(...xs) }))
    .sort((a, b) => b.mean - a.mean || b.runs - a.runs);
  return { runs: runs.length, scored: scored.length, hist, models, full: scored.filter((r) => r.reward >= 0.999).length,
           mean: scored.length ? scored.reduce((a, r) => a + r.reward, 0) / scored.length : null };
}

async function taskRuns(box, spec, ref, v, sync) {
  let d;
  try { d = await api(`${envApi(spec)}/runs?ref=${encodeURIComponent(ref)}`); } catch { d = { mine: [], public: [], elsewhere: [] }; }
  if (!box.isConnected) return;
  const note = box.closest(".tp-sec")?.querySelector(".aside");
  if (note && d.elsewhere.length) note.textContent = "yours, and everyone's public ones, of this task in every form";
  const mineIds = new Set(d.mine.map((r) => r.id));
  const all = [...d.mine.map((r) => ({ ...r, mine: true })), ...d.public.filter((r) => !mineIds.has(r.id))];
  if (!all.length) {
    box.innerHTML = `<p class="muted sm">No rollouts of this task yet.${v.run?.options?.some((o) => o.ok) ? " Run one and keep it public: it appears here for everyone, without your name." : ""}</p>`;
    return;
  }
  const pub = [...d.public, ...d.mine.filter((r) => r.visibility === "public" && r.status === "done" && r.reward != null && !d.public.some((x) => x.id === r.id))];
  const s = stats(pub);
  const peak = Math.max(1, ...s.hist);
  const via = (r) => (r.dataset === spec && (r.path === ref || (!r.path && r.row)) ? "" : ` · via ${esc((r.dataset || "").split("/")[1] || "")}`);
  const runner = (r) => (r.runner === "mimo" ? "MiMo harness" : r.harness || "");
  box.innerHTML = `${s.scored ? `<div class="cm-stats">
      <div class="stat"><span>Public rollouts</span><b>${s.runs}</b></div>
      <div class="stat"><span>Mean reward</span><b>${s.mean == null ? "–" : s.mean.toFixed(2)}</b></div>
      <div class="stat"><span>Full marks</span><b>${s.full} of ${s.scored}</b></div>
      <div class="stat"><span>Models tried</span><b>${s.models.length}</b></div></div>
    <div class="cm-grid">
      <div><div class="sec-label">Reward spread</div>
        <div class="hist" title="Rollouts per reward band">${s.hist.map((n, i) => `<i style="height:${Math.round((n / peak) * 100)}%" title="${(i / 10).toFixed(1)}–${((i + 1) / 10).toFixed(1)}: ${n}"></i>`).join("")}</div>
        <div class="hist-axis"><span>0</span><span>0.5</span><span>1</span></div></div>
      <div><div class="sec-label">By model</div>
        <table class="mtable"><thead><tr><th>Model</th><th>Runs</th><th>Mean</th><th>Best</th></tr></thead><tbody>
        ${s.models.slice(0, 8).map((m) => `<tr><td title="${esc(m.model)}">${esc(m.model.split("/")[1] || m.model)}</td><td>${m.runs}</td>
          <td><span class="bar" style="width:${Math.round(m.mean * 48)}px"></span>${m.mean.toFixed(2)}</td><td>${rewardText(m.best)}</td></tr>`).join("")}</tbody></table></div>
    </div>` : ""}
    <div class="sec-label" style="margin-top:${s.scored ? 18 : 0}px">Rollouts${all.length > 1 ? '<span class="hint">tick two or more to compare them</span>' : ""}</div>
    <div class="runlist cm-list">${all.map((r) => `<div class="runsel"><label class="pick" title="Select to compare">
      <input type="checkbox" data-cmp="${esc(r.id)}" aria-label="Select this rollout to compare"></label>
      <a class="runrow trow" href="/run/${esc(r.id)}"><span class="lead">${r.mine ? statusPill(r.status) + visibilityBadge(r.visibility) : rewardBadge(r.reward, r.status)}</span>
        <span class="m">${esc(r.endpoint ? r.model || "custom endpoint" : (r.model || "").split("/")[1] || r.model || "custom endpoint")}${r.endpoint ? ' <span class="chip">own endpoint</span>' : ""}
          <span class="faint xs">${esc(runner(r))}${r.mine ? " · yours" : ""}${via(r)}</span></span>
        <span class="rw">${r.mine ? rewardBadge(r.reward, r.status) : ""}</span><span class="c">${money((r.cost || {}).total)}</span>
        <span class="w2">${ago(r.created_at)}</span>${icon("chevronRight", 14)}</a></div>`).join("")}</div>`;
  sync();
}

function syncPicks(el, spec, ref, picks) {
  const full = picks.size >= COMPARE_MAX;
  $$("[data-cmp]", el).forEach((c) => { c.checked = picks.has(c.dataset.cmp); c.disabled = full && !c.checked; c.closest(".pick").title = c.disabled ? `Up to ${COMPARE_MAX} at once` : "Select to compare"; });
  const bar = $("#cmp-bar", el);
  if (!bar) return;
  bar.hidden = !picks.size;
  const n = picks.size;
  bar.innerHTML = `<span><b>${n}</b> selected${n < 2 ? " · pick one more to compare" : ""}</span>
    <button class="btn sm ghost" type="button" data-cmp-clear>Clear</button>
    ${n >= 2 ? `<a class="btn sm primary" href="${compareHref(spec, ref, [...picks])}">${icon("columns", 13)}Compare ${n}</a>`
      : `<button class="btn sm primary" type="button" disabled>${icon("columns", 13)}Compare</button>`}`;
}
