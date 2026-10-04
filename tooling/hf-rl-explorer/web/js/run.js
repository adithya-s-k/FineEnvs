// A rollout, live: where it is (task, sandbox, agent, grading), the agent's trajectory as the capture proxy recorded
// it, and the reward when it lands. Laid out like the MiMo explorer's run page.
import { $, api, esc, md, dur, ago, statusPill, toast, LIVE_STATUSES, rewardClass, rewardText, sk, emptyState, visibilityBadge,
  confetti } from "./util.js";
import { icon } from "./icons.js";
import { refreshActive } from "./session.js";

const PHASES = [["task", "Task"], ["sandbox", "Sandbox"], ["agent", "Agent"], ["grading", "Grading"], ["done", "Done"]];
const ORDER = { queued: -1, task: 0, sandbox: 1, agent: 2, grading: 3, done: 4 };
const TOOL_ICON = { bash: "terminal", shell: "terminal", execute_bash: "terminal", read: "file", read_file: "file", write: "pencil",
  write_file: "pencil", edit: "pencil", str_replace_editor: "pencil", grep: "search", glob: "search", todowrite: "check", list: "list" };
const enc = (s) => s.split("/").map(encodeURIComponent).join("/");
// a run's task page: its path is the task's ref in its environment (rollouts from before refs named a row apart)
const refOf = (r) => (r.row ? `${r.row.config === "default" ? "" : `${r.row.config}/`}${r.row.split}/${r.row.i}` : r.path);
const taskHref = (r) => `/t/${enc(r.dataset)}/${enc(refOf(r))}`;
let timer = null, alive = true;

let other = null;   // the MiMo harness's own run page, for its rollouts (run-mimo.js)
export function unmount() { alive = false; clearTimeout(timer); other?.unmount(); other = null; }
const isLive = (r) => r && LIVE_STATUSES.includes(r.status);
const served = (r) => (!r.provider ? "the HF router" : r.provider === "fastest" ? "the fastest provider" : r.provider === "cheapest" ? "the cheapest provider" : r.provider);

function skeleton() {
  return `<div class="wrap page"><div class="crumbs">${sk.line(18)}</div>
    <div class="rp-head">${sk.line(22, 22)}${sk.line(55, 24)}${sk.line(45)}</div>
    <div class="stepper">${PHASES.map(() => `<div class="step">${sk.box(20, "width:20px;border-radius:50%")}${sk.line(60)}</div>`).join("")}</div>
    <div class="rp-grid"><div style="display:grid;gap:10px">${[70, 90, 55, 85].map((w) => sk.box(30, `width:${w}%;border-radius:10px`)).join("")}</div>
      <div style="display:grid;gap:12px">${sk.card(sk.line(30, 14) + sk.line(25, 40) + sk.lines(90, 70))}</div></div></div>`;
}

export async function mount(el, { id }) {
  alive = true;
  el.innerHTML = skeleton();
  const st = { id, run: null, traj: [], follow: true, shown: 0 };
  let first;
  try { first = await api(`/api/runs/${encodeURIComponent(id)}`); } catch (e) {
    if (!alive) return;
    el.innerHTML = `<div class="wrap page">${emptyState("search", "Rollout not found", e.status === 404 ? "It may be private, or the link is wrong." : esc(e.message),
      `<a class="btn" href="/runs">${icon("arrowLeft")}Your rollouts</a>`)}</div>`;
    return;
  }
  if (!alive) return;
  if (["mimo", "nemo-gym"].includes(first.run.runner)) {
    other = await import("./run-mimo.js");
    if (alive) other.mount(el, id, first);
    return;
  }
  el.innerHTML = `<div class="wrap page rp fade-in">
    <nav class="crumbs" aria-label="Breadcrumb"><a href="/runs">Rollouts</a>${icon("chevronRight", 13)}<span>${esc(id)}</span></nav>
    <header class="rp-head" id="rp-head"></header>
    <div class="stepper" id="rp-phases" style="--n:${PHASES.length}"></div>
    <div class="rp-grid">
      <div class="rp-main"><div class="tl-bar"><h2>What the agent did</h2><span class="n" id="rp-n"></span>
        <div class="tools-right"><button class="btn sm ghost" id="rp-expand" type="button">Expand all</button>
        <label class="switch"><input type="checkbox" id="rp-follow" checked> Follow live</label></div></div>
        <ol class="timeline" id="rp-tl"></ol><div id="rp-tail"></div></div>
      <aside class="rp-side"><section class="panel grade" id="rp-grade"></section><section class="panel costcard" id="rp-cost"></section>
        <section class="panel meta-card" id="rp-meta"></section></aside>
    </div></div>`;
  $("#rp-follow", el).addEventListener("change", (e) => (st.follow = e.target.checked));
  $("#rp-expand", el).addEventListener("click", (e) => {
    const open = e.target.textContent === "Expand all";
    el.querySelectorAll("#rp-tl details").forEach((d) => (d.open = open));
    e.target.textContent = open ? "Collapse all" : "Expand all";
  });
  el.addEventListener("click", async (e) => {
    if (e.target.closest("#rp-stop")) {
      e.target.closest("#rp-stop").disabled = true;
      try { await api(`/api/runs/${encodeURIComponent(id)}/cancel`, { method: "POST" }); toast("Stopping: the sandbox is shut down"); } catch (err) { toast(err.message); }
    }
    const v = e.target.closest("[data-vis]");
    if (v) {
      try {
        const r = await api(`/api/runs/${encodeURIComponent(id)}/visibility`, { method: "POST", body: { visibility: v.dataset.vis } });
        st.run = r.run; renderHead(el, st);
        if (v.dataset.vis === "public") { const b = v.getBoundingClientRect(); confetti(b.left + b.width / 2, b.top); toast("Public: thank you for sharing"); }
        else toast("Private: only you can see this rollout");
      } catch (err) { toast(err.message); }
    }
  });
  apply(el, st, first);
  const poll = async () => {
    if (!alive) return;
    try { apply(el, st, await api(`/api/runs/${encodeURIComponent(id)}`)); } catch { /* next tick */ }
    if (alive && isLive(st.run)) timer = setTimeout(poll, 2000);
    else refreshActive();
  };
  if (isLive(st.run)) timer = setTimeout(poll, 2000);
}

function apply(el, st, d) {
  st.run = d.run;
  const grew = d.trajectory.length !== st.traj.length || JSON.stringify(d.trajectory.at(-1)) !== JSON.stringify(st.traj.at(-1));
  st.traj = d.trajectory;
  renderHead(el, st);
  renderPhases(el, st);
  if (grew) renderTimeline(el, st);
  renderTail(el, st);
  renderGrade(el, st);
  renderCost(el, st);
  renderMeta(el, st);
}

function renderHead(el, st) {
  const r = st.run, owner = r.is_owner;
  const end = r.finished_at || Date.now() / 1000, start = r.started_at || r.created_at;
  const vis = owner && !r.restricted ? `<span class="seg sm">${["public", "private"].map((v) => `<button type="button" data-vis="${v}" aria-pressed="${r.visibility === v}">${icon(v === "public" ? "globe" : "shield", 12)}${v === "public" ? "Public" : "Private"}</button>`).join("")}</span>`
    : visibilityBadge(r.visibility);
  const model = r.endpoint ? r.model : (r.model || "").split("/")[1] || r.model;
  $("#rp-head", el).innerHTML = `<div class="kick">${statusPill(r.status)}${r.reward != null ? `<span class="reward ${rewardClass(r.reward)}">${rewardText(r.reward)}</span>` : ""}
      ${owner ? "" : '<span class="when">shared anonymously</span>'}
      <span class="when">${icon("clock", 13)}${dur(end - start)} · started ${ago(r.created_at)}</span></div>
    <h1><a href="${taskHref(r)}">${esc(r.title)}</a></h1>
    <div class="rp-bar"><div class="facts">
      <span>${icon("database", 14)}<a class="u" href="/d/${enc(r.dataset)}"><code>${esc(r.dataset)}</code></a></span>
      <span>${icon("cpu", 14)}<b>${esc(model)}</b>${r.endpoint ? " via your endpoint" : ` via ${esc(served(r))}`}</span>
      <span>${icon("terminal", 14)}${esc(r.harness)}</span><span>${icon("box", 14)}HF Sandbox ${esc(r.flavor || "")}</span></div>
      <div class="rp-actions">${vis}<a class="btn sm" href="${taskHref(r)}">${icon("file", 13)}View task</a>
        ${owner && isLive(r) ? `<button class="btn sm danger" id="rp-stop" type="button">${icon("stop", 13)}Stop</button>` : ""}</div></div>
    ${r.error ? `<div class="note-box err">${icon("alert")}<span>${esc(r.error)}</span></div>` : ""}
    ${r.note ? `<div class="note-box warn">${icon("clock")}<span>${esc(r.note)}. The task's tests graded what it had done by then.</span></div>` : ""}`;
}

function renderPhases(el, st) {
  const r = st.run;
  const failed = ["failed", "cancelled", "interrupted"].includes(r.status);
  const at = r.status === "done" ? 4 : ORDER[r.phase] ?? -1;
  const pt = r.phase_timings || {};
  const took = { sandbox: pt.environment_setup, agent: (pt.agent_setup || 0) + (pt.agent_execution || 0) || null, grading: pt.verifier };
  $("#rp-phases", el).innerHTML = PHASES.map(([k, label], i) => {
    const state = i < at || (r.status === "done" && i === at) ? "ok" : i === at ? (failed ? "bad" : "on") : "off";
    const mk = state === "ok" ? icon("check", 12) : state === "bad" ? icon("x", 12) : "";
    return `<div class="step ${state}"><span class="mk">${mk}</span><b>${label}</b><span class="t">${took[k] ? dur(took[k]) : ""}</span></div>`;
  }).join("");
}

// ── the trajectory: OpenAI-style messages as timeline rows ───────────────────
const row = (cls, ic, body) => `<li class="ev ${cls}"><span class="ei">${icon(ic, 14)}</span><div class="eb">${body}</div><span class="ts"></span></li>`;
const summary = (inner) => `<summary>${icon("chevronRight", 13, "chev")}${inner}</summary>`;
const textOf = (c) => (typeof c === "string" ? c : Array.isArray(c) ? c.map((p) => (typeof p === "string" ? p : p?.text || "")).join("\n") : c == null ? "" : JSON.stringify(c));
function parseArgs(a) { if (typeof a !== "string") return a || {}; try { return JSON.parse(a); } catch { return { arguments: a }; } }

function rows(messages) {
  const results = {};
  messages.forEach((m) => { if (m.role === "tool" && m.tool_call_id) results[m.tool_call_id] = m; });
  let users = 0, steps = 0;
  const out = [];
  for (const m of messages) {
    const text = textOf(m.content);
    if (m.role === "system") {
      out.push(row("ev-prompt", "doc", `<details>${summary(`<b>System prompt</b><span class="peek">${text.length.toLocaleString()} characters, from the agent</span>`)}<div class="prompt-text">${esc(text)}</div></details>`));
    } else if (m.role === "user") {
      users++;
      out.push(row("ev-prompt", "doc", `<details ${users === 1 ? "open" : ""}>${summary(`<b>${users === 1 ? "Instructions to the model" : "Message to the model"}</b><span class="peek">${text.length.toLocaleString()} characters</span>`)}<div class="prompt-text">${esc(text)}</div></details>`));
    } else if (m.role === "assistant") {
      const thinking = m.reasoning_content || m.reasoning || "";
      if (thinking) out.push(row("ev-think", "brain", `<details>${summary(`<b>Thinking</b><span class="peek">${esc(String(thinking).replace(/\s+/g, " ").slice(0, 220))}</span>`)}<div class="thought">${esc(thinking)}</div></details>`));
      if (text.trim()) { out.push(row("ev-text", "message", `<div class="msg prose">${md(text)}</div>`)); steps++; }
      for (const c of m.tool_calls || []) {
        const fn = c.function || {}, args = parseArgs(fn.arguments), res = results[c.id];
        const head = args.command || args.cmd || args.keystrokes || args.filePath || args.path || args.pattern || Object.values(args).map(String).join(" ");
        const output = res ? textOf(res.content) : "";
        out.push(row(`ev-tool`, TOOL_ICON[fn.name] || "terminal", `<details>${summary(`<b>${esc(fn.name || "tool")}</b><code>${esc(String(head).split("\n")[0].slice(0, 200))}</code>`)}
          <div class="io"><h5>Input</h5><pre>${esc(JSON.stringify(args, null, 2))}</pre><h5>Output</h5><pre>${esc(output || (res ? "(empty)" : "(waiting)"))}</pre></div></details>`));
        steps++;
      }
    } else if (m.role === "tool" && !m.tool_call_id) {
      out.push(row("ev-log", "terminal", `<details>${summary(`<b>Tool output</b><code>${esc(text.split("\n")[0].slice(0, 160))}</code>`)}<pre class="code-block">${esc(text)}</pre></details>`));
    }
  }
  return { html: out.join(""), steps };
}

function renderTimeline(el, st) {
  const tl = $("#rp-tl", el);
  const open = new Set([...tl.querySelectorAll("details")].map((d, i) => (d.open ? i : -1)).filter((i) => i >= 0));
  const { html, steps } = rows(st.traj);
  tl.innerHTML = html;
  tl.querySelectorAll("details").forEach((d, i) => { if (open.has(i)) d.open = true; });
  $("#rp-n", el).textContent = steps ? `${steps} step${steps === 1 ? "" : "s"}` : "";
  if (st.follow && isLive(st.run)) tl.lastElementChild?.scrollIntoView({ block: "nearest", behavior: "smooth" });
}

function renderTail(el, st) {
  const r = st.run, t = $("#rp-tail", el);
  const msg = { queued: "Waiting to start…", task: "Fetching the task…", sandbox: "Starting the HF Sandbox from the task's image and installing the agent…",
    agent: "The agent is working…" }[r.phase];
  t.innerHTML = isLive(r) && msg ? `<ol class="timeline"><li class="ev ev-live still"><span class="ei"><span class="spinner"></span></span><div class="eb"><span class="live-msg">${esc(msg)}</span></div><span class="ts"></span></li></ol>`
    : !st.traj.length && !isLive(r) ? `<p class="tl-empty">No model calls were recorded.</p>` : "";
}

function renderGrade(el, st) {
  const r = st.run, box = $("#rp-grade", el);
  const h = `<div class="panel-h"><h3>${icon("scale", 14)}Grade</h3>${r.reward_key ? `<span class="aside">reward key <code>${esc(r.reward_key)}</code></span>` : ""}</div>`;
  if (r.reward == null) {
    box.innerHTML = `${h}<div class="panel-b"><p class="muted sm">${isLive(r) ? `Appears here once the agent finishes and the task's ${r.judge ? `grader has run (it asks <code>${esc(r.judge)}</code>, through a relay that only this rollout's grading can use)` : "tests have run"}.` : "This rollout was not graded."}</p></div>`;
    return;
  }
  const cls = rewardClass(r.reward);
  const named = Object.entries(r.rewards || {});
  box.innerHTML = `${h}<div class="panel-b">
    <div class="score"><span class="big ${cls}">${rewardText(r.reward)}</span><span class="of">reward, from the task's ${r.judge ? `grader, with <code>${esc(r.judge)}</code> as its judge${r.judge_calls != null ? ` (${r.judge_calls} call${r.judge_calls === 1 ? "" : "s"})` : ""}` : "tests"}</span></div>
    ${named.length > 1 ? `<ul class="gchecks">${named.map(([k, v]) => `<li class="na"><span class="mk">${icon("more", 14)}</span>
      <div><b>${esc(k)}</b></div><span class="sc">${Number(v).toFixed(2)}</span></li>`).join("")}</ul>` : ""}
    ${(r.findings || []).length ? `<details class="notes"><summary class="disclose">${icon("chevronRight", 14, "chev")}Capture notes (${r.findings.length})</summary>
      <ul class="rules">${r.findings.map((f) => `<li>${esc(f)}</li>`).join("")}</ul></details>` : ""}
  </div>`;
}

function renderCost(el, st) {
  const r = st.run, c = r.cost || {};
  const wall = r.wall_s ?? ((r.finished_at || Date.now() / 1000) - (r.started_at || r.created_at));
  $("#rp-cost", el).innerHTML = `<div class="panel-h"><h3>${icon("coins", 14)}Cost</h3>${isLive(r) ? `<span class="aside"><i class="pulse" style="color:var(--live)"></i> live</span>` : ""}</div>
    <div class="panel-b"><dl class="kv sm">
      <dt>Sandbox time</dt><dd>${dur(wall)}${c.sandbox != null ? ` <span class="faint">${c.sandbox < 0.01 ? "under $0.01" : `about $${c.sandbox.toFixed(2)}`}</span>` : ""}</dd>
      <dt>Model calls</dt><dd>${r.n_turns ?? 0}</dd>${r.judge ? `<dt>Judge calls</dt><dd>${r.judge_calls ?? (isLive(r) ? "after grading" : "–")}</dd>` : ""}</dl>
    <p class="fine">${r.endpoint ? "Tokens are billed by your endpoint; the sandbox by Hugging Face." : `Billed to ${r.is_owner ? "your" : "the runner's"} Hugging Face account: the sandbox by the hour, tokens at the provider's price.`}</p></div>`;
}

function renderMeta(el, st) {
  const r = st.run, p = r.params || {};
  const row2 = (k, v) => (v == null || v === "" ? "" : `<dt>${k}</dt><dd>${v}</dd>`);
  const code = (v) => `<code>${esc(v)}</code>`;
  $("#rp-meta", el).innerHTML = `<div class="panel-h"><h3>${icon("info", 14)}Run details</h3></div><div class="panel-b"><dl class="kv sm">
    ${row2("Model", code(r.model || ""))}
    ${row2("Served by", r.endpoint ? "an OpenAI-compatible endpoint" : esc(served(r)))}
    ${row2("Agent", esc(r.harness))}
    ${row2("Judge", r.judge ? code(r.judge) : "")}
    ${row2("Step cap", esc(p.steps || "none"))}
    ${row2("Time limit", p.timeout_min ? esc(`${p.timeout_min} min`) : "the task's")}
    ${row2("Sandbox", esc(`HF Sandbox, ${r.flavor || "cpu-basic"}`))}
    ${row2("Image", r.image ? code(r.image) : "")}
    ${row2("Task", code(`${r.dataset}/${r.path}`))}
    ${row2("Runner", "OpenEnv's Harbor runner and capture proxy")}
    ${row2("Rollout", code(r.id))}</dl></div>`;
}
