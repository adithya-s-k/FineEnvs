// Play: a person runs an episode exactly as an agent does. Tasks come from the env's Task API
// (/berth_planning/...), the episode is an OpenEnv WebSocket session (/ws): reset, list_tools, call_tool.
// The draft plan is edited on the berth chart; only the env's tool responses carry costs and the reward.
import { BerthChart, legendHtml } from "./chart.js";
import { DIFF_RANK, badgesHtml, callKinds, clockAt, cranePoolAt, divertWindows, escapeHtml, evaluatePlan, fmtNum, hasCranes, horizonOf, movesOf, parsePlanArgs, publishedPlan, sectionsLabel } from "./model.js";
import { createStage, sceneMarkup } from "./stage.js";
import { checkSummary, planTable } from "./transcript.js";

// Model rollouts and the eval live in their own Space; a local server keeps its own explorer.
const EXPLORER = /\.hf\.space$/.test(location.hostname) ? "https://fineenvs-portsimenv-eval.hf.space/viewer/" : "/viewer/";

/* ---------------- Task API ---------------- */

async function taskApi(path, body) {
  const r = await fetch(new URL(`/berth_planning/${path}`, location.origin).toString(), {
    method: body ? "POST" : "GET",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!r.ok) throw new Error(`Task API ${path}: ${r.status} ${r.statusText || ""}`.trim());
  return r.json();
}
const taskCache = new Map();
function splitTasks(split, n) {
  if (!taskCache.has(split)) {
    const p = taskApi("task_range", { split, start: 0, stop: n }).then((d) => d.tasks || []);
    p.catch(() => taskCache.delete(split));
    taskCache.set(split, p);
  }
  return taskCache.get(split);
}

/* ---------------- OpenEnv WebSocket session ---------------- */

class EnvSession {
  constructor(onStatus) {
    this.onStatus = onStatus;
    this.ws = null;
    this.pending = [];
    this.status = "idle";
    this.closing = false;
  }
  _set(s, detail) {
    this.status = s;
    if (this.onStatus) this.onStatus(s, detail);
  }
  connect() {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) return Promise.resolve();
    if (this._opening) return this._opening;
    this.closing = false;
    const url = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`;
    this._set("connecting");
    this._opening = new Promise((resolve, reject) => {
      let ws;
      try {
        ws = new WebSocket(url);
      } catch (e) {
        this._opening = null;
        this._set("closed", String(e));
        reject(e);
        return;
      }
      this.ws = ws;
      let opened = false;
      ws.onopen = () => {
        opened = true;
        this._opening = null;
        this._set("open");
        resolve();
      };
      ws.onmessage = (e) => {
        let msg;
        try {
          msg = JSON.parse(e.data);
        } catch {
          msg = { type: "error", data: { message: String(e.data) } };
        }
        const p = this.pending.shift();
        if (p) p.resolve(msg);
      };
      ws.onclose = (e) => {
        this._opening = null;
        for (const p of this.pending.splice(0)) p.reject(new Error("connection closed"));
        if (this.ws === ws) this.ws = null;
        if (!opened) reject(new Error(`could not open ${url}`));
        this._set(this.closing ? "idle" : "dropped", e.code);
      };
      ws.onerror = () => {};
    });
    return this._opening;
  }
  async request(msg) {
    await this.connect();
    return new Promise((resolve, reject) => {
      this.pending.push({ resolve, reject });
      try {
        this.ws.send(JSON.stringify(msg));
      } catch (e) {
        this.pending.pop();
        reject(e);
      }
    });
  }
  reset(data) {
    return this.request({ type: "reset", data });
  }
  listTools() {
    return this.request({ type: "step", data: { type: "list_tools" } });
  }
  callTool(name, args) {
    return this.request({ type: "step", data: { type: "call_tool", tool_name: name, arguments: args } });
  }
  state() {
    return this.request({ type: "state" });
  }
  close() {
    this.closing = true;
    if (this.ws) {
      try {
        if (this.ws.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify({ type: "close" }));
        this.ws.close();
      } catch {}
    }
    this.ws = null;
  }
}

/** Observation payload of a response; metadata may sit at the top level and/or inside the observation. */
function unwrap(resp) {
  if (!resp) return { error: "no response" };
  if (resp.type === "error") return { error: (resp.data && (resp.data.message || resp.data.code)) || "error" };
  const d = resp.data || {};
  const obs = d.observation || {};
  const meta = { ...(obs.metadata || {}), ...(d.metadata || {}) };
  return { obs, meta, reward: d.reward ?? obs.reward ?? null, done: !!(d.done ?? obs.done) };
}

/** Text content of a tool result, parsed as JSON when it is JSON. */
function toolResult(obs) {
  if (obs.error) return { error: obs.error.message || obs.error.error_type || JSON.stringify(obs.error) };
  const res = obs.result;
  if (res == null) return { error: "empty result" };
  let text = null;
  if (typeof res === "string") text = res;
  else if (Array.isArray(res.content)) text = res.content.filter((c) => c && c.type === "text").map((c) => c.text).join("\n");
  else if (res.structured_content && typeof res.structured_content.result === "string") text = res.structured_content.result;
  else if (res.data != null) text = typeof res.data === "string" ? res.data : JSON.stringify(res.data);
  if (text == null) return { json: res };
  try {
    const j = JSON.parse(text);
    if (j && typeof j === "object" && j.error && Object.keys(j).length <= 2) return { error: j.error, json: j };
    return { json: j, text };
  } catch {
    return { text };
  }
}

/* ---------------- small markdown renderer (situation text) ---------------- */

function inline(s) {
  return escapeHtml(s)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>");
}

export function mdToHtml(md) {
  const lines = String(md || "").split("\n");
  const out = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) {
      i++;
      continue;
    }
    const h = line.match(/^(#{1,4})\s+(.*)$/);
    if (h) {
      out.push(`<h${Math.min(6, h[1].length + 2)} class="md-h">${inline(h[2])}</h${Math.min(6, h[1].length + 2)}>`);
      i++;
      continue;
    }
    if (/^\s*\|/.test(line)) {
      const rows = [];
      while (i < lines.length && /^\s*\|/.test(lines[i])) rows.push(lines[i++]);
      const cells = (r) => r.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim());
      const body = rows.filter((r) => !/^\s*\|[\s:|-]+\|\s*$/.test(r));
      const head = cells(body.shift() || "");
      out.push(`<table class="tbl md-t"><thead><tr>${head.map((c) => `<th>${inline(c)}</th>`).join("")}</tr></thead><tbody>${body.map((r) => `<tr>${cells(r).map((c) => `<td class="${/^-?[\d.]+$/.test(c) ? "num" : ""}">${inline(c)}</td>`).join("")}</tr>`).join("")}</tbody></table>`);
      continue;
    }
    if (/^\s*[-*]\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) {
        let it = lines[i++].replace(/^\s*[-*]\s+/, "");
        while (i < lines.length && /^\s{2,}\S/.test(lines[i]) && !/^\s*[-*]\s+/.test(lines[i])) it += ` ${lines[i++].trim()}`;
        items.push(it);
      }
      out.push(`<ul class="md-l">${items.map((t) => `<li>${inline(t)}</li>`).join("")}</ul>`);
      continue;
    }
    const para = [];
    while (i < lines.length && lines[i].trim() && !/^(#{1,4}\s|\s*\||\s*[-*]\s)/.test(lines[i])) para.push(lines[i++]);
    out.push(`<p class="md-p">${inline(para.join(" "))}</p>`);
  }
  return out.join("");
}

/* ---------------- helpers ---------------- */

function disruptionSummary(list) {
  const counts = {};
  for (const d of list || []) {
    const k = typeof d === "string" ? d : d.type;
    counts[k] = (counts[k] || 0) + 1;
  }
  return Object.entries(counts)
    .map(([k, v]) => (v > 1 ? `${v} ${k}` : k))
    .join(", ");
}

function schemaSummary(schema) {
  const props = (schema && schema.properties) || {};
  const names = Object.keys(props);
  if (!names.length) return "no arguments";
  const typeOf = (p) => {
    if (!p) return "any";
    if (p.anyOf) return p.anyOf.map(typeOf).join(" or ");
    if (p.type === "array") {
      const it = p.items || {};
      if (it.type === "object" && it.properties) return `array of {${Object.keys(it.properties).join(", ")}}`;
      return `array of ${typeOf(it)}`;
    }
    return p.type || "any";
  };
  return names.map((n) => `${n}${(schema.required || []).includes(n) ? "" : "?"}: ${typeOf(props[n])}`).join(" · ");
}

const isPlanTool = (tool) => !!(tool && tool.input_schema && tool.input_schema.properties && tool.input_schema.properties.plan);

function classify(problem) {
  if (/overlaps ship|overlap in sections/.test(problem)) return "overlap";
  if (/cannot arrive/.test(problem)) return "before arrival";
  if (/outside the quay|but the quay has sections/.test(problem)) return "off the quay";
  if (/closed sections/.test(problem)) return "closed section";
  if (/alongside/.test(problem)) return "ship alongside";
  if (/no-movement window/.test(problem)) return "docks in wind";
  if (/cranes over the pool/.test(problem)) return "cranes over pool";
  if (/ships move \(limit/.test(problem)) return "too many moves";
  if (/can be worked by/.test(problem)) return "crane count";
  return "problem";
}

function draftProblemsSummary(ev) {
  if (!ev.issues.length) return '<i class="dot ok"></i>No rule problems in the draft';
  const c = {};
  for (const i of ev.issues) {
    const k = classify(i.problem);
    c[k] = (c[k] || 0) + 1;
  }
  const parts = Object.entries(c).map(([k, v]) => `${v} ${k}${v > 1 && k === "overlap" ? "s" : ""}`);
  return `<i class="dot bad"></i>${parts.join(" · ")}`;
}

function ago(ts) {
  const d = new Date(ts);
  return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}:${String(d.getSeconds()).padStart(2, "0")}`;
}

/* ---------------- page ---------------- */

const TOOL_LABEL = { get_situation: "get_situation", check_plan: "Check plan", submit_plan: "Submit plan" };

export async function playPage({ app, params, isCurrent, setTeardown, sceneMod }) {
  const st = {
    splits: [],
    split: params.get("split") || null,
    selected: params.get("task") || null,
    phase: "none",
    task: null,
    meta: null,
    tools: [],
    draft: new Map(),
    undo: [],
    lastCheck: null,
    lastSummary: "",
    log: [],
    unseenCalls: 0,
    tab: "ships",
    shipSel: null,
    done: false,
    endReason: null,
    grade: null,
    rubric: null,
    checksUsed: 0,
    toolCalls: 0,
    busy: false,
    H: 0,
  };
  let stage = null;
  let charts = [];
  let boxRO = null;
  let pickerTasks = [];
  const session = new EnvSession((s) => onConnStatus(s));
  const onKey = (e) => {
    if (e.key === "Escape") {
      closePopover();
      if (st.phase === "episode") closePicker();
    }
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z" && !e.target.closest("input, textarea") && st.phase === "episode") {
      e.preventDefault();
      undo();
    }
  };
  window.addEventListener("keydown", onKey);
  document.documentElement.classList.add("play-app");
  setTeardown(() => {
    window.removeEventListener("keydown", onKey);
    document.documentElement.classList.remove("play-app");
    session.close();
    destroyStage();
  });

  function destroyStage() {
    for (const c of charts) c.destroy();
    charts = [];
    if (boxRO) boxRO.disconnect();
    boxRO = null;
    if (stage) stage.destroy();
    stage = null;
  }

  /* ---------- shell ---------- */

  app.innerHTML = `
  <div class="pa">
    <header class="pa-top">
      <div class="pa-id"><b id="pa-task">Play</b><span class="muted" id="pa-meta">pick a task to start an episode</span></div>
      <div class="pa-state" id="pa-state"></div>
      <div class="pa-acts">
        <span class="pa-tb" id="pa-toolbtns"></span>
        <span class="pa-confirm" id="pa-confirm" hidden><span>Submit this plan? It ends the episode.</span><button type="button" class="btn sm primary" data-act="confirm-submit">Submit</button><button type="button" class="btn sm" data-act="cancel-submit">Cancel</button></span>
        <button type="button" class="btn sm" id="pa-tools-b" aria-expanded="false" aria-controls="pa-tools" title="What list_tools returned: descriptions, parameters, and the exact arguments that will be sent" hidden>Tools</button>
        <button type="button" class="btn sm" data-act="pick">Change task</button>
        <span class="muted small" id="conn"></span>
        <a class="ext" href="${EXPLORER}#/tasks" target="_blank" rel="noopener" title="Tasks and model rollouts, in a new tab">Eval explorer ↗</a>
      </div>
      <div class="pa-pop" id="pa-tools" hidden></div>
    </header>
    <div class="pa-drop" id="drop" hidden><i class="dot bad"></i>The connection to the env closed; the server ended this episode with it. Your draft is kept. <button type="button" class="btn sm" data-act="reconnect">Reconnect and restart this task</button></div>
    <div class="pa-body">
      <div class="pa-main" id="pa-main"><div class="pa-empty muted">Pick a task to start an episode.</div></div>
      <aside class="pa-side" id="pa-side" hidden>
        <div class="pa-grade" id="grade" hidden></div>
        <div class="pa-tabs" role="tablist">
          <button type="button" role="tab" data-tab="ships" aria-selected="true">Ships</button>
          <button type="button" role="tab" data-tab="sit" aria-selected="false">Situation</button>
          <button type="button" role="tab" data-tab="calls" aria-selected="false">Calls <span class="badge" id="calls-badge"></span></button>
        </div>
        <div class="pa-panel" data-panel="ships"><div id="ships"></div><div class="ship-detail" id="ship-detail"></div></div>
        <div class="pa-panel" data-panel="sit" hidden><div id="situation"></div></div>
        <div class="pa-panel" data-panel="calls" hidden><div id="log" class="transcript"></div></div>
      </aside>
    </div>
    <div class="pa-drawer" id="picker" hidden>
      <div class="pa-drawer-in" role="dialog" aria-label="Choose a task">
        <div class="pd-head"><h2>Choose a task</h2><span class="muted small" title="Tasks come from the env's Task API; the episode runs over an OpenEnv WebSocket session, with the same tools an agent gets">Task API</span><span class="grow"></span><button type="button" class="btn sm" data-act="close-pick" id="pick-close">Close</button></div>
        <div class="pick-bar">
          <label class="fld">Split <select id="p-split" aria-label="Split"><option>Loading…</option></select></label>
          <input id="p-q" type="search" placeholder="Filter" aria-label="Filter tasks" autocomplete="off">
          <select id="p-diff" aria-label="Difficulty"><option value="">All difficulties</option></select>
        </div>
        <div class="pd-table"><table class="tbl click" id="p-table"><thead></thead><tbody>${Array.from({ length: 8 }, () => `<tr class="skel">${"<td><i></i></td>".repeat(6)}</tr>`).join("")}</tbody></table></div>
        <div id="p-err"></div>
        <div class="pd-foot"><span class="muted small pick-sel" id="p-sel">Select a task.</span><span class="grow"></span><button type="button" class="btn" id="p-random" title="Start an episode on a random task from this split">Random task</button><button type="button" class="btn primary" id="p-start" disabled>Start episode</button></div>
      </div>
    </div>
  </div>`;

  const $ = (sel) => app.querySelector(sel);

  function onConnStatus(s) {
    const el = $("#conn");
    if (el) el.innerHTML = s === "open" ? "" : s === "connecting" ? '<i class="dot warn"></i>connecting' : s === "dropped" ? '<i class="dot bad"></i>disconnected' : "";
    const banner = $("#drop");
    if (banner) banner.hidden = !(s === "dropped" && st.phase === "episode" && !st.done);
  }

  function bindShell() {
    app.querySelector(".pa").addEventListener("click", (e) => {
      const b = e.target.closest("[data-act], [data-call], [data-tab], [data-cr], #pa-tools-b");
      if (!b) {
        if (!e.target.closest("#pa-tools")) closePopover();
        return;
      }
      if (b.id === "pa-tools-b") return togglePopover();
      if (b.dataset.cr && typeof st.shipSel === "number") {
        const cur = st.draft.get(st.shipSel);
        return cur && setCranes(st.shipSel, cur.cranes + Number(b.dataset.cr));
      }
      if (b.dataset.tab) return setTab(b.dataset.tab);
      if (b.dataset.call) return onToolButton(b.dataset.call);
      const act = b.dataset.act;
      if (act === "pick") openPicker();
      else if (act === "close-pick") closePicker();
      else if (act === "confirm-submit") {
        showConfirm(false);
        callTool("submit_plan");
      } else if (act === "cancel-submit") showConfirm(false);
      else if (act === "again") start({ task_id: st.task.task_id }, st.task);
      else if (act === "new") openPicker();
      else if (act === "reconnect") start({ task_id: st.task.task_id }, st.task, { keepDraft: true });
      else if (act === "undo") undo();
      else if (act === "published") {
        pushUndo();
        st.draft = initialDraft(st.task);
        refresh();
      } else if (act === "load") {
        const en = st.log.find((x) => x.n === Number(b.dataset.n));
        const plan = en && parsePlanArgs(en.args);
        if (!Array.isArray(plan)) return;
        pushUndo();
        for (const p of plan) if (st.draft.has(p.ship) && Number.isInteger(p.berth_hour) && Number.isInteger(p.section)) st.draft.set(p.ship, { berth_hour: p.berth_hour, section: p.section, cranes: Number.isInteger(p.cranes) ? p.cranes : st.draft.get(p.ship).cranes });
        refresh();
      }
    });
    $("#picker").addEventListener("click", (e) => {
      if (e.target.id === "picker" && st.phase === "episode") closePicker();
    });
  }

  function setTab(tab) {
    st.tab = tab;
    for (const b of app.querySelectorAll(".pa-tabs [data-tab]")) b.setAttribute("aria-selected", String(b.dataset.tab === tab));
    for (const p of app.querySelectorAll(".pa-panel")) p.hidden = p.dataset.panel !== tab;
    if (tab === "calls") {
      st.unseenCalls = 0;
      const panel = app.querySelector('.pa-panel[data-panel="calls"]');
      panel.scrollTop = panel.scrollHeight;
    }
    renderBadge();
  }

  function renderBadge() {
    const b = $("#calls-badge");
    if (b) b.textContent = st.unseenCalls ? String(st.unseenCalls) : "";
  }

  /* ---------- picker (drawer) ---------- */

  let pickerBound = false;
  let sortKey = "task_id";
  let dir = 1;
  const pcols = [
    { k: "task_id", label: "Task", get: (t) => t.task_id },
    { k: "quay", label: "Quay", get: (t) => t.quay, opt: true },
    { k: "week", label: "Week", get: (t) => t.week, num: true, opt: true },
    { k: "difficulty", label: "Difficulty", get: (t) => t.difficulty, sort: (t) => DIFF_RANK[t.difficulty] ?? 9 },
    { k: "ships", label: "Ships", get: (t) => t.ships.length, num: true },
    { k: "disruptions", label: "Disruptions", get: (t) => disruptionSummary(t.disruptions), sort: (t) => (t.disruptions || []).length, opt: true },
  ];

  function pickerFiltered() {
    const s = $("#p-q").value.trim().toLowerCase();
    const d = $("#p-diff").value;
    return pickerTasks.filter((t) => (!d || t.difficulty === d) && (!s || `${t.task_id} ${t.quay} ${t.terminal} ${t.week} ${disruptionSummary(t.disruptions)}`.toLowerCase().includes(s)));
  }

  function pickerRender() {
    $("#p-table thead").innerHTML = `<tr>${pcols.map((c) => `<th class="${c.num ? "num" : ""} ${c.opt ? "opt" : ""}"><button type="button" class="th-b${c.k === sortKey ? " on" : ""}" data-k="${c.k}">${escapeHtml(c.label)}<span class="arr">${c.k === sortKey ? (dir > 0 ? "▲" : "▼") : ""}</span></button></th>`).join("")}</tr>`;
    const col = pcols.find((c) => c.k === sortKey);
    const val = col.sort || col.get;
    const rows = pickerFiltered().sort((a, b) => {
      const va = val(a);
      const vb = val(b);
      return (va < vb ? -1 : va > vb ? 1 : 0) * dir;
    });
    $("#p-table tbody").innerHTML = rows.length
      ? rows.map((t) => `<tr data-row="${escapeHtml(t.task_id)}" class="${t.task_id === st.selected ? "sel" : ""}">${pcols.map((c) => `<td class="${c.num ? "num" : ""} ${c.opt ? "opt" : ""}">${escapeHtml(c.get(t))}</td>`).join("")}</tr>`).join("")
      : `<tr><td colspan="${pcols.length}" class="muted">No tasks.</td></tr>`;
    const sel = pickerTasks.find((t) => t.task_id === st.selected);
    $("#p-start").disabled = !sel;
    $("#p-sel").innerHTML = sel ? `<b>${escapeHtml(sel.task_id)}</b> · quay ${escapeHtml(sel.quay)} · week ${sel.week} · ${sel.ships.length} ships · ${escapeHtml(disruptionSummary(sel.disruptions))}` : "Select a task.";
  }

  async function pickerLoadSplit() {
    $("#p-table tbody").innerHTML = Array.from({ length: 8 }, () => `<tr class="skel">${"<td><i></i></td>".repeat(6)}</tr>`).join("");
    const sp = st.splits.find((s) => s.name === st.split);
    try {
      pickerTasks = await splitTasks(st.split, sp ? sp.num_tasks : 1000);
    } catch (err) {
      if (!isCurrent()) return;
      $("#p-err").innerHTML = `<div class="error">Could not load tasks. <span class="muted">${escapeHtml(err.message)}</span></div>`;
      return;
    }
    if (!isCurrent()) return;
    const diffSel = $("#p-diff");
    const diffs = [...new Set(pickerTasks.map((t) => t.difficulty))].sort((a, b) => (DIFF_RANK[a] ?? 9) - (DIFF_RANK[b] ?? 9));
    const keep = diffSel.value;
    diffSel.innerHTML = `<option value="">All difficulties</option>${diffs.map((d) => `<option value="${escapeHtml(d)}">${escapeHtml(d)}</option>`).join("")}`;
    if (diffs.includes(keep)) diffSel.value = keep;
    pickerRender();
  }

  async function openPicker() {
    closePopover();
    $("#picker").hidden = false;
    $("#pick-close").hidden = st.phase !== "episode";
    if (!pickerBound) {
      pickerBound = true;
      $("#p-table thead").addEventListener("click", (e) => {
        const b = e.target.closest("button[data-k]");
        if (!b) return;
        if (b.dataset.k === sortKey) dir = -dir;
        else {
          sortKey = b.dataset.k;
          dir = 1;
        }
        pickerRender();
      });
      $("#p-table tbody").addEventListener("click", (e) => {
        const tr = e.target.closest("tr[data-row]");
        if (!tr) return;
        st.selected = tr.dataset.row;
        pickerRender();
      });
      $("#p-table tbody").addEventListener("dblclick", (e) => {
        const tr = e.target.closest("tr[data-row]");
        if (tr) start({ task_id: tr.dataset.row }, pickerTasks.find((t) => t.task_id === tr.dataset.row));
      });
      $("#p-split").addEventListener("change", () => {
        st.split = $("#p-split").value;
        pickerLoadSplit();
      });
      $("#p-q").addEventListener("input", pickerRender);
      $("#p-diff").addEventListener("change", pickerRender);
      $("#p-start").addEventListener("click", () => {
        const t = pickerTasks.find((x) => x.task_id === st.selected);
        if (t) start({ task_id: t.task_id }, t);
      });
      $("#p-random").addEventListener("click", () => {
        const pool = pickerFiltered().length ? pickerFiltered() : pickerTasks;
        if (!pool.length) return;
        const t = pool[Math.floor(Math.random() * pool.length)];
        st.selected = t.task_id;
        start({ split: t.split || st.split, index: pickerTasks.indexOf(t) }, t);
      });
      try {
        if (!st.splits.length) st.splits = await taskApi("splits");
      } catch (err) {
        if (!isCurrent()) return;
        $("#p-err").innerHTML = `<div class="error">The env's Task API is not reachable here. <span class="muted">${escapeHtml(err.message)}</span></div>`;
        $("#p-table tbody").innerHTML = "";
        return;
      }
      if (!isCurrent()) return;
      if (!st.split || !st.splits.some((s) => s.name === st.split)) st.split = (st.splits.find((s) => s.name === "test") || st.splits[0] || {}).name;
      $("#p-split").innerHTML = st.splits.map((s) => `<option value="${escapeHtml(s.name)}">${escapeHtml(s.name)} (${s.num_tasks})</option>`).join("");
      $("#p-split").value = st.split;
      await pickerLoadSplit();
    } else pickerRender();
    const sel = $("#p-table tr.sel");
    if (sel) sel.scrollIntoView({ block: "nearest" });
  }

  function closePicker() {
    $("#picker").hidden = true;
  }

  /* ---------- reset ---------- */

  async function start(resetData, taskHint, { keepDraft = false } = {}) {
    const btns = app.querySelectorAll("#p-start, #p-random, [data-act='again'], [data-act='reconnect']");
    for (const b of btns) b.disabled = true;
    const errEl = $("#picker").hidden ? null : $("#p-err");
    if (errEl) errEl.innerHTML = '<p class="muted small">Starting episode…</p>';
    let r;
    let tools;
    try {
      r = unwrap(await session.reset(resetData));
      if (r.error) throw new Error(r.error);
      const tl = unwrap(await session.listTools());
      tools = (tl.obs && tl.obs.tools) || [];
    } catch (err) {
      if (!isCurrent()) return;
      for (const b of btns) b.disabled = false;
      const msg = `<div class="error">Could not start an episode over the env's WebSocket. <span class="muted">${escapeHtml(err.message)}</span></div>`;
      if (errEl) errEl.innerHTML = msg;
      else setSummary(msg);
      return;
    }
    if (!isCurrent()) return;
    const meta = r.meta;
    let task = taskHint && taskHint.task_id === meta.task_id ? taskHint : null;
    if (!task) {
      try {
        const list = await splitTasks(meta.split, (st.splits.find((s) => s.name === meta.split) || {}).num_tasks || 1000);
        task = list.find((t) => t.task_id === meta.task_id) || null;
      } catch {}
    }
    if (!isCurrent()) return;
    for (const b of btns) b.disabled = false;
    if (!task) {
      if (errEl) errEl.innerHTML = `<div class="error">The episode started on ${escapeHtml(meta.task_id)}, but that task is not in the Task API.</div>`;
      return;
    }
    if (errEl) errEl.innerHTML = "";
    const sameTask = st.task && st.task.task_id === task.task_id;
    Object.assign(st, { meta, task, done: false, endReason: null, grade: null, rubric: null, lastCheck: null, lastSummary: "", log: [], unseenCalls: 0, checksUsed: 0, toolCalls: 0, busy: false, phase: "episode" });
    st.tools = tools.length ? tools : (meta.tools || []).map((n) => (typeof n === "string" ? { name: n, description: "", input_schema: {} } : n));
    if (!(keepDraft && sameTask && st.draft.size)) {
      st.draft = initialDraft(task);
      st.undo = [];
      st.shipSel = null;
    }
    st.selected = task.task_id;
    st.split = meta.split || st.split;
    history.replaceState(null, "", `${location.pathname}${location.search}#/play?split=${encodeURIComponent(st.split)}&task=${encodeURIComponent(task.task_id)}`);
    closePicker();
    onConnStatus(session.status);
    await buildEpisode();
  }

  function initialDraft(task) {
    const m = new Map();
    for (const s of task.ships) {
      const cranes = hasCranes(task) ? s.std_cranes : null;
      if (s.planned_hour != null && s.planned_section != null) m.set(s.id, { berth_hour: s.planned_hour, section: s.planned_section, cranes });
      else m.set(s.id, { berth_hour: s.arrival, section: task.first_section, cranes });
    }
    return m;
  }

  const draftList = () => [...st.draft.entries()].sort((a, b) => a[0] - b[0]).map(([ship, v]) => ({ ship, berth_hour: v.berth_hour, section: v.section, ...(v.cranes != null ? { cranes: v.cranes } : {}) }));
  const sameEntry = (a, b) => !!(a && b && a.berth_hour === b.berth_hour && a.section === b.section && (a.cranes ?? null) === (b.cranes ?? null));

  /** Local rule problems of the draft, plus what the last check_plan said about ships that have not moved since. */
  function displayEval() {
    const ev = evaluatePlan(st.task, draftList());
    for (const row of ev.rows) row.local = [...row.conflicts];
    const lc = st.lastCheck;
    if (lc) {
      for (const row of ev.rows) {
        const was = lc.entries.get(row.id);
        const cur = st.draft.get(row.id);
        if (!sameEntry(was, cur)) continue;
        for (const p of lc.problems.get(row.id) || []) if (!row.conflicts.length) row.conflicts.push(`check_plan: ${p}`);
      }
    }
    return ev;
  }

  /* ---------- episode view ---------- */

  async function buildEpisode() {
    destroyStage();
    const t = st.task;
    const c0 = clockAt(t, 0);
    $("#pa-task").textContent = t.task_id;
    $("#pa-meta").textContent = `quay ${t.quay} · week ${t.week} · ${t.difficulty} · ${t.ships.length} ships`;
    $("#pa-meta").title = `${t.terminal} · quay ${t.quay} · week ${t.week} (from ${c0.day} ${c0.date}) · ${t.difficulty} · ${t.ships.length} ships`;
    $("#pa-side").hidden = false;
    $("#grade").hidden = true;
    $("#pa-main").innerHTML = `
      ${sceneMarkup()}
      <section class="pa-chart">
        <div class="pa-chart-bar">
          <h2>Plan</h2>
          <button type="button" class="btn sm" data-act="undo" title="Undo the last change (Ctrl+Z)">Undo</button>
          <button type="button" class="btn sm" data-act="published" title="Every ship back on its published slot; unscheduled and diverted calls at their arrival on the first section">Reset to published</button>
          <span class="small draft-st" id="draft-status"></span>
          <span class="grow"></span>
          <span class="legend">${legendHtml()}</span>
        </div>
        <div class="pa-chart-box"><div class="chart-host"></div></div>
      </section>`;
    st.H = horizonOf(t, [publishedPlan(t), draftList()]) + 24;
    renderToolButtons();
    renderToolsPopover();
    renderSituation();
    renderShips();
    renderLog();
    setTab(st.tab || "ships");
    renderHeader();
    const mod = await sceneMod();
    if (!isCurrent() || st.task !== t) return;
    let pendingPreview = 0;
    stage = createStage(app.querySelector(".pa-main"), {
      task: t,
      horizon: st.H,
      sceneMod: mod,
      editable: true,
      chartFill: true,
      onCranes: (id, delta) => {
        const cur = st.draft.get(id);
        if (cur) setCranes(id, cur.cranes + delta);
      },
      onMove: (id, hour, section, final) => {
        if (final) {
          setDraft(id, hour, section);
          return;
        }
        const tmp = new Map(st.draft);
        tmp.set(id, { berth_hour: hour, section });
        if (pendingPreview) cancelAnimationFrame(pendingPreview);
        pendingPreview = requestAnimationFrame(() => {
          pendingPreview = 0;
          if (stage) stage.preview(evaluatePlan(t, [...tmp.entries()].map(([ship, v]) => ({ ship, ...v }))));
        });
      },
      onSelect: (id) => {
        st.shipSel = id;
        markTable();
        renderShipDetail();
      },
      onHover: (id) => hoverTable(id),
    });
    stage.tl.innerHTML = `<span class="ovbox status" id="scene-status">Draft</span>`;
    const box = app.querySelector(".pa-chart-box");
    const fit = () => {
      if (!stage) return;
      const wide = window.matchMedia("(min-width: 901px)").matches;
      stage.setChartMaxHeight(wide ? box.clientHeight - 2 : null);
    };
    boxRO = new ResizeObserver(() => requestAnimationFrame(fit));
    boxRO.observe(box);
    fit();
    refresh();
    if (st.shipSel != null) stage.select(st.shipSel, { from: "restore" });
  }

  function pushUndo() {
    st.undo.push(new Map([...st.draft].map(([k, v]) => [k, { ...v }])));
    if (st.undo.length > 100) st.undo.shift();
  }
  function undo() {
    if (!st.undo.length || st.done) return;
    st.draft = st.undo.pop();
    refresh();
  }
  function setCranes(id, value) {
    const s = st.task.ships.find((x) => x.id === id);
    const cur = st.draft.get(id);
    if (!s || !cur || st.done || !hasCranes(st.task)) return;
    const c = Math.max(s.min_cranes, Math.min(s.max_cranes, Math.round(value)));
    if (c === cur.cranes) return refresh();
    pushUndo();
    st.draft.set(id, { ...cur, cranes: c });
    refresh();
  }

  function setDraft(id, hour, section) {
    if (st.done) {
      refresh();
      return;
    }
    const cur = st.draft.get(id);
    if (cur && cur.berth_hour === hour && cur.section === section) {
      refresh();
      return;
    }
    pushUndo();
    st.draft.set(id, { ...cur, berth_hour: hour, section });
    refresh();
  }

  function refresh() {
    if (st.phase !== "episode") return;
    const ev = displayEval();
    st.ev = ev;
    st.H = Math.max(st.H, horizonOf(st.task, [draftList()]) + 12);
    if (stage) stage.setEvaluation(ev, st.H);
    const local = evaluatePlan(st.task, draftList());
    const stale = st.lastCheck && st.lastCheck.key !== JSON.stringify(draftList());
    const ds = $("#draft-status");
    if (ds) {
      ds.innerHTML = `${draftProblemsSummary(local)}${stale ? ' <span class="muted">· changed since the last check</span>' : ""}`;
      ds.title = local.issues.map((i) => i.problem).join("\n");
    }
    const ss = $("#scene-status");
    if (ss) ss.innerHTML = `Draft · ${local.issues.length ? `<span class="bad-t">${local.issues.length} problem${local.issues.length > 1 ? "s" : ""}</span>` : "no rule problems"}`;
    const ub = app.querySelector('[data-act="undo"]');
    if (ub) ub.disabled = !st.undo.length || st.done;
    const pb = app.querySelector('[data-act="published"]');
    if (pb) pb.disabled = st.done;
    updateShips(ev);
    renderShipDetail();
    syncArgs();
    renderHeader();
  }

  function setSummary(html) {
    st.lastSummary = html;
    renderHeader();
  }

  function renderHeader() {
    const el = $("#pa-state");
    if (!el) return;
    if (st.phase !== "episode") {
      el.innerHTML = "";
      return;
    }
    const m = st.meta || {};
    const maxC = m.max_checks;
    const maxT = m.max_tool_calls;
    const status = st.done ? (st.grade ? '<i class="dot ok"></i>submitted' : `<i class="dot bad"></i>ended${st.endReason ? ` · ${escapeHtml(st.endReason.replace(/_/g, " "))}` : ""}`) : '<i class="dot warn"></i>open';
    el.innerHTML = `<span class="pa-s1" title="Episode ${escapeHtml(m.episode_id || "")}">${status}</span><span title="check_plan calls left">checks left <b>${maxC != null ? Math.max(0, maxC - st.checksUsed) : "–"}</b>${maxC != null ? `/${maxC}` : ""}</span><span title="Tool calls used">calls <b>${st.toolCalls}</b>${maxT != null ? `/${maxT}` : ""}</span>${st.lastSummary ? `<span class="pa-sum">${st.lastSummary}</span>` : ""}`;
  }

  /* ---------- tool buttons + popover ---------- */

  function renderToolButtons() {
    $("#pa-toolbtns").innerHTML = st.tools
      .map((tool) => {
        const desc = String(tool.description || "").replace(/\s+/g, " ").trim();
        const label = TOOL_LABEL[tool.name] || tool.name;
        return `<button type="button" class="btn sm${tool.name === "check_plan" ? " primary" : ""}${tool.name === "get_situation" ? " mono" : ""}" data-call="${escapeHtml(tool.name)}" title="${escapeHtml(`${tool.name}: ${desc}${isPlanTool(tool) ? " Sends your current draft." : ""}`)}">${escapeHtml(label)}</button>`;
      })
      .join("");
    $("#pa-tools-b").hidden = !st.tools.length;
    syncToolButtons();
  }

  function onToolButton(name) {
    closePopover();
    if (name === "submit_plan") return showConfirm(true);
    callTool(name);
  }

  function showConfirm(on) {
    $("#pa-confirm").hidden = !on;
    $("#pa-toolbtns").hidden = on;
  }

  function argsText() {
    return `{"plan": [\n${draftList().map((p) => `  {"ship": ${p.ship}, "berth_hour": ${p.berth_hour}, "section": ${p.section}${p.cranes != null ? `, "cranes": ${p.cranes}` : ""}}`).join(",\n")}\n]}`;
  }

  function renderToolsPopover() {
    const el = $("#pa-tools");
    el.innerHTML = `<div class="pop-head"><b>Tools</b><span class="muted small">as returned by list_tools</span></div>${st.tools
      .map((tool) => {
        const plan = isPlanTool(tool);
        return `<div class="tool" data-tool="${escapeHtml(tool.name)}">
          <div class="tool-h"><span class="fn">${escapeHtml(tool.name)}</span></div>
          <div class="tool-d small">${escapeHtml(String(tool.description || "").trim())}</div>
          <div class="tool-p muted small">${escapeHtml(schemaSummary(tool.input_schema))}${plan ? " · sends your current draft" : ""}</div>
          <details class="tool-a"><summary>Arguments</summary><textarea spellcheck="false" rows="${plan ? 9 : 2}" data-args="${escapeHtml(tool.name)}" aria-label="Arguments for ${escapeHtml(tool.name)}">${plan ? escapeHtml(argsText()) : "{}"}</textarea><div class="args-msg muted small"></div></details>
        </div>`;
      })
      .join("")}`;
    if (!el._bound) {
      el._bound = true;
      el.addEventListener("input", onArgsInput);
      el.addEventListener("focusout", (e) => {
        const ta = e.target.closest("textarea[data-args]");
        if (ta && ta.dataset.dirty && ta.closest(".tool").querySelector(".args-msg").textContent.startsWith("Applied")) {
          delete ta.dataset.dirty;
          syncArgs();
        }
      });
    }
  }

  function onArgsInput(e) {
    const ta = e.target.closest("textarea[data-args]");
    if (!ta) return;
    ta.dataset.dirty = "1";
    const row = ta.closest(".tool");
    const tool = st.tools.find((x) => x.name === row.dataset.tool);
    const msg = row.querySelector(".args-msg");
    let obj;
    try {
      obj = JSON.parse(ta.value);
    } catch (err) {
      msg.innerHTML = `<span class="bad-t">Not valid JSON: ${escapeHtml(err.message)}. The button sends the draft until this parses.</span>`;
      ta.dataset.bad = "1";
      return;
    }
    delete ta.dataset.bad;
    if (!isPlanTool(tool)) {
      msg.textContent = "Will be sent as written.";
      return;
    }
    const plan = parsePlanArgs(obj);
    const ok = Array.isArray(plan) && plan.every((p) => p && Number.isInteger(p.ship) && Number.isInteger(p.berth_hour) && Number.isInteger(p.section) && (p.cranes == null || Number.isInteger(p.cranes)) && st.draft.has(p.ship));
    if (ok) {
      clearTimeout(ta._t);
      ta._t = setTimeout(() => {
        pushUndo();
        for (const p of plan) st.draft.set(p.ship, { berth_hour: p.berth_hour, section: p.section, cranes: p.cranes != null ? p.cranes : st.draft.get(p.ship).cranes });
        msg.textContent = plan.length < st.draft.size ? "Applied to the draft. Ships left out stay in the draft but are not sent." : "Applied to the draft.";
        refresh();
      }, 350);
    } else msg.textContent = "Valid JSON, not a plan of known ships: the draft is unchanged and this text is sent as written.";
  }

  function syncArgs() {
    for (const ta of app.querySelectorAll("textarea[data-args]")) {
      const tool = st.tools.find((x) => x.name === ta.dataset.args);
      if (!isPlanTool(tool) || document.activeElement === ta || ta.dataset.dirty) continue;
      ta.value = argsText();
    }
  }

  function togglePopover() {
    const el = $("#pa-tools");
    el.hidden = !el.hidden;
    $("#pa-tools-b").setAttribute("aria-expanded", String(!el.hidden));
    if (!el.hidden) syncArgs();
  }
  function closePopover() {
    const el = $("#pa-tools");
    if (el && !el.hidden) {
      el.hidden = true;
      $("#pa-tools-b").setAttribute("aria-expanded", "false");
    }
  }

  function syncToolButtons() {
    for (const b of app.querySelectorAll("[data-call]")) b.disabled = st.done || st.busy || st.phase !== "episode";
    const cb = app.querySelector('[data-call="check_plan"]');
    const maxC = st.meta && st.meta.max_checks;
    if (cb && maxC != null && Math.max(0, maxC - st.checksUsed) === 0) cb.disabled = true;
  }

  /** Arguments for a call: the draft, or the Arguments text when the user edited it into something valid. */
  function argsFor(tool) {
    const ta = app.querySelector(`textarea[data-args="${CSS.escape(tool.name)}"]`);
    if (ta && ta.dataset.dirty && !ta.dataset.bad) {
      try {
        return { args: JSON.parse(ta.value), ta };
      } catch {}
    }
    return { args: isPlanTool(tool) ? { plan: draftList() } : {}, ta: null };
  }

  async function callTool(name) {
    if (st.busy || st.done) return;
    const tool = st.tools.find((x) => x.name === name) || { name, input_schema: {} };
    const { args, ta } = argsFor(tool);
    st.busy = true;
    syncToolButtons();
    const entry = { n: st.log.length + 1, tool: name, args, at: Date.now(), pending: true };
    st.log.push(entry);
    renderLog();
    setSummary(`<span class="muted">${escapeHtml(name)} · waiting for the env…</span>`);
    try {
      const r = unwrap(await session.callTool(name, args));
      entry.pending = false;
      if (r.error) entry.error = r.error;
      else {
        const res = toolResult(r.obs);
        entry.result = res;
        entry.reward = r.reward;
        entry.done = r.done;
        if (r.meta && r.meta.grade) {
          st.grade = r.meta.grade;
          st.rubric = r.meta.rubric || null;
        }
        if (r.meta && r.meta.end_reason) st.endReason = r.meta.end_reason;
        if (r.done) {
          st.done = true;
          if (!st.endReason && st.grade) st.endReason = "submitted";
        }
        if (name === "check_plan" && res.json && res.json.feasible != null) {
          const problems = new Map();
          for (const v of res.json.violations || []) {
            const id = Number(v.ship);
            if (!problems.has(id)) problems.set(id, []);
            problems.get(id).push(v.problem);
          }
          const sent = parsePlanArgs(args);
          const entries = new Map();
          if (Array.isArray(sent)) for (const p of sent) if (p && Number.isInteger(p.ship)) entries.set(p.ship, { berth_hour: p.berth_hour, section: p.section, cranes: hasCranes(st.task) ? (p.cranes ?? st.task.ships[p.ship]?.std_cranes ?? null) : null });
          st.lastCheck = { key: JSON.stringify(Array.isArray(sent) ? sent : []), entries, problems, ships: new Map((res.json.ships || []).map((s) => [Number(s.ship), s])) };
        }
      }
    } catch (err) {
      entry.pending = false;
      entry.error = String(err.message || err);
    }
    try {
      const s = await session.state();
      if (s && s.type === "state" && s.data) {
        st.checksUsed = s.data.checks_used ?? st.checksUsed;
        st.toolCalls = s.data.tool_calls ?? st.toolCalls;
        if (s.data.done) st.done = true;
        if (s.data.grade && !st.grade) st.grade = s.data.grade;
      }
    } catch {}
    st.busy = false;
    if (!isCurrent() || st.phase !== "episode") return;
    if (ta && !entry.error) {
      delete ta.dataset.dirty;
      ta.closest(".tool").querySelector(".args-msg").textContent = "";
    }
    if (st.tab !== "calls") st.unseenCalls++;
    renderBadge();
    renderLog();
    syncToolButtons();
    setSummary(summaryFor(entry));
    refresh();
    if (st.done) renderGrade();
  }

  function summaryFor(e) {
    const j = e.result && e.result.json;
    const tag = `<span class="muted">${escapeHtml(e.tool === "check_plan" ? "check" : e.tool === "submit_plan" ? "submit" : e.tool)} #${e.n}</span>`;
    if (e.error || (e.result && e.result.error)) return `${tag} <span class="bad-t">${escapeHtml(e.error || e.result.error)}</span>`;
    if (j && j.submitted != null) return `${tag} <i class="dot ${j.feasible ? "ok" : "bad"}"></i>submitted · reward ${fmtNum(e.reward ?? j.reward, 3)}`;
    if (j && j.feasible != null) return j.feasible ? `${tag} <i class="dot ok"></i>feasible · cost ${j.cost}` : `${tag} <i class="dot bad"></i>infeasible · ${(j.violations || []).length} violations`;
    return `${tag} done · see Calls`;
  }

  /* ---------- situation tab ---------- */

  function renderSituation() {
    const t = st.task;
    const m = st.meta || {};
    const closures = (t.blocks || []).filter((b) => b.kind === "closed");
    const along = (t.blocks || []).filter((b) => b.kind === "alongside");
    const fmt = (h) => {
      const c = clockAt(t, h);
      return `${c.day} ${c.hm}`;
    };
    const kinds = callKinds(t);
    const extras = t.ships.filter((s) => kinds.has(s.id));
    $("#situation").innerHTML = `
      <h3 class="md-h">Notices</h3><ul class="md-l">${(t.notices || []).map((n) => `<li>${escapeHtml(n)}</li>`).join("") || '<li class="muted">None.</li>'}</ul>
      <dl class="kv sit-kv">
        <dt>Quay</dt><dd>sections ${t.first_section}–${t.last_section}, about ${Math.round(t.section_m)} m each</dd>
        <dt>Hour 0</dt><dd>${escapeHtml(clockAt(t, 0).day)} ${escapeHtml(clockAt(t, 0).date)} 00:00 UTC</dd>
        <dt>Closed</dt><dd>${closures.length ? closures.map((b) => `sections ${b.first}–${b.last}, hours ${b.start}–${b.end} <span class="muted">${escapeHtml(b.label)} · ${fmt(b.start)} – ${fmt(b.end)}</span>`).join("<br>") : '<span class="muted">none</span>'}</dd>
        <dt>Alongside</dt><dd>${along.length ? along.map((b) => `${escapeHtml(b.label)} · sections ${b.first}–${b.last} until hour ${b.end} <span class="muted">cannot move</span>`).join("<br>") : '<span class="muted">none</span>'}</dd>
        ${divertWindows(t).map((d) => `<dt>Diverted</dt><dd>quay ${escapeHtml(d.from_quay)} closed hours ${d.start}–${d.end}; ${(d.ships || []).length} calls come here</dd>`).join("")}
        ${extras.length ? `<dt>No published slot</dt><dd>${extras.map((s) => `${s.id} ${escapeHtml(s.name)} <span class="muted">${kinds.get(s.id).kind === "divert" ? `from ${escapeHtml(kinds.get(s.id).from)}` : "extra call"}</span>`).join("<br>")}</dd>` : ""}
        ${hasCranes(t) ? cranesKv(t, fmt) : ""}
        <dt>Limits</dt><dd>${m.max_checks != null ? `${m.max_checks} checks` : "–"} · ${m.max_tool_calls != null ? `${m.max_tool_calls} tool calls` : "–"}</dd>
        <dt>Episode</dt><dd>${escapeHtml(m.episode_id || "–")} · <a class="ext" href="/viewer/#/live/${encodeURIComponent(m.episode_id || "")}" target="_blank" rel="noopener">in Explorer ↗</a></dd>
      </dl>
      ${m.instructions ? `<details class="plan-d"><summary>Instructions</summary><div class="md">${mdToHtml(m.instructions)}</div></details>` : ""}
      ${m.situation ? `<details class="plan-d"><summary>The situation as the agent reads it</summary><div class="md">${mdToHtml(m.situation)}</div></details>` : ""}`;
  }

  function cranesKv(t, fmt) {
    const r = t.rules;
    const outs = (r.crane_outages || []).map((o) => `${o.cranes} out hours ${o.start}–${o.end} <span class="muted">${fmt(o.start)} – ${fmt(o.end)}</span>`);
    const winds = (r.no_moves || []).map((w) => `hours ${w.start}–${w.end}: ${w.min_length > 0 ? `ships of ${w.min_length} m or more` : "all ships"} <span class="muted">${escapeHtml(w.reason || "")}</span>`);
    return `<dt>Cranes</dt><dd>${r.crane_pool} in the pool · ${r.crane_rate || 28} moves per crane-hour${outs.length ? `<br>${outs.join("<br>")}` : ""}</dd>
      <dt>Moves</dt><dd>at most ${r.max_moves_per_hour} ships dock or leave in any hour</dd>
      ${winds.length ? `<dt>Wind</dt><dd>no docking or leaving: ${winds.join("<br>")}<br><span class="muted">cranes keep working; a ship that finishes inside a window waits alongside</span></dd>` : ""}`;
  }

  /* ---------- ships tab ---------- */

  function renderShips() {
    const t = st.task;
    const cr = hasCranes(t);
    const kinds = callKinds(t);
    const rows = t.ships
      .map((s) => {
        const k = kinds.get(s.id);
        const tag = k ? `<span class="muted"> ${k.kind === "divert" ? `from ${escapeHtml(k.from)}` : "extra"}</span>` : "";
        const badge = (s.weight || 1) > 1 ? ` <em class="bdg">×${s.weight}</em>` : "";
        const emerg = s.berth_deadline != null ? ` <em class="bdg red">by ${s.berth_deadline}</em>` : "";
        const tip = `${s.id} ${s.name}${cr ? ` · ${movesOf(t, s)} moves · ${s.min_cranes}–${s.max_cranes} cranes` : ""}${(s.weight || 1) > 1 ? ` · priority ×${s.weight}` : ""}${s.berth_deadline != null ? ` · emergency: dock by hour ${s.berth_deadline}` : ""}`;
        return `<tr data-row="${s.id}">
          <td class="c-name"><span title="${escapeHtml(tip)}">${escapeHtml(s.name)}${badge}${emerg}${tag}</span></td>
          ${cr ? "" : `<td class="num">${s.sections}</td>`}
          <td class="num">${s.arrival}</td>
          ${cr ? "" : `<td class="num">${s.handling}</td>`}
          <td class="num">${s.due}</td>
          <td class="num"><input class="num-in" type="number" inputmode="numeric" step="1" min="0" data-f="h" data-id="${s.id}" aria-label="Docking hour of ${escapeHtml(s.name)}"></td>
          <td class="num"><input class="num-in" type="number" inputmode="numeric" step="1" min="${t.first_section}" max="${t.last_section - s.sections + 1}" data-f="s" data-id="${s.id}" aria-label="First section of ${escapeHtml(s.name)}"></td>
          ${cr ? `<td class="num"><input class="num-in cr-in" type="number" inputmode="numeric" step="1" min="${s.min_cranes}" max="${s.max_cranes}" data-f="c" data-id="${s.id}" aria-label="Cranes for ${escapeHtml(s.name)} (${s.min_cranes}-${s.max_cranes})" title="${s.min_cranes}–${s.max_cranes} cranes"></td>` : ""}
          <td class="c-st"><i class="dot"></i></td>
        </tr>`;
      })
      .join("");
    const cols = cr
      ? '<col class="w-name"><col class="w-n"><col class="w-n"><col class="w-in"><col class="w-in"><col class="w-cr"><col class="w-st">'
      : '<col class="w-name"><col class="w-n"><col class="w-n"><col class="w-n"><col class="w-n"><col class="w-in"><col class="w-in"><col class="w-st">';
    const head = cr
      ? '<th>Ship</th><th class="num" title="Arrival hour">Arr</th><th class="num" title="Due departure hour">Due</th><th class="num" title="Docking hour in your draft">Dock h</th><th class="num" title="First section in your draft">Sec</th><th class="num" title="Quay cranes working the ship (its min–max in the tooltip)">Cr</th><th title="Draft problems, and what the last check said">St</th>'
      : '<th>Ship</th><th class="num" title="Sections the ship needs">Len</th><th class="num" title="Arrival hour">Arr</th><th class="num" title="Hours alongside">Hrs</th><th class="num" title="Due departure hour">Due</th><th class="num" title="Docking hour in your draft">Dock h</th><th class="num" title="First section in your draft">Sec</th><th title="Draft problems, and what the last check said">St</th>';
    $("#ships").innerHTML = `<table class="tbl edit-tbl${cr ? " cr" : ""}">
      <colgroup>${cols}</colgroup>
      <thead><tr>${head}</tr></thead>
      <tbody>${rows}</tbody></table>`;
    const tbody = $("#ships tbody");
    tbody.addEventListener("change", (e) => {
      const inp = e.target.closest("input.num-in");
      if (!inp || st.done) return;
      const id = Number(inp.dataset.id);
      const s = t.ships.find((x) => x.id === id);
      const cur = st.draft.get(id);
      let v = Math.round(Number(inp.value));
      if (!Number.isFinite(v)) v = inp.dataset.f === "h" ? cur.berth_hour : cur.section;
      if (inp.dataset.f === "c") return setCranes(id, Number.isFinite(Number(inp.value)) ? Number(inp.value) : cur.cranes);
      if (inp.dataset.f === "h") setDraft(id, Math.max(0, v), cur.section);
      else setDraft(id, cur.berth_hour, Math.max(t.first_section, Math.min(t.last_section - s.sections + 1, v)));
    });
    tbody.addEventListener("focusin", (e) => {
      const tr = e.target.closest("tr[data-row]");
      if (tr && stage && stage.selected !== Number(tr.dataset.row)) stage.select(Number(tr.dataset.row), { from: "table" });
    });
    tbody.addEventListener("click", (e) => {
      if (e.target.closest("input")) return;
      const tr = e.target.closest("tr[data-row]");
      if (!tr || !stage) return;
      const id = Number(tr.dataset.row);
      if (stage.selected === id) return stage.select(null, { from: "table" });
      // show the ship alongside in the 3D view: jump to its berthing in the draft, then focus it
      const d = st.draft.get(id);
      const s = st.task.ships.find((x) => x.id === id);
      if (d && s && (stage.time < d.berth_hour || stage.time >= d.berth_hour + s.handling)) stage.setTime(d.berth_hour + Math.min(2, s.handling / 2));
      stage.select(id, { focus: true, from: "table" });
    });
    tbody.addEventListener("mouseover", (e) => {
      const tr = e.target.closest("tr[data-row]");
      if (stage) stage.hover(tr ? Number(tr.dataset.row) : null);
    });
    tbody.addEventListener("mouseleave", () => stage && stage.hover(null));
  }

  function shipVerdict(id) {
    const row = st.ev && st.ev.byId.get(id);
    const local = row ? row.local || [] : [];
    const lc = st.lastCheck;
    const d = st.draft.get(id);
    let env = null;
    if (lc && d) {
      const was = lc.entries.get(id);
      const same = sameEntry(was, d);
      env = { same, problems: lc.problems.get(id) || [], ship: lc.ships.get(id) || null };
    }
    return { row, local, env };
  }

  function updateShips() {
    const tbody = $("#ships tbody");
    if (!tbody) return;
    for (const tr of tbody.querySelectorAll("tr[data-row]")) {
      const id = Number(tr.dataset.row);
      const d = st.draft.get(id);
      const hIn = tr.querySelector('[data-f="h"]');
      const sIn = tr.querySelector('[data-f="s"]');
      if (document.activeElement !== hIn) hIn.value = d.berth_hour;
      if (document.activeElement !== sIn) sIn.value = d.section;
      hIn.disabled = sIn.disabled = st.done;
      const cIn = tr.querySelector('[data-f="c"]');
      if (cIn) {
        if (document.activeElement !== cIn) cIn.value = d.cranes;
        cIn.disabled = st.done;
      }
      const v = shipVerdict(id);
      const envBad = v.env && v.env.same && v.env.problems.length;
      const bad = v.local.length || envBad;
      const dot = tr.querySelector(".c-st .dot");
      dot.className = `dot ${bad ? "bad" : v.env && v.env.same ? "ok" : ""}`;
      const tips = [...v.local.map((p) => `draft: ${p}`), ...(v.env ? v.env.problems.map((p) => `${v.env.same ? "" : "before your last edit, "}check_plan: ${p}`) : [])];
      dot.parentElement.title = tips.length ? tips.join("\n") : v.env && v.env.same ? "No problems; the last check agrees" : "No rule problems in the draft";
      tr.classList.toggle("has-bad", !!bad);
    }
  }

  function renderShipDetail() {
    const el = $("#ship-detail");
    if (!el) return;
    const id = typeof st.shipSel === "number" ? st.shipSel : null;
    const s = id != null && st.task.ships.find((x) => x.id === id);
    if (!s) {
      el.innerHTML = '<p class="muted small">Select a ship for its details. Drag it on the chart, use the arrow keys (+ and − for cranes), or type its docking hour and section.</p>';
      return;
    }
    const d = st.draft.get(id);
    const v = shipVerdict(id);
    const k = callKinds(st.task).get(id);
    const row = st.ev && st.ev.byId.get(id);
    const cr = hasCranes(st.task);
    const dep = row && row.placed ? row.dep : d.berth_hour + s.handling;
    const pub = s.planned_hour != null ? `h ${s.planned_hour} @ ${sectionsLabel(s.planned_section, s.planned_section + s.sections - 1)}` : k && k.kind === "divert" ? `none · diverted from ${escapeHtml(k.from)}` : "none · extra call";
    let envTxt = '<span class="muted">not checked yet</span>';
    if (v.env) {
      const sh = v.env.ship;
      const pre = v.env.same ? "" : '<span class="muted">before your last edit: </span>';
      envTxt = v.env.problems.length
        ? `${pre}<span class="${v.env.same ? "bad-t" : ""}">${v.env.problems.map(escapeHtml).join("; ")}</span>`
        : sh
          ? `${pre}${sh.delay_h ? `+${sh.delay_h} h late` : "on time"}${sh.moved ? ", moved" : ""} · cost ${sh.cost}`
          : `${pre}ok`;
    }
    const stepper = cr
      ? `<span class="stepper"><button type="button" class="btn sm" data-cr="-1" aria-label="One crane less" ${st.done || d.cranes <= s.min_cranes ? "disabled" : ""}>−</button><b>${d.cranes}</b><button type="button" class="btn sm" data-cr="1" aria-label="One crane more" ${st.done || d.cranes >= s.max_cranes ? "disabled" : ""}>+</button></span> <span class="muted">of ${s.min_cranes}–${s.max_cranes} · ${movesOf(st.task, s)} moves → ${row ? row.work : "–"} h of work</span>`
      : "";
    el.innerHTML = `<div class="sd-h"><b>${escapeHtml(s.name)}</b>${badgesHtml(s)} <span class="muted">${s.id} · ${Math.round(s.length_m)} m · ${s.sections} sections · ${escapeHtml(s.from_port || "?")} → ${escapeHtml(s.to_port || "?")}</span></div>
      <dl class="kv sd-kv">
        <dt>Published</dt><dd>${pub}${cr && s.planned_hour != null ? ` · ${s.std_cranes} cranes` : ""}</dd>
        ${cr ? `<dt>Cranes</dt><dd>${stepper}</dd>` : ""}
        <dt>Draft</dt><dd>docks h ${d.berth_hour} @ ${sectionsLabel(d.section, d.section + s.sections - 1)} · leaves h ${dep}${row && row.hold ? ` <span class="muted">(done h ${row.finish}, held by the wind)</span>` : ""} <span class="muted">· due ${s.due}</span></dd>
        ${s.berth_deadline != null ? `<dt>Emergency</dt><dd class="${d.berth_hour > s.berth_deadline ? "bad-t" : ""}">dock by hour ${s.berth_deadline}; ${s.deadline_penalty} per hour later</dd>` : ""}
        <dt>Draft rules</dt><dd>${v.local.length ? `<span class="bad-t">${v.local.map(escapeHtml).join("; ")}</span>` : '<span class="muted">ok</span>'}</dd>
        <dt>Last check</dt><dd>${envTxt}</dd>
      </dl>`;
  }

  function markTable() {
    for (const tr of app.querySelectorAll("#ships tr[data-row]")) tr.classList.toggle("sel", Number(tr.dataset.row) === st.shipSel);
    const sel = app.querySelector("#ships tr.sel");
    if (sel && st.tab === "ships") sel.scrollIntoView({ block: "nearest" });
  }
  function hoverTable(id) {
    for (const tr of app.querySelectorAll("#ships tr[data-row]")) tr.classList.toggle("hov", Number(tr.dataset.row) === id);
  }

  /* ---------- calls tab ---------- */

  function renderLog() {
    const el = $("#log");
    if (!el) return;
    for (const c of charts) c.destroy();
    charts = [];
    if (!st.log.length) {
      el.innerHTML = '<p class="muted small">No calls yet.</p>';
      return;
    }
    let prev = publishedPlan(st.task);
    el.innerHTML = "";
    for (const e of st.log) {
      const d = document.createElement("div");
      d.className = "call";
      const plan = parsePlanArgs(e.args);
      let head = `<span class="fn">${escapeHtml(e.tool)}</span>`;
      if (Array.isArray(plan)) head += ` <span class="muted">${plan.length} ships</span>`;
      else if (e.args && e.args.plan != null) head += ' <span class="bad-t">plan sent as text</span>';
      head += `<span class="step-no muted">#${e.n} · ${ago(e.at)}</span>`;
      let body = "";
      if (Array.isArray(plan)) {
        const tbl = planTable(st.task, plan, prev);
        body += `<div class="mini-chart"></div><details class="plan-d"><summary>Plan sent <span class="muted">${tbl.changed} changed since the previous call</span></summary>${tbl.html}</details>`;
        prev = plan;
      } else if (e.args && Object.keys(e.args).length) body += `<pre class="txt args">${escapeHtml(JSON.stringify(e.args, null, 1))}</pre>`;
      if (e.pending) body += '<div class="result muted small">waiting for the env…</div>';
      else if (e.error) body += `<div class="result"><div class="res-line"><i class="dot bad"></i>${escapeHtml(e.error)}</div></div>`;
      else if (e.result) {
        const res = e.result;
        if (res.error) body += `<div class="result"><div class="res-line"><i class="dot bad"></i>${escapeHtml(res.error)}</div></div>`;
        else if (res.json && (res.json.feasible != null || res.json.submitted != null)) body += `<div class="result">${checkSummary(st.task, res.json)}${res.json.checks_left != null ? `<div class="muted small">${res.json.checks_left} checks left</div>` : ""}</div>`;
        else if (res.text != null) {
          const first = res.text.split("\n").find((l) => l.trim()) || "";
          body += `<details class="plan-d"><summary>Result <span class="muted">${escapeHtml(first.replace(/^#+\s*/, "").slice(0, 70))}</span></summary><div class="md">${mdToHtml(res.text)}</div></details>`;
        } else body += `<pre class="txt result">${escapeHtml(JSON.stringify(res.json, null, 1))}</pre>`;
        body += `<div class="muted small">reward ${e.reward == null ? "–" : fmtNum(e.reward, 3)} · done ${e.done ? "yes" : "no"}</div>`;
      }
      if (Array.isArray(plan) && !st.done) body += `<div class="call-act"><button type="button" class="btn sm" data-act="load" data-n="${e.n}" title="Replace the draft with the plan sent in this call">Load into draft</button></div>`;
      d.innerHTML = `<div class="call-h">${head}</div>${body}`;
      el.appendChild(d);
      const mc = d.querySelector(".mini-chart");
      if (mc) {
        const ch = new BerthChart(mc, { mini: true, onPick: () => {} });
        ch.setData(st.task, evaluatePlan(st.task, plan), st.H);
        charts.push(ch);
      }
    }
    const panel = app.querySelector('.pa-panel[data-panel="calls"]');
    if (panel && !panel.hidden) panel.scrollTop = panel.scrollHeight;
  }

  /* ---------- grade ---------- */

  function renderGrade() {
    const sec = $("#grade");
    sec.hidden = false;
    const g = st.grade;
    const actions = `<div class="grade-act"><button type="button" class="btn sm primary" data-act="new">New episode</button><button type="button" class="btn sm" data-act="again">Same task again</button><a class="ext" href="/viewer/#/live/${encodeURIComponent((st.meta || {}).episode_id || "")}" target="_blank" rel="noopener">Explorer ↗</a></div>`;
    if (!g) {
      sec.innerHTML = `<div class="sec-head"><h2>Episode ended</h2></div><p class="small">No plan was graded${st.endReason ? ` (${escapeHtml(st.endReason.replace(/_/g, " "))})` : ""}. Reward 0.</p>${actions}`;
      return;
    }
    const r = st.rubric || {};
    const cell = (k, v, tip) => `<dt${tip ? ` title="${escapeHtml(tip)}"` : ""}>${escapeHtml(k)}</dt><dd>${v}</dd>`;
    const v = g.violations || [];
    sec.innerHTML = `<div class="sec-head"><h2>Grade</h2><span class="muted small">from the env's rubric</span></div>
      <dl class="kv grade-kv">
        ${cell("Reward", `<b>${fmtNum(g.reward, 3)}</b>`)}
        ${cell("Feasible", g.feasible ? '<i class="dot ok"></i>yes' : '<i class="dot bad"></i>no')}
        <dt title="delay cost + 5 per moved ship">Cost</dt><dd class="wide">${g.cost == null ? '<span class="muted">– (infeasible plans have no cost)</span>' : `${g.cost} <span class="muted">= delay ${g.delay_cost} + ${g.moves} moves × 5</span>`}</dd>
        ${cell("Naive / optimum", `${fmtNum(g.naive_cost)} / ${fmtNum(g.optimal_cost)}`, "Cost of the naive re-plan and of the optimal plan")}
        ${cell("Quality", fmtNum(g.quality, 3), "Where your cost falls between the naive re-plan (0) and the optimum (1)")}
        ${cell("Clean", fmtNum(g.clean_fraction, 3), "Share of ships without a violation")}
        ${Object.entries(r).map(([k, val]) => cell(k.replace(/_/g, " "), fmtNum(val, 3), "Rubric component")).join("")}
      </dl>
      ${v.length ? `<details class="plan-d"><summary>${v.length} violation${v.length > 1 ? "s" : ""}</summary><ul class="viol">${v.map((x) => `<li><span class="sid">${escapeHtml(x.ship)} ${escapeHtml((st.task.ships.find((s) => s.id === Number(x.ship)) || {}).name || "")}</span> ${escapeHtml(x.problem)}</li>`).join("")}</ul></details>` : ""}
      ${actions}`;
  }

  onConnStatus(session.status);
  bindShell();
  // #/play?split=<s>&index=<i>&start=1 starts that task's episode straight away, without the picker (OpenEnv's /web
  // tab and embeds use it); the one task comes from the Task API, so a 1,050-task split is never listed first.
  const autoIndex = params.get("index");
  if (params.get("start") === "1" && st.split && autoIndex !== null && autoIndex !== "") {
    $("#pa-main").innerHTML = '<div class="pa-empty muted">Starting an episode…</div>';
    taskApi("task", { split: st.split, index: Number(autoIndex) })
      .then((d) => {
        const t = d.task || d;
        if (!isCurrent() || st.phase === "episode") return;
        st.selected = t.task_id;
        return start({ task_id: t.task_id }, t);
      })
      .catch(() => isCurrent() && openPicker());
    return;
  }
  // #/play?task=<id>&start=1 (older links): pick it from the split list, then start
  openPicker().then(() => {
    if (params.get("start") !== "1" || !isCurrent() || st.phase === "episode") return;
    const t = pickerTasks.find((x) => x.task_id === st.selected);
    if (t) start({ task_id: t.task_id }, t);
  });
}
