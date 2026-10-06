// Rollout transcript: messages as a conversation, tool calls paired with their outputs,
// plans as a mini berth chart + compact table, check results as violations + cost.
import { BerthChart } from "./chart.js";
import { escapeHtml, evaluatePlan, fmtNum, hasCranes, parsePlanArgs, publishedPlan } from "./model.js";

const PLAN_TOOLS = new Set(["check_plan", "submit_plan"]);

function parseJson(s) {
  if (s == null) return null;
  if (typeof s === "object") return s;
  try {
    return JSON.parse(s);
  } catch {
    return null;
  }
}

const planKey = (a) => (Array.isArray(a) ? JSON.stringify(a.map((p) => (p && typeof p === "object" ? [p.ship, p.berth_hour, p.section, p.cranes ?? null] : p))) : `raw:${String(a)}`);
const samePlan = (a, b) => planKey(a) === planKey(b);

/** Map each plan tool call (in message order) to its index in rollout.steps. */
export function mapCallsToSteps(rollout) {
  const calls = [];
  for (const m of rollout.messages || []) for (const tc of m.tool_calls || []) if (PLAN_TOOLS.has(tc.name)) calls.push(tc);
  const steps = rollout.steps || [];
  const map = new Map();
  let j = 0;
  for (const tc of calls) {
    const plan = parsePlanArgs(tc.arguments);
    let k = -1;
    for (let i = j; i < steps.length; i++) {
      if (steps[i].tool === tc.name && (plan == null || !Array.isArray(steps[i].plan) || samePlan(plan, steps[i].plan))) {
        k = i;
        break;
      }
    }
    if (k < 0 && j < steps.length && steps[j].tool === tc.name) k = j;
    if (k >= 0) {
      map.set(tc.id, k);
      j = k + 1;
    }
  }
  return map;
}

function firstLine(s, n = 90) {
  const line = String(s || "").trim().split("\n")[0] || "";
  return line.length > n ? `${line.slice(0, n - 1)}…` : line;
}

function shipName(task, id) {
  const s = task.ships.find((x) => x.id === Number(id));
  return s ? s.name : `ship ${id}`;
}

export function checkSummary(task, res) {
  if (!res) return "";
  if (res.grade) return gradeSummary(res.grade);
  if (res.submitted != null && res.grade == null) {
    const bits = [`Submitted${res.reward != null ? ` · reward ${fmtNum(res.reward, 2)}` : ""}`];
    if (res.feasible != null) bits.push(res.feasible ? "feasible" : "infeasible");
    if (res.cost != null) bits.push(`cost ${res.cost}${res.delay_cost != null ? ` <span class="muted">= delay ${res.delay_cost} + ${res.moves} moves × 5</span>` : ""}`);
    return `<div class="res-line">${res.feasible != null ? `<i class="dot ${res.feasible ? "ok" : "bad"}"></i>` : ""}${bits.join(" · ")}</div>`;
  }
  if (res.feasible == null) {
    if (res.error || res.parse_problems) return `<div class="res-line"><i class="dot bad"></i>${escapeHtml(res.error || [].concat(res.parse_problems).join("; "))}</div>`;
    return "";
  }
  const parts = [];
  if (res.feasible) {
    parts.push(`<div class="res-line"><i class="dot ok"></i>Feasible · cost ${res.cost} <span class="muted">= delay ${res.delay_cost} + ${res.moves} moves × 5</span></div>`);
  } else {
    const v = res.violations || [];
    parts.push(`<div class="res-line"><i class="dot bad"></i>Infeasible · ${v.length} violation${v.length === 1 ? "" : "s"}${res.cost != null ? ` <span class="muted">· cost ${res.cost}</span>` : ""}</div>`);
    if (v.length) parts.push(`<ul class="viol">${v.map((x) => `<li><span class="sid">${escapeHtml(x.ship)} ${escapeHtml(shipName(task, x.ship))}</span> ${escapeHtml(x.problem)}</li>`).join("")}</ul>`);
  }
  const late = (res.ships || []).filter((s) => s.cost > 0);
  if (late.length) {
    parts.push(`<div class="res-ships"><span class="lbl-c">Cost by ship</span>${late.map((s) => `<span title="departs hour ${s.departure}${s.moved ? ", moved" : ""}">${escapeHtml(shipName(task, s.ship))} ${[s.delay_h ? `+${s.delay_h} h` : "", s.moved ? "moved" : ""].filter(Boolean).join(", ")} = <b>${s.cost}</b></span>`).join("")}</div>`);
  }
  return parts.join("");
}

function gradeSummary(g) {
  const v = g.violations || [];
  const pp = g.parse_problems || [];
  let html = `<div class="res-line"><i class="dot ${g.feasible ? "ok" : "bad"}"></i>Submitted · reward ${fmtNum(g.reward, 2)} · ${g.feasible ? "feasible" : "infeasible"} · cost ${fmtNum(g.cost)} <span class="muted">(naive ${fmtNum(g.naive_cost)}, optimum ${fmtNum(g.optimal_cost)})</span></div>`;
  if (v.length) html += `<ul class="viol">${v.map((x) => `<li>${escapeHtml(typeof x === "string" ? x : `${x.ship} ${x.problem}`)}</li>`).join("")}</ul>`;
  if (pp.length) html += `<ul class="viol">${pp.map((x) => `<li>${escapeHtml(typeof x === "string" ? x : JSON.stringify(x))}</li>`).join("")}</ul>`;
  return html;
}

/** A plan the model wrote into its message instead of a tool call: {before, plan, after} or null. */
function planInText(text) {
  const t = String(text || "").replace(/```(?:json)?/g, "");
  const i = t.search(/[[{]/);
  const j = Math.max(t.lastIndexOf("]"), t.lastIndexOf("}"));
  if (i < 0 || j <= i) return null;
  let obj;
  try {
    obj = JSON.parse(t.slice(i, j + 1));
  } catch {
    return null;
  }
  const plan = parsePlanArgs(obj);
  if (!Array.isArray(plan) || !plan.length || !plan.every((p) => p && typeof p === "object" && "ship" in p)) return null;
  return { before: t.slice(0, i).trim(), plan, after: t.slice(j + 1).trim() };
}

export function planTable(task, plan, prev) {
  const cr = hasCranes(task);
  const prevBy = new Map((prev || []).map((p) => [Number(p.ship), p]));
  let changed = 0;
  const rows = [...plan]
    .sort((a, b) => Number(a.ship) - Number(b.ship))
    .map((p) => {
      const s = task.ships.find((x) => x.id === Number(p.ship));
      const q = prevBy.get(Number(p.ship));
      const c = cr ? (p.cranes ?? (s ? s.std_cranes : null)) : null;
      let ch = "";
      if (prev) {
        if (!q) ch = "new";
        else {
          const bits = [];
          if (q.berth_hour !== p.berth_hour) bits.push(`h ${q.berth_hour}→${p.berth_hour}`);
          if (q.section !== p.section) bits.push(`sec ${q.section}→${p.section}`);
          const qc = cr ? (q.cranes ?? (s ? s.std_cranes : null)) : null;
          if (cr && qc !== c) bits.push(`cr ${qc}→${c}`);
          ch = bits.join(", ");
        }
      }
      if (ch) changed++;
      const last = s ? Number(p.section) + s.sections - 1 : p.section;
      return `<tr class="${ch ? "chg" : ""}"><td class="num">${escapeHtml(p.ship)}</td><td>${escapeHtml(s ? s.name : "?")}</td><td class="num">${escapeHtml(p.berth_hour)}</td><td class="num">${escapeHtml(p.section)}–${escapeHtml(last)}</td>${cr ? `<td class="num">${escapeHtml(c ?? "–")}</td>` : ""}<td class="muted">${escapeHtml(ch)}</td></tr>`;
    })
    .join("");
  return { html: `<table class="tbl plan-tbl"><thead><tr><th class="num">#</th><th>Ship</th><th class="num">Dock h</th><th class="num">Sections</th>${cr ? '<th class="num">Cranes</th>' : ""}<th>${prev ? "Change" : ""}</th></tr></thead><tbody>${rows}</tbody></table>`, changed };
}

/**
 * Render into `root`. Returns {select(stepIndex)} to mark the selected step.
 * onStep(stepIndex) is called when the user clicks a tool call.
 */
export function renderTranscript(root, rollout, task, { onStep, horizon }) {
  root.innerHTML = "";
  const outputs = new Map();
  for (const m of rollout.messages || []) if (m.role === "tool" && m.tool_call_id) outputs.set(m.tool_call_id, m);
  const stepOf = mapCallsToSteps(rollout);
  const steps = rollout.steps || [];
  const callEls = new Map();
  const charts = [];
  let turn = 0;
  let prevPlan = publishedPlan(task);
  const used = new Set();

  for (const m of rollout.messages || []) {
    if (m.role === "tool") {
      if (m.tool_call_id && [...stepOf.keys()].includes(m.tool_call_id)) continue;
      if (m.tool_call_id && used.has(m.tool_call_id)) continue;
      const d = document.createElement("div");
      d.className = "msg tool";
      d.innerHTML = `<div class="who">Tool · ${escapeHtml(m.name || "")}</div><pre class="txt">${escapeHtml(m.content)}</pre>`;
      root.appendChild(d);
      continue;
    }
    if (m.role === "system" || m.role === "user") {
      const d = document.createElement("details");
      d.className = `msg ${m.role}`;
      d.innerHTML = `<summary><span class="who">${m.role === "system" ? "System" : "User"}</span> <span class="muted">${escapeHtml(firstLine(m.content))}</span></summary><pre class="txt">${escapeHtml(m.content)}</pre>`;
      root.appendChild(d);
      continue;
    }
    turn++;
    const d = document.createElement("div");
    d.className = "msg assistant";
    let html = `<div class="who">Assistant <span class="muted">· turn ${turn}</span></div>`;
    if (m.reasoning) html += `<details class="reason"><summary>Reasoning <span class="muted">${escapeHtml(firstLine(m.reasoning, 70))}</span></summary><pre class="txt">${escapeHtml(m.reasoning)}</pre></details>`;
    const inText = m.content ? planInText(m.content) : null;
    if (inText) {
      if (inText.before) html += `<div class="txt say">${escapeHtml(inText.before)}</div>`;
      const tbl = planTable(task, inText.plan, null);
      html += `<div class="call inline-plan"><div class="call-h"><span class="muted" title="The model wrote this plan in its reply instead of calling check_plan or submit_plan; it was not checked.">plan written in the message, not a tool call · ${inText.plan.length} ships</span></div><div class="mini-chart"></div><details class="plan-d"><summary>Plan table</summary>${tbl.html}</details></div>`;
      if (inText.after) html += `<div class="txt say">${escapeHtml(inText.after)}</div>`;
    } else if (m.content && String(m.content).trim()) html += `<div class="txt say">${escapeHtml(m.content)}</div>`;
    d.innerHTML = html;
    if (inText) {
      const ch = new BerthChart(d.querySelector(".inline-plan .mini-chart"), { mini: true, onPick: () => {} });
      ch.setData(task, evaluatePlan(task, inText.plan), horizon);
      charts.push(ch);
    }
    for (const tc of m.tool_calls || []) {
      used.add(tc.id);
      const out = outputs.get(tc.id);
      const k = stepOf.has(tc.id) ? stepOf.get(tc.id) : null;
      const step = k != null ? steps[k] : null;
      const parsed = parsePlanArgs(tc.arguments);
      const plan = Array.isArray(parsed) ? parsed : step && Array.isArray(step.plan) ? step.plan : null;
      const unreadable = !plan && (PLAN_TOOLS.has(tc.name));
      const call = document.createElement("div");
      call.className = `call${k != null ? " pick" : ""}`;
      if (k != null) call.dataset.step = k;
      const res = step && step.result ? step.result : parseJson(out && out.content);
      let head = `<span class="fn">${escapeHtml(tc.name)}</span>`;
      let body = "";
      if (plan) {
        const ev = evaluatePlan(task, plan);
        const tbl = planTable(task, plan, prevPlan);
        head += ` <span class="muted">${plan.length} ships${prevPlan ? ` · ${tbl.changed} changed` : ""}</span>`;
        if (k != null) head += `<span class="step-no muted">step ${k + 1}</span>`;
        body += `<div class="mini-chart" data-k="${k ?? ""}"></div>`;
        body += `<details class="plan-d"><summary>Plan table</summary>${tbl.html}</details>`;
        call._ev = ev;
        prevPlan = plan;
      } else if (unreadable) {
        head += ` <span class="bad-t">unreadable plan</span>`;
        if (k != null) head += `<span class="step-no muted">step ${k + 1}</span>`;
        const raw = typeof tc.arguments === "string" ? tc.arguments : JSON.stringify(tc.arguments);
        body += `<details class="plan-d"><summary>Arguments as sent</summary><pre class="txt args">${escapeHtml(raw)}</pre></details>`;
      } else {
        const args = parseJson(tc.arguments);
        if (!(args && typeof args === "object" && !Object.keys(args).length)) body += `<pre class="txt args">${escapeHtml(args ? JSON.stringify(args, null, 1) : tc.arguments)}</pre>`;
      }
      const resHtml = res ? checkSummary(task, res) : "";
      if (resHtml) body += `<div class="result">${resHtml}</div>`;
      else if (out) {
        const txt = String(out.content ?? "");
        body += txt.length > 300 || txt.includes("\n")
          ? `<details class="plan-d"><summary>Output <span class="muted">${escapeHtml(firstLine(txt, 70))}</span></summary><pre class="txt result">${escapeHtml(txt)}</pre></details>`
          : `<pre class="txt result">${escapeHtml(txt)}</pre>`;
      }
      call.innerHTML = `<div class="call-h">${head}</div>${body}`;
      d.appendChild(call);
      if (k != null) {
        callEls.set(k, call);
        call.addEventListener("click", (e) => {
          if (e.target.closest("summary, table, details[open] .plan-tbl")) return;
          onStep(k);
        });
      }
      const mc = call.querySelector(".mini-chart");
      if (mc && call._ev) {
        const ch = new BerthChart(mc, { mini: true, onPick: () => k != null && onStep(k) });
        ch.setData(task, call._ev, horizon);
        charts.push(ch);
      }
    }
    root.appendChild(d);
  }
  return {
    select(k) {
      for (const [i, c] of callEls) c.classList.toggle("sel", i === k);
    },
    scrollTo(k) {
      const c = callEls.get(k);
      if (c) c.scrollIntoView({ block: "nearest", behavior: "smooth" });
    },
    destroy() {
      for (const c of charts) c.destroy();
    },
  };
}
