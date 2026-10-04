// An environment Space, most often an OpenEnv server. Three sources, each named where it's used:
//   live       what its running server says now (/api/spaces/<id>/live: OpenAPI, /metadata, /schema, MCP tools, Task API)
//   last seen  that, kept from the last time it was seen running here (shown while it sleeps)
//   declared   what its repository's files say at one revision (/api/spaces/<id>/files: openenv.yaml, models.py,
//              server/app.py, the environment's reward code, Dockerfile, pyproject), read without waking it
// The page: an Interface table (how an agent acts, what it sees, the reward, tasks, web app, episodes), then the
// playground (awake only), its app, tools, tasks, rewards, schemas, files, connecting, `openenv validate`'s criteria,
// and the README. Status, wake and restart sit beside it.
import { $, api, esc, md, ago, sk, emptyState, toast, fmt, spinner, storage, setMeta, ownerLink } from "./util.js";
import { icon } from "./icons.js";
import { AGENTS, agentHtml, agentName, chosenAgent, copyBtn, snippet } from "./connect.js";
import { getSession } from "./session.js";
import { schemaForm, readForm, wireForm } from "./schemaform.js";
import { value } from "./render.js";
import { fileViewer } from "./files.js";
import { supportPanel } from "./support.js";
import { taskContent } from "./space-task-view.js";
import { spaceMedia } from "./space-media.js";

const enc = (s) => s.split("/").map(encodeURIComponent).join("/");
const STAGE = {
  RUNNING: ["Running", "ok"], SLEEPING: ["Asleep", "faint"], PAUSED: ["Paused", "faint"], STOPPED: ["Stopped", "faint"],
  APP_STARTING: ["Starting", "live"], BUILDING: ["Building", "live"], RUNNING_BUILDING: ["Rebuilding", "live"], RUNNING_APP_STARTING: ["Restarting", "live"],
  RUNTIME_ERROR: ["Runtime error", "err"], BUILD_ERROR: ["Build error", "err"], CONFIG_ERROR: ["Config error", "err"], NO_APP_FILE: ["No app file", "err"],
};
const stageOf = (x) => STAGE[x] || [String(x || "unknown").toLowerCase().replace(/_/g, " "), "faint"];
const STARTING = new Set(["APP_STARTING", "BUILDING", "RUNNING_BUILDING", "RUNNING_APP_STARTING"]);
const KIND = { harbor: "OpenEnv × Harbor", openenv: "OpenEnv", "nemo-gym": "NeMo Gym server", gymnasium: "Gymnasium HTTP server", ors: "ORS server", mcp: "MCP server", api: "HTTP API" };
const shortSplit = (n) => (String(n).includes("/") ? String(n).split("/").filter(Boolean).pop() : String(n));
const IMAGE = /((^|_)(image|img|frame|screenshot|pixels|png|jpe?g|picture|photo|render)s?(_?(b64|data|url|bytes))?$|base64|(^|_)b64$)/i;
const HIDDEN_OBS = new Set(["reward", "done", "metadata"]);

let page = null;   // this page's state; unmount ends its playground session, its polling and its file viewer
export function unmount() {
  if (!page) return;
  clearTimeout(page.poll);
  page.observer?.disconnect();
  page.viewer?.destroy();
  document.removeEventListener("keydown", page.onKey);
  if (page.session) {
    fetch(`/api/spaces/${enc(page.spec)}/play`, { method: "POST", keepalive: true, headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ op: "end", session: page.session }) }).catch(() => {});
  }
  page = null;
}

// ── where a fact comes from ──────────────────────────────────────────────────
const D = (me) => (me.F && !me.F.error ? me.F.declared || {} : {});
const seenOf = (me) => (me.L && !me.L.running ? me.L.last_seen || null : null);
const probeOf = (me) => (me.L?.running ? me.L : seenOf(me));           // the live answer, or the last one kept
const fileLink = (path, label) => `<a href="#" class="sp-fl" data-open-file="${esc(path)}" title="Open ${esc(path)} in Files">${esc(label || path.split("/").pop())}</a>`;
const SRC = {
  live: () => `<span class="sp-src">live</span>`,
  seen: (s) => `<span class="sp-src" title="${esc(new Date(s.checked_at * 1000).toLocaleString())}">last seen ${esc(ago(s.checked_at))}</span>`,
  file: (me, path) => `<span class="sp-src">${path ? fileLink(path) : "its files"} @ <code>${esc(me.F.short)}</code></span>`,
  session: () => `<span class="sp-src">this session</span>`,
};
const probeSrc = (me) => (me.L?.running ? SRC.live() : SRC.seen(seenOf(me)));
const codes = (xs, n = 8) => `${xs.slice(0, n).map((x) => `<code>${esc(x)}</code>`).join(" ")}${xs.length > n ? ` <span class="faint">+${fmt.format(xs.length - n)}</span>` : ""}`;
const line = (html, src = "") => `<div class="if-v"><span>${html}</span>${src}</div>`;
const NONE = (text) => `<span class="faint">${text}</span>`;

// the models.py class that is the Action (or Observation, State): the one server/app.py names, else the role's
// class with the most fields
function classOf(d, role) {
  const cls = d.models?.classes || [];
  const named = d.app?.[role];
  return cls.find((c) => c.name === named && (c.fields || []).length) || cls.filter((c) => c.role === role && !c.kind)
    .sort((a, b) => (b.fields || []).length - (a.fields || []).length)[0] || null;
}
const propsOf = (schema) => Object.entries(schema?.properties || {});
const isImage = (k, p = {}) => IMAGE.test(k) || /image/i.test(p.format || "") || p.contentEncoding === "base64" || /image/i.test(p.contentMediaType || "");
const fieldList = (rows) => rows.map(([k, img]) => `<code>${esc(k)}</code>${img ? ` <span class="faint xs">image</span>` : ""}`).join(" ");
const declaredTools = (d) => (d.environment?.tools?.length ? [d.environment.tools.map((t) => t.name), d.environment.path]
  : d.manifest?.validation?.capabilities?.declared_tools?.length ? [d.manifest.validation.capabilities.declared_tools, d.manifest.path]
  : d.manifest?.tools?.length ? [d.manifest.tools.map((t) => t.name), d.manifest.path] : [null, null]);
const declaredTasks = (d) => d.manifest?.tasks || null;
const taskCounts = (d) => d.manifest?.validation?.capabilities?.declared_task_count || {};
const reward = (d) => d.manifest?.validation?.reward || null;
const caps = (d) => d.manifest?.validation?.capabilities || {};

// ── the interface table ──────────────────────────────────────────────────────
function interfaceRows(me) {
  const P = probeOf(me), d = D(me), F = me.F && !me.F.error ? me.F : null;
  const rows = [];

  // how the agent acts
  const acts = [];
  if (P?.mcp?.length) acts.push(line(`MCP tools: ${codes(P.mcp.map((t) => t.name))}`, probeSrc(me)));
  else if (P?.schema?.action?.properties) acts.push(line(`Action fields: ${fieldList(propsOf(P.schema.action).filter(([k]) => k !== "metadata").map(([k]) => [k]))}`, probeSrc(me)));
  else if (me.L?.running && me.L.api?.length) acts.push(line(`HTTP routes: ${codes(me.L.api.map((r) => `${r.method} ${r.path}`), 6)}`, SRC.live()));
  if (!acts.length && F) {
    const [tools, where] = declaredTools(d);
    const A = classOf(d, "action");
    if (tools?.length) acts.push(line(`MCP tools: ${codes(tools)}`, SRC.file(me, where)));
    else if (d.app?.mcp) acts.push(line(`MCP tools (<code>${esc(d.app.action)}</code>)`, SRC.file(me, d.app.path)));
    else if (A) acts.push(line(`<code>${esc(A.name)}</code>${(A.fields || []).length ? `: ${fieldList(A.fields.map((f) => [f.name]))}` : ""}`, SRC.file(me, d.models.path)));
    else if (d.manifest?.legacy?.action) acts.push(line(`<code>${esc(d.manifest.legacy.action)}</code>`, SRC.file(me, d.manifest.path)));
  }
  rows.push(["Agent acts by", acts]);

  // what it sees
  const sees = [];
  if (P?.schema?.observation?.properties) {
    const obs = propsOf(P.schema.observation).filter(([k]) => !HIDDEN_OBS.has(k));
    if (obs.length) sees.push(line(`${fieldList(obs.map(([k, p]) => [k, isImage(k, p)]))}`, probeSrc(me)));
  }
  if (!sees.length && F) {
    const O = classOf(d, "observation");
    if (O && (O.fields || []).length) sees.push(line(`<code>${esc(O.name)}</code>: ${fieldList(O.fields.filter((f) => !HIDDEN_OBS.has(f.name)).map((f) => [f.name, f.image]))}`, SRC.file(me, d.models.path)));
    else if (d.app?.mcp) sees.push(line("tool results (MCP)", SRC.file(me, d.app.path)));
    else if (d.manifest?.legacy?.observation) sees.push(line(`<code>${esc(d.manifest.legacy.observation)}</code>`, SRC.file(me, d.manifest.path)));
  }
  rows.push(["Agent sees", sees]);

  // the reward
  const rw = [];
  if (me.rewards.length) {
    const lo = Math.min(...me.rewards), hi = Math.max(...me.rewards);
    rw.push(line(`observed ${lo === hi ? num(lo) : `${num(lo)} to ${num(hi)}`} over ${fmt.format(me.rewards.length)} step${me.rewards.length === 1 ? "" : "s"}`, SRC.session()));
  }
  if (P?.rewards) rw.push(line("every step returns one", probeSrc(me)));
  if (P?.graders?.length) rw.push(line(`graded on ${codes(P.graders, 4)}`, probeSrc(me)));
  if (F) {
    const r = reward(d), c = caps(d), mp = d.manifest?.path;
    if (r?.range) rw.push(line(`range ${esc(`[${r.range.map(num).join(", ")}]`)}`, SRC.file(me, mp)));
    if (c.verifier?.kind) rw.push(line(`verifier: <code>${esc(c.verifier.kind)}</code>${c.verifier.entry ? ` ${fileLinkIf(me, c.verifier.entry, d)}` : ""}`, SRC.file(me, mp)));
    if (c.llm_judged) rw.push(line(`LLM-judged${d.manifest.validation.judge?.model ? `: <code>${esc(d.manifest.validation.judge.model)}</code>` : ""}`, SRC.file(me, mp)));
    const blocks = Object.keys(d.manifest?.blocks || {}).filter((k) => /reward|rubric|grader|scoring/.test(k));
    if (blocks.length) rw.push(line(`${codes(blocks, 4)} in openenv.yaml`, SRC.file(me, mp)));
    const e = d.environment;
    if (e && !e.error) {
      if (e.llm_judge) rw.push(line(`an LLM judges${e.llm_modules?.length ? ` (${codes(e.llm_modules, 3)})` : ""}`, SRC.file(me, e.path)));
      if (e.rubric) rw.push(line("a Rubric scores it", SRC.file(me, e.path)));
      if (!e.llm_judge && !e.rubric && (e.functions?.length || e.reward_files?.length)) rw.push(line(`in code: ${e.functions?.length ? codes(e.functions, 3) : (e.reward_files || []).map((p) => fileLink(p)).join(", ")}`, SRC.file(me, e.path)));
    }
  }
  rows.push(["Reward", rw]);

  // tasks
  const tk = [];
  const splits = P?.task_api?.splits || [];
  if (splits.length) tk.push(line(splits.map((x) => `<code>${esc(shortSplit(x.name))}</code>${x.num_tasks != null ? ` ${fmt.format(x.num_tasks)}` : ""}`).join(" · "), probeSrc(me)));
  else if (P?.task_api) tk.push(line("a Task API, with no splits listed", probeSrc(me)));
  if (F && !tk.length) {
    const counts = Object.entries(taskCounts(d)), list = declaredTasks(d);
    if (counts.length) tk.push(line(counts.map(([k, n]) => `<code>${esc(k)}</code> ${fmt.format(n)}`).join(" · "), SRC.file(me, d.manifest.path)));
    else if (list?.length) tk.push(line(`${fmt.format(list.length)} declared: ${codes(list.map((t) => t.id || t.name || "?"), 6)}`, SRC.file(me, d.manifest.path)));
  }
  rows.push(["Tasks", tk]);

  // its web app
  const web = [];
  if (me.L?.running) web.push(line(me.L.ui ? `<a class="u" href="${esc(me.L.host + me.L.ui)}" target="_blank" rel="noopener"><code>${esc(me.L.ui)}</code></a>` : NONE("none"), SRC.live()));
  else if (seenOf(me)) web.push(line(seenOf(me).ui ? `<code>${esc(seenOf(me).ui)}</code>` : NONE("none"), SRC.seen(seenOf(me))));
  if (F && !me.L?.running) {
    const df = d.dockerfile, card = d.card || {};
    if (d.app?.gradio_builder) web.push(line(`a Gradio tab of its own${d.app.custom_tab_name ? `, <i>${esc(d.app.custom_tab_name)}</i>` : ""}`, SRC.file(me, d.app.path)));
    else if ((d.app?.routes || []).some((r) => /^GET \/web\/?$/.test(r))) web.push(line("a page of its own at <code>/web</code>", SRC.file(me, d.app.path)));
    if (df && df.web_interface != null) web.push(line(df.web_interface ? `OpenEnv's web interface at <code>${esc(card.base_path || "/web")}</code>` : "web interface off", SRC.file(me, df.path)));
  }
  rows.push(["Web app", web]);

  // episodes
  const ep = [];
  const mode = P?.mode;
  if (P && (P.openenv || P.framework === "openenv" || P.framework === "harbor")) {
    ep.push(line(mode === "production" ? "MCP only: production mode, no reset, step or state" : "reset, step, state; one WebSocket per session", probeSrc(me)));
  } else if (me.L?.running && me.L.framework) ep.push(line(esc(KIND[me.L.framework] || me.L.framework), SRC.live()));
  if (F && d.app && !d.app.error) {
    const bits = [d.app.max_concurrent_envs != null ? `up to ${esc(String(d.app.max_concurrent_envs))} at once` : null,
      d.app.mode ? `mode <code>${esc(String(d.app.mode))}</code>` : null, !P && d.app.factory ? `<code>${esc(d.app.factory)}</code>` : null].filter(Boolean);
    if (bits.length) ep.push(line(bits.join(", "), SRC.file(me, d.app.path)));
  }
  rows.push(["Episodes", ep]);
  return rows;
}
const num = (x) => String(+Number(x).toFixed(4));
const fileLinkIf = (me, path, d) => (me.F.files.some((f) => f.path === path) ? fileLink(path) : (d.manifest?.path && me.F.files.some((f) => f.path === `${me.F.root}/${path}`) ? fileLink(`${me.F.root}/${path}`, path) : `<code>${esc(path)}</code>`));

function interfaceTable(me) {
  if (!me.L && !me.F) return ifaceSkeleton();
  const rows = interfaceRows(me);
  const pending = !me.F || !me.L;
  return `${supportPanel(me.L?.support)}<dl class="kv sp-if">${rows.map(([k, vals]) => `<dt>${esc(k)}</dt><dd>${vals.length ? vals.join("") : pending ? sk.line(40) : NONE(emptyFor(me, k))}</dd>`).join("")}</dl>
    ${me.F?.error ? `<p class="fine sp-note">${icon("alert", 12)} Couldn't read its files: ${esc(me.F.error)}</p>` : ""}`;
}
const emptyFor = (me, k) => (k === "Tasks" ? "none declared" : k === "Web app" ? "none declared" : !me.L?.running && !seenOf(me) ? "not declared; wake it to ask its server" : "none");
const ifaceSkeleton = () => `<dl class="kv sp-if">${["Agent acts by", "Agent sees", "Reward", "Tasks", "Web app", "Episodes"].map((k, i) => `<dt>${k}</dt><dd>${sk.line([62, 74, 48, 36, 30, 52][i])}</dd>`).join("")}</dl>`;

// ── sections ─────────────────────────────────────────────────────────────────
// an OpenEnv × Harbor server: what it can run, and the two ways to run its tasks
function harborSection(L) {
  const c = L.harbor_caps;
  if (!c) return `<p class="muted sm">It runs Harbor tasks as rollouts (<code>run_rollout</code>), but didn't say what it can run.</p>`;
  const ok = c.sandboxes.filter((x) => x.available), validated = c.harnesses.filter((h) => h.status === "validated");
  const hub = c.datasets.filter((d) => d.hub);
  return `<dl class="kv sp-caps">
      <dt>Model engine</dt><dd>${c.engine.url ? `its own${c.engine.model ? `: <code>${esc(c.engine.model)}</code>` : ""}` : `none of its own: each <code>run_rollout</code> call names one (<code>llm_url</code>, <code>model</code>, <code>api_key</code>)`}</dd>
      <dt>Sandboxes</dt><dd>${ok.length ? ok.map((x) => `<code>${esc(x.name)}</code>`).join(" ") : `<span class="faint">none available</span>`}</dd>
      <dt>Agents</dt><dd>${validated.map((h) => `<code>${esc(h.name)}</code>`).join(" ")}${c.harnesses.length > validated.length ? ` <span class="faint">and ${c.harnesses.length - validated.length} more, not validated</span>` : ""}</dd>
      <dt>Serves</dt><dd>${c.datasets.map((d) => `<code>${esc(d.hub || d.name)}</code>${d.num_tasks != null ? ` <span class="faint">${fmt.format(d.num_tasks)} tasks</span>` : ""}`).join("<br>")}</dd></dl>
    <div class="sp-ways">
      <div class="sp-way"><b>${icon("user", 14)}On your account</b><p>${hub.length ? "Open the dataset in the explorer: every task runs on an HF Sandbox billed to you, with the agent and model you pick." : "Its datasets are local to the Space, so they can't be opened in the explorer."}</p>
        ${hub.map((d) => `<a class="btn sm" href="/d/${enc(d.hub)}">${icon("database", 13)}${esc(d.hub)}</a>`).join("")}</div>
      <div class="sp-way"><b>${icon("box", 14)}On this Space</b><p>${L.ui ? "Its own app runs rollouts with its sandboxes; sign in there." : ""} ${c.engine.url ? "Or call <code>run_rollout</code> from the playground, on its operator's engine." : "Calling <code>run_rollout</code> needs a model engine, which this explorer never fills in with your token."}</p>
        ${L.ui ? `<button class="btn sm" type="button" data-jump-to="app">${icon("window", 13)}Its app</button>` : ""}${c.engine.url ? `<button class="link sm" type="button" data-try="run_rollout">Try run_rollout</button>` : ""}</div>
    </div>`;
}

function appSection(me) {
  const { s, L } = me;
  if (!L) return sk.box(160);
  if (L.running && L.ui) {
    const src = L.host + L.ui;
    if (!me.appOpen) {
      return `<div class="sp-app-off"><span class="dot" style="--dc:var(--ok)"></span><code>${esc(src.replace("https://", ""))}</code><span class="grow"></span>
        <button class="btn sm" type="button" data-app="load">${icon("window", 13)}Open it here</button>
        <a class="btn sm ghost" href="${esc(src)}" target="_blank" rel="noopener">${icon("external", 13)}New tab</a></div>`;
    }
    return `<div class="sp-app"><div class="sp-app-bar"><span class="dot" style="--dc:var(--ok)"></span><code>${esc(src.replace("https://", ""))}</code><span class="grow"></span>
        <button class="icon-btn sm" type="button" data-app="reload" title="Reload">${icon("refresh", 15)}</button>
        <button class="icon-btn sm" type="button" data-app="full" title="Full view (Esc to close)">${icon("maximize", 15)}</button>
        <a class="icon-btn sm" href="${esc(src)}" target="_blank" rel="noopener" title="Open in a new tab">${icon("external", 15)}</a></div>
      <div class="sp-app-frame"><div class="sp-app-load">${spinner()}<span>Loading its app…</span></div>
      <iframe class="space-app" src="${esc(src)}" title="${esc(s.id)} app" allow="clipboard-write; fullscreen"
        sandbox="allow-scripts allow-same-origin allow-forms allow-popups allow-downloads" referrerpolicy="no-referrer"></iframe></div></div>`;
  }
  if (L.running) return `<p class="muted sm">This server has no web page of its own${L.base_path ? ` at <code>${esc(L.base_path)}</code>` : ""}, <code>/web</code> or <code>/</code>. The playground calls its API.</p>`;
  const thumb = `<img class="sp-shot" alt="The Hub's picture of ${esc(s.id)}" loading="lazy" onerror="this.remove()" src="https://cdn-thumbnails.huggingface.co/social-thumbnails/spaces/${enc(s.id)}.png">`;
  return `${thumb}<p class="muted sm sp-after">${STARTING.has(L.stage) ? "Starting: its app opens here when it's up." : "Its app opens here when the Space is running."}</p>`;
}

// what the playground can do here: call its MCP tools, take actions (reset/step), or call its own HTTP routes
const modesOf = (L) => [...(L.mcp?.length ? ["tools"] : []), ...(L.step_api && !L.mcp?.length ? ["action"] : []), ...(L.api?.length ? ["routes"] : [])];
const MODE = { tools: "Its tools", action: "Actions", routes: "Its routes" };

function playSection(me) {
  const L = me.L;
  if (!L) return sk.box(160);
  if (L.running) return playground(L);
  if (STARTING.has(L.stage)) return `<p class="muted sm sp-wait-line">${spinner("xs")}Starting. The playground opens when it's running.</p>`;
  if (L.stage === "SLEEPING") {
    return `<div class="sp-asleep"><p class="muted sm">Asleep. Wake it to play an episode on it; it takes a minute or two.</p>
      <button class="btn primary" type="button" data-act="wake">${icon("power", 14)}Wake it</button><div class="sp-msg" data-msg="wake"></div></div>`;
  }
  return `<p class="muted sm">It isn't running (${esc(stageOf(L.stage)[0].toLowerCase())}), so there is nothing to play. Its status, beside, says what can be done.</p>`;
}

function playground(L) {
  const modes = modesOf(L);
  if (!modes.length) return `<p class="muted sm">This Space has no API to play with: no <code>/reset</code> and <code>/step</code>, no MCP tools, no routes in an <code>/openapi.json</code>.</p>`;
  const tools = L.mcp || [];
  const splits = L.task_api?.splits || [];
  const resetOpts = `<details class="pg-opts"><summary class="disclose">${icon("chevronRight", 14, "chev")}Reset options</summary>
      <div class="pg-opts-b">${splits.length ? `<div class="pg-row"><label class="field"><span>Task split</span><select class="input" id="pg-split"><option value="">its choice</option>${splits.map((x) =>
        `<option value="${esc(x.name)}">${esc(shortSplit(x.name))}${x.num_tasks != null ? ` · ${fmt.format(x.num_tasks)}` : ""}</option>`).join("")}</select></label>
        <label class="field"><span>Task index</span><input class="input mono" id="pg-index" type="number" min="0" placeholder="random"></label></div>` : ""}
        <label class="field"><span>Other options <em>JSON, as this server takes them, e.g. {"seed": 1}</em></span><textarea class="input mono" id="pg-params" rows="2" spellcheck="false" placeholder="{}"></textarea></label></div></details>`;
  const pane = {
    tools: () => `<label class="field"><span>Tool</span><select class="input" id="pg-tool">${tools.map((t) => `<option value="${esc(t.name)}">${esc(t.name)}</option>`).join("")}</select></label>
      <p class="pg-desc" id="pg-desc"></p><div id="pg-form"></div><div id="pg-warn"></div>
      <button class="btn primary block" type="button" id="pg-call">${icon("send", 14)}Call</button>`,
    action: () => `<div class="sec-label">Action</div><div id="pg-form">${schemaForm(L.schema?.action || {}, { prefix: "pga" })}</div>
      <button class="btn primary block" type="button" id="pg-step">${icon("play", 14)}Step</button>`,
    routes: () => `<label class="field"><span>Route</span><select class="input mono" id="pg-route">${L.api.map((r, i) => `<option value="${i}">${esc(r.method)} ${esc(r.path)}</option>`).join("")}</select></label>
      <p class="pg-desc" id="pg-rdesc"></p><div id="pg-rparams"></div><div id="pg-query"></div><div id="pg-rform"></div>
      <button class="btn primary block" type="button" id="pg-send">${icon("send", 14)}Send</button>`,
  };
  return `<div class="pg">
    <div class="pg-ctl">
      <div class="pg-sess" id="pg-sess"></div>
      ${L.openenv && L.step_api ? `${resetOpts}<button class="btn block" type="button" id="pg-reset">${icon("refresh", 14)}New episode</button>` : ""}
      ${modes.length > 1 ? `<span class="seg pg-modes" role="tablist">${modes.map((m, i) => `<button type="button" role="tab" data-mode="${m}" aria-pressed="${i === 0}">${MODE[m]}</button>`).join("")}</span>` : ""}
      ${modes.map((m, i) => `<div class="pg-act" data-pane="${m}" ${i ? "hidden" : ""}>${pane[m]()}</div>`).join("")}
      <p class="fine">Requests go to the Space itself. ${L.openenv ? "An OpenEnv WebSocket keeps your episode together." : L.api?.length ? "HTTP cookies are isolated per explorer session; the server controls episode state." : "Tool state depends on the server's MCP implementation."} Sessions end after 10 idle minutes. The Space's owner pays for its hardware.</p>
    </div>
    <div class="pg-log" id="pg-log" aria-live="polite"></div>
  </div>`;
}

function tasksSection(me) {
  const P = probeOf(me), d = D(me);
  if (me.L?.running && me.L.task_api) return taskBrowser(me);
  const out = [];
  if (P?.task_api && !P.task_api.splits?.length) out.push(`<div class="sp-sub"><b>Its Task API lists no splits</b>${probeSrc(me)}</div>`);
  const splits = P?.task_api?.splits || [];
  if (splits.length) {
    out.push(`<div class="sp-sub"><b>Its Task API</b>${probeSrc(me)}</div>
      <div class="tbl"><table><thead><tr><th>Split</th><th>Tasks</th><th></th></tr></thead><tbody>${splits.map((x) =>
        `<tr><td><code>${esc(x.name)}</code></td><td class="num">${x.num_tasks != null ? fmt.format(x.num_tasks) : "–"}</td><td class="faint">${x.default ? "default" : ""}</td></tr>`).join("")}</tbody></table></div>
      <p class="fine sp-after">Wake it to browse them.</p>`);
  }
  if (!me.F) return out.join("") || sk.lines(80, 60, 70);
  const counts = Object.entries(taskCounts(d)), list = declaredTasks(d);
  if (counts.length) {
    out.push(`<div class="sp-sub"><b>Declared counts</b>${SRC.file(me, d.manifest.path)}</div>
      <div class="tbl"><table><thead><tr><th>Split</th><th>Tasks</th></tr></thead><tbody>${counts.map(([k, n]) => `<tr><td><code>${esc(k)}</code></td><td class="num">${fmt.format(n)}</td></tr>`).join("")}</tbody></table></div>`);
  }
  if (list?.length) {
    const cols = ["id", "name", "difficulty", "max_steps", "description"].filter((k) => list.some((t) => t[k] != null && t[k] !== ""));
    out.push(`<div class="sp-sub"><b>Declared tasks</b>${SRC.file(me, d.manifest.path)}</div>
      <div class="sp-ft sp-grid"><table><thead><tr>${cols.map((k) => `<th>${esc(k.replace("_", " "))}</th>`).join("")}</tr></thead><tbody>${list.slice(0, 200).map((t) =>
        `<tr>${cols.map((k) => `<td class="${k === "description" ? "ft-d" : k === "id" ? "ft-n" : "ft-s"}">${t[k] == null ? "" : k === "id" ? `<code>${esc(String(t[k]))}</code>` : k === "description" ? inl(typeof t[k] === "object" ? JSON.stringify(t[k]) : String(t[k])) : esc(typeof t[k] === "object" ? JSON.stringify(t[k]) : String(t[k]))}</td>`).join("")}</tr>`).join("")}</tbody></table></div>
      ${d.manifest.withheld?.length ? `<p class="fine">${icon("shield", 12)} Left out, as they may hold the answer: ${d.manifest.withheld.map((k) => `<code>${esc(k)}</code>`).join(", ")}.</p>` : ""}`);
  }
  return out.join("") || `<p class="muted sm">${me.L?.running ? "No Task API, and no tasks declared in its files." : "No tasks declared in its files."}${!me.L?.running && !seenOf(me) ? " Its server may serve some: wake it to ask." : ""}</p>`;
}

function taskCatalog(me) {
  const api = me.L.task_api, environments = api.environments || [api];
  const selected = environments.find((e) => e.env === me.taskPage.env) || environments[0];
  me.taskPage.env = selected.env;
  return selected;
}

function taskBrowser(me) {
  const { splits } = taskCatalog(me), envs = me.L.task_api.environments || [me.L.task_api];
  if (!splits.some((s) => s.name === me.taskPage.split)) me.taskPage.split = (splits.find((s) => s.default) || splits[0])?.name || "";
  return `<div class="tk">
    ${envs.length > 1 ? `<label class="field"><span>Environment</span><select class="input" id="tk-env">${envs.map((e) => `<option value="${esc(e.env)}" ${e.env === me.taskPage.env ? "selected" : ""}>${esc(e.env)}</option>`).join("")}</select></label>` : ""}
    <p class="fine">Choose a split, then a task to inspect its inputs. Counts come from this Space's Task API.</p>
    <div class="tk-bar"><span class="seg" id="tk-split">${splits.map((x) => `<button type="button" data-split="${esc(x.name)}" aria-pressed="${x.name === me.taskPage.split}" title="${esc(x.name)}">${esc(shortSplit(x.name))}${x.num_tasks != null ? `<span>${fmt.format(x.num_tasks)}</span>` : ""}</button>`).join("")}</span>
      <span class="grow"></span><span class="tk-pager"><button class="icon-btn sm" type="button" id="tk-prev" aria-label="Previous page">${icon("chevronRight", 15, "flip")}</button><span id="tk-range"></span><button class="icon-btn sm" type="button" id="tk-next" aria-label="Next page">${icon("chevronRight", 15)}</button></span></div>
    <div id="tk-table">${sk.lines(90, 90, 90, 90)}</div>
    <div class="tk-detail" id="tk-detail" hidden></div>
  </div>`;
}

function toolsSection(me) {
  const P = probeOf(me), d = D(me);
  if (P?.mcp?.length) {
    const own = new Set(P.mcp.map((t) => t.name));
    const extra = (P.bridge_tools || []).filter((t) => !own.has(t.name));
    return `${me.L?.running ? "" : `<div class="sp-sub"><b>${fmt.format(P.mcp.length)} tools</b>${probeSrc(me)}</div>`}${toolsRef(P.mcp, !!me.L?.running)}
      ${extra.length ? `<div class="sec-label">Also available through the explorer's bridge</div>${toolsRef(extra, false)}` : ""}`;
  }
  if (!me.F && !P) return sk.lines(70, 50, 60);
  const e = d.environment;
  if (e?.tools?.length) {
    return `<div class="sp-sub"><b>Registered in its code</b>${SRC.file(me, e.path)}</div>
      ${fieldTable(e.tools.map((t) => ({ name: t.name, typeHtml: t.params.length ? `<span class="ft-enum">${t.params.map((p) => `<code>${esc(p)}</code>`).join("")}</span>` : NONE("none"), desc: t.description || "" })), { head: ["Tool", "Parameters", "What it does"] })}
      ${me.L?.running ? "" : `<p class="fine sp-after">The ones a running server offers can depend on how it's configured.</p>`}`;
  }
  const [tools, where] = declaredTools(d);
  if (tools?.length) return `<div class="sp-sub"><b>Declared</b>${SRC.file(me, where)}</div><p class="sm">${codes(tools, 40)}</p>`;
  if (P?.bridge_tools?.length) {   // a reset/step server: its tools are its episode calls, with its own action's fields
    const [org, name] = me.spec.split("/");
    return `<div class="sp-sub"><b>${fmt.format(P.bridge_tools.length)} tools, through ${P.openenv ? "reset and step" : "its HTTP API"}</b>${SRC.live()}</div>
      <p class="fine">${P.openenv ? "The bridge exposes reset, step and state with the environment's action schema." : "The bridge exposes the server's published routes. Path parameters, query parameters and body have separate inputs."} These are the available tools
        (<code>/mcp/${esc(org)}/${esc(name)}</code>, under Connect); the playground calls the same ones.</p>
      ${toolsRef(P.bridge_tools, false)}`;
  }
  if (me.L?.running || P) return `<p class="muted sm">No MCP tools of its own${(P?.openenv || P?.step_api) ? ": an agent acts through reset and step (over MCP, the bridge offers those as tools)" : ""}.</p>`;
  const A = classOf(d, "action");
  if (A) return `<p class="muted sm">No MCP tools declared: an agent acts through reset and step, with <code>${esc(A.name)}</code>. ${SRC.file(me, d.models.path)}</p>`;
  return `<p class="muted sm">None declared in its files.${me.F ? " Wake it to ask its server." : ""}</p>`;
}

function toolsRef(tools, tryable) {
  return `<div class="mcp-tools">${tools.map((t) => {
    const props = Object.entries(t.inputSchema?.properties || {});
    return `<details class="mcp-tool"><summary>${icon("chevronRight", 13, "chev")}<code>${esc(t.name)}</code>
      ${props.length ? `<span class="faint xs">(${props.map(([k]) => esc(k)).join(", ")})</span>` : ""}<span class="d">${esc((t.description || t.title || "").split("\n")[0])}</span>
      ${tryable ? `<button class="link sm" type="button" data-try="${esc(t.name)}">Try it</button>` : ""}</summary>
      <div class="mcp-body">${t.description && t.description.includes("\n") ? docHtml(t.description, 1500) : ""}
        ${props.length ? fieldTable(schemaRows(t.inputSchema, t.inputSchema.$defs || {}, { images: false }), { head: ["Parameter", "Type", "What it is"] }) : `<p class="muted sm">${t.inputSchema ? "No parameters." : "The server doesn't list its parameters."}</p>`}</div></details>`;
  }).join("")}</div>`;
}

function rewardsSection(me) {
  const P = probeOf(me), d = D(me);
  if (!me.F && !me.L) return sk.lines(70, 50, 60);
  const rows = [];
  if (me.rewards.length) rows.push(["This session", `${me.rewards.slice(-12).map(num).join(", ")}${me.rewards.length > 12 ? " …" : ""} ${SRC.session()}`]);
  if (P?.rewards) rows.push(["Each step", `its observation carries <code>reward</code> ${probeSrc(me)}`]);
  if (P?.graders?.length) rows.push(["Graded on", `${codes(P.graders)} ${probeSrc(me)}`]);
  if (me.F && !me.F.error) {
    const v = d.manifest?.validation, mp = d.manifest?.path;
    if (v?.reward) {
      const r = v.reward;
      if (r.range) rows.push(["Range", `<code>[${r.range.map(num).join(", ")}]</code> ${SRC.file(me, mp)}`]);
      const tol = ["oracle_tolerance", "floor_margin", "variance_tolerance"].filter((k) => r[k] != null);
      if (tol.length) rows.push(["Tolerances", `${tol.map((k) => `${esc(k.replace("_", " "))} <code>${num(r[k])}</code>`).join(", ")} ${SRC.file(me, mp)}`]);
    }
    const c = v?.capabilities;
    if (c?.verifier?.kind) rows.push(["Verifier", `<code>${esc(c.verifier.kind)}</code>${c.verifier.entry ? ` ${fileLinkIf(me, c.verifier.entry, d)}` : ""} ${SRC.file(me, mp)}`]);
    if (c && "llm_judged" in c) rows.push(["LLM-judged", `${c.llm_judged ? "yes" : "no"}${v.judge?.model ? `: <code>${esc(v.judge.model)}</code>${v.judge.version ? ` <span class="faint">${esc(v.judge.version)}</span>` : ""}` : ""} ${SRC.file(me, mp)}`]);
    if (c?.oracle) rows.push(["Oracle", `${esc(c.oracle.form || "")} ${c.oracle.location ? fileLinkIf(me, c.oracle.location, d) : ""} ${SRC.file(me, mp)}`]);
    const e = d.environment;
    if (e && !e.error) {
      const files = [e.path, ...(e.reward_files || [])];
      rows.push(["Its code", `${files.map((p) => fileLink(p, p)).join(", ")}${e.functions?.length ? `<div class="sp-fns">${codes(e.functions, 12)}</div>` : ""}`]);
      if (e.llm_judge) rows.push(["An LLM judges", `${e.llm_modules?.length ? codes(e.llm_modules) : "yes"} ${SRC.file(me, e.path)}`]);
      if (e.rubric) rows.push(["Rubric", `an OpenEnv Rubric ${SRC.file(me, e.path)}`]);
    }
  }
  const blocks = Object.entries(D(me).manifest?.blocks || {}).filter(([k]) => /reward|rubric|grader|scoring/.test(k));
  const yaml = blocks.length ? `<div class="sp-sub"><b>In openenv.yaml</b>${SRC.file(me, d.manifest.path)}</div>${blocks.map(([k, x]) =>
    `<div class="sp-block"><code class="sp-bk">${esc(k)}</code>${isObj(x) ? fieldsView(x, 1) : `<div class="pgv-val">${value(x, k)}</div>`}</div>`).join("")}` : "";
  if (!rows.length && !yaml) return `<p class="muted sm">${me.F ? "Nothing about its reward is declared in its files." : ""}${!me.L?.running ? " Wake it and play an episode to see what it gives." : " Play an episode to see what it gives."}</p>`;
  return `${rows.length ? `<dl class="kv sp-rw">${rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${v}</dd>`).join("")}</dl>` : ""}${yaml}`;
}

// ── field tables: /schema, models.py classes and tool parameters, drawn the same way ──
// A field's name and type never break inside a word: unions break at `|`, enum values wrap as a list, and at phone
// width each field stacks into one block.
const inl = (s) => esc(s).replace(/`([^`\n]+)`/g, "<code>$1</code>");
const DOC_SECTION = /(^|\n)[ \t]*(Attributes|Args|Arguments|Parameters|Fields|Returns|Raises|Examples?|Notes?)[ \t]*:[ \t]*(\n|$)/;
// a docstring's own prose (its Attributes/Args lists repeat what the table says), rendered as Markdown
function docHtml(text, max = 700) {
  if (!text) return "";
  let t = String(text);
  const cut = t.search(DOC_SECTION);
  if (cut >= 0) t = t.slice(0, cut);
  t = t.replace(/^[ \t]+/gm, "").replace(/\[`~?([^`\]]+)`\]/g, (_, x) => `\`${x.split(".").pop()}\``).trim();
  if (!t) return "";
  if (t.length > max) t = t.slice(0, max).replace(/\s+\S*$/, "") + " …";
  return `<div class="prose sp-doc">${md(t)}</div>`;
}
function typeHtml(t) {
  const parts = String(t || "any").split(/\s*\|\s*/).filter(Boolean);
  return `<span class="ft-ty">${parts.map((x) => `<code>${esc(x)}</code>`).join(`<span class="ft-or">|</span>`)}</span>`;
}
const enumHtml = (vals) => (vals?.length ? `<span class="ft-enum">${vals.slice(0, 40).map((x) => `<code>${esc(typeof x === "string" ? x : JSON.stringify(x))}</code>`).join("")}${vals.length > 40 ? `<span class="faint xs">+${vals.length - 40}</span>` : ""}</span>` : "");

// a JSON-schema property as {type, enumVals, nested}: $refs into $defs followed, unions joined, arrays named by item
function propType(p, defs, depth = 0) {
  if (!p || typeof p !== "object" || depth > 4) return { type: "any" };
  if (p.$ref) {
    const name = String(p.$ref).split("/").pop(), d = defs?.[name];
    if (d?.enum) return { type: name, enumVals: d.enum };
    if (d?.properties) return { type: name, nested: d };
    return { type: name };
  }
  if (p.enum) return { type: p.type || "enum", enumVals: p.enum };
  if (p.const !== undefined) return { type: JSON.stringify(p.const) };
  const alts = p.anyOf || p.oneOf;
  if (alts) {
    const parts = alts.map((x) => propType(x, defs, depth + 1));
    return { type: parts.map((x) => x.type).join(" | "), enumVals: parts.find((x) => x.enumVals)?.enumVals, nested: parts.find((x) => x.nested)?.nested };
  }
  if (p.type === "array") { const it = propType(p.items, defs, depth + 1); return { type: `list[${it.type}]`, enumVals: it.enumVals, nested: it.nested }; }
  if (p.type === "object" && p.properties) return { type: p.title || "object", nested: p };
  if (p.type === "object" && p.additionalProperties && typeof p.additionalProperties === "object") return { type: `dict[str, ${propType(p.additionalProperties, defs, depth + 1).type}]` };
  return { type: Array.isArray(p.type) ? p.type.join(" | ") : p.type || "any" };
}

function schemaRows(schema, defs, { images = true, skip = [] } = {}) {
  const req = new Set(schema?.required || []);
  return Object.entries(schema?.properties || {}).filter(([k]) => !skip.includes(k)).map(([k, p]) => {
    const t = propType(p, defs);
    return { name: k, required: req.has(k), image: images && isImage(k, p), type: t.type, enumVals: t.enumVals, desc: p.description || "",
      def: p.default !== undefined && p.default !== null ? JSON.stringify(p.default) : null,
      nested: t.nested ? schemaRows(t.nested, defs, { images }) : null };
  });
}

function fieldTable(rows, { head = ["Field", "Type", "What it is"], depth = 0 } = {}) {
  if (!rows.length) return `<p class="muted sm">No fields.</p>`;
  return `<div class="sp-ft${depth ? " sub" : ""}"><table><thead><tr>${head.map((h) => `<th>${esc(h)}</th>`).join("")}</tr></thead><tbody>${rows.map((r) =>
    `<tr><td class="ft-n"><code>${esc(r.name)}</code>${r.required ? `<span class="ft-tag">required</span>` : ""}${r.image ? `<span class="ft-tag">image</span>` : ""}</td>
      <td class="ft-t">${r.typeHtml || typeHtml(r.type)}${enumHtml(r.enumVals)}${r.limits?.length ? `<span class="ft-tag">${esc(r.limits.join(", "))}</span>` : ""}</td>
      <td class="ft-d">${r.desc ? inl(r.desc) : ""}${r.def != null ? `<span class="ft-def">default <code>${esc(String(r.def).slice(0, 80))}</code></span>` : ""}</td></tr>
    ${r.nested?.length && depth < 2 ? `<tr class="ft-sub"><td colspan="3"><details><summary>${icon("chevronRight", 13, "chev")}<code>${esc(r.name)}</code>: ${fmt.format(r.nested.length)} field${r.nested.length === 1 ? "" : "s"}</summary>${fieldTable(r.nested, { depth: depth + 1 })}</details></td></tr>` : ""}`).join("")}</tbody></table></div>`;
}

function schemaTable(title, schema) {
  if (!schema || !schema.properties) return "";
  const defs = schema.$defs || schema.definitions || {};
  return `<div class="sec-label">${esc(title)}${schema.title ? ` <code>${esc(schema.title)}</code>` : ""}</div>
    ${docHtml(schema.description)}${fieldTable(schemaRows(schema, defs, { images: title !== "Action" }))}`;
}

// models.py's `Literal["a", "b"]` as enum values
function literalVals(t) {
  const m = String(t || "").match(/Literal\[(.*?)\]/);
  if (!m) return null;
  return (m[1].match(/"[^"]*"|'[^']*'|[^,\s][^,]*/g) || []).map((x) => x.trim().replace(/^["']|["']$/g, ""));
}

function classTable(title, c) {
  const rows = (c.fields || []).map((f) => {
    const lit = literalVals(f.type);
    return { name: f.name, required: f.required, image: f.image, type: lit ? f.type.replace(/Literal\[.*?\]/, "Literal") : f.type, enumVals: lit,
      limits: f.limits, desc: f.description || "", def: f.default ?? null };
  });
  return `<div class="sec-label">${esc(title)} <code>${esc(c.name)}</code>${c.bases?.length ? ` <span class="faint xs">(${esc(c.bases.join(", "))})</span>` : ""}</div>
    ${docHtml(c.doc)}${rows.length ? fieldTable(rows) : `<p class="muted sm" style="margin-bottom:16px">No fields of its own.</p>`}`;
}

function schemasSection(me) {
  const P = probeOf(me), d = D(me);
  const schema = P?.schema;
  if (schema && (schema.action || schema.observation)) {
    return `<div class="sp-sub"><b>From its <code>/schema</code></b>${probeSrc(me)}</div>${schemaTable("Action", schema.action) + schemaTable("Observation", schema.observation) + schemaTable("State", schema.state)}`;
  }
  if (P?.api?.length) return `<div class="sp-sub"><b>From its OpenAPI</b>${probeSrc(me)}</div>` + P.api.map((r) =>
    `<details><summary class="disclose"><code>${esc(r.method)} ${esc(r.path)}</code></summary>${r.summary ? `<p class="fine">${esc(r.summary)}</p>` : ""}
      ${r.query_schema?.properties && Object.keys(r.query_schema.properties).length ? schemaTable("Query parameters", r.query_schema) : ""}
      ${r.schema ? (schemaTable("Request body", r.schema) || snippet(JSON.stringify(r.schema, null, 2))) : '<p class="fine">No request body schema published.</p>'}</details>`).join("");
  if (!me.F) return sk.lines(70, 50, 60);
  const m = d.models;
  if (!m) return `<p class="muted sm">No <code>models.py</code> in its files.${!me.L?.running ? " Wake it to ask its server for <code>/schema</code>." : ""}</p>`;
  if (m.error) return `<p class="muted sm">${fileLink(m.path, m.path)}: ${esc(m.error)}.</p>`;
  const main = ["action", "observation", "state"].map((r) => [r, classOf(d, r)]).filter(([, c]) => c);
  const shown = new Set(main.map(([, c]) => c.name));
  const rest = (m.classes || []).filter((c) => !shown.has(c.name));
  return `<div class="sp-sub"><b>Declared in its code</b>${SRC.file(me, m.path)}</div>
    ${main.map(([r, c]) => classTable(r[0].toUpperCase() + r.slice(1), c)).join("") || `<p class="muted sm">No Action, Observation or State classes in it.</p>`}
    ${rest.length ? `<details class="sp-more"><summary class="disclose">${icon("chevronRight", 14, "chev")}${fmt.format(rest.length)} more class${rest.length === 1 ? "" : "es"} in it</summary><div class="sp-more-b">${rest.map((c) => c.kind === "enum"
      ? `<div class="sec-label">Enum <code>${esc(c.name)}</code></div><p class="sm">${codes(c.values || [], 30)}</p>` : classTable(c.role ? c.role[0].toUpperCase() + c.role.slice(1) : "Class", c)).join("")}</div></details>` : ""}`;
}

const KEY_LABEL = { manifest: "openenv.yaml", models: "models.py", app: "server/app.py", environment: "environment", client: "client.py", pyproject: "pyproject.toml",
  dockerfile: "Dockerfile", discovery: "discovery.json", inference: "inference.py", readme: "README", lock: "uv.lock" };
function filesSection(me) {
  if (!me.F) return `<div class="sp-keys">${sk.line(60, 14)}</div>${sk.box(320)}`;
  if (me.F.error) return `<p class="muted sm">Couldn't list its files: ${esc(me.F.error)}</p>`;
  const F = me.F;
  const keys = Object.entries(F.keys || {}).filter(([k]) => KEY_LABEL[k]);
  return `${keys.length ? `<div class="sp-keys">${keys.map(([k, p]) => `<a href="#" class="sp-fl" data-open-file="${esc(p)}" title="${esc(p)}">${esc(k === "environment" ? p.split("/").pop() : KEY_LABEL[k])}</a>`).join("")}</div>` : ""}
    <div class="sp-files" id="sp-files"></div>
    <p class="fine sp-after">${fmt.format(F.total)} file${F.total === 1 ? "" : "s"} at <a class="u" href="https://huggingface.co/spaces/${enc(me.spec)}/tree/${esc(F.sha)}" target="_blank" rel="noopener"><code>${esc(F.short)}</code></a>${F.dropped ? `; ${fmt.format(F.dropped)} more left out: vendored OpenEnv, build output, caches` : ""}${F.truncated ? "; the listing stops here" : ""}.</p>`;
}

// ── connect ──────────────────────────────────────────────────────────────────
const CONNECT = [["agents", "Coding agents"], ["python", "Python"], ["docker", "Docker"], ["fork", "Fork"]];
function connectSection(me) {
  const tab = CONNECT.some(([k]) => k === me.connectTab) ? me.connectTab : "agents";
  return `<div class="seg sp-tabs" role="tablist" aria-label="How to connect">${CONNECT.map(([k, label]) => `<button type="button" role="tab" data-ctab="${k}" aria-pressed="${k === tab}">${label}</button>`).join("")}</div>
    <div id="sp-connect">${connectPane(me, tab)}</div>`;
}

function connectPane(me, tab) {
  const spec = me.spec, L = me.L, P = probeOf(me), d = D(me);
  const host = L?.host || (me.F?.subdomain ? `https://${me.F.subdomain}.hf.space` : "");
  if (tab === "agents") {
    const url = `${location.origin}/mcp/${spec}`;
    const own = (P?.mcp || []).map((t) => t.name);
    const routeTools = L?.running ? (L.api || []).map((r) => r.path.replace(/[^A-Za-z0-9]+/g, "_").replace(/^_+|_+$/g, "").toLowerCase()).map((b, i) => (L.api[i].method === "POST" || b.startsWith("get_") ? b : `get_${b}`)).filter((n) => !own.includes(n)) : [];
    const openenv = P?.openenv || P?.step_api;
    const tools = P?.bridge_tools ? P.bridge_tools.map((t) => t.name) : P ? [...own, ...(openenv && P.step_api && !own.includes("reset") ? ["reset", ...(own.length ? [] : ["step"]), "state"] : []), ...routeTools, ...(P.task_api ? ["list_splits", "list_tasks", "get_task"] : [])] : [];
    const chosen = chosenAgent();
    return `<div class="sp-url"><div class="sp-url-row"><span class="sp-url-k">MCP</span><code>${esc(url)}</code>${copyBtn(url)}</div>
      <p class="fine">Connect through the explorer's MCP bridge. ${openenv ? "An OpenEnv WebSocket keeps each agent's episode together." : "Each agent connection has its own HTTP cookies and MCP session; the server controls episode state."}${tools.length ? ` Available tools: ${tools.slice(0, 16).map((t) => `<code>${esc(t)}</code>`).join(" ")}${tools.length > 16 ? " …" : ""}.` : ""}${L && !L.running ? " A sleeping Space is woken on the first call." : ""} Images in observations come back as images.</p></div>
      <div class="seg sp-agents" role="tablist" aria-label="Agent">${AGENTS.filter(([id]) => id !== "python").map(([id, label]) => `<button type="button" role="tab" data-agent="${id}" aria-pressed="${id === chosen}">${esc(label)}</button>`).join("")}</div>
      <div id="sp-agent">${agentHtml(chosen === "python" ? "claude" : chosen, agentName(spec), url)}</div>`;
  }
  if (tab === "python") {
    if (P && !P.openenv) return `<p class="fine sp-file">Use the same MCP bridge from Python:</p>${snippet(agentHtmlText(spec))}
      ${P.api?.length ? `<p class="fine sp-file">Or inspect the HTTP schema directly. Reuse the client for subsequent calls so the server's cookies persist:</p>${snippet(`# pip install httpx\nimport httpx\n\nwith httpx.Client(base_url=${JSON.stringify(host)}, timeout=120) as client:\n    schema = client.get("/openapi.json").json()\n    print(schema["paths"])\n    # Choose a route and supply its required path, query and body fields here.`)}` : ""}`;
    const action = actionExample(me);
    const generic = `# pip install openenv-core     (no package of its own: plain dictionaries)
from openenv.core.generic_client import GenericEnvClient

with GenericEnvClient(base_url="${host || `https://<subdomain>.hf.space`}").sync() as env:
    result = env.reset()
    print(result.observation)
    result = env.step(${action})
    print(result.reward, result.done)`;
    const root = me.F && !me.F.error ? me.F.root : "";
    const pyproject = d.pyproject && !d.pyproject.error ? d.pyproject.path : null;
    const sub = pyproject && pyproject.includes("/") ? `#subdirectory=${pyproject.split("/").slice(0, -1).join("/")}` : "";
    const auto = pyproject ? `# its own client and models, installed from the Space
# pip install "git+https://huggingface.co/spaces/${spec}${sub}"
from openenv import AutoEnv

env = AutoEnv.from_env("${spec}")` : null;
    return `<p class="fine sp-file">Any OpenEnv server, from its URL${me.L && !me.L.running ? " (wake it first)" : ""}:</p>${snippet(generic)}
      ${auto ? `<p class="fine sp-file">Or with its own typed client${root ? ` (from <code>${esc(root)}/</code>)` : ""}:</p>${snippet(auto)}` : ""}
      <p class="fine sp-file">Or over MCP, through this explorer's bridge:</p>${snippet(agentHtmlText(spec))}`;
  }
  if (tab === "docker") {
    const port = d.card?.app_port || d.dockerfile?.port || 7860;
    const sub = me.F?.subdomain || (host ? host.replace("https://", "").replace(".hf.space", "") : "");
    if (!sub) return me.F ? `<p class="muted sm">The Hub gave no image name for this Space.</p>` : sk.lines(70, 50);
    const path = L?.ui || d.card?.base_path || "/";
    return `<p class="fine sp-file">The Space's own image, from the Hub's registry, on your machine:</p>
      ${snippet(`docker run -it -p 8000:${port} --platform=linux/amd64 registry.hf.space/${sub}:latest`)}
      <p class="fine sp-file">Then <code>http://localhost:8000${esc(path)}</code>, or point a client at <code>http://localhost:8000</code>. Secrets it needs (its own API keys) go in with <code>-e NAME=value</code>.</p>`;
  }
  if (!L?.openenv_verified) return `<p class="fine sp-file">OpenEnv compatibility has not been verified. To copy this Space, use its Hub repository:</p><a class="btn" href="https://huggingface.co/spaces/${enc(spec)}?duplicate=true" target="_blank" rel="noopener">${icon("external", 14)}Duplicate on Hugging Face</a>`;
  return `<p class="fine sp-file">A copy of this Space on your account, to change and run as your own (needs <code>hf auth login</code>):</p>
    ${snippet(`pip install openenv-core\nopenenv fork ${spec}`)}
    <p class="fine sp-file">Options: <code>--repo-id you/name</code>, <code>--private</code>, <code>--hardware cpu-upgrade</code>, <code>--set-env KEY=VALUE</code>, <code>--set-secret KEY=VALUE</code>.</p>`;
}
const agentHtmlText = (spec) => AGENTS.find(([id]) => id === "python")[2](agentName(spec), `${location.origin}/mcp/${spec}`).text;

function actionExample(me) {
  const P = probeOf(me), d = D(me);
  if (P?.mcp?.length) return `{"type": "call_tool", "tool_name": "${P.mcp[0].name}", "arguments": {}}`;
  const props = propsOf(P?.schema?.action).filter(([k]) => k !== "metadata").slice(0, 3);
  if (props.length) return JSON.stringify(Object.fromEntries(props.map(([k, p]) => [k, p.default ?? (p.enum ? p.enum[0] : p.type === "string" ? "" : p.type === "integer" || p.type === "number" ? 0 : p.type === "boolean" ? false : null)])));
  const [tools] = declaredTools(d);
  if (tools?.length || d.app?.mcp) return `{"type": "call_tool", "tool_name": "${tools?.[0] || "<tool>"}", "arguments": {}}`;
  const A = classOf(d, "action");
  if (A?.fields?.length) {
    const lit = (f) => { try { return JSON.parse((f.default || "").replace(/^'(.*)'$/, '"$1"').replace(/^None$/, "null").replace(/^True$/, "true").replace(/^False$/, "false")); } catch { return /int|float/.test(f.type) ? 0 : /bool/.test(f.type) ? false : ""; } };
    return JSON.stringify(Object.fromEntries(A.fields.slice(0, 3).map((f) => [f.name, lit(f)])));
  }
  return "{}";
}

// ── conformance ──────────────────────────────────────────────────────────────
function conformanceSection(me) {
  const L = me.L;
  if (!L) return sk.lines(70, 70, 70, 70, 70, 70);
  const P = probeOf(me);
  const declared = me.s.declared_openenv || me.s.openenv || me.s.framework === "openenv" || me.s.manifest || D(me).manifest;
  if (P && !P.openenv && !declared) return `<p class="sm">This server exposes ${esc(P.support?.framework?.label || KIND[P.framework] || "an HTTP API")}. Its published interface is shown above; OpenEnv-specific checks do not apply.</p>`;
  const rows = L.running ? L.conformance : seenOf(me)?.conformance;
  const where = L.running ? `From the probe ${L.checked ? esc(ago(L.checked)) : "just now"}.` : seenOf(me) ? `From when it was last seen running, ${esc(ago(seenOf(me).checked_at))}.` : "";
  if (!rows?.length) return `<p class="muted sm">${L.running ? "Not checked." : "Not seen running here yet: wake it, and its server is checked against these."}</p>${criteriaList()}`;
  const host = L.host;
  return `${P && !P.openenv ? `<p class="sm">This repository has OpenEnv-related metadata, but its server did not return the expected OpenEnv schemas in this probe. The checks below show what answered.</p>` : ""}
    <p class="fine sp-cf-note">${where} Basic endpoint checks corresponding to <code>openenv validate</code>. Passing does not verify an episode, reward correctness, or training quality.</p>
    <p class="fine sp-cf-note">The MCP endpoint check accepts JSON-RPC error responses. ${P?.mcp?.length ? `Tool discovery returned ${fmt.format(P.mcp.length)} tools; tool execution is untested by this probe.` : `No usable MCP tools were discovered.${P?.mcp_error ? ` Server reply: <code>${esc(P.mcp_error)}</code>.` : ""}`} OpenEnv environments can use typed actions without MCP tools.</p>
    <div class="sp-ft sp-cf"><table><thead><tr><th>Criterion</th><th>Result</th><th>Why</th></tr></thead><tbody>${rows.map((c) =>
      `<tr><td class="ft-d"><code>${esc(c.id)}</code><span class="ft-tag">${esc(c.label)}</span></td><td class="ft-s"><span class="sp-cfs ${esc(c.status)}"><span class="dot"></span>${esc(c.status)}</span></td><td class="ft-d">${esc(c.reason)}</td></tr>`).join("")}</tbody></table></div>
    ${host ? `<p class="fine sp-file">Run it yourself:</p>${snippet(`openenv validate --url ${host}`)}` : ""}`;
}
const criteriaList = () => `<ul class="sp-crit">${[["openapi_version_available", "GET /openapi.json returns info.version"], ["health_endpoint", "GET /health returns healthy"],
  ["metadata_endpoint", "GET /metadata returns name and description"], ["schema_endpoint", "GET /schema returns action, observation and state"],
  ["mcp_endpoint", "POST /mcp returns JSON-RPC"], ["mode_endpoint_consistency", "its routes match its mode"]].map(([k, v]) => `<li><code>${k}</code> <span class="faint">${v}</span></li>`).join("")}</ul>`;

// ── the side panels ──────────────────────────────────────────────────────────
function statusPanel(me) {
  const { s, L } = me;
  const stage = L?.stage || s.stage || "UNKNOWN";
  const [label, tone] = stageOf(stage);
  let body;
  if (!L) body = `<p class="sp-st-note">${sk.line(70)}</p>`;
  else if (stage === "RUNNING") {
    body = `<p class="sp-st-note">${L.checked ? `Checked ${esc(ago(L.checked))}` : "Asking its server what it offers…"}${s.hardware ? ` · ${esc(s.hardware)}` : ""}</p>
      <button class="btn sm block" type="button" data-act="refresh">${icon("refresh", 13)}Check again</button>`;
  } else if (stage === "SLEEPING") {
    body = `<p class="sp-st-note">Spaces on free hardware sleep when nobody uses them. Anyone can wake one; it takes a minute or two.${seenOf(me) ? ` Last seen running ${esc(ago(seenOf(me).checked_at))}.` : ""}</p>
      <button class="btn primary block" type="button" data-act="wake">${icon("power", 14)}Wake it</button>`;
  } else if (STARTING.has(stage)) {
    const secs = me.started ? Math.round((Date.now() - me.started) / 1000) : null;
    body = `<ol class="sp-steps"><li class="done"><i></i>Asked to start</li><li class="now"><i></i>${stage === "BUILDING" ? "Building its image" : "Starting its server"}</li><li><i></i>Running</li></ol>
      <p class="sp-st-note" id="sp-elapsed">${secs != null ? `${secs} s so far. ` : ""}Checking every few seconds.</p>`;
  } else {
    const user = getSession().user;
    body = `<p class="sp-st-note">${stage === "PAUSED" ? "Its owner paused it." : stage === "STOPPED" ? "It was stopped." : "It stopped with an error."} Only someone who may write to it can restart it${user ? "" : ": sign in first"}.</p>
      ${user ? `<button class="btn block" type="button" data-act="restart">${icon("refresh", 14)}Restart it</button>` : `<button class="btn block" type="button" data-signin>${icon("user", 14)}Sign in to restart</button>`}
      <a class="btn sm ghost block" href="https://huggingface.co/spaces/${enc(s.id)}?logs=container" target="_blank" rel="noopener">${icon("external", 13)}Its logs on the Hub</a>`;
  }
  return `<div class="panel-h"><h3>${icon("zap", 14)}Status</h3></div><div class="panel-b">
    <div class="sp-st ${tone}"><span class="sp-st-dot"></span><b>${esc(label)}</b>${STARTING.has(stage) ? spinner("xs") : ""}</div>${body}<div class="sp-msg" data-msg="status"></div></div>`;
}

function glance(me) {
  const { s, rank } = me;
  const rows = [
    rank ? ["Trending", `#${fmt.format(rank.n)} <span class="faint">of ${fmt.format(rank.of)} Spaces</span>`] : null,
    ["Likes", esc(s.likes)],
    s.updated ? ["Updated", esc(ago(Date.parse(s.updated) / 1000))] : null,
    s.license || D(me).card?.license ? ["License", esc(s.license || D(me).card.license)] : null,
  ].filter(Boolean);
  return `<div class="panel-h"><h3>${icon("info", 14)}At a glance</h3></div><div class="panel-b"><dl class="kv">${rows.map(([a, b]) => `<dt>${a}</dt><dd>${b}</dd>`).join("")}</dl></div>`;
}

// ── the header ───────────────────────────────────────────────────────────────
function describe(me) {
  const d = D(me), P = probeOf(me);
  // OpenEnv's default /metadata says only "<Class> environment": not a description
  const meta = P?.metadata?.description && !/^\w+ environment\.?$/i.test(P.metadata.description.trim()) ? P.metadata.description : null;
  return d.manifest?.keys?.description || d.discovery?.description || meta || me.s.brief || d.card?.short_description || me.s.heading || "";
}

function factsLine(me) {
  const { s, L } = me, d = D(me), P = probeOf(me);
  const [label, tone] = stageOf(L?.stage || s.stage);
  const kind = L?.openenv_verified ? (L.harbor ? "OpenEnv × Harbor" : "OpenEnv · API checked") : P?.framework && !["openenv", "harbor"].includes(P.framework) ? KIND[P.framework] : s.declared_openenv || s.openenv || d.manifest ? "Unverified Space" : "Docker Space";
  const mode = P?.mode || (d.dockerfile?.mode ? String(d.dockerfile.mode).toLowerCase() : d.app?.mode ? String(d.app.mode).toLowerCase() : null);
  const py = d.pyproject && !d.pyproject.error && d.pyproject.openenv ? `${d.pyproject.openenv_package || "openenv-core"} ${d.pyproject.openenv_version || ""}`.trim() : s.openenv_version ? `openenv ${s.openenv_version}` : null;
  const hw = s.hardware || me.F?.hardware;
  const bits = [
    `<span class="sp-state ${tone}" id="sp-state"><span class="dot"></span>${esc(label)}</span>`,
    `<span>${esc(kind)}</span>`, mode ? `<span>${esc(mode)}</span>` : null, py ? `<span title="${esc(d.pyproject?.openenv || "")}">${esc(py)}</span>` : null,
    hw ? `<span>${esc(hw)}</span>` : null,
    me.F && !me.F.error ? `<span>rev <a class="u" href="https://huggingface.co/spaces/${enc(me.spec)}/commit/${esc(me.F.sha)}" target="_blank" rel="noopener"><code>${esc(me.F.short)}</code></a></span>` : !me.F ? `<span>${sk.line(100)}</span>` : null,
  ].filter(Boolean);
  return bits.join("");
}

function linksLine(me) {
  const { L, spec } = me;
  return `<a class="u" href="https://huggingface.co/spaces/${enc(spec)}" target="_blank" rel="noopener">${icon("external", 13)}Space</a>
    <a class="u" href="https://huggingface.co/spaces/${enc(spec)}/tree/${esc(me.F?.sha || "main")}" target="_blank" rel="noopener">${icon("folder", 13)}Files</a>
    ${L?.running && L.ui ? `<a class="u" href="${esc(L.host + L.ui)}" target="_blank" rel="noopener">${icon("window", 13)}App</a>` : ""}
    <button class="link" type="button" data-act="copy-link">${icon("link", 13)}Copy link</button>`;
}

// ── the page ─────────────────────────────────────────────────────────────────
const SECTIONS = [
  ["interface", "Interface", "gauge"], ["play", "Playground", "play"], ["harbor", "Rollouts here", "box"], ["app", "App", "window"], ["tools", "Tools", "wrench"],
  ["tasks", "Tasks", "list"], ["rewards", "Rewards", "target"], ["schema", "Schemas", "code"], ["files", "Files", "folder"],
  ["connect", "Connect", "plug"], ["conformance", "API checks", "check"], ["readme", "README", "doc"],
];

function sectionBody(me, id) {
  switch (id) {
    case "interface": return interfaceTable(me);
    case "play": return playSection(me);
    case "harbor": return harborSection(me.L);
    case "app": return appSection(me);
    case "tools": return toolsSection(me);
    case "tasks": return tasksSection(me);
    case "rewards": return rewardsSection(me);
    case "schema": return schemasSection(me);
    case "files": return filesSection(me);
    case "connect": return connectSection(me);
    case "conformance": return conformanceSection(me);
    case "readme": return me.s.readme ? `<div class="prose">${md(me.s.readme)}</div>` : `<p class="muted sm">No README.</p>`;
    default: return "";
  }
}

function noteOf(me, id) {
  const L = me.L;
  if (id === "play" && L?.running) return "an episode on the Space";
  if (id === "tools" && probeOf(me)?.mcp?.length) return `${fmt.format(probeOf(me).mcp.length)} over MCP`;
  if (id === "tasks" && L?.running && L.task_api?.splits?.length) return "from its Task API";
  if (id === "harbor") return "OpenEnv × Harbor";
  return "";
}

const sectionsOf = (me) => SECTIONS.filter(([id]) => id !== "harbor" || (me.L?.running && me.L.harbor));

function skeleton(spec) {
  const [org, name] = spec.split("/");
  return `<div class="wrap page"><nav class="crumbs">${sk.box(12, "width:220px")}</nav>
    <header class="tp-head ds-head"><h1>${ownerLink(org, "/")}${esc(name)}</h1>${sk.line(55, 14)}<div class="facts sp-facts"><div class="sp-fi">${sk.line(40, 13)}</div></div></header>
    <div class="tp-grid"><nav class="toc sk-in sk-toc">${SECTIONS.slice(0, 8).map(() => sk.line(70, 12)).join("")}</nav>
      <div class="tp-body"><section class="panel tp-sec"><div class="panel-h"><h2>${icon("gauge", 15)}Interface</h2></div><div class="panel-b">${ifaceSkeleton()}</div></section>
        <section class="panel tp-sec"><div class="panel-h"><h2>${icon("play", 15)}Playground</h2></div><div class="panel-b">${sk.box(160)}</div></section></div>
      <aside class="tp-side split"><section class="panel first"><div class="panel-h"><h3>${icon("zap", 14)}Status</h3></div><div class="panel-b sk-in">${sk.line(40, 16)}${sk.line(90)}${sk.box(30)}</div></section></aside>
    </div></div>`;
}

export async function mount(el, { spec }) {
  el.innerHTML = skeleton(spec);
  const me = page = { spec, s: null, F: null, L: null, el, session: null, log: [], episode: null, poll: null, started: null, taskPage: { split: null, start: 0 },
    rank: null, rewards: [], appOpen: false, connectTab: storage.get("sp-ctab") || "agents", filePath: null, viewer: null, rendered: false };
  const params = new URLSearchParams(location.search), index = Number(params.get("task"));
  if (params.has("task") && Number.isSafeInteger(index) && index >= 0) {
    me.taskPage = { env: params.get("env"), split: params.get("split"), start: Math.floor(index / 20) * 20 };
    me.linkedTask = index;
  }
  me.onKey = (e) => { if (e.key === "Escape") $(".sp-app.full", el)?.classList.remove("full"); };
  document.addEventListener("keydown", me.onKey);
  el.addEventListener("click", (e) => onClick(me, e));
  el.addEventListener("change", (e) => {
    if (e.target.id === "pg-tool") showTool(me);
    if (e.target.id === "tk-env") {
      me.taskPage = { env: e.target.value, split: null, start: 0 };
      $("#sb-tasks", me.el).innerHTML = taskBrowser(me);
      loadTasks(me);
    }
  });
  // all three at once: the card (and README), what its files declare, what its server says (or said)
  const card = api(`/api/spaces/${enc(spec)}`);
  loadFiles(me);
  loadLive(me, false);
  api(`/api/search/rank?scope=ready&key=${encodeURIComponent(`space:${spec}`)}`).then((rk) => {   // rank among recently checked environment Spaces
    if (page === me) { me.rank = rk; const g = $("#sp-glance", el); if (g) g.innerHTML = glance(me); }
  }).catch(() => {});   // not in the catalog yet: no rank shown
  try { me.s = await card; } catch (e) {
    if (page !== me) return;
    el.innerHTML = `<div class="wrap page">${emptyState("alert", "Couldn't open this Space", esc(e.message), `<a class="btn" href="/?k=space">${icon("arrowLeft", 15)}All Spaces</a>`)}</div>`;
    page = null;
    return;
  }
  if (page === me) render(me);
}

function render(me) {
  const { s, el, spec } = me;
  if (!s) return;
  const [org, name] = spec.split("/");
  const sections = sectionsOf(me);
  const kind = me.L?.openenv_verified ? "OpenEnv Space" : s.declared_openenv || s.openenv || s.manifest ? "Unverified Space" : s.framework === "ors" ? "ORS Space" : "environment Space";   // as the server-rendered title says
  setMeta({ title: `${s.heading || name} · ${kind}`, description: `${spec}: an RL environment Space on Hugging Face. ${describe(me)} See it live: its app, a playground with rewards, its tasks, and MCP for coding agents.` });
  me.viewer?.destroy();
  me.viewer = null;
  el.innerHTML = `<div class="wrap page${me.rendered ? "" : " fade-in"}">
    <nav class="crumbs" aria-label="Breadcrumb"><a href="/">Environments</a>${icon("chevronRight", 13)}<a href="/?k=space">Spaces</a>${icon("chevronRight", 13)}<span>${esc(name)}</span></nav>
    <header class="tp-head ds-head">
      <h1>${ownerLink(org, "/")}${esc(name)}</h1>
      <p class="lede" id="sp-lede">${esc(describe(me))}</p>
      <div class="facts sp-facts"><div class="sp-fi" id="sp-facts">${factsLine(me)}</div></div>
      <div class="facts sp-links" id="sp-links">${linksLine(me)}</div>
      ${(s.declared_openenv || s.openenv || s.manifest) && !me.L?.openenv_verified ? `<div class="note-box" id="sp-protocol-notice"><b>OpenEnv API not verified.</b> This Space was discovered from repository metadata. A file named <code>openenv.yaml</code> or a Hub tag does not establish compatibility.${me.L?.running ? " The current server did not pass the required API checks." : " It needs a successful check while running."}</div>` : ""}</header>
    <div class="tp-grid">
      <nav class="toc" aria-label="On this page"><p>On this page</p>${sections.map(([id, title]) => `<a href="#" data-jump="${id}">${esc(title)}</a>`).join("")}</nav>
      <div class="tp-body">${sections.map(([id, title, ic]) => `<section class="panel tp-sec" id="sec-${id}"><div class="panel-h"><h2>${icon(ic, 15)}${esc(title)}</h2>
        <span class="aside" id="note-${id}">${esc(noteOf(me, id))}</span></div><div class="panel-b sp-cq" id="sb-${id}">${sectionBody(me, id)}</div></section>`).join("")}</div>
      <aside class="tp-side split"><section class="panel first" id="sp-status">${statusPanel(me)}</section><section class="panel last" id="sp-glance">${glance(me)}</section></aside>
    </div></div>`;
  me.rendered = true;
  wirePlayground(me);
  wireApp(me);
  mountFiles(me);
  if (me.L?.running && me.L.task_api) loadTasks(me);
  me.observer?.disconnect();
  const links = new Map([...el.querySelectorAll("[data-jump]")].map((a) => [`sec-${a.dataset.jump}`, a]));
  const seen = new Map();
  me.observer = new IntersectionObserver((es) => {   // the contents list follows the section in view
    es.forEach((x) => seen.set(x.target.id, x.isIntersecting));
    const first = [...links.keys()].find((id) => seen.get(id));
    links.forEach((a, id) => a.classList.toggle("on", id === first));
  }, { rootMargin: "-80px 0px -55% 0px" });
  el.querySelectorAll(".tp-sec").forEach((x) => me.observer.observe(x));
}

// redraw parts in place (the header, the side, some sections), keeping a playground or an open app as they are
function paint(me, ids) {
  if (!me.rendered || page !== me) return;
  const set = (sel, html) => { const x = $(sel, me.el); if (x) x.innerHTML = html; };
  set("#sp-lede", esc(describe(me)));
  set("#sp-facts", factsLine(me));
  set("#sp-links", linksLine(me));
  set("#sp-status", statusPanel(me));
  set("#sp-glance", glance(me));
  for (const id of ids) {
    if (id === "files" && me.viewer) continue;
    set(`#sb-${id}`, sectionBody(me, id));
    set(`#note-${id}`, esc(noteOf(me, id)));
    if (id === "files") mountFiles(me);
  }
}

async function loadFiles(me) {
  try { me.F = await api(`/api/spaces/${enc(me.spec)}/files`); } catch (e) { me.F = { error: e.message }; }
  if (page !== me) return;
  paint(me, ["interface", "tools", ...(me.L?.running && me.L.task_api?.splits?.length ? [] : ["tasks"]), "rewards", "schema", "files", "connect"]);
}

async function loadLive(me, fresh) {
  let L;
  try { L = await api(`/api/spaces/${enc(me.spec)}/live${fresh ? "?fresh=1" : ""}`); } catch (e) {
    if (page !== me) return;
    if (!me.L) me.L = { running: false, stage: me.s?.stage || "UNKNOWN", error: e.message };
    const b = $("#sb-interface", me.el);
    if (b) b.insertAdjacentHTML("beforeend", `<div class="note-box err sp-after">${icon("alert")}<span>Couldn't ask the Space: ${esc(e.message)}</span></div>`);
    if (me.started) schedule(me, 6000);
    return;
  }
  if (page !== me) return;
  const was = me.L;
  me.L = L;
  if (L.task_api && me.indexKey) L.indexKey = me.indexKey;
  // the page is drawn again only when what the server offers changed (the first answer, or it woke up); otherwise
  // just the parts that say so, so a playground in progress keeps its episode
  if (!me.rendered) { /* render() runs when the card arrives and reads me.L */ }
  else if (!was || was.error || was.running !== L.running || was.openenv_verified !== L.openenv_verified) { const y = scrollY; render(me); scrollTo(0, y); }
  else if (!L.running && was.stage !== L.stage) paint(me, ["play", "app", "interface", "conformance"]);
  else paint(me, ["conformance"]);
  if (STARTING.has(L.stage)) { me.started ||= Date.now(); schedule(me, 4000); }
  else if (me.started && L.running) { me.started = null; toast("The Space is running"); }
}

function schedule(me, ms) {
  clearTimeout(me.poll);
  me.poll = setTimeout(async () => {
    if (page !== me) return;
    if (me.started && Date.now() - me.started > 10 * 60_000) {
      const t = $("#sp-elapsed", me.el);
      if (t) t.innerHTML = `Still not up after 10 minutes: <a class="u" href="https://huggingface.co/spaces/${enc(me.spec)}?logs=container" target="_blank" rel="noopener">check its logs</a>.`;
      return;
    }
    await loadLive(me, true);
  }, ms);
}

function mountFiles(me) {
  const box = $("#sp-files", me.el);
  if (!box || !me.F || me.F.error || me.viewer) return;
  const F = me.F;
  const byPath = new Map(F.files.map((f) => [f.path, f]));
  const blob = (p) => `https://huggingface.co/spaces/${enc(me.spec)}/blob/${F.sha}/${enc(p)}`;
  const raw = (p) => `https://huggingface.co/spaces/${enc(me.spec)}/resolve/${F.sha}/${enc(p)}`;
  const y = scrollY;   // the viewer brings its open file's row into view; the page stays where the reader is
  me.viewer = fileViewer(box, {
    files: F.files.map((f) => ({ path: f.path, size: f.size, withheld: !!f.withheld })),
    initial: me.filePath || F.keys?.manifest || F.keys?.readme || null,
    load: async (p) => {
      const f = byPath.get(p);
      if (f?.binary) return { binary: true, size: f.size };
      try { return await api(`/api/spaces/${enc(me.spec)}/file?f=${encodeURIComponent(p)}`); }
      catch (e) { return e.status === 415 ? { binary: true, size: f?.size } : { error: e.message }; }
    },
    hubUrl: blob,
    rawUrl: (p) => (byPath.get(p)?.binary && /\.(png|jpe?g|gif|webp|svg|avif)$/i.test(p) ? raw(p) : null),
    onChange: (p) => { me.filePath = p; },
  });
  if (scrollY !== y) scrollTo({ top: y, behavior: "instant" });
}

function openFile(me, path) {
  const sec = $("#sec-files", me.el);
  if (!sec) return;
  sec.scrollIntoView({ behavior: "smooth", block: "start" });
  if (me.viewer) me.viewer.open(path);
  else me.filePath = path;
}

function wireApp(me) {
  $(".space-app", me.el)?.addEventListener("load", (e) => e.target.closest(".sp-app-frame")?.classList.add("loaded"));
}

async function onClick(me, e) {
  const t = e.target;
  const j = t.closest("[data-jump]");
  if (j) { e.preventDefault(); $(`#sec-${j.dataset.jump}`, me.el)?.scrollIntoView({ behavior: "smooth", block: "start" }); return; }
  const jt = t.closest("[data-jump-to]");
  if (jt) { $(`#sec-${jt.dataset.jumpTo}`, me.el)?.scrollIntoView({ behavior: "smooth", block: "start" }); return; }
  const of = t.closest("[data-open-file]");
  if (of) { e.preventDefault(); openFile(me, of.dataset.openFile); return; }
  const c = t.closest("[data-copy]");
  if (c) { navigator.clipboard?.writeText(c.dataset.copy).then(() => toast("Copied"), () => toast("Couldn't copy")); return; }
  const act = t.closest("[data-act]")?.dataset.act;
  if (act === "copy-link") { navigator.clipboard?.writeText(location.href).then(() => toast("Link copied")); return; }
  if (act === "refresh") { const b = t.closest("button"); b.disabled = true; b.innerHTML = `${spinner()}Checking…`; await loadLive(me, true); return; }
  if (act === "wake" || act === "restart") {
    const b = t.closest("button");
    const msg = b.parentElement.querySelector(".sp-msg") || $('[data-msg="status"]', me.el);
    me.el.querySelectorAll(`[data-act="${act}"]`).forEach((x) => { x.disabled = true; x.innerHTML = `${spinner()}${act === "wake" ? "Waking…" : "Restarting…"}`; });
    try {
      const r = await api(`/api/spaces/${enc(me.spec)}/${act}`, { method: "POST", body: {} });
      me.started = Date.now();
      me.L = { ...(me.L || {}), running: r.stage === "RUNNING", stage: r.stage === "SLEEPING" || ERRORISH(r.stage) ? "APP_STARTING" : r.stage };
      paint(me, ["play", "app"]);
      schedule(me, 3000);
    } catch (err) {
      me.el.querySelectorAll(`[data-act="${act}"]`).forEach((x) => { x.disabled = false; x.innerHTML = act === "wake" ? `${icon("power", 14)}Wake it` : `${icon("refresh", 14)}Restart it`; });
      if (msg) msg.innerHTML = `<div class="note-box err">${icon("alert")}<span>${esc(err.message)}${err.status === 403 ? ` <a class="u" href="https://huggingface.co/spaces/${enc(me.spec)}/settings" target="_blank" rel="noopener">Its settings on the Hub</a>` : ""}</span></div>`;
    }
    return;
  }
  const app = t.closest("[data-app]")?.dataset.app;
  if (app === "load") { me.appOpen = true; $("#sb-app", me.el).innerHTML = appSection(me); wireApp(me); return; }
  if (app === "reload") { const f = $(".space-app", me.el); if (f) { f.closest(".sp-app-frame")?.classList.remove("loaded"); f.src = f.src; } return; }
  if (app === "full") { $(".sp-app", me.el)?.classList.toggle("full"); return; }
  const ct = t.closest("[data-ctab]");
  if (ct) {
    me.connectTab = ct.dataset.ctab;
    storage.set("sp-ctab", me.connectTab);
    me.el.querySelectorAll("[data-ctab]").forEach((x) => x.setAttribute("aria-pressed", String(x === ct)));
    $("#sp-connect", me.el).innerHTML = connectPane(me, me.connectTab);
    return;
  }
  const ag = t.closest("[data-agent]");
  if (ag) {
    storage.set("sp-agent", ag.dataset.agent);
    me.el.querySelectorAll("[data-agent]").forEach((x) => x.setAttribute("aria-pressed", String(x === ag)));
    $("#sp-agent", me.el).innerHTML = agentHtml(ag.dataset.agent, agentName(me.spec), `${location.origin}/mcp/${me.spec}`);
    return;
  }
  const tr = t.closest("[data-try]");
  if (tr) {
    e.preventDefault();
    const sel = $("#pg-tool", me.el);
    if (sel) {
      sel.value = tr.dataset.try;
      const modes = $(".pg-modes", me.el);
      if (modes) modes.querySelector('[data-mode="tools"]')?.click();
      showTool(me);
    }
    $("#sec-play", me.el)?.scrollIntoView({ behavior: "smooth", block: "start" });
    return;
  }
  const sp = t.closest("[data-split]");
  if (sp) {
    me.taskPage = { env: me.taskPage.env, split: sp.dataset.split, start: 0 };
    me.el.querySelectorAll("[data-split]").forEach((x) => x.setAttribute("aria-pressed", String(x === sp)));
    $("#tk-detail", me.el).hidden = true;
    loadTasks(me);
    return;
  }
  if (t.closest("#tk-prev")) { me.taskPage.start = Math.max(0, me.taskPage.start - 20); loadTasks(me); return; }
  if (t.closest("#tk-next")) { me.taskPage.start += 20; loadTasks(me); return; }
  const play = t.closest("[data-play-task]");
  if (play) {
    const split = $("#pg-split", me.el), idx = $("#pg-index", me.el);
    if (split) split.value = me.taskPage.split || "";
    if (idx) idx.value = play.dataset.playTask;
    const opts = $(".pg-opts", me.el);
    if (opts) opts.open = true;
    $("#sec-play", me.el)?.scrollIntoView({ behavior: "smooth", block: "start" });
    await reset(me);
    return;
  }
  const nextTask = t.closest("[data-task-move]");
  if (nextTask) { openTask(me, me.selectedTask + Number(nextTask.dataset.taskMove)); return; }
  if (t.closest("#tk-close")) { $("#tk-detail", me.el).hidden = true; me.el.querySelectorAll("[data-task]").forEach((r) => r.classList.remove("on")); return; }
  const row = t.closest("[data-task]");
  if (row) openTask(me, +row.dataset.task);
}
const ERRORISH = (x) => /ERROR|STOPPED|PAUSED/.test(x || "");

// ── playground ───────────────────────────────────────────────────────────────
function wirePlayground(me) {
  const ctl = $(".pg-ctl", me.el);
  if (!ctl) return;
  wireForm(ctl);
  if ($("#pg-tool", me.el)) showTool(me);
  $("#pg-reset", me.el)?.addEventListener("click", () => reset(me));
  $("#pg-step", me.el)?.addEventListener("click", () => step(me));
  $("#pg-call", me.el)?.addEventListener("click", () => call(me));
  $("#pg-send", me.el)?.addEventListener("click", () => sendRoute(me));
  $("#pg-route", me.el)?.addEventListener("change", () => showRoute(me));
  if ($("#pg-route", me.el)) showRoute(me);
  ctl.querySelector(".pg-modes")?.addEventListener("click", (e) => {
    const b = e.target.closest("[data-mode]");
    if (!b) return;
    ctl.querySelectorAll("[data-mode]").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    ctl.querySelectorAll("[data-pane]").forEach((p) => (p.hidden = p.dataset.pane !== b.dataset.mode));
  });
  $("#pg-log", me.el)?.addEventListener("click", (e) => { if (e.target.closest("#pg-end")) endSession(me); });
  $("#pg-sess", me.el)?.addEventListener("click", (e) => { if (e.target.closest("#pg-end")) endSession(me); });
  renderLog(me);
}

function showTool(me) {
  const name = $("#pg-tool", me.el)?.value;
  if (!name) return;
  const tool = (me.L.mcp || []).find((t) => t.name === name) || {};
  $("#pg-desc", me.el).textContent = (tool.description || "").split("\n\n")[0].slice(0, 400);
  // a server that lists a tool without its input schema: its arguments as JSON (named from its code when we have it)
  const params = D(me).environment?.tools?.find((t) => t.name === name)?.params || [];
  $("#pg-form", me.el).innerHTML = tool.inputSchema ? schemaForm(tool.inputSchema, { prefix: "pgt" })
    : `<label class="field"><span>Arguments <em>JSON; the server didn't list this tool's parameters${params.length ? `, its code takes ${params.map((p) => `<code>${esc(p)}</code>`).join(", ")}` : ""}</em></span>
      <textarea class="input mono" id="pg-args" rows="3" spellcheck="false" placeholder="${esc(JSON.stringify(Object.fromEntries(params.map((p) => [p, ""]))))}">${params.length ? esc(JSON.stringify(Object.fromEntries(params.map((p) => [p, ""])))) : "{}"}</textarea></label>`;
  $("#pg-warn", me.el).innerHTML = name === "run_rollout" ? `<label class="note-box warn pg-confirm"><input type="checkbox" id="pg-ok"><span>This runs a whole rollout on this Space, with its operator's sandbox and model: it takes minutes and they pay for it. To run a Harbor task on your own account, open its dataset in the explorer.</span></label>` : "";
}

function showRoute(me) {
  const r = me.L.api[+$("#pg-route", me.el).value];
  $("#pg-rdesc", me.el).textContent = r.summary || "";
  $("#pg-rparams", me.el).innerHTML = r.params.length ? `<div class="sf">${r.params.map((p) => `<label class="field sf-f"><span><code>${esc(p)}</code><em>in the path</em></span>
    <input class="input mono" data-rp="${esc(p)}" spellcheck="false" autocomplete="off"></label>`).join("")}</div>` : "";
  $("#pg-query", me.el).innerHTML = Object.keys(r.query_schema?.properties || {}).length ?
    `<div class="sec-label">Query parameters</div>${schemaForm(r.query_schema, { prefix: "pgq", skip: [] })}` : "";
  $("#pg-rform", me.el).innerHTML = r.method === "POST" ? (r.schema ? schemaForm(r.schema, { prefix: "pgr", skip: [] })
    : `<label class="field"><span>Body <em>JSON</em></span><textarea class="input mono" id="pg-rbody" rows="3" spellcheck="false">{}</textarea></label>`) : "";
}

async function sendRoute(me) {
  const r = me.L.api[+$("#pg-route", me.el).value];
  const params = {};
  for (const i of me.el.querySelectorAll("[data-rp]")) {
    if (!/^[\w.\-~]{1,200}$/.test(i.value) || [".", ".."].includes(i.value)) { toast(`${i.dataset.rp}: enter a path value using letters, digits, . - _`); return; }
    params[i.dataset.rp] = i.value;
  }
  let body = null, query = {};
  try { if (r.query_schema) query = readForm($("#pg-query", me.el), r.query_schema); }
  catch (e) { toast(e.message); return; }
  if (r.method === "POST") {
    try {
      if (r.schema) body = readForm($("#pg-rform", me.el), r.schema);
      else body = JSON.parse($("#pg-rbody", me.el)?.value || "{}");
    } catch (e) { toast(e.message || "The body isn't valid JSON"); return; }
  }
  await send(me, "http", { method: r.method, path: r.path, params, query, body }, `${r.method} ${r.path}`);
}

async function ensureSession(me) {
  if (!me.session) {
    const generation = me.sessionGeneration || 0;
    const id = (await api(`/api/spaces/${enc(me.spec)}/play`, { method: "POST", body: { op: "start" } })).session;
    if (page !== me || generation !== (me.sessionGeneration || 0)) {
      api(`/api/spaces/${enc(me.spec)}/play`, { method: "POST", body: { op: "end", session: id } }).catch(() => {});
      throw new Error("This playground was closed.");
    }
    me.session = id;
  }
  return me.session;
}

function endSession(me) {
  me.sessionGeneration = (me.sessionGeneration || 0) + 1;
  const id = me.session;
  me.session = null; me.episode = null; me.log = [];
  if (id) api(`/api/spaces/${enc(me.spec)}/play`, { method: "POST", body: { op: "end", session: id } }).catch(() => {});
  renderLog(me);
}

async function send(me, op, data, label) {
  if (me.sending) return;
  me.sending = true;
  const entry = { op, label, pending: true };
  if (op === "reset") me.log = [];
  me.log.push(entry);
  renderLog(me);
  busy(me, true);
  const post = () => api(`/api/spaces/${enc(me.spec)}/play`, { method: "POST", body: { op, session: me.session, data } });
  try {
    await ensureSession(me);
    if (!me.log.includes(entry)) return;
    let r;
    try { r = await post(); } catch (err) {
      if (err.status !== 410 || !me.log.includes(entry)) throw err;
      me.session = null;   // it timed out: a fresh session (an episode in progress is gone, so a step has to start over)
      if (op !== "reset") { me.episode = null; throw new Error("The session ended. Start a new episode before continuing."); }
      await ensureSession(me);
      r = await post();
    }
    if (page !== me || !me.log.includes(entry)) return;
    Object.assign(entry, { pending: false, result: spaceMedia(r.result, me.L.host), ms: r.ms, stateful: r.stateful });
    if (op === "reset") me.episode = { steps: 0, total: 0, done: false, task: taskOf(r.result) };
    else {
      me.episode ||= { steps: 0, total: 0, done: false, task: null };
      me.episode.steps++;
      const rw = rewardOf(op, r.result);
      if (typeof rw === "number") { me.episode.total += rw; me.rewards.push(rw); paint(me, ["interface", "rewards"]); }
      if (doneOf(r.result)) me.episode.done = true;
    }
  } catch (err) {
    Object.assign(entry, { pending: false, error: err.message });
  } finally {
    me.sending = false;
    if (page === me) { busy(me, false); renderLog(me); }
  }
}
const rewardOf = (op, r) => (op === "http" ? (typeof r?.json?.reward === "number" ? r.json.reward : null) : op === "call" ? (typeof r?.structuredContent?.reward === "number" ? r.structuredContent.reward : typeof r?.structured_content?.reward === "number" ? r.structured_content.reward : null)
  : r?.reward ?? r?.observation?.reward);
const doneOf = (r) => Boolean(r?.done || r?.observation?.done || r?.terminated || r?.truncated ||
  r?.json?.done || r?.json?.terminated || r?.json?.truncated || r?.structuredContent?.done);
const taskOf = (r) => { const o = r?.observation || {}; return o.task_id || o.metadata?.task_id || (o.index != null ? `#${o.index}` : null); };
const busy = (me, on) => me.el.querySelectorAll("#pg-reset, #pg-step, #pg-call, #pg-send, [data-play-task]").forEach((b) => (b.disabled = on));

function resetParams(me) {
  const out = {};
  const raw = $("#pg-params", me.el)?.value.trim();
  if (raw) {
    let p;
    try { p = JSON.parse(raw); } catch { throw new Error("Reset options aren't valid JSON"); }
    if (!p || typeof p !== "object" || Array.isArray(p)) throw new Error("Reset options should be a JSON object, like {\"seed\": 1}");
    Object.assign(out, p);
  }
  const split = $("#pg-split", me.el)?.value, idx = $("#pg-index", me.el)?.value;
  if (split) out.split = split;
  if (idx !== "" && idx != null) {
    if (!Number.isSafeInteger(Number(idx)) || Number(idx) < 0) throw new Error("The task index should be a whole number, 0 or more");
    out[me.indexKey || "index"] = Number(idx);   // servers name it their own way: the key their Task API rows use
  }
  return out;
}

async function reset(me) {
  let params;
  try { params = resetParams(me); } catch (e) { toast(e.message); return; }
  await send(me, "reset", params, Object.keys(params).length ? JSON.stringify(params) : "");
}

async function step(me) {
  let action;
  try { action = readForm($("#pg-form", me.el), me.L.schema?.action || {}); } catch (e) { toast(e.message); return; }
  await send(me, "step", action, JSON.stringify(action).slice(0, 90));
}

async function call(me) {
  const name = $("#pg-tool", me.el).value;
  const tool = (me.L.mcp || []).find((t) => t.name === name) || {};
  if (name === "run_rollout" && !$("#pg-ok", me.el)?.checked) { toast("Tick the box first: this runs on the Space operator's account"); return; }
  let args;
  try {
    if (tool.inputSchema) args = readForm($("#pg-form", me.el), tool.inputSchema);
    else {
      args = JSON.parse($("#pg-args", me.el)?.value.trim() || "{}");
      if (!args || typeof args !== "object" || Array.isArray(args)) throw new Error("Arguments should be a JSON object");
    }
  } catch (e) { toast(e instanceof SyntaxError ? "The arguments aren't valid JSON" : e.message); return; }
  await send(me, "call", { name, arguments: args }, `${name}(${JSON.stringify(args).slice(1, -1).slice(0, 80)})`);
}

function renderLog(me) {
  const box = $("#pg-log", me.el), sess = $("#pg-sess", me.el);
  if (!box) return;
  const ep = me.episode;
  if (sess) {
    sess.innerHTML = me.session
      ? `<span class="dot" style="--dc:var(--ok)"></span><span>${ep ? `${ep.task ? `<code title="${esc(ep.task)}">${esc(ep.task)}</code> · ` : ""}${fmt.format(ep.steps)} step${ep.steps === 1 ? "" : "s"}${ep.total ? ` · reward <b>${esc(String(+ep.total.toFixed(4)))}</b>` : ""}${ep.done ? " · done" : ""}` : "Session open"}</span><button class="link sm" type="button" id="pg-end">End</button>`
      : `<span class="dot" style="--dc:var(--faint)"></span><span>No episode yet</span>`;
  }
  if (!me.log.length) {
    box.innerHTML = `<div class="pg-empty">${icon("play", 22)}<b>${me.L?.openenv ? "Start an episode" : me.L?.api?.length ? "Send a request" : "Call a tool"}</b><p>Each request shows here with the server's response and any reward it returns.</p></div>`;
    return;
  }
  box.innerHTML = me.log.map((x, i) => {
    const recent = i >= me.log.length - 2;
    const rw = x.result ? rewardOf(x.op, x.result) : null;
    const head = `<span class="pg-op ${x.op}">${esc(x.op)}</span><span class="pg-lbl">${esc(x.label || "")}</span><span class="grow"></span>
      ${x.pending ? spinner("xs") : x.error ? `<span class="rv-done err">failed</span>` : `${rw != null ? `<span class="rv-reward ${rw > 0 ? "pos" : rw < 0 ? "neg" : ""}">${esc(String(+Number(rw).toFixed(4)))}</span>` : ""}<span class="faint xs">${fmt.format(x.ms)} ms</span>`}`;
    const body = x.pending ? `<div class="pg-wait">${x.op === "reset" ? "Starting an episode on the Space…" : "Waiting for the environment…"}</div>`
      : x.error ? `<div class="note-box err">${icon("alert")}<span>${esc(x.error)}</span></div>`
      : x.op === "call" ? callView(x.result) : x.op === "http" ? httpView(x.result) : obsView(x.result);
    return `<details class="pg-entry" ${recent || x.error ? "open" : ""}><summary>${head}</summary><div class="pg-body">${body}</div></details>`;
  }).join("") + (me.L?.openenv && me.log.at(-1)?.stateful === false ? `<p class="fine">The WebSocket session is unavailable. These tool calls use HTTP and may not retain episode state.</p>` : "");
}

// ── what the environment sent back ───────────────────────────────────────────
// One status line (reward, done, its feedback), then long text at full width (newlines kept, folded when long),
// images, the short fields as a wrapping key–value grid, and nested objects (metadata) as smaller grids. Nothing
// scrolls inside the log, and nothing scrolls sideways.
const STATUS_KEYS = ["feedback", "message", "status", "result", "error", "info", "action_result"];
const ORDER = ["prompt", "instruction", "question", "task", "observation", "text", "content", "dashboard", "output", "stdout", "stderr"];
const isObj = (v) => v && typeof v === "object" && !Array.isArray(v);
const isMedia = (html) => /^<(img|audio|video)\b/.test(html);
let foldN = 0;

function textBlock(v) {
  const s = String(v);
  const looksMd = /(^|\n)(#{1,4} |[-*] |\d+\. |```)/.test(s) && s.length < 40_000;
  const body = looksMd ? `<div class="prose pgv-md">${md(s)}</div>` : `<div class="pgv-txt">${esc(s.length > 60_000 ? s.slice(0, 60_000) + "\n… (cut)" : s)}</div>`;
  const long = s.length > 1400 || s.split("\n").length > 18;
  if (!long) return body;
  const id = `pgv-f${++foldN}`;
  return `<div class="pgv-fold"><input type="checkbox" class="pgv-tog" id="${id}"><div class="pgv-fold-b">${body}</div>
    <label for="${id}" class="pgv-more"><span class="o">Show all ${fmt.format(s.length)} characters</span><span class="c">Show less</span></label></div>`;
}

function factHtml(k, v, ctx) {
  if (v == null) return `<span class="faint">–</span>`;
  if (typeof v === "boolean") return `<span class="rv-bool ${v}">${v}</span>`;
  if (typeof v === "number") return `<span class="pgv-num">${esc(Number.isInteger(v) ? fmt.format(v) : String(+v.toFixed(6)))}</span>`;
  if (typeof v === "string") { const h = value(v, k, 0, ctx); return h.startsWith("<span class=\"rv-faint\"") ? h : `<span class="pgv-s">${esc(v)}</span>`; }
  if (Array.isArray(v)) return v.length ? `<span class="pgv-list">${v.map((x) => `<code>${esc(typeof x === "string" ? x : JSON.stringify(x))}</code>`).join("")}</span>` : `<span class="faint">none</span>`;
  return `<span class="pgv-inl">${Object.entries(v).map(([a, b]) => `<span><i>${esc(a)}</i> ${b == null ? "–" : esc(typeof b === "object" ? JSON.stringify(b) : String(b))}</span>`).join("")}</span>`;
}

// an object's fields, sorted into texts, media, short facts and nested groups
function fieldsView(o, depth = 0, hide = []) {
  const texts = [], media = [], facts = [], groups = [];
  const keys = Object.keys(o).filter((k) => !hide.includes(k));
  keys.sort((a, b) => (ORDER.includes(a) ? ORDER.indexOf(a) : 99) - (ORDER.includes(b) ? ORDER.indexOf(b) : 99));
  for (const k of keys) {
    const v = o[k];
    if (typeof v === "string") {
      const h = value(v, k, 0, o);
      if (isMedia(h)) media.push([k, h]);
      else if (v.length > 140 || v.includes("\n")) texts.push([k, v]);
      else facts.push([k, v]);
    } else if (Array.isArray(v)) {
      const short = v.length <= 40 && v.every((x) => x == null || ["string", "number", "boolean"].includes(typeof x)) && v.every((x) => String(x).length <= 60);
      if (short) facts.push([k, v]);
      else groups.push([k, `<div class="pgv-val">${value(v, k, 1, o)}</div>`]);
    } else if (isObj(v)) {
      const flat = Object.values(v).every((x) => x == null || typeof x !== "object") && Object.keys(v).length <= 8;
      if (depth > 0 && flat) facts.push([k, v]);
      else if (Object.keys(v).length) groups.push([k, fieldsView(v, depth + 1)]);
      else facts.push([k, null]);
    } else facts.push([k, v]);
  }
  return `<div class="pgv${depth ? " sub" : ""}">
    ${texts.map(([k, v]) => `<div class="pgv-sec"><div class="pgv-k">${esc(k)}</div>${textBlock(v)}</div>`).join("")}
    ${media.map(([k, h]) => `<div class="pgv-sec"><div class="pgv-k">${esc(k)}</div><div class="pgv-media">${h}</div></div>`).join("")}
    ${facts.length ? `<dl class="pgv-grid">${facts.map(([k, v]) => `<div class="${typeof v === "string" && v.length > 40 || isObj(v) || (Array.isArray(v) && v.length > 4) ? "wide" : ""}"><dt>${esc(k)}</dt><dd>${factHtml(k, v, o)}</dd></div>`).join("")}</dl>` : ""}
    ${groups.map(([k, h]) => `<div class="pgv-sec"><div class="pgv-k">${esc(k)}</div>${h}</div>`).join("")}
  </div>`;
}

function statusLine(bits) {
  const shown = bits.filter(Boolean);
  return shown.length ? `<div class="pgv-status">${shown.join("")}</div>` : "";
}

// reset/step: {observation, reward, done}
function obsView(data) {
  const obs = isObj(data) && "observation" in data ? data.observation : data;
  const reward = data?.reward ?? obs?.reward, done = data?.done ?? obs?.done;
  const said = isObj(obs) ? STATUS_KEYS.find((k) => typeof obs[k] === "string" && obs[k] && obs[k].length <= 240 && !obs[k].includes("\n")) : null;
  const head = statusLine([
    reward != null ? `<span class="pgv-rw ${Number(reward) > 0 ? "pos" : Number(reward) < 0 ? "neg" : ""}">reward <b>${esc(typeof reward === "number" ? num(reward) : String(reward))}</b></span>` : null,
    done != null ? `<span>${done ? "<b>done</b>" : "not done"}</span>` : null,
    said ? `<span class="pgv-said">${esc(obs[said])}</span>` : null]);
  if (!isObj(obs)) return head + (obs == null ? "" : typeof obs === "string" ? textBlock(obs) : `<div class="pgv-val">${value(obs, "observation")}</div>`);
  return head + fieldsView(obs, 0, ["reward", "done", ...(said ? [said] : [])]);
}

// an MCP tool's result: its content parts (JSON parts as fields), images, or its raw value
function callView(r) {
  if (!isObj(r) || !Array.isArray(r.content)) return isObj(r) ? fieldsView(r) : `<div class="pgv-val">${value(r)}</div>`;
  const sc = r.structuredContent || r.structured_content;
  const rw = typeof sc?.reward === "number" ? sc.reward : null;
  const head = statusLine([r.isError || r.is_error ? `<span class="pgv-err">error</span>` : null,
    rw != null ? `<span class="pgv-rw ${rw > 0 ? "pos" : rw < 0 ? "neg" : ""}">reward <b>${esc(num(rw))}</b></span>` : null]);
  const parts = r.content.map((c) => {
    if (c?.type === "text") {
      try { const j = JSON.parse(c.text); if (isObj(j)) return fieldsView(j); if (Array.isArray(j)) return `<div class="pgv-val">${value(j)}</div>`; } catch { /* plain text */ }
      return textBlock(c.text || "");
    }
    if (c?.type === "image" && /^image\/(png|jpeg|gif|webp)$/.test(c.mimeType || c.mime_type)) return `<div class="pgv-media"><img class="rv-img" src="data:${esc(c.mimeType || c.mime_type)};base64,${esc(c.data)}" alt=""></div>`;
    if (c?.type === "audio") return `<audio class="rv-audio" controls src="data:${esc(c.mimeType || "audio/wav")};base64,${esc(c.data)}"></audio>`;
    return `<div class="pgv-val">${value(c)}</div>`;
  }).join("");
  return head + (parts || `<span class="faint sm">no content</span>`);
}

// a route's answer: its status, then its body (JSON, text, or an image)
function httpView(r) {
  const ok = r.status >= 200 && r.status < 300;
  const head = statusLine([`<span class="${ok ? "" : "pgv-err"}">HTTP ${esc(r.status)}</span>`,
    typeof r.json?.reward === "number" ? `<span class="pgv-rw ${r.json.reward > 0 ? "pos" : ""}">reward <b>${esc(num(r.json.reward))}</b></span>` : null]);
  if (r.image) return head + `<div class="pgv-media"><img class="rv-img" src="${esc(r.image)}" alt=""></div>`;
  if (r.json !== undefined) return head + (isObj(r.json) ? fieldsView(r.json) : `<div class="pgv-val">${value(r.json)}</div>`);
  return head + (r.text ? textBlock(r.text) : `<span class="faint sm">no body</span>`);
}

// ── tasks ────────────────────────────────────────────────────────────────────
async function loadTasks(me) {
  const L = me.L, box = $("#tk-table", me.el);
  if (!box || !L?.task_api) return;
  const splits = taskCatalog(me).splits;
  me.taskPage.split ??= (splits.find((x) => x.default) || splits[0])?.name || "";
  const { split, start, env } = me.taskPage;
  me.selectedTask = null;
  $("#tk-detail", me.el).hidden = true;
  const total = splits.find((x) => x.name === split)?.num_tasks;
  box.innerHTML = sk.lines(90, 90, 90, 90);
  $("#tk-range", me.el).textContent = "";
  try {
    const r = await api(`/api/spaces/${enc(me.spec)}/tasks?env=${encodeURIComponent(env)}&split=${encodeURIComponent(split)}&start=${start}&stop=${start + 20}`);
    if (page !== me || me.taskPage.env !== env || me.taskPage.split !== split || me.taskPage.start !== start) return;
    me.tasks = r.tasks;
    if (r.tasks.length && me.indexKey == null) { me.indexKey = "task_index" in r.tasks[0] ? "task_index" : "index"; L.indexKey = me.indexKey; }
    const keys = columns(r.tasks);
    box.innerHTML = r.tasks.length ? `<div class="tbl tk-tbl"><table><thead><tr>${keys.map((k) => `<th>${esc(k)}</th>`).join("")}</tr></thead><tbody>${r.tasks.map((t, i) =>
      `<tr data-task="${i}" tabindex="0" aria-label="Inspect task ${start + i + 1}">${keys.map((k) => `<td title="${esc(typeof t[k] === "string" ? t[k] : "")}">${cellOf(t[k])}</td>`).join("")}</tr>`).join("")}</tbody></table></div>
      ${r.withheld.length ? `<p class="fine">${icon("shield", 12)} Left out, as they may hold the answer: ${r.withheld.map((k) => `<code>${esc(k)}</code>`).join(", ")}.</p>` : ""}`
      : `<p class="muted sm">No tasks here.</p>`;
    $("#tk-range", me.el).textContent = r.tasks.length ? `${fmt.format(start + 1)}–${fmt.format(start + r.tasks.length)}${total != null ? ` of ${fmt.format(total)}` : ""}` : "";
    $("#tk-prev", me.el).disabled = start === 0;
    $("#tk-next", me.el).disabled = r.tasks.length < 20 || (total != null && start + 20 >= total);
    box.querySelectorAll("[data-task]").forEach((tr) => tr.addEventListener("keydown", (e) => { if (e.key === "Enter") openTask(me, +tr.dataset.task); }));
    if (me.linkedTask != null) { const target = me.linkedTask; me.linkedTask = null; openTask(me, target - start); }
  } catch (e) { box.innerHTML = `<div class="note-box err">${icon("alert")}<span>${esc(e.message)}</span></div>`; }
}

function columns(rows) {
  const first = ["task_index", "index", "task_id", "id", "task_name", "name", "title"];
  const keys = [...new Set(rows.flatMap((r) => Object.keys(r)))];
  const short = keys.filter((k) => rows.every((r) => r[k] == null || (["string", "number", "boolean"].includes(typeof r[k]) && String(r[k]).length <= 80)));
  const pick = [...first.filter((k) => keys.includes(k)), ...short.filter((k) => !first.includes(k) && !["split", "dataset"].includes(k))].slice(0, 6);
  return pick.length ? pick : keys.slice(0, 4);
}
const cellOf = (x) => (x == null ? `<span class="faint">–</span>` : typeof x === "string" ? esc(x.length > 80 ? x.slice(0, 80) + "…" : x)
  : typeof x === "object" ? `<span class="faint">${Array.isArray(x) ? `${x.length} items` : "{…}"}</span>` : `<code>${esc(String(x))}</code>`);

async function openTask(me, i) {
  const t = me.tasks?.[i];
  if (!t) return;
  me.selectedTask = i;
  const requestKey = me.taskRequest = (me.taskRequest || 0) + 1;
  me.el.querySelectorAll("[data-task]").forEach((r) => r.classList.toggle("on", +r.dataset.task === i));
  const idx = t.task_index ?? t.index ?? me.taskPage.start + i;
  const d = $("#tk-detail", me.el);
  d.hidden = false;
  const hub = me.L.harbor_caps?.datasets.find((x) => x.name === me.taskPage.split)?.hub;
  const rel = hub && typeof t.task_id === "string" && t.task_id.startsWith(me.taskPage.split + "/") ? t.task_id.slice(me.taskPage.split.length + 1) : null;
  const title = t.task_name || t.title || t.task_id || t.id || `Task ${idx + 1}`;
  const link = new URL(location.href);
  link.search = new URLSearchParams({ env: me.taskPage.env, split: me.taskPage.split, task: idx });
  link.hash = "sec-tasks";
  const canPlay = me.L.openenv && !me.L.harbor && me.taskPage.env === me.L.task_api.env;
  d.innerHTML = `<div class="tk-detail-h"><div class="tk-title"><small>Task ${fmt.format(me.taskPage.start + i + 1)} · ${esc(shortSplit(me.taskPage.split))}</small><b title="${esc(title)}">${esc(title)}</b></div>
      <button class="icon-btn sm" type="button" data-task-move="-1" aria-label="Previous task in this page" ${i === 0 ? "disabled" : ""}>${icon("chevronRight", 15, "flip")}</button>
      <button class="icon-btn sm" type="button" data-task-move="1" aria-label="Next task in this page" ${i + 1 === me.tasks.length ? "disabled" : ""}>${icon("chevronRight", 15)}</button>
      <button class="icon-btn sm" type="button" id="tk-close" aria-label="Close task">${icon("x", 15)}</button></div>
    <div class="tk-actions">${copyBtn(link.href, "Copy task link")}
      ${rel ? `<a class="btn sm" href="/t/${enc(hub)}/${enc(rel)}" title="Open it in the explorer, where it runs on your account">${icon("database", 13)}Open in the explorer</a>` : ""}
      ${canPlay ? `<button class="btn sm primary" type="button" data-play-task="${esc(idx)}">${icon("play", 13)}Start this task</button>` : ""}</div>
    <div class="tk-detail-b">${taskContent(t)}<p class="fine" id="tk-full-status" role="status">Loading the full task…</p></div>`;
  d.scrollIntoView({ block: "nearest", behavior: "smooth" });
  try {
    const full = await api(`/api/spaces/${enc(me.spec)}/task?${new URLSearchParams({ env: me.taskPage.env, split: me.taskPage.split, index: idx })}`);
    if (page !== me || requestKey !== me.taskRequest || me.selectedTask !== i || !d.isConnected || d.hidden) return;
    if (full.task && typeof full.task === "object" && !Array.isArray(full.task)) {
      $(".tk-detail-b", d).innerHTML = taskContent(full.task);
    } else $("#tk-full-status", d).textContent = "Showing the task record returned by the list endpoint.";
  } catch (error) {
    if (page === me && requestKey === me.taskRequest && me.selectedTask === i && d.isConnected && !d.hidden)
      $("#tk-full-status", d).textContent = `Showing the list record. Full task unavailable: ${error.message}`;
  }
}
