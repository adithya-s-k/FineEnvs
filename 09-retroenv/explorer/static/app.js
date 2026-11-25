"use strict";
// RetroEnv explorer: the environment, the task table, task pages with known routes, live play, and eval runs (evals.js).

const $ = (selector, root = document) => root.querySelector(selector);
const esc = (s) =>
  String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const fmt = (n) => (n == null ? "–" : Number(n).toLocaleString("en-US"));
const num = (x, d = 3) => (x == null ? "–" : Number(x).toFixed(d));
const store = {
  get(key, fallback) {
    try {
      return localStorage.getItem(key) ?? fallback;
    } catch {
      return fallback;
    }
  },
  set(key, value) {
    try {
      localStorage.setItem(key, value);
    } catch {
      /* storage may be blocked */
    }
  },
};
const state = { benchmark: null };
let META = null;
const overviewCache = new Map();

async function api(path, params = {}) {
  const query = new URLSearchParams(Object.entries(params).filter(([, v]) => v !== "" && v != null)).toString();
  const response = await fetch(`/api/${path}${query ? `?${query}` : ""}`);
  if (!response.ok) {
    const text = await response.text();
    // FastAPI's bare "Not Found" means the route itself is missing: the server predates this page.
    if (response.status === 404 && text === '{"detail":"Not Found"}') {
      throw new Error("The explorer server is older than this page. Stop it (Ctrl+C) and start it again.");
    }
    throw new Error(`${response.status}: ${text}`);
  }
  return response.json();
}

function go(path, params = {}) {
  const query = new URLSearchParams(Object.entries(params).filter(([, v]) => v !== "" && v != null)).toString();
  location.hash = `${path}${query ? `?${query}` : ""}`;
}

// --- molecules: drawn by RDKit on the server, in currentColor so they follow the theme ---
const svgCache = new Map();
function mol(smiles, w = 200, h = 110, cls = "") {
  return `<span class="mol ${cls}" data-smi="${esc(smiles)}" data-w="${w}" data-h="${h}" title="${esc(smiles)}" style="width:${w}px;height:${h}px"></span>`;
}
async function hydrate(root = document) {
  const pending = [...root.querySelectorAll(".mol[data-smi]:not([data-done])")];
  await Promise.all(
    pending.map(async (el) => {
      el.dataset.done = "1";
      const key = `${el.dataset.smi}|${el.dataset.w}|${el.dataset.h}`;
      if (!svgCache.has(key)) {
        const query = new URLSearchParams({ smi: el.dataset.smi, w: el.dataset.w, h: el.dataset.h });
        svgCache.set(
          key,
          fetch(`/api/mol.svg?${query}`).then((r) => r.text()),
        );
      }
      el.innerHTML = await svgCache.get(key);
    }),
  );
}

function bars(rows, attrs = () => "") {
  const max = Math.max(1, ...rows.map((r) => r[1]));
  return `<div class="bars">${rows
    .map((r) => {
      const [label, value, shown] = r;
      return `<div class="row" ${attrs(r)}><span class="lbl">${esc(label)}</span><div class="track"><div class="fill" style="width:${((100 * value) / max).toFixed(1)}%"></div></div><span class="n">${shown ?? fmt(value)}</span></div>`;
    })
    .join("")}</div>`;
}
const STEP_ORDER = (a, b) => Number(a[0]) - Number(b[0]);
const VARIANTS = ["standard", "max_depth", "restricted_stock", "forbidden_class", "diversity"];
const TIER_ORDER = { easy: 0, medium: 1, hard: 2 };
const SIZE_ORDER = { "<=20": 0, "21-30": 1, "31-40": 2, "41+": 3 };

async function overviewData() {
  if (!overviewCache.has(state.benchmark)) overviewCache.set(state.benchmark, api("overview", { benchmark: state.benchmark }));
  return overviewCache.get(state.benchmark);
}

// --- environment (the RL side) ---
async function overviewView(view, params) {
  const data = await overviewData();
  const splits = Object.keys(data.splits);
  const focus =
    params.split && data.splits[params.split] ? params.split : splits.includes("train") ? "train" : splits[0];
  const s = data.splits[focus];
  const filterLink =
    (key) =>
    ([label]) =>
      `data-go='${esc(JSON.stringify({ split: focus, [key]: label }))}'`;
  const lib = META.benchmarks.find((b) => b.name === state.benchmark)?.manifest?.library || {};
  view.innerHTML = `
    <div class="grid" style="grid-template-columns:minmax(0,3fr) minmax(0,2fr)">
      <div class="panel"><div class="panel-h"><h2>${esc(state.benchmark)}: tasks by split</h2></div>
        <div class="tbl" style="border:0"><table>
          <tr><th>Split</th><th>Used for</th><th class="num">Targets</th><th class="num">Tasks</th><th class="num">Variants</th><th class="num">Easy / medium / hard</th></tr>
          ${splits
            .map((k) => {
              const x = data.splits[k];
              return `<tr class="click${k === focus ? " on" : ""}" data-split="${k}"><td>${k}</td>
            <td class="sm muted">${esc(META.split_use[k] || "")}</td>
            <td class="num">${fmt(x.parents)}</td><td class="num">${fmt(x.tasks)}</td><td class="num">${fmt(x.tasks - x.parents)}</td>
            <td class="num">${["easy", "medium", "hard"].map((t) => fmt(x.tier[t] || 0)).join(" / ")}</td></tr>`;
            })
            .join("")}
        </table></div></div>
      <div class="panel"><div class="panel-h"><h2>One episode</h2></div><div class="panel-b"><dl class="kv">
        <dt>Observation</dt><dd>The target, a depth budget (longest linear sequence), how many routes, and any constraint; no stock list and no answer</dd>
        <dt>Actions</dt><dd>Calls to ${META.tools.length} tools, none of which reads a hidden route</dd>
        <dt>End</dt><dd>One <span class="mono">emit_routes</span> call with the route trees</dd>
        <dt>Step check</dt><dd>A frozen library of ${fmt(lib.reactions)} reactions and ${fmt(lib.templates)} rdchiral templates (frequency ≥ ${lib.min_template_count ?? 5})</dd>
        <dt>Pass</dt><dd>Enough valid routes: supported steps, every leaf in stock, constraints met</dd>
      </dl></div></div>
    </div>
    <div class="section">
      <h2>${esc(focus)} <span class="faint sm">· click a bar to list those tasks</span></h2>
      <div class="grid">
        <div class="panel"><div class="panel-h"><h3>Shortest known route (reactions)</h3></div><div class="panel-b">${bars(
          Object.entries(s.min_depth).sort(STEP_ORDER),
          filterLink("min_depth"),
        )}</div></div>
        <div class="panel"><div class="panel-h"><h3>Task variants</h3></div><div class="panel-b">${bars(
          VARIANTS.filter((v) => s.variant[v]).map((v) => [v, s.variant[v]]),
          filterLink("variant"),
        )}</div></div>
        <div class="panel"><div class="panel-h"><h3>First reaction of the known route</h3></div><div class="panel-b">${bars(
          Object.entries(s.family).sort((a, b) => b[1] - a[1]).slice(0, 14),
          filterLink("family"),
        )}</div></div>
        <div class="panel"><div class="panel-h"><h3>Difficulty tier</h3></div><div class="panel-b">${bars(
          Object.entries(s.tier).sort((a, b) => (TIER_ORDER[a[0]] ?? 9) - (TIER_ORDER[b[0]] ?? 9)),
          filterLink("tier"),
        )}</div></div>
        <div class="panel"><div class="panel-h"><h3>Heavy atoms in the target</h3></div><div class="panel-b">${bars(
          Object.entries(s.size).sort((a, b) => SIZE_ORDER[a[0]] - SIZE_ORDER[b[0]]),
          filterLink("size"),
        )}</div></div>
        <div class="panel"><div class="panel-h"><h3>Targets with</h3></div><div class="panel-b">${bars([
          ["a convergent known route", s.convergent],
          ["defined stereocentres", s.stereo],
        ])}</div></div>
      </div>
    </div>
    <div class="section">
      <h2>Tools and reward</h2>
      <div class="grid">
        <div class="panel"><div class="panel-h"><h3>Tools the policy can call</h3></div><div class="tbl" style="border:0"><table>
          ${META.tools.map((t) => `<tr><td class="mono">${esc(t.name)}</td><td class="sm">${esc(t.description)}${t.assist ? ` <span class="faint xs">(dropped in the unaided ablation)</span>` : ""}</td></tr>`).join("")}
        </table></div></div>
        <div class="panel"><div class="panel-h"><h3>Reward components</h3></div><div class="panel-b">${bars(
          Object.entries(META.weights)
            .sort((a, b) => b[1] - a[1])
            .map(([k, w]) => [k, w, w.toFixed(2)]),
        )}</div></div>
      </div>
    </div>`;
  view
    .querySelectorAll("tr[data-split]")
    .forEach((row) => row.addEventListener("click", () => go("/", { split: row.dataset.split })));
  view
    .querySelectorAll("[data-go]")
    .forEach((row) => row.addEventListener("click", () => go("/tasks", JSON.parse(row.dataset.go))));
}

// --- task table ---
async function tasksView(view, params) {
  const options = await overviewData();
  const p = { split: "", variant: "", min_depth: "", family: "", tier: "", size: "", q: "", offset: "0", ...params };
  const data = await api("tasks", { benchmark: state.benchmark, ...p, limit: 50 });
  const all = Object.values(options.splits);
  const keys = (field) => [...new Set(all.flatMap((x) => Object.keys(x[field])))];
  const select = (name, label, values) =>
    `<select data-f="${name}" aria-label="${label}"><option value="">${label}: any</option>${values.map((v) => `<option value="${esc(v)}"${String(p[name]) === String(v) ? " selected" : ""}>${esc(v)}</option>`).join("")}</select>`;
  const offset = Number(p.offset) || 0;
  view.innerHTML = `
    <div class="toolbar">
      <input class="input" id="q" placeholder="Task ID, SMILES or substring" value="${esc(p.q)}">
      ${select("split", "Split", Object.keys(options.splits))}
      ${select("variant", "Variant", VARIANTS)}
      ${select("min_depth", "Shortest route", keys("min_depth").sort((a, b) => a - b))}
      ${select("family", "First reaction", keys("family").sort())}
      ${select("tier", "Tier", ["easy", "medium", "hard"])}
      ${select("size", "Heavy atoms", ["<=20", "21-30", "31-40", "41+"])}
    </div>
    <div class="tbl"><table>
      <tr><th></th><th>Task</th><th>Split</th><th>Variant</th><th class="num">Shortest / budget</th><th>First reaction</th><th>Tier</th><th class="num">Heavy atoms</th><th>Constraint</th></tr>
      ${data.rows
        .map(
          (r) => `<tr class="click" data-id="${esc(r.id)}"><td>${mol(r.smiles, 96, 52, "thumb")}</td><td class="mono">${esc(r.id)}</td><td>${r.split}</td>
        <td class="sm">${esc(r.variant)}</td><td class="num">${r.min_depth} / ${r.max_depth}</td><td class="sm">${esc(r.family)}</td><td class="sm">${esc(r.tier || "–")}</td>
        <td class="num">${r.heavy}</td><td class="sm">${esc(r.constraint || "")}</td></tr>`,
        )
        .join("")}
    </table></div>
    <div class="pager">${data.total ? `${fmt(offset + 1)}–${fmt(Math.min(offset + 50, data.total))} of ${fmt(data.total)}` : "No tasks match"}
      <button class="btn" id="prev" type="button">Previous</button><button class="btn" id="next" type="button">Next</button></div>`;
  const update = (patch) => go("/tasks", { ...p, ...patch, offset: patch.offset ?? 0 });
  view
    .querySelectorAll("select[data-f]")
    .forEach((el) => el.addEventListener("change", () => update({ [el.dataset.f]: el.value })));
  $("#q").addEventListener("keydown", (e) => {
    if (e.key === "Enter") update({ q: e.target.value.trim() });
  });
  $("#prev").addEventListener("click", () => update({ offset: Math.max(0, offset - 50) }));
  $("#next").addEventListener("click", () => {
    if (offset + 50 < data.total) update({ offset: offset + 50 });
  });
  view
    .querySelectorAll("tr[data-id]")
    .forEach((row) => row.addEventListener("click", () => go(`/task/${encodeURIComponent(row.dataset.id)}`)));
}

// --- route trees ---
function submittedTree(root) {
  const node = (n) => {
    const head = `<div class="node-head">${mol(n.smiles || "", 130, 70)}<div class="meta"><span class="mono" style="font-size:11px">${esc(n.smiles)}</span>
      ${n.children?.length ? `<span class="faint">expanded</span>` : `<span class="${n.in_stock ? "ok" : "warn"}"><i class="dot"></i>${n.in_stock ? "claimed in stock" : "claimed not in stock"}</span>`}</div></div>`;
    const reaction = n.children?.[0];
    if (!reaction) return `<div class="node">${head}</div>`;
    const md = reaction.metadata || {};
    return `<div class="node">${head}<div class="rxn">↳ ${esc(md.explanation || "")} <span class="faint">· ${esc(md.reaction_class || "")}${md.confidence != null ? ` · confidence ${md.confidence}` : ""}</span></div>
      <div class="kids">${(reaction.children || []).map(node).join("")}</div></div>`;
  };
  return `<div class="tree">${node(root)}</div>`;
}

// --- tool calls and their results ---
function callLabel(name, a) {
  const cut = (p, rs) => `${esc(p || "")} → ${(rs || []).map(esc).join(" + ")}`;
  switch (name) {
    case "validate_disconnection":
    case "reaction_class_lookup":
    case "reaction_conditions_search":
      return cut(a.product_smiles, a.reactants);
    case "stock_retrieve":
      return `${esc(a.query)} <span class="faint">(${esc(a.mode || "auto")})</span>`;
    case "reaction_precedent_search":
    case "search_literature":
      return `${esc(a.product_smiles || "")}${a.reaction_class ? ` · class ${esc(a.reaction_class)}` : ""}`;
    case "inspect_molecule":
      return esc(a.smiles);
    case "pubchem_lookup":
      return esc(a.query);
    case "emit_routes": {
      const routes = a.submission?.routes;
      return Array.isArray(routes) ? `${routes.length} route tree${routes.length === 1 ? "" : "s"}` : "submission";
    }
    default:
      return esc(JSON.stringify(a).slice(0, 160));
  }
}

function resultLine(name, out) {
  if (out.error) return `<li class="err"><i class="dot"></i>${esc(out.error)}</li>`;
  const left =
    out.model_turns_remaining != null ? ` <span class="faint">· ${out.model_turns_remaining} turns left</span>` : "";
  switch (name) {
    case "validate_disconnection":
      return out.supported
        ? `<li class="ok"><i class="dot"></i>Cut supported <span class="faint mono">${esc(out.support)}</span> <span class="faint">${esc(out.reaction_class || "")}</span>${left}</li>`
        : `<li class="warn"><i class="dot"></i>Cut not supported <span class="faint mono">${esc(out.support)}</span>${left}</li>`;
    case "stock_retrieve":
      if ((out.mode || "exact") !== "exact")
        return `<li><i class="dot"></i>${fmt(out.returned)} results for ${esc(out.query)} (${esc(out.mode)})${left}</li>`;
      return out.results?.length
        ? `<li class="ok"><i class="dot"></i>In stock: <span class="mono">${esc(out.query)}</span>${left}</li>`
        : `<li class="warn"><i class="dot"></i>Not in stock: <span class="mono">${esc(out.query)}</span>${left}</li>`;
    case "reaction_precedent_search": {
      const best = out.results?.[0];
      return `<li><i class="dot"></i>${fmt(out.returned)} precedents${best ? `, closest Tanimoto ${num(best.similarity, 2)} (${esc(best.reaction_class)})` : ""}${left}</li>`;
    }
    case "inspect_molecule":
      return `<li><i class="dot"></i>${esc(out.formula)} · ${out.rings} rings · ${out.chiral_centres} stereocentres · ${num(out.molecular_weight, 1)} g/mol${left}</li>`;
    case "emit_routes":
      return `<li class="${out.score?.valid ? "ok" : "warn"}"><i class="dot"></i>Scored ${num(out.score?.reward)} · ${out.score?.valid ? "passed" : "did not pass"}</li>`;
    case "reaction_class_lookup":
      return `<li class="${out.forbidden_in_this_task ? "warn" : ""}"><i class="dot"></i>Class: ${esc(out.reaction_class)}${out.forbidden_in_this_task ? " · forbidden in this task" : ""}${left}</li>`;
    case "reaction_conditions_search":
      return `<li><i class="dot"></i>${fmt((out.conditions || []).length)} conditions (${esc(out.source || "")})${left}</li>`;
    case "search_literature":
      return `<li><i class="dot"></i>${fmt(out.returned)} citations${left}</li>`;
    default:
      return `<li><i class="dot"></i>${esc(JSON.stringify(out).slice(0, 160))}${left}</li>`;
  }
}

// --- task page: what the policy sees, the hidden known routes, and live play ---
const TEMPLATES = {
  inspect_molecule: (t) => ({ smiles: t }),
  pubchem_lookup: (t) => ({ query: t }),
  stock_retrieve: (t) => ({ query: t, mode: "exact", limit: 5 }),
  reaction_precedent_search: (t) => ({ product_smiles: t, limit: 5 }),
  validate_disconnection: (t) => ({ product_smiles: t, reactants: ["", ""] }),
  reaction_class_lookup: (t) => ({ product_smiles: t, reactants: ["", ""] }),
  reaction_conditions_search: (t) => ({ product_smiles: t, reactants: ["", ""], limit: 3 }),
  search_literature: (t) => ({ product_smiles: t, limit: 5 }),
  emit_routes: () => ({ submission: { routes: [] } }),
};

function constraintsHtml(task) {
  const c = task.constraints;
  const items = [];
  if (c.forbidden_classes.length) items.push(`Forbidden reaction classes: ${c.forbidden_classes.map(esc).join(", ")}`);
  if (c.excluded_stock.length)
    items.push(`Unavailable building blocks: ${c.excluded_stock.map((s) => `<span class="mono">${esc(s)}</span>`).join(", ")}`);
  if (task.min_routes > 1) items.push(`At least ${task.min_routes} routes with different first disconnections`);
  if (task.variant === "max_depth") items.push(`Depth capped at the shortest known route (${task.max_depth})`);
  return items.length ? items.map((i) => `<div>${i}</div>`).join("") : `<span class="faint">none</span>`;
}

async function taskView(view, id, params) {
  const t = await api("task", { benchmark: state.benchmark, id, reveal: params.reveal || 0 });
  const r = t.row,
    task = t.task,
    d = t.difficulty || {};
  const canReveal = !t.hidden;
  const routeTitle = (route, i) =>
    `${route.kind === "patent" ? "Patent route" : "Witness route"} ${i + 1}${route.patents.length ? ` <span class="faint sm">· ${route.patents.map(esc).join(", ")}</span>` : ""}`;
  view.innerHTML = `
    <div class="crumbs"><a href="#/tasks">RL tasks</a><span>/</span><span class="mono">${esc(id)}</span></div>
    <div class="split equal">
      <div class="panel"><div class="panel-h"><h2>What the policy sees</h2><span class="grow"></span><span class="faint sm mono">${esc(task.variant)}</span></div>
        <div class="panel-b"><div class="molbox">${mol(r.smiles, 360, 170)}</div>
          <dl class="kv" style="margin-top:12px">
            <dt>Target</dt><dd class="mono">${esc(r.smiles)}</dd>
            <dt>Split</dt><dd>${esc(r.split)}</dd>
            <dt>Depth budget</dt><dd>${task.max_depth} reactions on the longest linear sequence</dd>
            <dt>Routes asked for</dt><dd>${task.min_routes === task.max_routes ? task.max_routes : `${task.min_routes} to ${task.max_routes}`}</dd>
            <dt>Constraints</dt><dd>${constraintsHtml(task)}</dd>
            <dt>Stock</dt><dd>reachable only through stock_retrieve${t.target_in_stock ? " (the target itself is in stock)" : ""}</dd>
          </dl>
          <details style="margin-top:10px"><summary>Exact prompt</summary><pre style="margin-top:6px">${esc(t.prompt)}</pre></details>
          <h3 style="margin:14px 0 6px">Difficulty <span class="faint sm">(not shown to the policy)</span></h3>
          <dl class="kv">
            <dt>Tier</dt><dd>${esc(d.tier || "–")}</dd>
            <dt>Shortest known route</dt><dd>${d.min_depth ?? "–"} reactions${d.constrained_min_depth && d.constrained_min_depth !== d.min_depth ? ` (${d.constrained_min_depth} under this constraint)` : ""}; patent route ${d.patent_min_depth ?? "–"}</dd>
            <dt>Heavy atoms</dt><dd>${d.heavy_atoms ?? "–"}${d.stereocentres ? ` · ${d.stereocentres} stereocentre${d.stereocentres > 1 ? "s" : ""}` : ""}${d.complex_ring ? " · bridged, spiro or macro ring" : ""}</dd>
            <dt>Convergent known route</dt><dd>${d.convergent ? "yes" : "no"}</dd>
            <dt>Rarest template</dt><dd>${d.rarest_template_count != null ? `seen ${fmt(d.rarest_template_count)} times in the corpus` : "–"}</dd>
            <dt>Closest other-patent target</dt><dd>${d.nn_similarity != null ? `Tanimoto ${num(d.nn_similarity, 2)}` : "–"}</dd>
            ${t.siblings.length ? `<dt>Same target</dt><dd>${t.siblings.map((s) => `<a class="link mono" href="#/task/${encodeURIComponent(s)}">${esc(s.split("-").slice(1).join("-") || s)}</a>`).join(" · ")}</dd>` : ""}
          </dl>
        </div></div>
      <div class="panel"><div class="panel-h"><h2>Known routes, hidden from the policy</h2></div>
        <div class="panel-b">${
          t.hidden
            ? `<div class="note">This is a held-out ${esc(r.split)} task, so its routes stay hidden by default. <button class="link" id="reveal" type="button">Show them</button></div>`
            : `<p class="faint sm" style="margin-bottom:10px">Evidence for the similarity bonus only: the verifier accepts any route whose steps the reaction library supports. Witness routes recombine corpus reactions to prove the task is solvable under its constraint.</p>` +
              t.routes
                .map((route, i) => `<h3 style="margin:${i ? "16px" : "0"} 0 8px">${routeTitle(route, i)}</h3><div data-route="${i}"></div>`)
                .join("")
        }</div></div>
    </div>

    <div class="section">
      <h2>Try it <span class="faint sm">· play this task through the same core session the server runs</span></h2>
      <div class="panel" id="play"><div class="panel-b"><div class="toolbar" style="margin:0">
        <select id="toolset" aria-label="Toolset"><option value="full">full toolset</option><option value="unaided">unaided (no step checker)</option></select>
        <button class="btn primary" id="start" type="button">Start an episode</button>
        <span class="faint sm">The first start loads the reaction library and precedent index, about a minute.</span></div></div></div>
    </div>`;
  $("#reveal")?.addEventListener("click", () => go(`/task/${encodeURIComponent(id)}`, { ...params, reveal: 1 }));
  view.querySelectorAll("[data-route]").forEach((host) => mountRouteGraph(host, t.routes[host.dataset.route], r.smiles));
  $("#start").addEventListener("click", () =>
    startPlay($("#play"), id, r.smiles, $("#toolset").value, canReveal, params.reveal || 0),
  );
}

function rewardHtml(result) {
  const score = result?.score;
  if (!score) return "";
  const parts = Object.entries(score.components || {});
  return `<div class="panel" style="margin-top:10px"><div class="panel-h"><h3>Reward ${num(score.reward)}</h3><span class="${score.valid ? "ok" : "warn"} sm"><i class="dot"></i>${score.valid ? "passed" : "did not pass"}</span>
      <span class="faint sm">${esc(score.verification_tier || "")}</span></div>
    <div class="panel-b">${bars(parts.map(([k, v]) => [`${k} × ${num(META.weights[k], 2)}`, v, num(v, 2)]))}
      ${(score.hard_failures || []).length ? `<ul class="calls" style="margin-top:10px">${score.hard_failures.map((f) => `<li class="warn"><i class="dot"></i>${esc(f)}</li>`).join("")}</ul>` : ""}</div></div>`;
}

async function startPlay(panel, id, target, toolset, canReveal, reveal) {
  panel.innerHTML = `<div class="panel-b"><div class="loading-row"><span class="spinner"></span>Starting the episode</div></div>`;
  let session;
  try {
    session = await fetch("/api/session", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ benchmark: state.benchmark, task: id, toolset }),
    }).then((r) => r.json());
  } catch (error) {
    panel.innerHTML = `<div class="panel-b note err">${esc(error.message)}</div>`;
    return;
  }
  const tools = session.tools.map((tool) => tool.function.name);
  const cuts = await api("disconnections", { smi: target });
  panel.innerHTML = `
    <div class="panel-h"><h3>Episode on ${esc(id)}</h3><span class="grow"></span><span class="faint sm" id="budget">32 tool calls left</span></div>
    <div class="panel-b">
      <div class="split" style="grid-template-columns:minmax(0,1fr) minmax(0,1fr)">
        <div>
          <div class="toolbar" style="margin-bottom:8px"><select id="tool" aria-label="Tool">${tools.map((name) => `<option>${name}</option>`).join("")}</select>
            <button class="btn" id="call" type="button">Call</button>
            ${canReveal ? `<button class="btn" id="answer" type="button">Submit the known routes</button>` : ""}
            <button class="btn" id="empty" type="button">Submit an empty route</button></div>
          <textarea id="args" class="input" spellcheck="false" style="width:100%;height:120px;padding:8px 10px;font-family:var(--mono);font-size:12px"></textarea>
          <p class="faint xs" style="margin-top:4px">Arguments as JSON. Pick a tool to load a template.</p>
        </div>
        <div>
          <h3 style="margin-bottom:6px">Candidate cuts of the target <span class="faint sm">(rule library)</span></h3>
          ${cuts.length ? `<ul class="calls">${cuts.map((c, i) => `<li><button class="link" data-cut="${i}" type="button">${tools.includes("validate_disconnection") ? "validate" : "use"}</button> ${esc(c.text)} <span class="faint">· ${esc(c.bond)}</span></li>`).join("")}</ul>` : `<p class="faint sm">No rule matches this target.</p>`}
        </div>
      </div>
      <div class="turns" id="log" style="margin-top:12px"></div>
    </div>`;
  const args = $("#args", panel),
    tool = $("#tool", panel),
    log = $("#log", panel);
  const fill = () => {
    args.value = JSON.stringify((TEMPLATES[tool.value] || (() => ({})))(target), null, 2);
  };
  tool.addEventListener("change", fill);
  fill();
  const show = (call, outcome) => {
    const result = outcome.result || {};
    log.insertAdjacentHTML(
      "beforeend",
      `<div class="msg asst"><div class="role">you</div><ol class="calls"><li><span class="fn">${esc(call.tool)}</span>${callLabel(call.tool, call.arguments)}</li></ol>
      ${call.tool === "emit_routes" && call.arguments.submission?.routes?.length ? `<details><summary>Submitted route</summary><div class="stack" style="margin-top:8px">${call.arguments.submission.routes.map(submittedTree).join("")}</div></details>` : ""}</div>
      <div class="msg ctx"><div class="role">environment</div><ul class="calls">${resultLine(call.tool, result)}</ul>
      <details><summary>Raw</summary><pre>${esc(JSON.stringify(result, null, 2))}</pre></details>${call.tool === "emit_routes" ? rewardHtml(result) : ""}</div>`,
    );
    if (outcome.tool_calls_remaining != null)
      $("#budget", panel).textContent =
        `${outcome.tool_calls_remaining} tool calls left${outcome.done ? " · episode over" : ""}`;
    hydrate(log);
  };
  const post = async (path, body) =>
    (
      await fetch(path, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) })
    ).json();
  $("#call", panel).addEventListener("click", async () => {
    let parsed;
    try {
      parsed = JSON.parse(args.value || "{}");
    } catch (error) {
      log.insertAdjacentHTML(
        "beforeend",
        `<p class="note err">Arguments are not valid JSON: ${esc(error.message)}</p>`,
      );
      return;
    }
    show(
      { tool: tool.value, arguments: parsed },
      await post(`/api/session/${session.session}/call`, { tool: tool.value, arguments: parsed }),
    );
  });
  $("#empty", panel).addEventListener("click", async () => {
    const call = { tool: "emit_routes", arguments: { submission: { routes: [] } } };
    show(call, await post(`/api/session/${session.session}/call`, call));
  });
  $("#answer", panel)?.addEventListener("click", async () => {
    const outcome = await post(`/api/session/${session.session}/reference?reveal=${reveal}`, {});
    show({ tool: "emit_routes", arguments: outcome.arguments || {} }, outcome);
  });
  panel.querySelectorAll("[data-cut]").forEach((button) =>
    button.addEventListener("click", async () => {
      const cut = cuts[Number(button.dataset.cut)];
      const name = tools.includes("validate_disconnection") ? "validate_disconnection" : "stock_retrieve";
      if (name === "stock_retrieve") {
        for (const piece of cut.reactants) {
          const call = { tool: name, arguments: { query: piece, mode: "exact", limit: 1 } };
          show(call, await post(`/api/session/${session.session}/call`, call));
        }
        return;
      }
      const call = { tool: name, arguments: { product_smiles: target, reactants: cut.reactants } };
      show(call, await post(`/api/session/${session.session}/call`, call));
    }),
  );
}

// --- shell ---
const systemTheme = () => (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
const currentTheme = () => document.documentElement.dataset.theme || systemTheme();
function withoutTransitions(change) {
  const style = document.createElement("style");
  style.textContent = "*,*::before,*::after{transition:none !important}";
  document.head.append(style);
  change();
  getComputedStyle(document.body).color;
  requestAnimationFrame(() => requestAnimationFrame(() => style.remove()));
}
function applyTheme(theme) {
  withoutTransitions(() => {
    document.documentElement.dataset.theme = theme;
  });
  $("#theme").textContent = theme === "dark" ? "Light mode" : "Dark mode";
}
function initTheme() {
  const saved = store.get("theme", "");
  if (saved) document.documentElement.dataset.theme = saved;
  applyTheme(currentTheme());
  $("#theme").addEventListener("click", () => {
    const next = currentTheme() === "dark" ? "light" : "dark";
    applyTheme(next);
    store.set("theme", next);
  });
}

function setBenchmark(name, reroute = true) {
  state.benchmark = name;
  store.set("benchmark", name);
  $("#benchmark").value = name;
  if (reroute) route();
}

async function route() {
  const [path, query] = location.hash.replace(/^#/, "").split("?");
  const parts = (path || "/").split("/").filter(Boolean);
  const params = Object.fromEntries(new URLSearchParams(query || ""));
  const view = $("#view");
  const active = parts[0] === "task" ? "tasks" : parts[0] || "overview";
  document.querySelectorAll(".nav a").forEach((a) => a.classList.toggle("on", a.dataset.nav === active));
  view.innerHTML = `<div class="loading-row"><span class="spinner"></span>Loading</div>`;
  try {
    if (!parts.length) await overviewView(view, params);
    else if (parts[0] === "tasks") await tasksView(view, params);
    else if (parts[0] === "task") await taskView(view, decodeURIComponent(parts[1]), params);
    else if (parts[0] === "evals" && !parts[1]) await evalsView(view, params);
    else if (parts[0] === "evals" && parts[1] === "run") await evalRunView(view, params);
    else if (parts[0] === "evals" && parts[1] === "episode") await evalEpisodeView(view, params);
    else view.innerHTML = `<p class="note">Nothing here. <a class="link" href="#/">Back to the overview</a></p>`;
  } catch (error) {
    view.innerHTML = `<p class="note err">${esc(error.message)}</p>`;
  }
  hydrate(view);
  window.scrollTo(0, 0);
}

async function boot() {
  initTheme();
  META = await api("meta");
  const benchmarks = META.benchmarks.map((b) => b.name);
  $("#benchmark").innerHTML = benchmarks.map((b) => `<option>${esc(b)}</option>`).join("");
  const saved = store.get("benchmark", "");
  state.benchmark = benchmarks.includes(saved) ? saved : benchmarks[0];
  $("#benchmark").value = state.benchmark;
  $("#benchmark").addEventListener("change", (e) => setBenchmark(e.target.value));
  window.addEventListener("hashchange", route);
  route();
}
boot();
