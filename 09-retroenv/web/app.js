(() => {
  "use strict";

  const data = window.RETRO_DATA;
  if (!data) {
    document.body.innerHTML = "<p style='padding:2rem'>Data bundle missing. Run <code>python web/build_data.py</code>.</p>";
    return;
  }

  const $ = (selector) => document.querySelector(selector);
  const SVG_NS = "http://www.w3.org/2000/svg";
  const XHTML_NS = "http://www.w3.org/1999/xhtml";
  const state = {
    modelKey: data.models[0].key,
    taskId: data.tasks[0].task_id,
    routeIndex: 0,
  };

  const pct = (value, digits = 0) => `${(Number(value || 0) * 100).toFixed(digits)}%`;
  const money = (value) => `$${Number(value || 0).toFixed(value < 0.01 ? 3 : 2)}`;
  const seconds = (value) => `${Number(value || 0).toFixed(1)}s`;
  const compact = (value, max = 74) => {
    const text = String(value ?? "");
    return text.length > max ? `${text.slice(0, max - 1)}…` : text;
  };
  const currentModel = () => data.models.find((model) => model.key === state.modelKey);
  const currentTask = () => data.tasks.find((task) => task.task_id === state.taskId);
  const currentEpisode = () => currentModel().episodes[state.taskId];

  function scoreClass(value) {
    return Number(value || 0) === 0 ? "score-zero" : "score-strong";
  }

  function episodeStatus(episode) {
    if (episode.valid) return { label: "Strict pass", className: "valid" };
    if (episode.metrics.exact_reference_match) return { label: "Exact route only", className: "exact" };
    return { label: "Rejected", className: "failed" };
  }

  function renderPrompt() {
    const task = data.tasks[0];
    $("#promptExample").textContent = data.meta.prompt_template
      .replace("{target}", task.target_smiles)
      .replace("{max_steps}", task.max_steps)
      .replace("{min_routes}", task.min_routes)
      .replace("{max_routes}", task.max_routes);
  }

  function renderReward() {
    const bar = $("#rewardBar");
    const legend = $("#rewardLegend");
    bar.replaceChildren();
    legend.replaceChildren();
    data.meta.reward_parts.forEach((part) => {
      const segment = document.createElement("div");
      segment.className = "reward-segment";
      segment.style.width = `${part.weight * 100}%`;
      segment.textContent = part.weight >= 0.1 ? `${part.weight * 100}%` : "";
      segment.title = `${part.label}: ${part.weight * 100}% — ${part.description}`;
      bar.appendChild(segment);

      const item = document.createElement("div");
      item.className = "legend-item";
      const dot = document.createElement("span");
      dot.className = "legend-dot";
      const label = document.createElement("span");
      label.textContent = part.label;
      const weight = document.createElement("b");
      weight.textContent = `${part.weight * 100}%`;
      item.append(dot, label, weight);
      legend.appendChild(item);
    });
  }

  function renderLeaderboard() {
    const body = $("#leaderboardBody");
    body.replaceChildren();
    data.models.forEach((model) => {
      const s = model.scores;
      const row = document.createElement("tr");
      row.tabIndex = 0;
      row.setAttribute("aria-label", `Explore ${model.label}`);
      const cells = [
        null,
        pct(s["pass@1"]),
        pct(s.exact_reference_match),
        pct(s.mean_reward),
        pct(s.graph_validity),
        pct(s.step_validity),
        pct(s.stock_completion),
        pct(s.route_diversity),
        pct(s.invalid_proposal_rate),
        money(s.reported_cost_usd),
      ];
      const modelCell = document.createElement("td");
      const modelWrap = document.createElement("div");
      modelWrap.className = "model-cell";
      const rank = document.createElement("span");
      rank.className = "rank";
      rank.textContent = model.rank;
      const name = document.createElement("span");
      name.textContent = model.label;
      modelWrap.append(rank, name);
      modelCell.appendChild(modelWrap);
      row.appendChild(modelCell);
      cells.slice(1).forEach((value, index) => {
        const cell = document.createElement("td");
        cell.textContent = value;
        if (index < 8) cell.className = scoreClass(parseFloat(value) / 100);
        row.appendChild(cell);
      });
      const activate = () => {
        state.modelKey = model.key;
        state.routeIndex = 0;
        renderExplorer();
        $("#explorer").scrollIntoView({ behavior: "smooth" });
      };
      row.addEventListener("click", activate);
      row.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          activate();
        }
      });
      body.appendChild(row);
    });
  }

  function renderModelTabs() {
    const tabs = $("#modelTabs");
    tabs.replaceChildren();
    data.models.forEach((model) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "model-tab";
      button.setAttribute("role", "tab");
      button.setAttribute("aria-selected", String(model.key === state.modelKey));
      const name = document.createElement("span");
      name.textContent = model.label;
      const score = document.createElement("small");
      score.textContent = `${pct(model.scores["pass@1"])} strict · ${pct(model.scores.exact_reference_match)} exact`;
      button.append(name, score);
      button.addEventListener("click", () => {
        state.modelKey = model.key;
        state.routeIndex = 0;
        renderExplorer();
      });
      tabs.appendChild(button);
    });
  }

  function renderTaskList() {
    const list = $("#taskList");
    const model = currentModel();
    list.replaceChildren();
    data.tasks.forEach((task, index) => {
      const episode = model.episodes[task.task_id];
      const status = episodeStatus(episode);
      const button = document.createElement("button");
      button.type = "button";
      button.className = "task-button";
      button.setAttribute("aria-selected", String(task.task_id === state.taskId));
      button.setAttribute("aria-label", `Task ${index + 1}, ${status.label}, reward ${pct(episode.reward)}`);

      const number = document.createElement("span");
      number.className = "task-index";
      number.textContent = String(index + 1).padStart(2, "0");
      const text = document.createElement("span");
      text.className = "task-name";
      const target = document.createElement("strong");
      target.textContent = task.target_smiles;
      const score = document.createElement("small");
      score.textContent = `${pct(episode.reward)} reward · ${episode.metrics.route_count || 0} routes`;
      text.append(target, score);
      const dot = document.createElement("span");
      dot.className = `task-dot ${status.className}`;
      button.append(number, text, dot);
      button.addEventListener("click", () => {
        state.taskId = task.task_id;
        state.routeIndex = 0;
        renderExplorer();
      });
      list.appendChild(button);
    });
  }

  function renderEpisodeHeader() {
    const model = currentModel();
    const task = currentTask();
    const episode = currentEpisode();
    const index = data.tasks.findIndex((item) => item.task_id === task.task_id) + 1;
    const status = episodeStatus(episode);
    $("#episodeLabel").textContent = `${model.label} · Task ${String(index).padStart(2, "0")}`;
    $("#targetSmiles").textContent = task.target_smiles;
    $("#taskMeta").textContent = [
      `${task.max_steps}-step cap`,
      `${task.min_routes} routes required`,
      `depth ${task.difficulty.depth}`,
      task.difficulty.stereochemistry ? "stereochemistry" : "no stereochemistry",
      `${episode.tool_calls} tool calls`,
      seconds(episode.usage.latency),
    ].join(" · ");
    const statusPill = $("#episodeStatus");
    statusPill.className = `status-pill ${status.className}`;
    statusPill.textContent = status.label;
  }

  function renderEpisodeStats() {
    const episode = currentEpisode();
    const values = [
      ["Reward", pct(episode.reward)],
      ["Strict", episode.valid ? "Pass" : "Fail"],
      ["Exact route", episode.metrics.exact_reference_match ? "Yes" : "No"],
      ["Graph", pct(episode.metrics.graph_validity)],
      ["Steps", pct(episode.metrics.step_validity)],
      ["Stock", pct(episode.metrics.building_block_completion)],
    ];
    const box = $("#episodeStats");
    box.replaceChildren();
    values.forEach(([label, value]) => {
      const item = document.createElement("div");
      item.className = "stat";
      const key = document.createElement("span");
      key.textContent = label;
      const score = document.createElement("strong");
      score.textContent = value;
      item.append(key, score);
      box.appendChild(item);
    });
  }

  function renderRouteTabs() {
    const episode = currentEpisode();
    const tabs = $("#routeTabs");
    tabs.replaceChildren();
    episode.routes.forEach((route, index) => {
      const result = episode.route_results[index] || {};
      const button = document.createElement("button");
      button.type = "button";
      button.className = "route-tab";
      button.setAttribute("role", "tab");
      button.setAttribute("aria-selected", String(index === state.routeIndex));
      button.textContent = `Route ${index + 1}${result.valid ? " ✓" : ""}`;
      button.addEventListener("click", () => {
        state.routeIndex = index;
        renderRouteArea();
      });
      tabs.appendChild(button);
    });
  }

  function addSummaryChip(container, text, kind = "") {
    const chip = document.createElement("span");
    chip.className = `summary-chip ${kind}`.trim();
    chip.textContent = text;
    container.appendChild(chip);
  }

  function renderRouteSummary() {
    const episode = currentEpisode();
    const result = episode.route_results[state.routeIndex] || {};
    const summary = $("#routeSummary");
    summary.replaceChildren();
    if (!episode.routes.length) {
      addSummaryChip(summary, "No route emitted", "bad");
      return;
    }
    addSummaryChip(summary, result.valid ? "Verifier passed" : "Verifier rejected", result.valid ? "good" : "bad");
    addSummaryChip(summary, result.exact_reference_match ? "Exact reference match" : `Similarity ${pct(result.reference_similarity)}`, result.exact_reference_match ? "good" : "");
    addSummaryChip(summary, `Steps ${pct(result.step_validity)}`);
    addSummaryChip(summary, `Stock claims ${pct(result.stock_claim_accuracy)}`);
    (result.hard_failures || []).slice(0, 2).forEach((failure) => addSummaryChip(summary, compact(failure, 80), "bad"));
  }

  function hierarchy(root) {
    let nextId = 0;
    let nextLeaf = 0;
    let maxDepth = 0;
    const nodes = [];
    const edges = [];

    function walk(value, depth, parentId = null) {
      const node = value && typeof value === "object" ? value : {};
      const id = `node-${nextId++}`;
      const isReaction = node.type === "reaction" || node.is_reaction === true;
      const children = Array.isArray(node.children) ? node.children : [];
      const item = { id, node, depth, isReaction, children: [] };
      nodes.push(item);
      maxDepth = Math.max(maxDepth, depth);
      if (parentId) edges.push({ source: parentId, target: id });
      item.children = children.map((child) => walk(child, depth + 1, id));
      if (!item.children.length) {
        item.y = 58 + nextLeaf++ * 118;
      } else {
        item.y = item.children.reduce((sum, child) => sum + child.y, 0) / item.children.length;
      }
      item.x = 42 + depth * 246;
      item.width = isReaction ? 206 : 194;
      item.height = isReaction ? 88 : 68;
      return item;
    }

    const tree = walk(root, 0);
    return {
      tree,
      nodes,
      edges,
      width: Math.max(680, 42 + maxDepth * 246 + 236),
      height: Math.max(340, 116 + Math.max(0, nextLeaf - 1) * 118),
    };
  }

  function svgElement(name, attrs = {}) {
    const element = document.createElementNS(SVG_NS, name);
    Object.entries(attrs).forEach(([key, value]) => element.setAttribute(key, value));
    return element;
  }

  function nodeTitle(item) {
    if (item.isReaction) {
      const meta = item.node.metadata || {};
      return meta.reaction_class || meta.classification || "Reaction";
    }
    return item.node.smiles || "Unknown molecule";
  }

  function selectGraphNode(group, item) {
    document.querySelectorAll(".graph-node.selected").forEach((node) => node.classList.remove("selected"));
    group.classList.add("selected");
    renderNodeDetail(item);
  }

  function renderNodeDetail(item) {
    const box = $("#nodeDetail");
    box.replaceChildren();
    const grid = document.createElement("div");
    grid.className = "node-detail-grid";
    const main = document.createElement("div");
    const side = document.createElement("div");
    if (item.isReaction) {
      const meta = item.node.metadata || {};
      const heading = document.createElement("strong");
      heading.textContent = meta.reaction_class || meta.classification || "Reaction";
      const explanation = document.createElement("p");
      explanation.textContent = meta.explanation || "No explanation supplied.";
      main.append(heading, explanation);
      const confidence = document.createElement("p");
      confidence.innerHTML = `<strong>Confidence</strong> ${Math.round(Number(meta.confidence || 0) * 100)}%`;
      side.appendChild(confidence);
      const roles = meta.precursor_roles || {};
      if (Object.keys(roles).length) {
        const roleText = document.createElement("p");
        roleText.textContent = Object.entries(roles).map(([smiles, role]) => `${role}: ${compact(smiles, 30)}`).join(" · ");
        side.appendChild(roleText);
      }
    } else {
      const heading = document.createElement("strong");
      heading.textContent = item.node.in_stock ? "Stock-confirmed molecule" : "Molecule";
      const smiles = document.createElement("p");
      const code = document.createElement("code");
      code.textContent = item.node.smiles || "Missing SMILES";
      smiles.appendChild(code);
      main.append(heading, smiles);
      const status = document.createElement("p");
      status.textContent = item.node.in_stock ? "Exact stock hit" : "Not claimed in stock";
      side.appendChild(status);
    }
    grid.append(main, side);
    box.appendChild(grid);
  }

  function renderGraph() {
    const svg = $("#routeGraph");
    const episode = currentEpisode();
    const route = episode.routes[state.routeIndex];
    while (svg.lastChild && !["title", "desc"].includes(svg.lastChild.tagName)) svg.removeChild(svg.lastChild);

    if (!route) {
      svg.setAttribute("viewBox", "0 0 680 340");
      svg.style.width = "100%";
      svg.style.height = "340px";
      const text = svgElement("text", { x: 340, y: 170, "text-anchor": "middle", fill: "#68726c", "font-size": 13 });
      text.textContent = "No route graph was emitted.";
      svg.appendChild(text);
      $("#nodeDetail").textContent = "The episode ended without a renderable route tree.";
      return;
    }

    const layout = hierarchy(route);
    svg.setAttribute("viewBox", `0 0 ${layout.width} ${layout.height}`);
    svg.style.width = `${layout.width}px`;
    svg.style.height = `${layout.height}px`;

    const defs = svgElement("defs");
    const marker = svgElement("marker", { id: "arrow-end", markerWidth: 8, markerHeight: 8, refX: 7, refY: 4, orient: "auto", markerUnits: "strokeWidth" });
    const path = svgElement("path", { d: "M0,0 L8,4 L0,8 Z", fill: "#aeb8b0" });
    marker.appendChild(path);
    defs.appendChild(marker);
    svg.appendChild(defs);

    const byId = Object.fromEntries(layout.nodes.map((item) => [item.id, item]));
    layout.edges.forEach((edge) => {
      const source = byId[edge.source];
      const target = byId[edge.target];
      const sx = source.x + source.width;
      const sy = source.y;
      const tx = target.x;
      const ty = target.y;
      const bend = (tx - sx) * 0.5;
      const line = svgElement("path", {
        class: "graph-edge",
        d: `M ${sx} ${sy} C ${sx + bend} ${sy}, ${tx - bend} ${ty}, ${tx} ${ty}`,
        "marker-end": "url(#arrow-end)",
      });
      svg.appendChild(line);
    });

    let firstGroup = null;
    layout.nodes.forEach((item) => {
      const group = svgElement("g", {
        class: `graph-node ${item.isReaction ? "reaction" : "molecule"}${item.node.in_stock ? " stock" : ""}`,
        transform: `translate(${item.x},${item.y - item.height / 2})`,
        role: "button",
        tabindex: "0",
        "aria-label": nodeTitle(item),
      });
      const rect = svgElement("rect", { width: item.width, height: item.height, rx: 11, ry: 11 });
      const foreign = svgElement("foreignObject", { width: item.width, height: item.height });
      const card = document.createElementNS(XHTML_NS, "div");
      card.className = "graph-node-card";
      const kind = document.createElement("span");
      kind.className = "graph-node-kind";
      kind.textContent = item.isReaction ? "Reaction" : item.node.in_stock ? "Molecule · in stock" : "Molecule";
      const title = document.createElement("span");
      title.className = "graph-node-title";
      title.textContent = nodeTitle(item);
      card.append(kind, title);
      if (item.isReaction) {
        const meta = item.node.metadata || {};
        const copy = document.createElement("span");
        copy.className = "graph-node-copy";
        copy.textContent = meta.explanation || "No rationale supplied";
        card.appendChild(copy);
      } else if (item.node.in_stock) {
        const badge = document.createElement("span");
        badge.className = "graph-node-badge";
        badge.textContent = "EXACT STOCK HIT";
        card.appendChild(badge);
      }
      foreign.appendChild(card);
      group.append(rect, foreign);
      const activate = () => selectGraphNode(group, item);
      group.addEventListener("click", activate);
      group.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          activate();
        }
      });
      svg.appendChild(group);
      if (!firstGroup) firstGroup = { group, item };
    });
    if (firstGroup) selectGraphNode(firstGroup.group, firstGroup.item);
  }

  function renderRouteArea() {
    renderRouteTabs();
    renderRouteSummary();
    renderGraph();
  }

  function toolState(step) {
    const result = step.result || {};
    if (result.error || (Array.isArray(result.errors) && result.errors.length) || result.valid === false) return { label: "Issue", className: "bad" };
    if (result.valid === true || result.support === "dataset_exact" || step.tool === "emit_routes") return { label: "Verified", className: "good" };
    return { label: "Returned", className: "" };
  }

  function factList(step) {
    const args = step.arguments || {};
    const result = step.result || {};
    const facts = [];
    if (args.smiles) facts.push(compact(args.smiles, 54));
    if (args.query) facts.push(`${args.mode || "query"}: ${compact(args.query, 50)}`);
    if (args.product_smiles) facts.push(`product: ${compact(args.product_smiles, 42)}`);
    if (Array.isArray(args.reactants)) facts.push(`${args.reactants.length} precursor${args.reactants.length === 1 ? "" : "s"}`);
    if (args.reaction_class) facts.push(compact(args.reaction_class, 40));
    if (Number.isFinite(args.route_count)) facts.push(`${args.route_count} routes emitted`);
    if (result.support) facts.push(`support: ${result.support}`);
    if (Number.isFinite(result.returned)) facts.push(`${result.returned} returned`);
    if (result.formula) facts.push(`${result.formula} · ${Number(result.molecular_weight || 0).toFixed(1)} Da`);
    if (result.classification || result.reaction_class) facts.push(compact(result.classification || result.reaction_class, 44));
    if (Number.isFinite(result.reward)) facts.push(`${pct(result.reward)} reward`);
    if (result.error) facts.push(compact(result.error, 64));
    return facts.slice(0, 5);
  }

  function renderTrajectory() {
    const episode = currentEpisode();
    const list = $("#trajectoryList");
    list.replaceChildren();
    $("#trajectoryCount").textContent = `${episode.trajectory.length} tool turns · ${episode.invalid_proposals} invalid proposals`;

    episode.trajectory.forEach((step) => {
      const stateInfo = toolState(step);
      const item = document.createElement("li");
      item.className = "trajectory-item";
      const number = document.createElement("span");
      number.className = "turn-number";
      number.textContent = String(step.turn).padStart(2, "0");
      const body = document.createElement("div");
      body.className = "trajectory-body";
      const top = document.createElement("div");
      top.className = "trajectory-top";
      const tool = document.createElement("span");
      tool.className = "tool-name";
      tool.textContent = step.tool;
      const badge = document.createElement("span");
      badge.className = `tool-state ${stateInfo.className}`.trim();
      badge.textContent = stateInfo.label;
      top.append(tool, badge);
      body.appendChild(top);
      if (step.note) {
        const note = document.createElement("p");
        note.className = "trajectory-note";
        note.textContent = step.note;
        body.appendChild(note);
      }
      const facts = factList(step);
      if (facts.length) {
        const factBox = document.createElement("div");
        factBox.className = "trajectory-facts";
        facts.forEach((fact) => {
          const tag = document.createElement("code");
          tag.textContent = fact;
          factBox.appendChild(tag);
        });
        body.appendChild(factBox);
      }
      const details = document.createElement("details");
      details.className = "trajectory-details";
      const summary = document.createElement("summary");
      summary.textContent = "Arguments and result";
      const pre = document.createElement("pre");
      pre.textContent = JSON.stringify({ arguments: step.arguments, result: step.result }, null, 2);
      details.append(summary, pre);
      body.appendChild(details);
      item.append(number, body);
      list.appendChild(item);
    });
  }

  function renderExplorer() {
    renderModelTabs();
    renderTaskList();
    renderEpisodeHeader();
    renderEpisodeStats();
    renderRouteArea();
    renderTrajectory();
  }

  renderPrompt();
  renderReward();
  renderLeaderboard();
  renderExplorer();
})();
