"use strict";
// Answer-key route graph: dagre layout, panzoom pan and zoom, a minimap and an inspector, as plain DOM.

const FLOW = { molW: 172, molH: 158, rxn: 40, nodesep: 24, ranksep: 64, margin: 12, dots: 22, minHeight: 220, maxHeight: 520 };
const FLOW_LABELS = { target: "Target", intermediate: "Intermediate", stocked: "In stock", missing: "Not in stock" };
const FLOW_ROLES = { target: "Target", intermediate: "Intermediate", leaf: "Precursor" };
const FLOW_MARKERS = ["flow-arrow", "flow-arrow-on"]
  .map(
    (id) =>
      `<marker id="${id}" class="${id}" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M0,1 L10,5 L0,9 z"/></marker>`,
  )
  .join("");

function flowTone(node) {
  return node.kind === "leaf" ? (node.inStock ? "stocked" : "missing") : node.kind;
}

function buildRouteGraph(route, target) {
  const byProduct = Object.fromEntries(route.steps.map((s) => [s.product, s]));
  const nodes = [];
  const edges = [];
  const walk = (smiles, path) => {
    const step = path.has(smiles) ? null : byProduct[smiles];
    const kind = smiles === target ? "target" : step ? "intermediate" : "leaf";
    const product = { id: `m${nodes.length}`, type: "mol", smiles, step, kind, inStock: Boolean(route.in_stock[smiles]) };
    nodes.push({ ...product, w: FLOW.molW, h: FLOW.molH });
    if (!step) return product.id;
    const reaction = { id: `r${nodes.length}`, type: "rxn", step, w: FLOW.rxn, h: FLOW.rxn };
    nodes.push(reaction);
    edges.push({ from: reaction.id, to: product.id, rxn: reaction.id });
    const inner = new Set(path).add(smiles);
    step.reactants.forEach((r) => edges.push({ from: walk(r, inner), to: reaction.id, rxn: reaction.id }));
    return product.id;
  };
  walk(target, new Set());
  return { nodes, edges };
}

function layoutRouteGraph({ nodes, edges }) {
  const g = new dagre.graphlib.Graph();
  g.setGraph({ rankdir: "LR", nodesep: FLOW.nodesep, ranksep: FLOW.ranksep, marginx: FLOW.margin, marginy: FLOW.margin });
  g.setDefaultEdgeLabel(() => ({}));
  nodes.forEach((n) => g.setNode(n.id, { width: n.w, height: n.h }));
  edges.forEach((e) => g.setEdge(e.from, e.to));
  dagre.layout(g);
  nodes.forEach((n) => {
    const { x, y } = g.node(n.id);
    n.x = x - n.w / 2;
    n.y = y - n.h / 2;
  });
  const { width, height } = g.graph();
  return { width, height };
}

function smoothStep(x1, y1, x2, y2, radius = 10) {
  const dy = y2 - y1;
  if (Math.abs(dy) < 1) return `M${x1},${y1}H${x2}`;
  const mid = (x1 + x2) / 2;
  const s = Math.sign(dy);
  const r = Math.min(radius, Math.abs(dy) / 2);
  return `M${x1},${y1}H${mid - r}Q${mid},${y1} ${mid},${y1 + s * r}V${y2 - s * r}Q${mid},${y2} ${mid + r},${y2}H${x2}`;
}

function flowEdgesHtml({ nodes, edges }) {
  const byId = new Map(nodes.map((n) => [n.id, n]));
  return edges
    .map((e) => {
      const a = byId.get(e.from);
      const b = byId.get(e.to);
      const d = smoothStep(a.x + a.w, a.y + a.h / 2, b.x, b.y + b.h / 2);
      return `<g class="fedge" data-rxn="${e.rxn}"><path class="hit" d="${d}"/><path class="line" d="${d}"/></g>`;
    })
    .join("");
}

function flowNodeHtml(n) {
  const pos = `left:${n.x}px;top:${n.y}px;width:${n.w}px;height:${n.h}px`;
  if (n.type === "rxn")
    return `<div class="fnode frxn" data-id="${n.id}" style="${pos}" title="${esc(n.step.family)}"><i></i></div>`;
  const tone = flowTone(n);
  return `<div class="fnode fmol" data-id="${n.id}" style="${pos}">
    <div class="fimg">${mol(n.smiles, n.w - 28, 92)}</div>
    <div class="flabel${n.kind === "target" ? " strong" : ""}"><i class="fdot ${tone}"></i>${FLOW_LABELS[tone]}</div>
    <div class="fsmi mono" title="${esc(n.smiles)}">${esc(n.smiles)}</div></div>`;
}

function flowReactionDetail(step) {
  return `<dl class="kv"><dt>Class</dt><dd>${esc(step.family)}</dd><dt>Type</dt><dd>${esc(step.phrase)}</dd><dt>Precursors</dt><dd>${step.reactants.length}</dd></dl>
    <h3 class="fh">Rationale</h3><p class="sm">${esc(step.text)}</p>
    <h3 class="fh">Reactants</h3>${step.reactants.map((r) => `<p class="mono fsmiles">${esc(r)}</p>`).join("")}`;
}

function flowStockText(n) {
  if (n.inStock) return `<span class="ok"><i class="dot"></i>${n.step ? "in stock, but made in this route" : "in stock"}</span>`;
  return `<span class="${n.step ? "warn" : "err"}"><i class="dot"></i>${n.step ? "made in this route" : "not in stock"}</span>`;
}

function flowInspectorHtml(n) {
  const title = n.type === "rxn" ? "Disconnection" : FLOW_ROLES[n.kind];
  const body =
    n.type === "rxn"
      ? flowReactionDetail(n.step)
      : `<div class="fimg wide">${mol(n.smiles, 276, 150)}</div>
        <h3 class="fh">SMILES</h3><p class="mono fsmiles">${esc(n.smiles)}</p>
        <dl class="kv"><dt>Role</dt><dd>${FLOW_ROLES[n.kind]}</dd><dt>Stock</dt><dd>${flowStockText(n)}</dd></dl>
        ${n.step ? `<h3 class="fh">Made by this disconnection</h3>${flowReactionDetail(n.step)}` : ""}`;
  return `<header><h3>${title}</h3><button class="link" type="button" data-close>Close</button></header><div class="finsp-b">${body}</div>`;
}

function flowLegendHtml() {
  return Object.entries(FLOW_LABELS)
    .map(([tone, label]) => `<span><i class="fdot ${tone}"></i>${label}</span>`)
    .join("");
}

function flowStageHeight(host, size) {
  const fitted = size.height * Math.min(1, host.clientWidth / size.width);
  return Math.round(Math.min(Math.max(fitted + 24, FLOW.minHeight), FLOW.maxHeight));
}

function flowShellHtml(graph, size, height) {
  const minimapRects = graph.nodes
    .map((n) => `<rect class="${n.type}" x="${n.x}" y="${n.y}" width="${n.w}" height="${n.h}" rx="${n.type === "rxn" ? n.w / 2 : 8}"/>`)
    .join("");
  return `<div class="flow-stage" style="min-height:${height}px">
      <div class="flow-view"><div class="flow-world" style="width:${size.width}px;height:${size.height}px">
        <svg class="flow-edges" width="${size.width}" height="${size.height}"><defs>${FLOW_MARKERS}</defs>${flowEdgesHtml(graph)}</svg>
        ${graph.nodes.map(flowNodeHtml).join("")}</div></div>
      <div class="flow-legend fctl">${flowLegendHtml()}</div>
      <div class="flow-hint fctl">Drag to pan · Ctrl/⌘ + scroll to zoom · click a node for details</div>
      <div class="flow-controls fctl">
        <button type="button" data-zoom="1.25" aria-label="Zoom in">+</button>
        <button type="button" data-zoom="0.8" aria-label="Zoom out">−</button>
        <button type="button" data-fit aria-label="Fit to view">⤢</button></div>
      <svg class="flow-minimap fctl" viewBox="0 0 ${size.width} ${size.height}">${minimapRects}<path class="mask" fill-rule="evenodd"/></svg>
    </div>
    <aside class="flow-inspector" hidden></aside>`;
}

function createViewport(host, size) {
  const view = $(".flow-view", host);
  const pz = panzoom($(".flow-world", host), {
    minZoom: 0.1,
    maxZoom: 2,
    zoomDoubleClickSpeed: 1,
    beforeWheel: (e) => !(e.ctrlKey || e.metaKey),
  });
  const center = () => [view.clientWidth / 2, view.clientHeight / 2];
  return {
    view,
    pz,
    zoom: (ratio) => pz.zoomTo(...center(), ratio),
    fit() {
      const scale = Math.min(1, view.clientWidth / size.width, view.clientHeight / size.height);
      pz.moveTo(0, 0);
      pz.zoomAbs(0, 0, scale);
      pz.moveTo((view.clientWidth - size.width * scale) / 2, (view.clientHeight - size.height * scale) / 2);
    },
  };
}

function syncBackground(view, { x, y, scale }) {
  const gap = FLOW.dots * scale;
  view.style.backgroundSize = `${gap}px ${gap}px`;
  view.style.backgroundPosition = `${x}px ${y}px`;
}

function syncMinimap(host, { view, pz }) {
  const { x, y, scale } = pz.getTransform();
  const [vx, vy, vw, vh] = [-x / scale, -y / scale, view.clientWidth / scale, view.clientHeight / scale];
  $(".flow-minimap .mask", host).setAttribute(
    "d",
    `M-1e5,-1e5H1e5V1e5H-1e5Z M${vx},${vy}h${vw}v${vh}h${-vw}Z`,
  );
}

function wireMinimap(host, { view, pz }) {
  const svg = $(".flow-minimap", host);
  const panTo = (e) => {
    const point = Object.assign(svg.createSVGPoint(), { x: e.clientX, y: e.clientY }).matrixTransform(
      svg.getScreenCTM().inverse(),
    );
    const { scale } = pz.getTransform();
    pz.moveTo(view.clientWidth / 2 - point.x * scale, view.clientHeight / 2 - point.y * scale);
  };
  svg.addEventListener("pointerdown", (e) => {
    svg.setPointerCapture(e.pointerId);
    panTo(e);
  });
  svg.addEventListener("pointermove", (e) => svg.hasPointerCapture(e.pointerId) && panTo(e));
}

function wireSelection(host, graph, view) {
  const aside = $(".flow-inspector", host);
  const byId = new Map(graph.nodes.map((n) => [n.id, n]));
  const select = (id) => {
    const node = byId.get(id);
    host.querySelectorAll(".fnode").forEach((el) => el.classList.toggle("on", el.dataset.id === id));
    host.querySelectorAll(".fedge").forEach((el) => el.classList.toggle("on", el.dataset.rxn === id));
    aside.hidden = !node;
    if (!node) return;
    aside.innerHTML = flowInspectorHtml(node);
    hydrate(aside);
  };
  let down = [0, 0];
  view.addEventListener("pointerdown", (e) => (down = [e.clientX, e.clientY]));
  view.addEventListener("click", (e) => {
    if (Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4) return;
    select(e.target.closest(".fnode")?.dataset.id ?? e.target.closest(".fedge")?.dataset.rxn);
  });
  aside.addEventListener("click", (e) => e.target.closest("[data-close]") && select(null));
}

// The stage can stretch after mount (e.g. to match a sibling card), so refit when its height changes.
function refitOnHeightChange(viewport) {
  let height = viewport.view.clientHeight;
  new ResizeObserver(() => {
    if (viewport.view.clientHeight === height) return;
    height = viewport.view.clientHeight;
    viewport.fit();
  }).observe(viewport.view);
}

function mountRouteGraph(host, route, target) {
  const graph = buildRouteGraph(route, target);
  const size = layoutRouteGraph(graph);
  host.classList.add("flow");
  host.innerHTML = flowShellHtml(graph, size, flowStageHeight(host, size));
  const viewport = createViewport(host, size);
  const onChange = () => {
    syncBackground(viewport.view, viewport.pz.getTransform());
    syncMinimap(host, viewport);
  };
  viewport.pz.on("transform", onChange);
  host.querySelectorAll("[data-zoom]").forEach((b) => b.addEventListener("click", () => viewport.zoom(Number(b.dataset.zoom))));
  $("[data-fit]", host).addEventListener("click", viewport.fit);
  wireMinimap(host, viewport);
  wireSelection(host, graph, viewport.view);
  viewport.fit();
  onChange();
  refitOnHeightChange(viewport);
  hydrate(host);
}
