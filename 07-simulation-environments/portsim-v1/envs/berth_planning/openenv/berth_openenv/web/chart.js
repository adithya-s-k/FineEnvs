// Berth chart (time-space diagram): x = hours, y = quay sections (port numbering).
// One rectangle per ship; closures/alongside blocks hatched; published slot as a dashed ghost;
// delay tail in red from due to departure; conflicts outlined red; a vertical line for the current hour.
import { callKinds, clockAt, cranePoolAt, departureAfter, divertWindows, escapeHtml, hasCranes, movesOf, publishedSlot } from "./model.js";

const NS = "http://www.w3.org/2000/svg";
let uid = 0;

function el(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v != null) e.setAttribute(k, String(v));
  if (parent) parent.appendChild(e);
  return e;
}

export class BerthChart {
  constructor(container, { mini = false, onHover, onPick, onScrub, maxHeight, editable = false, onMove, onCranes, fill = false } = {}) {
    this.container = container;
    this.onCranes = onCranes;
    this.mini = mini;
    this.fill = fill;
    this.editable = editable && !mini;
    this.onMove = onMove;
    this.onHover = onHover;
    this.onPick = onPick;
    this.onScrub = onScrub;
    this.maxHeight = maxHeight;
    this.id = `bc${++uid}`;
    this.time = 0;
    this.hover = null;
    this.selected = null;
    this.svg = el("svg", { class: mini ? "bchart mini" : this.editable ? "bchart edit" : "bchart", role: "img" });
    container.appendChild(this.svg);
    if (this.editable) {
      container.tabIndex = 0;
      container.dataset.keys = "chart";
      container.setAttribute("aria-label", "Plan editor: drag a ship to change its docking hour (left/right) and first section (up/down); arrow keys nudge the selected ship; + and - change its cranes");
    }
    this.ro = new ResizeObserver(() => {
      const w = container.clientWidth;
      if (!w || Math.abs(w - (this._w || 0)) < 1) return;
      cancelAnimationFrame(this._rf);
      this._rf = requestAnimationFrame(() => this.render());
    });
    this.ro.observe(container);
    this._bind();
  }

  destroy() {
    this.ro.disconnect();
    this.svg.remove();
  }

  setData(task, evaluation, horizon) {
    this.task = task;
    this.ev = evaluation;
    this.H = Math.max(1, horizon);
    this.render();
  }

  setMaxHeight(h) {
    if (h != null && this.maxHeight != null && Math.abs(h - this.maxHeight) < 1) return;
    this.maxHeight = h;
    this.render();
  }

  setTime(t) {
    this.time = t;
    if (this.nowLine && this.geom) {
      const x = this.geom.x(Math.max(0, Math.min(this.H, t)));
      this.nowLine.setAttribute("x1", x);
      this.nowLine.setAttribute("x2", x);
      if (this.nowHead) this.nowHead.setAttribute("transform", `translate(${x},${this.geom.top - 1})`);
    }
  }

  setHover(id) {
    this.hover = id;
    this._marks();
  }

  setSelected(id) {
    this.selected = id;
    this._marks();
  }

  _marks() {
    if (!this.shipEls) return;
    for (const [id, g] of this.shipEls) {
      g.classList.toggle("hov", this.hover === id);
      g.classList.toggle("sel", this.selected === id);
    }
  }

  render() {
    const task = this.task;
    const svg = this.svg;
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    if (!task || !this.ev) return;
    const W = Math.max(200, this.container.clientWidth);
    this._w = this.container.clientWidth;
    const n = task.last_section - task.first_section + 1;
    const mini = this.mini;
    const cr = hasCranes(task);
    const rules = task.rules || {};
    const winds = (rules.no_moves || []).slice().sort((a, b) => (b.min_length || 0) - (a.min_length || 0));
    const outages = rules.crane_outages || [];
    const diverts = divertWindows(task);
    const lane = !mini && (diverts.length || winds.length || outages.length);
    const left = mini ? 2 : W < 520 ? 24 : 30;
    const right = mini ? 2 : 8;
    const top = mini ? 2 : lane ? 46 : 20;
    const strips = cr && !mini && this.ev.series;
    const STRIP_C = 28;
    const STRIP_M = 20;
    const stripsH = strips ? 8 + STRIP_C + 6 + STRIP_M : 0;
    const bottom = mini ? 2 : 4;
    let rowH = mini ? Math.max(2, Math.min(4, 84 / n)) : this.editable ? (W < 560 ? 11 : 15) : W < 560 ? 9 : 12;
    if (!mini && this.maxHeight) {
      const fit = Math.floor((this.maxHeight - top - bottom - stripsH) / n);
      rowH = this.fill ? Math.max(8, Math.min(18, fit)) : Math.max(6, Math.min(rowH, fit));
    }
    const plotB = top + n * rowH;
    const Hpx = plotB + stripsH + bottom;
    svg.setAttribute("viewBox", `0 0 ${W} ${Hpx}`);
    svg.setAttribute("width", W);
    svg.setAttribute("height", Hpx);
    const H = this.H;
    const x = (h) => left + (h / H) * (W - left - right);
    const y = (s) => top + (s - task.first_section) * rowH;
    this.geom = { x, y, top, bottom: plotB, left, right: W - right, W, rowH, n };

    const defs = el("defs", {}, svg);
    const pat = (name, cls, size = 5) => {
      const p = el("pattern", { id: `${this.id}-${name}`, patternUnits: "userSpaceOnUse", width: size, height: size, patternTransform: "rotate(45)" }, defs);
      el("rect", { width: size, height: size, class: `${cls}-bg` }, p);
      el("line", { x1: 0, y1: 0, x2: 0, y2: size, class: `${cls}-ln` }, p);
    };
    pat("closed", "pc");
    pat("along", "pa");
    pat("divert", "pd");
    pat("hold", "ph", 4);
    pat("out", "po", 4);
    const kinds = callKinds(task);

    // grid: day bands, 6 h ticks, section lines
    const grid = el("g", { class: "grid" }, svg);
    for (let d = 0; d * 24 < H; d++) {
      const x0 = x(d * 24);
      const x1 = x(Math.min(H, (d + 1) * 24));
      if (d % 2 === 1) el("rect", { x: x0, y: top, width: x1 - x0, height: n * rowH, class: "band" }, grid);
      if (!mini) {
        const c = clockAt(task, d * 24);
        const label = x1 - x0 > 54 ? `${c.day} ${c.date.split(" ")[0]}` : c.day;
        if (x1 - x0 > 22) el("text", { x: (x0 + x1) / 2, y: 13, class: "ax ax-day", "text-anchor": "middle" }, grid).textContent = label;
      }
    }
    // wind windows: vertical bands through the whole chart (and the strips)
    for (const w of winds) {
      const all = !(w.min_length > 0);
      const wx0 = x(Math.max(0, w.start));
      const ww = Math.max(1, x(Math.min(H, w.end)) - wx0);
      const g = el("g", { class: `wind ${all ? "w-all" : "w-long"}` }, grid);
      el("rect", { x: wx0, y: mini ? 0 : top, width: ww, height: (mini ? Hpx : plotB + stripsH - top), class: "wind-r" }, g);
      el("title", {}, g).textContent = `Hours ${w.start}-${w.end}: ${w.reason || "wind"}: ${all ? "no ship" : `no ship of ${w.min_length} m or more`} may dock or leave. Cranes keep working; a ship that finishes inside waits alongside.`;
    }
    if (!mini) {
      for (let h = 6; h < H; h += 6) {
        el("line", { x1: x(h), x2: x(h), y1: top, y2: plotB, class: h % 24 === 0 ? "vday" : "v6" }, grid);
      }
      const every = rowH >= 11 ? 1 : 2;
      for (let sct = task.first_section; sct <= task.last_section; sct++) {
        if (sct > task.first_section) el("line", { x1: left, x2: W - right, y1: y(sct), y2: y(sct), class: "hsec" }, grid);
        if ((sct - task.first_section) % every === 0) el("text", { x: left - 4, y: y(sct) + rowH / 2 + 3, class: "ax ax-sec", "text-anchor": "end" }, grid).textContent = sct;
      }
      el("rect", { x: left, y: top, width: W - left - right, height: n * rowH, class: "frame" }, grid);
      // event lane above the plot: diverts, crane outages, wind labels
      const laneY = top - 9;
      const laneLabels = [];
      const laneText = (x0, x1, txt, cls) => laneLabels.push({ x0, x1, txt, cls });
      for (const d of diverts) {
        const dx0 = x(Math.max(0, d.start));
        const dw = Math.max(1, x(Math.min(H, d.end)) - dx0);
        const g = el("g", { class: "divert" }, grid);
        el("rect", { x: dx0, y: laneY, width: dw, height: 6, fill: `url(#${this.id}-divert)`, class: "dv-r" }, g);
        el("title", {}, g).textContent = `Quay ${d.from_quay} closed hours ${d.start}-${d.end}; ships ${(d.ships || []).join(", ")} diverted to this quay`;
        laneText(dx0, dx0 + dw, `${d.from_quay} closed · ${(d.ships || []).length} diverted here`, "dv-t");
      }
      for (const o of outages) {
        const ox0 = x(Math.max(0, o.start));
        const ow = Math.max(1, x(Math.min(H, o.end)) - ox0);
        const g = el("g", { class: "outage" }, grid);
        el("rect", { x: ox0, y: laneY, width: ow, height: 6, fill: `url(#${this.id}-out)`, class: "ou-r" }, g);
        el("title", {}, g).textContent = `Hours ${o.start}-${o.end}: ${o.cranes} of ${rules.crane_pool} quay cranes out of service`;
        laneText(ox0, ox0 + ow, `${o.cranes} cranes out`, "ou-t");
      }
      for (const w of winds) {
        const wx0 = x(Math.max(0, w.start));
        const wx1 = x(Math.min(H, w.end));
        const kn = (String(w.reason || "").match(/(\d+)\s*kn/) || [])[1];
        laneText(wx0, wx1, `${kn ? `${kn} kn` : "wind"} · ${w.min_length > 0 ? `≥${w.min_length} m` : "all ships"}`, w.min_length > 0 ? "wl-t" : "wa-t");
      }
      // place lane labels left to right without overlapping (two text rows if needed)
      laneLabels.sort((a, b) => a.x0 - b.x0);
      const rowsEnd = [-Infinity, -Infinity];
      for (const L of laneLabels) {
        const wTxt = L.txt.length * 5.1;
        let tx = L.x0;
        if (tx + wTxt > W - right) tx = Math.max(left, W - right - wTxt);
        const r = tx > rowsEnd[0] ? 0 : tx > rowsEnd[1] ? 1 : -1;
        if (r < 0) continue;
        rowsEnd[r] = tx + wTxt + 6;
        el("text", { x: tx, y: r === 0 ? laneY - 3 : laneY - 13, class: `ax lane-t ${L.cls}` }, grid).textContent = L.txt;
      }
    }

    // blocks
    const blocks = el("g", {}, svg);
    for (const b of task.blocks || []) {
      const bx = x(Math.max(0, b.start));
      const bw = x(Math.min(H, b.end)) - bx;
      const by = y(b.first);
      const bh = (b.last - b.first + 1) * rowH;
      const g = el("g", { class: `blk ${b.kind}` }, blocks);
      el("rect", { x: bx, y: by, width: Math.max(1, bw), height: bh, fill: `url(#${this.id}-${b.kind === "closed" ? "closed" : "along"})`, class: "blk-r" }, g);
      el("title", {}, g).textContent = b.kind === "closed" ? `Closed sections ${b.first}-${b.last}, hours ${b.start}-${b.end}: ${b.label}` : `${b.label} already alongside sections ${b.first}-${b.last} until hour ${b.end}${b.cranes ? `, working ${b.cranes} cranes` : ""}`;
      if (!mini && bw > 40 && bh >= 9) {
        const label = b.kind === "closed" ? `closed · ${b.label}` : `${b.label}${b.cranes ? ` · ${b.cranes} cr` : ""}`;
        this._text(g, bx + 4, by + Math.min(bh / 2, 10) + 3, label, bw - 8, "blk-t");
      }
    }

    // published ghosts
    if (!mini) {
      const ghosts = el("g", { class: "ghosts" }, svg);
      for (const sh of task.ships) {
        const slot = publishedSlot(sh);
        if (!slot) continue;
        const row = this.ev.byId.get(sh.id);
        if (row && row.placed && row.berth === slot.start && row.section === slot.first && row.dep === slot.end) continue;
        el("rect", { x: x(slot.start), y: y(slot.first) + 0.5, width: Math.max(1, x(slot.end) - x(slot.start)), height: sh.sections * rowH - 1, class: "ghost" }, ghosts);
      }
    }

    // ships
    const ships = el("g", { class: "ships" }, svg);
    this.shipEls = new Map();
    this.hits = [];
    for (const row of this.ev.rows) {
      if (!row.placed) continue;
      const sh = row.ship;
      const rx = x(row.berth);
      const rw = Math.max(1.5, x(row.dep) - rx);
      const ry = y(row.section) + 0.5;
      const rh = sh.sections * rowH - 1;
      const kind = kinds.get(sh.id);
      const g = el("g", { class: `ship${kind ? ` extra ${kind.kind}` : ""}${row.conflicts.length ? " bad" : ""}${(sh.weight || 1) > 1 ? " prio" : ""}${sh.berth_deadline != null ? " emerg" : ""}`, "data-id": sh.id }, ships);
      if (!mini && row.berth > sh.arrival) {
        const cy = ry + rh / 2;
        el("line", { x1: x(sh.arrival), x2: rx, y1: cy, y2: cy, class: "wait" }, g);
        el("line", { x1: x(sh.arrival), x2: x(sh.arrival), y1: cy - 3, y2: cy + 3, class: "wait" }, g);
      }
      el("rect", { x: rx, y: ry, width: rw, height: rh, class: "ship-r" }, g);
      if (row.hold > 0) {
        const hx = x(row.finish);
        el("rect", { x: hx, y: ry, width: Math.max(1, x(row.dep) - hx), height: rh, fill: `url(#${this.id}-hold)`, class: "hold" }, g);
      }
      if (row.delay > 0) {
        const tx = x(Math.max(row.berth, sh.due));
        el("rect", { x: tx, y: ry, width: Math.max(1, x(row.dep) - tx), height: rh, class: "late" }, g);
      }
      if (row.berth < sh.arrival && !mini) {
        el("line", { x1: x(sh.arrival), x2: x(sh.arrival), y1: ry, y2: ry + rh, class: "early" }, g);
      }
      if (sh.berth_deadline != null) {
        const dx = x(sh.berth_deadline);
        el("line", { x1: dx, x2: dx, y1: ry - 2, y2: ry + rh + 2, class: `ddl${row.late ? " missed" : ""}` }, g);
      }
      el("rect", { x: rx, y: ry, width: rw, height: rh, class: "ship-o" }, g);
      if (!mini && rw > 26 && rh >= 8) {
        const bits = [];
        if ((sh.weight || 1) > 1) bits.push(`×${sh.weight}`);
        bits.push(kind && kind.kind === "divert" && rw > 110 ? `${sh.name} · from ${kind.from}` : sh.name);
        let label = bits.join(" ");
        if (cr) label += ` · ${row.cranes} cr`;
        if (sh.berth_deadline != null && rw > 150) label += ` · dock by h ${sh.berth_deadline}`;
        this._text(g, rx + 3, ry + Math.min(rh / 2, 9) + 3, label, rw - 5, "ship-t");
      }
      const kindTip = kind ? (kind.kind === "divert" ? ` (diverted from ${kind.from})` : " (unscheduled call)") : "";
      const tip = [
        `${sh.id} ${sh.name}${kindTip}`,
        `docks at hour ${row.berth}, sections ${row.section}-${row.last}${cr ? `, ${row.cranes} cranes (${sh.min_cranes}-${sh.max_cranes})` : ""}`,
        cr ? `${movesOf(task, sh)} moves: ${row.work} h of work, done at hour ${row.finish}${row.hold ? `, then held by the wind until hour ${row.dep}` : ""}` : null,
        `arrives ${sh.arrival}, leaves ${row.dep}, due ${sh.due}`,
        row.delay > 0 ? `${row.delay} h late${(sh.weight || 1) > 1 ? ` (priority: counts ×${sh.weight})` : ""}` : "on time",
        sh.berth_deadline != null ? `emergency: must dock by hour ${sh.berth_deadline}${row.late ? `, docks ${row.berth - sh.berth_deadline} h late (+${sh.deadline_penalty} per hour)` : ""}` : null,
        row.moved ? "moved from its published sections" : null,
        ...row.conflicts,
      ].filter(Boolean).join("\n");
      el("title", {}, g).textContent = tip;
      this.shipEls.set(sh.id, g);
      this.hits.push({ id: sh.id, x0: rx, x1: rx + rw, y0: ry, y1: ry + rh, area: rw * rh });
      if (this.editable) g.classList.add("drag");
    }
    this.hits.sort((a, b) => a.area - b.area);

    if (strips) this._strips(svg, x, plotB + 8, STRIP_C, STRIP_M, left, W - right);

    // now line
    const nx = x(Math.max(0, Math.min(H, this.time)));
    this.nowLine = el("line", { x1: nx, x2: nx, y1: mini ? 0 : top - 2, y2: plotB + stripsH, class: "now" }, svg);
    this.nowHead = mini ? null : el("path", { d: "M-4,-5 L4,-5 L0,0 Z", class: "now-h", transform: `translate(${nx},${top - 1})` }, svg);
    this._marks();
  }

  /** Strips under the chart: cranes in use vs available, and ship movements per hour vs the limit. */
  _strips(svg, x, y0, hc, hm, xl, xr) {
    const task = this.task;
    const ser = this.ev.series;
    const H = Math.floor(this.H);
    const g = el("g", { class: "strips" }, svg);
    const pool = Number(task.rules.crane_pool);
    let maxC = pool;
    for (let t = 0; t < H; t++) maxC = Math.max(maxC, ser.use.get(t) || 0);
    const yc = (v) => y0 + hc - (v / maxC) * (hc - 2);
    el("rect", { x: xl, y: y0, width: xr - xl, height: hc, class: "strip-bg" }, g);
    // available (pool less outages) as a step area
    let d = `M${x(0)},${y0 + hc}`;
    for (let t = 0; t < H; t++) {
      const a = cranePoolAt(task, t);
      d += ` L${x(t)},${yc(a)} L${x(t + 1)},${yc(a)}`;
    }
    d += ` L${x(H)},${y0 + hc} Z`;
    el("path", { d, class: "avail" }, g);
    // in use, merged into runs of equal value
    let t = 0;
    while (t < H) {
      const u = ser.use.get(t) || 0;
      const overT = u > cranePoolAt(task, t);
      let t1 = t + 1;
      while (t1 < H && (ser.use.get(t1) || 0) === u && u > cranePoolAt(task, t1) === overT && cranePoolAt(task, t1) === cranePoolAt(task, t)) t1++;
      if (u > 0) {
        const r = el("rect", { x: x(t), y: yc(u), width: Math.max(0.8, x(t1) - x(t)), height: y0 + hc - yc(u), class: overT ? "use over" : "use" }, g);
        el("title", {}, r).textContent = `hours ${t}-${t1}: ${u} cranes in use, ${cranePoolAt(task, t)} available${overT ? " (over the pool)" : ""}`;
      }
      t = t1;
    }
    el("text", { x: xl + 3, y: y0 + 9, class: "ax strip-t" }, g).textContent = `cranes in use / available (pool ${pool})`;
    // moves per hour vs limit
    const y1 = y0 + hc + 6;
    const cap = ser.cap || 0;
    let maxM = Math.max(cap + 1, 2);
    for (const v of ser.moves.values()) maxM = Math.max(maxM, v);
    const ym = (v) => y1 + hm - (v / maxM) * (hm - 2);
    el("rect", { x: xl, y: y1, width: xr - xl, height: hm, class: "strip-bg" }, g);
    const bw = Math.max(1.2, (xr - xl) / this.H - 0.3);
    for (const [hh, v] of ser.moves) {
      if (!v || hh < 0 || hh > this.H) continue;
      const r = el("rect", { x: x(hh) - bw / 2 + ((xr - xl) / this.H) / 2, y: ym(v), width: bw, height: y1 + hm - ym(v), class: cap && v > cap ? "mv over" : "mv" }, g);
      el("title", {}, r).textContent = `hour ${hh}: ${v} ship${v > 1 ? "s" : ""} dock or leave${cap ? ` (limit ${cap})` : ""}`;
    }
    if (cap) el("line", { x1: xl, x2: xr, y1: ym(cap), y2: ym(cap), class: "cap" }, g);
    el("text", { x: xl + 3, y: y1 + 8, class: "ax strip-t" }, g).textContent = `moves per hour${cap ? ` (limit ${cap})` : ""}`;
  }

  _text(g, tx, ty, str, maxW, cls) {
    const per = this.geom.rowH >= 11 ? 5.6 : 5.0;
    const fit = Math.floor(maxW / per);
    if (fit < 3) return;
    const s = str.length > fit ? `${str.slice(0, Math.max(1, fit - 1))}…` : str;
    el("text", { x: tx, y: ty, class: cls }, g).textContent = s;
  }

  _hourAt(clientX) {
    const r = this.svg.getBoundingClientRect();
    const g = this.geom;
    const px = ((clientX - r.left) / r.width) * g.W;
    return Math.max(0, Math.min(this.H, ((px - g.left) / (g.right - g.left)) * this.H));
  }

  _shipAt(clientX, clientY) {
    if (!this.hits || !this.geom) return null;
    const r = this.svg.getBoundingClientRect();
    const scale = r.width / this.geom.W || 1;
    const px = (clientX - r.left) / scale;
    const py = (clientY - r.top) / scale;
    for (const h of this.hits) if (px >= h.x0 - 1 && px <= h.x1 + 1 && py >= h.y0 && py <= h.y1) return h.id;
    return null;
  }

  _rowOf(id) {
    return this.ev && this.ev.byId.get(id);
  }

  /** Clamp a proposed placement to the quay and the chart's hour range. */
  _clampPlace(row, hour, section) {
    const t = this.task;
    const n = row.ship.sections;
    return {
      hour: Math.max(0, Math.min(Math.max(0, Math.floor(this.H) - (row.work || row.ship.handling)), Math.round(hour))),
      section: Math.max(t.first_section, Math.min(t.last_section - n + 1, Math.round(section))),
    };
  }

  _preview(id, hour, section) {
    const g = this.shipEls && this.shipEls.get(id);
    const row = this._rowOf(id);
    if (!g || !row || !this.geom) return;
    const dx = this.geom.x(hour) - this.geom.x(row.berth);
    const dy = (section - row.section) * this.geom.rowH;
    g.setAttribute("transform", `translate(${dx},${dy})`);
    g.classList.add("dragging");
    if (!this.dragLbl) this.dragLbl = el("text", { class: "drag-t" }, this.svg);
    const last = section + row.ship.sections - 1;
    const fin = hour + (row.work || row.ship.handling);
    const dep = hasCranes(this.task) ? departureAfter(this.task, row.ship, fin) : fin;
    const txt = `h ${hour} · ${section}–${last} · leaves h ${dep}${dep > fin ? " (wind)" : ""}`;
    this.dragLbl.textContent = txt;
    const lx = Math.min(this.geom.right - txt.length * 5.6 - 2, Math.max(this.geom.left + 2, this.geom.x(hour)));
    const ly = this.geom.y(section) - 3 < this.geom.top + 8 ? this.geom.y(last + 1) + 11 : this.geom.y(section) - 3;
    this.dragLbl.setAttribute("x", lx);
    this.dragLbl.setAttribute("y", ly);
    this.svg.appendChild(this.dragLbl);
  }

  _clearPreview() {
    if (this.dragLbl) {
      this.dragLbl.remove();
      this.dragLbl = null;
    }
  }

  _bind() {
    let drag = null;
    const scale = () => {
      const r = this.svg.getBoundingClientRect();
      return r.width / this.geom.W || 1;
    };
    this.svg.addEventListener("pointermove", (e) => {
      if (!this.geom) return;
      if (drag && drag.kind === "ship") {
        const k = scale();
        const pxPerHour = (this.geom.right - this.geom.left) / this.H;
        const dh = (e.clientX - drag.x) / k / pxPerHour;
        const ds = (e.clientY - drag.y) / k / this.geom.rowH;
        if (!drag.moved && Math.hypot(e.clientX - drag.x, e.clientY - drag.y) < 4) return;
        drag.moved = true;
        const p = this._clampPlace(drag.row, drag.h0 + dh, drag.s0 + ds);
        if (p.hour !== drag.hour || p.section !== drag.section) {
          drag.hour = p.hour;
          drag.section = p.section;
          this._preview(drag.id, p.hour, p.section);
          if (this.onMove) this.onMove(drag.id, p.hour, p.section, false);
        }
        return;
      }
      if (drag && this.onScrub && !this.mini) {
        if (Math.abs(e.clientX - drag.x) > 3) drag.moved = true;
        if (drag.moved) this.onScrub(this._hourAt(e.clientX));
        return;
      }
      const id = this._shipAt(e.clientX, e.clientY);
      if (id !== this._hov) {
        this._hov = id;
        this.svg.style.cursor = id != null ? (this.editable ? "grab" : "pointer") : this.mini ? "pointer" : "crosshair";
        if (this.onHover) this.onHover(id);
      }
    });
    this.svg.addEventListener("pointerleave", () => {
      if (drag) return;
      if (this._hov != null && this.onHover) this.onHover(null);
      this._hov = null;
    });
    this.svg.addEventListener("pointerdown", (e) => {
      if (!this.geom || this.mini) return;
      const id = this.editable ? this._shipAt(e.clientX, e.clientY) : null;
      if (id != null) {
        const row = this._rowOf(id);
        if (!row || !row.placed) return;
        e.preventDefault();
        this.container.focus({ preventScroll: true });
        drag = { kind: "ship", id, row, x: e.clientX, y: e.clientY, h0: row.berth, s0: row.section, hour: row.berth, section: row.section, moved: false };
        this.svg.style.cursor = "grabbing";
      } else {
        drag = { kind: "time", x: e.clientX, moved: false };
      }
      this.svg.setPointerCapture(e.pointerId);
    });
    const end = (e, cancelled) => {
      if (this.mini) {
        if (!cancelled && this.onPick) this.onPick(null, null);
        return;
      }
      if (!drag) return;
      const d = drag;
      drag = null;
      try {
        this.svg.releasePointerCapture(e.pointerId);
      } catch {}
      if (d.kind === "ship") {
        this._clearPreview();
        this.svg.style.cursor = "grab";
        if (d.moved) {
          if (this.onMove) this.onMove(d.id, d.hour, d.section, true);
          else this.render();
        } else if (!cancelled && this.onPick) this.onPick(null, d.id);
        return;
      }
      if (d.moved || cancelled) return;
      const h = this._hourAt(e.clientX);
      const id = this._shipAt(e.clientX, e.clientY);
      if (this.onPick) this.onPick(h, id);
    };
    this.svg.addEventListener("pointerup", (e) => end(e, false));
    this.svg.addEventListener("pointercancel", (e) => end(e, true));
    if (this.editable) {
      this.container.addEventListener("keydown", (e) => {
        if (this.selected == null || typeof this.selected !== "number") return;
        const row = this._rowOf(this.selected);
        if (!row || !row.placed) return;
        const step = e.shiftKey ? 6 : 1;
        let h = row.berth;
        let sec = row.section;
        if (e.key === "ArrowLeft") h -= step;
        else if (e.key === "ArrowRight") h += step;
        else if (e.key === "ArrowUp") sec -= 1;
        else if (e.key === "ArrowDown") sec += 1;
        else if ((e.key === "+" || e.key === "=" || e.key === "-" || e.key === "_") && this.onCranes && hasCranes(this.task)) {
          e.preventDefault();
          e.stopPropagation();
          this.onCranes(row.id, e.key === "-" || e.key === "_" ? -1 : 1);
          return;
        } else return;
        e.preventDefault();
        e.stopPropagation();
        const p = this._clampPlace(row, h, sec);
        if (this.onMove && (p.hour !== row.berth || p.section !== row.section)) this.onMove(row.id, p.hour, p.section, true);
      });
    }
  }
}

export function legendHtml(opts = {}) {
  const sw = (cls) => `<i class="sw ${cls}"></i>`;
  return [
    `<span title="Ship alongside in this plan">${sw("lg-ship")} ship</span>`,
    `<span title="Hours past the ship's due departure">${sw("lg-late")} late</span>`,
    `<span title="Overlap, docking before arrival, a closed section, or (crane rules) cranes over the pool, too many moves in an hour, docking in a wind window">${sw("lg-bad")} conflict</span>`,
    `<span title="Slot in the published plan">${sw("lg-ghost")} published</span>`,
    `<span title="Sections closed for works">${sw("lg-closed")} closed</span>`,
    `<span title="Ship already alongside at hour 0 (cannot move)">${sw("lg-along")} alongside</span>`,
    `<span title="Unscheduled extra call, or a call diverted from the other quay">${sw("lg-extra")} extra / diverted</span>`,
    ...(opts.cranes ? [
      `<span title="Finished but held alongside: it may not leave inside a wind window">${sw("lg-hold")} held by wind</span>`,
      `<span title="Wind windows: no ship may dock or leave (darker: all ships; lighter: ships of 300 m or more)">${sw("lg-wind")} wind</span>`,
      `<span title="Emergency: must dock by the red line">${sw("lg-ddl")} dock-by</span>`,
    ] : []),
  ].join("");
}

export { escapeHtml };
