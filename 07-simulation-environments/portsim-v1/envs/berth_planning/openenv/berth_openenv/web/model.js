// Plan semantics shared by the chart, tables and 3D scene. Mirrors berth_core/check.py (`evaluate`,
// `ship_delay_cost`) and berth_core/model.py (`departure`, `no_move_windows`, `crane_pool_at`).
// A plan entry {ship, berth_hour, section[, cranes]} occupies sections section..section+sections-1 from
// berth_hour until it leaves. On crane-rule tasks: work = ceil(workload / cranes), finish = berth_hour + work,
// and a ship that finishes inside a wind window that applies to it waits alongside until the window ends.
// Cost = sum(sections x hours late x weight + deadline penalty) + 5 x ships moved off their planned sections.

export const MOVE_PENALTY = 5;
export const DAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

/** Crane-rule ("dock") task: plans carry a crane count per ship. */
export const hasCranes = (task) => !!(task && task.rules && task.rules.crane_pool);

export function cranePoolAt(task, hour) {
  let pool = Number(task.rules.crane_pool);
  for (const o of task.rules.crane_outages || []) if (o.start <= hour && hour < o.end) pool -= Number(o.cranes);
  return pool;
}

/** Merged [start, end) windows in which this ship may not dock or leave (wind limits by length). */
export function noMoveWindows(task, ship) {
  const ws = ((task.rules && task.rules.no_moves) || [])
    .filter((w) => ship.length_m >= (w.min_length || 0))
    .map((w) => [w.start, w.end])
    .sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const out = [];
  for (const [a, b] of ws) {
    if (out.length && a <= out[out.length - 1][1]) out[out.length - 1][1] = Math.max(out[out.length - 1][1], b);
    else out.push([a, b]);
  }
  return out;
}

/** A ship that finishes inside a no-movement window waits alongside until the window ends. */
export function departureAfter(task, ship, finish) {
  for (const [a, b] of noMoveWindows(task, ship)) if (a <= finish && finish < b) return b;
  return finish;
}

export function handlingFor(ship, cranes) {
  if (ship.workload == null || cranes == null) return ship.handling;
  return Math.max(1, Math.ceil(ship.workload / cranes));
}

export const movesOf = (task, ship) => (ship.workload != null ? ship.workload * Number((task.rules && task.rules.crane_rate) || 28) : null);

export function shipDelayCost(ship, berthHour, delayH) {
  let cost = ship.sections * delayH * (ship.weight || 1);
  if (ship.berth_deadline != null && berthHour > ship.berth_deadline) cost += (ship.deadline_penalty || 0) * (berthHour - ship.berth_deadline);
  return cost;
}

export function publishedPlan(task) {
  const cr = hasCranes(task);
  return task.ships
    .filter((s) => s.planned_hour != null && s.planned_section != null)
    .map((s) => ({ ship: s.id, berth_hour: s.planned_hour, section: s.planned_section, ...(cr ? { cranes: s.std_cranes } : {}) }));
}

/** The published slot of a ship: planned docking hour until its planned departure (= due). */
export function publishedSlot(ship) {
  if (ship.planned_hour == null || ship.planned_section == null) return null;
  return { start: ship.planned_hour, end: Math.max(ship.due, ship.planned_hour + 1), first: ship.planned_section, last: ship.planned_section + ship.sections - 1 };
}

const overlap = (a0, a1, b0, b1) => Math.max(a0, b0) < Math.min(a1, b1);
const span = (a0, a1, b0, b1) => [Math.max(a0, b0), Math.min(a1, b1)];
const spanTxt = (a, b) => (a === b ? `${a}` : `${a}-${b}`);

function hoursTxt(hs) {
  const out = [];
  let run = [];
  for (const h of [...new Set(hs)].sort((a, b) => a - b)) {
    if (run.length && h === run[run.length - 1] + 1) run.push(h);
    else {
      if (run.length) out.push(spanTxt(run[0], run[run.length - 1]));
      run = [h];
    }
  }
  if (run.length) out.push(spanTxt(run[0], run[run.length - 1]));
  return out.slice(0, 4).join(", ") + (out.length > 4 ? " …" : "");
}

/** Evaluate a plan locally: per-ship timing, delay, moves, cost and rule problems (as help; the env grades). */
export function evaluatePlan(task, plan) {
  const cr = hasCranes(task);
  const entries = new Map();
  const duplicates = [];
  for (const p of Array.isArray(plan) ? plan : []) {
    if (!p || typeof p !== "object") continue;
    const id = Number(p.ship);
    if (entries.has(id)) duplicates.push(id);
    else entries.set(id, { berth_hour: Number(p.berth_hour), section: Number(p.section), cranes: p.cranes == null ? null : Number(p.cranes) });
  }
  const issues = [];
  const rows = task.ships.map((s) => {
    const e = entries.get(s.id);
    if (!e || !Number.isFinite(e.berth_hour) || !Number.isFinite(e.section)) {
      return { ship: s, id: s.id, placed: false, conflicts: [], cost: null, delay: null, moved: false };
    }
    const h = e.berth_hour;
    const c = cr ? (e.cranes != null && Number.isFinite(e.cranes) ? e.cranes : s.std_cranes) : null;
    const badCranes = cr && !(s.min_cranes <= c && c <= s.max_cranes);
    const work = handlingFor(s, cr ? (badCranes ? s.std_cranes : c) : null);
    const finish = h + work;
    const dep = cr ? departureAfter(task, s, finish) : finish;
    const delay = Math.max(0, dep - s.due);
    const moved = s.planned_section != null && e.section !== s.planned_section;
    return {
      ship: s,
      id: s.id,
      placed: true,
      berth: h,
      section: e.section,
      last: e.section + s.sections - 1,
      cranes: c,
      work,
      finish,
      dep,
      hold: dep - finish,
      delay,
      moved,
      wait: Math.max(0, h - s.arrival),
      cost: shipDelayCost(s, h, delay) + (moved ? MOVE_PENALTY : 0),
      late: s.berth_deadline != null && h > s.berth_deadline,
      conflicts: [],
    };
  });
  const add = (r, problem) => {
    issues.push({ ships: [r.id], problem });
    r.conflicts.push(problem);
  };
  const placed = rows.filter((r) => r.placed);
  const lo = task.first_section;
  const hi = task.last_section;
  for (const r of placed) {
    const s = r.ship;
    if (cr) for (const [a, b] of noMoveWindows(task, s)) if (a <= r.berth && r.berth < b) add(r, `docks at hour ${r.berth}, inside the no-movement window ${a}-${b}`);
    if (cr && !(s.min_cranes <= r.cranes && r.cranes <= s.max_cranes)) add(r, `gets ${r.cranes} cranes but can be worked by ${s.min_cranes}-${s.max_cranes}`);
    if (r.berth < s.arrival) add(r, `docks at hour ${r.berth} but cannot arrive before hour ${s.arrival}`);
    if (r.section < lo || r.last > hi) add(r, `needs sections ${spanTxt(r.section, r.last)} but the quay has sections ${lo}-${hi}`);
    for (const b of task.blocks || []) {
      if (r.section <= b.last && b.first <= r.last && r.berth < b.end && b.start < r.dep) {
        const what = b.kind === "alongside" ? `${b.label} (alongside)` : "closed sections";
        add(r, `overlaps ${what} ${spanTxt(b.first, b.last)} during hours ${b.start}-${b.end}`);
      }
    }
  }
  for (let i = 0; i < placed.length; i++) {
    for (let j = i + 1; j < placed.length; j++) {
      const a = placed[i];
      const b = placed[j];
      if (overlap(a.berth, a.dep, b.berth, b.dep) && a.section <= b.last && b.section <= a.last) {
        const [s0, s1] = span(a.section, a.last + 1, b.section, b.last + 1);
        const [h0, h1] = span(a.berth, a.dep, b.berth, b.dep);
        const secs = spanTxt(s0, s1 - 1);
        issues.push({ ships: [a.id, b.id], problem: `${a.ship.name} and ${b.ship.name} overlap in sections ${secs} during hours ${h0}-${h1}` });
        a.conflicts.push(`overlaps ship ${b.id} (${b.ship.name}) in sections ${secs} during hours ${h0}-${h1}`);
        b.conflicts.push(`overlaps ship ${a.id} (${a.ship.name}) in sections ${secs} during hours ${h0}-${h1}`);
      }
    }
  }
  let series = null;
  if (cr) series = craneAndMoveLimits(task, placed, add);
  const missing = rows.filter((r) => !r.placed).map((r) => r.id);
  const delayCost = placed.reduce((acc, r) => acc + shipDelayCost(r.ship, r.berth, r.delay), 0);
  const moves = placed.filter((r) => r.moved).length;
  return {
    rows,
    byId: new Map(rows.map((r) => [r.id, r])),
    issues,
    missing,
    duplicates,
    delayCost,
    moves,
    cost: delayCost + MOVE_PENALTY * moves,
    feasible: issues.length === 0 && missing.length === 0,
    series,
  };
}

/** Crane pool and movement limit hour by hour (check.py `_crane_and_move_limits`); returns the hourly series. */
function craneAndMoveLimits(task, placed, add) {
  const use = new Map();
  const moves = new Map();
  const fixed = new Map();
  const bump = (m, k, v) => m.set(k, (m.get(k) || 0) + v);
  for (const b of task.blocks || []) {
    if (b.kind !== "alongside") continue;
    for (let t = b.start; t < b.end; t++) bump(use, t, b.cranes || 0);
    bump(fixed, b.end, 1);
  }
  for (const r of placed) {
    for (let t = r.berth; t < r.finish; t++) bump(use, t, r.cranes || 0);
    if (!moves.has(r.berth)) moves.set(r.berth, []);
    moves.get(r.berth).push(r);
    if (!moves.has(r.dep)) moves.set(r.dep, []);
    moves.get(r.dep).push(r);
  }
  const over = new Set([...use].filter(([t, u]) => u > cranePoolAt(task, t)).map(([t]) => t));
  if (over.size) {
    for (const r of placed) {
      const hs = [];
      for (let t = r.berth; t < r.finish; t++) if (over.has(t)) hs.push(t);
      if (!hs.length) continue;
      const worst = hs.reduce((a, t) => (use.get(t) - cranePoolAt(task, t) > use.get(a) - cranePoolAt(task, a) ? t : a), hs[0]);
      add(r, `cranes over the pool at hours ${hoursTxt(hs)} (e.g. hour ${worst}: ${use.get(worst)} in use, ${cranePoolAt(task, worst)} available)`);
    }
  }
  const cap = task.rules.max_moves_per_hour;
  const moveCount = new Map();
  for (const t of new Set([...moves.keys(), ...fixed.keys()])) moveCount.set(t, (moves.get(t) || []).length + (fixed.get(t) || 0));
  if (cap) {
    for (const [t, rs] of [...moves].sort((a, b) => a[0] - b[0])) {
      const n = moveCount.get(t);
      if (n > cap) for (const r of rs) add(r, `${r.berth === t ? "docks" : "leaves"} at hour ${t}, when ${n} ships move (limit ${cap} per hour)`);
    }
  }
  return { use, moves: moveCount, cap: cap || null, over };
}

/** Last hour worth showing: every departure, block end and arrival, plus time to sail out. */
export function horizonOf(task, plans) {
  let h = 0;
  for (const s of task.ships) h = Math.max(h, s.arrival, s.due);
  for (const b of task.blocks || []) h = Math.max(h, b.end);
  for (const w of (task.rules && task.rules.no_moves) || []) h = Math.max(h, w.end);
  for (const o of (task.rules && task.rules.crane_outages) || []) h = Math.max(h, o.end);
  const cr = hasCranes(task);
  for (const plan of plans) {
    if (!Array.isArray(plan)) continue;
    for (const p of plan) {
      if (!p || typeof p !== "object") continue;
      const s = task.ships.find((x) => x.id === Number(p.ship));
      if (!s) continue;
      const finish = Number(p.berth_hour) + handlingFor(s, cr ? (p.cranes != null ? Number(p.cranes) : s.std_cranes) : null);
      h = Math.max(h, cr ? departureAfter(task, s, finish) : finish);
    }
  }
  return Math.ceil(h + 4);
}

/** Ships without a published slot: {kind: "divert", from} for calls diverted from the other quay, else {kind: "extra"}. */
export function callKinds(task) {
  const out = new Map();
  for (const d of task.disruptions || []) {
    if (d.type === "divert") for (const id of d.ships || []) out.set(Number(id), { kind: "divert", from: d.from_quay, start: d.start, end: d.end });
  }
  for (const s of task.ships) {
    if (!out.has(s.id) && (s.planned_section == null || s.planned_hour == null)) out.set(s.id, { kind: "extra" });
  }
  return out;
}

export const divertWindows = (task) => (task.disruptions || []).filter((d) => d.type === "divert");

export const DIFF_RANK = { easy: 0, medium: 1, hard: 2, expert: 3, calm: 4, busy: 5, gale: 6, storm: 7 };

export function ago(unixSeconds) {
  if (unixSeconds == null) return "–";
  const s = Math.max(0, Date.now() / 1000 - Number(unixSeconds));
  if (s < 60) return `${Math.round(s)} s ago`;
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400) return `${Math.round(s / 3600)} h ago`;
  return `${Math.round(s / 86400)} d ago`;
}

export function clockAt(task, hour) {
  const t0 = Date.parse(task.week_start_utc || "2024-01-01T00:00:00Z");
  const d = new Date(t0 + hour * 3600e3);
  const hh = String(d.getUTCHours()).padStart(2, "0");
  const mm = String(d.getUTCMinutes()).padStart(2, "0");
  return { day: DAYS[d.getUTCDay()], date: `${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]}`, hm: `${hh}:${mm}`, dayIndex: Math.floor(hour / 24) };
}

export const fmtHour = (task, h) => {
  const c = clockAt(task, h);
  return `${c.day} ${c.hm}`;
};

export function hashStr(str) {
  let h = 2166136261;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export const fmtNum = (v, d = 0) => (v == null || Number.isNaN(v) ? "–" : Number(v).toFixed(d));
export const fmtPct = (v) => (v == null || Number.isNaN(v) ? "–" : `${Math.round(v * 100)}%`);
export const sectionsLabel = (a, b) => (a === b ? `${a}` : `${a}–${b}`);

/** Parse a plan out of a tool-call argument string (or object); null when it is not a plan. */
export function parsePlanArgs(args) {
  let obj = args;
  if (typeof args === "string") {
    try {
      obj = JSON.parse(args);
    } catch {
      return null;
    }
  }
  if (Array.isArray(obj)) return obj;
  if (obj && Array.isArray(obj.plan)) return obj.plan;
  return null;
}

export function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

/** Small inline badges for priority (×3) and emergency (dock by h N) ships. */
export function badgesHtml(s) {
  let out = "";
  if ((s.weight || 1) > 1) out += ` <span class="bdg" title="Priority: each hour late counts ×${s.weight}">×${s.weight}</span>`;
  if (s.berth_deadline != null) out += ` <span class="bdg red" title="Emergency: must dock by hour ${s.berth_deadline}; ${s.deadline_penalty} per hour later">dock by h ${s.berth_deadline}</span>`;
  return out;
}
