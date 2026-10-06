// Deterministic, metre/second cargo choreography. This is a visible deck-cargo
// sample: berth tasks provide crane-hours, not a bay manifest or yard inventory.
// A box keeps one identity from its ship slot through the hoist to its yard slot.
import { hullsOverlap } from './navigation.js';
export const CONTAINER = { length: 12.19, width: 2.44, height: 2.59 };
const H = CONTAINER.height;
const clamp = v => Math.max(0, Math.min(1, v));
const ease = v => { const u = clamp(v); return u * u * (3 - 2 * u); };
const mix = (a, b, u) => a + (b - a) * ease(u);
const between = (t, a, b) => (t - a) / Math.max(0.001, b - a);

/** A rounded route, with distance-based sampling and no instantaneous corners. */
export function cargoRoute(points, radius = 7) {
  const clean = points.filter((p, i) => !i || Math.hypot(p[0] - points[i - 1][0], p[1] - points[i - 1][1]) > 0.01);
  const out = [clean[0]];
  for (let i = 1; i < clean.length - 1; i++) {
    const a = clean[i - 1], b = clean[i], c = clean[i + 1];
    const ab = Math.hypot(b[0] - a[0], b[1] - a[1]), bc = Math.hypot(c[0] - b[0], c[1] - b[1]);
    const d = Math.min(radius, ab * 0.4, bc * 0.4);
    const p = b.map((v, k) => v + (a[k] - v) * d / ab), q = b.map((v, k) => v + (c[k] - v) * d / bc);
    out.push(p);
    for (let k = 1; k <= 12; k++) {
      const u = k / 12;
      out.push(p.map((v, j) => (1 - u) ** 2 * v + 2 * (1 - u) * u * b[j] + u * u * q[j]));
    }
  }
  out.push(clean.at(-1));
  let length = 0;
  const distances = [0];
  for (let i = 1; i < out.length; i++) { length += Math.hypot(out[i][0] - out[i - 1][0], out[i][1] - out[i - 1][1]); distances.push(length); }
  return { points: out, distances, length };
}

export function routePose(route, u) {
  const d = clamp(u) * route.length;
  let i = 1;
  while (i < route.distances.length - 1 && route.distances[i] < d) i++;
  const a = route.points[i - 1], b = route.points[i];
  const f = (d - route.distances[i - 1]) / Math.max(0.001, route.distances[i] - route.distances[i - 1]);
  return { x: a[0] + (b[0] - a[0]) * f, z: a[1] + (b[1] - a[1]) * f, ry: Math.atan2(a[1] - b[1], b[0] - a[0]) };
}

/** Top tiers first. A selected lower box always has every box above it selected. */
export function dischargeOrder(slots, bayX, side = 1) {
  return slots.filter(s => Math.abs(s.x - bayX) < 0.05).sort((a, b) => b.t - a.t || side * (a.z - b.z) || a.id - b.id);
}

export function makeCargoJob({ id, ship, slot, source, destination, start, laneZ, roadZ, clearY }) {
  const ground = 0.34 + H / 2;
  // Keep the carrier away until the STS spreader has lifted back out of its way.
  const approach = cargoRoute([[source.x - 19, laneZ - 14], [source.x - 19, laneZ], [source.x, laneZ]]);
  const via = destination.via;
  const end = destination.p;
  const outbound = cargoRoute([[source.x, laneZ], [source.x + 19, laneZ], [source.x + 19, roadZ], [via[0], roadZ], via, [end[0], end[2]]]);
  // Return along the same reserved access corridor only after setting down the box.
  const home = cargoRoute([[end[0], end[2]], via, [via[0], roadZ - 7], [source.x - 19, roadZ - 7], [source.x - 19, laneZ - 14]]);
  const mark = { start };
  let clock = start;
  const phase = (name, seconds) => { clock += seconds; mark[name] = clock; };
  // Smoothstep has peak speed 1.5 times its average: durations enforce the limits.
  phase('contact', Math.max(8, 1.5 * (clearY - source.y) / 1.4));
  phase('pickup', 5); // twistlock dwell
  phase('lift', Math.max(8, 1.5 * (clearY - source.y) / 1.1));
  phase('cross', Math.max(10, 1.5 * Math.abs(source.z - laneZ) / 2.4));
  phase('land', Math.max(10, 1.5 * (clearY - ground) / 1.1));
  phase('release', 5);
  phase('clear', Math.max(10, 1.5 * (clearY - ground) / 1.8));
  phase('collect', Math.max(14, 1.5 * approach.length / 3));
  const travelY = Math.max(ground + 1.4, destination.travelY ?? 9.2);
  phase('align', Math.max(8, 1.5 * (travelY - ground) / 1.4));
  phase('lock', 6);
  phase('carrierLift', Math.max(8, 1.5 * (travelY - ground) / 0.8));
  phase('atYard', Math.max(15, 1.5 * outbound.length / 4));
  phase('stored', Math.max(8, 1.5 * Math.abs(travelY - end[1]) / 0.8));
  phase('unlock', Math.max(6, 1.5 * (travelY - end[1]) / 1.4));
  phase('home', Math.max(15, 1.5 * home.length / 4));
  return { id, ship, slot, source, destination, laneZ, clearY, ground, travelY, approach, outbound, home, mark, duration: clock - start };
}

/** Exactly one owner, including at the pickup/release boundaries and after seeks. */
export function cargoState(job, t) {
  const { mark: m, source: s, destination: d, clearY: up, ground, laneZ } = job;
  let owner = 'ship', phase = 'ready', box = { ...s }, crane = { x: s.x, y: up, z: s.z }, carrier = null;
  if (t < m.start) return { owner, phase, box, crane, carrier };
  if (t < m.pickup) {
    phase = t < m.contact ? 'Lowering spreader' : 'Locking twistlocks';
    crane.y = mix(up, s.y, between(t, m.start, m.contact));
  } else if (t < m.release) {
    owner = 'crane';
    if (t < m.lift) { phase = 'Hoisting clear'; crane.y = mix(s.y, up, between(t, m.pickup, m.lift)); }
    else if (t < m.cross) { phase = 'Trolley to quay'; crane.z = mix(s.z, laneZ, between(t, m.lift, m.cross)); }
    else { phase = t < m.land ? 'Lowering to quay' : 'Unlocking spreader'; crane.z = laneZ; crane.y = mix(up, ground, between(t, m.cross, m.land)); }
    box = { ...crane, ry: s.ry };
  } else {
    crane.z = laneZ;
    crane.y = mix(ground, up, between(t, m.release, m.clear));
    box = { x: s.x, y: ground, z: laneZ, ry: s.ry };
    owner = 'quay'; phase = 'Quay handoff';
    if (t >= m.clear && t < m.collect) carrier = routePose(job.approach, ease(between(t, m.clear, m.collect)));
    else if (t >= m.collect && t < m.carrierLift) carrier = { x: s.x, z: laneZ, ry: 0 };
    else if (t >= m.carrierLift && t < m.atYard) carrier = routePose(job.outbound, ease(between(t, m.carrierLift, m.atYard)));
    else if (t >= m.atYard && t < m.unlock) carrier = { x: d.p[0], z: d.p[2], ry: d.ry };
    else if (t >= m.unlock && t < m.home) { carrier = routePose(job.home, ease(between(t, m.unlock, m.home))); carrier.ry += Math.PI; }
    if (t >= m.lock && t < m.stored) {
      owner = 'carrier'; phase = t < m.carrierLift ? 'Carrier lifting' : t < m.atYard ? 'Carrier to yard' : 'Stacking in yard';
      const y = t < m.carrierLift ? mix(ground, job.travelY, between(t, m.lock, m.carrierLift)) : t < m.atYard ? job.travelY : mix(job.travelY, d.p[1], between(t, m.atYard, m.stored));
      box = { ...carrier, y, ry: carrier.ry + s.ry };
    } else if (t >= m.stored) {
      owner = 'yard'; phase = 'Stored'; box = { x: d.p[0], y: d.p[1], z: d.p[2], ry: d.ry + s.ry };
    }
    if (carrier) carrier.hoistY = t < m.collect ? job.travelY : t < m.lock ? mix(job.travelY, ground, between(t, m.collect, m.align)) : t < m.stored ? box.y : mix(d.p[1], job.travelY, between(t, m.stored, m.unlock));
  }
  // An empty carrier waits beside the handover lane; it never materialises over a box.
  if (!carrier && t < m.home) carrier = { ...routePose(job.approach, 0), hoistY: job.travelY };
  if (carrier && carrier.hoistY == null) carrier.hoistY = job.travelY;
  return { owner, phase, box, crane, carrier };
}

/** Don't begin a lift that cannot finish before a closure or the ship's departure. */
export function nextCargoStart(start, duration, windows) {
  let s = start;
  for (const w of [...windows].sort((a, b) => a.start - b.start)) if (s < w.end && s + duration > w.start) s = w.end;
  return s;
}

export function carriersConflict(a, b) {
  const lo = Math.max(a.mark.clear, b.mark.clear), hi = Math.min(a.mark.home, b.mark.home);
  if (hi <= lo) return false;
  // At <=4 m/s, a one-second sample plus 3 m clearance covers motion between samples.
  for (let t = lo; t <= hi; t += 1) {
    const p = cargoState(a, t).carrier, q = cargoState(b, t).carrier;
    if (p && q && hullsOverlap({ ...p, th: p.ry }, { len: 14, beam: 5.6 }, { ...q, th: q.ry }, { len: 14, beam: 5.6 }, 3)) return true;
  }
  return false;
}

/** Allocate empty yard slots once, and reserve carrier routes before starting lifts. */
export function planDischarges(services, yardSlots, { laneZ, roadZ, windows = [] }) {
  const available = yardSlots.map((d, i) => ({ ...d, id: i }));
  const jobs = [], lanes = [], corridorReady = new Map();
  const ordered = [...services].filter(s => s.end > s.start && s.lanes.length).sort((a, b) => a.start - b.start);
  // Share finite, explicitly modelled storage across the replay. These containers
  // illustrate handling; the RL workload can also include holds, restows and exports.
  const share = Math.max(1, Math.floor(available.length / Math.max(1, ordered.length)));
  for (const service of ordered) {
    for (const [li, lane] of service.lanes.entries()) {
      const record = { ship: service.ship, x: lane.x, jobs: [], start: service.start, end: service.end };
      lanes.push(record);
      const wanted = Math.min(lane.slots.length, Math.max(1, Math.floor(share / service.lanes.length)), 50);
      const interval = (service.end - service.start - 120) / Math.max(1, wanted);
      let ready = service.start + 45 + li * 20;
      for (let k = 0; k < wanted && available.length; k++) {
        const slot = lane.slots[k];
        // Farther slots in a shared access row are filled first.
        const candidates = available.filter(d => !available.some(o => o.corridor === d.corridor && o.depth > d.depth + 0.1));
        candidates.sort((a, b) => Math.abs(a.via[0] - lane.x) - Math.abs(b.via[0] - lane.x));
        const destination = candidates[0];
        if (!destination) break;
        const source = { x: lane.x, y: slot.y + service.waterY, z: service.z + service.side * slot.z, ry: service.side < 0 ? Math.PI : 0 };
        let start = Math.max(ready, service.start + 45 + li * 20 + k * interval, corridorReady.get(destination.corridor) || 0);
        const create = () => makeCargoJob({ id: `${service.ship}:${slot.id}`, ship: service.ship, slot, source, destination, start, laneZ, roadZ, clearY: lane.clearY });
        let job = create();
        for (let attempt = 0; attempt <= jobs.length; attempt++) {
          start = nextCargoStart(start, job.duration, windows);
          job = create();
          const conflict = jobs.find(other => carriersConflict(job, other));
          if (!conflict) break;
          start = Math.max(start + 1, conflict.mark.home - (job.mark.clear - job.mark.start) + 3);
        }
        if (job.mark.home > service.end - 10) break;
        jobs.push(job); record.jobs.push(job);
        corridorReady.set(destination.corridor, job.mark.home + 5);
        available.splice(available.findIndex(d => d.id === destination.id), 1);
        ready = job.mark.home + 15;
      }
    }
  }
  return { jobs, lanes };
}
