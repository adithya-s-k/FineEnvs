// Deterministic, metre-scale harbour traffic for the visual replay. The planner's integer-hour
// assignments and rewards remain authoritative. Physical manoeuvres can wait for clearance;
// their offset is exposed to the viewer, never written back into the submitted plan.
import { noMoveWindows } from './model.js';

const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
const mix = (a, b, u) => a + (b - a) * u;
const smooth = u => { u = clamp(u); return u * u * (3 - 2 * u); };
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const heading = (a, b) => Math.atan2(a[1] - b[1], b[0] - a[0]);
const angle = (a, b, u) => a + Math.atan2(Math.sin(b - a), Math.cos(b - a)) * u;
const point = (a, b, u) => [mix(a[0], b[0], u), mix(a[1], b[1], u)];
const SAMPLE = 1 / 720; // five simulated seconds; all segment boundaries are also checked

/** The same stationary vessels are used by the scenery and the traffic controller. */
export function harbourObstacles(lay) {
  if (!lay.cruise) return [];
  const { a, b } = lay.cruise, L = dist(a, b), ux = (b[0] - a[0]) / L, uz = (b[1] - a[1]) / L;
  let nx = -uz, nz = ux;
  if (nz > 0) { nx = -nx; nz = -nz; }
  // The quay markers are not vessel centres: reserve each full hull plus a 30 m gap.
  const first = L * 0.27;
  return [[first, 298, 36], [first + (298 + 330) / 2 + 30, 330, 39]].map(([along, len, beam], i) => ({
    key: `cruise${i}`, len, beam: beam + 6, modelBeam: beam,
    x: a[0] + ux * along + nx * (beam / 2 + 6), z: a[1] + uz * along + nz * (beam / 2 + 6), th: Math.atan2(-uz, ux),
  }));
}

/** Escort positions are ship-local and keep their side throughout a turn. */
export function escortPoses(pose, size) {
  if (!pose?.tugs) return [];
  const side = pose.tugSide ?? 1, c = Math.cos(pose.th), s = Math.sin(pose.th);
  return [-1, 1].map(sign => {
    const x = sign * size.len * 0.3, z = side * (size.beam / 2 + 16);
    const ry = pose.lateral ? side * Math.PI / 2 : pose.reverse ? Math.PI : 0;
    return { x: pose.x + c * x + s * z, z: pose.z - s * x + c * z, th: pose.th + ry,
      localX: x, localZ: z, localHeading: ry, len: 30, beam: 11 };
  });
}

export function trafficBodies(pose, size) {
  return [{ ...pose, len: size.len, beam: size.beam }, ...escortPoses(pose, size)];
}

// Rounded polylines stay inside each corner's triangle (unlike an overshooting spline).
export function routePath(points, radius = 130) {
  const clean = points.filter((p, i) => !i || dist(p, points[i - 1]) > 0.01);
  const pts = [clean[0]];
  const line = end => {
    const start = pts.at(-1), n = Math.max(1, Math.ceil(dist(start, end) / 12));
    for (let i = 1; i <= n; i++) pts.push(point(start, end, i / n));
  };
  for (let i = 1; i < clean.length - 1; i++) {
    const a = clean[i - 1], b = clean[i], c = clean[i + 1];
    const trim = Math.min(radius, dist(a, b) * 0.4, dist(b, c) * 0.4);
    const p = point(b, a, trim / dist(a, b)), q = point(b, c, trim / dist(b, c));
    line(p);
    const n = Math.max(8, Math.ceil(trim / 5));
    for (let k = 1; k <= n; k++) {
      const u = k / n;
      pts.push(point(point(p, b, u), point(b, q, u), u));
    }
  }
  line(clean.at(-1));
  const lengths = [0];
  for (let i = 1; i < pts.length; i++) lengths.push(lengths.at(-1) + dist(pts[i - 1], pts[i]));
  const headings = pts.map((_, i) => heading(pts[Math.max(0, i - 1)], pts[Math.min(pts.length - 1, i + 1)]));
  const curvatures = [0];
  for (let i = 1; i < pts.length; i++) curvatures.push(Math.abs(Math.atan2(Math.sin(headings[i] - headings[i - 1]), Math.cos(headings[i] - headings[i - 1]))) / Math.max(0.00001, lengths[i] - lengths[i - 1]));
  return { pts, lengths, headings, curvatures, curvature: Math.max(...curvatures), length: lengths.at(-1) };
}

function onPath(path, u) {
  const d = clamp(u) * path.length;
  let lo = 1, hi = path.lengths.length - 1;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (path.lengths[mid] < d) lo = mid + 1; else hi = mid; }
  const a = path.pts[lo - 1], b = path.pts[lo];
  const f = (d - path.lengths[lo - 1]) / Math.max(0.00001, path.lengths[lo] - path.lengths[lo - 1]);
  const p = point(a, b, f);
  return { x: p[0], z: p[1], th: angle(path.headings[lo - 1], path.headings[lo], f) };
}

// No hull/escort corner travels more than two metres between clearance samples.
// The water envelope has 2 m padding and vessel separation has 6 m padding.
export function clearanceStep(s, size, t = null) {
  if (s.kind === 'hold') return Infinity;
  const radius = Math.hypot(size.len / 2, size.beam / 2 + (s.tugs ? 32 : 0));
  const duration = s.duration ?? s.t1 - s.t0;
  let curvature = s.path?.curvature || 0;
  if (t != null && s.path && s.th == null) {
    const p = s.path, d = smooth((t - s.t0) / duration) * p.length;
    const reach = p.length * 1.5 / duration * SAMPLE;
    let lo = 1, hi = p.lengths.length - 1;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (p.lengths[mid] < d) lo = mid + 1; else hi = mid; }
    curvature = 0;
    for (let i = lo; i < p.lengths.length && p.lengths[i - 1] <= d + reach; i++) curvature = Math.max(curvature, p.curvatures[i]);
  }
  const speed = s.kind === 'turn'
    ? radius * Math.abs(Math.atan2(Math.sin(s.th1 - s.th0), Math.cos(s.th1 - s.th0))) * 1.5 / duration
    : s.path.length * 1.5 / duration * (1 + (s.th == null ? radius * curvature : 0));
  return Math.min(SAMPLE, 2 / Math.max(1e-8, speed));
}

export function poseAt(schedule, t) {
  const segs = schedule?.segs || [];
  // Half-open intervals: docking and departure boundaries belong to the new phase.
  const s = segs.find(s => t >= s.t0 && t < s.t1);
  if (!s) return null;
  const u = clamp((t - s.t0) / (s.t1 - s.t0));
  if (s.kind === 'hold') return { x: s.x, z: s.z, th: s.th, phase: s.phase, u, speed: 0, tugs: false };
  if (s.kind === 'turn') return { x: s.x, z: s.z, th: angle(s.th0, s.th1, smooth(u)), phase: s.phase, u, speed: 0, tugs: true, tugSide: s.tugSide };
  const e = smooth(u);
  const p = onPath(s.path, e);
  if (s.th != null) p.th = s.th;
  else if (s.reverse) p.th += Math.PI;
  return { ...p, phase: s.phase, u, speed: s.path.length * 6 * u * (1 - u) / ((s.t1 - s.t0) * 3600), tugs: !!s.tugs, tugSide: s.tugSide, reverse: !!s.reverse, lateral: !!s.lateral };
}

/** Oriented hull safety envelopes, including a small clearance at bow/stern and sides. */
export function hullsOverlap(a, as, b, bs, gap = 6) {
  const dx = b.x - a.x, dz = b.z - a.z;
  if (Math.hypot(dx, dz) > (as.len + bs.len) / 2 + as.beam + bs.beam + gap) return false;
  const axes = p => [[Math.cos(p.th), -Math.sin(p.th)], [Math.sin(p.th), Math.cos(p.th)]];
  const A = axes(a), B = axes(b);
  for (const n of [...A, ...B]) {
    const project = (v, size) => Math.abs(n[0] * v[0][0] + n[1] * v[0][1]) * size.len / 2 + Math.abs(n[0] * v[1][0] + n[1] * v[1][1]) * size.beam / 2;
    if (Math.abs(dx * n[0] + dz * n[1]) >= project(A, as) + project(B, bs) + gap) return false;
  }
  return true;
}

export function createWaterMask(lay, twin) {
  const shore = new Map();
  const rings = twin.land.map(poly => poly.map(flat => {
    const pts = [];
    for (let i = 0; i < flat.length; i += 2) pts.push(lay.toLocal(flat[i], flat[i + 1]));
    const edges = new Map();
    for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
      const a = pts[i], b = pts[j];
      const edge = [a, b];
      for (let x = Math.floor(Math.min(a[0], b[0]) / 100); x <= Math.floor(Math.max(a[0], b[0]) / 100); x++) {
        for (let z = Math.floor(Math.min(a[1], b[1]) / 100); z <= Math.floor(Math.max(a[1], b[1]) / 100); z++) {
          const key = `${x},${z}`;
          if (!shore.has(key)) shore.set(key, []);
          shore.get(key).push(edge);
        }
      }
      for (let z = Math.floor(Math.min(a[1], b[1]) / 100); z <= Math.floor(Math.max(a[1], b[1]) / 100); z++) {
        if (!edges.has(z)) edges.set(z, []);
        edges.get(z).push([a, b]);
      }
    }
    return { pts, edges, x0: Math.min(...pts.map(p => p[0])), x1: Math.max(...pts.map(p => p[0])), z0: Math.min(...pts.map(p => p[1])), z1: Math.max(...pts.map(p => p[1])) };
  }));
  const inside = (x, z, r) => {
    if (x < r.x0 || x > r.x1 || z < r.z0 || z > r.z1) return false;
    let yes = false;
    for (const [[ax, az], [bx, bz]] of r.edges.get(Math.floor(z / 100)) || []) {
      if ((az > z) !== (bz > z) && x < (bx - ax) * (z - az) / (bz - az) + ax) yes = !yes;
    }
    return yes;
  };
  const water = (x, z) => !rings.some(poly => inside(x, z, poly[0]) && !poly.slice(1).some(r => inside(x, z, r)));
  return {
    water,
    fits(pose, size) {
      const c = Math.cos(pose.th), s = Math.sin(pose.th);
      if (!water(pose.x, pose.z)) return false;
      const hx = size.len / 2 + 2, hz = size.beam / 2 + 2;
      const bx = Math.abs(c) * hx + Math.abs(s) * hz, bz = Math.abs(s) * hx + Math.abs(c) * hz;
      const seen = new Set();
      // Segment/rectangle intersection catches thin piers and enclosed islands
      // between the old 27 hull probes, including coastline holes.
      for (let x = Math.floor((pose.x - bx) / 100); x <= Math.floor((pose.x + bx) / 100); x++) for (let z = Math.floor((pose.z - bz) / 100); z <= Math.floor((pose.z + bz) / 100); z++) {
        for (const edge of shore.get(`${x},${z}`) || []) {
          if (seen.has(edge)) continue;
          seen.add(edge);
          const local = ([X, Z]) => [(X - pose.x) * c - (Z - pose.z) * s, (X - pose.x) * s + (Z - pose.z) * c];
          const a = local(edge[0]), b = local(edge[1]);
          let lo = 0, hi = 1;
          for (const [axis, half] of [[0, hx], [1, hz]]) {
            const d = b[axis] - a[axis];
            if (Math.abs(d) < 1e-9) { if (Math.abs(a[axis]) > half) { hi = -1; break; } }
            else {
              const u = (-half - a[axis]) / d, v = (half - a[axis]) / d;
              lo = Math.max(lo, Math.min(u, v)); hi = Math.min(hi, Math.max(u, v));
            }
          }
          if (lo <= hi) return false;
        }
      }
      return true;
    },
  };
}

/** Every ship has a berth-sized offshore waiting slot; overflow grows out to sea, never reuses an occupied slot. */
export function anchorSlots(lay, vessels) {
  const A = lay.routes.anchorage, n = Math.hypot(...A.along), a = A.along.map(v => v / n);
  let p = [-a[1], a[0]];
  const sea = [lay.routes.sea[0] - A.centre[0], lay.routes.sea[1] - A.centre[1]];
  if (p[0] * sea[0] + p[1] * sea[1] < 0) p = p.map(v => -v);
  const pitch = Math.max(680, ...vessels.map(v => v.len + 220));
  const out = new Map();
  [...vessels].sort((a, b) => String(a.key).localeCompare(String(b.key))).forEach((v, i) => {
    const along = (i % 5 - 2) * pitch, across = Math.floor(i / 5) * pitch;
    const x = A.centre[0] + a[0] * along + p[0] * across, z = A.centre[1] + a[1] * along + p[1] * across;
    // Leave through the aisle between anchorage columns before joining the approach.
    // A direct chord to the pilot station would cut through the other waiting ships.
    const exit = [[x + p[0] * pitch / 2, z + p[1] * pitch / 2],
      [A.centre[0] - a[0] * pitch * 3.3 + p[0] * (across + pitch / 2), A.centre[1] - a[1] * pitch * 3.3 + p[1] * (across + pitch / 2)]];
    out.set(v.key, { x, z, th: Math.atan2(-p[1], p[0]), exit });
  });
  return out;
}

const hold = (t0, t1, p, phase) => ({ kind: 'hold', t0, t1, ...p, phase });
function travel(path, phase, { speed = 3, th = null, reverse = false, tugs = false, lateral = false } = {}) {
  // Smooth acceleration/deceleration: peak speed is 1.5 times the mean, capped in m/s.
  return { kind: 'path', path, duration: Math.max(0.015, 1.5 * path.length / (speed * 3600)), phase, th, reverse, tugs, lateral };
}

export function vesselRoute(lay, v, anchor) {
  const R = lay.routes, ms = R.mouthSide, th = ms > 0 ? Math.PI : 0;
  // APM's opposite quay has moored cruise ships projecting into the basin.
  // Keep the full cargo-ship/escort envelope on the container-terminal side.
  const lane = lay.quay === '24B' ? Math.min(R.lane, 170) : R.lane;
  const via = lay.quay === "36A" ? [lay.toLocal(620, -1120)] : R.via;
  // Approach parallel to the quay. No 180-degree pivot beside occupied berths.
  const gate = [R.entry[0], lane];
  const end = [v.x, lane];
  const sea = routePath([[anchor.x, anchor.z], ...anchor.exit, R.sea, R.mouthOut, R.mouthIn, ...via, R.entry, gate, [v.x + ms * Math.min(v.len, Math.abs(gate[0] - v.x) * 0.5), lane], end], Math.max(150, v.len * 0.7));
  const slide = routePath([end, [v.x, v.zMoor]]);
  const back = routePath([end, gate, R.entry], 100);
  const exit = routePath([R.entry, ...[...via].reverse(), R.mouthIn, R.mouthOut, R.sea], Math.max(150, v.len * 0.7));
  const inSegs = [travel(sea, 'berthing', { speed: 3.2 }), travel(slide, 'berthing', { speed: 0.45, th, tugs: true, lateral: true })];
  const outSegs = [travel(routePath([[v.x, v.zMoor], end]), 'departing', { speed: 0.45, th, tugs: true, lateral: true }), travel(back, 'departing', { speed: 1.4, th, reverse: true, tugs: true }), { kind: 'turn', duration: 0.14, x: R.entry[0], z: R.entry[1], th0: th, th1: onPath(exit, 0).th, phase: 'departing', tugs: true }, travel(exit, 'departing', { speed: 3.2 })];
  for (const segment of [...inSegs, ...outSegs]) segment.tugSide = ms > 0 ? -1 : 1;
  return { inSegs, outSegs, th, inDuration: inSegs.reduce((s, x) => s + x.duration, 0), outDuration: outSegs.reduce((s, x) => s + x.duration, 0) };
}

function stamp(parts, start) {
  let t = start;
  return parts.map(s => { const seg = { ...s, t0: t, t1: t + s.duration }; t = seg.t1; return seg; });
}
function bounds(s, size) {
  if (s.bounds) return s.bounds;
  const pts = s.path?.pts || [[s.x, s.z]];
  if (s.tugs) size = { ...size, beam: size.beam + 64 };
  const extent = s.kind === 'turn' || s.th == null ? Math.hypot(size.len, size.beam) / 2 + 6 : null;
  const xpad = extent ?? (Math.abs(Math.cos(s.th)) * size.len + Math.abs(Math.sin(s.th)) * size.beam) / 2 + 6;
  const zpad = extent ?? (Math.abs(Math.sin(s.th)) * size.len + Math.abs(Math.cos(s.th)) * size.beam) / 2 + 6;
  return s.bounds = { x0: Math.min(...pts.map(p=>p[0]))-xpad, x1: Math.max(...pts.map(p=>p[0]))+xpad, z0: Math.min(...pts.map(p=>p[1]))-zpad, z1: Math.max(...pts.map(p=>p[1]))+zpad };
}
export function schedulesConflict(a, av, b, bv) {
  const overlap = (ap, bp) => trafficBodies(ap, av).some(x => trafficBodies(bp, bv).some(y => hullsOverlap(x, x, y, y)));
  for (const sa of a.segs) for (const sb of b.segs) {
    const lo = Math.max(sa.t0, sb.t0), hi = Math.min(sa.t1, sb.t1);
    if (hi - lo < 1e-7) continue;
    const ab = bounds(sa, av), bb = bounds(sb, bv);
    if (ab.x1 < bb.x0 || bb.x1 < ab.x0 || ab.z1 < bb.z0 || bb.z1 < ab.z0) continue;
    for (let t = lo; t < hi; t += Math.min(hi - lo, clearanceStep(sa, av, t), clearanceStep(sb, bv, t))) {
      const ap = poseAt({ segs: [sa] }, t), bp = poseAt({ segs: [sb] }, t);
      if (overlap(ap, bp)) return true;
    }
    if (overlap(poseAt({ segs: [sa] }, hi - 1e-8), poseAt({ segs: [sb] }, hi - 1e-8))) return true;
  }
  return false;
}
function nextWindClear(windows, start, duration) {
  for (const [a, b] of windows) if (start < b && start + duration > a) start = b;
  return start;
}

export function buildTraffic(lay, vessels, task, horizon, mask = null) {
  const slots = anchorSlots(lay, vessels);
  const schedules = new Map(), accepted = [], movements = [], fixed = harbourObstacles(lay);
  for (const obstacle of fixed) {
    schedules.set(obstacle.key, { segs: [hold(-1e6, Infinity, obstacle, 'moored cruise ship')], visualDep: Infinity });
    accepted.push(obstacle);
  }
  // Work areas occupy water as well as quay sections. A delayed visual manoeuvre may
  // encounter a closure that was not active at its original planned hour.
  for (const [i, b] of (task.blocks || []).entries()) {
    if (b.kind !== 'closed') continue;
    const key = `closure${i}`, len = (b.last - b.first + 1) * lay.secM;
    const obstacle = { key, len, beam: 76, x: lay.secX(b.first) + len / 2, z: 42, th: 0 };
    schedules.set(key, { segs: [hold(b.start, b.end, obstacle, 'closed')], visualDep: b.end });
    accepted.push(obstacle);
  }
  const order = [...vessels].sort((a, b) => Number(!!b.preMoored) - Number(!!a.preMoored) || (a.preMoored && b.preMoored ? a.dep - b.dep : (a.berth ?? Infinity) - (b.berth ?? Infinity)) || String(a.key).localeCompare(String(b.key)));
  const waiting = (v, reason) => ({ segs: [hold(Math.min(v.arrival, v.berth ?? Infinity) - 2, Infinity, slots.get(v.key), reason ? 'plan blocked' : 'at anchor')], reason, visualBerth: null, delay: 0 });
  const heldAlongside = (v, route, reason) => ({ segs: [hold(-1e6, Infinity, { x: v.x, z: v.zMoor, th: route.th }, 'alongside')], visualBerth: v.berth, visualDep: null, delay: 0, reason });
  // An unplanned/blocked vessel still occupies water. Reserve all anchorages
  // before accepting routes, including ships processed later in the plan.
  for (const v of vessels) schedules.set(v.key, v.preMoored
    ? heldAlongside(v, { th: lay.routes.mouthSide > 0 ? Math.PI : 0 }, '')
    : waiting(v, v.reason));
  for (const v of order) {
    if (v.reason || v.berth == null) { schedules.set(v.key, waiting(v, v.reason)); continue; }
    const anchor = slots.get(v.key), route = vesselRoute(lay, v, anchor);
    const windows = noMoveWindows(task, { length_m: v.len });
    let dock = v.berth, result = null, departureWait = 0;
    let reason = '';
    // Pre-existing ships are immutable at hour zero. Other traffic yields to them.
    for (let attempt = 0; attempt < 160; attempt++) {
      let start = dock - route.inDuration;
      if (!v.preMoored) {
        while (true) {
          start = nextWindClear(windows, start, route.inDuration);
          const hit = movements.find(m => start < m[1] + 0.025 && start + route.inDuration > m[0] - 0.025);
          if (!hit) break;
          start = hit[1] + 0.025;
        }
        dock = Math.max(dock, start + route.inDuration);
      }
      let dep = Math.max(v.dep + (dock - v.berth), dock + 0.01, departureWait);
      dep = nextWindClear(windows, dep, route.outDuration);
      while (true) {
        const hit = movements.find(m => dep < m[1] + 0.025 && dep + route.outDuration > m[0] - 0.025);
        if (!hit) break;
        dep = nextWindClear(windows, hit[1] + 0.025, route.outDuration);
      }
      const segs = v.preMoored ? [] : [hold(Math.min(v.arrival - 2, start - 0.1), start, { ...anchor, th: onPath(route.inSegs[0].path, 0).th }, 'at anchor'), ...stamp(route.inSegs, start)];
      segs.push(hold(v.preMoored ? -1e6 : dock, dep, { x: v.x, z: v.zMoor, th: route.th }, 'alongside'), ...stamp(route.outSegs, dep));
      const candidate = { segs, visualBerth: dock, visualDep: dep, delay: dock - v.berth, reason: '' };
      if (attempt === 0) {
        // Check swept hull against the actual mapped land, not just its centre point.
        const underway = v.preMoored ? stamp(route.outSegs, 0) : [...stamp(route.inSegs, 0), ...stamp(route.outSegs, route.inDuration)];
        for (const s of underway) {
          const safe = t => trafficBodies(poseAt({ segs: [s] }, t), v).every(body => (!mask || mask.fits(body, body)) && !fixed.some(obstacle => hullsOverlap(body, body, obstacle, obstacle)));
          for (let t = s.t0; t < s.t1; t += clearanceStep(s, v, t)) {
            if (!safe(t)) { reason = 'route clearance'; break; }
          }
          if (!safe(s.t1 - 1e-8)) reason = 'route clearance';
          if (reason) break;
        }
        if (reason) {
          if (v.preMoored) result = heldAlongside(v, route, reason);
          break;
        }
      }
      const constraints = [...accepted, ...vessels.filter(b => b.key !== v.key && !accepted.some(a => a.key === b.key))];
      const collision = constraints.find(b => !(v.preMoored && b.key.startsWith('closure')) && schedulesConflict(candidate, v, schedules.get(b.key), b));
      if (collision) {
        const other = schedules.get(collision.key);
        if (!Number.isFinite(other.visualDep)) {
          reason = 'waiting for safe traffic clearance';
          if (v.preMoored) result = heldAlongside(v, route, reason);
          break;
        }
        // Release the occupied footprint before trying again, instead of giving up
        // after an arbitrary number of tiny time increments in a congested week.
        if (v.preMoored) departureWait = Math.max(dep + 0.1, other.visualDep + route.outDuration + 0.05);
        else dock = Math.max(dock + 0.1, other.visualDep + route.inDuration + 0.05);
        continue;
      }
      result = candidate;
      if (!v.preMoored) movements.push([dock - route.inDuration, dock]);
      movements.push([dep, dep + route.outDuration]);
      movements.sort((a, b) => a[0] - b[0]);
      break;
    }
    schedules.set(v.key, result || waiting(v, reason || 'waiting for safe traffic clearance'));
    if (result) accepted.push(v);
  }
  for (const v of accepted) if (v.key.startsWith("closure") || v.key.startsWith('cruise')) schedules.delete(v.key);
  return schedules;
}

/** Renderer-independent vessel specifications, shared by both viewers and regression tests. */
export function trafficVessels(lay, task, evaluation) {
  const vessels = evaluation.rows.map(row => {
    const beam = row.ship.beam_m || Math.max(16, row.ship.length_m / 7.2);
    return { key: `s${row.id}`, len: row.ship.length_m, beam, arrival: row.ship.arrival,
      berth: row.placed ? row.berth : null, dep: row.dep,
      x: row.placed ? lay.secX(row.section) + row.ship.sections * lay.secM / 2 : 0,
      zMoor: 6 + beam / 2,
      reason: row.conflicts.find(c => /overlaps|cannot arrive|quay has|no-movement window/.test(c)) };
  });
  (task.blocks || []).forEach((b, i) => {
    if (b.kind !== 'alongside') return;
    const len = (b.last - b.first + 1) * lay.secM - 14, beam = Math.max(18, len / 7.2);
    vessels.push({ key: `b${i}`, len, beam, arrival: b.start, berth: b.start, dep: b.end,
      x: (lay.secX(b.first) + lay.secX(b.last + 1)) / 2, zMoor: 6 + beam / 2, preMoored: b.start <= 0 });
  });
  return vessels;
}
