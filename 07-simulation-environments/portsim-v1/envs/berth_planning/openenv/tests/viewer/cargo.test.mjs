import assert from 'node:assert/strict';
import { test } from 'node:test';
import { CONTAINER, cargoState, makeCargoJob, dischargeOrder, planDischarges, carriersConflict, nextCargoStart } from '../../berth_openenv/web/cargo-handling.js';

const slot = { id: 0, x: 0, y: 25, z: 0, w: 2.44, t: 4, c: '#A94A39', b: 3, k: 1 };
const spec = { id: 's:0', ship: 's', slot, source: { x: 0, y: 25, z: 35, ry: 0 }, destination: { p: [50, 1.635, -110], via: [50, -70], ry: Math.PI / 2, travelY: 9.2 }, start: 100, laneZ: -23, roadZ: -50, clearY: 36 };
const job = makeCargoJob(spec);
const distance = (a, b) => Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);

test('a container has exactly one owner at each handoff boundary', () => {
  for (const [mark, before, after] of [['pickup', 'ship', 'crane'], ['release', 'crane', 'quay'], ['lock', 'quay', 'carrier'], ['stored', 'carrier', 'yard']]) {
    assert.equal(cargoState(job, job.mark[mark] - 0.001).owner, before);
    assert.equal(cargoState(job, job.mark[mark]).owner, after);
  }
  assert.equal(cargoState(job, 0).owner, 'ship');
  assert.equal(cargoState(job, 1e9).owner, 'yard');
});

test('position is continuous through pickup, hoisting, handover and placement', () => {
  for (const t of Object.values(job.mark)) {
    assert.ok(distance(cargoState(job, t - 1e-5).box, cargoState(job, t + 1e-5).box) < 0.001, `teleport at ${t}`);
  }
});

test('reverse-facing vessels keep container orientation through all transfers', () => {
  const reverse = makeCargoJob({ ...spec, source: { ...spec.source, ry: Math.PI } });
  for (const mark of ['pickup', 'release', 'lock', 'carrierLift', 'atYard', 'stored']) {
    const t = reverse.mark[mark];
    const a = cargoState(reverse, t - 1e-5).box, b = cargoState(reverse, t + 1e-5).box;
    assert.ok(Math.abs(Math.sin((a.ry - b.ry) / 2)) < 0.001, `rotation jump at ${mark}`);
  }
});

test('loaded trolley crosses only after the whole box clears the deck stack', () => {
  for (let t = job.mark.pickup; t < job.mark.release; t += 0.1) {
    const s = cargoState(job, t);
    if (s.box.z < spec.source.z - 0.01 && s.box.z > spec.laneZ + 0.01) assert.ok(s.box.y - CONTAINER.height / 2 > 30);
  }
  assert.equal(cargoState(job, job.mark.release).box.y - CONTAINER.height / 2, 0.3400000000000001);
});

test('loaded hoist, trolley and carrier stay within speed limits', () => {
  const dt = 0.01;
  for (let t = job.mark.pickup; t < job.mark.stored - dt; t += 0.1) {
    const a = cargoState(job, t), b = cargoState(job, t + dt);
    const speed = distance(a.box, b.box) / dt;
    const cap = t < job.mark.lift ? 1.11 : t < job.mark.cross ? 2.41 : t < job.mark.release ? 1.11 : t < job.mark.carrierLift ? 0.81 : t < job.mark.atYard ? 4.01 : 0.81;
    assert.ok(speed <= cap, `${a.phase}: ${speed} > ${cap}`);
  }
});

test('the carrier waits for STS clearance and lifts only after locking', () => {
  const a = cargoState(job, job.mark.clear - 0.01), b = cargoState(job, job.mark.clear);
  assert.ok(Math.hypot(a.carrier.x - job.source.x, a.carrier.z - job.laneZ) > 15);
  assert.equal(b.owner, 'quay');
  assert.equal(cargoState(job, job.mark.lock - 0.01).box.y, job.ground);
});

test('pause, forward seek and rewind produce the same cargo and preserve the manifest', () => {
  const before = JSON.stringify(job), t = job.mark.cross + 10, state = cargoState(job, t);
  cargoState(job, 1e8); cargoState(job, 0);
  assert.deepEqual(cargoState(job, t), state);
  assert.deepEqual(cargoState(job, t), cargoState(job, t));
  assert.equal(JSON.stringify(job), before);
  assert.equal(job.slot.c, '#A94A39'); assert.equal(job.slot.b, 3);
});

test('top-down discharge never picks through a remaining box', () => {
  const slots = Array.from({ length: 20 }, (_, id) => ({ ...slot, id, x: 10, z: id % 4 * 2.55, t: Math.floor(id / 4), y: 12 + Math.floor(id / 4) * 2.61 }));
  const order = dischargeOrder(slots, 10, -1), removed = new Set();
  for (const s of order) {
    assert.equal(slots.some(o => o.z === s.z && o.t > s.t && !removed.has(o.id)), false);
    removed.add(s.id);
  }
});

test('wind windows move the entire operation before it starts', () => {
  assert.equal(nextCargoStart(100, 50, [{ start: 120, end: 150 }, { start: 180, end: 220 }]), 220);
  assert.equal(nextCargoStart(10, 50, [{ start: 120, end: 150 }]), 10);
});

test('planner reserves unique storage, queues crossing routes, and finishes before departure', () => {
  const yard = Array.from({ length: 12 }, (_, i) => ({ p: [30 + (i % 4) * 12, 1.635, -90 - Math.floor(i / 4) * 15], via: [30 + (i % 4) * 12, -70], ry: Math.PI / 2, travelY: 9.2, corridor: `row${i % 4}`, depth: Math.floor(i / 4) * 15 }));
  const services = [0, 1].map(i => ({ ship: `s${i}`, start: 0, end: 7200, z: 35, side: 1, waterY: -3,
    lanes: [{ x: i * 45, clearY: 36, slots: Array.from({ length: 20 }, (_, id) => ({ ...slot, id, z: id * 2.55, y: 28 })) }] }));
  const plan = planDischarges(services, yard, { laneZ: -23, roadZ: -50, windows: [{ start: 1500, end: 2000 }] });
  assert.ok(plan.jobs.length >= 8);
  assert.equal(new Set(plan.jobs.map(j => j.destination.id)).size, plan.jobs.length);
  for (const j of plan.jobs) {
    assert.ok(j.mark.home < 7200);
    assert.ok(j.mark.home <= 1500 || j.mark.start >= 2000);
  }
  for (const j of plan.jobs) for (const other of plan.jobs) if (j !== other && j.destination.corridor === other.destination.corridor && j.destination.depth < other.destination.depth) assert.ok(j.mark.start >= other.mark.home, 'front row cannot be filled before rear access is complete');
  for (let i = 0; i < plan.jobs.length; i++) for (let k = 0; k < i; k++) assert.equal(carriersConflict(plan.jobs[i], plan.jobs[k]), false);
  assert.deepEqual(planDischarges(services, yard, { laneZ: -23, roadZ: -50, windows: [{ start: 1500, end: 2000 }] }), plan);
});
