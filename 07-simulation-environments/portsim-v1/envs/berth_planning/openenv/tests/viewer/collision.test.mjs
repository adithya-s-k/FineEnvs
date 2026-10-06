import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { siteFrame } from '../../berth_openenv/web/twin.js';
import { evaluatePlan, publishedPlan, horizonOf } from '../../berth_openenv/web/model.js';
import { createWaterMask, escortPoses, trafficBodies, harbourObstacles, hullsOverlap, poseAt, routePath, schedulesConflict, buildTraffic, trafficVessels, anchorSlots } from '../../berth_openenv/web/navigation.js';

const local = { toLocal: (x, z) => [x, z] };
test('whole hull detects a thin pier between the old sample points', () => {
  const mask = createWaterMask(local, { land: [[[10, -1, 12, -1, 12, 1, 10, 1]]] });
  assert.equal(mask.water(0, 0), true);
  assert.equal(mask.fits({ x: 0, z: 0, th: 0 }, { len: 200, beam: 30 }), false);
  assert.equal(mask.fits({ x: 0, z: 25, th: 0 }, { len: 200, beam: 30 }), true);
});

test('shoreline holes stay navigable while their boundaries remain solid', () => {
  const mask = createWaterMask(local, { land: [[[-500, -500, 500, -500, 500, 500, -500, 500], [-40, -40, 40, -40, 40, 40, -40, 40]]] });
  assert.equal(mask.fits({ x: 0, z: 0, th: 0.6 }, { len: 40, beam: 10 }), true);
  assert.equal(mask.fits({ x: 0, z: 0, th: 0.6 }, { len: 100, beam: 10 }), false);
});

test('a collision between five-second snapshots is still rejected', () => {
  const stationary = { segs: [{ kind: 'hold', t0: 0, t1: 1, x: 0, z: 0, th: 0, phase: 'alongside' }] };
  const crossing = { segs: [{ kind: 'path', t0: 0, t1: 5 / 3600, path: routePath([[-25, 0], [25, 0]]), phase: 'departing' }] };
  assert.equal(schedulesConflict(stationary, { len: 4, beam: 3 }, crossing, { len: 4, beam: 3 }), true);
});

test('escorts do not jump across the hull when a turn crosses 90 degrees', () => {
  const size = { len: 300, beam: 46 }, p = { x: 0, z: 0, tugs: true, tugSide: 1 };
  const a = escortPoses({ ...p, th: Math.PI / 2 - 1e-6 }, size), b = escortPoses({ ...p, th: Math.PI / 2 + 1e-6 }, size);
  for (let i = 0; i < 2; i++) {
    assert.equal(a[i].localZ, b[i].localZ);
    assert.ok(Math.hypot(a[i].x - b[i].x, a[i].z - b[i].z) < 0.001);
  }
});

const syntheticLay = { quay: 'test', secX: x => x * 40, routes: { mouthSide: -1, lane: 200, entry: [-500, 200], via: [], mouthIn: [-1000, 500], mouthOut: [-1400, 700], sea: [-2000, 1700], anchorage: { centre: [0, 2500], along: [1, 0] } } };
test('an already-moored ship waits safely if its departure route is obstructed', () => {
  const vessel = { key: 'fixed', len: 200, beam: 30, arrival: 0, berth: 0, dep: 5, x: 0, zMoor: 22, preMoored: true };
  const schedule = buildTraffic(syntheticLay, [vessel], {}, 24, { fits: () => false }).get('fixed');
  assert.equal(schedule.reason, 'route clearance');
  assert.equal(schedule.visualDep, null);
  assert.equal(poseAt(schedule, 100).phase, 'alongside');
  assert.equal(poseAt(schedule, 100).x, vessel.x);
});

test('an unplanned ship remains a collision obstacle even when processed last', () => {
  const vessels = [
    { key: 'a', len: 100, beam: 20, arrival: 10, berth: 10, dep: 12, x: 0, zMoor: 20 },
    { key: 'z', len: 100, beam: 20, arrival: 0, berth: null, x: 0, zMoor: 20 },
  ];
  const waiting = anchorSlots(syntheticLay, vessels).get('z');
  vessels[0].x = waiting.x; vessels[0].zMoor = waiting.z;
  const schedules = buildTraffic(syntheticLay, vessels, {}, 24);
  assert.equal(schedules.get('a').visualBerth, null);
  assert.equal(poseAt(schedules.get('a'), 11).phase, 'plan blocked');
});

const web = new URL('../../berth_openenv/web/', import.meta.url);
const twin = JSON.parse(gunzipSync(readFileSync(new URL('twin/twin.json.gz', web))));
const tasks = readFileSync(new URL('../../../tasks/dock-v1-eval/tasks.jsonl', import.meta.url), 'utf8').trim().split('\n').map(JSON.parse);
test('decorative cruise ships have separate berths and clear the shoreline', () => {
  const lay = siteFrame(twin, tasks.find(t => t.quay === '24B'));
  const mask = createWaterMask(lay, twin), ships = harbourObstacles(lay);
  assert.equal(ships.length, 2);
  for (const ship of ships) assert.ok(mask.fits(ship, ship), `${ship.key} clips the cruise quay`);
  assert.equal(hullsOverlap(ships[0], ships[0], ships[1], ships[1], 20), false);
});

for (const id of ['dock-24B-w06x1-busy-0', 'dock-36A-w05x1-busy-0']) for (const variant of ['optimal', 'published']) test(`hulls, cruise ships and escorts: ${id} / ${variant}`, () => {
  const task = tasks.find(t => t.task_id === id), plan = variant === 'optimal' ? task.reference.optimal_plan : publishedPlan(task);
  const lay = siteFrame(twin, task); lay.secX = s => lay.X0 + (s - lay.first) * lay.secM;
  const mask = createWaterMask(lay, twin), vessels = trafficVessels(lay, task, evaluatePlan(task, plan));
  const schedules = buildTraffic(lay, vessels, task, horizonOf(task, [plan]), mask), fixed = harbourObstacles(lay);
  const times = new Set([7.23, 127.8]); // recorded pre-fix cruise collisions
  for (const schedule of schedules.values()) for (const s of schedule.segs) if (s.kind !== 'hold') {
    times.add(Math.max(0, s.t0)); times.add(s.t1 - 1e-7);
    for (let t = Math.max(0, s.t0); t < s.t1; t += 1 / 1200) times.add(t); // independent 3-second audit
  }
  for (const t of times) {
    const fleet = vessels.map(v => ({ v, p: poseAt(schedules.get(v.key), t) })).filter(a => a.p);
    for (let i = 0; i < fleet.length; i++) {
      const a = fleet[i];
      for (const body of trafficBodies(a.p, a.v)) {
        assert.ok(mask.fits(body, body), `${a.v.key} clips shore at ${t}`);
        for (const obstacle of fixed) assert.equal(hullsOverlap(body, body, obstacle, obstacle, 0), false, `${a.v.key} hits ${obstacle.key} at ${t}`);
        for (const b of fleet.slice(i + 1)) for (const other of trafficBodies(b.p, b.v)) assert.equal(hullsOverlap(body, body, other, other, 0), false, `${a.v.key} / ${b.v.key} at ${t}`);
      }
    }
  }
});
