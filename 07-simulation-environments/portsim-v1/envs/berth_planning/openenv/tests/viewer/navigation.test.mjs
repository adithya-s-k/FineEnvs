import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { siteFrame } from '../../berth_openenv/web/twin.js';
import { evaluatePlan, horizonOf, publishedPlan, noMoveWindows } from '../../berth_openenv/web/model.js';
import { buildTraffic, createWaterMask, poseAt, hullsOverlap, trafficVessels, anchorSlots, routePath, trafficBodies, harbourObstacles } from '../../berth_openenv/web/navigation.js';
import { craneTargets } from '../../berth_openenv/web/crane-allocation.js';

const web = new URL('../../berth_openenv/web/', import.meta.url);
const twin = JSON.parse(gunzipSync(readFileSync(new URL('twin/twin.json.gz', web))));
const tasks = readFileSync(new URL('../../../tasks/dock-v1-eval/tasks.jsonl', import.meta.url), 'utf8').trim().split('\n').map(JSON.parse);
const train = gunzipSync(readFileSync(new URL('../../../tasks/dock-v1-train/tasks.jsonl.gz', import.meta.url))).toString().trim().split('\n').map(JSON.parse);
const denseTraining = ['24B', '36A'].map(quay => train.filter(t => t.quay === quay).sort((a, b) => b.ships.length - a.ships.length)[0]);
function setup(task, plan = task.reference.optimal_plan) {
  const lay = siteFrame(twin, task);
  lay.secX = s => lay.X0 + (s - lay.first) * lay.secM;
  const evaluation = evaluatePlan(task, plan);
  const vessels = trafficVessels(lay, task, evaluation);
  const mask = createWaterMask(lay, twin);
  return { lay, evaluation, vessels, mask, schedules: buildTraffic(lay, vessels, task, horizonOf(task, [plan]), mask) };
}

test('docking/departure boundaries use the next phase; seeking is deterministic', () => {
  const schedule = { segs: [
    { kind: 'path', t0: 0, t1: 1, path: routePath([[0, 100], [0, 20]]), th: 0, phase: 'berthing', tugs: true },
    { kind: 'hold', t0: 1, t1: 3, x: 0, z: 20, th: 0, phase: 'alongside' },
    { kind: 'path', t0: 3, t1: 4, path: routePath([[0, 20], [0, 100]]), th: 0, phase: 'departing' },
  ] };
  assert.equal(poseAt(schedule, 1).phase, 'alongside');
  assert.equal(poseAt(schedule, 1).tugs, false);
  assert.equal(poseAt(schedule, 3).phase, 'departing');
  assert.equal(poseAt(schedule, 4), null);
  const first = poseAt(schedule, 0.4);
  poseAt(schedule, 3.8); poseAt(schedule, 0.1);
  assert.deepEqual(poseAt(schedule, 0.4), first);
  assert.ok(poseAt(schedule, 0.001).speed < poseAt(schedule, 0.5).speed);
});

test('waiting slots remain separated beyond the old 15-vessel anchorage capacity', () => {
  const { lay } = setup(tasks[0]);
  const ships = Array.from({ length: 100 }, (_, i) => ({ key: `s${i}`, len: 400, beam: 60 }));
  const slots = anchorSlots(lay, ships);
  for (let i = 0; i < ships.length; i++) for (let j = i + 1; j < ships.length; j++) {
    assert.equal(hullsOverlap(slots.get(ships[i].key), ships[i], slots.get(ships[j].key), ships[j], 50), false);
  }
});

test('invalid spatial assignments wait offshore and never mutate the submitted plan', () => {
  const task = tasks[0], plan = publishedPlan(task), before = JSON.stringify(plan);
  const { evaluation, schedules } = setup(task, plan);
  let held = 0;
  for (const row of evaluation.rows) if (row.conflicts.some(c => /overlaps|cannot arrive|quay has|no-movement window/.test(c))) {
    const s = schedules.get(`s${row.id}`);
    assert.ok(s.reason);
    assert.equal(s.visualBerth, null);
    assert.equal(poseAt(s, Math.max(row.ship.arrival, row.berth)).phase, 'plan blocked');
    held++;
  }
  assert.ok(held > 0);
  assert.equal(JSON.stringify(plan), before);
});

test('cranes serve cargo bays only, obey requested counts and retain rail clearance', () => {
  const ship = { id: 'large', x1: -180, x2: 180, zNear: 6, zFar: 50, deckY: 12, want: 3,
    bays: [-150, -120, -90, -60, 20, 50, 80, 110, 140].map(x => ({ x, top: 27 })) };
  const targets = craneTargets([ship], 9, { railX0: -500, railX1: 500 });
  assert.equal(targets.filter(t => t.working).length, 3);
  assert.equal(targets.length, 9);
  for (const t of targets.filter(t => t.working)) assert.ok(ship.bays.some(b => b.x === t.x));
  for (let i = 1; i < targets.length; i++) assert.ok(targets[i].x - targets[i - 1].x >= 29 - 1e-8);
  assert.equal(craneTargets([{ ...ship, want: 0 }], 9, { railX0: -500, railX1: 500 }).filter(t => t.working).length, 0);
});

// Real mapped geography and independent time samples catch collisions between transitions,
// not just hand-picked poses. Adaptive clearance checks happen inside the controller; this
// audit samples every minute plus every boundary, across both terminals and all four tiers.
for (const task of [...tasks, ...denseTraining]) test(`water, separation, wind and completion: ${task.task_id}`, () => {
  const { vessels, mask, schedules, lay } = setup(task);
  const times = new Set([0]);
  for (const v of vessels) {
    const s = schedules.get(v.key);
    assert.equal(s.reason, '', `${v.key} must complete a feasible reference plan`);
    assert.ok(s.visualBerth >= v.berth);
    assert.ok(s.visualDep >= v.dep);
    for (const seg of s.segs) {
      times.add(Math.max(0, seg.t0));
      if (seg.kind !== 'hold') {
        for (const [a, b] of noMoveWindows(task, { length_m: v.len })) assert.ok(seg.t1 <= a + 1e-7 || seg.t0 >= b - 1e-7, `${v.key} moves in wind window`);
        for (let t = Math.max(0, seg.t0); t < seg.t1; t += 1 / 60) times.add(t);
      }
    }
  }
  for (const t of times) {
    const active = vessels.map(v => ({ v, p: poseAt(schedules.get(v.key), t) })).filter(a => a.p);
    for (let i = 0; i < active.length; i++) {
      const a = active[i];
      for (const body of trafficBodies(a.p, a.v)) {
        assert.ok(mask.fits(body, body), `${a.v.key} hull or escort touches land at ${t}`);
        for (const fixed of harbourObstacles(lay)) assert.equal(hullsOverlap(body, body, fixed, fixed, 0), false, `${a.v.key} hits ${fixed.key} at ${t}`);
        for (let j = i + 1; j < active.length; j++) for (const other of trafficBodies(active[j].p, active[j].v)) assert.equal(hullsOverlap(body, body, other, other, 0), false, `${a.v.key}/${active[j].v.key} overlap at ${t}`);
      }
      // Ships initially alongside may clear the berth in the closure's first hour;
      // the renderer withholds its workboat until their footprint clears. New traffic yields.
      for (const b of task.blocks.filter(b => !a.v.preMoored && b.kind === 'closed' && t >= b.start && t < b.end)) {
        const len = (b.last - b.first + 1) * lay.secM;
        for (const body of trafficBodies(a.p, a.v)) assert.equal(hullsOverlap(body, body, { x: lay.secX(b.first) + len / 2, z: 42, th: 0 }, { len, beam: 76 }, 0), false, `${a.v.key} crosses a closure`);
      }
    }
  }
});
