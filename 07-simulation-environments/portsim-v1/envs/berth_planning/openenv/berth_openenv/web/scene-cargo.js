import * as THREE from 'three';
import { Parts, C, containerMesh, setInstance, vcMat } from './scene-kit.js';
import { cargoState, CONTAINER } from './cargo-handling.js';

function carrierGeometry(tall) {
  const p = new Parts(), h = tall ? 14.6 : 9.4;
  for (const x of [-5.3, 5.3]) for (const z of [-2.25, 2.25]) {
    p.box(x, h / 2, z, 0.65, h, 0.65, '#E4E3DE');
    for (const dx of [-0.65, 0.65]) p.cylZ(x + dx, 0.72, z, 0.72, 0.52, '#202427', 12);
    p.box(x, 1.25, z, 3.1, 0.7, 0.9, '#47525A');
  }
  for (const z of [-2.25, 2.25]) p.box(0, h - 0.5, z, 13.3, 1.1, 0.85, '#E4E3DE');
  for (const x of [-5.3, 5.3]) p.box(x, h - 0.5, 0, 0.85, 1.1, 4.8, '#E4E3DE');
  p.box(-2, h + 0.55, 0, 3, 1.2, 2.3, '#596773');
  p.box(5, h + 0.7, -1.8, 2.3, 2, 1.7, '#EAE8DF');
  p.box(6.16, h + 0.8, -1.8, 0.08, 1.2, 1.5, C.window);
  p.cyl(5, h + 2, -1.8, 0.18, 0.25, '#F2AA37', 8);
  for (const z of [-2.25, 2.25]) p.box(6.88, 1.3, z, 0.08, 0.3, 0.3, '#FFF2C7');
  return { geometry: p.geometry(), height: h };
}

/** Render every discharged box with its original colour, brand, ink and dimensions. */
export function buildCargoReplay(plan, actors, tall) {
  const root = new THREE.Group();
  const { jobs, lanes } = plan;
  const boxes = containerMesh(jobs.map(j => ({ p: [0, -100, 0], s: [0, 0, 0], c: j.slot.c, b: j.slot.b, k: j.slot.k })));
  boxes.geometry.userData.shared = false;
  boxes.frustumCulled = false;
  root.add(boxes);
  const shape = carrierGeometry(tall);
  const carriers = new THREE.InstancedMesh(shape.geometry, vcMat, Math.max(1, lanes.length));
  carriers.count = lanes.length; carriers.frustumCulled = false; carriers.castShadow = true; carriers.receiveShadow = true;
  root.add(carriers);
  const spreaderParts = new Parts();
  for (const z of [-1.1, 1.1]) spreaderParts.box(0, 0, z, 12.4, 0.28, 0.2, '#E7B335');
  for (const x of [-5.95, 5.95]) spreaderParts.box(x, 0, 0, 0.25, 0.28, 2.4, '#E7B335');
  const spreaders = new THREE.InstancedMesh(spreaderParts.geometry(), vcMat, Math.max(1, lanes.length));
  spreaders.count = lanes.length; spreaders.frustumCulled = false; spreaders.castShadow = true;
  const ropes = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.045, 0.045, 1, 5), new THREE.MeshStandardMaterial({ color: '#363C40', metalness: 0.5, roughness: 0.7 }), Math.max(1, lanes.length * 4));
  ropes.count = lanes.length * 4; ropes.frustumCulled = false;
  root.add(spreaders, ropes);
  let lastTime = null;
  const craneStates = new Map();
  const summary = { lifting: 0, quay: 0, transit: 0, stored: 0, phase: '' };
  function update(t) {
    if (t === lastTime) return { craneStates, summary };
    lastTime = t;
    const seconds = t * 3600, removed = new Map([...actors.keys()].map(k => [k, new Set()]));
    Object.assign(summary, { lifting: 0, quay: 0, transit: 0, stored: 0, phase: '' });
    for (let i = 0; i < jobs.length; i++) {
      const j = jobs[i], s = cargoState(j, seconds), b = s.box;
      if (s.owner === 'ship') setInstance(boxes, i, 0, -100, 0, 0, 0, 0, 0);
      else {
        removed.get(j.ship)?.add(j.slot.id);
        setInstance(boxes, i, b.x, b.y, b.z, b.ry, CONTAINER.length, CONTAINER.height, Math.min(CONTAINER.width, j.slot.w));
      }
      if (s.owner === 'crane') { summary.lifting++; summary.phase ||= s.phase; }
      else if (s.owner === 'quay') summary.quay++;
      else if (s.owner === 'carrier') { summary.transit++; summary.phase ||= s.phase; }
      else if (s.owner === 'yard') summary.stored++;
    }
    for (const [id, actor] of actors) actor.model.setDischarged(removed.get(id));
    craneStates.clear();
    lanes.forEach((lane, i) => {
      const current = lane.jobs.find(j => seconds >= j.mark.start && seconds < j.mark.home);
      const next = lane.jobs.find(j => seconds < j.mark.start);
      const j = current || next || lane.jobs.at(-1);
      const key = `${lane.ship}:${lane.x.toFixed(3)}`;
      if (!j) {
        setInstance(carriers, i, 0, -100, 0, 0, 0, 0, 0);
        setInstance(spreaders, i, 0, -100, 0, 0, 0, 0, 0);
        for (let k = 0; k < 4; k++) setInstance(ropes, i * 4 + k, 0, -100, 0, 0, 0, 0, 0);
        return;
      }
      const state = cargoState(j, seconds);
      const activeLift = !!current && seconds <= j.mark.clear;
      // The empty trolley returns high to the next row; it never cuts through stacks.
      let crane = state.crane;
      if (!activeLift) {
        const previous = lane.jobs.filter(v => v.mark.clear <= seconds).at(-1);
        const targetZ = next?.source.z ?? j.laneZ;
        const u = previous ? Math.min(1, Math.max(0, (seconds - previous.mark.clear) / 35)) : 1;
        crane = { x: lane.x, y: j.clearY, z: previous ? previous.laneZ + (targetZ - previous.laneZ) * u * u * (3 - 2 * u) : targetZ };
      }
      craneStates.set(key, { ...crane, active: activeLift, phase: activeLift ? state.phase : 'Waiting for next lift' });
      // One carrier per crane lane, with continuous parking between jobs.
      const c = state.carrier || { ...cargoState(j, j.mark.start).carrier, hoistY: j.travelY };
      const visible = seconds >= lane.start && seconds <= lane.end;
      if (!visible || !c) {
        setInstance(carriers, i, 0, -100, 0, 0, 0, 0, 0);
        setInstance(spreaders, i, 0, -100, 0, 0, 0, 0, 0);
        for (let k = 0; k < 4; k++) setInstance(ropes, i * 4 + k, 0, -100, 0, 0, 0, 0, 0);
        return;
      }
      setInstance(carriers, i, c.x, 0.34, c.z, c.ry);
      const y = c.hoistY + CONTAINER.height / 2 + 0.17;
      setInstance(spreaders, i, c.x, y, c.z, c.ry);
      const length = Math.max(0.05, shape.height - 0.6 - y), co = Math.cos(c.ry), si = Math.sin(c.ry);
      let k = 0;
      for (const x of [-5.3, 5.3]) for (const z of [-1, 1]) setInstance(ropes, i * 4 + k++, c.x + co * x + si * z, y + length / 2, c.z - si * x + co * z, 0, 1, length, 1);
    });
    for (const mesh of [boxes, carriers, spreaders, ropes]) mesh.instanceMatrix.needsUpdate = true;
    return { craneStates, summary };
  }
  return { root, update, jobs, lanes };
}
