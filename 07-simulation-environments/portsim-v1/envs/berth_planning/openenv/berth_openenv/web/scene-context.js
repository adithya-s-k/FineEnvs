// Small-scale activity on mapped parking areas and street verges. These details
// are illustrative; the footprints, roads and shoreline remain the map's geometry.
import * as THREE from "three";
import { mulberry32 } from "./model.js";
import { Parts, instanced, mat, vcMat } from "./scene-kit.js";
import { coverAt, surfaceAt } from "./twin.js";

function inside(x, z, ring) {
  let hit = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [ax, az] = ring[i], [bx, bz] = ring[j];
    if ((az > z) !== (bz > z) && x < (bx - ax) * (z - az) / (bz - az) + ax) hit = !hit;
  }
  return hit;
}

/** Spatial lookup prevents scenery details intersecting mapped buildings. */
function buildingMask(lay, twin) {
  const grid = new Map(), size = 50;
  for (const b of twin.buildings) {
    const ring = [];
    for (let i = 3; i < b.length; i += 2) ring.push([b[i], b[i + 1]]);
    const local = lay.toLocal(...ring[0]);
    if (Math.hypot(...local) > 4600) continue;
    const x0 = Math.floor(Math.min(...ring.map(p => p[0])) / size), x1 = Math.floor(Math.max(...ring.map(p => p[0])) / size);
    const z0 = Math.floor(Math.min(...ring.map(p => p[1])) / size), z1 = Math.floor(Math.max(...ring.map(p => p[1])) / size);
    for (let x = x0; x <= x1; x++) for (let z = z0; z <= z1; z++) {
      const key = `${x},${z}`;
      if (!grid.has(key)) grid.set(key, []);
      grid.get(key).push(ring);
    }
  }
  return (x, z) => (grid.get(`${Math.floor(x / size)},${Math.floor(z / size)}`) || []).some(r => inside(x, z, r));
}

/** Keep planted verges and parked cars clear of the mapped carriageways. */
function roadMask(lay, twin) {
  const grid = new Map(), size = 50;
  for (const rd of twin.roads) {
    if (rd[1]) continue;
    const near = lay.toLocal(rd[2], rd[3]);
    if (Math.hypot(...near) > 4800) continue;
    for (let i = 2; i + 3 < rd.length; i += 2) {
      const a = [rd[i], rd[i + 1]], b = [rd[i + 2], rd[i + 3]], margin = rd[0] / 2 + 5;
      const segment = { a, b, half: rd[0] / 2 };
      for (let x = Math.floor((Math.min(a[0], b[0]) - margin) / size); x <= Math.floor((Math.max(a[0], b[0]) + margin) / size); x++) {
        for (let y = Math.floor((Math.min(a[1], b[1]) - margin) / size); y <= Math.floor((Math.max(a[1], b[1]) + margin) / size); y++) {
          const key = `${x},${y}`;
          if (!grid.has(key)) grid.set(key, []);
          grid.get(key).push(segment);
        }
      }
    }
  }
  return (x, y, radius) => (grid.get(`${Math.floor(x / size)},${Math.floor(y / size)}`) || []).some(({ a, b, half }) => {
    const dx = b[0] - a[0], dy = b[1] - a[1];
    const t = Math.max(0, Math.min(1, ((x - a[0]) * dx + (y - a[1]) * dy) / Math.max(0.001, dx * dx + dy * dy)));
    return Math.hypot(x - a[0] - dx * t, y - a[1] - dy * t) < half + radius;
  });
}

export function buildContext(lay, twin, { lite = false } = {}) {
  const root = new THREE.Group(), fixtures = new Parts();
  const blocked = buildingMask(lay, twin), onRoad = roadMask(lay, twin), random = mulberry32(4106);
  const ground = (X, Y) => surfaceAt(twin, X, Y, lite ? 2 : 1);
  const cars = [], glass = [], trees = [], trunks = [];
  const colors = ["#E4E5DF", "#C1C7C9", "#404E59", "#657F8A", "#C4C4B7", "#9C5145"];
  const nearest = twin.parking.map(flat => {
    const ring = [];
    for (let i = 0; i < flat.length; i += 2) ring.push([flat[i], flat[i + 1]]);
    const p = lay.toLocal(...ring[0]);
    return { ring, distance: Math.hypot(...p) };
  }).filter(p => p.distance < (lite ? 2000 : 3800)).sort((a, b) => a.distance - b.distance).slice(0, lite ? 15 : 45);
  for (const { ring } of nearest) {
    let longest = 0, ax = [1, 0];
    for (let i = 0; i < ring.length; i++) {
      const a = ring[i], b = ring[(i + 1) % ring.length];
      const L = Math.hypot(b[0] - a[0], b[1] - a[1]);
      if (L > longest) { longest = L; ax = [(b[0] - a[0]) / L, (b[1] - a[1]) / L]; }
    }
    const nx = [-ax[1], ax[0]], us = ring.map(p => p[0] * ax[0] + p[1] * ax[1]), vs = ring.map(p => p[0] * nx[0] + p[1] * nx[1]);
    const direction = lay.toLocal(lay.O[0] + nx[0], lay.O[1] + nx[1]);
    const angle = Math.atan2(-direction[1], direction[0]);
    for (let v = Math.min(...vs) + 4; v < Math.max(...vs) - 4; v += 14) {
      for (let u = Math.min(...us) + 3; u < Math.max(...us) - 3; u += 2.8) {
        const X = ax[0] * u + nx[0] * v, Y = ax[1] * u + nx[1] * v;
        if (![-1.4, 1.4].every(du => [-2.8, 2.8].every(dv => inside(X + ax[0] * du + nx[0] * dv, Y + ax[1] * du + nx[1] * dv, ring)))) continue;
        if ([-1, 0, 1].some(du => [-2.4, 0, 2.4].some(dv => blocked(X + ax[0] * du + nx[0] * dv, Y + ax[1] * du + nx[1] * dv))) || onRoad(X, Y, 2.4) || coverAt(twin, X, Y) === "water") continue;
        const y = ground(X, Y);
        if (Math.abs(ground(X + nx[0] * 3, Y + nx[1] * 3) - y) > 0.5) continue;
        const [x, z] = lay.toLocal(X, Y);
        const [mx, mz] = lay.toLocal(X + ax[0] * 1.3, Y + ax[1] * 1.3);
        if (!lite) fixtures.box(mx, y + 0.23, mz, 5, 0.025, 0.1, "#DADBD1", angle);
        if (random() > 0.67 || cars.length >= (lite ? 600 : 3000)) continue;
        cars.push({ p: [x, y + 0.65, z], s: [4.4, 0.85, 1.85], r: angle, c: colors[Math.floor(random() * colors.length)] });
        glass.push({ p: [x, y + 1.28, z], s: [2.4, 0.55, 1.65], r: angle, c: "#354A56" });
      }
    }
  }

  // Boulevard trees follow existing roads; every canopy has space beside the road
  // and outside building footprints. A stable grid prevents duplicate plantings.
  const occupied = new Set();
  for (const road of twin.roads) {
    if (road[0] < 7 || road[0] > 15 || road[1]) continue;
    for (let i = 2; i + 3 < road.length; i += 2) {
      const a = [road[i], road[i + 1]], b = [road[i + 2], road[i + 3]];
      const L = Math.hypot(b[0] - a[0], b[1] - a[1]);
      if (L < 18) continue;
      const nx = -(b[1] - a[1]) / L, ny = (b[0] - a[0]) / L;
      for (let d = 12; d < L - 5; d += lite ? 55 : 32) for (const side of [-1, 1]) {
        const X = a[0] + (b[0] - a[0]) * d / L + nx * (road[0] / 2 + 2.6) * side;
        const Y = a[1] + (b[1] - a[1]) * d / L + ny * (road[0] / 2 + 2.6) * side;
        const [x, z] = lay.toLocal(X, Y), cls = coverAt(twin, X, Y);
        if (Math.hypot(x, z) > (lite ? 2400 : 4000) || !["urban", "residential", "commercial", "park"].includes(cls)) continue;
        if ([[0, 0], [-2, 0], [2, 0], [0, -2], [0, 2]].some(([dx, dy]) => blocked(X + dx, Y + dy))) continue;
        const cell = `${Math.round(X / 12)},${Math.round(Y / 12)}`;
        if (onRoad(X, Y, 0.8) || occupied.has(cell) || trees.length >= (lite ? 1200 : 5500)) continue;
        occupied.add(cell);
        const y = ground(X, Y), h = 6.5 + random() * 2;
        trees.push({ p: [x, y + h * 0.7, z], s: [4.2, h * 0.65, 4.2], c: random() < 0.5 ? "#5F7847" : "#72844C", r: random() * 6.28 });
        trunks.push({ p: [x, y + h * 0.27, z], s: [0.28, h * 0.54, 0.28], c: "#7E7460" });
      }
    }
  }
  if (cars.length) root.add(instanced(new THREE.BoxGeometry(1, 1, 1), mat("#FFFFFF", { rough: 0.4, metal: 0.15 }), cars));
  if (glass.length) root.add(instanced(new THREE.BoxGeometry(1, 1, 1), mat("#FFFFFF", { rough: 0.2, metal: 0.3 }), glass));
  if (trees.length) root.add(instanced(new THREE.IcosahedronGeometry(0.6, 1), mat("#FFFFFF", { rough: 1 }), trees, { cast: !lite, receive: true }));
  if (trunks.length) root.add(instanced(new THREE.CylinderGeometry(0.5, 0.65, 1, 5), mat("#FFFFFF"), trunks, { cast: false }));
  root.add(fixtures.mesh(vcMat, { cast: false, receive: true }));
  return root;
}
