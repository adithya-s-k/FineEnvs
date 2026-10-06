// The static port around a quay, as a digital twin of the Port of Barcelona: real coastline, terrain, buildings,
// tanks and landmarks from the twin layers (OpenStreetMap, Sentinel-2, terrain tiles), and the terminal itself built
// from what the terminals have: BEST (quay 36A) with 34 automated stacking blocks, blue stacking cranes, shuttle
// carriers and an on-dock rail terminal; APM (quay 24B) with a straddle-carrier yard, the CLH tank farm and
// Montjuic behind it and cruise ships at the Moll Adossat across the channel.
import * as THREE from "three";
import { buildContext } from "./scene-context.js";
import { harbourObstacles } from './navigation.js';
import { mulberry32 } from "./model.js";
import { NO_REFLECT, WATER_Y } from "./scene-env.js";
import { coverAt, siteFrame, surfaceAt, terrainAt } from "./twin.js";
import { C, Parts, boxSpec, clamp, concreteTexture, containerMesh, instanced, mat, setInstance, smooth, steelMat, textTexture, vcMat } from "./scene-kit.js";

export { WATER_Y };
export const BOX_L = 12.19;
export const BOX_W = 2.44;
export const BOX_H = 2.59;
export const GAUGE = 30.48; // STS rail gauge (100 ft)

/** Layout of a task's quay in the scene frame (see twin.js). */
export function layoutFor(task, twin) {
  const f = siteFrame(twin, task);
  const n = f.last - f.first + 1;
  const seaRail = f.railC + GAUGE / 2;
  const landRail = f.railC - GAUGE / 2;
  return {
    ...f,
    n,
    secX: (s) => f.X0 + (s - f.first) * f.secM,
    seaRail,
    landRail,
    craneLane: (seaRail + landRail) / 2 - 2,
    // camera bounds: the quay, its basin and what lies just behind the yard
    zWater: f.quay === "24B" ? 440 : 900,
  };
}

const V2 = (x, y) => new THREE.Vector2(x, y);

function colorFromInt(v) {
  return [((v >> 16) & 255) / 255, ((v >> 8) & 255) / 255, (v & 255) / 255];
}

const _c = new THREE.Color();
function lin(hex) {
  _c.set(hex);
  return [_c.r, _c.g, _c.b];
}
function srgbToLin(rgb) {
  _c.setRGB(rgb[0], rgb[1], rgb[2], THREE.SRGBColorSpace);
  return [_c.r, _c.g, _c.b];
}
const mix3 = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];

/** Simple growable geometry with positions, colours and (optional) uvs. */
class Geo {
  constructor(uv = false) {
    this.p = [];
    this.c = [];
    this.uv = uv ? [] : null;
  }
  v(x, y, z, col, u, w) {
    this.p.push(x, y, z);
    this.c.push(col[0], col[1], col[2]);
    if (this.uv) this.uv.push(u || 0, w || 0);
  }
  tri(a, b, c, col) {
    this.v(...a, col);
    this.v(...b, col);
    this.v(...c, col);
  }
  quad(a, b, c, d, col) {
    this.tri(a, b, c, col);
    this.tri(a, c, d, col);
  }
  build() {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(this.p, 3));
    g.setAttribute("color", new THREE.Float32BufferAttribute(this.c, 3));
    if (this.uv) g.setAttribute("uv", new THREE.Float32BufferAttribute(this.uv, 2));
    g.computeVertexNormals();
    return g;
  }
}

function pointInPoly(x, z, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, zi] = poly[i];
    const [xj, zj] = poly[j];
    if (zi > z !== zj > z && x < ((xj - xi) * (z - zi)) / (zj - zi + 1e-9) + xi) inside = !inside;
  }
  return inside;
}

/* ================================================================ the twin around the quay */

function groundMaterial(twin) {
  // A georeferenced material retains real streets and roof grain at every zoom.
  const atlas = new THREE.Texture(twin.surface);
  atlas.colorSpace = THREE.SRGBColorSpace;
  atlas.anisotropy = 8;
  atlas.needsUpdate = true;
  const [x0, y0, x1, y1] = twin.frame.extent;
  atlas.repeat.set(DETAIL / (x1 - x0), DETAIL / (y1 - y0));
  atlas.offset.set(-x0 / (x1 - x0), -y0 / (y1 - y0));
  const m = new THREE.MeshStandardMaterial({ map: atlas, roughness: 0.98, metalness: 0, side: THREE.DoubleSide });
  return m;
}

const DETAIL = 70; // metres per detail-texture tile

/** Flat land from the real coastline, in the colour of streets and paving. */
function buildLand(lay, twin, gmat) {
  const g = new Geo(true);
  const T = lay.toLocal;
  const col = [1, 1, 1];
  for (const poly of twin.land) addPolygon(g, poly, T, 0, col);
  const m = new THREE.Mesh(g.build(), gmat);
  m.receiveShadow = true;
  return m;
}

function addPolygon(g, poly, T, y, col) {
  const rings = poly.map((r) => {
    const pts = [];
    for (let i = 0; i < r.length; i += 2) pts.push(V2(r[i], r[i + 1]));
    return pts;
  });
  if (rings[0].length < 3) return;
  const tris = THREE.ShapeUtils.triangulateShape(rings[0], rings.slice(1));
  const all = rings.flat();
  const loc = all.map((p) => T(p.x, p.y));
  for (const [a, b, c] of tris) {
    for (const k of [a, c, b]) g.v(loc[k][0], y, loc[k][1], col, all[k].x / DETAIL, all[k].y / DETAIL);
  }
}

/** Mapped roads draped over the terrain, with kerbs, lane markings and bridge decks. */
function buildRoads(lay, twin, lite) {
  const g = new Geo();
  const ground = (X, Y) => surfaceAt(twin, X, Y, lite ? 2 : 1);
  const piers = new Parts();
  const asph = lin("#646D70"), path = lin("#B7A584"), edge = lin("#B7B6AA"), side = lin("#ACADA5"), paint = lin("#D8D8C8");
  for (const rd of twin.roads) {
    const [w, layer] = rd;
    const raw = [];
    for (let i = 2; i < rd.length; i += 2) raw.push([rd[i], rd[i + 1]]);
    const centre = raw[Math.floor(raw.length / 2)];
    const local = lay.toLocal(...centre);
    const dist = Math.hypot(local[0], local[1]);
    if (dist > (lite ? 3800 : 7500)) continue; // the atlas retains distant streets
    const pts = [];
    let along = 0;
    for (let i = 0; i + 1 < raw.length; i++) {
      const a = raw[i], b = raw[i + 1];
      const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
      const count = Math.max(1, Math.ceil(len / (lite ? 18 : 9)));
      for (let k = 0; k < count; k++) {
        const u = k / count;
        pts.push([a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u, along + len * u]);
      }
      along += len;
    }
    pts.push([...raw[raw.length - 1], along]);
    // Elevated segments interpolate their deck height and do not dive into ravines.
    const bridgeY0 = ground(...raw[0]) + 7.5;
    const bridgeY1 = ground(...raw[raw.length - 1]) + 7.5;
    for (let i = 0; i + 1 < pts.length; i++) {
      const a = pts[i], b = pts[i + 1];
      const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
      if (len < 0.1) continue;
      const nx = -(b[1] - a[1]) / len, ny = (b[0] - a[0]) / len;
      const vertex = (p, off, lift = 0) => {
        const X = p[0] + nx * off, Y = p[1] + ny * off;
        const [x, z] = lay.toLocal(X, Y);
        const y = layer ? bridgeY0 + (bridgeY1 - bridgeY0) * p[2] / Math.max(1, along) : ground(X, Y);
        return [x, y + 0.19 + lift, z];
      };
      const ribbon = (left, right, col, lift = 0) => g.quad(vertex(a, left, lift), vertex(b, left, lift), vertex(b, right, lift), vertex(a, right, lift), col);
      ribbon(-w / 2 - (w > 3 ? 1 : 0.25), w / 2 + (w > 3 ? 1 : 0.25), edge);
      ribbon(-w / 2, w / 2, w > 3 ? asph : path, 0.035);
      if (!lite && w >= 9 && dist < 4200) {
        for (const sd of [-1, 1]) ribbon(sd * (w / 2 - 0.3) - 0.09, sd * (w / 2 - 0.3) + 0.09, paint, 0.065);
        if (Math.floor((a[2] + b[2]) / 2 / 7) % 2 === 0) ribbon(-0.1, 0.1, paint, 0.065);
      }
      if (layer) {
        for (const sd of [-1, 1]) {
          const A = vertex(a, sd * (w / 2 + 0.7)), B = vertex(b, sd * (w / 2 + 0.7));
          g.quad(A, B, [B[0], B[1] - 1.2, B[2]], [A[0], A[1] - 1.2, A[2]], side);
          piers.beam([A[0], A[1] + 0.65, A[2]], [B[0], B[1] + 0.65, B[2]], 0.18, "#A4A9A6");
        }
        if (Math.floor(a[2] / 40) !== Math.floor(b[2] / 40)) {
          const P = vertex(b, 0);
          const floor = ground(b[0], b[1]);
          const h = P[1] - floor - 1.2;
          if (h > 0) piers.cyl(P[0], floor + h / 2, P[2], 1, h, "#ACADA5", 6);
        }
      }
    }
  }
  const mesh = new THREE.Mesh(g.build(), new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.96, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: -1 }));
  mesh.receiveShadow = true;
  const group = new THREE.Group();
  group.add(mesh, piers.mesh(vcMat));
  return group;
}

/** Montjuic, the city's rise to Collserola and the delta: terrain from the elevation tiles, coloured by land cover and slope. */
function buildTerrain(lay, twin, gmat, lite) {
  const f = twin.frame;
  const { w, h } = f.terrain;
  const step = lite ? 2 : 1;
  const cols = Math.floor((w - 1) / step) + 1;
  const rows = Math.floor((h - 1) / step) + 1;
  const pos = new Float32Array(cols * rows * 3);
  const uv = new Float32Array(cols * rows * 2);
  const twinXY = new Float32Array(cols * rows * 2);
  const H = twin.heights;
  const T = lay.toLocal;
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      const ii = i * step;
      const jj = j * step;
      const X = f.extent[0] + jj * f.grid;
      const Y = f.extent[3] - ii * f.grid;
      const [x, z] = T(X, Y);
      const k = i * cols + j;
      pos[k * 3] = x;
      pos[k * 3 + 1] = H[ii * w + jj];
      pos[k * 3 + 2] = z;
      uv[k * 2] = X / DETAIL;
      uv[k * 2 + 1] = Y / DETAIL;
      twinXY[k * 2] = X;
      twinXY[k * 2 + 1] = Y;
    }
  }
  const idx = [];
  for (let i = 0; i < rows - 1; i++) {
    for (let j = 0; j < cols - 1; j++) {
      const a = i * cols + j;
      const b = a + 1;
      const c = a + cols;
      const d = c + 1;
      const top = Math.max(pos[a * 3 + 1], pos[b * 3 + 1], pos[c * 3 + 1], pos[d * 3 + 1]);
      if (top < -0.4) continue; // under the flat land or the water
      idx.push(a, c, b, b, c, d);
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geo.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
  geo.setIndex(idx);
  geo.computeVertexNormals();
  const m = new THREE.Mesh(geo, gmat);
  m.receiveShadow = true;
  m.castShadow = false;
  return m;
}

/** Quay walls (concrete faces down to the water), breakwater rock armour and natural shore slopes. */
function buildEdges(lay, twin, r, lite) {
  const T = lay.toLocal;
  const g = new Geo();
  const coping = lin(C.coping);
  const face = lin(C.quayWall);
  const algae = lin("#4F5546");
  const sand = lin("#CDBB97");
  const wet = lin("#9C8C6E");
  const seg = (line, fn) => {
    for (let i = 0; i + 3 < line.length; i += 2) {
      const a = T(line[i], line[i + 1]);
      const b = T(line[i + 2], line[i + 3]);
      fn(a, b);
    }
  };
  for (const line of twin.walls) {
    seg(line, (a, b) => {
      g.quad([a[0], 0.06, a[1]], [b[0], 0.06, b[1]], [b[0], -0.5, b[1]], [a[0], -0.5, a[1]], coping);
      g.quad([a[0], -0.5, a[1]], [b[0], -0.5, b[1]], [b[0], -2.55, b[1]], [a[0], -2.55, a[1]], face);
      g.quad([a[0], -2.55, a[1]], [b[0], -2.55, b[1]], [b[0], -5, b[1]], [a[0], -5, a[1]], algae);
    });
  }
  for (const line of twin.shores) {
    seg(line, (a, b) => {
      // rings keep land on their left in the twin (x east, y north); the scene's (x, z) is that frame mirrored, so
      // here the sea lies on the left of a -> b
      const dx = b[0] - a[0];
      const dz = b[1] - a[1];
      const L = Math.hypot(dx, dz) || 1;
      const nx = -dz / L;
      const nz = dx / L;
      const sx = nx * 22;
      const sz = nz * 22;
      g.quad([a[0], 0.02, a[1]], [b[0], 0.02, b[1]], [b[0] + sx * 0.5, -2.6, b[1] + sz * 0.5], [a[0] + sx * 0.5, -2.6, a[1] + sz * 0.5], sand);
      g.quad([a[0] + sx * 0.5, -2.6, a[1] + sz * 0.5], [b[0] + sx * 0.5, -2.6, b[1] + sz * 0.5], [b[0] + sx, -5, b[1] + sz], [a[0] + sx, -5, a[1] + sz], wet);
    });
  }
  const mesh = new THREE.Mesh(g.build(), new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.92, side: THREE.DoubleSide }));
  mesh.receiveShadow = true;
  // rock armour: two rows of boulders and concrete blocks along every breakwater edge
  const rocks = [];
  const rowsN = lite ? 1 : 2;
  for (const line of twin.armour) {
    seg(line, (a, b) => {
      const dx = b[0] - a[0];
      const dz = b[1] - a[1];
      const L = Math.hypot(dx, dz) || 1;
      const nx = -dz / L;
      const nz = dx / L;
      const mid = lay.toTwin((a[0] + b[0]) / 2 - nx * 8, (a[1] + b[1]) / 2 - nz * 8);
      if (coverAt(twin, mid[0], mid[1]) === "water") return;
      for (let s = 0; s < L; s += lite ? 4.5 : 3.2) {
        const x = a[0] + (dx * s) / L;
        const z = a[1] + (dz * s) / L;
        for (let k = 0; k < rowsN; k++) {
          const off = 2 + k * 5 + r() * 3;
          const y = 1.2 - k * 2.2 - r() * 1.2;
          const sz = 2.6 + r() * 2.2;
          const shade = r();
          rocks.push({ p: [x + nx * off, y, z + nz * off], s: [sz, sz * (0.6 + r() * 0.4), sz], c: shade < 0.25 ? "#A8A49B" : shade < 0.6 ? C.rock : C.rockDark, r: r() * 6 });
        }
      }
    });
  }
  const rockMesh = instanced(new THREE.DodecahedronGeometry(0.62, 0), new THREE.MeshStandardMaterial({ roughness: 1, flatShading: true }), rocks, { cast: false, receive: true });
  return [mesh, rockMesh];
}

const KIND_WALLS = [
  ["#E3DDD2", "#DCD6CB", "#E8E3DA"], // generic
  ["#E2D4BC", "#DAC8AC", "#E6DDCD", "#D2BE9F", "#DFD0B8", "#D5C3A8", "#CBB79A"], // residential: Barcelona's warm stucco
  ["#DCDCD8", "#CFD2D3", "#C3C9CC", "#D8D5CD", "#CBCFD1"], // industrial sheds and warehouses
  ["#C9D2D8", "#D8DBDC", "#B4C1CA", "#E0DDD6"], // offices, hotels, retail
  ["#E2D6C1", "#D8CCB5"], // public buildings
  ["#C8B48F", "#BCA680"], // historic stone (Montjuic castle, churches)
];
const KIND_ROOFS = [
  ["#CFC8BC", "#C6C0B5"],
  ["#D2C8B8", "#C9BBA6", "#B9A58E", "#D6CEC0"], // flat terraces, some tile
  ["#BFC2C3", "#AEB4B7", "#CBCCC9", "#9FA6AA"],
  ["#BFC4C7", "#CACCCB"],
  ["#CBBFA9"],
  ["#B9A27B"],
];

/** OSM buildings extruded to their tagged (or typical) heights, in palettes by building type. */
function buildBuildings(lay, twin, lite) {
  const T = lay.toLocal;
  const g = new Geo();
  const details = new Parts();
  const glass = lin("#8EA2B1");
  const cx = lay.toTwin(0, -300);
  for (const b of twin.buildings) {
    const [h0, base, kind] = b;
    const h = kind === 2 ? Math.min(h0, 40) : h0; // sheds and warehouses: cap mis-tagged heights
    const ring = [];
    for (let i = 3; i < b.length; i += 2) ring.push([b[i], b[i + 1]]);
    const dist = Math.hypot(ring[0][0] - cx[0], ring[0][1] - cx[1]);
    if (h < 40 && dist > (lite ? 3500 : 7000)) continue;
    const hs = Math.abs(Math.round(ring[0][0] * 7.3 + ring[0][1] * 3.1));
    const walls = KIND_WALLS[kind] || KIND_WALLS[0];
    const roofs = KIND_ROOFS[kind] || KIND_ROOFS[0];
    let wall = lin(walls[hs % walls.length]);
    const roof = lin(roofs[(hs >> 2) % roofs.length]);
    if (h > 55) wall = mix3(wall, glass, 0.6);
    const loc = ring.map(([X, Y]) => T(X, Y));
    const y0 = base - 0.5;
    const y1 = base + h;
    let area = 0;
    for (let i = 0; i < loc.length; i++) {
      const p = loc[i];
      const q = loc[(i + 1) % loc.length];
      area += p[0] * q[1] - q[0] * p[1];
    }
    const sgn = area > 0 ? 1 : -1;
    for (let i = 0; i < loc.length; i++) {
      const p = loc[i];
      const q = loc[(i + 1) % loc.length];
      if (sgn > 0) g.quad([p[0], y0, p[1]], [p[0], y1, p[1]], [q[0], y1, q[1]], [q[0], y0, q[1]], wall);
      else g.quad([q[0], y0, q[1]], [q[0], y1, q[1]], [p[0], y1, p[1]], [p[0], y0, p[1]], wall);
      // Nearby industrial facades: inset loading doors, a concrete plinth and
      // narrow glazing. Positions follow each mapped wall, including rotated sheds.
      if (!lite && kind === 2 && dist < 2500 && h > 5) {
        const len = Math.hypot(q[0] - p[0], q[1] - p[1]);
        const nx = (q[1] - p[1]) / Math.max(1, len) * sgn;
        const nz = -(q[0] - p[0]) / Math.max(1, len) * sgn;
        const panel = (lo, hi, ya, yb, col) => {
          const a = [p[0] + (q[0] - p[0]) * lo + nx * 0.05, p[1] + (q[1] - p[1]) * lo + nz * 0.05];
          const b = [p[0] + (q[0] - p[0]) * hi + nx * 0.05, p[1] + (q[1] - p[1]) * hi + nz * 0.05];
          g.quad([a[0], ya, a[1]], [a[0], yb, a[1]], [b[0], yb, b[1]], [b[0], ya, b[1]], lin(col));
        };
        panel(0, 1, y0, y0 + 0.8, "#9B9A90");
        for (let d = 5; d + 5 < len; d += 14) {
          panel(d / len, (d + 4.5) / len, y0 + 0.2, y0 + Math.min(4.3, h * 0.6), "#6C797D");
          for (let yy = 0.8; yy < Math.min(4, h * 0.6); yy += 0.5) panel(d / len, (d + 4.5) / len, y0 + yy, y0 + yy + 0.04, "#A2AAA6");
          panel(d / len, (d + 6) / len, y1 - 2.2, y1 - 1.25, "#607F8B");
        }
      }
      // window bands on taller buildings, a hair proud of the wall
      if (h > 14 && kind !== 2 && dist < (lite ? 1800 : 3200)) {
        const wb = mix3(wall, lin("#55606B"), 0.35);
        const L = Math.hypot(q[0] - p[0], q[1] - p[1]) || 1;
        const ox = ((q[1] - p[1]) / L) * 0.08 * sgn;
        const oz = (-(q[0] - p[0]) / L) * 0.08 * sgn;
        const P = [p[0] + ox, p[1] + oz];
        const Q = [q[0] + ox, q[1] + oz];
        const top = h < 30 ? Math.min(y1 - 2, y0 + 16) : y1 - 2;
        for (let y = y0 + 4; y < top; y += 3.2) {
          const ya = y + 1.0;
          const yb = y + 2.2;
          if (sgn > 0) g.quad([P[0], ya, P[1]], [P[0], yb, P[1]], [Q[0], yb, Q[1]], [Q[0], ya, Q[1]], wb);
          else g.quad([Q[0], ya, Q[1]], [Q[0], yb, Q[1]], [P[0], yb, P[1]], [P[0], ya, P[1]], wb);
        }
      }
    }
    if (!lite && dist < 2500 && h > 5) {
      const midX = loc.reduce((v, p) => v + p[0], 0) / loc.length;
      const midZ = loc.reduce((v, p) => v + p[1], 0) / loc.length;
      // Roof fixtures must fit fully inside the real footprint.
      if ([-4, 4].every(dx => [-4, 4].every(dz => pointInPoly(midX + dx, midZ + dz, loc)))) {
        details.box(midX, y1 + 0.25, midZ, 6.2, 0.5, 4.2, "#8F9797");
        details.box(midX, y1 + 1, midZ, 5.5, 1.1, 3.6, "#BBC1BE");
        for (const dx of [-1.4, 1.4]) details.cyl(midX + dx, y1 + 1.65, midZ, 0.85, 0.16, "#5E696D", 12);
      }
    }
    const tris = THREE.ShapeUtils.triangulateShape(loc.map(([x, z]) => V2(x, z)), []);
    for (const [a, bb, c] of tris) {
      const A = loc[a];
      const B = loc[bb];
      const Cc = loc[c];
      const cross = (B[0] - A[0]) * (Cc[1] - A[1]) - (B[1] - A[1]) * (Cc[0] - A[0]);
      if (cross < 0) g.tri([A[0], y1, A[1]], [B[0], y1, B[1]], [Cc[0], y1, Cc[1]], roof);
      else g.tri([A[0], y1, A[1]], [Cc[0], y1, Cc[1]], [B[0], y1, B[1]], roof);
    }
  }
  const m = new THREE.Mesh(g.build(), new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.85, flatShading: true, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: 0 }));
  m.castShadow = true;
  m.receiveShadow = true;
  const group = new THREE.Group();
  group.add(m, details.mesh(steelMat));
  return group;
}

/** Storage tanks: the CLH and Moll de l'Energia farms, with the big domed LNG tanks of the Enagas plant. */
function buildTanks(lay, twin) {
  const T = lay.toLocal;
  const shells = [];
  const domes = [];
  const fittings = new Parts();
  for (const [X, Y, rad, h, kind, rgb, base] of twin.tanks) {
    const [x, z] = T(X, Y);
    const v = Math.abs(Math.round(X * 3 + Y)) % 5;
    const col = kind === 1 ? "#DCD3C0" : ["#EEEDE9", "#E7E6E1", "#F1F0EC", "#DEDDD7", "#E9E9E4"][v];
    shells.push({ p: [x, base + h / 2, z], s: [rad * 2, h, rad * 2], c: col });
    domes.push({ p: [x, base + h, z], s: [rad, kind === 1 ? rad * 0.22 : rad * 0.08, rad], c: col });
    if (Math.hypot(x, z) < 2700 && rad > 6) {
      for (const yy of [h * 0.32, h * 0.66, h + 0.8]) {
        const ring = new THREE.TorusGeometry(rad + 0.1, yy > h ? 0.1 : 0.07, 4, 40);
        ring.rotateX(Math.PI / 2); ring.translate(x, base + yy, z);
        fittings.geo(ring, yy > h ? "#939C9A" : "#B6B9B3");
      }
      for (let a = 0; a < Math.PI * 2; a += Math.PI / 12) {
        fittings.cyl(x + Math.cos(a) * rad, base + h + 0.4, z + Math.sin(a) * rad, 0.07, 0.8, "#969F9D", 4);
      }
      for (const dx of [-0.36, 0.36]) fittings.cyl(x + dx, base + h / 2, z + rad + 0.2, 0.09, h, "#949C9B", 4);
      for (let yy = 0.4; yy < h; yy += 0.55) fittings.box(x, base + yy, z + rad + 0.2, 0.8, 0.06, 0.1, "#A2A9A5");
      fittings.cyl(x + rad * 0.25, base + h + rad * 0.08 + 0.5, z, 0.5, 1, "#8E9695", 8);
    }
  }
  const shell = instanced(new THREE.CylinderGeometry(0.5, 0.5, 1, 28), new THREE.MeshStandardMaterial({ roughness: 0.55, metalness: 0.15 }), shells);
  const dome = instanced(new THREE.SphereGeometry(1, 24, 8, 0, Math.PI * 2, 0, Math.PI / 2), new THREE.MeshStandardMaterial({ roughness: 0.55, metalness: 0.15 }), domes);
  return [shell, dome, fittings.mesh(steelMat)];
}

/** Towers and spires: Sagrada Familia, the port cable-car towers, the Collserola tower on Tibidabo. */
function buildSpires(lay, twin) {
  const T = lay.toLocal;
  const p = new Parts();
  for (const [X, Y, h, base, type] of twin.spires) {
    const [x, z] = T(X, Y);
    if (type === "bell_tower") {
      p.cyl(x, base + h * 0.42, z, 5.2, h * 0.84, "#B9A688", 12, 3.4);
      p.cyl(x, base + h * 0.92, z, 3.4, h * 0.16, "#C9B897", 12, 0.3);
    } else if (type === "communication") {
      p.cyl(x, base + h / 2, z, 1.6, h, "#D9D6D0", 8, 0.6);
    } else {
      p.cyl(x, base + h / 2, z, 3.2, h, "#8A8E92", 8, 2.2);
      p.box(x, base + h - 4, z, 12, 8, 12, "#7B8086");
    }
  }
  const col = twin.landmarks && twin.landmarks.collserola;
  if (col) {
    const [x, z] = T(col[0], col[1]);
    const b = col[3];
    p.cyl(x, b + 42, z, 4.6, 84, "#C9C6BE", 16, 2.8);
    for (let k = 0; k < 13; k++) p.cyl(x, b + 86 + k * 3.8, z, 17 - Math.abs(k - 6) * 0.6, 2.6, k % 2 ? "#9AA3AA" : "#E7E5DF", 20);
    p.cyl(x, b + 175, z, 2.4, 70, "#D9D6D0", 10, 1.4);
    p.cyl(x, b + 250, z, 1.2, 80, "#C74B3A", 8, 0.5);
  }
  const m = p.mesh(vcMat, { cast: false, receive: false });
  return m;
}

/** Trees where the land cover says so: Montjuic's pine woods and scrub, parks and gardens, the cemetery. */
function buildTrees(lay, twin, r, lite) {
  const items = [];
  const trunks = [];
  const R = lite ? 2800 : 4600;
  const cx = lay.toTwin(0, -800);
  const max = lite ? 8000 : 42000;
  const dens = { forest: 3.2, park: 2.2, scrub: 1.1, cemetery: 0.7, grass: 0.18, farm: 0.05 };
  const f = twin.frame;
  const res = f.cover_res;
  for (let Y = cx[1] - R; Y < cx[1] + R && items.length < max; Y += res) {
    for (let X = cx[0] - R; X < cx[0] + R && items.length < max; X += res) {
      if ((X - cx[0]) ** 2 + (Y - cx[1]) ** 2 > R * R) continue;
      const cls = coverAt(twin, X, Y);
      let d = dens[cls] || 0;
      if (!d) continue;
      const y0 = terrainAt(twin, X, Y);
      // Montjuic and the hills: pine woods over whatever the map calls park, grass or scrub
      const hill = y0 > 12 && (cls === "park" || cls === "grass" || cls === "scrub" || cls === "forest");
      if (hill) d = Math.max(d, 3.0);
      let n = Math.floor(d) + (r() < d - Math.floor(d) ? 1 : 0);
      for (; n > 0; n--) {
        const jx = X + (r() - 0.5) * res;
        const jy = Y + (r() - 0.5) * res;
        const [x, z] = lay.toLocal(jx, jy);
        if (z > -60 && z < 40 && x > lay.quayX0 - 80 && x < lay.quayX1 + 80) continue;
        const y = surfaceAt(twin, jx, jy, lite ? 2 : 1);
        const pine = hill || cls === "forest" || cls === "scrub";
        const s = (pine ? 7 : 6) + r() * (pine ? 6 : 5);
        if (!lite && Math.hypot(jx - cx[0], jy - cx[1]) < 2400) trunks.push({ p: [x, y + s * 0.43, z], s: [s * 0.065, s * 0.86, s * 0.065], c: "#635846" });
        items.push({ p: [x, y + s * (pine ? 0.95 : 0.7), z], s: [s * (pine ? 0.95 : 0.8), s * (pine ? 0.55 : 0.8), s * (pine ? 0.95 : 0.8)], c: pine ? (r() > 0.5 ? "#3A5530" : "#2F4728") : r() > 0.5 ? "#4C6E39" : "#3F5E31", r: r() * 6 });
      }
    }
  }
  // Irregular overlapping crowns give stone pines their broad, broken silhouette.
  // Instancing keeps the whole hillside to two draw calls, including the trunks.
  const crown = new Parts();
  for (const [x, y, z, sx, sy, sz, col] of [
    [-0.26, -0.02, 0.04, 0.85, 0.84, 0.82, "#C6CEAF"],
    [0.22, 0.04, -0.16, 0.85, 0.94, 0.9, "#E1E3CF"],
    [0.02, 0.19, 0.22, 0.72, 0.86, 0.78, "#F0EFDC"],
  ]) {
    const geo = new THREE.IcosahedronGeometry(0.62, 1);
    const pos = geo.attributes.position;
    for (let i = 0; i < pos.count; i++) {
      const px = pos.getX(i), py = pos.getY(i), pz = pos.getZ(i);
      const ragged = 1 + 0.12 * Math.sin(px * 31 + py * 17) * Math.cos(pz * 27 - py * 13);
      pos.setXYZ(i, px * ragged, py * ragged, pz * ragged);
    }
    geo.scale(sx, sy, sz); geo.translate(x, y, z); geo.computeVertexNormals();
    crown.geo(geo, col);
  }
  const group = new THREE.Group();
  group.add(instanced(crown.geometry(), new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 1 }), items, { cast: !lite, receive: !lite }));
  if (trunks.length) group.add(instanced(new THREE.CylinderGeometry(0.35, 0.5, 1, 5), mat("#FFFFFF"), trunks));
  return group;
}

/* ================================================================ the terminal */

function texturedPlane(x0, z0, x1, z1, y, tex, tile, color = "#FFFFFF") {
  const t = tex.clone();
  t.needsUpdate = true;
  t.repeat.set(Math.abs(x1 - x0) / tile, Math.abs(z1 - z0) / tile);
  t.userData.shared = false;
  const m = new THREE.Mesh(new THREE.PlaneGeometry(Math.abs(x1 - x0), Math.abs(z1 - z0)), new THREE.MeshStandardMaterial({ map: t, bumpMap: t, bumpScale: 0.07, roughnessMap: t, roughness: 0.96, color }));
  m.rotation.x = -Math.PI / 2;
  m.position.set((x0 + x1) / 2, y, (z0 + z1) / 2);
  m.receiveShadow = true;
  return m;
}

/** Quay deck: apron paving, crane rails, painted lanes, bollards and fenders, section numbers. */
function buildQuayDeck(lay, apronD, root, r) {
  const { quayX0, quayX1, seaRail, landRail, X0, secM, n, first } = lay;
  root.add(texturedPlane(quayX0, -apronD, quayX1, -0.3, 0.035, concreteTexture(), 20, "#EAE8E3"));
  const p = new Parts();
  for (const z of [seaRail, landRail]) {
    p.box((quayX0 + quayX1) / 2, 0.07, z, quayX1 - quayX0, 0.08, 1.4, "#8F887D");
    p.box((quayX0 + quayX1) / 2, 0.12, z, quayX1 - quayX0, 0.1, 0.22, C.rail);
  }
  // lane paint: yellow edge lines, the crane lane and the hatch-cover laydown
  p.box((quayX0 + quayX1) / 2, 0.06, -2.2, quayX1 - quayX0, 0.03, 0.25, C.laneYellow);
  p.box((quayX0 + quayX1) / 2, 0.06, landRail - 3, quayX1 - quayX0, 0.03, 0.2, C.laneYellow);
  for (let x = quayX0 + 4; x < quayX1 - 4; x += 9) p.box(x, 0.06, (seaRail + landRail) / 2, 4.5, 0.03, 0.18, C.lane);
  for (let x = quayX0 + 8; x < quayX1; x += 26) {
    for (const z of [landRail - 8, landRail - 18]) p.box(x, 0.06, z, 12.6, 0.03, 0.14, C.lane);
  }
  // bollards on the coping and cone fenders on the face, every half section
  for (let x = quayX0 + secM / 4; x < quayX1; x += secM / 2) {
    p.cyl(x, 0.55, -0.9, 0.45, 0.9, "#3A3D41", 10, 0.38);
    p.cyl(x, 1.02, -0.9, 0.62, 0.12, "#3A3D41", 10);
    p.box(x + 6, -1.6, 0.55, 2.6, 2.4, 1.1, "#1E2023");
    p.box(x + 6, -1.6, 1.25, 3.2, 3.0, 0.3, "#2A2C30");
  }
  // section boundary ticks
  for (let s = 0; s <= n; s++) p.box(X0 + s * secM, 0.07, -1.8, 0.3, 0.03, 2.2, C.lane);
  root.add(p.mesh(vcMat, { cast: true, receive: true }));
  const numGeo = new THREE.PlaneGeometry(10, 5);
  for (let s = 0; s < n; s++) {
    const m = new THREE.Mesh(numGeo, new THREE.MeshStandardMaterial({ map: textTexture(String(first + s), { size: 100, color: "rgba(255,255,255,0.85)" }), transparent: true, roughness: 1, depthWrite: false }));
    m.rotation.x = -Math.PI / 2;
    m.position.set(X0 + s * secM + secM / 2, 0.09, -4.4);
    m.renderOrder = 1;
    root.add(m);
  }
}

function lightMast(p, x, z, h = 36) {
  p.cyl(x, h / 2, z, 0.5, h, "#8D9196", 8, 0.32);
  p.box(x, h + 0.6, z, 5, 1.2, 1.4, "#5B6066");
  for (const dx of [-1.8, -0.6, 0.6, 1.8]) p.box(x + dx, h - 0.2, z + 0.4, 1.0, 0.7, 0.4, "#F4F1E6");
}

/** Rail tracks from OSM inside the terminal, with a container train on some of them. */
function railTracks(lay, twin, zMin, zMax, root, r) {
  const T = lay.toLocal;
  const p = new Parts();
  const tracks = [];
  for (const line of twin.rail) {
    const pts = [];
    for (let i = 0; i < line.length; i += 2) pts.push(T(line[i], line[i + 1]));
    const zs = pts.map((q) => q[1]);
    const xs = pts.map((q) => q[0]);
    if (Math.max(...zs) < zMin || Math.min(...zs) > zMax) continue;
    if (Math.max(...xs) < lay.quayX0 - 400 || Math.min(...xs) > lay.quayX1 + 400) continue;
    for (let i = 0; i + 1 < pts.length; i++) {
      const [ax, az] = pts[i];
      const [bx, bz] = pts[i + 1];
      const L = Math.hypot(bx - ax, bz - az);
      if (L < 0.5) continue;
      const ang = Math.atan2(-(bz - az), bx - ax);
      p.box((ax + bx) / 2, 0.08, (az + bz) / 2, L, 0.12, 2.6, "#6F675C", ang);
      for (const o of [-0.72, 0.72]) {
        const ox = Math.sin(ang) * o;
        const oz = Math.cos(ang) * o;
        p.box((ax + bx) / 2 + ox, 0.2, (az + bz) / 2 + oz, L, 0.14, 0.12, "#4A4743", ang);
      }
      if (L > 40) tracks.push({ ax, az, bx, bz, L, ang });
    }
  }
  root.add(p.mesh(vcMat, { cast: false, receive: true }));
  // trains: flat wagons with containers on a few long straight segments
  const wag = new Parts();
  const boxes = [];
  tracks.sort((a, b) => b.L - a.L);
  tracks.slice(0, 5).forEach((t, k) => {
    if (k % 2 === 1 && r() < 0.5) return;
    const nW = Math.floor((t.L * (0.4 + r() * 0.5)) / 20);
    const ux = (t.bx - t.ax) / t.L;
    const uz = (t.bz - t.az) / t.L;
    for (let i = 0; i < nW; i++) {
      const s = 10 + i * 20;
      const x = t.ax + ux * s;
      const z = t.az + uz * s;
      wag.box(x, 1.05, z, 19.2, 0.5, 2.6, "#3B3633", t.ang);
      wag.box(x, 0.55, z, 2.4, 0.9, 2.2, "#26231F", t.ang);
      if (r() < 0.85) {
        const two = r() < 0.3;
        if (two) for (const o of [-4.8, 4.8]) boxes.push({ p: [x + ux * o, 1.3 + BOX_H / 2, z + uz * o], s: [6.06, BOX_H, BOX_W], r: t.ang, ...boxSpec(r()) });
        else boxes.push({ p: [x, 1.3 + BOX_H / 2, z], s: [BOX_L, BOX_H, BOX_W], r: t.ang, ...boxSpec(r()) });
      }
    }
  });
  root.add(wag.mesh(vcMat));
  root.add(containerMesh(boxes));
  return tracks;
}


/* ---------------- horizontal transport ---------------- */

/* ---------------- BEST ---------------- */

function ascGeometry(span, color) {
  const p = new Parts();
  const H = 23.5;
  for (const sx of [-span / 2, span / 2]) {
    for (const z of [-5.5, 5.5]) {
      p.box(sx, H / 2, z, 1.2, H, 1.2, color);
      p.box(sx, 1.0, z, 1.6, 2.0, 3.2, C.steelDark);
    }
    p.box(sx, H - 0.6, 0, 1.6, 1.4, 13, color);
    p.beam([sx, 3, -5.5], [sx, H - 2, 5.5], 0.6, color);
  }
  for (const z of [-5.2, 5.2]) p.box(0, H + 0.6, z, span + 1.6, 2.2, 1.4, color);
  p.box(0, H + 2.0, 0, span * 0.5, 1.0, 11, "#E7E6E1");
  p.box(-span / 2 + 3, H + 2.6, 0, 4.5, 2.4, 6, "#E7E6E1");
  // trolley + spreader
  p.box(0, H - 0.4, 0, 5.4, 1.6, 7, "#3A4048");
  p.box(0, H - 9, 0, 2.6, 0.7, 12.4, "#E6B42E");
  for (const dx of [-0.8, 0.8]) for (const dz of [-3, 3]) p.box(dx, H - 4.7, dz, 0.12, 8.6, 0.12, "#2B2E33");
  return p.geometry();
}

function buildBEST(lay, twin, root, r, lite) {
  const T = lay.toLocal;
  const blocks = twin.best_blocks.map(([ref, X, Y, len, wid, ang]) => {
    const [x, z] = T(X, Y);
    return { ref, x, z, len, wid, zSea: z + len / 2, zLand: z - len / 2 };
  });
  const apronD = -Math.max(...blocks.map((b) => b.zSea)) + 2;
  buildQuayDeck(lay, apronD, root, r);
  const bx0 = Math.min(...blocks.map((b) => b.x - b.wid / 2)) - 14;
  const bx1 = Math.max(...blocks.map((b) => b.x + b.wid / 2)) + 14;
  const zFar = Math.min(...blocks.map((b) => b.zLand));
  root.add(texturedPlane(bx0, zFar - 40, bx1, -apronD, 0.03, concreteTexture(), 24, "#D9D6D0"));
  const p = new Parts();
  const boxes = [];
  const ascs = [];
  const transferSlots = [];
  for (const b of blocks) {
    // Reserved ground slots at the real block transfer end. Fill from the back
    // so a later container cannot block the carrier's route to an earlier one.
    for (const dx of [-12, -4, 4, 12]) {
      if (Math.abs(dx) + 2.8 > b.wid / 2) continue;
      for (const depth of [25, 10]) transferSlots.push({ p: [b.x + dx, 0.34 + BOX_H / 2, b.zSea - depth], via: [b.x + dx, b.zSea + 7], ry: Math.PI / 2, travelY: 3.1, corridor: `best-${b.ref}-${dx}`, depth });
    }
    // block pad, ASC rails, transfer zones at both ends
    p.box(b.x, 0.06, b.z, b.wid + 2, 0.04, b.len, "#B8B2A6");
    for (const s of [-1, 1]) {
      p.box(b.x + s * (b.wid / 2 + 0.6), 0.1, b.z, 0.9, 0.12, b.len, "#6B665E");
      p.box(b.x + s * (b.wid / 2 + 0.6), 0.18, b.z, 0.18, 0.1, b.len, C.rail);
    }
    for (const [z0, sgn] of [
      [b.zSea, -1],
      [b.zLand, 1],
    ]) {
      for (let k = 0; k < 5; k++) p.box(b.x - b.wid / 2 + 3 + k * 5.8, 0.07, z0 + sgn * 9, 0.18, 0.03, 13, C.lane);
    }
    const fullness = 0.55 + r() * 0.35;
    const rows = 10;
    const pitch = (b.wid - 2.6) / rows;
    const z0 = b.zLand + 24;
    const z1 = b.zSea - 42;
    const rowCol = Array.from({ length: rows }, () => boxSpec(r()));
    for (let zc = z0 + BOX_L / 2; zc + BOX_L / 2 <= z1; zc += 12.8) {
      if (r() < 0.08) rowCol[Math.floor(r() * rows)] = boxSpec(r());
      for (let k = 0; k < rows; k++) {
        if (r() > fullness + 0.15) continue;
        const tiers = Math.min(5, 1 + Math.floor(r() * 5.5 * fullness + 0.4));
        for (let t = 0; t < tiers; t++) {
          boxes.push({ p: [b.x - b.wid / 2 + 1.3 + pitch * (k + 0.5), 0.08 + BOX_H / 2 + t * (BOX_H + 0.03), zc], s: [BOX_W, BOX_H, BOX_L], ...(r() < 0.72 ? rowCol[k] : boxSpec(r())) });
        }
      }
    }
    // two stacking cranes per block, one each half
    ascs.push({ x: b.x, z0: (b.z + b.zSea) / 2, span: (b.zSea - b.z) / 2 - 16, ph: r() * 100, sp: 0.01 + r() * 0.01 });
    ascs.push({ x: b.x, z0: (b.z + b.zLand) / 2, span: (b.z - b.zLand) / 2 - 16, ph: r() * 100, sp: 0.01 + r() * 0.01 });
  }
  // light masts between block groups and along the apron's land side
  for (let i = 0; i < blocks.length; i += 3) {
    const b = blocks[i];
    for (const z of [b.zSea + 4, b.z, b.zLand - 6]) lightMast(p, b.x - b.wid / 2 - 2.5, z, 36);
  }
  root.add(p.mesh(vcMat, { cast: true, receive: true }));
  // containers rotated so their length runs across the quay (along the block)
  const boxItems = boxes.map((it) => ({ ...it, s: [it.s[2], it.s[1], it.s[0]], r: Math.PI / 2 }));
  root.add(noReflect(containerMesh(boxItems)));

  const ascGeo = ascGeometry(blocks[0].wid + 1.2, C.bestBlue);
  const ascMesh = new THREE.InstancedMesh(ascGeo, steelMat, ascs.length);
  ascMesh.castShadow = true;
  ascMesh.receiveShadow = true;
  ascMesh.frustumCulled = false;
  const tracks = railTracks(lay, twin, zFar - 260, zFar + 20, root, r);
  const rmg = new Parts();
  if (tracks.length) {
    const t = tracks[0];
    for (const s of [0.3, 0.62]) {
      const x = t.ax + (t.bx - t.ax) * s;
      const z = t.az + (t.bz - t.az) * s;
      const ux = Math.sin(t.ang);
      const uz = Math.cos(t.ang);
      const span = 46;
      for (const o of [-span / 2, span / 2]) {
        rmg.box(x + ux * o, 13, z + uz * o, 1.4, 26, 10, C.bestBlue, t.ang);
      }
      rmg.box(x, 26.5, z, 3, 2.2, span + 8, C.bestBlue, t.ang);
      rmg.box(x - Math.cos(t.ang) * 3, 26.5, z + Math.sin(t.ang) * 3, 3, 2.2, span + 8, C.bestBlue, t.ang);
    }
  }
  root.add(rmg.mesh(steelMat));

  const scr = mulberry32(99);
  // trucks waiting at the landside ends of the blocks
  const trucks = [];
  for (const b of blocks) {
    for (let k = 0; k < 3; k++) if (scr() < 0.4) trucks.push({ x: b.x - b.wid / 2 + 4 + k * 8, z: b.zLand - 12 });
  }
  root.add(truckMesh(trucks, Math.PI / 2, r));

  // Yard cranes remain parked until a yard job is represented. Decorative
  // oscillation with randomly appearing loads is misleading in a simulation.
  ascs.forEach((a, i) => setInstance(ascMesh, i, a.x, 0, a.z0, 0));
  ascMesh.instanceMatrix.needsUpdate = true;
  root.add(ascMesh);
  return { update() {}, apronD, transferSlots, cargoRoadZ: -apronD + 14 };
}

/* ---------------- APM ---------------- */

function buildAPM(lay, twin, root, r, lite) {
  const T = lay.toLocal;
  const slabs = twin.apm_slabs.map((ring) => {
    const pts = [];
    for (let i = 0; i < ring.length; i += 2) pts.push(T(ring[i], ring[i + 1]));
    return pts;
  });
  // a ~60 m apron like the real one: crane portal, then the straddle-carrier road, then the stacks
  const apronD = Math.max(60, clamp(-Math.max(...slabs.flat().map((q) => q[1])) + 4, 40, 90));
  buildQuayDeck(lay, apronD, root, r);
  const zFar = Math.min(...slabs.flat().map((q) => q[1]));
  root.add(texturedPlane(lay.quayX0 + 10, zFar - 30, lay.quayX1 - 10, -apronD, 0.03, concreteTexture(), 24, "#D9D6D0"));
  const p = new Parts();
  const boxes = [];
  const pads = [];
  const transferSlots = [];
  for (const poly of slabs) {
    // rows run along the slab's longest edge, 4.1 m apart (box + straddle-carrier leg lane); 1-over-3 stacking
    let best = 0;
    let ax = [1, 0];
    for (let i = 0; i < poly.length; i++) {
      const a = poly[i];
      const b = poly[(i + 1) % poly.length];
      const L = Math.hypot(b[0] - a[0], b[1] - a[1]);
      if (L > best) {
        best = L;
        ax = [(b[0] - a[0]) / L, (b[1] - a[1]) / L];
      }
    }
    const nx = [-ax[1], ax[0]];
    const ts = poly.map((q) => q[0] * ax[0] + q[1] * ax[1]);
    const ss = poly.map((q) => q[0] * nx[0] + q[1] * nx[1]);
    const t0 = Math.min(...ts);
    const t1 = Math.max(...ts);
    const s0 = Math.min(...ss);
    const s1 = Math.max(...ss);
    const ang = Math.atan2(-ax[1], ax[0]);
    const full = 0.6 + r() * 0.3;
    pads.push(slabGeometry(poly));
    let rk = 0;
    for (let s = s0 + 2.5; s < s1 - 2; s += 4.1, rk++) {
      let runCol = boxSpec(r());
      for (let t = t0 + 7; t < t1 - 7; t += 12.8) {
        const x = ax[0] * t + nx[0] * s;
        const z = ax[1] * t + nx[1] * s;
        if (!pointInPoly(x, z, poly) || z > -apronD - 2) continue;
        const at = u => [ax[0] * u + nx[0] * s, ax[1] * u + nx[1] * s];
        const near0 = at(t0)[1] > at(t1)[1];
        const via = at(near0 ? t0 - 7 : t1 + 7);
        const depth = Math.hypot(x - via[0], z - via[1]);
        const cornersInside = [-1, 1].every(a => [-1, 1].every(b => pointInPoly(x + a * ax[0] * 6.2 + b * nx[0] * 2.8, z + a * ax[1] * 6.2 + b * nx[1] * 2.8, poly)));
        if (rk % 4 === 0 && via[1] > -apronD - 25 && depth < 105 && depth > 15 && cornersInside) {
          transferSlots.push({ p: [x, 0.34 + BOX_H / 2, z], via, ry: Math.atan2(via[1] - z, x - via[0]), travelY: 10.2, corridor: `apm-${slabs.indexOf(poly)}-${rk}`, depth });
          continue;
        }
        if (r() > full) continue;
        const tiers = 1 + Math.floor(r() * 3);
        if (r() < 0.18) runCol = boxSpec(r());
        for (let k = 0; k < tiers; k++) boxes.push({ p: [x, 0.08 + BOX_H / 2 + k * (BOX_H + 0.03), z], s: [BOX_L, BOX_H, BOX_W], r: ang, ...(r() < 0.7 ? runCol : boxSpec(r())) });
      }
    }
  }
  for (let x = lay.quayX0 + 60; x < lay.quayX1 - 40; x += 120) lightMast(p, x, -apronD + 3, 32);
  root.add(p.mesh(vcMat, { cast: true, receive: true }));
  for (const g of pads) {
    const m = new THREE.Mesh(g, mat("#B9B3A7", { rough: 0.95 }));
    m.receiveShadow = true;
    root.add(m);
  }
  root.add(noReflect(containerMesh(boxes)));

  const tracks = railTracks(lay, twin, zFar - 220, zFar + 40, root, r);
  if (tracks.length) {
    const t = tracks[0];
    const rmg = new Parts();
    const x = t.ax + (t.bx - t.ax) * 0.5;
    const z = t.az + (t.bz - t.az) * 0.5;
    const ux = Math.sin(t.ang);
    const uz = Math.cos(t.ang);
    for (const o of [-20, 20]) rmg.box(x + ux * o, 12, z + uz * o, 1.4, 24, 9, "#4DAEC9", t.ang);
    rmg.box(x, 24.5, z, 3, 2, 50, "#4DAEC9", t.ang);
    root.add(rmg.mesh(steelMat));
  }
  // cruise ships at the Moll Adossat opposite: alongside its west face, on the side facing APM
  for (const obstacle of harbourObstacles(lay)) {
    const m = cruiseShip(obstacle.len, obstacle.modelBeam, r);
    m.position.set(obstacle.x, WATER_Y, obstacle.z);
    m.rotation.y = obstacle.th;
    root.add(m);
  }
  return { update() {}, apronD, transferSlots, cargoRoadZ: -apronD + 14 };
}

function slabGeometry(poly) {
  const tris = THREE.ShapeUtils.triangulateShape(poly.map(([x, z]) => V2(x, z)), []);
  const pos = [];
  for (const [a, b, c] of tris) {
    for (const k of [a, b, c]) pos.push(poly[k][0], 0.05, poly[k][1]);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  g.computeVertexNormals();
  // make sure it faces up
  const nrm = g.attributes.normal;
  if (nrm.count && nrm.getY(0) < 0) {
    for (let i = 0; i < pos.length; i += 9) {
      const t = [pos[i + 3], pos[i + 4], pos[i + 5]];
      pos[i + 3] = pos[i + 6];
      pos[i + 4] = pos[i + 7];
      pos[i + 5] = pos[i + 8];
      pos[i + 6] = t[0];
      pos[i + 7] = t[1];
      pos[i + 8] = t[2];
    }
    g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    g.computeVertexNormals();
  }
  return g;
}

function truckMesh(items, ry, r) {
  const t = new Parts();
  t.box(5.2, 1.9, 0, 2.4, 2.8, 2.5, "#F1F0EC");
  t.box(6.35, 2.4, 0, 0.1, 1.2, 2.2, C.window);
  t.box(-0.6, 1.0, 0, 12.6, 0.4, 2.4, "#2F3237");
  for (const x of [5.4, -3.5, -5.2]) for (const z of [-1.05, 1.05]) t.cylZ(x, 0.5, z, 0.5, 0.4, "#1E2023", 10);
  const geo = t.geometry();
  const cabs = items.map((it) => ({ p: [it.x, 0, it.z], s: [1, 1, 1], r: ry, c: ["#F1F0EC", "#2E5A93", "#B8322A", "#E6B42E", "#3F4A55"][Math.floor(r() * 5)] }));
  const g = new THREE.Group();
  g.add(instanced(geo, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.7 }), cabs));
  g.add(containerMesh(items.map((it) => ({ p: [it.x, 1.2 + BOX_H / 2, it.z + 0.6], s: [BOX_L, BOX_H, BOX_W], r: ry, ...boxSpec(r()) }))));
  return g;
}

/** A large cruise ship as seen across the channel: white hull with a navy boot band, a long block of balcony decks
 * stepping back towards the bow, lifeboats along both sides and a blue-topped funnel aft. */
function cruiseShip(L, B, r) {
  const plan = (len, beam) => {
    const sh = new THREE.Shape();
    const hw = beam / 2;
    sh.moveTo(-len / 2, -hw * 0.8);
    sh.quadraticCurveTo(-len / 2, -hw, -len / 2 + hw * 0.6, -hw);
    sh.lineTo(len / 2 - len * 0.16, -hw);
    sh.bezierCurveTo(len / 2 - len * 0.05, -hw, len / 2, -hw * 0.35, len / 2, 0);
    sh.bezierCurveTo(len / 2, hw * 0.35, len / 2 - len * 0.05, hw, len / 2 - len * 0.16, hw);
    sh.lineTo(-len / 2 + hw * 0.6, hw);
    sh.quadraticCurveTo(-len / 2, hw, -len / 2, hw * 0.8);
    sh.closePath();
    return sh;
  };
  const slab = (len, beam, y0, h, color) => {
    const g = new THREE.ExtrudeGeometry(plan(len, beam), { depth: h, bevelEnabled: false, curveSegments: 10 });
    g.rotateX(-Math.PI / 2);
    g.translate(0, y0, 0);
    return [g, color];
  };
  const parts = [];
  parts.push(slab(L, B, -6, 4.2, "#1F3A63"));
  parts.push(slab(L, B, -1.8, 13.8, "#F4F4F1"));
  parts.push(slab(L * 0.91, B + 0.1, 10.2, 1.2, "#334854")); // promenade glazing
  const decks = 11;
  for (let k = 0; k < decks; k++) {
    const len = L * (k < 7 ? 0.83 : 0.83 - (k - 6) * 0.06);
    const y = 12 + k * 3.05;
    const shift = -L * 0.025 - Math.max(0, k - 6) * 2.5;
    const [g, c] = slab(len, B - 0.6, y, 3.05, "#F4F4F1");
    g.translate(shift, 0, 0);
    parts.push([g, c]);
    const [gb] = slab(len + 0.2, B - 0.4, y + 0.8, 1.3, "#4A6079"); // balcony glass
    gb.translate(shift, 0, 0);
    parts.push([gb, k % 3 === 2 ? "#3C5068" : "#506A84"]);
  }
  const p = new Parts();
  for (const [g, c] of parts) p.geo(g, c);
  // Balcony dividers interrupt the glass bands; lower decks have individual portholes.
  for (const side of [-1, 1]) {
    for (let k = 0; k < decks; k++) {
      const len = L * (k < 7 ? 0.83 : 0.83 - (k - 6) * 0.06);
      const shift = -L * 0.025 - Math.max(0, k - 6) * 2.5;
      for (let x = -len / 2 + 9; x < len / 2 - L * 0.13; x += 4.2) {
        p.box(x + shift, 13.4 + k * 3.05, side * (B / 2 - 0.16), 0.16, 2.5, 0.65, '#D6DEDF');
      }
    }
    for (let x = -L * 0.43; x < L * 0.32; x += 4.4) for (const y of [3.6, 6.4]) p.box(x, y, side * (B / 2 + 0.05), 1.5, 0.9, 0.1, '#334552');
  }
  const top = 12 + decks * 3.05;
  p.box(-L * 0.3, top + 6, 0, 24, 12, 14, "#F4F4F1");
  p.box(-L * 0.3, top + 10.5, 0, 24.2, 3.4, 14.2, "#1D4F91");
  p.box(L * 0.18, top + 1.6, 0, 40, 3.2, B * 0.7, "#E8E8E4");
  // Open sun deck, pools, satellite domes and forward bridge wings.
  p.box(-L * 0.05, top + 0.18, 0, L * 0.37, 0.35, B * 0.72, '#C8BCA1');
  for (const x of [-L * 0.1, L * 0.07]) {
    p.box(x, top + 0.42, 0, 18, 0.5, 9, '#EAECE8');
    p.box(x, top + 0.69, 0, 16, 0.04, 7, '#3C9FB6');
    for (const side of [-1, 1]) for (let dx = -10; dx < 10; dx += 3) p.box(x + dx, top + 0.5, side * 8, 2.1, 0.4, 0.8, '#F0ECE1');
  }
  p.box(L * 0.32, 36.5, 0, 14, 3.5, B + 5, '#E9EEEB');
  p.box(L * 0.32 + 7.1, 37, 0, 0.2, 1.5, B + 4, '#23444F');
  for (const side of [-1, 1]) {
    const dome = new THREE.SphereGeometry(3.5, 12, 8);
    dome.translate(L * 0.23, top + 5, side * 9);
    p.geo(dome, '#EEF1ED');
    for (let i = 0; i < 13; i++) {
      const x = -L * 0.34 + i * L * 0.049;
      const boat = new THREE.CapsuleGeometry(1.25, 6.4, 3, 8);
      boat.rotateZ(Math.PI / 2);
      boat.translate(x, 9, side * (B / 2 + 1));
      p.geo(boat, '#E98332');
      p.box(x, 10, side * (B / 2 + 1), 6, 0.6, 1.8, '#F3EADA');
      p.beam([x - 2.7, 12, side * (B / 2 - 1)], [x - 2.7, 10.5, side * (B / 2 + 2)], 0.25, '#D6DFDE');
    }
  }
  const m = p.mesh(vcMat, { cast: true, receive: true });
  const g = new THREE.Group();
  g.add(m);
  return g;
}

function noReflect(obj) {
  obj.traverse((o) => o.layers.set(NO_REFLECT));
  return obj;
}

/* ================================================================ world */

export function buildWorld(lay, twin, { lite = false } = {}) {
  const root = new THREE.Group();
  const r = mulberry32(1234 + Math.round(lay.L) + (lay.quay === "24B" ? 7 : 0));
  const gmat = groundMaterial(twin);
  root.add(noReflect(buildLand(lay, twin, gmat)));
  // Land cover is baked into the surface atlas and follows the actual terrain.
  root.add(noReflect(buildRoads(lay, twin, lite)));
  root.add(noReflect(buildTerrain(lay, twin, gmat, lite)));
  const [edges, rocks] = buildEdges(lay, twin, r, lite);
  root.add(edges, noReflect(rocks));
  root.add(noReflect(buildBuildings(lay, twin, lite)));
  for (const m of buildTanks(lay, twin)) root.add(m);
  root.add(buildSpires(lay, twin));
  root.add(noReflect(buildTrees(lay, twin, r, lite)));
  root.add(noReflect(buildContext(lay, twin, { lite })));
  // the terminal sits a hand's breadth above the land-cover layers so none of them can show through its paving
  const termRoot = new THREE.Group();
  termRoot.position.y = 0.24;
  root.add(termRoot);
  const term = lay.quay === "24B" ? buildAPM(lay, twin, termRoot, r, lite) : buildBEST(lay, twin, termRoot, r, lite);
  return {
    root,
    apronD: term.apronD,
    transferSlots: term.transferSlots,
    cargoRoadZ: term.cargoRoadZ,
    /** time: wall-clock seconds; work: x of every quay crane working a ship now. */
    update(time, work) {
      term.update(time, work);
    },
  };
}
