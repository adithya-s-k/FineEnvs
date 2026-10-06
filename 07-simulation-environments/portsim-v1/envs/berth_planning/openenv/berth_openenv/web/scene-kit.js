// Shared building blocks for the port twin: palette, procedural textures (concrete, corrugated container steel),
// merged vertex-coloured parts and instancing helpers.
import * as THREE from "three";
import { mulberry32 } from "./model.js";
import { mergeGeometries } from "three/addons/utils/BufferGeometryUtils.js";

export const C = {
  concrete: "#C8C4BC", // apron paving: light, slightly warm grey concrete
  concreteDark: "#B9AD9D",
  asphalt: "#8C8A85",
  quayWall: "#A9A196",
  coping: "#E2DBCF",
  lane: "#F2EFE6",
  laneYellow: "#E6B42E",
  rail: "#5E5B56",
  steelDark: "#2F343B",
  white: "#F1F0EC",
  window: "#2B3540",
  signal: "#EE6232",
  red: "#B8322A",
  conflict: "#D93A2B",
  rock: "#8F8B84",
  rockDark: "#6E6A64",
  tree: "#4C6B3C",
  treeDark: "#3A5530",
  // BEST: white and carmine STS cranes, blue Konecranes stacking cranes
  bestWhite: "#EDECE7",
  bestCarmine: "#9C1F2F",
  bestBlue: "#2D62A8",
  // APM: yellow STS cranes with grey-blue machinery houses; the 2025 "Triple-E" cranes in light APM blue
  apmYellow: "#EEAA24",
  apmHouse: "#5F6775",
  apmBlue: "#4DAEC9",
};

/**
 * Containers as they look in the sun at a Mediterranean hub: the lines' own boxes (Maersk sky blue, MSC gold and
 * cream, CMA CGM navy, Hapag-Lloyd orange, ONE magenta, Evergreen green, COSCO steel blue), the lessors' rust-reds
 * and greys (Triton, Textainer, Seaco, Florens), white reefers. [colour, weight, marking row, ink: 0 white / 1 dark]
 */
export const MARKINGS = ["", "MAERSK", "MSC", "CMA CGM", "Hapag-Lloyd", "ONE", "EVERGREEN", "COSCO", "TRITON", "TEX", "SEACO", "FLORENS", "HMM", "ZIM"];
const PALETTE = [
  ["#6A98C0", 8, 1, 0],
  ["#CC9E3E", 5, 2, 0], ["#D8CBA9", 2, 2, 1],
  ["#2A4677", 8, 3, 0],
  ["#D56C2C", 5, 4, 0],
  ["#AE4074", 2, 5, 0],
  ["#357046", 3, 6, 0],
  ["#44679A", 4, 7, 0],
  ["#984633", 5, 8, 0], ["#984633", 3, 9, 0], ["#78352C", 5, 9, 0], ["#B2432F", 2, 0, 0],
  ["#959BA0", 4, 10, 0], ["#959BA0", 3, 11, 0], ["#B6BBBE", 2, 12, 1], ["#B6BBBE", 2, 13, 1],
  ["#E3E3DE", 5, 0, 0],
  ["#BBAD86", 2, 0, 0], ["#3A7B77", 1, 8, 0], ["#725540", 2, 0, 0],
];
const TOTAL = PALETTE.reduce((a, p) => a + p[1], 0);
const _bc = new THREE.Color();
function entry(u) {
  let acc = 0;
  const x = u * TOTAL;
  for (const e of PALETTE) {
    acc += e[1];
    if (x < acc) return e;
  }
  return PALETTE[0];
}
/** A container: colour (with wear), marking row and lettering ink, all from one uniform draw u. */
export function boxSpec(u) {
  const e = entry(u);
  return { c: wearColor(e[0], (u * 9301.17) % 1), b: e[2], k: e[3] };
}
export function boxColor(u) {
  return boxSpec(u).c;
}
/** A container colour with a little wear (v in 0..1: +-4 % lightness, slightly less saturated). */
export function wearColor(hex, v) {
  _bc.set(hex).offsetHSL(0, -0.04 * v, (v - 0.5) * 0.08);
  return `#${_bc.getHexString()}`;
}
export const BOX_COLORS = PALETTE.map((p) => p[0]);

export const unitBox = new THREE.BoxGeometry(1, 1, 1);
export const unitCyl = new THREE.CylinderGeometry(0.5, 0.5, 1, 12);

const matCache = new Map();
export function mat(color, opts = {}) {
  const key = `${color}|${opts.rough ?? 0.85}|${opts.metal ?? 0}|${opts.flat ? 1 : 0}`;
  let m = matCache.get(key);
  if (!m) {
    m = new THREE.MeshStandardMaterial({ color, roughness: opts.rough ?? 0.85, metalness: opts.metal ?? 0, flatShading: !!opts.flat });
    m.userData.shared = true;
    matCache.set(key, m);
  }
  return m;
}

export const vcMat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.78, metalness: 0.05 });
vcMat.userData.shared = true;
/** Painted steel (cranes, gantries): a touch of sheen. */
export const steelMat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.42, metalness: 0.32 });
steelMat.userData.shared = true;

const tmpColor = new THREE.Color();
const UP = new THREE.Vector3(0, 1, 0);

function colorize(geo, color) {
  tmpColor.set(color);
  const n = geo.attributes.position.count;
  const arr = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    arr[i * 3] = tmpColor.r;
    arr[i * 3 + 1] = tmpColor.g;
    arr[i * 3 + 2] = tmpColor.b;
  }
  geo.setAttribute("color", new THREE.BufferAttribute(arr, 3));
  if (geo.attributes.uv) geo.deleteAttribute("uv");
  return geo;
}

/** Accumulates coloured boxes/cylinders/beams and merges them into one vertex-coloured geometry (one draw call). */
export class Parts {
  constructor() {
    this.geos = [];
  }
  box(x, y, z, sx, sy, sz, color, ry = 0, rx = 0, rz = 0) {
    const g = new THREE.BoxGeometry(sx, sy, sz);
    if (rx) g.rotateX(rx);
    if (rz) g.rotateZ(rz);
    if (ry) g.rotateY(ry);
    g.translate(x, y, z);
    this.geos.push(colorize(g, color));
    return this;
  }
  cyl(x, y, z, r, h, color, seg = 10, rTop = r) {
    const g = new THREE.CylinderGeometry(rTop, r, h, seg);
    g.translate(x, y, z);
    this.geos.push(colorize(g, color));
    return this;
  }
  /** Horizontal cylinder along x (wheels, drums). */
  cylX(x, y, z, r, len, color, seg = 10) {
    const g = new THREE.CylinderGeometry(r, r, len, seg);
    g.rotateZ(Math.PI / 2);
    g.translate(x, y, z);
    this.geos.push(colorize(g, color));
    return this;
  }
  /** Horizontal cylinder along z. */
  cylZ(x, y, z, r, len, color, seg = 10) {
    const g = new THREE.CylinderGeometry(r, r, len, seg);
    g.rotateX(Math.PI / 2);
    g.translate(x, y, z);
    this.geos.push(colorize(g, color));
    return this;
  }
  /** Box stretched between two points (truss members, stays). */
  beam(a, b, size, color, size2 = size) {
    const va = new THREE.Vector3(...a);
    const vb = new THREE.Vector3(...b);
    const dir = vb.clone().sub(va);
    const len = dir.length();
    const g = new THREE.BoxGeometry(size, len, size2);
    g.applyQuaternion(new THREE.Quaternion().setFromUnitVectors(UP, dir.normalize()));
    const mid = va.add(vb).multiplyScalar(0.5);
    g.translate(mid.x, mid.y, mid.z);
    this.geos.push(colorize(g, color));
    return this;
  }
  geo(g, color) {
    this.geos.push(colorize(g, color));
    return this;
  }
  geometry() {
    if (!this.geos.length) return new THREE.BufferGeometry();
    // extrusions are non-indexed and boxes indexed: merge them all non-indexed when mixed
    if (this.geos.some((g) => !g.index) && this.geos.some((g) => g.index)) {
      this.geos = this.geos.map((g) => {
        if (!g.index) return g;
        const n = g.toNonIndexed();
        g.dispose();
        return n;
      });
    }
    for (const g of this.geos) g.clearGroups();
    const merged = mergeGeometries(this.geos, false);
    for (const g of this.geos) g.dispose();
    this.geos = [];
    return merged;
  }
  mesh(material = vcMat, { cast = true, receive = true } = {}) {
    const m = new THREE.Mesh(this.geometry(), material);
    m.castShadow = cast;
    m.receiveShadow = receive;
    return m;
  }
}

const dummy = new THREE.Object3D();

/** items: [{p:[x,y,z], s:[sx,sy,sz], c?:color, r?:ry}] -> one InstancedMesh. */
export function instanced(geometry, material, items, { cast = true, receive = true } = {}) {
  const m = new THREE.InstancedMesh(geometry, material, Math.max(1, items.length));
  m.count = items.length;
  items.forEach((it, i) => {
    dummy.position.set(it.p[0], it.p[1], it.p[2]);
    dummy.rotation.set(0, it.r || 0, 0);
    dummy.scale.set(it.s[0], it.s[1], it.s[2]);
    dummy.updateMatrix();
    m.setMatrixAt(i, dummy.matrix);
    if (it.c) m.setColorAt(i, tmpColor.set(it.c));
  });
  m.instanceMatrix.needsUpdate = true;
  if (m.instanceColor) m.instanceColor.needsUpdate = true;
  m.castShadow = cast;
  m.receiveShadow = receive;
  m.computeBoundingSphere();
  return m;
}

export function setInstance(mesh, i, x, y, z, ry, sx = 1, sy = 1, sz = 1) {
  dummy.position.set(x, y, z);
  dummy.rotation.set(0, ry, 0);
  dummy.scale.set(sx, sy, sz);
  dummy.updateMatrix();
  mesh.setMatrixAt(i, dummy.matrix);
}

/* ---------------- procedural textures ---------------- */

function canvas(w, h) {
  const cv = document.createElement("canvas");
  cv.width = w;
  cv.height = h;
  return [cv, cv.getContext("2d")];
}

function finish(cv, { repeat = true, srgb = true } = {}) {
  const tex = new THREE.CanvasTexture(cv);
  if (srgb) tex.colorSpace = THREE.SRGBColorSpace;
  if (repeat) tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.anisotropy = 8;
  tex.userData.shared = true;
  return tex;
}

/*
 * Container atlas: one 128 px row per marking. Columns: a 40 ft side (1024 px: corrugation, rails, corner posts and
 * the name), the door end (256 px: doors, lock rods, castings) and the roof (256 px: transverse ribs). Red = shading,
 * green = lettering mask; the colour itself comes from each instance, the ink from aInk.
 */
const ATLAS_W = 1536;
const ROW_H = 128;
let _atlas = null;
export function containerAtlas() {
  const random = mulberry32(73);
  if (_atlas) return _atlas;
  const rows = MARKINGS.length;
  const H = rows * ROW_H;
  const [sh, g] = canvas(ATLAS_W, H);
  const [mk, m] = canvas(ATLAS_W, H);
  m.fillStyle = "#000";
  m.fillRect(0, 0, ATLAS_W, H);
  for (let row = 0; row < rows; row++) {
    const y0 = row * ROW_H;
    // side: corrugation
    g.fillStyle = "#E6E6E6";
    g.fillRect(0, y0, 1024, ROW_H);
    for (let x = 0; x < 1024; x += 20.5) {
      const grd = g.createLinearGradient(x, 0, x + 20.5, 0);
      grd.addColorStop(0, "#D6D6D6");
      grd.addColorStop(0.3, "#F7F7F7");
      grd.addColorStop(0.55, "#E4E4E4");
      grd.addColorStop(0.8, "#C9C9C9");
      grd.addColorStop(1, "#D6D6D6");
      g.fillStyle = grd;
      g.fillRect(x, y0 + 7, 20.5, ROW_H - 14);
    }
    g.fillStyle = "#CFCFCF";
    g.fillRect(0, y0, 1024, 7);
    g.fillStyle = "#BDBDBD";
    g.fillRect(0, y0 + ROW_H - 7, 1024, 7);
    g.fillStyle = "#C3C3C3";
    g.fillRect(0, y0, 8, ROW_H);
    g.fillRect(1016, y0, 8, ROW_H);
    // door end
    g.fillStyle = "#E2E2E2";
    g.fillRect(1024, y0, 256, ROW_H);
    g.fillStyle = "#C8C8C8";
    g.fillRect(1024 + 126, y0 + 6, 4, ROW_H - 12);
    g.fillStyle = "#B4B4B4";
    for (const x of [1024 + 44, 1024 + 86, 1024 + 170, 1024 + 212]) g.fillRect(x, y0 + 8, 5, ROW_H - 16);
    for (const x of [1024 + 36, 1024 + 80, 1024 + 162, 1024 + 206]) for (const yy of [y0 + 40, y0 + 84]) g.fillRect(x, yy, 18, 5);
    g.fillStyle = "#A8A8A8";
    g.fillRect(1024, y0, 256, 8);
    g.fillRect(1024, y0 + ROW_H - 8, 256, 8);
    g.fillRect(1024, y0, 10, ROW_H);
    g.fillRect(1024 + 246, y0, 10, ROW_H);
    // roof: transverse ribs
    g.fillStyle = "#DEDEDE";
    g.fillRect(1280, y0, 256, ROW_H);
    for (let x = 1280; x < 1536; x += 9) {
      g.fillStyle = "#CDCDCD";
      g.fillRect(x, y0, 3, ROW_H);
    }
    // weathering streaks on the side
    for (let i = 0; i < 30; i++) {
      g.fillStyle = `rgba(90,80,70,${0.02 + random() * 0.04})`;
      g.fillRect(random() * 1024, y0 + 7, 1 + random() * 3, 30 + random() * 70);
    }
    // lettering
    const text = MARKINGS[row];
    if (text) {
      m.fillStyle = "#FFF";
      let size = 70;
      m.font = `800 ${size}px "Helvetica Neue", Arial, sans-serif`;
      while (m.measureText(text).width > 760 && size > 30) {
        size -= 2;
        m.font = `800 ${size}px "Helvetica Neue", Arial, sans-serif`;
      }
      m.textAlign = "center";
      m.textBaseline = "middle";
      m.fillText(text, 512, y0 + ROW_H / 2 + 2);
      // the small owner code and number top right, as on every box
      m.font = `600 13px "Helvetica Neue", Arial, sans-serif`;
      m.textAlign = "right";
      m.fillText(`${text.replace(/[^A-Z]/g, "").slice(0, 3).padEnd(3, "U")}U ${String(100000 + row * 7919).slice(0, 6)}`, 1000, y0 + 22);
    }
  }
  const a = g.getImageData(0, 0, ATLAS_W, H);
  const b = m.getImageData(0, 0, ATLAS_W, H);
  for (let i = 0; i < a.data.length; i += 4) {
    a.data[i + 1] = b.data[i];
    a.data[i + 2] = 0;
    a.data[i + 3] = 255;
  }
  g.putImageData(a, 0, 0);
  _atlas = finish(sh, { repeat: false, srgb: false });
  _atlas.generateMipmaps = true;
  return _atlas;
}

let _containerGeo = null;
/** Unit box (length along x) whose faces map to the atlas columns: sides, door ends, roof. */
export function containerGeometry() {
  if (_containerGeo) return _containerGeo;
  const g = new THREE.BoxGeometry(1, 1, 1);
  const uv = g.attributes.uv;
  // face order: +x, -x, +y, -y, +z, -z (4 vertices each)
  const cols = [
    [1024, 1280], [1024, 1280], // door ends
    [1280, 1536], [1280, 1536], // roof / floor
    [0, 1024], [0, 1024], // long sides
  ];
  for (let f = 0; f < 6; f++) {
    const [u0, u1] = cols[f];
    for (let k = 0; k < 4; k++) uv.setX(f * 4 + k, (u0 + uv.getX(f * 4 + k) * (u1 - u0)) / ATLAS_W);
  }
  uv.needsUpdate = true;
  g.userData.shared = true;
  _containerGeo = g;
  return g;
}

let _containerMat = null;
export function containerMaterial() {
  if (!_containerMat) {
    const rows = MARKINGS.length;
    const m = new THREE.MeshStandardMaterial({ map: containerAtlas(), roughness: 0.58, metalness: 0.12 });
    m.onBeforeCompile = (sh) => {
      sh.vertexShader = sh.vertexShader
        .replace("#include <common>", "#include <common>\nattribute float aBrand;\nattribute float aInk;\nvarying vec2 vAtlas;\nvarying float vInk;\nvarying vec2 vContainerUv;")
        .replace("#include <uv_vertex>", `#include <uv_vertex>\n  vContainerUv = uv;\n  vAtlas = vec2(uv.x, 1.0 - (1.0 - uv.y + aBrand) / ${rows.toFixed(1)});\n  vInk = aInk;`);
      sh.fragmentShader = sh.fragmentShader
        .replace("#include <common>", "#include <common>\nvarying vec2 vAtlas;\nvarying float vInk;\nvarying vec2 vContainerUv;")
        .replace("#include <map_fragment>", "vec4 atl = texture2D(map, vAtlas);")
        .replace("#include <color_fragment>", "#include <color_fragment>\n  diffuseColor.rgb = mix(diffuseColor.rgb, mix(vec3(0.93), vec3(0.06, 0.08, 0.14), vInk), atl.g) * atl.r;")
        .replace("#include <normal_fragment_maps>", `#include <normal_fragment_maps>
          // Physical corrugation on the long walls, filtered out below a pixel.
          // Derivatives construct the surface tangent without per-instance tangents.
          float ribPhase = vContainerUv.x * 1536.0 / 20.5 * 6.283185;
          float resolved = 1.0 - smoothstep(0.8, 2.8, fwidth(ribPhase));
          float sideWall = 1.0 - step(0.666, vContainerUv.x);
          vec3 q0 = dFdx(-vViewPosition), q1 = dFdy(-vViewPosition);
          vec2 st0 = dFdx(vContainerUv), st1 = dFdy(vContainerUv);
          vec3 tangent = q0 * st1.y - q1 * st0.y;
          tangent *= inversesqrt(max(dot(tangent, tangent), 1e-12));
          normal = normalize(normal + tangent * cos(ribPhase) * 0.32 * resolved * sideWall);
        `);
    };
    m.customProgramCacheKey = () => "container-corrugated-v2";
    m.userData.shared = true;
    _containerMat = m;
  }
  return _containerMat;
}

/** Instanced containers: items {p:[x,y,z], s:[sx,sy,sz], r?, c, b?, k?} (b = marking row, k = ink). */
export function containerMesh(items, { cast = true, receive = true } = {}) {
  const g = containerGeometry().clone();
  const n = Math.max(1, items.length);
  const brand = new Float32Array(n);
  const ink = new Float32Array(n);
  items.forEach((it, i) => {
    brand[i] = it.b || 0;
    ink[i] = it.k || 0;
  });
  g.setAttribute("aBrand", new THREE.InstancedBufferAttribute(brand, 1));
  g.setAttribute("aInk", new THREE.InstancedBufferAttribute(ink, 1));
  const mesh = instanced(g, containerMaterial(), items, { cast, receive });
  return mesh;
}

let _concrete = null;
/** Apron paving: 5 m slabs with joints, tyre marks and stains (512 px = 20 m). */
export function concreteTexture() {
  const random = mulberry32(97);
  if (_concrete) return _concrete;
  const [cv, g] = canvas(512, 512);
  g.fillStyle = C.concrete;
  g.fillRect(0, 0, 512, 512);
  const img = g.getImageData(0, 0, 512, 512);
  for (let i = 0; i < img.data.length; i += 4) {
    const n = (random() - 0.5) * 16;
    img.data[i] += n;
    img.data[i + 1] += n;
    img.data[i + 2] += n;
  }
  g.putImageData(img, 0, 0);
  for (let i = 0; i < 60; i++) {
    g.fillStyle = `rgba(90,80,70,${0.02 + random() * 0.05})`;
    g.beginPath();
    g.ellipse(random() * 512, random() * 512, 10 + random() * 60, 6 + random() * 30, random() * 3, 0, 7);
    g.fill();
  }
  g.strokeStyle = "rgba(60,55,50,0.10)";
  for (let i = 0; i < 26; i++) {
    g.lineWidth = 3 + random() * 5;
    const y = random() * 512;
    g.beginPath();
    g.moveTo(0, y);
    g.bezierCurveTo(170, y + (random() - 0.5) * 80, 340, y + (random() - 0.5) * 80, 512, y + (random() - 0.5) * 40);
    g.stroke();
  }
  g.strokeStyle = "rgba(70,64,58,0.45)";
  g.lineWidth = 1.5;
  for (let k = 0; k <= 512; k += 128) {
    g.beginPath();
    g.moveTo(k, 0);
    g.lineTo(k, 512);
    g.moveTo(0, k);
    g.lineTo(512, k);
    g.stroke();
  }
  _concrete = finish(cv);
  return _concrete;
}

let _detail = null;
/** Soft grey surface noise (mean ~0.93) multiplied over flat ground so large areas are not flat colour. */
export function detailTexture() {
  const random = mulberry32(113);
  if (_detail) return _detail;
  const [cv, g] = canvas(256, 256);
  const img = g.createImageData(256, 256);
  // value noise at two scales, tileable
  const grid = (n, seed) => {
    let x = seed;
    const r = () => ((x = (x * 16807) % 2147483647) / 2147483647);
    return Array.from({ length: n * n }, r);
  };
  const g1 = grid(8, 3);
  const g2 = grid(32, 7);
  const sample = (gr, n, u, v) => {
    const x = u * n;
    const y = v * n;
    const x0 = Math.floor(x);
    const y0 = Math.floor(y);
    const fx = x - x0;
    const fy = y - y0;
    const at = (i, j) => gr[((j % n) + n) % n * n + (((i % n) + n) % n)];
    const sx = fx * fx * (3 - 2 * fx);
    const sy = fy * fy * (3 - 2 * fy);
    return at(x0, y0) * (1 - sx) * (1 - sy) + at(x0 + 1, y0) * sx * (1 - sy) + at(x0, y0 + 1) * (1 - sx) * sy + at(x0 + 1, y0 + 1) * sx * sy;
  };
  for (let j = 0; j < 256; j++) {
    for (let i = 0; i < 256; i++) {
      const v = 0.86 + 0.09 * sample(g1, 8, i / 256, j / 256) + 0.05 * sample(g2, 32, i / 256, j / 256) + (random() - 0.5) * 0.03;
      const k = (j * 256 + i) * 4;
      img.data[k] = img.data[k + 1] = img.data[k + 2] = Math.round(Math.min(1, v) * 255);
      img.data[k + 3] = 255;
    }
  }
  g.putImageData(img, 0, 0);
  _detail = finish(cv, { srgb: false });
  return _detail;
}

let _asphalt = null;
export function asphaltTexture() {
  const random = mulberry32(127);
  if (_asphalt) return _asphalt;
  const [cv, g] = canvas(256, 256);
  g.fillStyle = "#9B968E";
  g.fillRect(0, 0, 256, 256);
  const img = g.getImageData(0, 0, 256, 256);
  for (let i = 0; i < img.data.length; i += 4) {
    const n = (random() - 0.5) * 22;
    img.data[i] += n;
    img.data[i + 1] += n;
    img.data[i + 2] += n;
  }
  g.putImageData(img, 0, 0);
  for (let i = 0; i < 30; i++) {
    g.fillStyle = `rgba(60,55,50,${0.03 + random() * 0.06})`;
    g.beginPath();
    g.ellipse(random() * 256, random() * 256, 6 + random() * 30, 4 + random() * 20, random() * 3, 0, 7);
    g.fill();
  }
  _asphalt = finish(cv);
  return _asphalt;
}

/** Flat painted text on the ground (section numbers). */
export function textTexture(text, { size = 96, color = "rgba(255,255,255,0.92)", weight = 700, w = 256, h = 128, font = "system-ui, -apple-system, Segoe UI, Roboto, sans-serif" } = {}) {
  const [cv, ctx] = canvas(w, h);
  ctx.fillStyle = color;
  ctx.font = `${weight} ${size}px ${font}`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, w / 2, h / 2 + size * 0.04);
  const tex = new THREE.CanvasTexture(cv);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  return tex;
}

/** Diagonal hazard stripes, tiled. */
export function stripeTexture(a, b, { size = 64, ratio = 0.5 } = {}) {
  const [cv, ctx] = canvas(size, size);
  ctx.fillStyle = b;
  ctx.fillRect(0, 0, size, size);
  ctx.strokeStyle = a;
  ctx.lineWidth = size * ratio * 0.7071;
  ctx.beginPath();
  for (let k = -2; k <= 2; k++) {
    ctx.moveTo(k * size, size);
    ctx.lineTo(k * size + size, 0);
  }
  ctx.stroke();
  const tex = new THREE.CanvasTexture(cv);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.anisotropy = 4;
  return tex;
}

/** Thin flat slab on the ground (roads, pads, paint). */
export function flat(parts, x0, z0, x1, z1, y, color, t = 0.2) {
  parts.box((x0 + x1) / 2, y - t / 2, (z0 + z1) / 2, Math.abs(x1 - x0), t, Math.abs(z1 - z0), color);
}

export const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
export const smooth = (u) => {
  const x = clamp(u, 0, 1);
  return x * x * (3 - 2 * x);
};
export const lerp = (a, b, t) => a + (b - a) * t;

export function disposeTree(obj) {
  obj.traverse((o) => {
    if (o.geometry && o.geometry !== unitBox && o.geometry !== unitCyl && o.geometry !== _containerGeo && !o.geometry.userData.shared) o.geometry.dispose();
    if (o.material) {
      const ms = Array.isArray(o.material) ? o.material : [o.material];
      for (const m of ms) {
        if (m.userData && m.userData.shared) continue;
        if (m.map && !(m.map.userData && m.map.userData.shared)) m.map.dispose();
        m.dispose();
      }
    }
  });
}
