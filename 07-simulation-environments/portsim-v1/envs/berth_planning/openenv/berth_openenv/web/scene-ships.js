// Container ships: a lofted hull scaled to length and beam, in the livery of the line whose name the ship carries,
// with hatch covers, lashing bridges and deck stacks; and a deterministic motion schedule along the port's real
// routes (sea -> anchorage offshore -> south entrance -> channel/basin -> berth -> back out), evaluated at any
// fractional hour so scrubbing is exact.
import * as THREE from "three";
import { mergeVertices } from "three/addons/utils/BufferGeometryUtils.js";
import { hashStr, mulberry32 } from "./model.js";
import { C, MARKINGS, Parts, boxSpec, clamp, containerMesh, setInstance, vcMat, wearColor } from "./scene-kit.js";
import { BOX_H, BOX_L, BOX_W, WATER_Y } from "./scene-world.js";

/* ---------------- liveries ---------------- */

// Hull, boot-topping, superstructure, funnel, funnel top, and the box colours the line's own containers bring.
const LIVERIES = {
  msc: { word: "MSC", hull: "#1D2127", boot: "#7C2A22", house: "#F2F0EA", funnel: "#D7A92F", top: "#1D2127", boxes: ["#D6AC48", "#E4D7B6", "#EDEDE8", "#A2A8AC", "#D6AC48"] },
  maersk: { word: "MAERSK", hull: "#3F8DBF", boot: "#9A2F27", house: "#F4F3EF", funnel: "#3F8DBF", top: "#3F8DBF", star: true, boxes: ["#78A6CF", "#78A6CF", "#EDEDE8", "#78A6CF"] },
  cma: { word: "CMA CGM", hull: "#16244A", boot: "#A3262A", house: "#F2F0EA", funnel: "#F2F0EA", top: "#16244A", boxes: ["#2F4F86", "#2F4F86", "#EDEDE8", "#A2A8AC"] },
  one: { word: "ONE", hull: "#C2246D", boot: "#5B1A33", house: "#F2F0EA", funnel: "#C2246D", top: "#2B2B30", boxes: ["#C64C86", "#C64C86", "#EDEDE8"] },
  hapag: { word: "Hapag-Lloyd", hull: "#1E2B4A", boot: "#8E2A23", house: "#F2F0EA", funnel: "#E0702C", top: "#1E2B4A", boxes: ["#E57A33", "#E57A33", "#A2A8AC"] },
  evergreen: { word: "EVERGREEN", hull: "#2E6E46", boot: "#7A2A23", house: "#F2F0EA", funnel: "#2E6E46", top: "#E0702C", boxes: ["#3F8152", "#3F8152", "#EDEDE8"] },
  cosco: { word: "COSCO SHIPPING", hull: "#22314F", boot: "#8E2A23", house: "#F2F0EA", funnel: "#2E5A93", top: "#22314F", boxes: ["#4C719E", "#EDEDE8", "#A2A8AC"] },
  zim: { word: "ZIM", hull: "#2B2F36", boot: "#7C2A22", house: "#F2F0EA", funnel: "#E9E8E3", top: "#2B2F36", boxes: ["#A2A8AC", "#EDEDE8", "#A5503A"] },
  hmm: { word: "HMM", hull: "#34404F", boot: "#8E2A23", house: "#F2F0EA", funnel: "#D86A2B", top: "#34404F", boxes: ["#C6CACC", "#E57A33", "#A2A8AC"] },
};
const GENERIC = [
  { hull: "#2B3442", boot: "#7C2A22", house: "#F0EEE8", funnel: "#2B3442", top: "#1D2127" },
  { hull: "#3D4A3A", boot: "#7C2A22", house: "#F0EEE8", funnel: "#C9A43F", top: "#1D2127" },
  { hull: "#5A2A27", boot: "#3A1D1B", house: "#F0EEE8", funnel: "#E9E8E3", top: "#5A2A27" },
  { hull: "#1F3550", boot: "#8E2A23", house: "#F0EEE8", funnel: "#B8322A", top: "#1F3550" },
  { hull: "#4A4F55", boot: "#7C2A22", house: "#F0EEE8", funnel: "#2E5A93", top: "#1D2127" },
  { hull: "#23262B", boot: "#7C2A22", house: "#F0EEE8", funnel: "#E9E8E3", top: "#B8322A" },
];

export function liveryFor(name, imo) {
  const n = String(name || "").toUpperCase();
  const w = n.split(/\s+/)[0];
  let key = null;
  if (w === "MSC") key = "msc";
  else if (n.startsWith("MAERSK") || n.startsWith("MERETE MAERSK")) key = "maersk";
  else if (n.startsWith("CMA CGM") || w === "CMA" || w === "APL") key = "cma";
  else if (w === "ONE") key = "one";
  else if (w === "EVER") key = "evergreen";
  else if (w === "COSCO" || w === "CSCL" || w === "OOCL") key = "cosco";
  else if (w === "ZIM") key = "zim";
  else if (w === "HMM" || w === "HYUNDAI") key = "hmm";
  else if (/EXPRESS$/.test(n)) key = "hapag";
  if (key) return { key, ...LIVERIES[key] };
  const h = hashStr(`${name}|${imo || ""}`);
  return { key: "generic", ...GENERIC[h % GENERIC.length], boxes: null };
}

/* ---------------- lettering ---------------- */

const letterCache = new Map();
/** Painted hull letters on a transparent canvas texture (cached by text). */
function lettering(text, { color = "#FFFFFF", bold = true } = {}) {
  const key = `${text}|${color}|${bold}`;
  if (letterCache.has(key)) return letterCache.get(key);
  const cv = document.createElement("canvas");
  cv.width = 1024;
  cv.height = 128;
  const g = cv.getContext("2d");
  g.fillStyle = color;
  let size = 104;
  g.font = `${bold ? 800 : 600} ${size}px "Helvetica Neue", Arial, sans-serif`;
  while (g.measureText(text).width > 1000 && size > 30) {
    size -= 4;
    g.font = `${bold ? 800 : 600} ${size}px "Helvetica Neue", Arial, sans-serif`;
  }
  g.textAlign = "center";
  g.textBaseline = "middle";
  g.fillText(text, 512, 66);
  const tex = new THREE.CanvasTexture(cv);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 8;
  tex.userData.shared = true;
  letterCache.set(key, tex);
  return tex;
}

/* ---------------- hull ---------------- */

function halfWidths(L, B, x) {
  // x in [-L/2, L/2]; waterline and deck half-breadths, the deck flaring wider than the waterline at the bow
  const u = (x + L / 2) / L; // 0 stern .. 1 bow
  const bowStart = 0.74;
  const sternEnd = 0.08;
  let wl = B / 2;
  let dk = B / 2;
  if (u > bowStart) {
    const v = (u - bowStart) / (1 - bowStart);
    wl = (B / 2) * Math.sqrt(Math.max(0, 1 - Math.pow(Math.min(1, v * 1.04), 1.9)));
    dk = (B / 2) * Math.sqrt(Math.max(0, 1 - Math.pow(v, 2.8)));
  } else if (u < sternEnd) {
    const v = u / sternEnd;
    wl = (B / 2) * (0.62 + 0.38 * Math.sqrt(v));
    dk = (B / 2) * (0.93 + 0.07 * v);
  }
  return [Math.max(0.05, wl), Math.max(0.05, dk)];
}

function loftHull(L, B, F, T, liv, band) {
  const NS = 44;
  const xs = [];
  for (let i = 0; i <= NS; i++) {
    const u = i / NS;
    xs.push(-L / 2 + L * (u < 0.7 ? u : 0.7 + (u - 0.7) * 1.0));
  }
  // section rows (y, which width, colour): keel .. boot top | hull .. sheer band .. deck edge
  const hull = new THREE.Color(liv.hull);
  const boot = new THREE.Color(liv.boot);
  const sheer = band ? new THREE.Color(band) : hull;
  const rows = [
    [-T, -1, boot],
    [-T * 0.45, 0, boot],
    [0.9, 0, boot],
    [0.95, 0, hull],
    [F * 0.55, 0.5, hull],
    [F - 1.4, 0.85, hull],
    [F - 1.35, 0.85, sheer],
    [F, 1, sheer],
  ];
  const pos = [];
  const col = [];
  const P = (x, y, z, c) => {
    pos.push(x, y, z);
    col.push(c.r, c.g, c.b);
  };
  const ringAt = (x) => {
    const [wl, dk] = halfWidths(L, B, x);
    return rows.map(([y, t, c]) => [y, t < 0 ? wl * 0.55 : wl + (dk - wl) * t, c]);
  };
  for (let i = 0; i < NS; i++) {
    const a = ringAt(xs[i]);
    const b = ringAt(xs[i + 1]);
    for (let k = 0; k < rows.length - 1; k++) {
      for (const s of [-1, 1]) {
        const [ya0, wa0, c0] = a[k];
        const [ya1, wa1, c1] = a[k + 1];
        const [yb0, wb0] = b[k];
        const [yb1, wb1] = b[k + 1];
        const q = [
          [xs[i], ya0, s * wa0, c0],
          [xs[i + 1], yb0, s * wb0, c0],
          [xs[i + 1], yb1, s * wb1, c1],
          [xs[i], ya1, s * wa1, c1],
        ];
        const order = s > 0 ? [0, 1, 2, 0, 2, 3] : [0, 2, 1, 0, 3, 2];
        for (const o of order) P(q[o][0], q[o][1], q[o][2], q[o][3]);
      }
    }
  }
  // transom (stern face)
  const st = ringAt(xs[0]);
  for (let k = 0; k < rows.length - 1; k++) {
    const [y0, w0, c0] = st[k];
    const [y1, w1, c1] = st[k + 1];
    const q = [
      [xs[0], y0, -w0, c0],
      [xs[0], y0, w0, c0],
      [xs[0], y1, w1, c1],
      [xs[0], y1, -w1, c1],
    ];
    for (const o of [0, 2, 1, 0, 3, 2]) P(q[o][0], q[o][1], q[o][2], q[o][3]);
  }
  // flat bottom between the keel lines (keeps the hull closed from every angle)
  for (let i = 0; i < NS; i++) {
    const a = ringAt(xs[i])[0];
    const b = ringAt(xs[i + 1])[0];
    const q = [
      [xs[i], -T, -a[1]],
      [xs[i + 1], -T, -b[1]],
      [xs[i + 1], -T, b[1]],
      [xs[i], -T, a[1]],
    ];
    for (const o of [0, 1, 2, 0, 2, 3]) P(q[o][0], q[o][1], q[o][2], boot);
  }
  // deck plate
  const deck = new THREE.Color("#4F5A57");
  for (let i = 0; i < NS; i++) {
    const [, da] = halfWidths(L, B, xs[i]);
    const [, db] = halfWidths(L, B, xs[i + 1]);
    const q = [
      [xs[i], F, -da],
      [xs[i + 1], F, -db],
      [xs[i + 1], F, db],
      [xs[i], F, da],
    ];
    for (const o of [0, 2, 1, 0, 3, 2]) P(q[o][0], q[o][1], q[o][2], deck);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
  // Weld each coloured plating band so the bow catches a continuous highlight.
  const welded = mergeVertices(g, 0.0001);
  g.dispose();
  welded.computeVertexNormals();
  return welded;
}

function roundedRectShape(w, h, r) {
  const s = new THREE.Shape();
  const x = -w / 2;
  const y = -h / 2;
  s.moveTo(x + r, y);
  s.lineTo(x + w - r, y);
  s.quadraticCurveTo(x + w, y, x + w, y + r);
  s.lineTo(x + w, y + h - r);
  s.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  s.lineTo(x + r, y + h);
  s.quadraticCurveTo(x, y + h, x, y + h - r);
  s.lineTo(x, y + r);
  s.quadraticCurveTo(x, y, x + r, y);
  return s;
}

function buildHarbourTug() {
  const group = new THREE.Group(), p = new Parts(), windows = new Parts();
  const body = '#993E32', deck = '#4D5956', ivory = '#DBDED5', rubber = '#1C2529';
  const prism=(parts,x,y,z,w,h,d,r,color)=>{
    const geo=new THREE.ExtrudeGeometry(roundedRectShape(w,d,r),{depth:h,bevelEnabled:false,curveSegments:5,steps:1});
    geo.rotateX(-Math.PI/2);geo.translate(x,y-h/2,z);parts.geo(geo,color);
  };
  prism(p,0,.45,0,29.7,3.4,10.8,4.2,body);
  prism(p,0,2.13,0,30,.68,11,4.15,rubber);
  prism(p,0,2.5,0,28.2,.2,9.8,3.8,deck);
  // Raised bulwarks and continuous rubber fendering, with a working aft deck.
  for(const sd of [-1,1]){
    p.box(-1,3.02,sd*4.75,19,.8,.22,body);
    for(let x=-9;x<11;x+=3.2){
      const tyre=new THREE.TorusGeometry(.62,.22,7,12);tyre.translate(x,1.75,sd*5.32);p.geo(tyre,rubber);
    }
    for(let x=-8;x<9;x+=2.4){p.cyl(x,3.8,sd*4.48,.055,1.0,ivory,5);}
    p.beam([-8,4.3,sd*4.48],[9,4.3,sd*4.48],.07,ivory);
    // Exhausts flank the accommodation, below the radar mast.
    p.cyl(1.5,6.0,sd*3.75,.47,4.8,rubber,10);
    p.cyl(1.5,8.25,sd*3.75,.6,.4,'#555E5C',10);
  }
  prism(p,4,4.0,0,10,2.9,7.3,1.5,ivory);
  prism(p,5.0,5.7,0,8.9,.35,7.3,1.4,'#98A5A3');
  // Glazing wraps a low, chamfered pilothouse rather than a window band on a cube.
  prism(windows,5.1,6.6,0,8.4,1.65,6.9,1.55,'#233C45');
  prism(p,5.1,7.55,0,8.95,.35,7.35,1.7,ivory);
  for(const sd of [-1,1])for(let x=2.8;x<=7.4;x+=1.55)p.box(x,6.6,sd*3.47,.13,1.75,.12,ivory);
  for(const z of [-1.6,0,1.6])p.box(9.31,6.6,z,.12,1.75,.13,ivory);
  p.cyl(4.0,9.15,0,.11,2.8,ivory,7);
  p.beam([4,9.55,-1.5],[4,9.55,1.5],.13,ivory);
  p.box(4.0,10.55,0,2.1,.15,.3,'#C7CBBF');
  p.cyl(6.2,8.5,0,.085,1.55,ivory,6);
  p.box(8.1,7.95,-2.8,.35,.3,.4,'#AB4237');
  p.box(8.1,7.95,2.8,.35,.3,.4,'#4D8970');
  // Towing winch, rope drum, towing pins and bollards are kept visible aft.
  p.box(-6.6,3.1,0,4.8,1.2,4.2,'#727C75');
  p.cylZ(-6.6,4.05,0,.9,2.9,'#B3A184',14);
  for(const sd of [-1,1]){
    p.cylZ(-6.6,4.05,sd*1.6,1.1,.2,'#3C4748',14);
    p.cyl(-10.7,3.65,sd*1.4,.3,1.8,'#CDD1C8',8);
    for(const x of [-11.7,11.5]){p.cyl(x,2.9,sd*2.6,.32,.65,rubber,8);p.box(x,3.2,sd*2.6,1.2,.17,.5,rubber);}
  }
  const hull=p.mesh(vcMat);hull.castShadow=true;hull.receiveShadow=true;group.add(hull);
  group.add(windows.mesh(new THREE.MeshPhysicalMaterial({vertexColors:true,roughness:.2,metalness:.2,clearcoat:1})));
  return group;
}

/** Container ship facing +x, origin at the waterline amidships. */
export function buildShipModel({ length, beam, seed, livery, band = null, lite = false, name = "" }) {
  const shipName = String(name || "").toUpperCase();
  const L = length;
  const B = beam;
  const r = mulberry32(seed);
  const liv = livery;
  const group = new THREE.Group();
  const F = clamp(0.028 * L + 2.5, 6.5, 14);
  const T = clamp(0.034 * L, 5, 15);
  const hullMat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.44, metalness: 0.22, side: THREE.DoubleSide });
  // Welded plate seams, oxidised scuppers and a damp boot-top, in ship-local metres.
  hullMat.onBeforeCompile = sh => {
    sh.uniforms.uFreeboard = { value: F };
    sh.vertexShader = 'varying vec3 vHull;\n' + sh.vertexShader.replace('#include <begin_vertex>', '#include <begin_vertex>\n vHull = position;');
    sh.fragmentShader = 'varying vec3 vHull;\nuniform float uFreeboard;\n' + sh.fragmentShader
      .replace('#include <color_fragment>', `#include <color_fragment>
        vec2 plate = fract(vHull.xy / vec2(7.8, 2.1));
        vec2 edge = min(plate, 1.0 - plate);
        vec2 aa = max(fwidth(vHull.xy / vec2(7.8, 2.1)), vec2(0.002));
        float seam = (1.0 - smoothstep(0.0, aa.x * 1.4, edge.x)) * 0.065
                   + (1.0 - smoothstep(0.0, aa.y * 1.4, edge.y)) * 0.035;
        float streak = pow(max(0.0, sin(vHull.x * 1.71 + sin(vHull.x * 0.37))), 14.0);
        float rust = streak * exp(-abs(vHull.y - uFreeboard + 0.5) * 0.38) * 0.3;
        float salt = (0.5 + 0.5 * sin(vHull.x * 0.93)) * exp(-abs(vHull.y - 1.0) * 1.4) * 0.1;
        diffuseColor.rgb = mix(diffuseColor.rgb * (1.0 - seam - salt), vec3(0.16, 0.065, 0.025), rust);`)
      .replace('#include <roughnessmap_fragment>', `#include <roughnessmap_fragment>
        roughnessFactor *= mix(0.58, 1.0, smoothstep(0.4, 2.3, vHull.y));`);
  };
  hullMat.customProgramCacheKey = () => 'hull-plate-wear-v2';
  const hullMesh = new THREE.Mesh(loftHull(L, B, F, T, liv, band), hullMat);
  hullMesh.castShadow = true;
  hullMesh.receiveShadow = true;
  group.add(hullMesh);

  const twinIsland = L >= 330;
  const tiersMax = L >= 360 ? 9 : L >= 300 ? 8 : L >= 250 ? 7 : L >= 200 ? 5 : 4;
  // Bridge sightline clears the stack, without giving a feeder the tower of a
  // large liner. These are procedural vessel classes, not surveyed ship models.
  const Hs = clamp(tiersMax * BOX_H + 7, 16, 33);
  const sternX = -L / 2;
  const bowX = L / 2;
  const p = new Parts();
  const glazing = new Parts();
  const house = liv.house;
  const glass = "#3A4A57";
  const trim = "#C9C6BE";
  // deckhouse: stepped accommodation decks, a glazed wheelhouse with bridge wings, radar mast on top
  const houseX = twinIsland ? bowX - 0.33 * L : sternX + 0.15 * L;
  const houseW = clamp(B * 0.64, 12, 23);
  const houseB = B * 0.72;
  const decks = Math.max(4, Math.round((Hs - 4) / 2.9));
  for (let k = 0; k < decks; k++) {
    const y = F + k * 2.9;
    const w = houseW - (k > decks - 3 ? 1.5 : 0);
    p.box(houseX, y + 1.45, 0, w, 2.9, houseB - (k % 3 === 2 ? 0.6 : 0), house);
    p.box(houseX, y + 2.86, 0, w + 0.24, 0.13, houseB + 0.18, '#C3C8C4');
    glazing.box(houseX + w / 2 + 0.02, y + 1.8, 0, 0.06, 0.8, houseB * 0.86, glass); // forward windows
    for (const sd of [-1, 1]) for (let wx = -w / 2 + 1.2; wx < w / 2 - 1; wx += 2.2) glazing.box(houseX + wx + 0.6, y + 1.8, sd * (houseB / 2 + 0.02), 1.3, 0.8, 0.06, glass); // side windows
  }
  const wy = F + decks * 2.9;
  p.box(houseX + 0.8, wy + 1.6, 0, houseW - 2.5, 3.2, B + 1.2, house); // wheelhouse + wings across the beam
  glazing.box(houseX + 0.8 + (houseW - 2.5) / 2 + 0.03, wy + 1.9, 0, 0.08, 1.7, B + 0.8, glass);
  for (const sd of [-1, 1]) glazing.box(houseX + 0.8, wy + 1.9, sd * (B / 2 + 0.62), houseW - 3, 1.5, 0.06, glass);
  p.box(houseX + 0.8, wy + 3.35, 0, houseW - 2.3, 0.3, B + 1.4, trim);
  p.box(houseX, wy + 4.6, 0, 3.4, 2.2, 3.4, house);
  p.cyl(houseX, wy + 8.5, 0, 0.35, 6, "#D9D6D0", 6);
  p.box(houseX, wy + 9.4, 0, 0.3, 0.3, 6, "#D9D6D0");
  p.box(houseX, wy + 6.2, 0, 4.6, 0.25, 0.6, "#E6B42E"); // radar scanner
  // funnel: tapered casing in the line's colour with its top band
  const funnelX = twinIsland ? sternX + 0.12 * L : houseX - houseW / 2 - 5;
  const fh = twinIsland ? Hs * 0.48 : Hs * 0.56;
  const fBase = twinIsland ? F + 6 : F + Hs * 0.18;
  if (twinIsland) {
    p.box(funnelX + 2, F + 3, 0, 20, 6, B * 0.6, house); // engine casing aft
    p.box(funnelX + 2, F + 6.2, 0, 20.2, 0.4, B * 0.6 + 0.2, trim);
  }
  else {
    p.box(funnelX, F + Hs * 0.09, 0, 8.4, Hs * 0.18, B * 0.43, house);
    for (const sd of [-1, 1]) for (let vent = 0; vent < 5; vent++)
      p.box(funnelX - 2.4 + vent * 1.15, F + 1.8, sd * (B * 0.215 + 0.02), .55, 1.35, .08, '#526066');
  }
  const funnelB=Math.min(8.2,B*.25);
  const fg = new THREE.ExtrudeGeometry(roundedRectShape(7.8,funnelB,.7),{depth:fh,bevelEnabled:false,curveSegments:3});
  fg.rotateX(-Math.PI/2);fg.translate(funnelX,fBase,0);p.geo(fg,liv.funnel);
  const ft = new THREE.ExtrudeGeometry(roundedRectShape(7.9,funnelB+.1,.7),{depth:.85,bevelEnabled:false,curveSegments:3});
  ft.rotateX(-Math.PI/2);ft.translate(funnelX,fBase+fh,0);p.geo(ft,liv.top);
  for(const sd of [-1,1])p.cyl(funnelX-1.4,fBase+fh+1.25,sd*funnelB*.23,.55,1.25,'#30373A',10);
  if (liv.star) for (const sd of [-1, 1]) p.box(funnelX, fBase + fh * 0.62, sd * (Math.min(9, B * 0.2) * 0.36 + 0.05), 3.2, 3.2, 0.1, "#F4F3EF");
  // forecastle with its V-shaped breakwater, mooring deck aft, stern free-fall lifeboat
  const fcX = bowX - 0.055 * L;
  // Follow the hull taper: a rectangular forecastle used to overhang the bow.
  const forecastle=new THREE.Shape(), outline=[];
  for(let i=0;i<=10;i++){const x=bowX-L*.11+L*.11*i/10;outline.push([x,halfWidths(L,B,x)[1]*.97]);}
  for(let i=10;i>=0;i--){const x=bowX-L*.11+L*.11*i/10;outline.push([x,-halfWidths(L,B,x)[1]*.97]);}
  outline.forEach(([x,z],i)=>i?forecastle.lineTo(x,z):forecastle.moveTo(x,z));forecastle.closePath();
  const fgDeck=new THREE.ExtrudeGeometry(forecastle,{depth:2.2,bevelEnabled:false,steps:1});
  fgDeck.rotateX(-Math.PI/2);fgDeck.translate(0,F,0);p.geo(fgDeck,liv.hull);
  const fcTop=new THREE.ShapeGeometry(forecastle);fcTop.rotateX(-Math.PI/2);fcTop.translate(0,F+2.21,0);p.geo(fcTop,'#65716B');
  const bwX = bowX - 0.12 * L;
  for (const sd of [-1, 1]) p.box(bwX, F + 1.8, sd * B * 0.24, .45, 3.6, B * 0.5, '#747C78', sd * 0.42);
  p.cyl(fcX + 2, F + 7, 0, 0.2, 10, "#E3E0D8", 8);
  p.box(sternX + 4.5, F + 1.1, 0, 8, 2.2, B * 0.8, "#E3E0D8");
  const lb = new THREE.CapsuleGeometry(1.6, 6.4, 4, 8);
  lb.rotateZ(Math.PI / 2 - 0.5);
  lb.translate(sternX + 2.5, F + 5.6, 0);
  p.geo(lb, "#E8762B");
  for(const sd of [-1,1]){
    p.beam([sternX+1,F+3,sd*1.3],[sternX+7,F+6,sd*1.3],.2,'#929B98');
    p.beam([sternX+7,F+2.2,sd*1.3],[sternX+7,F+6,sd*1.3],.18,'#929B98');
  }
  for (const sd of [-1, 1]) p.box(bowX - 0.05 * L, F - 2.6, sd * (halfWidths(L, B, bowX - 0.05 * L)[1] + 0.05), 2.4, 2.2, 0.3, "#1D1F22");
  // Open railings and deck fittings make the silhouette read as a ship at eye level.
  for (const sd of [-1, 1]) {
    for (let x = sternX + 3; x < bowX - 4; x += lite ? 12 : 6) {
      const w = halfWidths(L, B, x)[1] - 0.25;
      const nx = Math.min(bowX - 3, x + (lite ? 12 : 6));
      const nw = halfWidths(L, B, nx)[1] - 0.25;
      p.cyl(x, F + 0.55, sd * w, 0.07, 1.1, '#C7CBCC', 4);
      p.beam([x, F + 1.1, sd * w], [nx, F + 1.1, sd * nw], 0.075, '#D4D7D6');
    }
    for (const x of [sternX + 12, fcX - 3]) {
      p.cyl(x, F + 0.45, sd * B * 0.19, 0.8, 0.9, '#43494B', 8);
      p.box(x, F + 1, sd * B * 0.19, 2.4, 0.3, 1.1, '#313739');
    }
    for (let y = 1.5; y < F - 1; y += 0.6) p.box(sternX + L * 0.065, y, sd * (halfWidths(L, B, sternX + L * 0.065)[0] + 0.09), 0.6, 0.12, 0.12, '#E4E5DE');
  }
  const top = new THREE.Mesh(p.geometry(), vcMat);
  top.castShadow = true;
  top.receiveShadow = true;
  group.add(top);
  const glassMat = new THREE.MeshPhysicalMaterial({ vertexColors: true, roughness: 0.16, metalness: 0.25, clearcoat: 1, clearcoatRoughness: 0.12 });
  group.add(glazing.mesh(glassMat));
  // the line's name along the hull and the ship's name on the bow (painted letters)
  if (!lite) {
    const width = (x) => {
      const [wl, dk] = halfWidths(L, B, x);
      return wl + (dk - wl) * 0.85;
    };
    const decal = (tex, x, y, len, h, sd) => {
      const dw = (width(x + 2) - width(x - 2)) / 4;
      const m = new THREE.Mesh(new THREE.PlaneGeometry(len, h), new THREE.MeshStandardMaterial({ map: tex, transparent: true, roughness: 0.6, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -2 }));
      m.position.set(x, y, sd * (width(x) + 0.1));
      m.rotation.y = sd > 0 ? -Math.atan(dw) : Math.PI + Math.atan(dw);
      m.renderOrder = 1;
      group.add(m);
    };
    if (liv.word && L > 150) {
      const len = Math.min(L * 0.42, liv.word.length * F * 0.55);
      const tex = lettering(liv.word, { color: "#F4F4F0" });
      for (const sd of [-1, 1]) decal(tex, L * 0.02, F * 0.52, len, F * 0.62, sd);
    }
    const nameTex = lettering(shipName, { color: "#F2F2EE", bold: false });
    const nl = Math.min(L * 0.16, Math.max(10, shipName.length * 1.6));
    for (const sd of [-1, 1]) decal(nameTex, bowX - 0.12 * L, F - 1.9, nl, 2.0, sd);
  }

  // hatch covers, lashing bridges and the deck stacks: 40 ft bays (13.4 m pitch) from forecastle to stern
  const lash = new Parts();
  const slots = [];
  const usable = B - 1.2;
  const R = Math.max(3, Math.floor(usable / 2.55));
  const rowW = usable / R;
  const bayP = 13.4;
  const x0 = sternX + 0.06 * L + (twinIsland ? 22 : 0);
  const x1 = bowX - 0.13 * L;
  const bias = liv.boxes;
  let bi = 0;
  for (let xc = x0 + BOX_L / 2; xc + BOX_L / 2 <= x1; xc += bayP, bi++) {
    if (Math.abs(xc - houseX) < houseW / 2 + BOX_L / 2 + 1) continue;
    if (!twinIsland && xc < houseX) continue;
    if (twinIsland && Math.abs(xc - funnelX - 2) < 12) continue;
    const hw = halfWidths(L, B, xc + BOX_L / 2)[1] - 0.6;
    lash.box(xc, F + 0.4, 0, BOX_L + 0.6, 0.8, Math.min(B - 1, hw * 2), "#3E4A47");
    lash.box(xc + BOX_L / 2 + 0.55, F + 2.6, 0, 0.5, 5, Math.min(B - 1.2, hw * 2), "#9BA2A6");
    const nearBow = xc > x1 - bayP * 2.5;
    const fwd = twinIsland ? xc > houseX : false;
    let Tm = tiersMax - Math.floor(r() * 2.2) - (nearBow ? 2 : 0) - (fwd ? 1 : 0);
    Tm = Math.max(1, Tm);
    const own = (u) => {
      const c = bias[Math.floor(u * bias.length)];
      const mark = MARKINGS.indexOf(liv.word === "COSCO SHIPPING" ? "COSCO" : liv.word || "");
      return { c, b: c === "#EDEDE8" ? 0 : Math.max(0, mark), k: c === "#E4D7B6" || c === "#D8CBA9" ? 1 : 0 };
    };
    const base = bias && r() < 0.7 ? own(r()) : boxSpec(r());
    for (let j = 0; j < R; j++) {
      const zc = -usable / 2 + rowW * (j + 0.5);
      if (Math.abs(zc) + rowW / 2 > hw) continue;
      const tiers = Math.max(1, Tm - (j === 0 || j === R - 1 ? (r() < 0.5 ? 1 : 0) : 0) - (r() < 0.12 ? 1 : 0));
      for (let t = 0; t < tiers; t++) {
        const sp = r() < 0.55 ? base : bias && r() < 0.5 ? own(r()) : boxSpec(r());
        slots.push({ x: xc, y: F + 0.8 + BOX_H / 2 + t * (BOX_H + 0.02), z: zc, w: rowW - 0.12, t, o: r(), c: wearColor(sp.c, r()), b: sp.b, k: sp.k });
      }
    }
  }
  const lashMesh = lash.mesh(vcMat);
  group.add(lashMesh);
  slots.sort((a, b) => a.t - b.t || a.o - b.o);
  slots.forEach((slot, i) => { slot.id = i; });
  const deck = containerMesh(slots.map((sl) => ({ p: [sl.x, sl.y, sl.z], s: [BOX_L, BOX_H, Math.min(BOX_W, sl.w)], c: sl.c, b: sl.b, k: sl.k })));
  group.add(deck);

  // selection ring on the water
  const ringShape = roundedRectShape(L + 22, B + 22, 10);
  ringShape.holes.push(roundedRectShape(L + 13, B + 13, 6));
  const ringGeo = new THREE.ShapeGeometry(ringShape, 4);
  ringGeo.rotateX(-Math.PI / 2);
  const ringMat = new THREE.MeshBasicMaterial({ color: C.signal, transparent: true, opacity: 0.95, depthWrite: false });
  const ring = new THREE.Mesh(ringGeo, ringMat);
  ring.position.y = 0.4;
  ring.visible = false;
  ring.renderOrder = 2;
  group.add(ring);

  // wake: Kelvin arms and a churned centre strip fading astern
  const wpos = [];
  const wcol = [];
  const quad = (a0, a1, b0, b1, alphaNear, alphaFar) => {
    const pts = [a0, a1, b1, a0, b1, b0];
    const al = [alphaNear, alphaNear, alphaFar, alphaNear, alphaFar, alphaFar];
    pts.forEach((pt, i) => {
      wpos.push(pt[0], 0, pt[1]);
      wcol.push(1, 1, 1, al[i]);
    });
  };
  const WL = Math.max(260, L * 1.1);
  const spread = Math.tan(0.34);
  for (const sd of [-1, 1]) {
    const z0 = sd * B * 0.48;
    const z1 = sd * (B * 0.48 + WL * spread);
    quad([sternX + 4, z0], [sternX + 4, z0 + sd * 2], [sternX - WL, z1], [sternX - WL, z1 + sd * 10], 0.38, 0);
  }
  quad([sternX + 2, -B * 0.3], [sternX + 2, B * 0.3], [sternX - WL * 0.7, -B * 0.8], [sternX - WL * 0.7, B * 0.8], 0.32, 0);
  const wakeGeo = new THREE.BufferGeometry();
  wakeGeo.setAttribute("position", new THREE.Float32BufferAttribute(wpos, 3));
  wakeGeo.setAttribute("color", new THREE.Float32BufferAttribute(wcol, 4));
  const wakeMat = new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, opacity: 0, depthWrite: false, side: THREE.DoubleSide });
  const wakeTime = { value: 0 };
  wakeMat.userData.time = wakeTime;
  wakeMat.onBeforeCompile = sh => {
    sh.uniforms.uWakeTime = wakeTime;
    sh.vertexShader = 'varying vec3 vWake;\n' + sh.vertexShader.replace('#include <begin_vertex>', '#include <begin_vertex>\n vWake = position;');
    sh.fragmentShader = 'uniform float uWakeTime; varying vec3 vWake;\n' + sh.fragmentShader.replace('#include <color_fragment>', `#include <color_fragment>
      float foam = sin(vWake.x * 0.24 + uWakeTime * 2.0 + sin(vWake.z * 0.7)) * sin(vWake.z * 1.1 - uWakeTime);
      diffuseColor.a *= 0.32 + 0.68 * smoothstep(-0.7, 0.6, foam);`);
  };
  wakeMat.customProgramCacheKey = () => 'ship-foam-v1';
  const wake = new THREE.Mesh(wakeGeo, wakeMat);
  wake.position.y = 0.25;
  wake.renderOrder = 1;
  group.add(wake);

  // Generic harbour-assist tugs: hull envelope matches navigation's 30 x 11 m.
  // The visible deck machinery and faceted bridge replace the old stacked boxes.
  const tugs = [];
  for (let i = 0; i < 2; i++) {
    const tug = buildHarbourTug();
    tug.visible = false;
    group.add(tug);
    tugs.push(tug);
  }

  const moorGeo = new THREE.BufferGeometry();
  moorGeo.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(24), 3));
  const moorings = new THREE.LineSegments(moorGeo, new THREE.LineBasicMaterial({ color: '#BBAE91' }));
  moorings.frustumCulled = false;
  moorings.visible = false;
  group.add(moorings);
  const towGeo=new THREE.BufferGeometry();
  towGeo.setAttribute('position',new THREE.Float32BufferAttribute(new Float32Array(12),3));
  const towlines=new THREE.LineSegments(towGeo,new THREE.LineBasicMaterial({color:'#A99B7C'}));
  towlines.frustumCulled=false;towlines.visible=false;group.add(towlines);
  let moorKey = '';
  let removedCargo = new Set();
  const cargoBays = [...new Set(slots.map(s => s.x))].map(x => ({ x, top: Math.max(...slots.filter(s => s.x === x).map(s => s.y + BOX_H / 2)) }));
  const height = F + Hs + 4;
  const pick = new THREE.Mesh(new THREE.BoxGeometry(L, height + 4, B + 8), new THREE.MeshBasicMaterial({ visible: false }));
  pick.position.y = height / 2;
  group.add(pick);

  return {
    group,
    hullMat,
    deck,
    nDeck: slots.length,
    cargoSlots: slots,
    cargoBays,
    ring,
    ringMat,
    wake,
    wakeMat,
    tugs,
    pick,
    height,
    houseX, houseW,
    setTowlines(escorts) {
      towlines.visible=escorts.length>0;
      const a=towGeo.attributes.position;
      escorts.forEach((e,i)=>{
        a.setXYZ(i*2,e.localX,F*.65,Math.sign(e.localZ)*B*.47);
        // The winch sits aft of the tug pilothouse; rotate with its heading.
        a.setXYZ(i*2+1,e.localX-6.6*Math.cos(e.localHeading),4.05,e.localZ+6.6*Math.sin(e.localHeading));
      });
      a.needsUpdate=true;
    },
    setDischarged(ids) {
      let changed = false;
      for (const id of new Set([...removedCargo, ...ids])) {
        if (removedCargo.has(id) === ids.has(id)) continue;
        const s = slots[id];
        if (ids.has(id)) setInstance(deck, id, s.x, -100, s.z, 0, 0, 0, 0);
        else setInstance(deck, id, s.x, s.y, s.z, 0, BOX_L, BOX_H, Math.min(BOX_W, s.w));
        changed = true;
      }
      if (changed) deck.instanceMatrix.needsUpdate = true;
      removedCargo = ids;
    },
    setMooring(on, th, z) {
      moorings.visible = on;
      const key = `${th.toFixed(3)}|${z.toFixed(2)}`;
      if (!on || key === moorKey) return;
      moorKey = key;
      const side = Math.cos(th) > 0 ? -1 : 1;
      const a = moorGeo.attributes.position;
      let k = 0;
      for (const end of [-1, 1]) for (const spring of [false, true]) {
        const x = end * L * 0.39;
        a.setXYZ(k++, x, F - 0.5, side * B * 0.45);
        a.setXYZ(k++, x + (spring ? -end * 32 : end * 18), -WATER_Y + 0.8, side * (z + 2));
      }
      a.needsUpdate = true;
    },
    deckY: F + 0.8,
    length: L,
    beam: B,
  };
}

// Stable livery seed; navigation chooses the berth heading from the approach channel.
export function shipColors(ship) {
  const h = hashStr(`${ship.name}|${ship.imo || ""}`);
  return { seed: h, livery: liveryFor(ship.name, ship.imo) };
}
