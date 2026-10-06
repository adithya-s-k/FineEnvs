// The Port of Barcelona digital twin: open-data layers (OpenStreetMap geometry and land cover, terrain tiles) built by
// tools/twin/build_twin.py into ./twin/, and the frame that puts one quay's sections along the scene's x axis.
//
// Twin frame: UTM 31N metres minus an origin (X east, Y north). Scene frame of a task: x along the quay from its
// south-west end towards the north-east, z from the quay edge out over the water, y up; x = 0 is the middle of the
// task's sections, so section s starts at X0 + (s - first) * secM as before.

let loading = null;

async function gunzipJson(res) {
  const buf = new Uint8Array(await res.arrayBuffer());
  if (buf[0] === 0x1f && buf[1] === 0x8b) {
    const ds = new Blob([buf]).stream().pipeThrough(new DecompressionStream("gzip"));
    return JSON.parse(await new Response(ds).text());
  }
  return JSON.parse(new TextDecoder().decode(buf)); // already decoded by a proxy
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const im = new Image();
    im.crossOrigin = "anonymous";
    im.onload = () => resolve(im);
    im.onerror = () => reject(new Error(`could not load ${url}`));
    im.src = url;
  });
}

function pixels(im) {
  const cv = document.createElement("canvas");
  cv.width = im.naturalWidth;
  cv.height = im.naturalHeight;
  const ctx = cv.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(im, 0, 0);
  return ctx.getImageData(0, 0, cv.width, cv.height);
}

export function loadTwin() {
  if (!loading) {
    loading = (async () => {
      const base = new URL("./twin/", import.meta.url);
      const [data, terrain, cover, scenery, surface] = await Promise.all([
        fetch(new URL("twin.json.gz", base)).then((r) => {
          if (!r.ok) throw new Error(`twin.json.gz: HTTP ${r.status}`);
          return gunzipJson(r);
        }),
        loadImage(new URL("terrain.png", base).href),
        loadImage(new URL("cover.png", base).href),
        fetch(new URL("scenery.json.gz", base)).then(r => {
          if (!r.ok) throw new Error(`scenery.json.gz: HTTP ${r.status}`);
          return gunzipJson(r);
        }),
        loadImage(new URL("surface.webp", base).href),
      ]);
      const tp = pixels(terrain);
      const { w, h } = data.frame.terrain;
      const heights = new Float32Array(w * h);
      for (let i = 0; i < w * h; i++) heights[i] = (tp.data[i * 4] * 256 + tp.data[i * 4 + 1]) / 10 - 50;
      const cp = pixels(cover);
      const cls = new Uint8Array(cp.width * cp.height);
      for (let i = 0; i < cls.length; i++) cls[i] = cp.data[i * 4];
      return { ...data, buildings: [...data.buildings, ...scenery.buildings], roads: scenery.roads, parking: scenery.parking, surface, heights, cover: { w: cp.width, h: cp.height, cls }, coverPolys: data.cover };
    })();
  }
  return loading;
}

/** Height of the rendered terrain (m, scene y) at a twin point; 0 on flat land, negative under water. */
export function terrainAt(twin, X, Y) {
  const f = twin.frame;
  const { w, h } = f.terrain;
  const j = (X - f.extent[0]) / f.grid;
  const i = (f.extent[3] - Y) / f.grid; // row 0 = north
  const j0 = Math.max(0, Math.min(w - 2, Math.floor(j)));
  const i0 = Math.max(0, Math.min(h - 2, Math.floor(i)));
  const fj = Math.min(1, Math.max(0, j - j0));
  const fi = Math.min(1, Math.max(0, i - i0));
  const H = twin.heights;
  return H[i0 * w + j0] * (1 - fi) * (1 - fj) + H[i0 * w + j0 + 1] * (1 - fi) * fj + H[(i0 + 1) * w + j0] * fi * (1 - fj) + H[(i0 + 1) * w + j0 + 1] * fi * fj;
}

/** Height of a rendered terrain triangle, clamped to the flat harbour apron. */
export function surfaceAt(twin, X, Y, step = 1) {
  const f = twin.frame;
  const { w, h } = f.terrain;
  const cols = Math.floor((w - 1) / step) + 1, rows = Math.floor((h - 1) / step) + 1;
  const j = Math.max(0, Math.min(cols - 1.00001, (X - f.extent[0]) / (f.grid * step)));
  const i = Math.max(0, Math.min(rows - 1.00001, (f.extent[3] - Y) / (f.grid * step)));
  const x = Math.floor(j) * step, y = Math.floor(i) * step, u = j - Math.floor(j), v = i - Math.floor(i);
  const H = twin.heights;
  const a = H[y * w + x], b = H[y * w + x + step], c = H[(y + step) * w + x], d = H[(y + step) * w + x + step];
  return Math.max(0, u + v <= 1 ? a + (b - a) * u + (c - a) * v : d + (c - d) * (1 - u) + (b - d) * (1 - v));
}

/** Land-cover class name at a twin point ("water" off the land). */
export function coverAt(twin, X, Y) {
  const f = twin.frame;
  const c = twin.cover;
  const j = Math.floor((X - f.extent[0]) / f.cover_res);
  const i = Math.floor((f.extent[3] - Y) / f.cover_res);
  if (j < 0 || i < 0 || j >= c.w || i >= c.h) return "water";
  const k = c.cls[i * c.w + j];
  return k === 255 ? "water" : twin.cover_classes[k];
}

/** The scene frame of a task's quay. */
export function siteFrame(twin, task) {
  const quay = twin.sites[task.quay] ? task.quay : "36A";
  const s = twin.sites[quay];
  const dx = s.ne[0] - s.sw[0];
  const dy = s.ne[1] - s.sw[1];
  const len = Math.hypot(dx, dy);
  const d = [dx / len, dy / len];
  const n = [d[1], -d[0]]; // south-east: the water side at both quays
  const secM = task.section_m || 42;
  const first = task.first_section;
  const last = task.last_section;
  const L = (last - first + 1) * secM;
  const o = (first - 1) * secM + L / 2; // section 1 starts at the south-west end of the quay
  const O = [s.sw[0] + d[0] * o, s.sw[1] + d[1] * o];
  const toLocal = (X, Y) => {
    const ax = X - O[0];
    const ay = Y - O[1];
    return [ax * d[0] + ay * d[1], ax * n[0] + ay * n[1]];
  };
  const toTwin = (x, z) => [O[0] + d[0] * x + n[0] * z, O[1] + d[1] * x + n[1] * z];
  const r = twin.routes;
  const loc = (p) => toLocal(p[0], p[1]);
  const site = r[quay] || {};
  const anch = r.anchorage;
  const routes = {
    sea: loc(r.sea),
    mouthOut: loc(r.mouth_out),
    mouthIn: loc(r.mouth_in),
    via: (site.via || []).map(loc),
    entry: loc(site.entry || r.mouth_in),
    lane: site.lane || 300,
    anchorage: { centre: loc(anch.centre), along: [anch.along[0] * d[0] + anch.along[1] * d[1], anch.along[0] * n[0] + anch.along[1] * n[1]], half: anch.half },
  };
  // which end of the quay ships come in from (+1: the north-east end)
  routes.mouthSide = routes.entry[0] > 0 ? 1 : -1;
  return {
    quay,
    site: s,
    d,
    nv: n, // unit vector towards the water (twin x, y)
    O,
    secM,
    first,
    last,
    L,
    X0: -L / 2,
    X1: L / 2,
    quayX0: -o,
    quayX1: len - o,
    railC: -s.rail_offset,
    toLocal,
    toTwin,
    routes,
    cruise: r.cruise && quay === "24B" ? { a: loc(r.cruise.a), b: loc(r.cruise.b) } : null,
  };
}
