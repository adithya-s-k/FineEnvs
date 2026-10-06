"""Digital-twin assets of the Port of Barcelona for the 3D viewer, built from open data.

    python build_twin.py fetch --cache DIR      # OSM (Overpass), terrain tiles, Sentinel-2 window  (~25 MB)
    python build_twin.py build --cache DIR      # -> openenv/berth_openenv/web/twin/

Sources (see web/twin/SOURCES.md):
  OpenStreetMap contributors (ODbL): coastline, breakwaters, buildings, tanks, BEST stacking blocks, APM yard slabs
  Terrain Tiles on AWS (Mapzen terrarium, z13): elevation of Montjuic, the city and Collserola
  Copernicus Sentinel-2 L2A true colour, 17 June 2026, tile 31TDF (via Element 84 Earth Search): only used to check
  the frame against reality and to place routes and the anchorage; the viewer draws its own materials

Frame: UTM zone 31N metres minus ORIGIN (x east, y north). The viewer rotates it per quay so x runs along the quay.
Needs: numpy, shapely, pillow, pyproj, rasterio (fetch only).
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "openenv" / "berth_openenv" / "web" / "twin"

ORIGIN_LL = (41.33, 2.155)
EXTENT = (-9000.0, -7500.0, 8000.0, 12000.0)  # xmin, ymin, xmax, ymax (frame metres)
GRID = 40.0          # terrain grid spacing (m)
GROUND_RES = 10.0    # ground image metres per pixel (Sentinel-2 native)
S2_ITEM = "https://e84-earth-search-sentinel-data.s3.us-west-2.amazonaws.com/sentinel-2-c1-l2a/31/T/DF/2026/6/S2A_T31TDF_20260617T104044_L2A/TCI.tif"
DEM_Z = 13
OVERPASS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]

QUERIES = {
    "osm_a": """[out:json][timeout:180];(
  way["natural"="coastline"](41.26,2.06,41.40,2.24);
  way["man_made"~"breakwater|pier|groyne|quay"](41.26,2.06,41.40,2.24);
  relation["man_made"~"breakwater|pier"](41.26,2.06,41.40,2.24);
  way["landuse"](41.285,2.095,41.385,2.205);
  relation["landuse"](41.285,2.095,41.385,2.205);
  way["natural"~"water|wood|scrub|grassland|beach|sand|wetland|heath"](41.26,2.06,41.40,2.24);
  relation["natural"~"water|wood|scrub|grassland"](41.26,2.06,41.40,2.24);
  way["leisure"~"park|garden"](41.33,2.13,41.38,2.18);
  way["waterway"~"river|canal"](41.26,2.06,41.40,2.24);
  way["aeroway"~"runway|taxiway|apron"](41.27,2.06,41.32,2.12);
);out tags geom qt;""",
    "osm_b": """[out:json][timeout:180];(
  way["building"](41.285,2.095,41.362,2.205);
  relation["building"](41.285,2.095,41.362,2.205);
);out tags geom qt;""",
    "osm_c": """[out:json][timeout:180];(
  way["man_made"="storage_tank"](41.285,2.095,41.385,2.205);
  way["railway"~"rail|light_rail"](41.285,2.095,41.385,2.205);
  way["highway"~"motorway|trunk|primary|secondary|motorway_link|trunk_link|service"](41.285,2.095,41.375,2.205);
  way["man_made"="crane"](41.285,2.095,41.385,2.205);
  node["man_made"~"crane|lighthouse|mast|tower|chimney"](41.26,2.06,41.42,2.24);
  way["man_made"~"tower|chimney|silo"](41.26,2.06,41.42,2.24);
  way["building"]["height"](41.33,2.08,41.43,2.24);
  way["building"]["building:levels"~"^[1-9][0-9]$"](41.33,2.08,41.43,2.24);
  node["natural"="peak"](41.30,2.05,41.45,2.25);
);out tags geom qt;""",
}

# land cover classes drawn on the flat land (index = class id; colours live in the viewer)
COVER = ["urban", "port", "industrial", "residential", "commercial", "rail", "dirt", "grass", "park", "forest", "scrub",
         "sand", "farm", "cemetery", "apron", "runway"]
COVER_TAGS = {
    ("landuse", "harbour"): "port", ("landuse", "port"): "port", ("landuse", "industrial"): "industrial",
    ("landuse", "residential"): "residential", ("landuse", "commercial"): "commercial", ("landuse", "retail"): "commercial",
    ("landuse", "railway"): "rail", ("landuse", "construction"): "dirt", ("landuse", "brownfield"): "dirt",
    ("landuse", "greenfield"): "dirt", ("landuse", "landfill"): "dirt", ("landuse", "grass"): "grass",
    ("landuse", "meadow"): "grass", ("landuse", "village_green"): "grass", ("landuse", "recreation_ground"): "park",
    ("landuse", "flowerbed"): "park", ("landuse", "allotments"): "farm", ("landuse", "farmland"): "farm",
    ("landuse", "orchard"): "farm", ("landuse", "vineyard"): "farm", ("landuse", "forest"): "forest",
    ("landuse", "cemetery"): "cemetery", ("leisure", "park"): "park", ("leisure", "garden"): "park",
    ("natural", "wood"): "forest", ("natural", "scrub"): "scrub", ("natural", "heath"): "scrub",
    ("natural", "grassland"): "grass", ("natural", "wetland"): "grass", ("natural", "beach"): "sand",
    ("natural", "sand"): "sand", ("aeroway", "apron"): "apron", ("aeroway", "runway"): "runway",
}
# drawing order: broad zones first, then detail on top
COVER_ORDER = ["port", "industrial", "residential", "commercial", "rail", "farm", "dirt", "apron", "cemetery", "grass",
               "scrub", "forest", "park", "sand", "runway"]
BUILDING_KIND = {"house": 1, "detached": 1, "terrace": 1, "residential": 1, "apartments": 1, "semidetached_house": 1,
                 "industrial": 2, "warehouse": 2, "hangar": 2, "service": 2, "roof": 2, "garage": 2, "garages": 2,
                 "shed": 2, "storage_tank": 2, "transportation": 2, "office": 3, "commercial": 3, "retail": 3,
                 "hotel": 3, "supermarket": 3, "public": 4, "school": 4, "university": 4, "hospital": 4, "civic": 4,
                 "government": 4, "church": 5, "cathedral": 5, "chapel": 5, "castle": 5, "ruins": 5, "fort": 5}
ROAD_W = {"motorway": 26, "trunk": 20, "primary": 15, "secondary": 11, "motorway_link": 8, "trunk_link": 8}


# ------------------------------------------------------------------ frame

def _proj():
    from pyproj import Transformer
    fwd = Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=True)
    inv = Transformer.from_crs("EPSG:32631", "EPSG:4326", always_xy=True)
    e0, n0 = fwd.transform(ORIGIN_LL[1], ORIGIN_LL[0])
    return fwd, inv, round(e0 / 10) * 10, round(n0 / 10) * 10


FWD, INV, E0, N0 = _proj()


def to_xy(lon, lat):
    e, n = FWD.transform(np.asarray(lon, float), np.asarray(lat, float))
    return np.stack([np.asarray(e) - E0, np.asarray(n) - N0], axis=-1)


def geom_xy(geometry):
    pts = [(p["lon"], p["lat"]) for p in geometry if p and p.get("lon") is not None]
    if not pts:
        return np.zeros((0, 2))
    a = np.array(pts)
    return to_xy(a[:, 0], a[:, 1])


# ------------------------------------------------------------------ fetch

def fetch(cache: Path):
    import time
    import urllib.parse
    import urllib.request
    cache.mkdir(parents=True, exist_ok=True)
    for name, q in QUERIES.items():
        path = cache / f"{name}.json"
        if path.is_file():
            continue
        for attempt in range(8):
            ep = OVERPASS[attempt % len(OVERPASS)]
            req = urllib.request.Request(ep, data=urllib.parse.urlencode({"data": q}).encode(),
                                         headers={"User-Agent": "FineEnvs-dock-twin/0.1"})
            try:
                body = urllib.request.urlopen(req, timeout=240).read()
                if body.lstrip().startswith(b"{"):
                    path.write_bytes(body)
                    break
            except Exception as e:  # busy mirrors: retry
                print(name, ep, e)
            time.sleep(15)
    dem = cache / "dem"
    dem.mkdir(exist_ok=True)
    for x, y in _dem_tiles():
        p = dem / f"{DEM_Z}_{x}_{y}.png"
        if not p.is_file():
            try:
                p.write_bytes(urllib.request.urlopen(
                    f"https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{DEM_Z}/{x}/{y}.png", timeout=60).read())
            except Exception as e:
                print("dem", x, y, e)
    s2 = cache / "s2_tci.npz"
    if not s2.is_file():
        import rasterio
        from rasterio.windows import from_bounds
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", AWS_NO_SIGN_REQUEST="YES"):
            with rasterio.open(S2_ITEM) as src:
                x0, y0, x1, y1 = EXTENT
                w = from_bounds(E0 + x0, N0 + y0, E0 + x1, N0 + y1, src.transform)
                arr = src.read(window=w, boundless=True, fill_value=0)
                t = src.window_transform(w)
                np.savez_compressed(s2, rgb=arr, transform=np.array([t.a, t.b, t.c, t.d, t.e, t.f]))


def _dem_tiles():
    lon0, lat0 = INV.transform(E0 + EXTENT[0], N0 + EXTENT[3])
    lon1, lat1 = INV.transform(E0 + EXTENT[2], N0 + EXTENT[1])
    n = 2 ** DEM_Z

    def tile(lat, lon):
        return (int((lon + 180) / 360 * n),
                int((1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n))
    xa, ya = tile(lat0 + 0.02, lon0 - 0.02)
    xb, yb = tile(lat1 - 0.02, lon1 + 0.02)
    return [(x, y) for x in range(xa, xb + 1) for y in range(ya, yb + 1)]


# ------------------------------------------------------------------ build

def load_osm(cache: Path):
    seen, out = set(), []
    for name in QUERIES:
        for e in json.loads((cache / f"{name}.json").read_text())["elements"]:
            k = (e["type"], e["id"])
            if k not in seen:
                seen.add(k)
                out.append(e)
    return out


def land_polygons(els):
    """Land = faces of (coastline + frame boundary) not connected to the open sea, minus inland water."""
    from shapely.geometry import LineString, Polygon, box, Point
    from shapely.ops import polygonize, unary_union
    frame = box(*EXTENT)
    lines = []
    for e in els:
        t = e.get("tags", {})
        if e["type"] == "way" and t.get("natural") == "coastline" and e.get("geometry"):
            p = geom_xy(e["geometry"])
            if len(p) >= 2:
                lines.append(LineString(p))
    from shapely.ops import linemerge
    coast = linemerge(unary_union(lines))
    parts = list(coast.geoms) if hasattr(coast, "geoms") else [coast]
    # the coastline download stops short of the frame in places: run open ends straight on to the frame edge
    ext = []
    for ln in parts:
        c = list(ln.coords)
        if np.allclose(c[0], c[-1]) or ln.length < 500:
            ext.append(ln)
            continue
        for end, prev in ((0, 1), (-1, -2)):
            p = np.array(c[end]); q = np.array(c[prev])
            if frame.exterior.distance(Point(p)) > 1:
                d = (p - q) / (np.linalg.norm(p - q) or 1)
                far = p + d * 40000
                ray = LineString([p, far]).intersection(frame)
                if not ray.is_empty:
                    ext.append(LineString([p, ray.coords[-1] if ray.geom_type == "LineString" else far]))
        ext.append(ln)
    merged = unary_union(ext + [frame.exterior])
    faces = list(polygonize(merged))
    sea_probe = Point(EXTENT[2] - 50, EXTENT[1] + 50)  # south-east corner: open sea
    land = [f for f in faces if not f.buffer(1).intersects(sea_probe)]
    land_u = unary_union(land)
    water = []
    for e in els:
        t = e.get("tags", {})
        if t.get("natural") == "water" or t.get("waterway") == "riverbank":
            for ring in _rings(e):
                if len(ring) >= 4:
                    pg = Polygon(ring).buffer(0)
                    if pg.area > 2000:
                        water.append(pg)
    if water:
        land_u = land_u.difference(unary_union(water))
    return land_u.intersection(frame).buffer(0)


def _rings(e):
    if e["type"] == "way" and e.get("geometry"):
        p = geom_xy(e["geometry"])
        if len(p) >= 4 and np.allclose(p[0], p[-1]):
            return [p]
        return []
    if e["type"] == "relation":
        # outer members assembled naively (closed ways only)
        out = []
        for m in e.get("members", []):
            if m.get("role") in ("outer", "") and m.get("geometry"):
                p = geom_xy(m["geometry"])
                if len(p) >= 4 and np.allclose(p[0], p[-1]):
                    out.append(p)
        return out
    return []


def dem_grid(cache: Path):
    from PIL import Image
    tiles = _dem_tiles()
    xs = sorted({t[0] for t in tiles}); ys = sorted({t[1] for t in tiles})
    mos = np.zeros((len(ys) * 256, len(xs) * 256), np.float32)
    for (x, y) in tiles:
        p = cache / "dem" / f"{DEM_Z}_{x}_{y}.png"
        if not p.is_file():
            continue
        a = np.asarray(Image.open(p).convert("RGB"), np.float32)
        h = a[..., 0] * 256 + a[..., 1] + a[..., 2] / 256 - 32768
        mos[(y - ys[0]) * 256:(y - ys[0] + 1) * 256, (x - xs[0]) * 256:(x - xs[0] + 1) * 256] = h
    gx = np.arange(EXTENT[0], EXTENT[2] + 0.1, GRID)
    gy = np.arange(EXTENT[1], EXTENT[3] + 0.1, GRID)
    X, Y = np.meshgrid(gx, gy)
    lon, lat = INV.transform(E0 + X, N0 + Y)
    n = 2 ** DEM_Z * 256
    px = (lon + 180) / 360 * n - xs[0] * 256
    py = (1 - np.log(np.tan(np.radians(lat)) + 1 / np.cos(np.radians(lat))) / np.pi) / 2 * n - ys[0] * 256
    x0 = np.clip(np.floor(px).astype(int), 0, mos.shape[1] - 2); y0 = np.clip(np.floor(py).astype(int), 0, mos.shape[0] - 2)
    fx = px - x0; fy = py - y0
    h = (mos[y0, x0] * (1 - fx) * (1 - fy) + mos[y0, x0 + 1] * fx * (1 - fy) + mos[y0 + 1, x0] * (1 - fx) * fy
         + mos[y0 + 1, x0 + 1] * fx * fy)
    return gx, gy, h


def ground_image(cache: Path):
    """Sentinel-2 true colour cropped to the frame, graded towards what the eye sees (lifted shadows, less blue cast)."""
    from PIL import Image
    d = np.load(cache / "s2_tci.npz")
    rgb = np.moveaxis(d["rgb"], 0, -1).astype(np.float32) / 255.0
    rgb = np.flipud(rgb) if d["transform"][4] > 0 else rgb
    # TCI is already stretched; lift mid-tones a little and warm it slightly
    rgb = np.clip(rgb, 0, 1) ** 0.86
    rgb = rgb * np.array([1.04, 1.0, 0.94])
    lum = rgb.mean(-1, keepdims=True)
    rgb = lum + (rgb - lum) * 0.88
    img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    W = int(round((EXTENT[2] - EXTENT[0]) / GROUND_RES)); H = int(round((EXTENT[3] - EXTENT[1]) / GROUND_RES))
    return img.resize((W, H), Image.LANCZOS)


# ------------------------------------------------------------------ layers

def _height(t):
    def num(v):
        try:
            return float(str(v).replace(",", ".").split()[0])
        except (ValueError, IndexError):
            return None
    h = num(t.get("height")) if t.get("height") else None
    if h is None and t.get("building:levels"):
        lv = num(t.get("building:levels"))
        if lv is not None:
            h = lv * 3.2 + 1.5
    if h is None:
        h = {"house": 7, "detached": 7, "residential": 16, "apartments": 18, "industrial": 11, "warehouse": 12,
             "roof": 6, "office": 18, "commercial": 12, "retail": 9, "hotel": 30, "school": 12, "public": 12,
             "hangar": 14, "service": 5, "garage": 4, "garages": 4, "shed": 4, "kiosk": 3, "church": 18,
             "ruins": 4, "construction": 8}.get(t.get("building"), 10)
    return float(min(max(h, 3), 320))


def _sampler(img):
    a = np.asarray(img, np.float32)
    H, W = a.shape[:2]

    def at(x, y, r=1):
        j = int((x - EXTENT[0]) / GROUND_RES); i = int((EXTENT[3] - y) / GROUND_RES)
        i0, i1 = max(0, i - r), min(H, i + r + 1); j0, j1 = max(0, j - r), min(W, j + r + 1)
        if i0 >= i1 or j0 >= j1:
            return (200, 200, 200)
        return tuple(int(v) for v in a[i0:i1, j0:j1].reshape(-1, 3).mean(0))
    return at


def _q(a, d=1):
    return [round(float(v), d) for v in np.asarray(a).ravel()]


def _fit_line(pts, tol=20.0, iters=4):
    # RANSAC over point pairs (cranes stand on several quay faces), then refine on the inliers
    best = None
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = pts[j] - pts[i]
            if np.linalg.norm(d) < 50:
                continue
            d = d / np.linalg.norm(d)
            n = np.array([-d[1], d[0]])
            k = np.abs((pts - pts[i]) @ n) < tol
            if best is None or k.sum() > best.sum():
                best = k
    keep = best if best is not None else np.ones(len(pts), bool)
    for _ in range(iters):
        c = pts[keep].mean(0)
        _, _, vt = np.linalg.svd(pts[keep] - c)
        d = vt[0]
        n = np.array([-d[1], d[0]])
        keep = np.abs((pts - c) @ n) < tol
    if d[0] + d[1] < 0:
        d = -d  # point roughly north-east
    return c, d, keep


def quay_site(els, land, lat_rng, lon_rng, name):
    """Quay edge from the STS crane rail (OSM seamark cranes) snapped to the coastline: SW and NE ends, frame metres."""
    from shapely.geometry import LineString
    pts = np.array([to_xy(e["lon"], e["lat"]) for e in els if e["type"] == "node"
                    and e.get("tags", {}).get("seamark:type") == "crane"
                    and lat_rng[0] < e["lat"] < lat_rng[1] and lon_rng[0] < e["lon"] < lon_rng[1]])
    c, d, keep = _fit_line(pts)
    n = np.array([d[1], -d[0]])  # to the right of north-east = south-east, the water side here
    # walk the land boundary: points within 60 m on the water side of the crane line and near-parallel
    bnd = []
    for g in getattr(land, "geoms", [land]):
        for ring in [g.exterior, *g.interiors]:
            a = np.asarray(ring.coords)
            seg = LineString(a).segmentize(5.0)
            bnd.append(np.asarray(seg.coords))
    b = np.concatenate(bnd)
    off = (b - c) @ n; along = (b - c) @ d
    near = (off > 0) & (off < 60) & (np.abs(along) < 1200)
    # the quay edge is the cluster of boundary points at a constant offset from the crane line
    o = np.median(off[near])
    edge = near & (np.abs(off - o) < 4)
    t = along[edge]
    # longest run without gaps > 30 m
    ts = np.sort(t); gaps = np.where(np.diff(ts) > 30)[0]
    runs = np.split(ts, gaps + 1); run = max(runs, key=lambda r: r[-1] - r[0])
    sw = c + d * run[0] + n * o; ne = c + d * run[-1] + n * o
    print(f"{name}: {keep.sum()} cranes on the line, quay edge {run[-1] - run[0]:.0f} m, crane rail {o:.1f} m behind the edge, "
          f"bearing {math.degrees(math.atan2(d[0], d[1])):.1f} deg")
    return {"sw": _q(sw), "ne": _q(ne), "rail_offset": round(float(o), 1)}


def build(cache: Path, out: Path):
    from PIL import Image
    from shapely import contains_xy
    from shapely.geometry import LineString, Polygon, Point
    from shapely.ops import unary_union
    out.mkdir(parents=True, exist_ok=True)
    els = load_osm(cache)
    land = land_polygons(els)
    print("land", round(land.area / 1e6, 1), "km2")

    # ground image + terrain
    img = ground_image(cache)  # used only to read the real water colour; the viewer draws its own ground
    gx, gy, dem = dem_grid(cache)
    X, Y = np.meshgrid(gx, gy)
    onland = contains_xy(land, X, Y)
    y = np.where(onland, np.maximum(dem - 8.0, -6.0), -8.0)
    enc = np.clip(np.round((y + 50.0) * 10.0), 0, 65535).astype(np.uint16)
    rgb = np.zeros(enc.shape + (3,), np.uint8)
    rgb[..., 0] = enc >> 8; rgb[..., 1] = enc & 255
    Image.fromarray(np.flipud(rgb)).save(out / "terrain.png", optimize=True)  # row 0 = north

    def ground_y(x, yy):
        j = (x - EXTENT[0]) / GRID; i = (yy - EXTENT[1]) / GRID
        i0 = int(np.clip(np.floor(i), 0, len(gy) - 2)); j0 = int(np.clip(np.floor(j), 0, len(gx) - 2))
        fi, fj = i - i0, j - j0
        v = (y[i0, j0] * (1 - fi) * (1 - fj) + y[i0, j0 + 1] * (1 - fi) * fj + y[i0 + 1, j0] * fi * (1 - fj)
             + y[i0 + 1, j0 + 1] * fi * fj)
        return max(0.0, float(v))

    # land polygons (oriented: land on the left of every ring) and the shoreline split into quay walls, breakwater
    # armour and natural shore, so the viewer can give each edge the right face
    from shapely.geometry.polygon import orient
    harbour = unary_union([Polygon(r).buffer(0) for e in els if e.get("tags", {}).get("landuse") == "harbour"
                           for r in _rings(e) if len(r) >= 4]).buffer(40)
    armour_zone = unary_union([Polygon(r).buffer(0) for e in els if e.get("tags", {}).get("man_made") == "breakwater"
                               for r in _rings(e) if len(r) >= 4]).buffer(12)
    land_s = land.simplify(1.0)
    land_out, edges = [], {"wall": [], "armour": [], "shore": []}
    for g in getattr(land_s, "geoms", [land_s]):
        g = orient(g, 1.0)
        land_out.append([_q(np.asarray(g.exterior.coords)[:-1])] + [_q(np.asarray(r.coords)[:-1]) for r in g.interiors])
        for ring in [g.exterior, *g.interiors]:
            a = np.asarray(LineString(ring.coords).segmentize(25.0).coords)
            cur, kind = [a[0]], None
            for p0, p1 in zip(a[:-1], a[1:]):
                mid = Point((p0 + p1) / 2)
                if not (EXTENT[0] + 5 < mid.x < EXTENT[2] - 5 and EXTENT[1] + 5 < mid.y < EXTENT[3] - 5):
                    k = None
                elif armour_zone.contains(mid):
                    k = "armour"
                elif harbour.contains(mid):
                    k = "wall"
                else:
                    k = "shore"
                if k != kind:
                    if kind and len(cur) > 1:
                        edges[kind].append(_q(LineString(cur).simplify(0.5).coords))
                    cur = [p0]
                kind = k
                cur.append(p1)
            if kind and len(cur) > 1:
                edges[kind].append(_q(LineString(cur).simplify(0.5).coords))
    walls, armour, shores = edges["wall"], edges["armour"], edges["shore"]
    print("edges", {k: len(v) for k, v in edges.items()})

    # buildings (port, Zona Franca, Montjuic, Poble-sec) and the tall city skyline
    tanks, buildings = [], []
    for e in els:
        t = e.get("tags", {})
        is_tank = t.get("man_made") == "storage_tank" or t.get("building") == "storage_tank"
        if not (is_tank or t.get("building")):
            continue
        for r in _rings(e):
            try:
                pg = Polygon(r).buffer(0)
            except Exception:
                continue
            if pg.is_empty or pg.area < 25:
                continue
            ct = pg.centroid
            if not (EXTENT[0] < ct.x < EXTENT[2] and EXTENT[1] < ct.y < EXTENT[3]):
                continue
            if is_tank:
                rad = math.sqrt(pg.area / math.pi)
                if rad < 2.5:
                    continue
                h = _height(t) if t.get("height") else (40.0 if rad >= 30 else min(max(rad * 1.15, 8), 24))
                kind = 1 if rad >= 30 else 0  # 1: LNG tank with a dome
                tanks.append([round(ct.x, 1), round(ct.y, 1), round(rad, 1), round(h, 1), kind, 0,
                              round(ground_y(ct.x, ct.y), 1)])
                continue
            pg = pg.simplify(0.6)
            if pg.geom_type != "Polygon":
                continue
            ring = np.asarray(pg.exterior.coords)[:-1]
            if len(ring) < 3:
                continue
            kind = BUILDING_KIND.get(t.get("building"), 0)
            if "castell" in str(t.get("name", "")).lower() or t.get("historic") in ("castle", "fort"):
                kind = 5
            buildings.append([round(_height(t), 1), round(ground_y(ct.x, ct.y), 1), kind] + _q(ring))
    print("buildings", len(buildings), "tanks", len(tanks))

    # spires and towers (Sagrada Familia, cable-car towers, Columbus) and the Collserola tower
    spires = []
    for e in els:
        t = e.get("tags", {})
        if t.get("man_made") in ("tower", "communications_tower") and t.get("height"):
            h = _height(t)
            if h < 35:
                continue
            g = e.get("geometry") or [{"lat": e.get("lat"), "lon": e.get("lon")}]
            p = geom_xy(g).mean(0)
            if EXTENT[0] < p[0] < EXTENT[2] and EXTENT[1] < p[1] < EXTENT[3]:
                spires.append([round(p[0], 1), round(p[1], 1), round(h, 1), round(ground_y(*p), 1), t.get("tower:type", "")])
    cx = to_xy(2.11412, 41.41740)
    landmarks = {"collserola": [round(cx[0], 1), round(cx[1], 1), 288.0, round(ground_y(*cx), 1)]}

    # terminal yards
    best_blocks, apm_slabs = [], []
    for e in els:
        t = e.get("tags", {})
        if e["type"] == "way" and t.get("crane:type") == "gantry_crane" and t.get("ref") and e["geometry"][0]["lat"] < 41.32:
            r = geom_xy(e["geometry"])[:-1]
            e1 = r[1] - r[0]; e2 = r[2] - r[1]
            if np.linalg.norm(e1) < np.linalg.norm(e2):
                e1, e2 = e2, e1
            best_blocks.append([int(t["ref"])] + _q(r.mean(0)) + [round(float(np.linalg.norm(e1)), 1),
                                round(float(np.linalg.norm(e2)), 1), round(math.atan2(e1[1], e1[0]), 5)])
        if t.get("industrial") == "depot" and str(t.get("name", "")).startswith("Container Storage"):
            for r in _rings(e):
                apm_slabs.append(_q(r[:-1]))
    best_blocks.sort()
    rail = []
    for e in els:
        t = e.get("tags", {})
        if t.get("railway") == "rail" and e.get("geometry") and t.get("service") in (None, "yard", "siding", "spur"):
            p = geom_xy(e["geometry"])
            m = p.mean(0)
            if -1800 < m[0] < 1800 and -3200 < m[1] < 3600:
                rail.append(_q(LineString(p).simplify(1.0).coords))
    breakwaters = [_q(Polygon(r).buffer(0).simplify(2.0).exterior.coords) for e in els
                   if e.get("tags", {}).get("man_made") == "breakwater" for r in _rings(e) if len(r) >= 4]

    sites = {
        "36A": quay_site(els, land, (41.300, 41.320), (2.13, 2.16), "36A BEST"),
        "24B": quay_site(els, land, (41.344, 41.3575), (2.1655, 2.1735), "24B APM"),
    }
    # land cover polygons by class (clipped to land), roads as ribbons, and a 20 m class raster for terrain/trees
    from shapely.geometry import MultiPolygon
    cover = {k: [] for k in COVER_ORDER}
    for e in els:
        t = e.get("tags", {})
        cls = None
        for k in ("aeroway", "leisure", "landuse", "natural"):
            if (k, t.get(k)) in COVER_TAGS:
                cls = COVER_TAGS[(k, t.get(k))]
                break
        if not cls:
            continue
        for rr in _rings(e):
            if len(rr) < 4:
                continue
            pg = Polygon(rr).buffer(0)
            if pg.is_empty or pg.area < 400:
                continue
            cover[cls].append(pg)
    cover_out = {}
    raster = np.zeros((int((EXTENT[3] - EXTENT[1]) / 20), int((EXTENT[2] - EXTENT[0]) / 20)), np.uint8)
    rx = EXTENT[0] + (np.arange(raster.shape[1]) + 0.5) * 20; ry = EXTENT[3] - (np.arange(raster.shape[0]) + 0.5) * 20
    RX, RY = np.meshgrid(rx, ry)
    for cls in COVER_ORDER:
        if not cover[cls]:
            continue
        u = unary_union(cover[cls]).intersection(land).simplify(1.5)
        polys = [g for g in getattr(u, "geoms", [u]) if g.geom_type == "Polygon" and g.area > 400]
        cover_out[cls] = [[_q(np.asarray(g.exterior.coords)[:-1])] + [_q(np.asarray(r2.coords)[:-1]) for r2 in g.interiors if len(r2.coords) > 3]
                          for g in polys]
        raster[contains_xy(MultiPolygon(polys), RX, RY)] = COVER.index(cls)
    raster[~contains_xy(land, RX, RY)] = 255
    Image.fromarray(raster).save(out / "cover.png", optimize=True)
    print("cover", {k: len(v) for k, v in cover_out.items()})
    roads = []
    for e in els:
        t = e.get("tags", {})
        w = ROAD_W.get(t.get("highway"))
        if not w or not e.get("geometry") or t.get("tunnel") == "yes":
            continue
        pts = geom_xy(e["geometry"])
        if len(pts) < 2:
            continue
        lay = 1 if (t.get("bridge") == "yes" or str(t.get("layer", "0")) in ("1", "2")) else 0
        roads.append([w, lay] + _q(LineString(pts).simplify(1.5).coords))
    print("roads", len(roads))

    # ship routes (frame metres, read off the imagery): the south entrance between the Dic del Sud and Dic de l'Est
    # heads, the channel north past the Moll de l'Energia to the Moll Sud, and the anchorage offshore where waiting
    # ships lie (visible as ship dots in the Sentinel-2 scene)
    routes = {
        "sea": [5200.0, -4200.0],
        "mouth_out": [1250.0, -2050.0],
        "mouth_in": [1110.0, -1250.0],
        "anchorage": {"centre": [3150.0, -350.0], "along": [0.12, 0.99], "half": [1150.0, 650.0]},
        "36A": {"entry": [420.0, -1720.0], "lane": 330.0},
        "24B": {"entry": [1180.0, 1650.0], "via": [[960.0, -450.0], [900.0, 450.0], [1030.0, 1250.0]], "lane": 250.0},
    }
    # the Moll Adossat face opposite APM (cruise ships berth there)
    bnd = np.concatenate([np.asarray(LineString(r.coords).segmentize(5).coords) for g in getattr(land, "geoms", [land])
                          for r in [g.exterior, *g.interiors]])
    sel = bnd[(bnd[:, 0] > 1560) & (bnd[:, 0] < 1820) & (bnd[:, 1] > 1950) & (bnd[:, 1] < 3150)]
    c0, d0, k0 = _fit_line(sel, tol=8)
    t0 = (sel[k0] - c0) @ d0
    routes["cruise"] = {"a": _q(c0 + d0 * t0.min()), "b": _q(c0 + d0 * t0.max())}
    print("cruise berth", routes["cruise"], round(float(t0.max() - t0.min())), "m")
    body = {
        "routes": routes,
        "frame": {"origin_ll": ORIGIN_LL, "crs": "EPSG:32631", "e0": E0, "n0": N0, "extent": EXTENT, "grid": GRID,
                  "ground_res": GROUND_RES, "terrain": {"w": len(gx), "h": len(gy), "enc": "(R*256+G)/10-50 m"},
                  "cover_res": 20.0},
        "sites": sites, "land": land_out, "walls": walls, "armour": armour, "shores": shores, "breakwaters": breakwaters,
        "buildings": buildings, "tanks": tanks, "spires": spires, "landmarks": landmarks, "cover": cover_out,
        "cover_classes": COVER, "roads": roads,
        "best_blocks": best_blocks, "apm_slabs": apm_slabs, "rail": rail,
    }
    raw = json.dumps(body, separators=(",", ":")).encode()
    (out / "twin.json.gz").write_bytes(gzip.compress(raw, 9, mtime=0))
    print("twin.json", len(raw) // 1024, "KB raw,", (out / "twin.json.gz").stat().st_size // 1024, "KB gz;",
          "cover.png", (out / "cover.png").stat().st_size // 1024, "KB; terrain.png",
          (out / "terrain.png").stat().st_size // 1024, "KB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "build"])
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    if a.cmd == "fetch":
        fetch(a.cache)
    else:
        build(a.cache, a.out)


if __name__ == "__main__":
    main()
